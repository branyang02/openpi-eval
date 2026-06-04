"""
Evaluate every task in a LIBERO-Pro suite in parallel by launching one subprocess
per task_id. Each subprocess is an independent ``main.py`` invocation, which
gives each env its own MuJoCo/EGL context (necessary because MuJoCo + EGL
is not safe to share across envs in a single process).

For sequential execution with inline stack traces on crash, pass
``--num_workers 1``.

Examples:
    MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_goal_task
    MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_10_object --num_episodes 10 --num_workers 5
    MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_goal_object --output_dir /tmp/liberopro_eval_run

One run's entire output tree lives under a single directory (by default
``examples/liberopro_env/output/{task_suite_name}/``, or whatever ``--output_dir``
points at)::

    <output_dir>/
    |-- results.json
    |-- parallel_logs/task_NN.log
    `-- <task_id:02d>-<task_name>/episode_NNN.mp4
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import math
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np
import tyro

logger = logging.getLogger(__name__)

# Pulls the last ``success_rate=0.50`` (or similar) from the main.py log stream.
# main.py logs this once at the end of eval_task via ``logger.info``, e.g.:
#   [libero_goal_task/open.../task_00] success_rate=1.00 (1/1)
SUCCESS_RATE_RE = re.compile(r"success_rate=([0-9.]+)")


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # LIBERO-Pro suite name.
    task_suite_name: str = "libero_goal_task"
    # Number of episodes / initial states per task.
    num_episodes: int = 15
    # Override the suite default max steps. If None, uses main.BASE_SUITE_MAX_STEPS.
    max_steps: Optional[int] = None
    # Number of settling steps before policy actions.
    num_steps_wait: int = 10
    # Number of steps to execute from the model's action plan before re-planning.
    replan_steps: int = 5

    # Image resize size for the policy input.
    resize_size: int = 224

    # Cameras to tile into each subprocess's per-episode video output. Forwarded
    # to main.py as a single --render_cameras flag followed by all camera names
    # (see _build_command for the exact shape). Must match one of the keys in
    # main.py's CAMERA_KEYS dict (``agentview``, ``eye_in_hand``).
    render_cameras: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "eye_in_hand"]
    )

    fps: int = 10
    seed: int = 7

    # Max number of tasks to run concurrently. Each task is its own subprocess,
    # so this caps concurrent MuJoCo/EGL contexts. Higher = faster but more
    # pressure on the shared policy server and more host memory.
    num_workers: int = 10

    # Top-level run directory. See module docstring for layout. Relative paths
    # are resolved against the user's shell cwd.
    output_dir: Optional[str] = None


def _build_command(
    args: Args,
    task_id: int,
    output_dir: str,
) -> List[str]:
    """Build the ``main.py`` CLI invocation for one task_id.

    ``output_dir`` is the absolute path where this subprocess should write its
    per-task video directory. It is unconditionally forwarded as ``--output_dir``
    so that main.py does not fall back to its own default (which would land
    videos in a separate ``output/{suite}-task{id:02d}/`` tree).

    """
    cmd = [
        sys.executable,
        "main.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--task_suite_name",
        args.task_suite_name,
        "--task_id",
        str(task_id),
        "--num_episodes",
        str(args.num_episodes),
        "--num_steps_wait",
        str(args.num_steps_wait),
        "--replan_steps",
        str(args.replan_steps),
        "--resize_size",
        str(args.resize_size),
        "--fps",
        str(args.fps),
        "--seed",
        str(args.seed),
        "--output_dir",
        output_dir,
    ]
    # Forward render_cameras as a single --render_cameras flag followed by all
    # values (tyro's List[str] fields use nargs='+' semantics in this venv, so
    # repeated --render_cameras flags would silently keep only the last value).
    # subprocess.run takes an argv list, so no shell-splitting concerns.
    if args.render_cameras:
        cmd.append("--render_cameras")
        cmd.extend(args.render_cameras)
    if args.max_steps is not None:
        cmd.extend(["--max_steps", str(args.max_steps)])
    return cmd


def _run_one_task(
    args: Args,
    task_id: int,
    log_dir: str,
    cwd: str,
    output_dir: str,
) -> Dict[str, object]:
    """Launch main.py for a single task_id and return a parsed result dict.

    Writes the subprocess's combined stdout+stderr to ``log_dir/task_{id}.log``
    so the main process doesn't have to deal with interleaved output, and so
    the user can re-inspect the per-task logs after the run.
    """
    cmd = _build_command(args, task_id, output_dir)
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")

    log_path = os.path.join(log_dir, f"task_{task_id:02d}.log")
    with open(log_path, "w") as log_file:
        proc = subprocess.run(  # noqa: S603
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            check=False,
        )

    with open(log_path) as log_file:
        log_text = log_file.read()

    matches = SUCCESS_RATE_RE.findall(log_text)
    if not matches:
        logger.error(
            "task_%02d produced no success_rate line (returncode=%d). See %s",
            task_id,
            proc.returncode,
            log_path,
        )
        success_rate = float("nan")
    else:
        success_rate = float(matches[-1])

    return {
        "task_id": task_id,
        "success_rate": success_rate,
        "returncode": proc.returncode,
        "log_path": log_path,
    }


def main(args: Args) -> None:
    from main import get_task_suite

    np.random.seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    task_suite = get_task_suite(args.task_suite_name)
    # One output dir holds everything this run produces: results.json,
    # parallel_logs/, and each subprocess's per-task video directory. The same
    # dir is forwarded to main.py subprocesses via --output_dir so their
    # per-task video dirs land alongside results.json instead of in a
    # sibling ``output/{suite}-task{id:02d}/`` tree.
    #
    # ``os.path.abspath`` matters when the user passes a relative --output_dir:
    # main.py subprocesses run with cwd=script_dir, so a relative path would
    # otherwise resolve against the wrong directory. Resolving it here uses
    # eval_all's own cwd (the user's shell cwd), which is what they mean.
    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(script_dir, "output", args.task_suite_name)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "parallel_logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.info(
        "Evaluating %d tasks from %s in parallel (num_workers=%d)",
        task_suite.n_tasks,
        args.task_suite_name,
        args.num_workers,
    )
    logger.info(
        "Per-task stdout/stderr is captured to %s. Tail a single task with:\n"
        "    tail -f %s/task_00.log\n"
        "or follow every task at once with:\n"
        "    tail -f %s/task_*.log",
        log_dir,
        log_dir,
        log_dir,
    )
    # Pre-build per-task metadata in the parent process so the aggregated
    # results.json can include task_name / task_description without having to
    # parse them out of each subprocess's stdout. This is cheap: no env
    # construction, just a lookup against the benchmark registry.
    task_metadata: Dict[int, Dict[str, str]] = {}
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        task_metadata[task_id] = {
            "task_name": getattr(task, "name", f"task_{task_id:02d}"),
            "task_description": str(task.language),
        }

    results: List[Dict[str, object]] = []
    results_path = os.path.join(output_dir, "results.json")

    # ThreadPoolExecutor instead of ProcessPoolExecutor: each worker just blocks
    # on subprocess.run, so there's no Python-side compute to parallelize. This
    # avoids the double-fork (pool worker -> main.py subprocess) and sidesteps
    # all pickling concerns. The max_workers cap is still enforced by the pool.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(
                _run_one_task,
                args,
                task_id,
                log_dir,
                script_dir,
                output_dir,
            ): task_id
            for task_id in range(task_suite.n_tasks)
        }

        for future in concurrent.futures.as_completed(futures):
            task_id = futures[future]
            try:
                parsed = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception("task_%02d crashed in worker thread: %s", task_id, exc)
                parsed = {
                    "task_id": task_id,
                    "success_rate": float("nan"),
                    "returncode": -1,
                    "log_path": os.path.join(log_dir, f"task_{task_id:02d}.log"),
                }

            task_summary = {
                "task_id": task_id,
                "task_name": task_metadata[task_id]["task_name"],
                "task_description": task_metadata[task_id]["task_description"],
                "success_rate": parsed["success_rate"],
            }
            results.append(task_summary)
            logger.info(
                "[task_%02d/%s] success_rate=%.2f",
                task_id,
                task_summary["task_name"],
                task_summary["success_rate"],
            )

            # Incremental save so progress isn't lost on early exit. Sort by
            # task_id for a stable on-disk order during the run; the final save
            # below re-sorts by success_rate (descending) for human-friendly
            # display in the summary.
            results.sort(key=lambda item: item["task_id"])
            with open(results_path, "w") as file_handle:
                json.dump(results, file_handle, indent=2)

    valid = [
        item
        for item in results
        if not (
            isinstance(item["success_rate"], float) and math.isnan(item["success_rate"])
        )
    ]
    mean_success = (
        float(np.mean([item["success_rate"] for item in valid])) if valid else 0.0
    )

    summary = {
        "task_suite_name": args.task_suite_name,
        "mean_success_rate": mean_success,
        "per_task": sorted(
            results, key=lambda item: item["success_rate"], reverse=True
        ),
    }
    with open(results_path, "w") as file_handle:
        json.dump(summary, file_handle, indent=2)

    logger.info("Results saved to %s", results_path)
    logger.info("=" * 60)
    logger.info(
        "[%s] mean success rate: %.2f (%.0f%%)",
        args.task_suite_name,
        mean_success,
        mean_success * 100.0,
    )
    for task in summary["per_task"]:
        logger.info(
            "  task_%02d %-35s %.2f",
            task["task_id"],
            task["task_name"],
            task["success_rate"],
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
