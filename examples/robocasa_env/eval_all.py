"""
Evaluate every environment in a RoboCasa task set in parallel by launching one
subprocess per env_name. Each subprocess is an independent ``main.py`` invocation,
which gives each env its own MuJoCo/EGL context (necessary because MuJoCo + EGL
is not safe to share across envs in a single process).

Available task sets:
- ``subset``              — curated 7-task smoke subset (default).
                            Parallels MetaWorld's ``--split subset``.
- ``atomic_seen``         — 18 atomic target tasks
- ``composite_seen``      — 16 seen composite target tasks
- ``composite_unseen``    — 16 unseen composite target tasks
- ``target50``            — atomic_seen + composite_seen + composite_unseen
- ``pretrain50`` / ``pretrain100`` / ``pretrain200`` / ``pretrain300`` — pretraining task sets

All named task sets other than ``subset`` resolve via
``robocasa.utils.dataset_registry.TASK_SET_REGISTRY``. Pass ``--tasks t1 t2 ...``
to override the task set with an explicit list (parallels metaworld's ``--tasks``).

RoboCasa env stepping is roughly 10x slower than libero (~400 ms per step), so
parallel subprocess orchestration is a substantial wall-clock win over the old
sequential eval_all. For sequential execution with inline stack traces on crash,
pass ``--num_workers 1``.

Examples:
    MUJOCO_GL=egl uv run python eval_all.py                                 # curated subset
    MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen
    MUJOCO_GL=egl uv run python eval_all.py --task_set composite_seen --num_episodes 3 --num_workers 5
    MUJOCO_GL=egl uv run python eval_all.py --tasks OpenDrawer CloseFridge  # explicit task list

One run's entire output tree lives under a single directory (by default
``examples/robocasa_env/output/{task_set}-{split}/``, or whatever
``--output_dir`` points at)::

    <output_dir>/
    ├── results.json
    ├── parallel_logs/task_NN_<env_name>.log
    └── <env_name>/episode_NNN.mp4
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
import robocasa  # noqa: F401
import tyro
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

logger = logging.getLogger(__name__)

# Curated 7-task subset for quick smoke evaluations, parallel to MetaWorld's
# SUBSET and faster than the full 18-task atomic_seen set.
SUBSET = [
    "CloseFridge",
    "CoffeeSetupMug",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "TurnOnElectricKettle",
]

# Pulls the last ``success_rate=0.50`` (or similar) from the main.py log stream.
# main.py logs this once at the end of eval_task via ``logger.info``, e.g.:
#   [CloseBlenderLid/pretrain] success_rate=1.00 (1/1)
SUCCESS_RATE_RE = re.compile(r"success_rate=([0-9.]+)")


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # Task set name. ``subset`` picks the curated 7-task list (default; see
    # module-level SUBSET). Any other value must be a key of
    # ``TASK_SET_REGISTRY``. Overridden by ``--tasks`` if that is non-empty.
    task_set: str = "subset"
    # Explicit list of env_names to evaluate. When non-empty, overrides
    # ``--task_set``. Parallels metaworld's ``--tasks`` override.
    tasks: List[str] = dataclasses.field(default_factory=list)
    # Dataset split: "pretrain" (in-distribution object instances) or "target" (held-out).
    split: str = "pretrain"
    # Number of episodes to run per task. Drop to --num_episodes 1 for quick
    # smoke tests because RoboCasa env stepping is slow (~400 ms/step).
    num_episodes: int = 15
    # Override the maximum steps per episode. If None, uses 1.5 * task horizon
    # via ``main.get_task_horizon`` in the subprocess.
    max_steps: Optional[int] = None
    # Number of steps to execute from the model's action plan before re-planning.
    replan_steps: int = 5

    # Image resize size for the policy input.
    resize_size: int = 224

    # Cameras to tile into each subprocess's per-episode video output. Forwarded
    # to main.py as a single --render_cameras flag followed by all camera names
    # (see _build_command for the exact shape). Must match one of the keys in
    # main.py's CAMERA_KEYS dict (``agentview_left``, ``agentview_right``,
    # ``eye_in_hand``).
    render_cameras: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview_left", "agentview_right", "eye_in_hand"]
    )

    fps: int = 24
    seed: int = 7

    # Max number of tasks to run concurrently. Each task is its own subprocess,
    # so this caps concurrent MuJoCo/EGL contexts. Higher = faster but more
    # pressure on the shared policy server and more host memory. RoboCasa env
    # stepping is ~400 ms/step so parallelism helps a lot; 5-10 workers is the
    # sweet spot.
    num_workers: int = 10

    # Top-level run directory. See module docstring for layout. Relative paths
    # are resolved against the user's shell cwd.
    output_dir: Optional[str] = None


def _sanitize_env_name(env_name: str) -> str:
    """Defensive slug of env_name for log filenames. RoboCasa env names like
    ``CloseBlenderLid`` and ``OpenCabinet`` are already filesystem-safe, but
    this keeps behavior robust if a future env name introduces unusual chars.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", env_name.strip()).strip("-")
    return slug or "task"


def _build_command(
    args: Args,
    env_name: str,
    output_dir: str,
    task_idx: int = 0,
) -> List[str]:
    """Build the ``main.py`` CLI invocation for one env_name.

    ``output_dir`` is the absolute path where this subprocess should write its
    per-task video directory. It is unconditionally forwarded as ``--output_dir``
    so that main.py does not fall back to its own bare ``output/`` default,
    which — via ``eval_task``'s per-env nesting — would scatter videos into a
    sibling ``output/{env_name}/`` tree instead of alongside ``results.json``.

    ``task_idx`` is unused in the argv but accepted here so the call site can
    pass it uniformly alongside env_name; it's only consumed for log filename
    ordering in ``_run_one_task``.
    """
    del task_idx  # only used in _run_one_task for log filename stability
    cmd = [
        sys.executable,
        "main.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--env_name",
        env_name,
        "--split",
        args.split,
        "--num_episodes",
        str(args.num_episodes),
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
    env_name: str,
    task_idx: int,
    log_dir: str,
    cwd: str,
    output_dir: str,
) -> Dict[str, object]:
    """Launch main.py for a single env_name and return a parsed result dict.

    Writes the subprocess's combined stdout+stderr to
    ``log_dir/task_{idx:02d}_{env_name}.log`` so the main process doesn't have
    to deal with interleaved output, and so the user can re-inspect the
    per-task logs after the run. The ``task_idx`` prefix makes the log files
    sort in submission order even when env_names are alphabetical.
    """
    cmd = _build_command(args, env_name, output_dir, task_idx=task_idx)
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")

    log_name = f"task_{task_idx:02d}_{_sanitize_env_name(env_name)}.log"
    log_path = os.path.join(log_dir, log_name)
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
            "%s produced no success_rate line (returncode=%d). See %s",
            env_name,
            proc.returncode,
            log_path,
        )
        success_rate = float("nan")
    else:
        success_rate = float(matches[-1])

    return {
        "env_name": env_name,
        "task_idx": task_idx,
        "success_rate": success_rate,
        "returncode": proc.returncode,
        "log_path": log_path,
    }


def _resolve_tasks(args: Args) -> List[str]:
    """Return the list of env_names to evaluate.

    Order of precedence, mirroring ``examples/metaworld/eval_all.py``:
    1. ``--tasks t1 t2 ...`` explicit override, if non-empty.
    2. ``--task_set subset`` — the curated ``SUBSET`` list.
    3. Any other ``--task_set`` value — resolved via ``TASK_SET_REGISTRY``.
    """
    if args.tasks:
        return list(args.tasks)
    if args.task_set == "subset":
        return list(SUBSET)
    if args.task_set not in TASK_SET_REGISTRY:
        raise ValueError(
            f"Unknown task_set '{args.task_set}'. Available: "
            f"{['subset', *sorted(TASK_SET_REGISTRY.keys())]}"
        )
    return list(TASK_SET_REGISTRY[args.task_set])


def main(args: Args) -> None:
    env_names: List[str] = _resolve_tasks(args)

    np.random.seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # One output dir holds everything this run produces: results.json,
    # parallel_logs/, and each subprocess's per-task video directory. The same
    # dir is forwarded to main.py subprocesses via --output_dir so their
    # per-task video dirs land alongside results.json instead of in a sibling
    # ``output/{env_name}/`` tree (main.py's default parent is the bare
    # ``output/``; ``eval_task`` inside main.py then nests ``{env_name}/``).
    #
    # ``os.path.abspath`` matters when the user passes a relative --output_dir:
    # main.py subprocesses run with cwd=script_dir, so a relative path would
    # otherwise resolve against the wrong directory. Resolving it here uses
    # eval_all's own cwd (the user's shell cwd), which is what they mean.
    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(script_dir, "output", f"{args.task_set}-{args.split}")
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "parallel_logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.info(
        "Evaluating %d tasks from %s (split=%s) in parallel (num_workers=%d)",
        len(env_names),
        args.task_set,
        args.split,
        args.num_workers,
    )
    logger.info(
        "Per-task stdout/stderr is captured to %s. Tail a single task with:\n"
        "    tail -f %s/task_00_<env_name>.log\n"
        "or follow every task at once with:\n"
        "    tail -f %s/task_*.log",
        log_dir,
        log_dir,
        log_dir,
    )
    # Pre-build per-task metadata (just the submission-order index) so the
    # aggregated results.json can reference it later. RoboCasa dispatches by
    # env_name string, so there's no integer task_id concept — task_idx is
    # purely a client-side ordering handle.
    task_metadata: Dict[str, Dict[str, object]] = {
        env_name: {"task_idx": idx} for idx, env_name in enumerate(env_names)
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
                env_name,
                task_metadata[env_name]["task_idx"],
                log_dir,
                script_dir,
                output_dir,
            ): env_name
            for env_name in env_names
        }

        for future in concurrent.futures.as_completed(futures):
            env_name = futures[future]
            task_idx = task_metadata[env_name]["task_idx"]
            try:
                parsed = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s crashed in worker thread: %s", env_name, exc)
                log_name = f"task_{task_idx:02d}_{_sanitize_env_name(env_name)}.log"
                parsed = {
                    "env_name": env_name,
                    "task_idx": task_idx,
                    "success_rate": float("nan"),
                    "returncode": -1,
                    "log_path": os.path.join(log_dir, log_name),
                }

            task_summary = {
                "env_name": env_name,
                "task_idx": task_idx,
                "success_rate": parsed["success_rate"],
            }
            results.append(task_summary)
            logger.info(
                "[%s] success_rate=%.2f",
                env_name,
                task_summary["success_rate"],
            )

            # Incremental save so progress isn't lost on early exit. Sort by
            # task_idx for a stable on-disk order during the run; the final
            # save below re-shapes into the mean+per_task summary schema.
            results.sort(key=lambda item: item["task_idx"])
            interim = {item["env_name"]: item["success_rate"] for item in results}
            with open(results_path, "w") as file_handle:
                json.dump(interim, file_handle, indent=2)

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

    # Preserve the same schema the old sequential eval_all produced:
    # {"task_set", "split", "mean_success_rate", "per_task": {env_name: rate, ...}}
    # sorted by success_rate descending for human-friendly display.
    per_task_sorted = dict(
        sorted(
            ((item["env_name"], item["success_rate"]) for item in results),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    summary = {
        "task_set": args.task_set,
        "split": args.split,
        "mean_success_rate": mean_success,
        "per_task": per_task_sorted,
    }
    with open(results_path, "w") as file_handle:
        json.dump(summary, file_handle, indent=2)

    logger.info("Results saved to %s", results_path)
    logger.info("=" * 60)
    logger.info(
        "[%s/%s] mean success rate: %.2f (%.0f%%)",
        args.task_set,
        args.split,
        mean_success,
        mean_success * 100.0,
    )
    for env_name, rate in per_task_sorted.items():
        logger.info("  %-40s %.2f", env_name, rate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
