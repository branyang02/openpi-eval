"""
Evaluate RoboLab task sets by launching one ``main.py`` subprocess per task.

RoboLab already vectorizes episodes within one Isaac Sim process via
``--num-envs``. This script adds the same task-set orchestration pattern used by
the LIBERO and RoboCasa examples: one top-level output directory, per-task logs,
and an aggregate ``results.json``.

Examples:
    CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES uv run python eval_all.py
    CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES uv run python eval_all.py --num-envs 10
    CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES uv run python eval_all.py --tasks BananaInBowlTask OneBottleInSquarePailTask

Output layout:
    <output_dir>/
    ├── results.json
    ├── episode_results.jsonl
    ├── parallel_logs/task_NN_<task_name>.log
    └── <task_name>/
        ├── env_cfg.json
        ├── log_<run>_env<env>.json
        ├── run_<run>.hdf5
        └── *.mp4  # when --video-mode is not none
"""

from __future__ import annotations

import ast
import concurrent.futures
import dataclasses
import json
import logging
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import tyro

from main import PolicyVariant, VideoMode, _ensure_output_dir_policy_compatible

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

SUBSET = [
    "BananaInBowlTask",
    "BananasInBinThreeTotalTask",
    "OneBottleInSquarePailTask",
    "MustardAboveRaisinTask",
    "BananaThenRubiksCubeTask",
]


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    policy: PolicyVariant = "pi05"
    # ``all`` discovers every benchmark task class in third_party/robolab.
    # ``subset`` is available for quick smoke runs.
    task_set: str = "all"
    # Explicit task class names. Overrides --task-set when non-empty.
    tasks: List[str] = dataclasses.field(default_factory=list)
    # RoboLab task subdirectories to register.
    task_dirs: List[str] = dataclasses.field(default_factory=lambda: ["benchmark"])

    num_envs: int = 1
    num_runs: int = 1
    num_episodes_adaptive: Optional[int] = None
    ci_pp_width: float = 0.14

    open_loop_horizon: Optional[int] = None
    remote_uri: Optional[str] = None

    instruction_type: str = "default"
    video_mode: VideoMode = "all"
    headless: bool = True
    device: str = "cuda:0"
    enable_subtask: bool = False
    enable_verbose: bool = False
    enable_debug: bool = False
    record_image_data: bool = False
    randomize_background: bool = False
    background_seed: Optional[int] = None

    # Max number of tasks to evaluate concurrently. Keep this at 1 unless the
    # host has enough CPU/GPU memory for multiple Isaac Sim processes. Increase
    # num_envs first when you want more episodes per task.
    num_workers: int = 1

    # Top-level run directory. Relative paths are resolved against the user's
    # shell cwd before being forwarded to main.py subprocesses.
    output_dir: Optional[str] = None


def _sanitize_task_name(task_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", task_name.strip()).strip("-")
    return slug or "task"


def _base_name(base: ast.expr) -> str | None:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def _is_task_class(node: ast.ClassDef) -> bool:
    return any(_base_name(base) == "Task" for base in node.bases)


def _discover_benchmark_tasks(repo_root: Path = _REPO_ROOT) -> List[str]:
    task_dir = repo_root / "third_party" / "robolab" / "robolab" / "tasks" / "benchmark"
    if not task_dir.exists():
        raise FileNotFoundError(
            "RoboLab benchmark tasks are missing. Run: "
            "git submodule update --init --recursive third_party/robolab"
        )

    task_names: list[str] = []
    for task_file in sorted(task_dir.glob("*.py")):
        if task_file.name == "__init__.py":
            continue
        tree = ast.parse(task_file.read_text(), filename=str(task_file))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and _is_task_class(node):
                task_names.append(node.name)

    return sorted(task_names)


def _resolve_tasks(args: Args, repo_root: Path = _REPO_ROOT) -> List[str]:
    if args.tasks:
        return list(args.tasks)
    if args.task_set == "subset":
        return list(SUBSET)
    if args.task_set == "all":
        return _discover_benchmark_tasks(repo_root)
    raise ValueError(
        "Unknown task_set {!r}. Available: subset, all".format(args.task_set)
    )


def _run_label(args: Args) -> str:
    return "explicit" if args.tasks else args.task_set


def _append_list_flag(cmd: list[str], flag: str, values: list[str]) -> None:
    if values:
        cmd.append(flag)
        cmd.extend(values)


def _build_command(args: Args, task_name: str, output_dir: str) -> List[str]:
    cmd = [
        sys.executable,
        "main.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--policy",
        args.policy,
        "--task",
        task_name,
        "--num-envs",
        str(args.num_envs),
        "--num-runs",
        str(args.num_runs),
        "--ci-pp-width",
        str(args.ci_pp_width),
        "--instruction-type",
        args.instruction_type,
        "--video-mode",
        args.video_mode,
        "--device",
        args.device,
        "--output-dir",
        output_dir,
    ]

    _append_list_flag(cmd, "--task-dirs", list(args.task_dirs))

    if args.remote_uri is not None:
        cmd.extend(["--remote-uri", args.remote_uri])
    if args.open_loop_horizon is not None:
        cmd.extend(["--open-loop-horizon", str(args.open_loop_horizon)])
    if args.num_episodes_adaptive is not None:
        cmd.extend(["--num-episodes-adaptive", str(args.num_episodes_adaptive)])
    if not args.headless:
        cmd.append("--no-headless")
    if args.enable_subtask:
        cmd.append("--enable-subtask")
    if args.enable_verbose:
        cmd.append("--enable-verbose")
    if args.enable_debug:
        cmd.append("--enable-debug")
    if args.record_image_data:
        cmd.append("--record-image-data")
    if args.randomize_background:
        cmd.append("--randomize-background")
    if args.background_seed is not None:
        cmd.extend(["--background-seed", str(args.background_seed)])

    return cmd


def _episode_matches_task(
    episode: dict, task_name: str, policy: str | None = None
) -> bool:
    if policy is not None and episode.get("policy") not in (None, policy):
        return False
    return episode.get("env_name") == task_name or episode.get("task_name") == task_name


def _load_episode_results(
    output_dir: str,
    task_name: str | None = None,
    policy: str | None = None,
) -> list[dict]:
    jsonl_path = os.path.join(output_dir, "episode_results.jsonl")
    json_path = os.path.join(output_dir, "episode_results.json")
    episodes: list[dict] = []

    if os.path.exists(jsonl_path):
        with open(jsonl_path) as file_handle:
            for line in file_handle:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))

    elif os.path.exists(json_path):
        with open(json_path) as file_handle:
            data = json.load(file_handle)
        if isinstance(data, list):
            episodes = data
        elif isinstance(data, dict):
            episodes = [data]

    if task_name is not None:
        return [
            episode
            for episode in episodes
            if _episode_matches_task(episode, task_name, policy=policy)
        ]

    if policy is not None:
        return [
            episode for episode in episodes if episode.get("policy") in (None, policy)
        ]

    return episodes


def _summarize_episodes(episodes: list[dict]) -> dict[str, object]:
    num_episodes = len(episodes)
    num_success = sum(1 for episode in episodes if episode.get("success") is True)
    success_rate = float(num_success / num_episodes) if num_episodes else float("nan")
    return {
        "num_episodes": num_episodes,
        "num_success": num_success,
        "success_rate": success_rate,
    }


def _expected_min_episodes(args: Args) -> int:
    if args.num_episodes_adaptive is not None:
        return args.num_envs
    return args.num_envs * args.num_runs


def _task_failure_reason(
    args: Args, returncode: int, episodes: list[dict]
) -> str | None:
    if returncode != 0:
        return f"subprocess exited with return code {returncode}"

    expected_min = _expected_min_episodes(args)
    if len(episodes) < expected_min:
        return (
            f"expected at least {expected_min} matching episode result(s), "
            f"found {len(episodes)}"
        )

    return None


def _run_one_task(
    args: Args,
    task_name: str,
    task_idx: int,
    log_dir: str,
    cwd: str,
    output_dir: str,
) -> Dict[str, object]:
    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    cmd = _build_command(args, task_name, output_dir)
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    log_name = f"task_{task_idx:02d}_{_sanitize_task_name(task_name)}.log"
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

    episodes = _load_episode_results(
        output_dir, task_name=task_name, policy=args.policy
    )
    summary = _summarize_episodes(episodes)
    failure_reason = _task_failure_reason(args, proc.returncode, episodes)
    summary.update(
        {
            "task_name": task_name,
            "task_idx": task_idx,
            "returncode": proc.returncode,
            "log_path": log_path,
            "output_dir": task_output_dir,
            "episode_results_path": os.path.join(output_dir, "episode_results.jsonl"),
            "expected_min_episodes": _expected_min_episodes(args),
        }
    )
    if failure_reason is not None:
        summary["failure_reason"] = failure_reason
    return summary


def _success_sort_key(item: dict[str, object]) -> tuple[int, float, int]:
    rate = item["success_rate"]
    if isinstance(rate, float) and math.isnan(rate):
        return (1, 0.0, int(item["task_idx"]))
    return (0, -float(rate), int(item["task_idx"]))


def _build_final_summary(
    args: Args,
    run_label: str,
    task_names: list[str],
    results: list[dict[str, object]],
    mean_success: float,
) -> dict[str, object]:
    per_task = []
    for item in sorted(results, key=_success_sort_key):
        task_summary = {
            "task_name": item["task_name"],
            "task_idx": item["task_idx"],
            "num_episodes": item["num_episodes"],
            "num_success": item["num_success"],
            "success_rate": item["success_rate"],
        }
        failure_reason = item.get("failure_reason")
        if failure_reason is not None:
            task_summary["failure_reason"] = failure_reason
            task_summary["returncode"] = item["returncode"]
            task_summary["log_path"] = item["log_path"]
        per_task.append(task_summary)

    return {
        "task_set": run_label,
        "requested_task_set": args.task_set,
        "tasks": task_names,
        "policy": args.policy,
        "num_envs": args.num_envs,
        "num_runs": args.num_runs,
        "mean_success_rate": mean_success,
        "per_task": per_task,
    }


def _resolve_output_dir(args: Args, script_dir: str) -> str:
    if args.output_dir is not None:
        return os.path.abspath(args.output_dir)
    return os.path.join(script_dir, "output", f"{args.policy}-{_run_label(args)}")


def main(args: Args) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    task_names = _resolve_tasks(args)
    run_label = _run_label(args)

    output_dir = _resolve_output_dir(args, script_dir)
    _ensure_output_dir_policy_compatible(output_dir, args.policy)
    os.makedirs(output_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, "parallel_logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.info(
        "Evaluating %d RoboLab tasks from %s (num_workers=%d, num_envs=%d, num_runs=%d)",
        len(task_names),
        run_label,
        args.num_workers,
        args.num_envs,
        args.num_runs,
    )
    logger.info("Per-task stdout/stderr is captured to %s", log_dir)

    results: list[dict[str, object]] = []
    results_path = os.path.join(output_dir, "results.json")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(
                _run_one_task,
                args,
                task_name,
                task_idx,
                log_dir,
                script_dir,
                output_dir,
            ): (task_idx, task_name)
            for task_idx, task_name in enumerate(task_names)
        }

        for future in concurrent.futures.as_completed(futures):
            task_idx, task_name = futures[future]
            try:
                parsed = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s crashed in worker thread: %s", task_name, exc)
                parsed = {
                    "task_name": task_name,
                    "task_idx": task_idx,
                    "num_episodes": 0,
                    "num_success": 0,
                    "success_rate": float("nan"),
                    "returncode": -1,
                    "failure_reason": str(exc),
                    "log_path": os.path.join(
                        log_dir,
                        f"task_{task_idx:02d}_{_sanitize_task_name(task_name)}.log",
                    ),
                    "output_dir": os.path.join(output_dir, task_name),
                    "episode_results_path": os.path.join(
                        output_dir, "episode_results.jsonl"
                    ),
                }

            results.append(parsed)
            results.sort(key=lambda item: int(item["task_idx"]))
            with open(results_path, "w") as file_handle:
                json.dump(results, file_handle, indent=2)

            logger.info(
                "[%s] success_rate=%.2f (%d/%d, returncode=%d)",
                task_name,
                parsed["success_rate"],
                parsed["num_success"],
                parsed["num_episodes"],
                parsed["returncode"],
            )

    valid = [
        item
        for item in results
        if not (
            isinstance(item["success_rate"], float) and math.isnan(item["success_rate"])
        )
    ]
    mean_success = (
        sum(float(item["success_rate"]) for item in valid) / len(valid)
        if valid
        else 0.0
    )

    summary = _build_final_summary(args, run_label, task_names, results, mean_success)
    with open(results_path, "w") as file_handle:
        json.dump(summary, file_handle, indent=2)

    logger.info("Results saved to %s", results_path)
    logger.info("Mean success rate: %.2f", mean_success)

    failures = [item for item in results if item.get("failure_reason") is not None]
    if failures:
        for item in failures:
            logger.error(
                "[%s] failed: %s. See %s",
                item["task_name"],
                item["failure_reason"],
                item["log_path"],
            )
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
