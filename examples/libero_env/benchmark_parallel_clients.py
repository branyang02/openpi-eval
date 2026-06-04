"""Benchmark many LIBERO simulator clients against one policy server.

This is an end-to-end stress runner: each client is an independent ``main.py``
subprocess with its own MuJoCo/EGL context, image observations, policy-server
requests, and environment actions. Unlike ``eval_all.py``, this intentionally
allows repeated task IDs so you can launch more clients than the suite has
unique tasks.

Example:
    MUJOCO_GL=egl uv run python benchmark_parallel_clients.py \
        --num_clients 64 \
        --max_workers 64 \
        --task_suite_name libero_spatial \
        --max_steps 50 \
        --video_mode none \
        --output_dir /tmp/libero_e2e_64
"""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import json
import logging
import math
import os
import re
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Literal, Optional, Pattern

import numpy as np
import tyro
from openpi_client import websocket_client_policy

from main import get_task_suite

logger = logging.getLogger(__name__)

SUCCESS_RATE_RE = re.compile(r"success_rate=([0-9.]+)")
POLICY_CALLS_RE = re.compile(r"policy_calls=(\d+)")
ENV_STEPS_RE = re.compile(r"env_steps=(\d+)")
METRICS_JSON_RE = re.compile(r"metrics_json=(\{.*\})")


@dataclasses.dataclass(frozen=True)
class ClientSpec:
    client_id: int
    task_id: int
    seed: int
    task_name: str
    task_description: str


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # LIBERO suite name.
    task_suite_name: Literal[
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    ] = "libero_spatial"
    # Number of independent main.py subprocess clients to launch. Task IDs are
    # cycled when this is larger than the suite's task count.
    num_clients: int = 64
    # Maximum subprocesses to run concurrently. Defaults to num_clients.
    max_workers: Optional[int] = None

    # Number of episodes / initial states per subprocess.
    num_episodes: int = 1
    # Short horizon by default because this is a throughput benchmark, not a
    # full success-rate evaluation. Set to None only when you want suite
    # defaults from main.py.
    max_steps: Optional[int] = 50
    # Number of settling steps before policy actions.
    num_steps_wait: int = 1
    # Number of steps to execute from the model's action plan before re-planning.
    replan_steps: int = 5

    # Image resize size for the policy input.
    resize_size: int = 224

    # Cameras to tile into videos when video_mode=record. Policy observations
    # still render LIBERO's agentview and wrist images even when video_mode=none.
    render_cameras: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "eye_in_hand"]
    )

    fps: int = 10
    video_mode: Literal["record", "none"] = "none"
    progress_mode: Literal["auto", "off"] = "off"
    seed: int = 7
    mujoco_gl: str = "egl"
    # Forwarded to main.py / robosuite. -1 keeps robosuite's default device.
    render_gpu_device_id: int = -1
    # Delay between launching subprocesses. Useful when many EGL contexts fail
    # if they are created at exactly the same time.
    launch_stagger_s: float = 0.0
    # If set, clients wait until this many seconds after benchmark start once
    # their env has reset, so policy requests still arrive as a concurrent wave.
    synchronized_start_delay_s: Optional[float] = None

    # Kill an individual client if it exceeds this many seconds. None disables
    # subprocess-level timeouts.
    client_timeout_s: Optional[float] = None

    # Top-level benchmark directory. Relative paths are resolved against the
    # user's shell cwd.
    output_dir: Optional[str] = None


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = round((len(values) - 1) * percentile / 100.0)
    return sorted(values)[index]


def _make_client_specs(args: Args) -> List[ClientSpec]:
    task_suite = get_task_suite(args.task_suite_name)
    specs = []
    for client_id in range(args.num_clients):
        task_id = client_id % task_suite.n_tasks
        task = task_suite.get_task(task_id)
        specs.append(
            ClientSpec(
                client_id=client_id,
                task_id=task_id,
                seed=args.seed + client_id * args.num_episodes,
                task_name=getattr(task, "name", f"task_{task_id:02d}"),
                task_description=str(task.language),
            )
        )
    return specs


def _build_command(
    args: Args,
    spec: ClientSpec,
    output_dir: str,
    *,
    start_after_unix_s: Optional[float] = None,
) -> List[str]:
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
        str(spec.task_id),
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
        "--video_mode",
        args.video_mode,
        "--progress_mode",
        args.progress_mode,
        "--seed",
        str(spec.seed),
        "--render_gpu_device_id",
        str(args.render_gpu_device_id),
        "--output_dir",
        output_dir,
    ]
    if start_after_unix_s is not None:
        cmd.extend(["--start_after_unix_s", "{:.6f}".format(start_after_unix_s)])
    if args.render_cameras:
        cmd.append("--render_cameras")
        cmd.extend(args.render_cameras)
    if args.max_steps is not None:
        cmd.extend(["--max_steps", str(args.max_steps)])
    return cmd


def _last_int(pattern: Pattern[str], text: str) -> Optional[int]:
    matches = pattern.findall(text)
    return int(matches[-1]) if matches else None


def _last_float(pattern: Pattern[str], text: str) -> Optional[float]:
    matches = pattern.findall(text)
    return float(matches[-1]) if matches else None


def _last_json(pattern: Pattern[str], text: str) -> Optional[Dict[str, object]]:
    matches = pattern.findall(text)
    if not matches:
        return None
    return json.loads(matches[-1])


def _run_one_client(
    args: Args,
    spec: ClientSpec,
    *,
    script_dir: str,
    log_dir: str,
    output_dir: str,
    start_after_unix_s: Optional[float],
) -> Dict[str, object]:
    client_output_dir = os.path.join(
        output_dir,
        "clients",
        "client_{:03d}_task_{:02d}".format(spec.client_id, spec.task_id),
    )
    os.makedirs(client_output_dir, exist_ok=True)

    if args.launch_stagger_s > 0:
        time.sleep(spec.client_id * args.launch_stagger_s)

    cmd = _build_command(
        args,
        spec,
        client_output_dir,
        start_after_unix_s=start_after_unix_s,
    )
    env = os.environ.copy()
    env["MUJOCO_GL"] = args.mujoco_gl

    log_path = os.path.join(log_dir, "client_{:03d}.log".format(spec.client_id))
    start = time.perf_counter()
    timed_out = False
    returncode = 0
    with open(log_path, "w") as log_file:
        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=script_dir,
                check=False,
                timeout=args.client_timeout_s,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1
            log_file.write(
                "\nTIMED_OUT after {:.1f}s\n".format(args.client_timeout_s or 0.0)
            )

    wall_s = time.perf_counter() - start
    with open(log_path) as log_file:
        log_text = log_file.read()
    metrics = _last_json(METRICS_JSON_RE, log_text) or {}

    return {
        "client_id": spec.client_id,
        "task_id": spec.task_id,
        "task_name": spec.task_name,
        "task_description": spec.task_description,
        "seed": spec.seed,
        "success_rate": metrics.get(
            "success_rate", _last_float(SUCCESS_RATE_RE, log_text)
        ),
        "policy_calls": metrics.get(
            "policy_calls", _last_int(POLICY_CALLS_RE, log_text)
        ),
        "env_steps": metrics.get("env_steps", _last_int(ENV_STEPS_RE, log_text)),
        "server_batch_size_counts": metrics.get("server_batch_size_counts", {}),
        "server_padded_batch_size_counts": metrics.get(
            "server_padded_batch_size_counts", {}
        ),
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_s": wall_s,
        "log_path": log_path,
        "output_dir": client_output_dir,
        "command": cmd,
    }


def _summarize_results(
    args: Args,
    *,
    results: List[Dict[str, object]],
    wall_s: float,
    server_metadata: Dict[str, object],
) -> Dict[str, object]:
    client_wall_s = [
        float(result["wall_s"])
        for result in results
        if not bool(result["timed_out"]) and int(result["returncode"]) == 0
    ]
    success_rates = [
        float(result["success_rate"])
        for result in results
        if result["success_rate"] is not None
    ]
    total_policy_calls = sum(
        int(result["policy_calls"])
        for result in results
        if result["policy_calls"] is not None
    )
    total_env_steps = sum(
        int(result["env_steps"])
        for result in results
        if result["env_steps"] is not None
    )
    returncode_counts = collections.Counter(
        str(result["returncode"]) for result in results
    )

    def merge_counts(key: str) -> Dict[str, int]:
        merged = collections.Counter()
        for result in results:
            counts = result.get(key) or {}
            for batch_size, count in counts.items():
                merged[str(batch_size)] += int(count)
        return dict(sorted(merged.items(), key=lambda item: int(item[0])))

    return {
        "benchmark": "libero_parallel_clients",
        "task_suite_name": args.task_suite_name,
        "num_clients": args.num_clients,
        "max_workers": args.max_workers or args.num_clients,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "num_steps_wait": args.num_steps_wait,
        "replan_steps": args.replan_steps,
        "video_mode": args.video_mode,
        "mujoco_gl": args.mujoco_gl,
        "render_gpu_device_id": args.render_gpu_device_id,
        "launch_stagger_s": args.launch_stagger_s,
        "synchronized_start_delay_s": args.synchronized_start_delay_s,
        "server_metadata": server_metadata,
        "aggregate": {
            "wall_s": wall_s,
            "completed_clients": sum(
                1
                for result in results
                if not bool(result["timed_out"]) and int(result["returncode"]) == 0
            ),
            "timed_out_clients": sum(1 for result in results if result["timed_out"]),
            "returncode_counts": dict(sorted(returncode_counts.items())),
            "server_batch_size_counts": merge_counts("server_batch_size_counts"),
            "server_padded_batch_size_counts": merge_counts(
                "server_padded_batch_size_counts"
            ),
            "mean_success_rate": (
                statistics.fmean(success_rates) if success_rates else None
            ),
            "total_policy_calls": total_policy_calls,
            "policy_calls_per_s": total_policy_calls / wall_s if wall_s > 0 else 0.0,
            "total_env_steps": total_env_steps,
            "env_steps_per_s": total_env_steps / wall_s if wall_s > 0 else 0.0,
            "client_wall_s": {
                "mean": statistics.fmean(client_wall_s) if client_wall_s else 0.0,
                "p50": _percentile(client_wall_s, 50),
                "p95": _percentile(client_wall_s, 95),
                "max": max(client_wall_s, default=0.0),
            },
        },
        "clients": sorted(results, key=lambda result: int(result["client_id"])),
    }


def _write_results(path: str, summary: Dict[str, object]) -> None:
    with open(path, "w") as file_handle:
        json.dump(summary, file_handle, indent=2)
        file_handle.write("\n")


def main(args: Args) -> None:
    if args.num_clients <= 0:
        raise ValueError("--num_clients must be positive")
    if args.max_workers is not None and args.max_workers <= 0:
        raise ValueError("--max_workers must be positive")
    if args.num_episodes <= 0:
        raise ValueError("--num_episodes must be positive")
    if args.client_timeout_s is not None and args.client_timeout_s <= 0:
        raise ValueError("--client_timeout_s must be positive when set")
    if args.launch_stagger_s < 0:
        raise ValueError("--launch_stagger_s must be non-negative")
    if (
        args.synchronized_start_delay_s is not None
        and args.synchronized_start_delay_s <= 0
    ):
        raise ValueError("--synchronized_start_delay_s must be positive when set")

    np.random.seed(args.seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(
            script_dir,
            "output",
            "benchmark_parallel_clients",
            "{}-{}c-{}".format(args.task_suite_name, args.num_clients, stamp),
        )
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "parallel_logs")
    os.makedirs(log_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()

    specs = _make_client_specs(args)
    max_workers = args.max_workers or args.num_clients
    start_after_unix_s = (
        time.time() + args.synchronized_start_delay_s
        if args.synchronized_start_delay_s is not None
        else None
    )
    logger.info(
        "Launching %d LIBERO clients against %s:%d (max_workers=%d, suite=%s)",
        args.num_clients,
        args.host,
        args.port,
        max_workers,
        args.task_suite_name,
    )
    logger.info("Per-client logs: %s", log_dir)

    results: List[Dict[str, object]] = []
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_one_client,
                args,
                spec,
                script_dir=script_dir,
                log_dir=log_dir,
                output_dir=output_dir,
                start_after_unix_s=start_after_unix_s,
            ): spec
            for spec in specs
        }
        for future in concurrent.futures.as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "client_%03d crashed in worker: %s", spec.client_id, exc
                )
                result = {
                    "client_id": spec.client_id,
                    "task_id": spec.task_id,
                    "task_name": spec.task_name,
                    "task_description": spec.task_description,
                    "seed": spec.seed,
                    "success_rate": None,
                    "policy_calls": None,
                    "env_steps": None,
                    "server_batch_size_counts": {},
                    "server_padded_batch_size_counts": {},
                    "returncode": -1,
                    "timed_out": False,
                    "wall_s": 0.0,
                    "log_path": os.path.join(
                        log_dir, "client_{:03d}.log".format(spec.client_id)
                    ),
                    "output_dir": "",
                    "command": [],
                }
            results.append(result)
            summary = _summarize_results(
                args,
                results=results,
                wall_s=time.perf_counter() - start,
                server_metadata=server_metadata,
            )
            _write_results(results_path, summary)
            logger.info(
                "client_%03d task_%02d rc=%s success=%s wall=%.1fs policy_calls=%s",
                spec.client_id,
                spec.task_id,
                result["returncode"],
                result["success_rate"],
                result["wall_s"],
                result["policy_calls"],
            )

    wall_s = time.perf_counter() - start
    summary = _summarize_results(
        args,
        results=results,
        wall_s=wall_s,
        server_metadata=server_metadata,
    )
    _write_results(results_path, summary)
    aggregate = summary["aggregate"]
    logger.info("Results saved to %s", results_path)
    logger.info(
        "wall=%.1fs completed=%d/%d policy_calls/s=%.2f env_steps/s=%.2f",
        aggregate["wall_s"],
        aggregate["completed_clients"],
        args.num_clients,
        aggregate["policy_calls_per_s"],
        aggregate["env_steps_per_s"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
