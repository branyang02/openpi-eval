"""Tests for benchmark_parallel_clients.py's subprocess orchestration."""

from __future__ import annotations

import benchmark_parallel_clients as benchmark


class _FakeTask:
    def __init__(self, name: str, language: str) -> None:
        self.name = name
        self.language = language


class _FakeSuite:
    n_tasks = 3

    def get_task(self, task_id: int) -> _FakeTask:
        return _FakeTask(f"task_{task_id}", f"language_{task_id}")


def test_make_client_specs_cycles_tasks_and_offsets_seeds(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "get_task_suite", lambda _: _FakeSuite())

    args = benchmark.Args(num_clients=8, num_episodes=2, seed=10)
    specs = benchmark._make_client_specs(args)

    assert [spec.client_id for spec in specs] == list(range(8))
    assert [spec.task_id for spec in specs] == [0, 1, 2, 0, 1, 2, 0, 1]
    assert [spec.seed for spec in specs] == [10, 12, 14, 16, 18, 20, 22, 24]
    assert specs[2].task_name == "task_2"
    assert specs[2].task_description == "language_2"


def test_build_command_forwards_benchmark_flags() -> None:
    args = benchmark.Args(
        host="1.2.3.4",
        port=9999,
        task_suite_name="libero_spatial",
        num_episodes=3,
        max_steps=25,
        num_steps_wait=2,
        replan_steps=4,
        resize_size=128,
        fps=12,
        video_mode="none",
        progress_mode="off",
        render_gpu_device_id=0,
        render_cameras=["agentview"],
    )
    spec = benchmark.ClientSpec(
        client_id=5,
        task_id=2,
        seed=17,
        task_name="task_2",
        task_description="language_2",
    )

    cmd = benchmark._build_command(args, spec, "/tmp/client_output")

    expected_pairs = {
        "--host": "1.2.3.4",
        "--port": "9999",
        "--task_suite_name": "libero_spatial",
        "--task_id": "2",
        "--num_episodes": "3",
        "--max_steps": "25",
        "--num_steps_wait": "2",
        "--replan_steps": "4",
        "--resize_size": "128",
        "--fps": "12",
        "--video_mode": "none",
        "--progress_mode": "off",
        "--seed": "17",
        "--render_gpu_device_id": "0",
        "--output_dir": "/tmp/client_output",
    }
    for flag, value in expected_pairs.items():
        assert flag in cmd
        assert cmd[cmd.index(flag) + 1] == value

    render_idx = cmd.index("--render_cameras")
    assert cmd[render_idx + 1] == "agentview"
    assert cmd.count("--render_cameras") == 1


def test_build_command_omits_max_steps_when_none() -> None:
    spec = benchmark.ClientSpec(
        client_id=0,
        task_id=0,
        seed=7,
        task_name="task_0",
        task_description="language_0",
    )

    cmd = benchmark._build_command(
        benchmark.Args(max_steps=None), spec, "/tmp/client_output"
    )

    assert "--max_steps" not in cmd


def test_build_command_forwards_synchronized_start() -> None:
    spec = benchmark.ClientSpec(
        client_id=0,
        task_id=0,
        seed=7,
        task_name="task_0",
        task_description="language_0",
    )

    cmd = benchmark._build_command(
        benchmark.Args(), spec, "/tmp/client_output", start_after_unix_s=123.456789
    )

    assert "--start_after_unix_s" in cmd
    assert cmd[cmd.index("--start_after_unix_s") + 1] == "123.456789"


def test_summarize_results_reports_end_to_end_throughput() -> None:
    args = benchmark.Args(num_clients=2, max_workers=2, max_steps=5)
    results = [
        {
            "client_id": 0,
            "task_id": 0,
            "task_name": "task_0",
            "task_description": "language_0",
            "seed": 7,
            "success_rate": 1.0,
            "policy_calls": 3,
            "env_steps": 6,
            "server_batch_size_counts": {"1": 3},
            "server_padded_batch_size_counts": {"1": 3},
            "returncode": 0,
            "timed_out": False,
            "wall_s": 10.0,
            "log_path": "/tmp/client_000.log",
            "output_dir": "/tmp/client_000",
            "command": ["python", "main.py"],
        },
        {
            "client_id": 1,
            "task_id": 1,
            "task_name": "task_1",
            "task_description": "language_1",
            "seed": 8,
            "success_rate": 0.0,
            "policy_calls": 2,
            "env_steps": 4,
            "server_batch_size_counts": {"2": 2},
            "server_padded_batch_size_counts": {"4": 2},
            "returncode": 0,
            "timed_out": False,
            "wall_s": 12.0,
            "log_path": "/tmp/client_001.log",
            "output_dir": "/tmp/client_001",
            "command": ["python", "main.py"],
        },
    ]

    summary = benchmark._summarize_results(
        args,
        results=results,
        wall_s=5.0,
        server_metadata={"microbatch": {"max_batch_size": 32}},
    )

    assert summary["aggregate"]["completed_clients"] == 2
    assert summary["aggregate"]["mean_success_rate"] == 0.5
    assert summary["aggregate"]["total_policy_calls"] == 5
    assert summary["aggregate"]["policy_calls_per_s"] == 1.0
    assert summary["aggregate"]["total_env_steps"] == 10
    assert summary["aggregate"]["env_steps_per_s"] == 2.0
    assert summary["aggregate"]["returncode_counts"] == {"0": 2}
    assert summary["aggregate"]["server_batch_size_counts"] == {"1": 3, "2": 2}
    assert summary["aggregate"]["server_padded_batch_size_counts"] == {
        "1": 3,
        "4": 2,
    }
