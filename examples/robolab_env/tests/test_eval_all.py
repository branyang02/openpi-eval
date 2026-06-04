from __future__ import annotations

import json
import math
import pathlib
import textwrap

import pytest

import eval_all


def _default_args(**overrides) -> eval_all.Args:
    return eval_all.Args(**overrides)


def test_resolve_subset_tasks() -> None:
    args = _default_args()

    assert eval_all._resolve_tasks(args) == eval_all.SUBSET


def test_explicit_tasks_override_task_set() -> None:
    args = _default_args(task_set="all", tasks=["A", "B"])

    assert eval_all._resolve_tasks(args) == ["A", "B"]


def test_run_label_marks_explicit_task_lists() -> None:
    assert eval_all._run_label(_default_args()) == "subset"
    assert eval_all._run_label(_default_args(task_set="all")) == "all"
    assert eval_all._run_label(_default_args(tasks=["BananaInBowlTask"])) == "explicit"


def test_default_video_mode_matches_main() -> None:
    assert eval_all.Args().video_mode == "all"


def test_default_output_dir_includes_policy() -> None:
    args = _default_args(policy="pi0_fast", tasks=["BananaInBowlTask"])

    output_dir = eval_all._resolve_output_dir(args, "/tmp/robolab-example")

    assert output_dir == "/tmp/robolab-example/output/pi0_fast-explicit"


def test_discover_benchmark_tasks_from_ast(tmp_path: pathlib.Path) -> None:
    task_dir = tmp_path / "third_party" / "robolab" / "robolab" / "tasks" / "benchmark"
    task_dir.mkdir(parents=True)
    (task_dir / "b_task.py").write_text(
        textwrap.dedent(
            """\
            class Helper:
                pass

            class BTask(Task):
                pass
            """
        )
    )
    (task_dir / "a_task.py").write_text("class ATask(Task):\n    pass\n")

    assert eval_all._discover_benchmark_tasks(tmp_path) == ["ATask", "BTask"]


def test_build_command_forwards_main_args() -> None:
    args = _default_args(
        host="127.0.0.1",
        port=9000,
        policy="pi0_fast",
        task_dirs=["benchmark", "custom"],
        num_envs=8,
        num_runs=3,
        open_loop_horizon=10,
        remote_uri="wss://policy.example",
        video_mode="sensor",
        device="cuda:1",
        enable_subtask=True,
        enable_verbose=True,
        background_seed=123,
    )

    cmd = eval_all._build_command(args, "BananaInBowlTask", "/tmp/robolab-run")

    expected_pairs = {
        "--host": "127.0.0.1",
        "--port": "9000",
        "--policy": "pi0_fast",
        "--task": "BananaInBowlTask",
        "--num-envs": "8",
        "--num-runs": "3",
        "--open-loop-horizon": "10",
        "--remote-uri": "wss://policy.example",
        "--video-mode": "sensor",
        "--device": "cuda:1",
        "--output-dir": "/tmp/robolab-run",
        "--background-seed": "123",
    }
    for flag, value in expected_pairs.items():
        assert flag in cmd
        assert cmd[cmd.index(flag) + 1] == value

    assert cmd[0].endswith("python") or cmd[0].endswith("python3")
    assert cmd[1] == "main.py"
    assert cmd.count("--task-dirs") == 1
    assert cmd[cmd.index("--task-dirs") + 1 : cmd.index("--task-dirs") + 3] == [
        "benchmark",
        "custom",
    ]
    assert "--enable-subtask" in cmd
    assert "--enable-verbose" in cmd


def test_summarize_episodes() -> None:
    summary = eval_all._summarize_episodes(
        [{"success": True}, {"success": False}, {"success": True}]
    )

    assert summary == {
        "num_episodes": 3,
        "num_success": 2,
        "success_rate": 2 / 3,
    }


def test_summarize_empty_episodes_is_nan() -> None:
    summary = eval_all._summarize_episodes([])

    assert summary["num_episodes"] == 0
    assert summary["num_success"] == 0
    assert math.isnan(summary["success_rate"])


def test_run_one_task_parses_episode_results(tmp_path: pathlib.Path) -> None:
    fake_main = tmp_path / "main.py"
    fake_main.write_text(
        textwrap.dedent(
            """\
            import json
            import os
            import sys

            output_dir = sys.argv[sys.argv.index("--output-dir") + 1]
            task = sys.argv[sys.argv.index("--task") + 1]
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, task), exist_ok=True)
            with open(os.path.join(output_dir, task, "env_cfg.json"), "w") as f:
                f.write("{}")
            with open(os.path.join(output_dir, "episode_results.jsonl"), "w") as f:
                f.write(json.dumps({"env_name": task, "policy": "pi05", "episode": 0, "success": True}) + "\\n")
                f.write(json.dumps({"env_name": task, "policy": "pi05", "episode": 1, "success": False}) + "\\n")
                f.write(json.dumps({"env_name": task, "policy": "pi0_fast", "episode": 2, "success": True}) + "\\n")
                f.write(json.dumps({"env_name": "OtherTask", "policy": "pi05", "episode": 0, "success": True}) + "\\n")
            """
        )
    )

    log_dir = tmp_path / "logs"
    output_dir = tmp_path / "output"
    log_dir.mkdir()
    output_dir.mkdir()

    result = eval_all._run_one_task(
        _default_args(num_envs=2),
        "BananaInBowlTask",
        0,
        str(log_dir),
        str(tmp_path),
        str(output_dir),
    )

    assert result["task_name"] == "BananaInBowlTask"
    assert result["num_episodes"] == 2
    assert result["num_success"] == 1
    assert result["success_rate"] == 0.5
    assert result["returncode"] == 0
    assert result["expected_min_episodes"] == 2
    assert "failure_reason" not in result
    assert result["output_dir"] == str(output_dir / "BananaInBowlTask")
    assert pathlib.Path(result["log_path"]).exists()
    assert pathlib.Path(result["episode_results_path"]).exists()
    assert not (output_dir / "BananaInBowlTask" / "BananaInBowlTask").exists()


def test_run_one_task_flags_missing_episode_results(tmp_path: pathlib.Path) -> None:
    fake_main = tmp_path / "main.py"
    fake_main.write_text("import sys\nsys.exit(0)\n")

    log_dir = tmp_path / "logs"
    output_dir = tmp_path / "output"
    log_dir.mkdir()
    output_dir.mkdir()

    result = eval_all._run_one_task(
        _default_args(num_envs=2),
        "BananaInBowlTask",
        0,
        str(log_dir),
        str(tmp_path),
        str(output_dir),
    )

    assert result["returncode"] == 0
    assert result["num_episodes"] == 0
    assert result["expected_min_episodes"] == 2
    assert result["failure_reason"] == (
        "expected at least 2 matching episode result(s), found 0"
    )


def test_load_episode_results_supports_json_fallback(tmp_path: pathlib.Path) -> None:
    episodes = [{"env_name": "A", "success": True}]
    (tmp_path / "episode_results.json").write_text(json.dumps(episodes))

    assert eval_all._load_episode_results(str(tmp_path)) == episodes


def test_load_episode_results_filters_by_task(tmp_path: pathlib.Path) -> None:
    (tmp_path / "episode_results.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"env_name": "A", "success": True}),
                json.dumps({"task_name": "A", "success": False}),
                json.dumps({"env_name": "B", "success": True}),
            ]
        )
    )

    episodes = eval_all._load_episode_results(str(tmp_path), task_name="A")

    assert len(episodes) == 2
    assert [episode["success"] for episode in episodes] == [True, False]


def test_load_episode_results_filters_by_task_and_policy(
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "episode_results.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"env_name": "A", "policy": "pi05", "success": True}),
                json.dumps({"env_name": "A", "policy": "pi0_fast", "success": False}),
            ]
        )
    )

    episodes = eval_all._load_episode_results(
        str(tmp_path), task_name="A", policy="pi05"
    )

    assert len(episodes) == 1
    assert episodes[0]["policy"] == "pi05"


def test_build_final_summary_labels_explicit_task_runs() -> None:
    args = _default_args(task_set="all", tasks=["BananaInBowlTask"])
    summary = eval_all._build_final_summary(
        args=args,
        run_label="explicit",
        task_names=["BananaInBowlTask"],
        results=[
            {
                "task_name": "BananaInBowlTask",
                "task_idx": 0,
                "num_episodes": 1,
                "num_success": 1,
                "success_rate": 1.0,
                "returncode": 0,
                "log_path": "/tmp/log",
                "output_dir": "/tmp/output",
            }
        ],
        mean_success=1.0,
    )

    assert summary["task_set"] == "explicit"
    assert summary["requested_task_set"] == "all"
    assert summary["tasks"] == ["BananaInBowlTask"]
    assert summary["mean_success_rate"] == 1.0
    assert summary["per_task"] == [
        {
            "task_name": "BananaInBowlTask",
            "task_idx": 0,
            "num_episodes": 1,
            "num_success": 1,
            "success_rate": 1.0,
        }
    ]


def test_build_final_summary_keeps_failure_context() -> None:
    args = _default_args(tasks=["BananaInBowlTask"])

    summary = eval_all._build_final_summary(
        args=args,
        run_label="explicit",
        task_names=["BananaInBowlTask"],
        results=[
            {
                "task_name": "BananaInBowlTask",
                "task_idx": 0,
                "num_episodes": 0,
                "num_success": 0,
                "success_rate": float("nan"),
                "returncode": 0,
                "failure_reason": "expected at least 1 matching episode result(s), found 0",
                "log_path": "/tmp/robolab.log",
                "output_dir": "/tmp/output/BananaInBowlTask",
            }
        ],
        mean_success=0.0,
    )

    [task_summary] = summary["per_task"]
    assert task_summary["task_name"] == "BananaInBowlTask"
    assert task_summary["task_idx"] == 0
    assert task_summary["num_episodes"] == 0
    assert task_summary["num_success"] == 0
    assert math.isnan(task_summary["success_rate"])
    assert task_summary["failure_reason"] == (
        "expected at least 1 matching episode result(s), found 0"
    )
    assert task_summary["returncode"] == 0
    assert task_summary["log_path"] == "/tmp/robolab.log"


def test_main_exits_nonzero_after_failed_task(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _default_args(tasks=["BananaInBowlTask"], output_dir=str(tmp_path / "out"))

    def fake_run_one_task(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "task_name": "BananaInBowlTask",
            "task_idx": 0,
            "num_episodes": 0,
            "num_success": 0,
            "success_rate": float("nan"),
            "returncode": 0,
            "failure_reason": "expected at least 1 matching episode result(s), found 0",
            "log_path": str(tmp_path / "task.log"),
            "output_dir": str(tmp_path / "out" / "BananaInBowlTask"),
            "episode_results_path": str(tmp_path / "out" / "episode_results.jsonl"),
        }

    monkeypatch.setattr(eval_all, "_run_one_task", fake_run_one_task)

    with pytest.raises(SystemExit) as exc_info:
        eval_all.main(args)

    assert exc_info.value.code == 1
    results = json.loads((tmp_path / "out" / "results.json").read_text())
    assert results["per_task"][0]["failure_reason"] == (
        "expected at least 1 matching episode result(s), found 0"
    )
