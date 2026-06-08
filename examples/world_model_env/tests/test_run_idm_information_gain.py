from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_idm_information_gain import (
    REPORT_JSON_FILENAME,
    build_command_plan,
    build_report,
    filter_episodes_for_request,
    launch_plan,
    main,
    render_markdown_report,
    resolve_split_episodes,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_resolve_split_episodes_prefers_cli_over_cache_config(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    _write_json(cache_dir / "config.json", {"dataset_config": {"episodes": [1, 2, 3]}})

    resolved = resolve_split_episodes(
        cli_episodes=(7, 8),
        cache_config_path=cache_dir,
        split_name="train",
    )

    assert resolved == {"episodes": (7, 8), "source": "cli", "cache_config": None}


def test_resolve_split_episodes_reads_existing_cache_config(tmp_path) -> None:
    cache_config = tmp_path / "eval_cache" / "config.json"
    _write_json(cache_config, {"dataset_config": {"episodes": [11, 12]}})

    resolved = resolve_split_episodes(
        cli_episodes=None,
        cache_config_path=cache_config.parent,
        split_name="eval",
    )

    assert resolved["episodes"] == (11, 12)
    assert resolved["source"] == "cache_config"
    assert resolved["cache_config"] == str(cache_config)


def test_filter_episodes_skips_short_windows_and_tracks_unknown_lengths() -> None:
    result = filter_episodes_for_request(
        episodes=(1, 2, 3),
        episode_lengths={1: 10, 2: 6},
        frame_delta=2,
        num_future_frames=3,
        action_horizon=4,
        samples_per_episode=3,
    )

    assert result["required_future_offset"] == 6
    assert result["required_action_offset"] == 3
    assert result["required_offset"] == 6
    assert result["kept"] == [1, 3]
    assert result["unknown_length"] == [3]
    assert result["skipped"] == [
        {
            "episode": 2,
            "length": 6,
            "valid_windows": 0,
            "reason": "valid_windows=0 < required min_valid_windows=3 for required_offset=6",
        }
    ]


def test_build_command_plan_sweeps_conditionings_and_frame_deltas(tmp_path) -> None:
    plan = build_command_plan(
        output_dir=tmp_path,
        frame_deltas=(1, 4),
        future_conditionings=("current_only", "future_only", "full"),
        train_episodes=(1, 2),
        eval_episodes=(3,),
        episode_lengths={1: 20, 2: 12, 3: 20},
        script_dir="/repo/examples/world_model_env",
        dataset_source="lerobot",
        repo_id="test/repo",
        image_keys=("corner4.image",),
        samples_per_episode=2,
        num_future_frames=2,
        action_horizon=4,
        epochs=1,
        batch_size=4,
        device="cpu",
        seed=5,
    )

    assert len(plan["runs"]) == 6
    run = plan["runs"][0]
    assert run["run_name"] == "idm_fd1_current_only_seed5"
    assert run["train_episodes"] == [1, 2]
    assert run["eval_episodes"] == [3]
    assert "--idm-future-conditioning current_only" in run["train_command_display"]
    assert "--frame-delta 1" in run["train_command_display"]
    assert str(tmp_path / "idm_fd1_current_only_seed5" / "best_idm_checkpoint.pt") in run["eval_command"]
    fd4_filter = plan["episode_filters"]["train"]["4"]
    assert fd4_filter["kept"] == [1, 2]


def test_build_command_plan_skips_runs_when_filter_removes_all_train_episodes(tmp_path) -> None:
    plan = build_command_plan(
        output_dir=tmp_path,
        frame_deltas=(4,),
        future_conditionings=("full",),
        train_episodes=(1,),
        eval_episodes=(2,),
        episode_lengths={1: 5, 2: 20},
        dataset_source="lerobot",
        samples_per_episode=2,
        num_future_frames=2,
        action_horizon=4,
    )

    assert plan["runs"] == []
    assert plan["skipped_runs"] == [
        {
            "run_name": "idm_fd4_full_seed7",
            "frame_delta": 4,
            "idm_future_conditioning": "full",
            "reason": "no_train_episodes_after_filter",
        }
    ]


def test_build_command_plan_rejects_samples_per_episode_for_synthetic(tmp_path) -> None:
    with pytest.raises(ValueError, match="samples_per_episode requires dataset_source='lerobot'"):
        build_command_plan(
            output_dir=tmp_path,
            frame_deltas=(1,),
            future_conditionings=("full",),
            train_episodes=(1,),
            eval_episodes=(2,),
            samples_per_episode=2,
        )


def test_build_report_summarizes_existing_metrics_and_information_gain(tmp_path) -> None:
    plan = build_command_plan(
        output_dir=tmp_path,
        frame_deltas=(1,),
        future_conditionings=("current_only", "full"),
        train_episodes=None,
        eval_episodes=None,
        script_dir="/repo/examples/world_model_env",
    )
    for item, eval_mse, train_mse in zip(plan["runs"], (0.50, 0.35), (0.55, 0.40), strict=True):
        train_dir = Path(item["train_output_dir"])
        eval_dir = Path(item["eval_output_dir"])
        _write_json(
            train_dir / "metrics.json",
            {
                "history": [{"epoch": 1}],
                "model_config": {"idm_future_conditioning": item["idm_future_conditioning"]},
                "best": {"idm_mse": train_mse, "idm_smooth_l1": train_mse / 2},
                "final": {"idm_mse": train_mse + 0.1},
            },
        )
        _write_json(
            eval_dir / "eval_metrics.json",
            {
                "idm_mse": eval_mse,
                "idm_smooth_l1": eval_mse / 2,
                "mean_action_baseline": {"idm_mse": 0.9},
            },
        )

    report = build_report(plan)
    current, full = report["runs"]

    assert current["metrics_status"] == "complete"
    assert current["idm_mse_delta_vs_current"] == pytest.approx(0.0)
    assert full["eval_metrics"]["idm_mse"] == pytest.approx(0.35)
    assert full["idm_mse_delta_vs_current"] == pytest.approx(0.15)
    assert report["best_by_frame_delta"]["1"]["run_name"] == "idm_fd1_full_seed7"
    markdown = render_markdown_report(report)
    assert "# IDM Information Gain Report" in markdown
    assert "| 1 | full | complete | 0.35 | 0.4 | 0.15 |" in markdown


def test_launch_plan_uses_injected_runner_without_training(tmp_path) -> None:
    plan = build_command_plan(
        output_dir=tmp_path,
        frame_deltas=(1,),
        future_conditionings=("full",),
        train_episodes=None,
        eval_episodes=None,
    )
    launched = []

    def fake_runner(command, *, check):
        launched.append((command, check))

    launch_plan(plan, runner=fake_runner)

    assert len(launched) == 2
    assert launched[0][0][1].endswith("train_idm.py")
    assert launched[1][0][1].endswith("eval_idm.py")
    assert launched[0][1] is True
    assert launched[1][1] is True


def test_main_writes_plan_and_report_without_launching(tmp_path, capsys) -> None:
    rc = main(
        [
            "--output-dir",
            str(tmp_path),
            "--dataset-source",
            "lerobot",
            "--frame-deltas",
            "1",
            "--future-conditionings",
            "full",
            "--train-episodes",
            "1",
            "2",
            "--eval-episodes",
            "3",
            "--episode-length",
            "1=20",
            "--episode-length",
            "2=20",
            "--episode-length",
            "3=20",
            "--samples-per-episode",
            "2",
            "--action-horizon",
            "4",
            "--device",
            "cpu",
        ]
    )

    assert rc == 0
    plan = json.loads((tmp_path / "idm_information_gain_plan.json").read_text())
    report = json.loads((tmp_path / REPORT_JSON_FILENAME).read_text())
    stdout_report = json.loads(capsys.readouterr().out)
    assert len(plan["runs"]) == 1
    assert report["runs"][0]["metrics_status"] == "missing"
    assert stdout_report["num_planned_runs"] == 1
