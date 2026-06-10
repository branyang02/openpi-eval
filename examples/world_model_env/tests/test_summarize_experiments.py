from __future__ import annotations

import json
from pathlib import Path

import pytest

from summarize_experiments import build_summary, main, render_markdown


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _idm_metrics() -> dict:
    return {
        "history": [
            {"epoch": 1, "idm_mse": 0.5, "train_loss": 0.52},
            {"epoch": 2, "idm_mse": 0.3, "train_loss": 0.31},
        ],
        "final": {"epoch": 2, "idm_mse": 0.3, "idm_smooth_l1": 0.15, "train_loss": 0.31},
        "best": {"epoch": 2, "idm_mse": 0.3, "idm_smooth_l1": 0.15},
        "model_config": {"idm_arch": "delta", "action_dim": 4},
        "training_target": "idm",
        "stopped_early": False,
    }


def _ranking_summary() -> dict:
    run_a = {
        "label": "loraA",
        "idm_decodability_gap": 0.03,
        "idm": {"idm_mse": 0.08, "idm_smooth_l1": 0.04},
        "pixel": {
            "future_mse": 0.001,
            "future_psnr": 28.8,
            "contact_sheet": "ranking/loraA/pixel/future_cache_contact_sheet.png",
        },
    }
    run_b = {
        "label": "loraB",
        "idm_decodability_gap": 0.06,
        "idm": {"idm_mse": 0.11, "idm_smooth_l1": 0.06},
        "pixel": {
            "future_mse": 0.002,
            "future_psnr": 25.1,
            "contact_sheet": "ranking/loraB/pixel/future_cache_contact_sheet.png",
        },
    }
    return {
        "rank_by": "idm_decodability_gap",
        "best": {
            "by_future_mse": "loraA",
            "by_idm_decodability_gap": "loraA",
            "by_idm_mse": "loraA",
        },
        "ground_truth_reference": {"idm_mse": 0.05, "idm_smooth_l1": 0.02, "num_samples": 64},
        "runs": [run_a, run_b],
        "ranked": [run_a, run_b],
    }


def _diagnostics() -> dict:
    return {
        "model_config": {"idm_arch": "delta"},
        "num_samples": 64,
        "num_valid_actions": 256,
        "idm_mse": 0.045,
        "idm_smooth_l1": 0.021,
        "mean_action_baseline": {"idm_mse": 0.29, "idm_smooth_l1": 0.13, "mean_action": [0.1, 0.2]},
        "future_sensitivity": {
            "zero": {"target_mse": 0.64},
            "current_repeated": {"target_mse": 0.45},
            "shuffled": {"target_mse": 0.076},
            "noise": {"target_mse": 0.53},
        },
        "error_stats": {"mean": 0.04, "std": 0.2, "mae": 0.11, "max_abs": 1.7},
        "action_trace": "diag/action_trace.png",
        "action_histograms": "diag/action_histograms.png",
    }


def _eval_idm() -> dict:
    return {
        "idm_mse": 0.37,
        "idm_smooth_l1": 0.15,
        "checkpoint": "output/run/best_idm_checkpoint.pt",
        "cached_future_dir": "output/cache",
        "flow_eval_seed": None,
        "dataset_config": {"frame_delta": 1},
    }


def _eval_minimal() -> dict:
    return {"wm_mse": 0.01, "wm_psnr": 20.0, "idm_mse": 0.4, "idm_generated_mse": 0.5}


def _future_cache() -> dict:
    return {
        "cache_dir": "output/cache_run",
        "num_samples": 16,
        "future_mse": 0.0024,
        "future_mae": 0.026,
        "future_psnr": 26.1,
        "max_abs_error": 0.83,
        "per_sample_metrics": "output/cache_run_quality/per_sample_metrics.jsonl",
        "contact_sheet": "output/cache_run_quality/future_cache_contact_sheet.png",
        "visual_indices": [0, 1, 2, 3],
    }


def test_build_summary_extracts_idm_train_run(tmp_path) -> None:
    _write(tmp_path / "idm_run" / "metrics.json", _idm_metrics())

    summary = build_summary(tmp_path)

    assert summary["counts"]["idm_train"] == 1
    entry = summary["idm_train"][0]
    assert entry["experiment"] == "idm_run"
    assert entry["path"] == "idm_run/metrics.json"
    assert entry["epochs"] == 2
    assert entry["idm_arch"] == "delta"
    assert entry["training_target"] == "idm"
    assert entry["stopped_early"] is False
    assert entry["final"]["idm_mse"] == 0.3
    assert entry["best"]["idm_mse"] == 0.3


def test_build_summary_extracts_wan_ranking(tmp_path) -> None:
    _write(tmp_path / "ranking" / "ranking_summary.json", _ranking_summary())

    summary = build_summary(tmp_path)

    assert summary["counts"]["wan_ranking"] == 1
    entry = summary["wan_ranking"][0]
    assert entry["experiment"] == "ranking"
    assert entry["rank_by"] == "idm_decodability_gap"
    assert entry["num_runs"] == 2
    assert entry["best"]["by_idm_mse"] == "loraA"
    assert entry["ground_truth_reference"]["idm_mse"] == 0.05
    top = entry["ranked"][0]
    assert top["label"] == "loraA"
    assert top["idm_decodability_gap"] == 0.03
    assert top["idm_mse"] == 0.08
    assert top["future_mse"] == 0.001
    assert top["future_psnr"] == 28.8
    # Contact-sheet paths referenced in the JSON are surfaced as visual artifacts.
    assert "ranking/loraA/pixel/future_cache_contact_sheet.png" in entry["visual_artifacts"]


def test_build_summary_extracts_diagnostics(tmp_path) -> None:
    _write(tmp_path / "diag" / "idm_diagnostics.json", _diagnostics())

    summary = build_summary(tmp_path)

    assert summary["counts"]["diagnostics"] == 1
    entry = summary["diagnostics"][0]
    assert entry["idm_mse"] == 0.045
    assert entry["num_samples"] == 64
    assert entry["num_valid_actions"] == 256
    assert entry["idm_arch"] == "delta"
    assert entry["mean_action_baseline"]["idm_mse"] == 0.29
    assert entry["future_sensitivity"]["shuffled"] == 0.076
    assert entry["error_stats"]["mae"] == 0.11
    assert "diag/action_trace.png" in entry["visual_artifacts"]
    assert "diag/action_histograms.png" in entry["visual_artifacts"]


def test_build_summary_extracts_eval_idm_metrics(tmp_path) -> None:
    _write(tmp_path / "eval_run" / "eval_metrics.json", _eval_idm())

    summary = build_summary(tmp_path)

    assert summary["counts"]["eval"] == 1
    entry = summary["eval"][0]
    assert entry["metrics"]["idm_mse"] == 0.37
    assert entry["metrics"]["idm_smooth_l1"] == 0.15
    assert entry["checkpoint"] == "output/run/best_idm_checkpoint.pt"
    assert entry["cached_future_dir"] == "output/cache"


def test_build_summary_handles_minimal_eval_metrics(tmp_path) -> None:
    _write(tmp_path / "wm_eval" / "eval_metrics.json", _eval_minimal())

    summary = build_summary(tmp_path)

    entry = summary["eval"][0]
    assert entry["metrics"]["wm_mse"] == 0.01
    assert entry["metrics"]["wm_psnr"] == 20.0
    assert entry["metrics"]["idm_generated_mse"] == 0.5
    assert "checkpoint" not in entry


def test_build_summary_extracts_future_cache(tmp_path) -> None:
    _write(tmp_path / "cache_run_quality" / "future_cache_metrics.json", _future_cache())

    summary = build_summary(tmp_path)

    assert summary["counts"]["future_cache"] == 1
    entry = summary["future_cache"][0]
    assert entry["num_samples"] == 16
    assert entry["future_mse"] == 0.0024
    assert entry["future_psnr"] == 26.1
    assert entry["cache_dir"] == "output/cache_run"
    assert entry["contact_sheet"] == "output/cache_run_quality/future_cache_contact_sheet.png"


def test_future_cache_metrics_discovered_when_nested(tmp_path) -> None:
    nested = tmp_path / "ranking" / "loraA" / "pixel" / "future_cache_metrics.json"
    _write(nested, _future_cache())

    summary = build_summary(tmp_path)

    assert summary["counts"]["future_cache"] == 1
    assert summary["future_cache"][0]["experiment"] == "ranking/loraA/pixel"
    assert summary["future_cache"][0]["path"] == "ranking/loraA/pixel/future_cache_metrics.json"


def test_visual_artifacts_collected_from_disk(tmp_path) -> None:
    run = tmp_path / "viz_run"
    _write(run / "eval_metrics.json", _eval_minimal())
    (run / "grid.png").write_bytes(b"\x89PNG")
    (run / "clip.mp4").write_bytes(b"\x00\x00\x00\x18")
    (run / "notes.txt").write_text("not an image")

    summary = build_summary(tmp_path)

    artifacts = summary["eval"][0]["visual_artifacts"]
    assert "viz_run/grid.png" in artifacts
    assert "viz_run/clip.mp4" in artifacts
    assert "viz_run/notes.txt" not in artifacts


def test_build_summary_empty_root(tmp_path) -> None:
    summary = build_summary(tmp_path)

    assert summary["counts"] == {
        "idm_train": 0,
        "wan_ranking": 0,
        "eval": 0,
        "diagnostics": 0,
        "future_cache": 0,
    }
    assert summary["idm_train"] == []
    assert summary["errors"] == []


def test_build_summary_records_malformed_json_without_crashing(tmp_path) -> None:
    _write(tmp_path / "good" / "metrics.json", _idm_metrics())
    bad = tmp_path / "bad" / "metrics.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not valid json")

    summary = build_summary(tmp_path)

    assert summary["counts"]["idm_train"] == 1
    assert summary["idm_train"][0]["experiment"] == "good"
    assert len(summary["errors"]) == 1
    assert summary["errors"][0]["path"] == "bad/metrics.json"


def test_entries_sorted_by_experiment(tmp_path) -> None:
    _write(tmp_path / "zzz" / "metrics.json", _idm_metrics())
    _write(tmp_path / "aaa" / "metrics.json", _idm_metrics())

    summary = build_summary(tmp_path)

    names = [e["experiment"] for e in summary["idm_train"]]
    assert names == ["aaa", "zzz"]


def test_render_markdown_includes_sections_and_names(tmp_path) -> None:
    _write(tmp_path / "idm_run" / "metrics.json", _idm_metrics())
    _write(tmp_path / "ranking" / "ranking_summary.json", _ranking_summary())

    summary = build_summary(tmp_path)
    markdown = render_markdown(summary)

    assert markdown.startswith("# Experiment Summary")
    assert "## IDM Training Runs (1)" in markdown
    assert "## Wan Rankings (1)" in markdown
    assert "## Visual Artifacts" in markdown
    assert "idm_run" in markdown
    assert "ranking" in markdown


def test_main_writes_json_and_markdown_files(tmp_path, capsys) -> None:
    out = tmp_path / "output"
    _write(out / "idm_run" / "metrics.json", _idm_metrics())
    json_out = tmp_path / "summary.json"
    md_out = tmp_path / "summary.md"

    rc = main(["--root", str(out), "--json-out", str(json_out), "--markdown-out", str(md_out)])

    assert rc == 0
    written = json.loads(json_out.read_text())
    assert written["counts"]["idm_train"] == 1
    assert md_out.read_text().startswith("# Experiment Summary")
    # JSON is also echoed to stdout for piping.
    assert json.loads(capsys.readouterr().out)["counts"]["idm_train"] == 1


def test_main_markdown_flag_prints_markdown(tmp_path, capsys) -> None:
    out = tmp_path / "output"
    _write(out / "idm_run" / "metrics.json", _idm_metrics())

    rc = main(["--root", str(out), "--markdown"])

    assert rc == 0
    assert capsys.readouterr().out.startswith("# Experiment Summary")


def test_main_errors_on_missing_root(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path / "does_not_exist")])
