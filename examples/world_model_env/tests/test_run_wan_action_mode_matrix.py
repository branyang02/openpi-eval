from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

WORLD_MODEL_ENV_DIR = Path(__file__).resolve().parents[1]
if str(WORLD_MODEL_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_ENV_DIR))

from run_wan_action_mode_matrix import compare_action_modes, main  # noqa: E402


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _basic_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    decoded = _write_json(tmp_path / "decoded.json", {"mode": "decoded_video_idm", "idm_mse": 1.0})
    current = _write_json(
        tmp_path / "current.json",
        {"mode": "current_wan_prefix_action_expert", "val_model_sample_mse": 2.0},
    )
    partial = _write_json(
        tmp_path / "partial.json",
        {"mode": "partial_wan_prefix_action_expert", "val_model_zero_noise_mse": 3.0},
    )
    return decoded, current, partial


def _rows_by_mode(payload: dict) -> dict[str, dict]:
    return {row["mode"]: row for row in payload["modes"]}


def test_comparison_normalizes_aliases_dirs_history_notes_and_timing(tmp_path: Path) -> None:
    decoded = _write_json(
        tmp_path / "decoded_eval.json",
        {
            "action_mode": "decoded_video_idm",
            "idm_mse": 0.42,
            "per_action_dim_mse": [0.1, 0.2],
            "latency_ms": 12.5,
        },
    )
    current_dir = tmp_path / "current_run"
    _write_json(
        current_dir / "metrics.json",
        {
            "wan_action_mode": "current_wan_prefix_action_expert",
            "decoder_arch": "encoder",
            "mse": 0.31,
            "val_model_sample_mse": 0.99,
            "val_model_zero_noise_mse_per_action_dim": [0.3, 0.4],
            "timestep_conditioning": "additive",
            "elapsed_seconds": 1.5,
        },
    )
    partial = _write_json(
        tmp_path / "partial_metrics.json",
        {
            "mode": "partial_wan_prefix_action_expert",
            "history": [
                {"epoch": 1, "val_model_sample_mse": 0.6},
                {"epoch": 2, "val_model_zero_noise_mse": 0.4},
                {"epoch": 3, "val_model_sample_mse": 0.5},
            ],
            "duration_seconds": 9.0,
        },
    )

    payload = compare_action_modes(
        decoded_video_idm=decoded,
        current_wan_prefix_action_expert=current_dir,
        partial_wan_prefix_action_expert=partial,
    )

    assert "not directly comparable" in payload["comparison_warning"]
    assert "same action metric" in payload["comparison_warning"]
    assert "No row reports true native Wan attention KV-cache reuse" in payload["wan_contract_warning"]
    assert "not cached Wan attention KV" in payload["wan_contract_warning"]

    rows = _rows_by_mode(payload)
    decoded_row = rows["decoded_video_idm"]
    assert decoded_row["decoder_arch"] is None
    assert decoded_row["inference_path"] == "full Wan video generation -> IDM"
    assert decoded_row["true_kv_cache"] is False
    assert decoded_row["source_path"] == str(decoded)
    assert decoded_row["metric_name"] == "idm_mse"
    assert decoded_row["metric_family"] == "idm_action_mse"
    assert decoded_row["lower_is_better"] is True
    assert decoded_row["best_mse"] == 0.42
    assert decoded_row["last_mse"] == 0.42
    assert decoded_row["best_mse_metric_name"] == "idm_mse"
    assert decoded_row["last_mse_metric_name"] == "idm_mse"
    assert decoded_row["best_mse_source_field"] == "idm_mse"
    assert decoded_row["last_mse_source_field"] == "idm_mse"
    assert decoded_row["per_dim_mse"] == [0.1, 0.2]
    assert decoded_row["timing"] == {"latency_ms": 12.5}
    assert decoded_row["notes"] == [
        "inference_path=full Wan video generation -> IDM",
        "runs_wan_generation=yes",
        "decoded_action_video=yes",
        "reusable_action_memory=no",
        "native_wan_kv_cache=no",
    ]

    current_row = rows["current_wan_prefix_action_expert"]
    assert current_row["decoder_arch"] == "encoder"
    assert current_row["inference_path"] == "current Wan prefix run once -> action expert"
    assert current_row["true_kv_cache"] is False
    assert current_row["source_path"] == str(current_dir / "metrics.json")
    assert current_row["metric_name"] == "mse"
    assert current_row["metric_family"] == "pi05_action_expert_mse"
    assert current_row["lower_is_better"] is True
    assert current_row["best_mse"] == 0.31
    assert current_row["last_mse"] == 0.31
    assert current_row["best_mse_metric_name"] == "mse"
    assert current_row["last_mse_metric_name"] == "mse"
    assert current_row["best_mse_source_field"] == "mse"
    assert current_row["last_mse_source_field"] == "mse"
    assert current_row["per_dim_mse"] == [0.3, 0.4]
    assert current_row["timing"] == {"elapsed_seconds": 1.5}
    assert current_row["notes"] == [
        "inference_path=current Wan prefix run once -> action expert",
        "runs_wan_generation=no",
        "decoded_action_video=no",
        "reusable_action_memory=yes",
        "native_wan_kv_cache=no",
    ]

    partial_row = rows["partial_wan_prefix_action_expert"]
    assert partial_row["decoder_arch"] is None
    assert partial_row["inference_path"] == "hybrid Wan future latents/memory -> action expert"
    assert partial_row["true_kv_cache"] is False
    assert partial_row["metric_name"] == "val_model_zero_noise_mse"
    assert partial_row["metric_family"] == "pi05_action_expert_mse"
    assert partial_row["lower_is_better"] is True
    assert partial_row["best_mse"] == 0.4
    assert partial_row["last_mse"] == 0.5
    assert partial_row["best_mse_metric_name"] == "val_model_zero_noise_mse"
    assert partial_row["last_mse_metric_name"] == "val_model_sample_mse"
    assert partial_row["best_mse_source_field"] == "history[1].val_model_zero_noise_mse"
    assert partial_row["last_mse_source_field"] == "history[2].val_model_sample_mse"
    assert partial_row["per_dim_mse"] is None
    assert partial_row["timing"] == {"duration_seconds": 9.0}
    assert partial_row["notes"] == [
        "inference_path=hybrid Wan future latents/memory -> action expert",
        "runs_wan_generation=yes",
        "decoded_action_video=no",
        "reusable_action_memory=yes",
        "native_wan_kv_cache=no",
    ]


def test_best_and_final_objects_supply_best_and_last_mse(tmp_path: Path) -> None:
    decoded, current, _partial = _basic_sources(tmp_path)
    partial = _write_json(
        tmp_path / "partial_nested.json",
        {
            "mode": "partial_wan_prefix_action_expert",
            "best": {"val_model_zero_noise_mse": 0.25},
            "final": {"val_model_zero_noise_mse": 0.75},
        },
    )

    payload = compare_action_modes(
        decoded_video_idm=decoded,
        current_wan_prefix_action_expert=current,
        partial_wan_prefix_action_expert=partial,
    )

    partial_row = _rows_by_mode(payload)["partial_wan_prefix_action_expert"]
    assert partial_row["metric_name"] == "val_model_zero_noise_mse"
    assert partial_row["best_mse"] == 0.25
    assert partial_row["last_mse"] == 0.75
    assert partial_row["best_mse_source_field"] == "best.val_model_zero_noise_mse"
    assert partial_row["last_mse_source_field"] == "final.val_model_zero_noise_mse"


def test_decoded_metric_prefers_idm_mse_over_generic_mse(tmp_path: Path) -> None:
    decoded = _write_json(
        tmp_path / "decoded.json",
        {
            "mode": "decoded_video_idm",
            "mse": 9.0,
            "idm_mse": 0.5,
        },
    )
    _, current, partial = _basic_sources(tmp_path / "support")

    payload = compare_action_modes(
        decoded_video_idm=decoded,
        current_wan_prefix_action_expert=current,
        partial_wan_prefix_action_expert=partial,
    )

    decoded_row = _rows_by_mode(payload)["decoded_video_idm"]
    assert decoded_row["metric_name"] == "idm_mse"
    assert decoded_row["metric_family"] == "idm_action_mse"
    assert decoded_row["best_mse"] == 0.5
    assert decoded_row["last_mse"] == 0.5
    assert decoded_row["best_mse_source_field"] == "idm_mse"


def test_dataset_action_mse_is_preferred_shared_metric_for_all_modes(tmp_path: Path) -> None:
    decoded = _write_json(
        tmp_path / "decoded.json",
        {
            "mode": "decoded_video_idm",
            "dataset_action_mse": 0.11,
            "idm_mse": 9.0,
            "dataset_action_mse_per_action_dim": [0.1, 0.2],
            "per_action_dim_mse": [9.0, 9.0],
        },
    )
    current = _write_json(
        tmp_path / "current.json",
        {
            "mode": "current_wan_prefix_action_expert",
            "dataset_action_mse": 0.22,
            "val_model_zero_noise_mse": 8.0,
            "dataset_action_mse_per_action_dim": [0.2, 0.3],
        },
    )
    partial = _write_json(
        tmp_path / "partial.json",
        {
            "mode": "partial_wan_prefix_action_expert",
            "history": [
                {"epoch": 1, "dataset_action_mse": 0.44},
                {"epoch": 2, "dataset_action_mse": 0.33, "val_model_sample_mse": 7.0},
            ],
        },
    )

    payload = compare_action_modes(
        decoded_video_idm=decoded,
        current_wan_prefix_action_expert=current,
        partial_wan_prefix_action_expert=partial,
    )

    rows = _rows_by_mode(payload)
    decoded_row = rows["decoded_video_idm"]
    assert decoded_row["metric_name"] == "dataset_action_mse"
    assert decoded_row["metric_family"] == "dataset_action_mse"
    assert decoded_row["best_mse"] == 0.11
    assert decoded_row["best_mse_source_field"] == "dataset_action_mse"
    assert decoded_row["per_dim_mse"] == [0.1, 0.2]

    current_row = rows["current_wan_prefix_action_expert"]
    assert current_row["metric_name"] == "dataset_action_mse"
    assert current_row["metric_family"] == "dataset_action_mse"
    assert current_row["best_mse"] == 0.22
    assert current_row["last_mse"] == 0.22
    assert current_row["per_dim_mse"] == [0.2, 0.3]

    partial_row = rows["partial_wan_prefix_action_expert"]
    assert partial_row["metric_name"] == "dataset_action_mse"
    assert partial_row["metric_family"] == "dataset_action_mse"
    assert partial_row["best_mse"] == 0.33
    assert partial_row["last_mse"] == 0.33
    assert partial_row["best_mse_source_field"] == "history[1].dataset_action_mse"


def test_current_prefix_decoder_arch_variants_are_distinguishable(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    decoded, _current, partial = _basic_sources(tmp_path)
    suffix_checkpoint = tmp_path / "suffix_checkpoint.pt"
    joint_checkpoint = tmp_path / "joint_checkpoint.pt"
    torch.save({"model_kwargs": {"decoder_arch": "suffix_prefix_cache"}}, suffix_checkpoint)
    torch.save({"args": {"decoder_arch": "joint_softmax_prefix_cache"}}, joint_checkpoint)
    suffix_current = _write_json(
        tmp_path / "suffix_current.json",
        {
            "mode": "current_wan_prefix_action_expert",
            "checkpoint": str(suffix_checkpoint),
            "dataset_action_mse": 0.2,
        },
    )
    joint_current = _write_json(
        tmp_path / "joint_current.json",
        {
            "mode": "current_wan_prefix_action_expert",
            "checkpoint": str(joint_checkpoint),
            "dataset_action_mse": 0.2,
        },
    )

    suffix_payload = compare_action_modes(
        decoded_video_idm=decoded,
        current_wan_prefix_action_expert=suffix_current,
        partial_wan_prefix_action_expert=partial,
    )
    joint_payload = compare_action_modes(
        decoded_video_idm=decoded,
        current_wan_prefix_action_expert=joint_current,
        partial_wan_prefix_action_expert=partial,
    )

    suffix_identity = {
        key: _rows_by_mode(suffix_payload)["current_wan_prefix_action_expert"][key]
        for key in ("mode", "decoder_arch", "inference_path")
    }
    joint_identity = {
        key: _rows_by_mode(joint_payload)["current_wan_prefix_action_expert"][key]
        for key in ("mode", "decoder_arch", "inference_path")
    }
    assert suffix_identity["mode"] == joint_identity["mode"] == "current_wan_prefix_action_expert"
    assert suffix_identity["inference_path"] == joint_identity["inference_path"]
    assert suffix_identity != joint_identity
    assert suffix_identity["decoder_arch"] == "suffix_prefix_cache"
    assert joint_identity["decoder_arch"] == "joint_softmax_prefix_cache"


def test_sample_fingerprints_match_when_explicit_metadata_matches(tmp_path: Path) -> None:
    dataset_fingerprint = {"repo_id": "metaworld", "split": "eval"}
    sample_fingerprint = {"dataset_id": "metaworld-eval", "sample_ids_sha256": "abc123"}
    decoded = _write_json(
        tmp_path / "decoded.json",
        {
            "mode": "decoded_video_idm",
            "dataset_action_mse": 0.11,
            "dataset_fingerprint": dataset_fingerprint,
            "sample_fingerprint": sample_fingerprint,
        },
    )
    current = _write_json(
        tmp_path / "current.json",
        {
            "mode": "current_wan_prefix_action_expert",
            "dataset_action_mse": 0.22,
            "dataset_fingerprint": dataset_fingerprint,
            "sample_fingerprint": sample_fingerprint,
        },
    )
    partial = _write_json(
        tmp_path / "partial.json",
        {
            "mode": "partial_wan_prefix_action_expert",
            "dataset_action_mse": 0.33,
            "metadata": {
                "dataset_fingerprint": dataset_fingerprint,
                "sample_fingerprint": sample_fingerprint,
            },
        },
    )

    payload = compare_action_modes(
        decoded_video_idm=decoded,
        current_wan_prefix_action_expert=current,
        partial_wan_prefix_action_expert=partial,
    )

    assert payload["sample_sets_match"] is True
    assert payload["sample_set_warning"] is None
    for row in payload["modes"]:
        assert row["dataset_fingerprint"] == dataset_fingerprint
        assert row["sample_fingerprint"] == sample_fingerprint


def test_sample_fingerprints_detect_different_derived_sample_sets(tmp_path: Path) -> None:
    dataset_config = {
        "repo_id": "metaworld",
        "episodes": [1, 2],
        "max_samples": 4,
        "frame_delta": 1,
    }
    common_metadata = {
        "dataset_config": dataset_config,
        "cache_path": "prefix-cache/eval",
        "cached_future_dir": "future-cache/eval",
        "num_valid_action_steps": 12,
    }
    decoded = _write_json(
        tmp_path / "decoded.json",
        {
            "mode": "decoded_video_idm",
            "dataset_action_mse": 0.11,
            **common_metadata,
            "num_samples": 4,
        },
    )
    current = _write_json(
        tmp_path / "current.json",
        {
            "mode": "current_wan_prefix_action_expert",
            "dataset_action_mse": 0.22,
            **common_metadata,
            "num_samples": 4,
        },
    )
    partial = _write_json(
        tmp_path / "partial.json",
        {
            "mode": "partial_wan_prefix_action_expert",
            "dataset_action_mse": 0.33,
            **common_metadata,
            "num_samples": 5,
        },
    )

    payload = compare_action_modes(
        decoded_video_idm=decoded,
        current_wan_prefix_action_expert=current,
        partial_wan_prefix_action_expert=partial,
    )

    expected_dataset_fingerprint = {"dataset_config": dataset_config}
    assert payload["sample_sets_match"] is False
    assert "metrics share a family but sample sets differ" in payload["sample_set_warning"]
    rows = _rows_by_mode(payload)
    assert rows["decoded_video_idm"]["dataset_fingerprint"] == expected_dataset_fingerprint
    assert rows["decoded_video_idm"]["sample_fingerprint"] == {
        "dataset_fingerprint": expected_dataset_fingerprint,
        "cache_path": "prefix-cache/eval",
        "cached_future_dir": "future-cache/eval",
        "num_samples": 4,
        "num_valid_action_steps": 12,
    }
    assert rows["partial_wan_prefix_action_expert"]["sample_fingerprint"]["num_samples"] == 5


def test_sample_fingerprints_are_unknown_when_missing(tmp_path: Path) -> None:
    decoded, current, partial = _basic_sources(tmp_path)

    payload = compare_action_modes(
        decoded_video_idm=decoded,
        current_wan_prefix_action_expert=current,
        partial_wan_prefix_action_expert=partial,
    )

    assert payload["sample_sets_match"] is None
    assert "unknown" in payload["sample_set_warning"]
    for row in payload["modes"]:
        assert row["dataset_fingerprint"] is None
        assert row["sample_fingerprint"] is None


def test_main_prints_and_optionally_writes_json(tmp_path: Path) -> None:
    decoded, current, partial = _basic_sources(tmp_path)
    output_json = tmp_path / "matrix" / "comparison.json"
    out = io.StringIO()

    assert (
        main(
            [
                "--decoded-video-idm",
                str(decoded),
                "--current-wan-prefix-action-expert",
                str(current),
                "--partial-wan-prefix-action-expert",
                str(partial),
                "--output-json",
                str(output_json),
            ],
            out=out,
        )
        == 0
    )

    printed = json.loads(out.getvalue())
    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert printed == written
    assert "same action metric" in printed["comparison_warning"]
    assert "No row reports true native Wan attention KV-cache reuse" in printed["wan_contract_warning"]
    assert [row["mode"] for row in printed["modes"]] == [
        "decoded_video_idm",
        "current_wan_prefix_action_expert",
        "partial_wan_prefix_action_expert",
    ]
    assert [row["inference_path"] for row in printed["modes"]] == [
        "full Wan video generation -> IDM",
        "current Wan prefix run once -> action expert",
        "hybrid Wan future latents/memory -> action expert",
    ]
    assert [row["true_kv_cache"] for row in printed["modes"]] == [False, False, False]


def test_declared_mode_must_match_explicit_slot(tmp_path: Path) -> None:
    decoded, current, partial = _basic_sources(tmp_path)
    mismatched_current = _write_json(
        tmp_path / "mismatched_current.json",
        {"mode": "partial_wan_prefix_action_expert", "mse": 2.0},
    )

    with pytest.raises(ValueError, match="declares Wan action mode"):
        compare_action_modes(
            decoded_video_idm=decoded,
            current_wan_prefix_action_expert=mismatched_current,
            partial_wan_prefix_action_expert=partial,
        )
    assert current.exists()


def test_output_dir_with_multiple_known_json_files_errors(tmp_path: Path) -> None:
    decoded, current, partial = _basic_sources(tmp_path)
    ambiguous_dir = tmp_path / "ambiguous"
    _write_json(ambiguous_dir / "metrics.json", {"mode": "decoded_video_idm", "mse": 1.0})
    _write_json(ambiguous_dir / "eval_metrics.json", {"mode": "decoded_video_idm", "mse": 1.0})

    with pytest.raises(ValueError, match="multiple candidate result JSON"):
        compare_action_modes(
            decoded_video_idm=ambiguous_dir,
            current_wan_prefix_action_expert=current,
            partial_wan_prefix_action_expert=partial,
        )
    assert decoded.exists()


def test_output_dir_without_known_json_file_errors(tmp_path: Path) -> None:
    decoded, current, partial = _basic_sources(tmp_path)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="does not contain exactly one known result JSON"):
        compare_action_modes(
            decoded_video_idm=decoded,
            current_wan_prefix_action_expert=current,
            partial_wan_prefix_action_expert=empty_dir,
        )
