from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from summarize_wan_action_modes import load_results, main, render_contract_table, render_result_table


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_list_modes_renders_contract_table() -> None:
    out = io.StringIO()

    assert main(["--list-modes"], out=out) == 0

    rendered = out.getvalue()
    assert (
        "| mode | inference_path | runs_wan_generation | decoded_action_video | consumes_future_pixels | "
        "reusable_action_memory | native_wan_kv |"
    ) in rendered
    assert "| `decoded_video_idm` | full Wan video generation -> IDM | yes | yes | yes | no | no |" in rendered
    assert (
        "| `current_wan_prefix_action_expert` | current Wan prefix run once -> action expert | no | no | no | yes | no |"
        in rendered
    )
    assert (
        "| `partial_wan_prefix_action_expert` | hybrid Wan future latents/memory -> action expert | yes | no | no | "
        "yes | no |"
        in rendered
    )


def test_result_table_uses_mode_from_json_and_optional_metrics(tmp_path: Path) -> None:
    result_path = tmp_path / "decoded.json"
    _write(
        result_path,
        {
            "wan_action_mode": "decoded_video_idm",
            "mse": 0.125,
            "per_dim_mse": [0.1, 0.2],
        },
    )

    rendered = render_result_table(load_results([result_path]))

    assert (
        "| `decoded_video_idm` | n/a | full Wan video generation -> IDM | yes | yes | yes | no | no | 0.125 | "
        "[0.1, 0.2] |"
    ) in rendered
    assert f"`{result_path}`" in rendered


def test_explicit_metrics_take_priority_over_training_aliases(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    _write(
        result_path,
        {
            "wan_action_mode": "decoded_video_idm",
            "mse": 0.1,
            "val_model_sample_mse": 0.2,
            "val_model_zero_noise_mse": 0.3,
            "model_sample_mse": 0.4,
            "model_zero_noise_mse": 0.5,
            "per_dim_mse": [1.0, 2.0],
            "val_model_zero_noise_mse_per_action_dim": [3.0, 4.0],
            "model_zero_noise_mse_per_action_dim": [5.0, 6.0],
            "per_action_dim_mse": [7.0, 8.0],
        },
    )

    (row,) = load_results([result_path])

    assert row.mse == 0.1
    assert row.per_dim_mse == [1.0, 2.0]


@pytest.mark.parametrize(
    ("metric_key", "expected"),
    [
        ("val_model_sample_mse", 0.2),
        ("val_model_zero_noise_mse", 0.3),
        ("model_sample_mse", 0.4),
        ("model_zero_noise_mse", 0.5),
    ],
)
def test_mse_uses_training_aliases_in_priority_order(tmp_path: Path, metric_key: str, expected: float) -> None:
    result_path = tmp_path / "result.json"
    payload = {
        "wan_action_mode": "decoded_video_idm",
        "val_model_sample_mse": 0.2,
        "val_model_zero_noise_mse": 0.3,
        "model_sample_mse": 0.4,
        "model_zero_noise_mse": 0.5,
    }
    keys_before_selected = list(payload)[1 : list(payload).index(metric_key)]
    for key in keys_before_selected:
        payload.pop(key)
    _write(result_path, payload)

    (row,) = load_results([result_path])

    assert row.mse == expected


@pytest.mark.parametrize(
    ("metric_key", "expected"),
    [
        ("val_model_zero_noise_mse_per_action_dim", [0.1, 0.2]),
        ("model_zero_noise_mse_per_action_dim", [0.3, 0.4]),
        ("per_action_dim_mse", [0.5, 0.6]),
    ],
)
def test_per_dim_mse_uses_training_aliases_in_priority_order(
    tmp_path: Path, metric_key: str, expected: list[float]
) -> None:
    result_path = tmp_path / "result.json"
    payload = {
        "wan_action_mode": "decoded_video_idm",
        "val_model_zero_noise_mse_per_action_dim": [0.1, 0.2],
        "model_zero_noise_mse_per_action_dim": [0.3, 0.4],
        "per_action_dim_mse": [0.5, 0.6],
    }
    keys_before_selected = list(payload)[1 : list(payload).index(metric_key)]
    for key in keys_before_selected:
        payload.pop(key)
    _write(result_path, payload)

    (row,) = load_results([result_path])

    assert row.per_dim_mse == expected


def test_real_training_style_metrics_are_summarized(tmp_path: Path) -> None:
    result_path = tmp_path / "metrics.json"
    _write(
        result_path,
        {
            "mode": "current_wan_prefix_action_expert",
            "epoch": 3,
            "train_loss": 0.42,
            "val_model_zero_noise_mse": 0.125,
            "val_model_zero_noise_mse_per_action_dim": [0.1, 0.15, 0.2],
        },
    )

    rendered = render_result_table(load_results([result_path]))

    assert (
        "| `current_wan_prefix_action_expert` | n/a | current Wan prefix run once -> action expert | no | no | no | "
        "yes | no | "
        "0.125 | [0.1, 0.15, 0.2] |"
    ) in rendered


def test_result_table_keeps_decoder_arch_for_same_current_mode(tmp_path: Path) -> None:
    suffix = tmp_path / "suffix.json"
    joint = tmp_path / "joint.json"
    _write(
        suffix,
        {
            "wan_action_mode": "current_wan_prefix_action_expert",
            "decoder_arch": "suffix_prefix_cache",
            "mse": 0.2,
        },
    )
    _write(
        joint,
        {
            "wan_action_mode": "current_wan_prefix_action_expert",
            "decoder_arch": "joint_softmax_prefix_cache",
            "mse": 0.2,
        },
    )

    rows = load_results([suffix, joint])
    rendered = render_result_table(rows)

    assert [row.mode.value for row in rows] == [
        "current_wan_prefix_action_expert",
        "current_wan_prefix_action_expert",
    ]
    assert [row.decoder_arch for row in rows] == ["suffix_prefix_cache", "joint_softmax_prefix_cache"]
    assert "| mode | decoder_arch | inference_path |" in rendered
    assert "| `current_wan_prefix_action_expert` | suffix_prefix_cache |" in rendered
    assert "| `current_wan_prefix_action_expert` | joint_softmax_prefix_cache |" in rendered


def test_checkpoint_style_decoder_arch_metadata_is_summarized(tmp_path: Path) -> None:
    result_path = tmp_path / "checkpoint_metadata.json"
    _write(
        result_path,
        {
            "mode": "current_wan_prefix_action_expert",
            "model_kwargs": {"decoder_arch": "suffix_prefix_cache"},
            "args": {"decoder_arch": "suffix_prefix_cache"},
            "metrics": {"decoder_arch": "suffix_prefix_cache"},
            "val_model_sample_mse": 0.25,
        },
    )

    (row,) = load_results([result_path])

    assert row.decoder_arch == "suffix_prefix_cache"


def test_standalone_dataset_action_eval_metrics_are_summarized(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_kwargs": {"decoder_arch": "joint_softmax_prefix_cache"}}, checkpoint_path)
    result_path = tmp_path / "eval_metrics.json"
    _write(
        result_path,
        {
            "wan_action_mode": "current_wan_prefix_action_expert",
            "checkpoint": str(checkpoint_path),
            "dataset_action_mse": 0.25,
            "dataset_action_mse_per_action_dim": [0.1, 0.2, 0.3],
        },
    )

    rows = load_results([result_path])
    rendered = render_result_table(rows)
    (row,) = rows

    assert row.decoder_arch == "joint_softmax_prefix_cache"
    assert (
        "| `current_wan_prefix_action_expert` | joint_softmax_prefix_cache | current Wan prefix run once -> "
        "action expert | no | no | no | yes | no | 0.25 | [0.1, 0.2, 0.3] |"
    ) in rendered


def test_decoded_mode_does_not_load_checkpoint_for_decoder_arch(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "decoded_checkpoint.pt"
    checkpoint_path.write_text("not a torch checkpoint", encoding="utf-8")
    result_path = tmp_path / "decoded_eval_metrics.json"
    _write(
        result_path,
        {
            "mode": "decoded_video_idm",
            "checkpoint": str(checkpoint_path),
            "idm_mse": 0.25,
        },
    )

    (row,) = load_results([result_path])

    assert row.decoder_arch is None


def test_single_explicit_mode_applies_to_all_paths(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _write(first, {"mse": 1.25})
    _write(second, {"per_dim_mse": [2.0, 3.0]})

    rows = load_results([first, second], modes=["current_wan_prefix_action_expert"])

    assert [row.mode.value for row in rows] == [
        "current_wan_prefix_action_expert",
        "current_wan_prefix_action_expert",
    ]
    assert rows[0].mse == 1.25
    assert rows[1].per_dim_mse == [2.0, 3.0]


def test_explicit_modes_can_be_passed_once_per_path(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _write(first, {})
    _write(second, {})

    rows = load_results(
        [first, second],
        modes=["decoded_video_idm", "partial_wan_prefix_action_expert"],
    )

    assert [row.mode.value for row in rows] == ["decoded_video_idm", "partial_wan_prefix_action_expert"]


def test_missing_mode_without_explicit_label_errors(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    _write(result_path, {"mse": 0.5})

    with pytest.raises(ValueError, match="missing Wan action mode"):
        load_results([result_path])


def test_unknown_json_mode_errors(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    _write(result_path, {"mode": "bogus"})

    with pytest.raises(ValueError, match="unknown Wan action mode 'bogus'"):
        load_results([result_path])


def test_mode_count_must_match_path_count(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _write(first, {})
    _write(second, {})

    with pytest.raises(ValueError, match="once or once per result path"):
        load_results(
            [first, second],
            modes=[
                "decoded_video_idm",
                "current_wan_prefix_action_expert",
                "partial_wan_prefix_action_expert",
            ],
        )


def test_main_renders_empty_result_table() -> None:
    out = io.StringIO()

    assert main([], out=out) == 0

    assert out.getvalue() == (
        "| mode | decoder_arch | inference_path | runs_wan_generation | decoded_action_video | consumes_future_pixels | "
        "reusable_action_memory | native_wan_kv | mse | per_dim_mse | result path |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )


def test_contract_renderer_accepts_default_specs() -> None:
    assert render_contract_table().startswith("| mode | inference_path |")
