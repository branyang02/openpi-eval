from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from infer_pi05_wan_action_expert import Args, main
from world_model.pi05_wan_action_expert import WanPi05ActionExpert


def _tiny_model() -> WanPi05ActionExpert:
    model = WanPi05ActionExpert(
        prefix_dim=4,
        state_dim=3,
        action_dim=2,
        action_horizon=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    return model


def _tiny_model_kwargs() -> dict[str, object]:
    return {
        "prefix_dim": 4,
        "state_dim": 3,
        "action_dim": 2,
        "action_horizon": 2,
        "hidden_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "dropout": 0.0,
        "conditioning_mode": "wan_prefix_state",
        "timestep_conditioning": "additive",
        "decoder_arch": "encoder",
    }


def _write_checkpoint(path: Path, *, wan_action_mode: str | None = None) -> None:
    model = _tiny_model()
    args = {}
    metrics = {}
    if wan_action_mode is not None:
        args["wan_action_mode"] = wan_action_mode
        metrics["wan_action_mode"] = wan_action_mode
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": _tiny_model_kwargs(),
            "args": args,
            "metrics": metrics,
            "action_normalization": {
                "enabled": True,
                "mean": torch.tensor([1.0, 2.0]),
                "std": torch.tensor([10.0, 20.0]),
            },
        },
        path,
    )


def _write_row(path: Path, *, wan_action_mode: str | None = None) -> None:
    metadata = {"task": "fake"}
    if wan_action_mode is not None:
        metadata["wan_action_mode"] = wan_action_mode
    torch.save(
        {
            "prefix_tokens": torch.zeros(3, 4),
            "state": torch.zeros(3),
            "actions": torch.zeros(2, 2),
            "action_mask": torch.ones(2),
            "task": "fake",
            "metadata": metadata,
        },
        path,
    )


def test_infer_pi05_wan_action_expert_writes_json_and_prints_mode(tmp_path, capsys) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    row_path = tmp_path / "row.pt"
    output_dir = tmp_path / "out"
    _write_checkpoint(checkpoint_path, wan_action_mode="current_wan_prefix_action_expert")
    _write_row(row_path, wan_action_mode="current_wan_prefix_action_expert")

    output = main(
        Args(
            checkpoint=str(checkpoint_path),
            prefix_cache_row=str(row_path),
            output_dir=str(output_dir),
            sample_steps=2,
            device="cpu",
            flow_seed=123,
            zero_noise=True,
        )
    )

    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((output_dir / "pi05_wan_action.json").read_text(encoding="utf-8"))
    assert printed == saved == output
    assert saved["action_chunk"] == [[1.0, 2.0], [1.0, 2.0]]
    assert saved["wan_action_mode"] == "current_wan_prefix_action_expert"
    assert saved["sample_steps"] == 2
    assert saved["noise"] == "zero"
    assert saved["flow_seed"] is None


def test_infer_pi05_wan_action_expert_propagates_row_only_action_mode(tmp_path, capsys) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    row_path = tmp_path / "row.pt"
    _write_checkpoint(checkpoint_path)
    _write_row(row_path, wan_action_mode="row_mode")

    output = main(
        Args(
            checkpoint=str(checkpoint_path),
            prefix_cache_row=str(row_path),
            output_dir=str(tmp_path / "out"),
            sample_steps=1,
            device="cpu",
            zero_noise=True,
        )
    )

    capsys.readouterr()
    assert output["wan_action_mode"] == "row_mode"


def test_infer_pi05_wan_action_expert_rejects_action_mode_disagreement(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    row_path = tmp_path / "row.pt"
    _write_checkpoint(checkpoint_path, wan_action_mode="checkpoint_mode")
    _write_row(row_path, wan_action_mode="row_mode")

    with pytest.raises(ValueError, match="wan_action_mode disagrees"):
        main(
            Args(
                checkpoint=str(checkpoint_path),
                prefix_cache_row=str(row_path),
                output_dir=str(tmp_path / "out"),
                device="cpu",
                zero_noise=True,
            )
        )


def test_infer_pi05_wan_action_expert_rejects_future_leakage_row(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    row_path = tmp_path / "row.pt"
    _write_checkpoint(checkpoint_path)
    torch.save(
        {
            "prefix_tokens": torch.zeros(3, 4),
            "state": torch.zeros(3),
            "actions": torch.zeros(2, 2),
            "future_images": torch.zeros(1, 3, 8, 8),
        },
        row_path,
    )

    with pytest.raises(ValueError, match="current-only.*future_images"):
        main(
            Args(
                checkpoint=str(checkpoint_path),
                prefix_cache_row=str(row_path),
                output_dir=str(tmp_path / "out"),
                device="cpu",
            )
        )


def test_infer_pi05_wan_action_expert_rejects_malformed_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    row_path = tmp_path / "row.pt"
    torch.save({"model_state_dict": {}}, checkpoint_path)
    _write_row(row_path)

    with pytest.raises(ValueError, match="missing required key 'model_kwargs'"):
        main(
            Args(
                checkpoint=str(checkpoint_path),
                prefix_cache_row=str(row_path),
                output_dir=str(tmp_path / "out"),
                device="cpu",
            )
        )
