from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

WORLD_MODEL_ENV_DIR = Path(__file__).resolve().parents[1]
if str(WORLD_MODEL_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_ENV_DIR))

from eval_idm import _build_sample_fingerprints as _build_idm_sample_fingerprints  # noqa: E402
from eval_pi05_wan_action_expert import Args, _build_sample_fingerprints, main  # noqa: E402
from world_model.pi05_wan_action_expert import WanPi05ActionExpert  # noqa: E402

WAN_ACTION_MODE = "current_wan_prefix_action_expert"


def _tiny_model(decoder_arch: str = "encoder") -> WanPi05ActionExpert:
    model = WanPi05ActionExpert(
        prefix_dim=4,
        state_dim=3,
        action_dim=2,
        action_horizon=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        decoder_arch=decoder_arch,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def _tiny_model_kwargs(decoder_arch: str = "encoder") -> dict[str, object]:
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
        "decoder_arch": decoder_arch,
    }


def _dataset_config_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": "lerobot",
        "repo_id": "brandonyang/metaworld_ml45",
        "image_keys": ["corner4.image"],
        "state_key": "observation.state",
        "action_key": "actions",
        "task_key": "task",
        "frame_delta": 1,
        "num_future_frames": 1,
        "action_horizon": 2,
        "image_size": 32,
        "max_samples": 2,
        "samples_per_episode": None,
        "episodes": [3, 5],
        "seed": 11,
    }
    payload.update(overrides)
    return payload


def _write_cache_config(cache_dir: Path, dataset_config: dict[str, object]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "config.json").write_text(
        json.dumps(
            {
                "cache_kind": "pi05_wan_current_prefix",
                "dataset_config": dataset_config,
                "prefix_compression": {"image": "mode-specific and ignored"},
                "future_provider_path": "mode-specific-and-ignored",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_checkpoint(
    path: Path,
    *,
    normalize_actions: bool = False,
    per_task_normalization: bool = False,
    wan_action_mode: str = WAN_ACTION_MODE,
    decoder_arch: str = "encoder",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    action_normalization: dict[str, object] = {"enabled": False}
    if per_task_normalization:
        action_normalization = {
            "enabled": True,
            "scope": "per_task",
            "tasks": {
                "fake": {
                    "mean": torch.tensor([1.0, 2.0]),
                    "std": torch.tensor([10.0, 20.0]),
                }
            },
        }
    elif normalize_actions:
        action_normalization = {
            "enabled": True,
            "scope": "global",
            "mean": torch.tensor([1.0, 2.0]),
            "std": torch.tensor([10.0, 20.0]),
        }
    torch.save(
        {
            "model_state_dict": _tiny_model(decoder_arch=decoder_arch).state_dict(),
            "model_kwargs": _tiny_model_kwargs(decoder_arch=decoder_arch),
            "args": {"wan_action_mode": wan_action_mode},
            "metrics": {"wan_action_mode": wan_action_mode},
            "action_normalization": action_normalization,
        },
        path,
    )


def _write_row(
    path: Path,
    *,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    task: str = "fake",
    wan_action_mode: str = WAN_ACTION_MODE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prefix_tokens": torch.zeros(3, 4),
            "state": torch.zeros(3),
            "actions": actions.to(dtype=torch.float32),
            "action_mask": action_mask.to(dtype=torch.float32),
            "task": task,
            "metadata": {"task": task, "wan_action_mode": wan_action_mode},
        },
        path,
    )


def test_eval_pi05_wan_action_expert_masks_actions_mean_baseline_and_writes_json(tmp_path, capsys) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out"
    _write_checkpoint(checkpoint_path)
    _write_row(
        cache_dir / "row_000.pt",
        actions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        action_mask=torch.tensor([1.0, 0.0]),
    )
    _write_row(
        cache_dir / "row_001.pt",
        actions=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        action_mask=torch.tensor([1.0, 1.0]),
    )

    output = main(
        Args(
            checkpoint=str(checkpoint_path),
            cache_path=str(cache_dir),
            output_dir=str(output_dir),
            sample_steps=2,
            batch_size=1,
            device="cpu",
        )
    )

    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((output_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    assert printed == saved == output
    assert saved["dataset_action_mse"] == pytest.approx(179.0 / 6.0)
    assert saved["dataset_action_smooth_l1"] == pytest.approx(26.0 / 6.0)
    assert saved["dataset_action_mse_per_action_dim"] == pytest.approx([25.0, 104.0 / 3.0])
    assert saved["mean_action_baseline"]["dataset_action_mse"] == pytest.approx(56.0 / 9.0)
    assert saved["mean_action_baseline"]["dataset_action_smooth_l1"] == pytest.approx(47.0 / 27.0)
    assert saved["mean_action_baseline"]["dataset_action_mse_per_action_dim"] == pytest.approx(
        [56.0 / 9.0, 56.0 / 9.0]
    )
    assert saved["mean_action_baseline"]["mean_action"] == pytest.approx([13.0 / 3.0, 16.0 / 3.0])
    assert saved["num_samples"] == 2
    assert saved["num_valid_action_steps"] == 3
    assert saved["checkpoint"] == str(checkpoint_path)
    assert saved["cache_path"] == str(cache_dir)
    assert saved["sample_steps"] == 2
    assert saved["zero_noise"] is True
    assert saved["flow_seed"] is None
    assert saved["metric_family"] == "dataset_action_mse"
    assert saved["eval_device"] == "cpu"
    assert saved["eval_elapsed_ms"] >= 0.0
    assert saved["eval_ms_per_sample"] == pytest.approx(saved["eval_elapsed_ms"] / saved["num_samples"])
    assert saved["wan_action_mode"] == WAN_ACTION_MODE
    assert "dataset_fingerprint" not in saved
    assert "sample_fingerprint" not in saved


def test_eval_pi05_wan_action_expert_loads_suffix_prefix_cache_checkpoint(
    tmp_path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    ) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    cache_dir = tmp_path / "cache"
    calls = 0
    original_forward_with_action_context = WanPi05ActionExpert.forward_with_action_context

    def recording_forward_with_action_context(
        self: WanPi05ActionExpert,
        context,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_forward_with_action_context(self, context, noisy_actions, time)

    monkeypatch.setattr(
        WanPi05ActionExpert,
        "forward_with_action_context",
        recording_forward_with_action_context,
    )
    _write_checkpoint(checkpoint_path, decoder_arch="suffix_prefix_cache")
    _write_row(
        cache_dir / "row_000.pt",
        actions=torch.zeros(2, 2),
        action_mask=torch.ones(2),
    )

    output = main(
        Args(
            checkpoint=str(checkpoint_path),
            cache_path=str(cache_dir),
            output_dir=str(tmp_path / "out"),
            sample_steps=2,
            device="cpu",
        )
    )

    capsys.readouterr()
    assert output["num_samples"] == 1
    assert output["dataset_action_mse"] == pytest.approx(0.0)
    assert calls == 2


def test_sample_fingerprint_helpers_match_when_future_only_fields_differ() -> None:
    decoded_config = _dataset_config_payload(
        num_future_frames=4,
        cached_future_dir="decoded-future-cache",
        checkpoint="idm.pt",
    )
    prefix_config = _dataset_config_payload(
        num_future_frames=1,
        prefix_compression={"image": "current-only wan prefix"},
        future_provider_path="wan-prefix-cache",
    )

    decoded = _build_idm_sample_fingerprints(decoded_config, num_samples=7)
    prefix = _build_sample_fingerprints(prefix_config, num_samples=7)

    assert decoded == prefix
    fingerprint_config = decoded["dataset_fingerprint"]["dataset_config"]
    assert "num_future_frames" not in fingerprint_config
    assert "cached_future_dir" not in fingerprint_config
    assert "checkpoint" not in fingerprint_config
    assert "prefix_compression" not in fingerprint_config
    assert "future_provider_path" not in fingerprint_config


def test_eval_pi05_wan_action_expert_writes_sample_fingerprints_from_cache_config(tmp_path, capsys) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out"
    dataset_config = _dataset_config_payload(num_future_frames=8)
    _write_checkpoint(checkpoint_path)
    _write_cache_config(cache_dir, dataset_config)
    _write_row(
        cache_dir / "row_000.pt",
        actions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        action_mask=torch.ones(2),
    )

    output = main(
        Args(
            checkpoint=str(checkpoint_path),
            cache_path=str(cache_dir),
            output_dir=str(output_dir),
            sample_steps=1,
            device="cpu",
        )
    )

    capsys.readouterr()
    saved = json.loads((output_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    expected_fingerprints = _build_sample_fingerprints(dataset_config, num_samples=1)

    assert saved == output
    assert saved["dataset_fingerprint"] == expected_fingerprints["dataset_fingerprint"]
    assert saved["sample_fingerprint"] == expected_fingerprints["sample_fingerprint"]
    assert saved["num_samples"] == 1
    assert "num_future_frames" not in saved["dataset_fingerprint"]["dataset_config"]
    assert "prefix_compression" not in json.dumps(saved["sample_fingerprint"], sort_keys=True)


def test_eval_pi05_wan_action_expert_rejects_cache_config_missing_dataset_config(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    cache_dir = tmp_path / "cache"
    _write_checkpoint(checkpoint_path)
    _write_row(
        cache_dir / "row_000.pt",
        actions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        action_mask=torch.ones(2),
    )
    (cache_dir / "config.json").write_text(json.dumps({"cache_kind": "pi05_wan_current_prefix"}) + "\n")

    with pytest.raises(ValueError, match="dataset_config"):
        main(
            Args(
                checkpoint=str(checkpoint_path),
                cache_path=str(cache_dir),
                output_dir=str(tmp_path / "out"),
                device="cpu",
            )
        )


def test_eval_pi05_wan_action_expert_compares_denormalized_predictions(tmp_path, capsys) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    cache_dir = tmp_path / "cache"
    output_json = tmp_path / "custom" / "metrics.json"
    _write_checkpoint(checkpoint_path, normalize_actions=True)
    _write_row(
        cache_dir / "row_000.pt",
        actions=torch.tensor([[1.0, 2.0], [9.0, 9.0]]),
        action_mask=torch.tensor([1.0, 0.0]),
    )

    output = main(
        Args(
            checkpoint=str(checkpoint_path),
            cache_path=str(cache_dir),
            output_json=str(output_json),
            sample_steps=1,
            device="cpu",
        )
    )

    capsys.readouterr()
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved == output
    assert saved["dataset_action_mse"] == pytest.approx(0.0)
    assert saved["dataset_action_smooth_l1"] == pytest.approx(0.0)
    assert saved["dataset_action_mse_per_action_dim"] == pytest.approx([0.0, 0.0])


def test_eval_pi05_wan_action_expert_compares_per_task_denormalized_predictions(tmp_path, capsys) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    cache_dir = tmp_path / "cache"
    _write_checkpoint(checkpoint_path, per_task_normalization=True)
    _write_row(
        cache_dir / "row_000.pt",
        actions=torch.tensor([[1.0, 2.0], [9.0, 9.0]]),
        action_mask=torch.tensor([1.0, 0.0]),
        task="fake",
    )

    output = main(
        Args(
            checkpoint=str(checkpoint_path),
            cache_path=str(cache_dir),
            output_dir=str(tmp_path / "out"),
            sample_steps=1,
            device="cpu",
        )
    )

    capsys.readouterr()
    assert output["dataset_action_mse"] == pytest.approx(0.0)
    assert output["dataset_action_smooth_l1"] == pytest.approx(0.0)
    assert output["dataset_action_mse_per_action_dim"] == pytest.approx([0.0, 0.0])


def test_eval_pi05_wan_action_expert_rejects_missing_per_task_normalization(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    cache_dir = tmp_path / "cache"
    _write_checkpoint(checkpoint_path, per_task_normalization=True)
    _write_row(
        cache_dir / "row_000.pt",
        actions=torch.tensor([[1.0, 2.0], [9.0, 9.0]]),
        action_mask=torch.tensor([1.0, 0.0]),
        task="unseen",
    )

    with pytest.raises(ValueError, match="missing action normalization stats"):
        main(
            Args(
                checkpoint=str(checkpoint_path),
                cache_path=str(cache_dir),
                output_dir=str(tmp_path / "out"),
                sample_steps=1,
                device="cpu",
            )
        )


def test_eval_pi05_wan_action_expert_rejects_action_mode_disagreement(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    cache_dir = tmp_path / "cache"
    _write_checkpoint(checkpoint_path, wan_action_mode="checkpoint_mode")
    _write_row(
        cache_dir / "row_000.pt",
        actions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        action_mask=torch.ones(2),
        wan_action_mode="row_mode",
    )

    with pytest.raises(ValueError, match="wan_action_mode disagrees"):
        main(
            Args(
                checkpoint=str(checkpoint_path),
                cache_path=str(cache_dir),
                output_dir=str(tmp_path / "out"),
                device="cpu",
            )
        )
