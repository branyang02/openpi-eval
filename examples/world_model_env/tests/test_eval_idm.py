from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from eval_idm import Args, _build_sample_fingerprints, compute_mean_action_baseline, main
from world_model.config import DatasetConfig, ModelConfig, TrainConfig
from world_model.models import InverseDynamicsModel
from world_model.train_lib import (
    ActionNormalizer,
    create_dataset_with_optional_cache,
    evaluate_idm,
    load_idm_checkpoint,
    run_idm_training,
    temporary_flow_num_samples,
    temporary_flow_sampling_config,
)


def _train_idm_checkpoint(output_dir: Path, *, frame_delta: int) -> Path:
    run_idm_training(
        TrainConfig(
            dataset=DatasetConfig(
                source="synthetic",
                image_keys=("corner4.image",),
                image_size=32,
                frame_delta=frame_delta,
                max_samples=16,
                synthetic_samples=16,
                num_future_frames=4,
                action_horizon=8,
            ),
            output_dir=str(output_dir),
            epochs=1,
            batch_size=4,
            device="cpu",
            seed=11,
        )
    )
    return output_dir / "idm_checkpoint.pt"


def test_eval_idm_rejects_frame_delta_mismatch(tmp_path) -> None:
    checkpoint = _train_idm_checkpoint(tmp_path / "train", frame_delta=1)

    with pytest.raises(ValueError, match="frame_delta=1"):
        main(
            Args(
                checkpoint=str(checkpoint),
                dataset_source="synthetic",
                output_dir=str(tmp_path / "eval"),
                image_keys=("corner4.image",),
                max_samples=16,
                synthetic_samples=16,
                image_size=32,
                frame_delta=2,
                num_future_frames=4,
                action_horizon=8,
                batch_size=4,
                device="cpu",
            )
        )


def test_eval_idm_accepts_matching_frame_delta(tmp_path) -> None:
    checkpoint = _train_idm_checkpoint(tmp_path / "train", frame_delta=1)
    output_dir = tmp_path / "eval"

    main(
        Args(
            checkpoint=str(checkpoint),
            dataset_source="synthetic",
            output_dir=str(output_dir),
            image_keys=("corner4.image",),
            max_samples=16,
            synthetic_samples=16,
            image_size=32,
            frame_delta=1,
            num_future_frames=4,
            action_horizon=8,
            batch_size=4,
            device="cpu",
        )
    )

    output = json.loads((output_dir / "eval_metrics.json").read_text())
    assert output["dataset_config"]["frame_delta"] == 1


def test_mean_action_baseline_scores_constant_mean_prediction() -> None:
    # Valid actions are [1, 1, 10] -> empirical mean 4.0; the second sample's
    # second step is masked out and must not contribute.
    batches = [
        {
            "action_chunk": torch.tensor([[[1.0], [1.0]], [[10.0], [0.0]]]),
            "action_mask": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
        }
    ]

    baseline = compute_mean_action_baseline(batches, torch.device("cpu"), None)

    assert baseline["mean_action"] == pytest.approx([4.0])
    # Predicting the constant 4.0: residuals (-3, -3, 6) -> squared (9, 9, 36).
    assert baseline["idm_mse"] == pytest.approx((9.0 + 9.0 + 36.0) / 3.0)
    # smooth_l1 of |residual| 3, 3, 6 (all > beta=1) is |r| - 0.5.
    assert baseline["idm_smooth_l1"] == pytest.approx((2.5 + 2.5 + 5.5) / 3.0)
    assert baseline["dataset_action_mse"] == pytest.approx(baseline["idm_mse"])
    assert baseline["dataset_action_smooth_l1"] == pytest.approx(baseline["idm_smooth_l1"])


def test_mean_action_baseline_uses_normalizer_mean_when_present() -> None:
    batches = [
        {
            "action_chunk": torch.tensor([[[1.0], [1.0]], [[10.0], [0.0]]]),
            "action_mask": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
        }
    ]
    # Normalizer mean (0.0) deliberately differs from the empirical mean (4.0),
    # so the baseline value reveals which constant was actually used.
    normalizer = ActionNormalizer(mean=torch.tensor([0.0]), std=torch.tensor([1.0]))

    baseline = compute_mean_action_baseline(batches, torch.device("cpu"), normalizer)

    assert baseline["mean_action"] == pytest.approx([0.0])
    # Predicting the constant 0.0: squared residuals (1, 1, 100).
    assert baseline["idm_mse"] == pytest.approx((1.0 + 1.0 + 100.0) / 3.0)
    assert baseline["dataset_action_mse"] == pytest.approx(baseline["idm_mse"])


def test_evaluate_idm_context_action_uses_flow_context_head_with_history_and_latents() -> None:
    class DirectContextHead(torch.nn.Module):
        def forward(self, context, *, history_tokens=None):
            if history_tokens is not None:
                context = context + history_tokens.mean(dim=1)
            first = context[:, 0] + 1.0
            second = context[:, 0] + 2.0
            return torch.stack([first, second], dim=1).unsqueeze(-1)

    class DirectContextFlowIdm(torch.nn.Module):
        uses_flow_matching = True

        def __init__(self) -> None:
            super().__init__()
            self.config = ModelConfig(
                num_views=1,
                image_size=8,
                state_dim=4,
                action_dim=1,
                action_horizon=2,
                num_future_frames=1,
                latent_dim=1,
                idm_arch="flow_transformer",
                idm_history_length=1,
                idm_flow_num_samples=3,
                idm_flow_sample_noise_scale=0.5,
            )
            self.context_action_head = DirectContextHead()
            self.forward_called = False
            self.saw_wan_vae_latents = False

        def forward(self, *args, **kwargs):
            self.forward_called = True
            raise AssertionError("context_action eval should not call the sampler forward path")

        def _transition_context(self, current_images, future_images, state, *, wan_vae_latents=None):
            del current_images
            assert wan_vae_latents is not None
            self.saw_wan_vae_latents = True
            visual = future_images.mean(dim=(1, 2, 3, 4, 5), keepdim=False).view(-1, 1)
            latent = wan_vae_latents.mean(dim=(1, 2, 3, 4), keepdim=False).view(-1, 1)
            return state[:, :1] + visual + latent

        def _history_tokens(self, prev_state_history, prev_action_history, history_mask):
            mask = history_mask.to(dtype=prev_state_history.dtype).unsqueeze(-1)
            return (prev_state_history[..., :1] + prev_action_history) * mask

    idm = DirectContextFlowIdm()
    dataset = [
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.full((1, 1, 3, 8, 8), 3.0),
            "state": torch.tensor([2.0, 0.0, 0.0, 0.0]),
            "task_id": torch.tensor(0),
            "wan_vae_latents": torch.full((8, 1, 1, 1), 4.0),
            "prev_state_history": torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
            "prev_action_history": torch.tensor([[6.0]]),
            "history_mask": torch.tensor([1.0]),
            "action_chunk": torch.tensor([[21.0], [22.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        }
    ]

    metrics = evaluate_idm(
        idm,
        DataLoader(dataset, batch_size=1),
        torch.device("cpu"),
        prediction_mode="context_action",
    )

    assert metrics["idm_mse"] == pytest.approx(0.0)
    assert metrics["idm_smooth_l1"] == pytest.approx(0.0)
    assert metrics["flow_num_samples"] == 3
    assert metrics["flow_noise_scale"] == pytest.approx(0.5)
    assert idm.saw_wan_vae_latents
    assert not idm.forward_called


def test_evaluate_idm_context_action_rejects_non_flow_idm() -> None:
    class ZeroIdm(torch.nn.Module):
        def forward(self, current_images, future_images, state, task_id, *, sample_noise=None):
            del future_images, state, task_id, sample_noise
            return torch.zeros((current_images.shape[0], 2, 1))

    dataset, _ = _zero_idm_dataset_and_config()

    with pytest.raises(ValueError, match="context_action.*flow_transformer"):
        evaluate_idm(
            ZeroIdm(),
            DataLoader(dataset, batch_size=1),
            torch.device("cpu"),
            prediction_mode="context_action",
        )


def _zero_idm_dataset_and_config() -> tuple[list[dict[str, torch.Tensor]], ModelConfig]:
    dataset = [
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.zeros(4),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[1.0], [1.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        },
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.zeros(4),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[10.0], [0.0]]),
            "action_mask": torch.tensor([1.0, 0.0]),
        },
    ]
    model_config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=4,
        action_dim=1,
        action_horizon=2,
        num_future_frames=1,
    )
    return dataset, model_config


class _TinyEvalDataset:
    def __init__(self) -> None:
        self.config = DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=8,
            frame_delta=4,
            num_future_frames=1,
            action_horizon=2,
            synthetic_samples=2,
        )
        self.items = [
            {
                "current_images": torch.zeros(1, 3, 8, 8),
                "future_images": torch.full((1, 1, 3, 8, 8), 0.25),
                "state": torch.zeros(4),
                "task_id": torch.tensor(0),
                "action_chunk": torch.tensor([[1.0], [1.0]]),
                "action_mask": torch.tensor([1.0, 1.0]),
            },
            {
                "current_images": torch.ones(1, 3, 8, 8),
                "future_images": torch.full((1, 1, 3, 8, 8), 0.75),
                "state": torch.ones(4),
                "task_id": torch.tensor(1),
                "action_chunk": torch.tensor([[10.0], [0.0]]),
                "action_mask": torch.tensor([1.0, 0.0]),
            },
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def _wan_vae_model_config(*, idm_visual_encoder: str = "wan_vae") -> ModelConfig:
    return ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=4,
        action_dim=1,
        action_horizon=2,
        num_future_frames=1,
        latent_dim=32,
        idm_arch="flow_transformer",
        idm_visual_encoder=idm_visual_encoder,
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        wan_vae_latent_channels=8,
        wan_vae_spatial_stride=8,
        wan_vae_use_cached_latents=True,
    )


def test_temporary_flow_num_samples_updates_and_restores_module_and_flow_head_configs() -> None:
    idm = InverseDynamicsModel(
        ModelConfig(
            num_views=1,
            image_size=8,
            state_dim=4,
            action_dim=1,
            action_horizon=2,
            num_future_frames=1,
            latent_dim=32,
            idm_arch="flow_transformer",
            idm_transformer_layers=1,
            idm_transformer_heads=4,
            idm_transformer_patch_size=4,
            idm_flow_num_samples=1,
        )
    )
    original_config = idm.config
    original_flow_head_config = idm.flow_head.config

    with pytest.raises(RuntimeError, match="restore check"):
        with temporary_flow_num_samples(idm, 5) as effective:
            assert effective == 5
            assert idm.config.idm_flow_num_samples == 5
            assert idm.flow_head.config.idm_flow_num_samples == 5
            assert idm.config is not original_config
            assert idm.flow_head.config is not original_flow_head_config
            raise RuntimeError("restore check")

    assert idm.config is original_config
    assert idm.flow_head.config is original_flow_head_config
    assert idm.config.idm_flow_num_samples == 1
    assert idm.flow_head.config.idm_flow_num_samples == 1


def test_temporary_flow_sampling_config_updates_and_restores_noise_scale() -> None:
    idm = InverseDynamicsModel(
        ModelConfig(
            num_views=1,
            image_size=8,
            state_dim=4,
            action_dim=1,
            action_horizon=2,
            num_future_frames=1,
            latent_dim=32,
            idm_arch="flow_transformer",
            idm_transformer_layers=1,
            idm_transformer_heads=4,
            idm_transformer_patch_size=4,
            idm_flow_num_samples=1,
            idm_flow_sample_noise_scale=1.0,
        )
    )
    original_config = idm.config
    original_flow_head_config = idm.flow_head.config

    with temporary_flow_sampling_config(idm, num_samples=5, noise_scale=0.25) as (
        effective_num_samples,
        effective_noise_scale,
    ):
        assert effective_num_samples == 5
        assert effective_noise_scale == pytest.approx(0.25)
        assert idm.config.idm_flow_num_samples == 5
        assert idm.flow_head.config.idm_flow_num_samples == 5
        assert idm.config.idm_flow_sample_noise_scale == pytest.approx(0.25)
        assert idm.flow_head.config.idm_flow_sample_noise_scale == pytest.approx(0.25)

    assert idm.config is original_config
    assert idm.flow_head.config is original_flow_head_config
    assert idm.config.idm_flow_num_samples == 1
    assert idm.flow_head.config.idm_flow_num_samples == 1
    assert idm.config.idm_flow_sample_noise_scale == pytest.approx(1.0)
    assert idm.flow_head.config.idm_flow_sample_noise_scale == pytest.approx(1.0)


def test_temporary_flow_num_samples_rejects_non_flow_idm() -> None:
    idm = InverseDynamicsModel(
        ModelConfig(
            num_views=1,
            image_size=8,
            state_dim=4,
            action_dim=1,
            action_horizon=2,
            num_future_frames=1,
        )
    )

    with pytest.raises(ValueError, match="flow-matching IDM"):
        with temporary_flow_num_samples(idm, 2):
            pass


def _write_generated_wan_config(cache_dir: Path, generator: dict[str, object] | None = None) -> dict[str, object]:
    if generator is None:
        generator = {
            "source": "wan_lora",
            "checkpoint": "Wan2.2-TI2V-5B",
            "seed": 123,
        }
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "config.json").write_text(json.dumps({"generator": generator}) + "\n")
    return generator


def test_eval_idm_reports_mean_action_baseline(tmp_path, monkeypatch) -> None:
    class ZeroIdm(torch.nn.Module):
        def forward(self, current_images, future_images, state, task_id, *, sample_noise=None):
            return torch.zeros((current_images.shape[0], 2, 1))

    dataset, model_config = _zero_idm_dataset_and_config()
    monkeypatch.setattr("eval_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "eval_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (ZeroIdm().to(device), model_config),
    )
    monkeypatch.setattr("eval_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    main(
        Args(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
        )
    )

    metrics = json.loads((tmp_path / "eval_metrics.json").read_text())

    # The zero IDM (MSE 34.0) is *worse* than predicting the mean action (18.0);
    # surfacing the baseline makes that obvious instead of silently passing.
    assert metrics["idm_mse"] == pytest.approx((1.0 + 1.0 + 100.0) / 3.0)
    assert metrics["dataset_action_mse"] == pytest.approx(metrics["idm_mse"])
    assert metrics["dataset_action_smooth_l1"] == pytest.approx(metrics["idm_smooth_l1"])
    assert metrics["metric_family"] == "dataset_action_mse"
    assert metrics["mean_action_baseline"]["idm_mse"] == pytest.approx((9.0 + 9.0 + 36.0) / 3.0)
    assert metrics["mean_action_baseline"]["idm_smooth_l1"] == pytest.approx((2.5 + 2.5 + 5.5) / 3.0)
    assert metrics["mean_action_baseline"]["dataset_action_mse"] == pytest.approx(
        metrics["mean_action_baseline"]["idm_mse"]
    )
    assert metrics["mean_action_baseline"]["dataset_action_smooth_l1"] == pytest.approx(
        metrics["mean_action_baseline"]["idm_smooth_l1"]
    )
    assert metrics["mean_action_baseline"]["mean_action"] == pytest.approx([4.0])
    expected_fingerprints = _build_sample_fingerprints(
        DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=8,
            frame_delta=4,
            action_horizon=2,
            seed=7,
        ),
        num_samples=len(dataset),
    )
    assert metrics["dataset_fingerprint"] == expected_fingerprints["dataset_fingerprint"]
    assert metrics["sample_fingerprint"] == expected_fingerprints["sample_fingerprint"]
    assert metrics["num_samples"] == len(dataset)
    assert "num_future_frames" not in metrics["dataset_fingerprint"]["dataset_config"]


def test_eval_idm_future_usage_flag_records_metrics(tmp_path, monkeypatch) -> None:
    class ZeroFlowIdm(torch.nn.Module):
        uses_flow_matching = True

        def __init__(self) -> None:
            super().__init__()
            self.config = ModelConfig(
                num_views=1,
                image_size=8,
                state_dim=4,
                action_dim=1,
                action_horizon=2,
                num_future_frames=1,
                idm_arch="flow_transformer",
            )

        def forward(self, current_images, future_images, state, task_id, *, sample_noise=None):
            del future_images, state, task_id, sample_noise
            return torch.zeros((current_images.shape[0], 2, 1))

    dataset, _ = _zero_idm_dataset_and_config()
    model_config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=4,
        action_dim=1,
        action_horizon=2,
        num_future_frames=1,
        idm_arch="flow_transformer",
    )
    monkeypatch.setattr("eval_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "eval_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (ZeroFlowIdm().to(device), model_config),
    )
    monkeypatch.setattr("eval_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)
    monkeypatch.setattr(
        "eval_idm.evaluate_idm_future_usage",
        lambda *args, **kwargs: {
            "future_usage_gate_pass": False,
            "future_usage_rank_accuracy": 0.0,
            "future_usage_gate_reasons": "rank_accuracy",
        },
    )

    main(
        Args(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
            future_usage_eval=True,
        )
    )

    metrics = json.loads((tmp_path / "eval_metrics.json").read_text())

    assert metrics["future_usage_gate_pass"] is False
    assert metrics["future_usage_rank_accuracy"] == pytest.approx(0.0)
    assert metrics["future_usage_gate_reasons"] == "rank_accuracy"


def test_eval_idm_forwards_future_usage_score_mode(tmp_path, monkeypatch) -> None:
    class ZeroFlowIdm(torch.nn.Module):
        uses_flow_matching = True

        def __init__(self) -> None:
            super().__init__()
            self.config = ModelConfig(
                num_views=1,
                image_size=8,
                state_dim=4,
                action_dim=1,
                action_horizon=2,
                num_future_frames=1,
                idm_arch="flow_transformer",
            )

        def forward(self, current_images, future_images, state, task_id, *, sample_noise=None):
            del future_images, state, task_id, sample_noise
            return torch.zeros((current_images.shape[0], 2, 1))

    dataset, _ = _zero_idm_dataset_and_config()
    model_config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=4,
        action_dim=1,
        action_horizon=2,
        num_future_frames=1,
        idm_arch="flow_transformer",
    )
    captured = {}

    def capture_future_usage(*args, **kwargs):
        del args
        captured["score_mode"] = kwargs.get("score_mode")
        return {"future_usage_gate_pass": True, "future_usage_score_mode": kwargs.get("score_mode")}

    monkeypatch.setattr("eval_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "eval_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (ZeroFlowIdm().to(device), model_config),
    )
    monkeypatch.setattr("eval_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)
    monkeypatch.setattr("eval_idm.evaluate_idm_future_usage", capture_future_usage)

    main(
        Args(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
            future_usage_eval=True,
            future_usage_score_mode="sampled_action",
        )
    )

    assert captured["score_mode"] == "sampled_action"
    metrics = json.loads((tmp_path / "eval_metrics.json").read_text())
    assert metrics["future_usage_score_mode"] == "sampled_action"


@pytest.mark.parametrize(
    "cache_args",
    [
        {"cached_future_dir": "future_cache", "wan_vae_latent_cache_dir": "wan_cache"},
        {"cached_future_dir": "future_cache", "generated_wan_latent_cache_dir": "generated_cache"},
        {"wan_vae_latent_cache_dir": "wan_cache", "generated_wan_latent_cache_dir": "generated_cache"},
    ],
)
def test_eval_idm_rejects_multiple_future_latent_cache_modes(cache_args) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        main(Args(checkpoint="fake.pt", **cache_args))


def test_eval_idm_generated_wan_latent_cache_requires_generator_config(tmp_path) -> None:
    missing_config_dir = tmp_path / "missing"
    missing_config_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Generated Wan latent cache config not found"):
        main(Args(checkpoint="fake.pt", generated_wan_latent_cache_dir=str(missing_config_dir)))

    invalid_json_dir = tmp_path / "invalid"
    invalid_json_dir.mkdir()
    (invalid_json_dir / "config.json").write_text("{")
    with pytest.raises(ValueError, match="invalid JSON"):
        main(Args(checkpoint="fake.pt", generated_wan_latent_cache_dir=str(invalid_json_dir)))

    missing_generator_dir = tmp_path / "missing_generator"
    missing_generator_dir.mkdir()
    (missing_generator_dir / "config.json").write_text(json.dumps({"cache_schema": "generated_wan_latents"}))
    with pytest.raises(ValueError, match="generator JSON object"):
        main(Args(checkpoint="fake.pt", generated_wan_latent_cache_dir=str(missing_generator_dir)))


def test_eval_idm_generated_wan_latent_cache_wires_checkpoint_wrapper_and_metrics(tmp_path, monkeypatch) -> None:
    class FakeIdm(torch.nn.Module):
        pass

    class FakeGeneratedWanLatentDataset:
        instances = []

        def __init__(self, base_dataset, cache_dir, model_config, *, generator_metadata):
            self.base_dataset = base_dataset
            self.cache_dir = cache_dir
            self.model_config = model_config
            self.generator_metadata = generator_metadata
            self.instances.append(self)

        def __len__(self) -> int:
            return len(self.base_dataset)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            item = dict(self.base_dataset[index])
            item["wan_vae_latents"] = torch.full((8, 1, 1, 1), float(index))
            return item

    cache_dir = tmp_path / "generated_cache"
    generator = _write_generated_wan_config(
        cache_dir,
        {
            "source": "wan_lora",
            "checkpoint": "Wan2.2-TI2V-5B",
            "seed": 123,
            "stop_after_steps": 4,
        },
    )
    base_dataset = _TinyEvalDataset()
    model_config = _wan_vae_model_config()
    load_calls = []
    create_cache_args = []
    evaluated_batches = []

    def fake_load_idm_checkpoint(path, device, *, use_cached_wan_vae_latents=False):
        load_calls.append(use_cached_wan_vae_latents)
        return FakeIdm().to(device), model_config

    def fake_evaluate_idm(
        idm,
        loader,
        device,
        *,
        flow_eval_seed=None,
        flow_num_samples=None,
        flow_noise_scale=None,
        prediction_mode="sample",
    ):
        del flow_eval_seed, flow_num_samples, flow_noise_scale, prediction_mode
        batch = next(iter(loader))
        evaluated_batches.append(batch)
        assert torch.allclose(batch["future_images"][0], base_dataset[0]["future_images"])
        assert "wan_vae_latents" in batch
        return {"idm_mse": 0.0, "idm_smooth_l1": 0.0}

    monkeypatch.setattr("eval_idm.load_idm_checkpoint", fake_load_idm_checkpoint)
    monkeypatch.setattr("eval_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)
    monkeypatch.setattr(
        "eval_idm.create_dataset_with_optional_cache",
        lambda config, cache: create_cache_args.append(cache) or base_dataset,
    )
    monkeypatch.setattr("eval_idm.GeneratedWanLatentDataset", FakeGeneratedWanLatentDataset)
    monkeypatch.setattr("eval_idm.evaluate_idm", fake_evaluate_idm)

    output_dir = tmp_path / "eval"
    main(
        Args(
            checkpoint="fake.pt",
            output_dir=str(output_dir),
            generated_wan_latent_cache_dir=str(cache_dir),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
        )
    )

    metrics = json.loads((output_dir / "eval_metrics.json").read_text())
    wrapper = FakeGeneratedWanLatentDataset.instances[0]

    assert load_calls == [True]
    assert create_cache_args == [None]
    assert wrapper.base_dataset is base_dataset
    assert wrapper.cache_dir == str(cache_dir)
    assert wrapper.model_config is model_config
    assert wrapper.generator_metadata == generator
    assert evaluated_batches
    assert metrics["generated_wan_latent_cache_dir"] == str(cache_dir)
    assert metrics["generated_wan_latent_generator"] == generator


def test_eval_idm_default_cache_behavior_remains_unchanged(tmp_path, monkeypatch) -> None:
    class FakeIdm(torch.nn.Module):
        pass

    base_dataset = _TinyEvalDataset()
    model_config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=4,
        action_dim=1,
        action_horizon=2,
        num_future_frames=1,
    )
    load_calls = []
    create_cache_args = []

    def fake_load_idm_checkpoint(path, device, *, use_cached_wan_vae_latents=False):
        load_calls.append(use_cached_wan_vae_latents)
        return FakeIdm().to(device), model_config

    monkeypatch.setattr("eval_idm.load_idm_checkpoint", fake_load_idm_checkpoint)
    monkeypatch.setattr("eval_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)
    monkeypatch.setattr(
        "eval_idm.create_dataset_with_optional_cache",
        lambda config, cache: create_cache_args.append(cache) or base_dataset,
    )
    monkeypatch.setattr("eval_idm.GeneratedWanLatentDataset", lambda *args, **kwargs: pytest.fail("unexpected wrapper"))
    monkeypatch.setattr("eval_idm.CachedWanVaeLatentDataset", lambda *args, **kwargs: pytest.fail("unexpected wrapper"))
    monkeypatch.setattr(
        "eval_idm.evaluate_idm",
        lambda idm,
        loader,
        device,
        *,
        flow_eval_seed=None,
        flow_num_samples=None,
        flow_noise_scale=None,
        prediction_mode="sample": {
            "idm_mse": 0.0,
            "idm_smooth_l1": 0.0,
        },
    )

    output_dir = tmp_path / "eval"
    main(
        Args(
            checkpoint="fake.pt",
            output_dir=str(output_dir),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
        )
    )

    metrics = json.loads((output_dir / "eval_metrics.json").read_text())

    assert load_calls == [False]
    assert create_cache_args == [None]
    assert metrics["cached_future_dir"] is None
    assert metrics["wan_vae_latent_cache_dir"] is None
    assert metrics["generated_wan_latent_cache_dir"] is None
    assert metrics["generated_wan_latent_generator"] is None
    assert metrics["prediction_mode"] == "sample"


def test_eval_idm_main_forwards_and_records_prediction_mode(tmp_path, monkeypatch) -> None:
    class FakeFlowIdm(torch.nn.Module):
        uses_flow_matching = True

        def __init__(self) -> None:
            super().__init__()
            self.config = ModelConfig(
                num_views=1,
                image_size=8,
                state_dim=4,
                action_dim=1,
                action_horizon=2,
                num_future_frames=1,
                idm_arch="flow_transformer",
                idm_flow_num_samples=1,
            )

    idm = FakeFlowIdm()
    forwarded_prediction_modes = []

    def fake_evaluate_idm(
        idm,
        loader,
        device,
        *,
        flow_eval_seed=None,
        flow_num_samples=None,
        flow_noise_scale=None,
        prediction_mode="sample",
    ):
        del idm, loader, device, flow_eval_seed, flow_num_samples, flow_noise_scale
        forwarded_prediction_modes.append(prediction_mode)
        return {"idm_mse": 0.0, "idm_smooth_l1": 0.0}

    monkeypatch.setattr(
        "eval_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (idm.to(device), idm.config),
    )
    monkeypatch.setattr("eval_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)
    monkeypatch.setattr("eval_idm.create_dataset_with_optional_cache", lambda config, cache: _TinyEvalDataset())
    monkeypatch.setattr("eval_idm.evaluate_idm", fake_evaluate_idm)

    main(
        Args(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
            prediction_mode="context_action",
        )
    )

    metrics = json.loads((tmp_path / "eval_metrics.json").read_text())

    assert forwarded_prediction_modes == ["context_action"]
    assert metrics["prediction_mode"] == "context_action"


def test_eval_idm_uses_checkpoint_history_length_for_dataset(tmp_path, monkeypatch) -> None:
    class FakeIdm(torch.nn.Module):
        pass

    captured_configs: list[DatasetConfig] = []
    dataset, _ = _zero_idm_dataset_and_config()
    model_config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=4,
        action_dim=1,
        action_horizon=2,
        num_future_frames=1,
        idm_arch="flow_transformer",
        idm_history_length=2,
    )

    monkeypatch.setattr(
        "eval_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (FakeIdm().to(device), model_config),
    )
    monkeypatch.setattr("eval_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)
    monkeypatch.setattr(
        "eval_idm.create_dataset_with_optional_cache",
        lambda config, cache: captured_configs.append(config) or dataset,
    )
    monkeypatch.setattr(
        "eval_idm.evaluate_idm",
        lambda idm,
        loader,
        device,
        *,
        flow_eval_seed=None,
        flow_num_samples=None,
        flow_noise_scale=None,
        prediction_mode="sample": {
            "idm_mse": 0.0,
            "idm_smooth_l1": 0.0,
        },
    )

    main(Args(checkpoint="fake.pt", output_dir=str(tmp_path), image_size=8, action_horizon=2, device="cpu"))

    metrics = json.loads((tmp_path / "eval_metrics.json").read_text())

    assert captured_configs[0].idm_history_length == 2
    assert metrics["dataset_config"]["idm_history_length"] == 2


def test_eval_idm_flow_num_samples_override_is_forwarded_and_recorded(tmp_path, monkeypatch) -> None:
    class FakeFlowIdm(torch.nn.Module):
        uses_flow_matching = True

        def __init__(self) -> None:
            super().__init__()
            self.config = ModelConfig(
                num_views=1,
                image_size=8,
                state_dim=4,
                action_dim=1,
                action_horizon=2,
                num_future_frames=1,
                idm_arch="flow_transformer",
                idm_flow_num_samples=1,
            )
            self.flow_head = torch.nn.Module()
            self.flow_head.config = self.config

    idm = FakeFlowIdm()
    base_dataset = _TinyEvalDataset()
    forwarded_flow_num_samples = []

    def fake_evaluate_idm(
        idm,
        loader,
        device,
        *,
        flow_eval_seed=None,
        flow_num_samples=None,
        flow_noise_scale=None,
        prediction_mode="sample",
    ):
        del idm, loader, device, flow_eval_seed, flow_noise_scale, prediction_mode
        forwarded_flow_num_samples.append(flow_num_samples)
        return {"idm_mse": 0.0, "idm_smooth_l1": 0.0}

    monkeypatch.setattr(
        "eval_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (idm.to(device), idm.config),
    )
    monkeypatch.setattr("eval_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)
    monkeypatch.setattr("eval_idm.create_dataset_with_optional_cache", lambda config, cache: base_dataset)
    monkeypatch.setattr("eval_idm.evaluate_idm", fake_evaluate_idm)

    main(
        Args(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
            flow_num_samples=16,
        )
    )

    metrics = json.loads((tmp_path / "eval_metrics.json").read_text())

    assert forwarded_flow_num_samples == [16]
    assert metrics["flow_num_samples"] == 16


def test_eval_idm_flow_noise_scale_override_is_recorded_and_changes_sampling(tmp_path, monkeypatch) -> None:
    class NoiseEchoFlowIdm(torch.nn.Module):
        uses_flow_matching = True

        def __init__(self) -> None:
            super().__init__()
            self.config = ModelConfig(
                num_views=1,
                image_size=8,
                state_dim=4,
                action_dim=1,
                action_horizon=2,
                num_future_frames=1,
                idm_arch="flow_transformer",
                idm_flow_num_samples=1,
                idm_flow_sample_noise_scale=1.0,
            )
            self.flow_head = torch.nn.Module()
            self.flow_head.config = self.config
            self.sample_noise_max_abs: list[float] = []

        def forward(self, current_images, future_images, state, task_id, *, sample_noise=None):
            del future_images, state, task_id
            assert sample_noise is not None
            self.sample_noise_max_abs.append(float(sample_noise.abs().max().detach().cpu()))
            return sample_noise.view(
                current_images.shape[0],
                self.config.idm_flow_num_samples,
                self.config.action_horizon,
                self.config.action_dim,
            ).mean(dim=1)

    idm = NoiseEchoFlowIdm()
    base_dataset = _TinyEvalDataset()
    monkeypatch.setattr(
        "eval_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (idm.to(device), idm.config),
    )
    monkeypatch.setattr("eval_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)
    monkeypatch.setattr("eval_idm.create_dataset_with_optional_cache", lambda config, cache: base_dataset)

    main(
        Args(
            checkpoint="fake.pt",
            output_dir=str(tmp_path / "zero"),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
            flow_noise_scale=0.0,
        )
    )
    zero_metrics = json.loads((tmp_path / "zero" / "eval_metrics.json").read_text())
    zero_noise_max_abs = list(idm.sample_noise_max_abs)
    idm.sample_noise_max_abs.clear()

    main(
        Args(
            checkpoint="fake.pt",
            output_dir=str(tmp_path / "scaled"),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
            flow_noise_scale=0.5,
        )
    )
    scaled_metrics = json.loads((tmp_path / "scaled" / "eval_metrics.json").read_text())

    assert zero_metrics["flow_noise_scale"] == pytest.approx(0.0)
    assert scaled_metrics["flow_noise_scale"] == pytest.approx(0.5)
    assert zero_metrics["idm_mse"] != scaled_metrics["idm_mse"]
    assert zero_noise_max_abs and all(value == pytest.approx(0.0) for value in zero_noise_max_abs)
    assert idm.sample_noise_max_abs and max(idm.sample_noise_max_abs) > 0.0
    assert idm.config.idm_flow_sample_noise_scale == pytest.approx(1.0)
    assert idm.flow_head.config.idm_flow_sample_noise_scale == pytest.approx(1.0)


def test_eval_idm_rejects_generated_wan_latent_cache_for_non_wan_vae_checkpoint(tmp_path, monkeypatch) -> None:
    class FakeIdm(torch.nn.Module):
        pass

    cache_dir = tmp_path / "generated_cache"
    _write_generated_wan_config(cache_dir)
    load_calls = []

    def fake_load_idm_checkpoint(path, device, *, use_cached_wan_vae_latents=False):
        load_calls.append(use_cached_wan_vae_latents)
        return FakeIdm().to(device), _wan_vae_model_config(idm_visual_encoder="patch")

    monkeypatch.setattr("eval_idm.load_idm_checkpoint", fake_load_idm_checkpoint)
    monkeypatch.setattr("eval_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)
    monkeypatch.setattr(
        "eval_idm.create_dataset_with_optional_cache",
        lambda config, cache: pytest.fail("generated cache should be rejected before dataset construction"),
    )

    with pytest.raises(ValueError, match="idm_visual_encoder='wan_vae'"):
        main(Args(checkpoint="fake.pt", generated_wan_latent_cache_dir=str(cache_dir), device="cpu"))

    assert load_calls == [True]


# --- Flow-matching eval determinism / flow_eval_seed ---
#
# A flow_transformer IDM samples actions from noise, so eval is only reproducible when the noise is
# seeded. flow_eval_seed controls that noise; these tests pin that it makes eval reproducible AND
# that it genuinely drives the result, that None falls back to the global RNG, and that the seed is
# inert (and recorded as None) for a regression IDM that never samples.

_FLOW_DATASET = DatasetConfig(
    source="synthetic",
    image_keys=("corner4.image",),
    image_size=32,
    frame_delta=1,
    max_samples=12,
    synthetic_samples=12,
    num_future_frames=2,
    action_horizon=4,
)


def _train_flow_idm_checkpoint(output_dir: Path) -> Path:
    run_idm_training(
        TrainConfig(
            dataset=_FLOW_DATASET,
            model=ModelConfig(
                idm_arch="flow_transformer",
                latent_dim=64,
                idm_transformer_layers=1,
                idm_transformer_heads=4,
                idm_transformer_patch_size=16,
                idm_transformer_dropout=0.0,
                idm_flow_sampling_steps=2,
                idm_flow_num_samples=2,
            ),
            output_dir=str(output_dir),
            epochs=1,
            batch_size=4,
            device="cpu",
            seed=13,
        )
    )
    return output_dir / "idm_checkpoint.pt"


@pytest.fixture(scope="module")
def flow_idm_checkpoint(tmp_path_factory) -> Path:
    """Train one tiny flow-matching IDM and reuse it across the determinism tests."""
    return _train_flow_idm_checkpoint(tmp_path_factory.mktemp("flow_idm"))


def _flow_loader() -> DataLoader:
    return DataLoader(create_dataset_with_optional_cache(_FLOW_DATASET), batch_size=4, shuffle=False)


def test_eval_idm_flow_eval_seed_is_reproducible_and_drives_noise(flow_idm_checkpoint) -> None:
    idm, _ = load_idm_checkpoint(flow_idm_checkpoint, torch.device("cpu"))
    loader = _flow_loader()

    first = evaluate_idm(idm, loader, torch.device("cpu"), flow_eval_seed=123)
    repeat = evaluate_idm(idm, loader, torch.device("cpu"), flow_eval_seed=123)
    other = evaluate_idm(idm, loader, torch.device("cpu"), flow_eval_seed=999)

    # A fixed flow_eval_seed makes flow-matching eval bit-for-bit reproducible...
    assert first == repeat
    # ...and the seed genuinely drives the sampling noise: a different seed moves the metric, so the
    # reproducibility above is the seed's doing, not an accidentally noise-invariant model.
    assert first["idm_mse"] != other["idm_mse"]


def test_eval_idm_flow_eval_seed_none_uses_global_rng(flow_idm_checkpoint) -> None:
    idm, _ = load_idm_checkpoint(flow_idm_checkpoint, torch.device("cpu"))
    loader = _flow_loader()

    torch.manual_seed(0)
    seeded = evaluate_idm(idm, loader, torch.device("cpu"), flow_eval_seed=None)
    torch.manual_seed(0)
    seeded_again = evaluate_idm(idm, loader, torch.device("cpu"), flow_eval_seed=None)
    advanced = evaluate_idm(idm, loader, torch.device("cpu"), flow_eval_seed=None)

    # flow_eval_seed=None deliberately uses the *global* RNG instead of a dedicated generator, so
    # eval is reproducible only while the global seed is pinned and drifts once it advances. This is
    # exactly why flow_eval_seed defaults to 0 for stable reporting.
    assert seeded == seeded_again
    assert advanced["idm_mse"] != seeded["idm_mse"]


def test_eval_idm_main_records_flow_eval_seed_for_flow_checkpoint(flow_idm_checkpoint, tmp_path) -> None:
    def run(out_dir: Path) -> dict:
        main(
            Args(
                checkpoint=str(flow_idm_checkpoint),
                dataset_source="synthetic",
                output_dir=str(out_dir),
                image_keys=("corner4.image",),
                max_samples=12,
                synthetic_samples=12,
                image_size=32,
                frame_delta=1,
                num_future_frames=2,
                action_horizon=4,
                batch_size=4,
                device="cpu",
                flow_eval_seed=123,
            )
        )
        return json.loads((out_dir / "eval_metrics.json").read_text())

    first = run(tmp_path / "a")
    second = run(tmp_path / "b")

    # The seed is live for a flow IDM, so it is recorded and makes the whole CLI path reproducible:
    # the seeded generator is independent of global RNG, so a re-run reproduces idm_mse exactly.
    assert first["flow_eval_seed"] == 123
    assert second["flow_eval_seed"] == 123
    assert first["idm_mse"] == second["idm_mse"]


def test_eval_idm_non_flow_checkpoint_ignores_flow_eval_seed(tmp_path) -> None:
    checkpoint = _train_idm_checkpoint(tmp_path / "train", frame_delta=1)

    def run(seed: int, out_dir: Path) -> dict:
        main(
            Args(
                checkpoint=str(checkpoint),
                dataset_source="synthetic",
                output_dir=str(out_dir),
                image_keys=("corner4.image",),
                max_samples=16,
                synthetic_samples=16,
                image_size=32,
                frame_delta=1,
                num_future_frames=4,
                action_horizon=8,
                batch_size=4,
                device="cpu",
                flow_eval_seed=seed,
            )
        )
        return json.loads((out_dir / "eval_metrics.json").read_text())

    seven = run(7, tmp_path / "eval7")
    eight = run(8, tmp_path / "eval8")

    # A regression IDM never samples noise, so the flow seed cannot change its metrics, and main
    # records flow_eval_seed as None to make clear the seed was inert for this checkpoint.
    assert seven["idm_mse"] == eight["idm_mse"]
    assert seven["flow_eval_seed"] is None
    assert eight["flow_eval_seed"] is None
