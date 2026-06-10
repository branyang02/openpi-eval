from __future__ import annotations

import json

import pytest
import torch

from diagnose_idm import Args as DiagnoseArgs
from diagnose_idm import future_sensitivity_gate, stepwise_action_diagnostics
from diagnose_idm import main as diagnose_main
from world_model.config import DatasetConfig, ModelConfig, TrainConfig
from world_model.train_lib import (
    ActionNormalizer,
    StateNormalizer,
    attach_action_normalizer,
    attach_state_normalizer,
    run_idm_training,
)


def _future_sensitivity(*, output_delta_mse: float, target_mse: float) -> dict[str, dict[str, float]]:
    variant = {"target_mse": target_mse, "target_smooth_l1": target_mse, "output_delta_mse": output_delta_mse}
    return {"current_repeated": variant, "zero": variant, "shuffled": variant, "noise": variant}


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


class _ZeroIdm(torch.nn.Module):
    def forward(self, current_images, future_images, state, task_id, *, sample_noise=None):
        return torch.zeros((current_images.shape[0], 2, 1))


class _HistoryRequiredIdm(torch.nn.Module):
    uses_flow_matching = False

    def __init__(self):
        super().__init__()
        self.seen_history: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        current_images,
        future_images,
        state,
        task_id,
        *,
        sample_noise=None,
        prev_state_history=None,
        prev_action_history=None,
        history_mask=None,
    ):
        del future_images, state, task_id, sample_noise
        if prev_state_history is None or prev_action_history is None or history_mask is None:
            raise AssertionError("history kwargs were not forwarded")
        self.seen_history.append(
            (
                prev_state_history.detach().cpu(),
                prev_action_history.detach().cpu(),
                history_mask.detach().cpu(),
            )
        )
        return torch.zeros((current_images.shape[0], 2, 1), device=current_images.device)


class _LatentAwareZeroIdm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_latents: list[torch.Tensor | None] = []

    def forward(self, current_images, future_images, state, task_id, *, sample_noise=None, wan_vae_latents=None):
        del future_images, state, task_id, sample_noise
        self.seen_latents.append(None if wan_vae_latents is None else wan_vae_latents.detach().cpu())
        return torch.zeros((current_images.shape[0], 2, 1))


class _StateTargetTransitionEncoder(torch.nn.Module):
    def forward(self, current_images, future_images, state):
        del future_images
        return state.view(current_images.shape[0], 2, 1)


class _FutureMeanTransitionEncoder(torch.nn.Module):
    def forward(self, current_images, future_images, state):
        del state
        value = future_images.mean(dim=(1, 2, 3, 4, 5))
        return value.view(current_images.shape[0], 1, 1).expand(-1, 2, -1)


class _FutureLatentTransitionEncoder(torch.nn.Module):
    def forward(self, current_images, future_images, state, *, wan_vae_latents=None):
        del future_images, state
        if wan_vae_latents is None:
            raise AssertionError("wan_vae_latents were not forwarded to the ranking probe")
        value = wan_vae_latents[:, :, 1:].mean(dim=(1, 2, 3, 4))
        return value.view(current_images.shape[0], 1, 1).expand(-1, 2, -1)


class _FutureLatentTokenTransitionEncoder(torch.nn.Module):
    def forward(self, current_images, future_images, state, *, wan_vae_latents=None, return_tokens=False):
        del future_images, state
        if wan_vae_latents is None:
            raise AssertionError("wan_vae_latents were not forwarded to the ranking probe")
        value = wan_vae_latents[:, :, 1:].mean(dim=(1, 2, 3, 4))
        context = value.view(current_images.shape[0], 1, 1).expand(-1, 2, -1)
        if return_tokens:
            return context, value.view(current_images.shape[0], 1, 1)
        return context


class _PerfectVelocityFlowHead(torch.nn.Module):
    def forward(self, context, noisy_action, time):
        time_view = time.view(-1, 1, 1)
        return (context - noisy_action) / (1.0 - time_view).clamp_min(1e-6)


class _VisualTokenRequiredPerfectVelocityFlowHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_token_values: list[torch.Tensor] = []

    def forward(self, context, noisy_action, time, *, history_tokens=None, visual_context_tokens=None):
        del history_tokens
        if visual_context_tokens is None:
            raise AssertionError("visual_context_tokens were not forwarded to the diagnostic flow probe")
        self.visual_token_values.append(visual_context_tokens[..., 0].detach().cpu())
        time_view = time.view(-1, 1, 1)
        return (context - noisy_action) / (1.0 - time_view).clamp_min(1e-6)


class _FutureAwareTeacherForcedFlowIdm(torch.nn.Module):
    uses_flow_matching = True

    def __init__(
        self,
        transition_encoder,
        *,
        idm_visual_encoder="patch",
        idm_flow_visual_token_conditioning=False,
        idm_flow_visual_token_conditioning_mode="prefix",
    ):
        super().__init__()
        self.config = ModelConfig(
            num_views=1,
            image_size=8,
            state_dim=2,
            action_dim=1,
            action_horizon=2,
            num_future_frames=1,
            idm_arch="flow_transformer",
            idm_visual_encoder=idm_visual_encoder,
            idm_flow_num_samples=1,
            wan_vae_latent_channels=1,
            wan_vae_use_cached_latents=idm_visual_encoder == "wan_vae",
            idm_flow_visual_token_conditioning=idm_flow_visual_token_conditioning,
            idm_flow_visual_token_conditioning_mode=idm_flow_visual_token_conditioning_mode,
        )
        self.transition_encoder = transition_encoder
        self.flow_head = _PerfectVelocityFlowHead()
        self.flow_head.config = self.config

    def forward(
        self,
        current_images,
        future_images,
        state,
        task_id,
        *,
        sample_noise=None,
        target_action=None,
        action_mask=None,
        wan_vae_latents=None,
        mode="sample",
    ):
        del future_images, state, task_id, sample_noise, target_action, action_mask, wan_vae_latents
        if mode != "sample":
            raise ValueError("fake flow IDM only supports sample mode")
        return torch.zeros((current_images.shape[0], 2, 1), device=current_images.device)

    def sample_action(
        self,
        current_images,
        future_images,
        state,
        *,
        sample_noise=None,
        wan_vae_latents=None,
        prev_state_history=None,
        prev_action_history=None,
        history_mask=None,
    ):
        del sample_noise, prev_state_history, prev_action_history, history_mask
        # Deterministic stand-in for the flow sampler that reproduces the teacher-forced endpoint
        # (transition-encoder context), so sampled-action ranking mirrors the endpoint ranking.
        if self.config.idm_visual_encoder == "wan_vae":
            return self.transition_encoder(current_images, future_images, state, wan_vae_latents=wan_vae_latents)
        return self.transition_encoder(current_images, future_images, state)


class _TeacherForcedFlowIdm(torch.nn.Module):
    uses_flow_matching = True

    def __init__(self):
        super().__init__()
        self.config = ModelConfig(
            num_views=1,
            image_size=8,
            state_dim=2,
            action_dim=1,
            action_horizon=2,
            num_future_frames=1,
            idm_arch="flow_transformer",
            idm_flow_num_samples=1,
        )
        self.transition_encoder = _StateTargetTransitionEncoder()
        self.flow_head = _PerfectVelocityFlowHead()
        self.flow_head.config = self.config

    def forward(
        self,
        current_images,
        future_images,
        state,
        task_id,
        *,
        sample_noise=None,
        target_action=None,
        action_mask=None,
        mode="sample",
    ):
        del future_images, state, task_id, sample_noise, target_action, action_mask
        if mode != "sample":
            raise ValueError("fake flow IDM only supports sample mode")
        return torch.zeros((current_images.shape[0], 2, 1), device=current_images.device)


def _patch_zero_idm(monkeypatch, dataset, model_config) -> None:
    monkeypatch.setattr("diagnose_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "diagnose_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (_ZeroIdm().to(device), model_config),
    )
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)


class _FakeCachedWanVaeLatentDataset:
    instances = []

    def __init__(self, base_dataset, cache_dir, *, model_config):
        self.base_dataset = base_dataset
        self.cache_dir = cache_dir
        self.model_config = model_config
        self.instances.append(self)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = dict(self.base_dataset[index])
        item["wan_vae_latents"] = torch.full((1, 2, 1, 1), float(index + 1))
        return item


def test_future_sensitivity_gate_flags_collapsed_output() -> None:
    gate = future_sensitivity_gate(
        idm_mse=0.05,
        future_sensitivity=_future_sensitivity(output_delta_mse=1e-9, target_mse=0.6),
        output_delta_mse_min=1e-4,
        degradation_min=1e-4,
    )

    assert gate["future_blind"] is True
    assert gate["output_delta_mse_collapsed"] is True
    assert gate["degradation_collapsed"] is False
    assert gate["current_repeated_output_delta_mse"] == pytest.approx(1e-9)
    assert gate["real_vs_current_repeated_degradation"] == pytest.approx(0.55)
    assert len(gate["reasons"]) == 1


def test_future_sensitivity_gate_flags_low_degradation() -> None:
    gate = future_sensitivity_gate(
        idm_mse=0.5,
        future_sensitivity=_future_sensitivity(output_delta_mse=0.2, target_mse=0.5),
        output_delta_mse_min=1e-4,
        degradation_min=1e-4,
    )

    assert gate["future_blind"] is True
    assert gate["output_delta_mse_collapsed"] is False
    assert gate["degradation_collapsed"] is True
    assert gate["real_vs_current_repeated_degradation"] == pytest.approx(0.0)
    assert len(gate["reasons"]) == 1


def test_future_sensitivity_gate_passes_responsive_model() -> None:
    gate = future_sensitivity_gate(
        idm_mse=0.05,
        future_sensitivity=_future_sensitivity(output_delta_mse=0.3, target_mse=0.8),
        output_delta_mse_min=1e-4,
        degradation_min=1e-4,
    )

    assert gate["future_blind"] is False
    assert gate["output_delta_mse_collapsed"] is False
    assert gate["degradation_collapsed"] is False
    assert gate["reasons"] == []


def test_future_sensitivity_gate_thresholds_are_configurable() -> None:
    # A model that passes the default floor can still be flagged under a stricter gate.
    sensitivity = _future_sensitivity(output_delta_mse=0.02, target_mse=0.1)
    lenient = future_sensitivity_gate(
        idm_mse=0.05, future_sensitivity=sensitivity, output_delta_mse_min=1e-4, degradation_min=1e-4
    )
    strict = future_sensitivity_gate(
        idm_mse=0.05, future_sensitivity=sensitivity, output_delta_mse_min=0.1, degradation_min=0.1
    )

    assert lenient["future_blind"] is False
    assert strict["future_blind"] is True


def test_diagnose_idm_cached_wan_latent_arg_is_forwarded_and_dataset_is_wrapped(tmp_path, monkeypatch) -> None:
    dataset, _ = _zero_idm_dataset_and_config()
    model_config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=4,
        action_dim=1,
        action_horizon=2,
        num_future_frames=1,
        idm_visual_encoder="wan_vae",
    )
    idm = _LatentAwareZeroIdm()
    load_calls = []
    create_calls = []
    _FakeCachedWanVaeLatentDataset.instances = []
    cache_dir = tmp_path / "wan_latents"

    def fake_create_dataset(config, cache):
        create_calls.append((config, cache))
        return dataset

    def fake_load_idm_checkpoint(path, device, *, use_cached_wan_vae_latents=False):
        load_calls.append(use_cached_wan_vae_latents)
        return idm.to(device), model_config

    monkeypatch.setattr("diagnose_idm.create_dataset_with_optional_cache", fake_create_dataset)
    monkeypatch.setattr("diagnose_idm.load_idm_checkpoint", fake_load_idm_checkpoint)
    monkeypatch.setattr("diagnose_idm.CachedWanVaeLatentDataset", _FakeCachedWanVaeLatentDataset)
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
            wan_vae_latent_cache_dir=str(cache_dir),
        )
    )

    metrics = json.loads((tmp_path / "idm_diagnostics.json").read_text())
    wrapper = _FakeCachedWanVaeLatentDataset.instances[0]

    assert load_calls == [True]
    assert create_calls[0][1] is None
    assert wrapper.base_dataset is dataset
    assert wrapper.cache_dir == str(cache_dir)
    assert wrapper.model_config is model_config
    assert metrics["wan_vae_latent_cache_dir"] == str(cache_dir)
    assert all(latents is not None for latents in idm.seen_latents)


def test_diagnose_idm_default_does_not_use_cached_wan_latents(tmp_path, monkeypatch) -> None:
    dataset, model_config = _zero_idm_dataset_and_config()
    idm = _LatentAwareZeroIdm()
    load_calls = []
    create_calls = []

    def fake_create_dataset(config, cache):
        create_calls.append((config, cache))
        return dataset

    def fake_load_idm_checkpoint(path, device, *, use_cached_wan_vae_latents=False):
        load_calls.append(use_cached_wan_vae_latents)
        return idm.to(device), model_config

    def fail_cached_wan_wrapper(*args, **kwargs):
        raise AssertionError("CachedWanVaeLatentDataset should not be constructed by default.")

    monkeypatch.setattr("diagnose_idm.create_dataset_with_optional_cache", fake_create_dataset)
    monkeypatch.setattr("diagnose_idm.load_idm_checkpoint", fake_load_idm_checkpoint)
    monkeypatch.setattr("diagnose_idm.CachedWanVaeLatentDataset", fail_cached_wan_wrapper)
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
        )
    )

    metrics = json.loads((tmp_path / "idm_diagnostics.json").read_text())

    assert load_calls == [False]
    assert create_calls[0][1] is None
    assert metrics["wan_vae_latent_cache_dir"] is None
    assert all(latents is None for latents in idm.seen_latents)


def test_diagnose_idm_rejects_cached_future_with_wan_latent_cache(tmp_path, monkeypatch) -> None:
    def fail_load_idm_checkpoint(*args, **kwargs):
        raise AssertionError("checkpoint should not be loaded when cache options are incompatible.")

    monkeypatch.setattr("diagnose_idm.load_idm_checkpoint", fail_load_idm_checkpoint)

    with pytest.raises(ValueError, match="cannot be combined with cached futures"):
        diagnose_main(
            DiagnoseArgs(
                checkpoint="fake.pt",
                output_dir=str(tmp_path),
                cached_future_dir=str(tmp_path / "future_cache"),
                wan_vae_latent_cache_dir=str(tmp_path / "wan_latents"),
            )
        )


def test_diagnose_idm_reports_future_sensitivity_gate(tmp_path, monkeypatch) -> None:
    dataset, model_config = _zero_idm_dataset_and_config()
    _patch_zero_idm(monkeypatch, dataset, model_config)

    # Default fail flag is off: the gate is reported but never raises.
    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
        )
    )

    gate = json.loads((tmp_path / "idm_diagnostics.json").read_text())["future_sensitivity_gate"]
    # The zero IDM ignores its future input entirely, so both signals collapse.
    assert gate["future_blind"] is True
    assert gate["output_delta_mse_collapsed"] is True
    assert gate["degradation_collapsed"] is True
    assert gate["reasons"]


def test_diagnose_idm_fails_on_future_blind_when_requested(tmp_path, monkeypatch) -> None:
    dataset, model_config = _zero_idm_dataset_and_config()
    _patch_zero_idm(monkeypatch, dataset, model_config)

    with pytest.raises(SystemExit):
        diagnose_main(
            DiagnoseArgs(
                checkpoint="fake.pt",
                output_dir=str(tmp_path),
                image_size=8,
                action_horizon=2,
                batch_size=1,
                device="cpu",
                fail_on_future_blind=True,
            )
        )

    # The diagnostics artifact is still written before the gate fails the run.
    metrics = json.loads((tmp_path / "idm_diagnostics.json").read_text())
    assert metrics["future_sensitivity_gate"]["future_blind"] is True


def test_diagnose_idm_writes_metrics_and_plots(tmp_path) -> None:
    train_dir = tmp_path / "train"
    diagnostics_dir = tmp_path / "diagnostics"
    run_idm_training(
        TrainConfig(
            dataset=DatasetConfig(
                source="synthetic",
                image_keys=("corner4.image",),
                image_size=32,
                synthetic_samples=16,
                num_future_frames=4,
                action_horizon=8,
            ),
            output_dir=str(train_dir),
            epochs=1,
            batch_size=4,
            device="cpu",
            seed=13,
        )
    )

    diagnose_main(
        DiagnoseArgs(
            checkpoint=str(train_dir / "idm_checkpoint.pt"),
            dataset_source="synthetic",
            image_keys=("corner4.image",),
            output_dir=str(diagnostics_dir),
            image_size=32,
            synthetic_samples=16,
            num_future_frames=4,
            action_horizon=8,
            batch_size=4,
            device="cpu",
            seed=13,
        )
    )

    metrics = json.loads((diagnostics_dir / "idm_diagnostics.json").read_text())

    assert metrics["num_samples"] == 16
    assert metrics["num_valid_actions"] == 16 * 8
    assert metrics["idm_mse"] >= 0.0
    assert set(metrics["future_sensitivity"]) == {"current_repeated", "noise", "shuffled", "zero"}
    assert metrics["future_sensitivity"]["shuffled"]["output_delta_mse"] >= 0.0
    assert set(metrics["state_sensitivity"]) == {"zero"}
    assert metrics["state_sensitivity"]["zero"]["output_delta_mse"] >= 0.0
    assert metrics["mean_action_baseline"]["idm_mse"] >= 0.0
    assert len(metrics["per_action_dim_mse"]) == 4
    # Stepwise diagnostics span the full action horizon on the real (flow-matching) path.
    assert len(metrics["per_action_step_mse"]) == 8
    assert len(metrics["per_action_step_mae"]) == 8
    assert len(metrics["per_action_step_valid_count"]) == 8
    assert len(metrics["per_action_step_dim_mse"]) == 8
    assert all(len(row) == 4 for row in metrics["per_action_step_dim_mse"])
    assert (diagnostics_dir / "action_trace.png").exists()
    assert (diagnostics_dir / "action_histograms.png").exists()


def test_diagnose_idm_weights_metrics_by_valid_action_elements(tmp_path, monkeypatch) -> None:
    class ZeroIdm(torch.nn.Module):
        def forward(self, current_images, future_images, state, task_id, *, sample_noise=None):
            return torch.zeros((current_images.shape[0], 2, 1))

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
    monkeypatch.setattr("diagnose_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "diagnose_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (ZeroIdm().to(device), model_config),
    )
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
        )
    )

    metrics = json.loads((tmp_path / "idm_diagnostics.json").read_text())

    assert metrics["num_valid_actions"] == 3
    assert metrics["idm_mse"] == pytest.approx((1.0 + 1.0 + 100.0) / 3.0)
    assert metrics["idm_smooth_l1"] == pytest.approx((0.5 + 0.5 + 9.5) / 3.0)
    assert metrics["future_sensitivity"]["zero"]["target_mse"] == pytest.approx(metrics["idm_mse"])
    assert metrics["state_sensitivity"]["zero"]["target_mse"] == pytest.approx(metrics["idm_mse"])
    assert metrics["state_sensitivity"]["zero"]["output_delta_mse"] == pytest.approx(0.0)


def test_diagnose_idm_forwards_history_kwargs_from_dataset(tmp_path, monkeypatch) -> None:
    item = {
        "current_images": torch.zeros(1, 3, 8, 8),
        "future_images": torch.zeros(1, 1, 3, 8, 8),
        "state": torch.zeros(4),
        "task_id": torch.tensor(0),
        "action_chunk": torch.zeros(2, 1),
        "action_mask": torch.ones(2),
        "prev_state_history": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "prev_action_history": torch.tensor([[5.0]]),
        "history_mask": torch.tensor([1.0]),
    }
    dataset = [item]
    idm = _HistoryRequiredIdm()
    model_config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=4,
        action_dim=1,
        action_horizon=2,
        num_future_frames=1,
        idm_arch="flow_transformer",
        idm_history_length=1,
    )
    captured_configs: list[DatasetConfig] = []
    monkeypatch.setattr(
        "diagnose_idm.create_dataset_with_optional_cache",
        lambda config, cache: captured_configs.append(config) or dataset,
    )
    monkeypatch.setattr(
        "diagnose_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (idm.to(device), model_config),
    )
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
        )
    )

    metrics = json.loads((tmp_path / "idm_diagnostics.json").read_text())

    assert captured_configs[0].idm_history_length == 1
    assert metrics["dataset_config"]["idm_history_length"] == 1
    assert idm.seen_history
    assert torch.allclose(idm.seen_history[0][0], item["prev_state_history"].unsqueeze(0))
    assert torch.allclose(idm.seen_history[0][1], item["prev_action_history"].unsqueeze(0))
    assert torch.allclose(idm.seen_history[0][2], item["history_mask"].unsqueeze(0))


def test_diagnose_idm_applies_state_normalizer_explicitly(tmp_path, monkeypatch) -> None:
    class StateEchoIdm(torch.nn.Module):
        def forward(self, current_images, future_images, state, task_id, *, sample_noise=None):
            del future_images, task_id, sample_noise
            return state.view(current_images.shape[0], 2, 1)

    dataset = [
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.tensor([12.0, 14.0]),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[1.0], [2.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        }
    ]
    model_config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=2,
        action_dim=1,
        action_horizon=2,
        num_future_frames=1,
    )
    idm = StateEchoIdm()
    normalizer = StateNormalizer(mean=torch.tensor([10.0, 10.0]), std=torch.tensor([2.0, 2.0]))
    attach_state_normalizer(idm, normalizer, normalize_forward=False)
    monkeypatch.setattr("diagnose_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "diagnose_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (idm.to(device), model_config),
    )
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
        )
    )

    metrics = json.loads((tmp_path / "idm_diagnostics.json").read_text())

    assert metrics["idm_mse"] == pytest.approx(0.0)
    assert metrics["state_normalizer"] == {"mean": [10.0, 10.0], "std": [2.0, 2.0]}


def test_diagnose_idm_reports_state_sensitivity(tmp_path, monkeypatch) -> None:
    class StateEchoIdm(torch.nn.Module):
        def forward(self, current_images, future_images, state, task_id, *, sample_noise=None):
            del future_images, task_id, sample_noise
            return state.view(current_images.shape[0], 2, 1)

    dataset = [
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.tensor([1.0, 3.0]),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[1.0], [3.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        }
    ]
    model_config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=2,
        action_dim=1,
        action_horizon=2,
        num_future_frames=1,
    )
    monkeypatch.setattr("diagnose_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "diagnose_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (StateEchoIdm().to(device), model_config),
    )
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
        )
    )

    metrics = json.loads((tmp_path / "idm_diagnostics.json").read_text())
    state_sensitivity = metrics["state_sensitivity"]["zero"]

    assert metrics["idm_mse"] == pytest.approx(0.0)
    assert state_sensitivity["target_mse"] == pytest.approx((1.0 + 9.0) / 2.0)
    assert state_sensitivity["output_delta_mse"] == pytest.approx((1.0 + 9.0) / 2.0)


def test_diagnose_idm_reports_flow_teacher_forced_metrics_with_masks(tmp_path, monkeypatch) -> None:
    dataset = [
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.tensor([10.0, 12.0]),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[1.0], [3.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        },
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.tensor([14.0, 1008.0]),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[5.0], [999.0]]),
            "action_mask": torch.tensor([1.0, 0.0]),
        },
    ]
    idm = _TeacherForcedFlowIdm()
    action_normalizer = ActionNormalizer(mean=torch.tensor([1.0]), std=torch.tensor([2.0]))
    state_normalizer = StateNormalizer(mean=torch.tensor([10.0, 10.0]), std=torch.tensor([2.0, 2.0]))
    attach_action_normalizer(idm, action_normalizer)
    attach_state_normalizer(idm, state_normalizer, normalize_forward=True)
    monkeypatch.setattr("diagnose_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "diagnose_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (idm.to(device), idm.config),
    )
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
            flow_eval_seed=123,
        )
    )

    metrics = json.loads((tmp_path / "idm_diagnostics.json").read_text())
    flow_metrics = metrics["flow_teacher_forced"]

    assert metrics["num_valid_actions"] == 3
    assert metrics["idm_mse"] == pytest.approx((0.0 + 4.0 + 16.0) / 3.0)
    assert flow_metrics["normalized_t0_endpoint_mse"] == pytest.approx(0.0)
    assert flow_metrics["normalized_t0_5_endpoint_mse"] == pytest.approx(0.0)
    assert flow_metrics["normalized_t0_velocity_mse"] == pytest.approx(0.0)
    assert flow_metrics["normalized_t0_5_velocity_mse"] == pytest.approx(0.0)
    assert flow_metrics["denormalized_t0_endpoint_mse"] == pytest.approx(0.0)
    assert flow_metrics["denormalized_t0_5_endpoint_mse"] == pytest.approx(0.0)
    assert flow_metrics["normalized_sampled_vs_t0_endpoint_mse"] == pytest.approx((0.0 + 1.0 + 4.0) / 3.0)
    assert flow_metrics["sampled_vs_endpoint_mse"] == pytest.approx((0.0 + 4.0 + 16.0) / 3.0)
    assert set(flow_metrics["future_sensitivity"]) == {"current_repeated", "noise", "shuffled", "zero"}
    assert flow_metrics["future_sensitivity"]["zero"]["normalized_t0_endpoint_output_delta_mse"] == pytest.approx(0.0)
    assert flow_metrics["future_sensitivity"]["zero"]["normalized_t0_5_velocity_output_delta_mse"] == pytest.approx(0.0)
    assert set(flow_metrics["state_sensitivity"]) == {"zero"}
    assert flow_metrics["state_sensitivity"]["zero"]["normalized_t0_5_endpoint_target_mse"] > 0.0
    assert flow_metrics["state_sensitivity"]["zero"]["normalized_t0_5_endpoint_output_delta_mse"] > 0.0


def test_diagnose_idm_flow_num_samples_override_is_applied_and_recorded(tmp_path, monkeypatch) -> None:
    dataset = [
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.tensor([1.0, 2.0]),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[1.0], [2.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        }
    ]

    class RecordingFlowIdm(_TeacherForcedFlowIdm):
        def __init__(self) -> None:
            super().__init__()
            self.sample_noise_shapes: list[tuple[int, ...] | None] = []

        def forward(self, *args, sample_noise=None, **kwargs):
            self.sample_noise_shapes.append(None if sample_noise is None else tuple(sample_noise.shape))
            assert self.config.idm_flow_num_samples == 4
            assert self.flow_head.config.idm_flow_num_samples == 4
            return super().forward(*args, sample_noise=sample_noise, **kwargs)

    idm = RecordingFlowIdm()
    monkeypatch.setattr("diagnose_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "diagnose_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (idm.to(device), idm.config),
    )
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
            flow_num_samples=4,
        )
    )

    metrics = json.loads((tmp_path / "idm_diagnostics.json").read_text())

    assert metrics["flow_num_samples"] == 4
    assert idm.sample_noise_shapes
    assert all(shape == (4, 2, 1) for shape in idm.sample_noise_shapes)
    assert idm.config.idm_flow_num_samples == 1
    assert idm.flow_head.config.idm_flow_num_samples == 1


def test_diagnose_idm_flow_noise_scale_override_is_applied_and_recorded(tmp_path, monkeypatch) -> None:
    dataset = [
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.tensor([1.0, 2.0]),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[1.0], [2.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        }
    ]

    class RecordingFlowIdm(_TeacherForcedFlowIdm):
        def __init__(self) -> None:
            super().__init__()
            self.sample_noise_abs_sums: list[float] = []

        def forward(self, *args, sample_noise=None, **kwargs):
            assert self.config.idm_flow_sample_noise_scale == pytest.approx(0.0)
            assert self.flow_head.config.idm_flow_sample_noise_scale == pytest.approx(0.0)
            if sample_noise is not None:
                self.sample_noise_abs_sums.append(float(sample_noise.abs().sum().item()))
            return super().forward(*args, sample_noise=sample_noise, **kwargs)

    idm = RecordingFlowIdm()
    monkeypatch.setattr("diagnose_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "diagnose_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (idm.to(device), idm.config),
    )
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
            flow_noise_scale=0.0,
        )
    )

    metrics = json.loads((tmp_path / "idm_diagnostics.json").read_text())

    assert metrics["flow_noise_scale"] == pytest.approx(0.0)
    assert idm.sample_noise_abs_sums
    assert all(value == pytest.approx(0.0) for value in idm.sample_noise_abs_sums)
    assert idm.config.idm_flow_sample_noise_scale == pytest.approx(1.0)
    assert idm.flow_head.config.idm_flow_sample_noise_scale == pytest.approx(1.0)


def test_diagnose_idm_reports_flow_future_ranking_for_image_encoder(tmp_path, monkeypatch) -> None:
    dataset = [
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.full((1, 1, 3, 8, 8), 2.0),
            "state": torch.zeros(2),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[2.0], [2.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        },
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.full((1, 1, 3, 8, 8), -2.0),
            "state": torch.zeros(2),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[-2.0], [-2.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        },
    ]
    idm = _FutureAwareTeacherForcedFlowIdm(_FutureMeanTransitionEncoder())
    monkeypatch.setattr("diagnose_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "diagnose_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (idm.to(device), idm.config),
    )
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=2,
            device="cpu",
            flow_eval_seed=123,
        )
    )

    ranking = json.loads((tmp_path / "idm_diagnostics.json").read_text())["flow_teacher_forced"]["future_ranking"]
    candidate_mse = ranking["candidate_teacher_forced_endpoint_mse"]

    assert ranking["candidate_order"] == ["real_gt", "current_repeated", "shuffled", "zero", "noise"]
    assert set(candidate_mse) == {"real_gt", "current_repeated", "shuffled", "zero", "noise"}
    assert candidate_mse["real_gt"] == pytest.approx(0.0)
    assert ranking["real_candidate_rank"] == 1
    assert ranking["mean_real_candidate_rank"] == pytest.approx(1.0)
    assert ranking["rank_accuracy"] == pytest.approx(1.0)
    assert ranking["real_vs_best_negative_gap"] > 0.0
    assert ranking["num_ranked_samples"] == 2
    assert ranking["score_mode"] == "teacher_forced_endpoint"
    assert ranking["time_value"] == pytest.approx(0.5)


def test_diagnose_idm_flow_future_ranking_sampled_action_mode(tmp_path, monkeypatch) -> None:
    dataset = [
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.full((1, 1, 3, 8, 8), 2.0),
            "state": torch.zeros(2),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[2.0], [2.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        },
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.full((1, 1, 3, 8, 8), -2.0),
            "state": torch.zeros(2),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[-2.0], [-2.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        },
    ]
    idm = _FutureAwareTeacherForcedFlowIdm(_FutureMeanTransitionEncoder())
    monkeypatch.setattr("diagnose_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "diagnose_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (idm.to(device), idm.config),
    )
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=2,
            device="cpu",
            flow_eval_seed=123,
            future_usage_score_mode="sampled_action",
        )
    )

    ranking = json.loads((tmp_path / "idm_diagnostics.json").read_text())["flow_teacher_forced"]["future_ranking"]

    # The sampled-action scorer replaces the teacher-forced endpoint candidate key while keeping the
    # same ranking semantics; the fake sampler mirrors the endpoint so the ranking is unchanged.
    assert ranking["score_mode"] == "sampled_action"
    assert ranking["time_value"] is None
    assert "candidate_teacher_forced_endpoint_mse" not in ranking
    candidate_mse = ranking["candidate_sampled_action_mse"]
    assert ranking["candidate_order"] == ["real_gt", "current_repeated", "shuffled", "zero", "noise"]
    assert set(candidate_mse) == {"real_gt", "current_repeated", "shuffled", "zero", "noise"}
    assert candidate_mse["real_gt"] == pytest.approx(0.0)
    assert ranking["real_candidate_rank"] == 1
    assert ranking["rank_accuracy"] == pytest.approx(1.0)
    assert ranking["real_vs_best_negative_gap"] > 0.0
    assert ranking["num_ranked_samples"] == 2


def test_diagnose_idm_flow_future_ranking_uses_cached_wan_latents(tmp_path, monkeypatch) -> None:
    dataset = [
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.zeros(2),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[2.0], [2.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        },
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.zeros(2),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[-2.0], [-2.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        },
    ]

    class RankingLatentDataset:
        def __init__(self, base_dataset, cache_dir, *, model_config):
            del cache_dir, model_config
            self.base_dataset = base_dataset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, index):
            item = dict(self.base_dataset[index])
            value = float(item["action_chunk"][0, 0])
            item["wan_vae_latents"] = torch.tensor([[[[0.0]], [[value]]]])
            return item

    idm = _FutureAwareTeacherForcedFlowIdm(
        _FutureLatentTransitionEncoder(),
        idm_visual_encoder="wan_vae",
    )
    monkeypatch.setattr("diagnose_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "diagnose_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (idm.to(device), idm.config),
    )
    monkeypatch.setattr("diagnose_idm.CachedWanVaeLatentDataset", RankingLatentDataset)
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=2,
            device="cpu",
            flow_eval_seed=123,
            wan_vae_latent_cache_dir=str(tmp_path / "wan_latents"),
        )
    )

    ranking = json.loads((tmp_path / "idm_diagnostics.json").read_text())["flow_teacher_forced"]["future_ranking"]
    candidate_mse = ranking["candidate_teacher_forced_endpoint_mse"]

    assert candidate_mse["real_gt"] == pytest.approx(0.0)
    assert candidate_mse["current_repeated"] > 0.0
    assert candidate_mse["zero"] > 0.0
    assert ranking["real_candidate_rank"] == 1
    assert ranking["rank_accuracy"] == pytest.approx(1.0)
    assert ranking["real_vs_best_negative_gap"] > 0.0


def test_diagnose_idm_flow_cross_attention_uses_cached_wan_latent_tokens(tmp_path, monkeypatch) -> None:
    dataset = [
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.zeros(2),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[2.0], [2.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        },
        {
            "current_images": torch.zeros(1, 3, 8, 8),
            "future_images": torch.zeros(1, 1, 3, 8, 8),
            "state": torch.zeros(2),
            "task_id": torch.tensor(0),
            "action_chunk": torch.tensor([[-2.0], [-2.0]]),
            "action_mask": torch.tensor([1.0, 1.0]),
        },
    ]

    class RankingLatentDataset:
        def __init__(self, base_dataset, cache_dir, *, model_config):
            del cache_dir, model_config
            self.base_dataset = base_dataset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, index):
            item = dict(self.base_dataset[index])
            value = float(item["action_chunk"][0, 0])
            item["wan_vae_latents"] = torch.tensor([[[[0.0]], [[value]]]])
            return item

    idm = _FutureAwareTeacherForcedFlowIdm(
        _FutureLatentTokenTransitionEncoder(),
        idm_visual_encoder="wan_vae",
        idm_flow_visual_token_conditioning=True,
        idm_flow_visual_token_conditioning_mode="cross_attention",
    )
    idm.flow_head = _VisualTokenRequiredPerfectVelocityFlowHead()
    monkeypatch.setattr("diagnose_idm.create_dataset_with_optional_cache", lambda config, cache: dataset)
    monkeypatch.setattr(
        "diagnose_idm.load_idm_checkpoint",
        lambda path, device, *, use_cached_wan_vae_latents=False: (idm.to(device), idm.config),
    )
    monkeypatch.setattr("diagnose_idm.CachedWanVaeLatentDataset", RankingLatentDataset)
    monkeypatch.setattr("diagnose_idm.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=2,
            device="cpu",
            flow_eval_seed=123,
            wan_vae_latent_cache_dir=str(tmp_path / "wan_latents"),
        )
    )

    ranking = json.loads((tmp_path / "idm_diagnostics.json").read_text())["flow_teacher_forced"]["future_ranking"]
    seen_tokens = [values.squeeze(1) for values in idm.flow_head.visual_token_values]

    assert ranking["candidate_teacher_forced_endpoint_mse"]["real_gt"] == pytest.approx(0.0)
    assert ranking["rank_accuracy"] == pytest.approx(1.0)
    assert torch.equal(seen_tokens[0], torch.tensor([2.0, -2.0]))
    assert torch.equal(seen_tokens[1], torch.tensor([2.0, -2.0]))
    assert any(torch.equal(values, torch.tensor([-2.0, 2.0])) for values in seen_tokens)
    assert any(torch.equal(values, torch.tensor([0.0, 0.0])) for values in seen_tokens)


def test_stepwise_action_diagnostics_aggregates_per_horizon_step() -> None:
    # All-zero predictions so each horizon step has a hand-checkable error.
    predicted = torch.zeros(2, 2, 1)
    target = torch.tensor([[[1.0], [1.0]], [[10.0], [0.0]]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    diag = stepwise_action_diagnostics(predicted, target, mask)

    # Step 0: errors 1^2 and 10^2 over 2 valid samples -> (1 + 100) / 2 = 50.5.
    # Step 1: only sample 0 is valid -> 1^2 / 1 = 1.0.
    assert diag["per_action_step_mse"] == pytest.approx([50.5, 1.0])
    assert diag["per_action_step_mae"] == pytest.approx([5.5, 1.0])
    assert diag["per_action_step_valid_count"] == [2, 1]
    assert diag["first_action_mse"] == pytest.approx(50.5)
    assert diag["first_action_mae"] == pytest.approx(5.5)
    assert diag["last_action_mse"] == pytest.approx(1.0)
    assert diag["last_action_mae"] == pytest.approx(1.0)
    assert diag["per_action_step_dim_mse"][0] == pytest.approx([50.5])
    assert diag["per_action_step_dim_mse"][1] == pytest.approx([1.0])


def test_stepwise_action_diagnostics_averages_over_action_dims() -> None:
    # Single valid sample/step with two action dims: per-step MSE averages the dims.
    predicted = torch.zeros(1, 1, 2)
    target = torch.tensor([[[2.0, 4.0]]])
    mask = torch.ones(1, 1)

    diag = stepwise_action_diagnostics(predicted, target, mask)

    assert diag["per_action_step_mse"] == pytest.approx([(4.0 + 16.0) / 2.0])
    assert diag["per_action_step_mae"] == pytest.approx([(2.0 + 4.0) / 2.0])
    assert diag["per_action_step_dim_mse"][0] == pytest.approx([4.0, 16.0])


def test_stepwise_action_diagnostics_handles_fully_masked_step() -> None:
    # The trailing horizon step has no valid samples: report 0.0 with a 0 count
    # rather than dividing by zero.
    predicted = torch.zeros(2, 3, 1)
    target = torch.ones(2, 3, 1)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    diag = stepwise_action_diagnostics(predicted, target, mask)

    assert diag["per_action_step_valid_count"] == [2, 2, 0]
    assert diag["per_action_step_mse"] == pytest.approx([1.0, 1.0, 0.0])
    assert diag["last_action_mse"] == pytest.approx(0.0)
    assert diag["last_action_mae"] == pytest.approx(0.0)


def test_diagnose_idm_reports_per_action_step_metrics(tmp_path, monkeypatch) -> None:
    dataset, model_config = _zero_idm_dataset_and_config()
    _patch_zero_idm(monkeypatch, dataset, model_config)

    diagnose_main(
        DiagnoseArgs(
            checkpoint="fake.pt",
            output_dir=str(tmp_path),
            image_size=8,
            action_horizon=2,
            batch_size=1,
            device="cpu",
        )
    )

    metrics = json.loads((tmp_path / "idm_diagnostics.json").read_text())

    assert metrics["per_action_step_mse"] == pytest.approx([50.5, 1.0])
    assert metrics["per_action_step_mae"] == pytest.approx([5.5, 1.0])
    assert metrics["per_action_step_valid_count"] == [2, 1]
    assert metrics["first_action_mse"] == pytest.approx(50.5)
    assert metrics["first_action_mae"] == pytest.approx(5.5)
    assert metrics["last_action_mse"] == pytest.approx(1.0)
    assert metrics["last_action_mae"] == pytest.approx(1.0)
    assert metrics["per_action_step_dim_mse"][0] == pytest.approx([50.5])
    assert metrics["per_action_step_dim_mse"][1] == pytest.approx([1.0])
    # The per-step valid counts partition the aggregate valid-action total...
    assert sum(metrics["per_action_step_valid_count"]) == metrics["num_valid_actions"]
    # ...and the count-weighted mean of the per-step MSEs reproduces the aggregate idm_mse.
    counts = metrics["per_action_step_valid_count"]
    weighted = sum(mse * count for mse, count in zip(metrics["per_action_step_mse"], counts)) / sum(counts)
    assert weighted == pytest.approx(metrics["idm_mse"])
