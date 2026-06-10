from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import world_model.train_lib as train_lib
from train_idm import Args as TrainIdmArgs
from train_idm import main as train_idm_main
from world_model.config import DatasetConfig, ModelConfig, TrainConfig
from world_model.models import InverseDynamicsModel
from world_model.train_lib import (
    ActionNormalizer,
    StateNormalizer,
    apply_current_conditioning_dropout,
    apply_future_augmentation,
    apply_wan_vae_latent_noise,
    attach_state_normalizer,
    compute_idm_losses,
    context_action_loss_weight_for_epoch,
    create_dataset_with_optional_cache,
    evaluate_idm,
    evaluate_idm_future_usage,
    future_ranking_weight_for_epoch,
    get_state_normalizer,
    is_better_idm_checkpoint_row,
    load_idm_checkpoint,
    module_state_dict_for_checkpoint,
    normalize_state_for_idm,
    run_idm_training,
    run_training,
    save_idm_state_checkpoint,
    state_normalizer_applies_in_forward,
    train_idm_one_epoch,
)
from world_model.wan_vae_encoder import FakeWanVaeEncoder


class _ControlledStateDataset(Dataset):
    def __init__(self) -> None:
        self.states = [
            torch.tensor([0.0, 10.0]),
            torch.tensor([1.0, 11.0]),
            torch.tensor([2.0, 12.0]),
            torch.tensor([3.0, 13.0]),
            torch.tensor([4.0, 14.0]),
            torch.tensor([5.0, 15.0]),
            torch.tensor([1000.0, 2000.0]),
            torch.tensor([3000.0, 4000.0]),
        ]

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "current_images": torch.zeros(1, 3, 16, 16),
            "future_images": torch.zeros(1, 1, 3, 16, 16),
            "future_image_mask": torch.ones(1),
            "state": self.states[index],
            "action_chunk": torch.zeros(2, 1),
            "action_mask": torch.ones(2),
            "task_id": torch.tensor(0, dtype=torch.long),
        }


class _TaskMetadataDataset(Dataset):
    def __init__(self, tasks: list[int], *, key: str = "task_index") -> None:
        self.tasks = tasks
        self.key = key

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            self.key: torch.tensor(self.tasks[index], dtype=torch.long),
            "dataset_index": torch.tensor(index, dtype=torch.long),
        }


class _NoTaskMetadataDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"dataset_index": torch.tensor(index, dtype=torch.long)}


class _ExternalWanVaeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.external_weight = torch.nn.Parameter(torch.ones(1))

    @torch.no_grad()
    def encode_videos(self, videos: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_frames, height, width = videos.shape
        return torch.zeros(
            batch_size,
            48,
            (num_frames + 3) // 4,
            height // 16,
            width // 16,
            device=videos.device,
            dtype=torch.float32,
        )


class _ForbiddenTransitionEncoder(torch.nn.Module):
    def forward(self, current_images, future_images, state):
        del current_images, future_images, state
        raise AssertionError("contrastive endpoint path should not run when its weight is zero")


class _FutureScalarTransitionEncoder(torch.nn.Module):
    def forward(self, current_images, future_images, state):
        del current_images, state
        return future_images[:, :1, :1, :1, :1, :1].reshape(future_images.shape[0], 1, 1)


class _CurrentScalarTransitionEncoder(torch.nn.Module):
    def forward(self, current_images, future_images, state):
        del future_images, state
        return current_images[:, :1, :1, :1, :1].reshape(current_images.shape[0], 1, 1)


class _WanLatentScalarTransitionEncoder(torch.nn.Module):
    def forward(self, current_images, future_images, state, *, wan_vae_latents=None):
        del current_images, future_images, state
        if wan_vae_latents is None:
            raise AssertionError("wan_vae_latents are required for this fake latent IDM")
        return wan_vae_latents[:, :1, 1, :1, :1].reshape(wan_vae_latents.shape[0], 1, 1)


class _RecordingWanLatentScalarTransitionEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_latents: list[torch.Tensor] = []

    def forward(self, current_images, future_images, state, *, wan_vae_latents=None):
        del current_images, future_images, state
        if wan_vae_latents is None:
            raise AssertionError("wan_vae_latents are required for this fake latent IDM")
        self.seen_latents.append(wan_vae_latents.detach().cpu())
        return wan_vae_latents[:, :1, 1, :1, :1].reshape(wan_vae_latents.shape[0], 1, 1)


class _WanLatentLastScalarTransitionEncoder(torch.nn.Module):
    def forward(self, current_images, future_images, state, *, wan_vae_latents=None):
        del current_images, future_images, state
        if wan_vae_latents is None:
            raise AssertionError("wan_vae_latents are required for this fake latent IDM")
        return wan_vae_latents[:, :1, -1, :1, :1].reshape(wan_vae_latents.shape[0], 1, 1)


class _WanLatentTokenTransitionEncoder(torch.nn.Module):
    def forward(self, current_images, future_images, state, *, wan_vae_latents=None, return_tokens=False):
        del current_images, future_images, state
        if wan_vae_latents is None:
            raise AssertionError("wan_vae_latents are required for this fake token IDM")
        future_value = wan_vae_latents[:, :1, 1, :1, :1].reshape(wan_vae_latents.shape[0], 1, 1)
        if return_tokens:
            return future_value, future_value
        return future_value


class _RecordingWanLatentFlowIdm(torch.nn.Module):
    uses_flow_matching = True

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.seen_current_images: list[torch.Tensor] = []
        self.seen_latents: list[torch.Tensor] = []
        self.config = ModelConfig(
            num_views=1,
            image_size=1,
            state_dim=1,
            action_dim=1,
            action_horizon=1,
            num_future_frames=1,
            idm_arch="flow_transformer",
            idm_visual_encoder="wan_vae",
        )

    def forward(
        self,
        current_images,
        future_images,
        state,
        task_id,
        *,
        wan_vae_latents=None,
        sample_noise=None,
        target_action=None,
        action_mask=None,
        mode="sample",
    ):
        del future_images, state, task_id, sample_noise, action_mask
        self.seen_current_images.append(current_images.detach().cpu())
        if wan_vae_latents is not None:
            self.seen_latents.append(wan_vae_latents.detach().cpu())
        if mode == "loss":
            if target_action is None:
                raise ValueError("target_action is required")
            return {"loss": self.scale * 0.0 + target_action.new_tensor(1.0)}
        return torch.zeros((current_images.shape[0], 1, 1), device=current_images.device)


class _EndpointFlowHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.noisy_actions: list[torch.Tensor] = []

    def forward(self, context, noisy_action, time, *, history_tokens=None):
        del history_tokens
        self.noisy_actions.append(noisy_action.detach().cpu())
        endpoint = context * self.scale
        return (endpoint - noisy_action) / (1.0 - time.view(-1, 1, 1)).clamp_min(1e-6)


class _VisualTokenRequiredEndpointFlowHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.visual_token_values: list[torch.Tensor] = []

    def forward(self, context, noisy_action, time, *, history_tokens=None, visual_context_tokens=None):
        del history_tokens
        if visual_context_tokens is None:
            raise AssertionError("visual_context_tokens were not forwarded to the cross-attention flow head")
        self.visual_token_values.append(visual_context_tokens[..., 0].detach().cpu())
        endpoint = context * self.scale
        return (endpoint - noisy_action) / (1.0 - time.view(-1, 1, 1)).clamp_min(1e-6)


class _AnchorAwareTransitionEncoder(torch.nn.Module):
    def forward(self, current_images, future_images, state):
        current_value = current_images[:, :1, :1, :1, :1].reshape(current_images.shape[0], 1, 1)
        future_value = future_images[:, :1, :1, :1, :1, :1].reshape(future_images.shape[0], 1, 1)
        state_value = state[:, :1].reshape(state.shape[0], 1, 1)
        return 10.0 * current_value + future_value + 100.0 * state_value


class _HistoryAwareEndpointFlowHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.noisy_actions: list[torch.Tensor] = []
        self.history_values: list[torch.Tensor] = []

    def forward(self, context, noisy_action, time, *, history_tokens=None):
        self.noisy_actions.append(noisy_action.detach().cpu())
        history_value = torch.zeros_like(context) if history_tokens is None else history_tokens.mean(dim=1)
        self.history_values.append(history_value.detach().cpu())
        endpoint = (context + 1000.0 * history_value) * self.scale
        return (endpoint - noisy_action) / (1.0 - time.view(-1, 1, 1)).clamp_min(1e-6)


class _AnchorAwareFlowIdm(torch.nn.Module):
    uses_flow_matching = True

    def __init__(self) -> None:
        super().__init__()
        self.config = ModelConfig(
            num_views=1,
            image_size=1,
            state_dim=1,
            action_dim=1,
            action_horizon=1,
            idm_history_length=1,
            num_future_frames=1,
            idm_arch="flow_transformer",
        )
        self.transition_encoder = _AnchorAwareTransitionEncoder()
        self.flow_head = _HistoryAwareEndpointFlowHead()

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
        prev_state_history=None,
        prev_action_history=None,
        history_mask=None,
    ):
        del future_images, state, task_id, sample_noise, action_mask
        del prev_state_history, prev_action_history, history_mask
        if mode == "loss":
            if target_action is None:
                raise ValueError("target_action is required")
            return {"loss": self.flow_head.scale * target_action.new_tensor(0.0)}
        return torch.zeros((current_images.shape[0], 1, 1), device=current_images.device)

    def _history_tokens(self, prev_state_history, prev_action_history, history_mask):
        mask = history_mask.to(dtype=prev_state_history.dtype).unsqueeze(-1)
        return (prev_state_history + prev_action_history) * mask


class _FakeFlowIdm(torch.nn.Module):
    uses_flow_matching = True

    def __init__(
        self,
        *,
        endpoint_consistency_loss: float = 0.0,
        zero_start_endpoint_loss: float = 0.0,
        sampled_action_loss: float = 0.0,
        forbid_contrastive_path: bool = False,
        forbid_context_action_path: bool = False,
    ) -> None:
        super().__init__()
        self.endpoint_consistency_loss = endpoint_consistency_loss
        self.zero_start_endpoint_loss = zero_start_endpoint_loss
        self.sampled_action_loss = sampled_action_loss
        self.forbid_context_action_path = forbid_context_action_path
        self.seen_context_action_targets: list[torch.Tensor] = []
        self.config = ModelConfig(
            num_views=1,
            image_size=1,
            state_dim=1,
            action_dim=1,
            action_horizon=1,
            num_future_frames=1,
            idm_arch="flow_transformer",
        )
        self.transition_encoder = (
            _ForbiddenTransitionEncoder() if forbid_contrastive_path else _FutureScalarTransitionEncoder()
        )
        self.flow_head = _EndpointFlowHead()
        self.context_action_scale = torch.nn.Parameter(torch.tensor(0.0))

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
        del future_images, state, task_id, sample_noise, action_mask
        if mode == "loss":
            if target_action is None:
                raise ValueError("target_action is required")
            return {
                "loss": target_action.new_tensor(3.0) * self.flow_head.scale,
                "endpoint_consistency_loss": target_action.new_tensor(self.endpoint_consistency_loss),
                "zero_start_endpoint_loss": target_action.new_tensor(self.zero_start_endpoint_loss),
                "sampled_action_loss": target_action.new_tensor(self.sampled_action_loss),
            }
        return torch.zeros((current_images.shape[0], 1, 1), device=current_images.device)

    def context_action_loss(self, current_images, future_images, state, target_action, action_mask):
        del current_images, future_images, state
        if self.forbid_context_action_path:
            raise AssertionError("context action path should not run when its weight is zero")
        self.seen_context_action_targets.append(target_action.detach().cpu())
        predicted_action = torch.zeros_like(target_action) + self.context_action_scale
        mask = action_mask
        while mask.ndim < target_action.ndim:
            mask = mask.unsqueeze(-1)
        loss = ((predicted_action - target_action).square() * mask).sum() / mask.expand_as(target_action).sum()
        return {"loss": loss, "predicted_action": predicted_action}

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
        del sample_noise, wan_vae_latents, prev_state_history, prev_action_history, history_mask
        # Deterministic stand-in for the flow sampler: returns the same endpoint the teacher-forced
        # path produces (context * scale), so sampled-action ranking metrics mirror the endpoint
        # metrics for this fake and stay exactly assertable.
        context = self.transition_encoder(current_images, future_images, state)
        endpoint = context * self.flow_head.scale
        batch_size = current_images.shape[0]
        return endpoint.reshape(batch_size, 1, 1).expand(batch_size, self.config.action_horizon, self.config.action_dim)


class _FakeWanLatentFlowIdm(torch.nn.Module):
    uses_flow_matching = True

    def __init__(self) -> None:
        super().__init__()
        self.config = ModelConfig(
            num_views=1,
            image_size=1,
            state_dim=1,
            action_dim=1,
            action_horizon=1,
            num_future_frames=1,
            idm_arch="flow_transformer",
            idm_visual_encoder="wan_vae",
            wan_vae_use_cached_latents=True,
        )
        self.transition_encoder = _WanLatentScalarTransitionEncoder()
        self.flow_head = _EndpointFlowHead()

    def forward(
        self,
        current_images,
        future_images,
        state,
        task_id,
        *,
        wan_vae_latents=None,
        sample_noise=None,
        target_action=None,
        action_mask=None,
        mode="sample",
    ):
        del future_images, state, task_id, wan_vae_latents, sample_noise, action_mask
        if mode == "loss":
            if target_action is None:
                raise ValueError("target_action is required")
            return {"loss": target_action.new_tensor(3.0) * self.flow_head.scale}
        return torch.zeros((current_images.shape[0], 1, 1), device=current_images.device)


class _FakeCrossAttentionWanLatentFlowIdm(torch.nn.Module):
    uses_flow_matching = True

    def __init__(self) -> None:
        super().__init__()
        self.config = ModelConfig(
            num_views=1,
            image_size=1,
            state_dim=1,
            action_dim=1,
            action_horizon=1,
            num_future_frames=1,
            idm_arch="flow_transformer",
            idm_visual_encoder="wan_vae",
            wan_vae_use_cached_latents=True,
            idm_flow_visual_token_conditioning=True,
            idm_flow_visual_token_conditioning_mode="cross_attention",
        )
        self.transition_encoder = _WanLatentTokenTransitionEncoder()
        self.flow_head = _VisualTokenRequiredEndpointFlowHead()

    def forward(
        self,
        current_images,
        future_images,
        state,
        task_id,
        *,
        wan_vae_latents=None,
        sample_noise=None,
        target_action=None,
        action_mask=None,
        mode="sample",
    ):
        del future_images, state, task_id, wan_vae_latents, sample_noise, action_mask
        if mode == "loss":
            if target_action is None:
                raise ValueError("target_action is required")
            return {"loss": target_action.new_tensor(3.0) * self.flow_head.scale}
        return torch.zeros((current_images.shape[0], 1, 1), device=current_images.device)


def _future_contrastive_batch(future_values: torch.Tensor, target_values: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size = int(future_values.shape[0])
    return {
        "current_images": torch.full((batch_size, 1, 1, 1, 1), 2.0),
        "future_images": future_values.reshape(batch_size, 1, 1, 1, 1, 1),
        "state": torch.zeros(batch_size, 1),
        "task_id": torch.zeros(batch_size, dtype=torch.long),
        "action_chunk": target_values.reshape(batch_size, 1, 1),
        "action_mask": torch.ones(batch_size, 1),
    }


def _future_contrastive_latent_batch(
    future_values: torch.Tensor,
    target_values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch = _future_contrastive_batch(future_values, target_values)
    latents = torch.zeros((future_values.shape[0], 1, 2, 1, 1), dtype=future_values.dtype)
    latents[:, 0, 0, 0, 0] = torch.arange(future_values.shape[0], dtype=future_values.dtype) + 10.0
    latents[:, 0, 1, 0, 0] = future_values
    return {**batch, "wan_vae_latents": latents}


def _future_contrastive_samples(
    future_values: torch.Tensor, target_values: torch.Tensor
) -> list[dict[str, torch.Tensor]]:
    batch = _future_contrastive_batch(future_values, target_values)
    return [{key: value[index] for key, value in batch.items()} for index in range(int(future_values.shape[0]))]


def test_synthetic_training_writes_artifacts(tmp_path) -> None:
    config = TrainConfig(
        dataset=DatasetConfig(source="synthetic", image_size=32, synthetic_samples=16, action_horizon=8),
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=4,
        device="cpu",
        seed=11,
    )

    metrics = run_training(config)

    assert "final" in metrics
    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "prediction_grid.png").exists()

    written = json.loads((tmp_path / "metrics.json").read_text())
    assert written["final"]["wm_mse"] >= 0.0
    assert written["final"]["idm_mse"] >= 0.0
    assert written["final"]["idm_generated_mse"] >= 0.0


def test_train_idm_main_trains_gt_only_synthetic_smoke(tmp_path) -> None:
    train_idm_main(
        TrainIdmArgs(
            dataset_source="synthetic",
            image_keys=("corner4.image",),
            output_dir=str(tmp_path),
            epochs=1,
            batch_size=4,
            image_size=32,
            synthetic_samples=16,
            action_horizon=8,
            latent_dim=32,
            idm_transformer_layers=1,
            idm_transformer_heads=4,
            idm_transformer_ff_dim=64,
            idm_flow_sampling_steps=2,
            device="cpu",
            seed=11,
        )
    )

    assert (tmp_path / "idm_checkpoint.pt").exists()
    assert (tmp_path / "best_idm_checkpoint.pt").exists()
    assert (tmp_path / "metrics.json").exists()

    written = json.loads((tmp_path / "metrics.json").read_text())
    assert written["training_target"] == "idm"
    assert written["cached_future_dir"] is None
    assert not written["include_gt_futures_with_cache"]
    assert written["final"]["idm_mse"] >= 0.0
    assert written["final"]["idm_smooth_l1"] >= 0.0
    assert written["best"]["idm_mse"] >= 0.0
    assert written["best_checkpoint"] == str(tmp_path / "best_idm_checkpoint.pt")


def test_run_idm_training_writes_best_checkpoint_during_progress(tmp_path, monkeypatch) -> None:
    dataset = _future_contrastive_samples(
        torch.tensor([0.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
    )
    train_epochs_started = 0
    progress_checkpoint_metrics = None

    def fake_train_idm_one_epoch(*args, **kwargs):
        del args, kwargs
        nonlocal train_epochs_started, progress_checkpoint_metrics
        train_epochs_started += 1
        if train_epochs_started == 2:
            progress_checkpoint = tmp_path / "best_idm_checkpoint.pt"
            assert progress_checkpoint.exists()
            progress_checkpoint_metrics = torch.load(
                progress_checkpoint,
                map_location="cpu",
                weights_only=False,
            )["metrics"]
        return {"loss": 0.0, "idm_loss": 0.0, "action_smoothness_loss": 0.0}

    eval_mses = iter([2.0, 1.0])

    def fake_evaluate_idm(*args, **kwargs):
        del args, kwargs
        return {"idm_mse": next(eval_mses), "idm_smooth_l1": 0.0}

    monkeypatch.setattr(train_lib, "create_dataset_with_optional_cache", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(train_lib, "create_idm_model", lambda config, device: _FakeFlowIdm().to(device))
    monkeypatch.setattr(train_lib, "train_idm_one_epoch", fake_train_idm_one_epoch)
    monkeypatch.setattr(train_lib, "evaluate_idm", fake_evaluate_idm)
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=1,
            synthetic_samples=len(dataset),
            action_horizon=1,
            num_future_frames=1,
        ),
        model=ModelConfig(
            num_views=1,
            image_size=1,
            state_dim=1,
            action_dim=1,
            action_horizon=1,
            num_future_frames=1,
            idm_arch="flow_transformer",
        ),
        output_dir=str(tmp_path),
        epochs=2,
        batch_size=2,
        eval_fraction=0.25,
        split_gap=0,
        normalize_actions=False,
        device="cpu",
    )

    run_idm_training(config)

    assert progress_checkpoint_metrics is not None
    assert progress_checkpoint_metrics["final"]["epoch"] == 1
    assert progress_checkpoint_metrics["best"]["epoch"] == 1


def test_run_idm_training_keeps_final_metrics_compatible_and_streams_epochs(tmp_path, monkeypatch) -> None:
    dataset = _future_contrastive_samples(
        torch.tensor([0.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
    )

    monkeypatch.setattr(train_lib, "create_dataset_with_optional_cache", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(train_lib, "create_idm_model", lambda config, device: _FakeFlowIdm().to(device))
    monkeypatch.setattr(
        train_lib,
        "train_idm_one_epoch",
        lambda *args, **kwargs: {"loss": 0.0, "idm_loss": 0.0, "action_smoothness_loss": 0.0},
    )
    eval_mses = iter([2.0, 1.0])
    monkeypatch.setattr(
        train_lib,
        "evaluate_idm",
        lambda *args, **kwargs: {"idm_mse": next(eval_mses), "idm_smooth_l1": 0.0},
    )
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=1,
            synthetic_samples=len(dataset),
            action_horizon=1,
            num_future_frames=1,
        ),
        model=ModelConfig(
            num_views=1,
            image_size=1,
            state_dim=1,
            action_dim=1,
            action_horizon=1,
            num_future_frames=1,
            idm_arch="flow_transformer",
        ),
        output_dir=str(tmp_path),
        epochs=2,
        batch_size=2,
        eval_fraction=0.25,
        split_gap=0,
        normalize_actions=False,
        device="cpu",
    )

    metrics = run_idm_training(config)

    written = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    stream_rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert written["history"] == metrics["history"]
    assert written["final"] == metrics["final"]
    assert written["best"] == metrics["best"]
    assert written["final_checkpoint"] == str(tmp_path / "idm_checkpoint.pt")
    assert written["best_checkpoint"] == str(tmp_path / "best_idm_checkpoint.pt")
    assert written["best"]["epoch"] == 2
    assert stream_rows == written["history"]

    final_checkpoint = torch.load(tmp_path / "idm_checkpoint.pt", map_location="cpu", weights_only=False)
    best_checkpoint = torch.load(tmp_path / "best_idm_checkpoint.pt", map_location="cpu", weights_only=False)
    assert final_checkpoint["metrics"]["final_checkpoint"] == written["final_checkpoint"]
    assert best_checkpoint["metrics"]["best_checkpoint"] == written["best_checkpoint"]


def test_train_idm_main_forwards_flow_train_time_range(tmp_path, monkeypatch) -> None:
    captured_configs = []
    captured_kwargs = []

    def fake_run_idm_training(config, **kwargs):
        captured_configs.append(config)
        captured_kwargs.append(kwargs)

    monkeypatch.setattr("train_idm.run_idm_training", fake_run_idm_training)

    train_idm_main(
        TrainIdmArgs(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            num_future_frames=4,
            idm_visual_encoder="wan_vae",
            idm_flow_visual_token_conditioning=True,
            idm_flow_visual_token_conditioning_mode="cross_attention",
            idm_flow_visual_token_scope="future_only",
            idm_flow_visual_token_representation="future_delta",
            idm_flow_endpoint_consistency_loss_weight=0.2,
            idm_flow_zero_start_endpoint_loss_weight=0.3,
            idm_flow_sampled_action_loss_weight=0.35,
            idm_flow_sample_noise_scale=0.0,
            idm_flow_train_time_min=0.25,
            idm_flow_train_time_max=0.75,
            idm_context_action_warmup_epochs=3,
            idm_same_task_future_delta_weight=0.4,
            idm_same_task_future_delta_time_value=0.6,
            idm_same_task_future_delta_max_state_distance=0.25,
            idm_same_task_future_delta_min_action_delta_mse=0.01,
            idm_future_ranking_same_task_negative=True,
            idm_same_task_batching=True,
        )
    )

    assert len(captured_configs) == 1
    assert captured_configs[0].model.idm_visual_encoder == "wan_vae"
    assert captured_configs[0].model.idm_flow_visual_token_conditioning is True
    assert captured_configs[0].model.idm_flow_visual_token_conditioning_mode == "cross_attention"
    assert captured_configs[0].model.idm_flow_visual_token_scope == "future_only"
    assert captured_configs[0].model.idm_flow_visual_token_representation == "future_delta"
    assert captured_configs[0].model.idm_flow_endpoint_consistency_loss_weight == pytest.approx(0.2)
    assert captured_configs[0].model.idm_flow_zero_start_endpoint_loss_weight == pytest.approx(0.3)
    assert captured_configs[0].model.idm_flow_sampled_action_loss_weight == pytest.approx(0.35)
    assert captured_configs[0].model.idm_flow_sample_noise_scale == pytest.approx(0.0)
    assert captured_configs[0].model.idm_flow_train_time_min == pytest.approx(0.25)
    assert captured_configs[0].model.idm_flow_train_time_max == pytest.approx(0.75)
    assert captured_configs[0].idm_wan_vae_latent_noise_prob == pytest.approx(0.0)
    assert captured_configs[0].idm_wan_vae_latent_noise_time_mode == "all"
    assert captured_configs[0].idm_future_ranking_weight == pytest.approx(0.0)
    assert not captured_configs[0].idm_future_ranking_repeated_current_negative
    assert captured_configs[0].idm_future_ranking_same_task_negative
    assert captured_configs[0].idm_same_task_batching
    assert captured_configs[0].idm_same_task_future_delta_weight == pytest.approx(0.4)
    assert captured_configs[0].idm_same_task_future_delta_time_value == pytest.approx(0.6)
    assert captured_configs[0].idm_same_task_future_delta_max_state_distance == pytest.approx(0.25)
    assert captured_configs[0].idm_same_task_future_delta_min_action_delta_mse == pytest.approx(0.01)
    assert captured_kwargs == [{"idm_context_action_warmup_epochs": 3}]


def test_train_idm_main_forwards_idm_history_length(tmp_path, monkeypatch) -> None:
    captured_configs = []

    def fake_run_idm_training(config, **kwargs):
        del kwargs
        captured_configs.append(config)

    monkeypatch.setattr("train_idm.run_idm_training", fake_run_idm_training)

    train_idm_main(
        TrainIdmArgs(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            idm_arch="flow_transformer",
            idm_history_length=3,
        )
    )

    assert len(captured_configs) == 1
    assert captured_configs[0].dataset.idm_history_length == 3
    assert captured_configs[0].model.idm_history_length == 3


def test_train_idm_main_forwards_idm_future_conditioning(tmp_path, monkeypatch) -> None:
    captured_configs = []

    def fake_run_idm_training(config, **kwargs):
        del kwargs
        captured_configs.append(config)

    monkeypatch.setattr("train_idm.run_idm_training", fake_run_idm_training)

    train_idm_main(
        TrainIdmArgs(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            idm_arch="flow_transformer",
            idm_future_conditioning="current_only",
        )
    )

    assert len(captured_configs) == 1
    assert captured_configs[0].model.idm_future_conditioning == "current_only"


def test_train_idm_main_forwards_wan_vae_latent_noise_options(tmp_path, monkeypatch) -> None:
    captured_configs = []

    def fake_run_idm_training(config, **kwargs):
        del kwargs
        captured_configs.append(config)

    monkeypatch.setattr("train_idm.run_idm_training", fake_run_idm_training)

    train_idm_main(
        TrainIdmArgs(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            idm_wan_vae_latent_noise_prob=0.25,
            idm_wan_vae_latent_noise_s_min=0.2,
            idm_wan_vae_latent_noise_s_max=0.8,
            idm_wan_vae_latent_noise_time_mode="future_only",
        )
    )

    assert len(captured_configs) == 1
    assert captured_configs[0].idm_wan_vae_latent_noise_prob == pytest.approx(0.25)
    assert captured_configs[0].idm_wan_vae_latent_noise_s_min == pytest.approx(0.2)
    assert captured_configs[0].idm_wan_vae_latent_noise_s_max == pytest.approx(0.8)
    assert captured_configs[0].idm_wan_vae_latent_noise_time_mode == "future_only"


def test_train_idm_main_forwards_current_conditioning_dropout(tmp_path, monkeypatch) -> None:
    captured_configs = []

    def fake_run_idm_training(config, **kwargs):
        del kwargs
        captured_configs.append(config)

    monkeypatch.setattr("train_idm.run_idm_training", fake_run_idm_training)

    train_idm_main(
        TrainIdmArgs(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            idm_current_frame_dropout=0.25,
            idm_wan_vae_current_latent_dropout=0.5,
        )
    )

    assert len(captured_configs) == 1
    assert captured_configs[0].idm_current_frame_dropout == pytest.approx(0.25)
    assert captured_configs[0].idm_wan_vae_current_latent_dropout == pytest.approx(0.5)


def test_train_idm_main_forwards_future_ranking_options(tmp_path, monkeypatch) -> None:
    captured_configs = []

    def fake_run_idm_training(config, **kwargs):
        del kwargs
        captured_configs.append(config)

    monkeypatch.setattr("train_idm.run_idm_training", fake_run_idm_training)

    train_idm_main(
        TrainIdmArgs(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            idm_future_ranking_weight=0.75,
            idm_future_ranking_start_epoch=4,
            idm_future_ranking_ramp_epochs=3,
            idm_future_ranking_temperature=0.25,
            idm_future_ranking_noise_std=0.5,
            idm_future_ranking_repeated_current_negative=True,
            idm_future_ranking_shuffled_future_negative=True,
            idm_future_ranking_noisy_future_negative=True,
            idm_future_ranking_zero_future_negative=True,
            idm_future_ranking_same_task_negative=True,
            idm_future_ranking_score_mode="sampled_action",
        )
    )

    assert len(captured_configs) == 1
    assert captured_configs[0].idm_future_ranking_weight == pytest.approx(0.75)
    assert captured_configs[0].idm_future_ranking_start_epoch == 4
    assert captured_configs[0].idm_future_ranking_ramp_epochs == 3
    assert captured_configs[0].idm_future_ranking_temperature == pytest.approx(0.25)
    assert captured_configs[0].idm_future_ranking_noise_std == pytest.approx(0.5)
    assert captured_configs[0].idm_future_ranking_repeated_current_negative
    assert captured_configs[0].idm_future_ranking_shuffled_future_negative
    assert captured_configs[0].idm_future_ranking_noisy_future_negative
    assert captured_configs[0].idm_future_ranking_zero_future_negative
    assert captured_configs[0].idm_future_ranking_same_task_negative
    assert captured_configs[0].idm_future_ranking_score_mode == "sampled_action"


def test_train_idm_main_forwards_future_usage_eval_options(tmp_path, monkeypatch) -> None:
    captured_configs = []

    def fake_run_idm_training(config, **kwargs):
        del kwargs
        captured_configs.append(config)

    monkeypatch.setattr("train_idm.run_idm_training", fake_run_idm_training)

    train_idm_main(
        TrainIdmArgs(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            idm_future_usage_eval=True,
            idm_future_usage_rank_accuracy_min=0.8,
            idm_future_usage_gap_min=0.05,
            idm_future_usage_degradation_min=0.02,
            idm_future_usage_output_delta_mse_min=0.03,
            idm_future_usage_score_mode="sampled_action",
        )
    )

    assert len(captured_configs) == 1
    config = captured_configs[0]
    assert config.idm_future_usage_eval is True
    assert config.idm_future_usage_rank_accuracy_min == pytest.approx(0.8)
    assert config.idm_future_usage_gap_min == pytest.approx(0.05)
    assert config.idm_future_usage_degradation_min == pytest.approx(0.02)
    assert config.idm_future_usage_output_delta_mse_min == pytest.approx(0.03)
    assert config.idm_future_usage_score_mode == "sampled_action"


def test_idm_state_normalizer_uses_train_split_only_and_reloads(tmp_path, monkeypatch) -> None:
    dataset = _ControlledStateDataset()
    monkeypatch.setattr(train_lib, "create_dataset_with_optional_cache", lambda *args, **kwargs: dataset)
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=16,
            synthetic_samples=len(dataset),
            num_future_frames=1,
            action_horizon=2,
        ),
        model=ModelConfig(num_views=1, image_size=16, latent_dim=32, action_horizon=2),
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=4,
        eval_fraction=0.25,
        split_gap=0,
        normalize_actions=False,
        device="cpu",
        seed=11,
    )

    metrics = run_idm_training(config)

    train_states = torch.stack(dataset.states[:6])
    expected_mean = train_states.mean(dim=0)
    expected_std = train_states.std(dim=0, unbiased=False).clamp_min(1e-4)
    full_mean = torch.stack(dataset.states).mean(dim=0)
    metrics_mean = torch.tensor(metrics["state_normalizer"]["mean"])
    metrics_std = torch.tensor(metrics["state_normalizer"]["std"])
    assert torch.allclose(metrics_mean, expected_mean)
    assert torch.allclose(metrics_std, expected_std)
    assert not torch.allclose(metrics_mean, full_mean)

    checkpoint = torch.load(tmp_path / "idm_checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["state_normalizer"] == metrics["state_normalizer"]
    assert checkpoint["metrics"]["state_normalizer"] == metrics["state_normalizer"]

    idm, _ = load_idm_checkpoint(tmp_path / "idm_checkpoint.pt", torch.device("cpu"))
    loaded = get_state_normalizer(idm, torch.device("cpu"))
    assert loaded is not None
    assert state_normalizer_applies_in_forward(idm)
    assert torch.allclose(loaded.mean, expected_mean)
    assert torch.allclose(loaded.std, expected_std)


def test_idm_loss_and_eval_pass_normalized_state_to_model() -> None:
    class RecordingIdm(torch.nn.Module):
        uses_flow_matching = False

        def __init__(self) -> None:
            super().__init__()
            self.seen_states: list[torch.Tensor] = []
            self.seen_task_ids: list[torch.Tensor] = []

        def forward(self, current_images, future_images, state, task_id, *, sample_noise=None):
            del future_images, sample_noise
            self.seen_states.append(state.detach().cpu())
            self.seen_task_ids.append(task_id.detach().cpu())
            return torch.zeros((current_images.shape[0], 2, 1), device=current_images.device)

    idm = RecordingIdm()
    normalizer = StateNormalizer(mean=torch.tensor([10.0, 20.0]), std=torch.tensor([2.0, 5.0]))
    batch = {
        "current_images": torch.zeros(2, 1, 3, 8, 8),
        "future_images": torch.zeros(2, 1, 1, 3, 8, 8),
        "state": torch.tensor([[12.0, 10.0], [8.0, 25.0]]),
        "task_id": torch.tensor([7, 8], dtype=torch.long),
        "action_chunk": torch.ones(2, 2, 1),
        "action_mask": torch.ones(2, 2),
    }
    expected_state = torch.tensor([[1.0, -2.0], [-1.0, 1.0]])

    compute_idm_losses(idm, batch, state_normalizer=normalizer)
    assert torch.allclose(idm.seen_states[-1], expected_state)
    assert torch.equal(idm.seen_task_ids[-1], batch["task_id"])

    samples = [
        {
            "current_images": batch["current_images"][index],
            "future_images": batch["future_images"][index],
            "state": batch["state"][index],
            "task_id": batch["task_id"][index],
            "action_chunk": batch["action_chunk"][index],
            "action_mask": batch["action_mask"][index],
        }
        for index in range(2)
    ]
    evaluate_idm(idm, DataLoader(samples, batch_size=2), torch.device("cpu"), state_normalizer=normalizer)

    assert torch.allclose(idm.seen_states[-1], expected_state)
    assert torch.equal(idm.seen_task_ids[-1], batch["task_id"])


def test_future_contrastive_default_zero_preserves_flow_loss_path() -> None:
    idm = _FakeFlowIdm(forbid_contrastive_path=True, forbid_context_action_path=True)
    batch = _future_contrastive_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))

    losses = compute_idm_losses(idm, batch)

    assert torch.equal(losses["idm_loss"], torch.tensor(3.0))
    assert torch.equal(losses["idm_endpoint_consistency_loss"], torch.tensor(0.0))
    assert torch.equal(losses["idm_zero_start_endpoint_loss"], torch.tensor(0.0))
    assert torch.equal(losses["idm_sampled_action_loss"], torch.tensor(0.0))
    assert torch.equal(losses["idm_context_action_loss"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_contrastive_loss"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_contrastive_real_endpoint_mse"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_contrastive_corrupted_endpoint_mse"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_contrastive_singleton_fraction"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_loss"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_real_endpoint_mse"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_best_negative_endpoint_mse"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_mean_negative_endpoint_mse"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_rank_accuracy"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_negative_count"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_singleton_fraction"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_same_task_valid_fraction"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_same_task_no_donor_fraction"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_same_task_different_episode_fraction"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_same_task_wan_latent_too_short_fraction"], torch.tensor(0.0))


def test_future_usage_eval_config_defaults_and_validation() -> None:
    config = TrainConfig()

    assert config.idm_future_usage_eval is False
    assert config.idm_future_usage_rank_accuracy_min == pytest.approx(0.55)
    assert config.idm_future_usage_gap_min == pytest.approx(0.0)
    assert config.idm_future_usage_degradation_min == pytest.approx(1e-4)
    assert config.idm_future_usage_output_delta_mse_min == pytest.approx(1e-4)

    with pytest.raises(ValueError, match="idm_future_usage_rank_accuracy_min"):
        TrainConfig(idm_future_usage_rank_accuracy_min=1.1)
    with pytest.raises(ValueError, match="idm_future_usage_gap_min"):
        TrainConfig(idm_future_usage_gap_min=-0.1)
    with pytest.raises(ValueError, match="idm_future_usage_degradation_min"):
        TrainConfig(idm_future_usage_degradation_min=-0.1)
    with pytest.raises(ValueError, match="idm_future_usage_output_delta_mse_min"):
        TrainConfig(idm_future_usage_output_delta_mse_min=-0.1)
    with pytest.raises(ValueError, match="flow_transformer"):
        TrainConfig(idm_future_usage_eval=True)


def test_future_usage_eval_future_blind_flow_model_fails_gate() -> None:
    idm = _FakeFlowIdm()
    idm.transition_encoder = _CurrentScalarTransitionEncoder()
    samples = _future_contrastive_samples(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 4.0]))

    regular_metrics = evaluate_idm(idm, DataLoader(samples, batch_size=2), torch.device("cpu"))
    metrics = evaluate_idm_future_usage(idm, DataLoader(samples, batch_size=2), torch.device("cpu"))

    assert regular_metrics["idm_mse"] >= 0.0
    assert metrics["future_usage_gate_pass"] is False
    assert metrics["future_usage_rank_accuracy"] == pytest.approx(0.0)
    assert metrics["future_usage_current_repeated_degradation"] == pytest.approx(0.0)
    assert metrics["future_usage_current_repeated_output_delta_mse"] == pytest.approx(0.0)
    assert "current_repeated_output_delta_mse" in metrics["future_usage_gate_reasons"]


def test_future_usage_eval_future_aware_flow_model_passes_gate() -> None:
    idm = _FakeFlowIdm()
    samples = _future_contrastive_samples(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 4.0]))

    metrics = evaluate_idm_future_usage(idm, DataLoader(samples, batch_size=2), torch.device("cpu"))

    assert metrics["future_usage_gate_pass"] is True
    assert metrics["future_usage_current_repeated_degradation"] > 0.0
    assert metrics["future_usage_current_repeated_output_delta_mse"] > 0.0
    assert metrics["future_usage_real_vs_best_negative_gap"] > 0.0
    assert metrics["future_usage_gap_norm"] > 0.0
    assert metrics["future_usage_rank_accuracy"] == pytest.approx(1.0)
    assert metrics["future_usage_num_ranked_samples"] == 2
    assert metrics["future_usage_gate_reasons"] == ""
    assert metrics["future_usage_score_mode"] == "teacher_forced_endpoint"


def test_future_usage_eval_sampled_action_mode_matches_mirrored_endpoint() -> None:
    # The fake sampler reproduces the teacher-forced endpoint, so sampled-action future-usage
    # scoring must yield the same gate decision and metrics, only tagged with the new score mode.
    idm = _FakeFlowIdm()
    samples = _future_contrastive_samples(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 4.0]))

    teacher = evaluate_idm_future_usage(idm, DataLoader(samples, batch_size=2), torch.device("cpu"))
    sampled = evaluate_idm_future_usage(
        idm,
        DataLoader(samples, batch_size=2),
        torch.device("cpu"),
        score_mode="sampled_action",
    )

    assert teacher["future_usage_score_mode"] == "teacher_forced_endpoint"
    assert sampled["future_usage_score_mode"] == "sampled_action"
    assert sampled["future_usage_gate_pass"] is True
    assert sampled["future_usage_gate_reasons"] == ""
    assert sampled["future_usage_num_ranked_samples"] == teacher["future_usage_num_ranked_samples"]
    for key in (
        "future_usage_real_endpoint_mse",
        "future_usage_current_repeated_endpoint_mse",
        "future_usage_current_repeated_degradation",
        "future_usage_current_repeated_output_delta_mse",
        "future_usage_rank_accuracy",
        "future_usage_real_vs_best_negative_gap",
        "future_usage_gap_norm",
    ):
        assert sampled[key] == pytest.approx(teacher[key])


def test_future_usage_eval_sampled_action_mode_flags_future_blind() -> None:
    idm = _FakeFlowIdm()
    idm.transition_encoder = _CurrentScalarTransitionEncoder()
    samples = _future_contrastive_samples(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 4.0]))

    metrics = evaluate_idm_future_usage(
        idm,
        DataLoader(samples, batch_size=2),
        torch.device("cpu"),
        score_mode="sampled_action",
    )

    assert metrics["future_usage_score_mode"] == "sampled_action"
    assert metrics["future_usage_gate_pass"] is False
    assert metrics["future_usage_rank_accuracy"] == pytest.approx(0.0)
    assert metrics["future_usage_current_repeated_output_delta_mse"] == pytest.approx(0.0)
    assert "current_repeated_output_delta_mse" in metrics["future_usage_gate_reasons"]


def test_future_usage_eval_rejects_invalid_score_mode() -> None:
    idm = _FakeFlowIdm()
    samples = _future_contrastive_samples(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 4.0]))

    with pytest.raises(ValueError, match="score_mode"):
        evaluate_idm_future_usage(
            idm,
            DataLoader(samples, batch_size=2),
            torch.device("cpu"),
            score_mode="bogus",
        )


def test_future_usage_best_selection_prefers_passing_epoch_over_lower_mse_failing_epoch() -> None:
    failing_low_mse = {"epoch": 1, "idm_mse": 0.1, "future_usage_gate_pass": False}
    passing_higher_mse = {"epoch": 2, "idm_mse": 0.2, "future_usage_gate_pass": True}
    failing_even_lower_mse = {"epoch": 3, "idm_mse": 0.01, "future_usage_gate_pass": False}
    passing_lower_mse = {"epoch": 4, "idm_mse": 0.15, "future_usage_gate_pass": True}

    assert is_better_idm_checkpoint_row(
        passing_higher_mse,
        failing_low_mse,
        future_usage_eval=True,
    )
    assert not is_better_idm_checkpoint_row(
        failing_even_lower_mse,
        passing_higher_mse,
        future_usage_eval=True,
    )
    assert is_better_idm_checkpoint_row(
        passing_lower_mse,
        passing_higher_mse,
        future_usage_eval=True,
    )
    assert is_better_idm_checkpoint_row(
        failing_even_lower_mse,
        failing_low_mse,
        future_usage_eval=False,
    )


def test_future_contrastive_positive_weight_adds_nonzero_flow_ranking_loss() -> None:
    idm = _FakeFlowIdm()
    batch = _future_contrastive_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))

    losses = compute_idm_losses(
        idm,
        batch,
        future_contrastive_weight=1.0,
        future_contrastive_margin=1.5,
    )

    assert torch.allclose(losses["idm_future_contrastive_real_endpoint_mse"], torch.tensor(0.0))
    assert torch.allclose(losses["idm_future_contrastive_corrupted_endpoint_mse"], torch.tensor(1.0))
    assert torch.allclose(losses["idm_future_contrastive_loss"], torch.tensor(0.5))


def test_future_contrastive_wan_vae_latents_add_nonzero_flow_ranking_loss() -> None:
    idm = _FakeWanLatentFlowIdm()
    batch = _future_contrastive_latent_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))

    losses = compute_idm_losses(
        idm,
        batch,
        future_contrastive_weight=1.0,
        future_contrastive_margin=1.5,
    )

    assert torch.allclose(losses["idm_future_contrastive_real_endpoint_mse"], torch.tensor(0.0))
    assert torch.allclose(losses["idm_future_contrastive_corrupted_endpoint_mse"], torch.tensor(1.0))
    assert torch.allclose(losses["idm_future_contrastive_loss"], torch.tensor(0.5))
    assert torch.equal(losses["idm_future_contrastive_singleton_fraction"], torch.tensor(0.0))


def test_future_contrastive_cross_attention_wan_latents_passes_visual_tokens() -> None:
    idm = _FakeCrossAttentionWanLatentFlowIdm()
    batch = _future_contrastive_latent_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))

    losses = compute_idm_losses(
        idm,
        batch,
        future_contrastive_weight=1.0,
        future_contrastive_margin=1.5,
    )

    visual_token_values = idm.flow_head.visual_token_values
    assert torch.allclose(losses["idm_future_contrastive_loss"], torch.tensor(0.5))
    assert len(visual_token_values) == 2
    assert torch.equal(visual_token_values[0].squeeze(1), torch.tensor([0.0, 1.0]))
    assert torch.equal(visual_token_values[1].squeeze(1), torch.tensor([1.0, 0.0]))


def test_train_config_future_ranking_score_mode_defaults_to_teacher_forced_endpoint() -> None:
    assert TrainConfig().idm_future_ranking_score_mode == "teacher_forced_endpoint"


def test_train_config_accepts_sampled_action_future_ranking_score_mode() -> None:
    config = TrainConfig(idm_future_ranking_score_mode="sampled_action")
    assert config.idm_future_ranking_score_mode == "sampled_action"


def test_train_config_rejects_invalid_future_ranking_score_mode() -> None:
    with pytest.raises(ValueError, match="idm_future_ranking_score_mode"):
        TrainConfig(idm_future_ranking_score_mode="bogus")


def test_train_config_future_usage_score_mode_defaults_to_teacher_forced_endpoint() -> None:
    assert TrainConfig().idm_future_usage_score_mode == "teacher_forced_endpoint"


def test_train_config_accepts_sampled_action_future_usage_score_mode() -> None:
    config = TrainConfig(idm_future_usage_score_mode="sampled_action")
    assert config.idm_future_usage_score_mode == "sampled_action"


def test_train_config_rejects_invalid_future_usage_score_mode() -> None:
    with pytest.raises(ValueError, match="idm_future_usage_score_mode"):
        TrainConfig(idm_future_usage_score_mode="bogus")


def test_future_ranking_image_negatives_add_listwise_loss_and_metrics() -> None:
    idm = _FakeFlowIdm()
    batch = _future_contrastive_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))

    losses = compute_idm_losses(
        idm,
        batch,
        future_ranking_weight=1.0,
        future_ranking_temperature=1.0,
        future_ranking_repeated_current_negative=True,
        future_ranking_shuffled_future_negative=True,
        future_ranking_zero_future_negative=True,
    )

    assert torch.allclose(losses["idm_future_ranking_real_endpoint_mse"], torch.tensor(0.0))
    assert torch.allclose(losses["idm_future_ranking_best_negative_endpoint_mse"], torch.tensor(0.5))
    assert torch.allclose(losses["idm_future_ranking_mean_negative_endpoint_mse"], torch.tensor(4.0 / 3.0))
    assert torch.allclose(losses["idm_future_ranking_rank_accuracy"], torch.tensor(0.5))
    assert torch.equal(losses["idm_future_ranking_negative_count"], torch.tensor(3.0))
    assert torch.isfinite(losses["idm_future_ranking_loss"])
    # Default teacher_forced_endpoint mode leaves the sampled-action metrics at zero.
    assert torch.equal(losses["idm_future_ranking_real_sampled_action_mse"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_best_negative_sampled_action_mse"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_mean_negative_sampled_action_mse"], torch.tensor(0.0))


def test_future_ranking_sampled_action_mode_scores_sampler_and_zeroes_endpoint_metrics() -> None:
    idm = _FakeFlowIdm()
    batch = _future_contrastive_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))

    losses = compute_idm_losses(
        idm,
        batch,
        future_ranking_weight=1.0,
        future_ranking_temperature=1.0,
        future_ranking_repeated_current_negative=True,
        future_ranking_shuffled_future_negative=True,
        future_ranking_zero_future_negative=True,
        future_ranking_score_mode="sampled_action",
    )

    # The fake sampler reproduces the teacher-forced endpoint, so the sampled-action metrics match
    # the endpoint values asserted in the teacher-mode test, while the endpoint metrics are zeroed.
    assert torch.allclose(losses["idm_future_ranking_real_sampled_action_mse"], torch.tensor(0.0))
    assert torch.allclose(losses["idm_future_ranking_best_negative_sampled_action_mse"], torch.tensor(0.5))
    assert torch.allclose(losses["idm_future_ranking_mean_negative_sampled_action_mse"], torch.tensor(4.0 / 3.0))
    assert torch.equal(losses["idm_future_ranking_real_endpoint_mse"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_best_negative_endpoint_mse"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_mean_negative_endpoint_mse"], torch.tensor(0.0))
    # The ranking/CE/rank logic is unchanged across score modes.
    assert torch.allclose(losses["idm_future_ranking_rank_accuracy"], torch.tensor(0.5))
    assert torch.equal(losses["idm_future_ranking_negative_count"], torch.tensor(3.0))
    assert torch.isfinite(losses["idm_future_ranking_loss"])


def test_future_ranking_rejects_invalid_score_mode() -> None:
    idm = _FakeFlowIdm()
    batch = _future_contrastive_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))

    with pytest.raises(ValueError, match="future_ranking_score_mode"):
        compute_idm_losses(
            idm,
            batch,
            future_ranking_weight=1.0,
            future_ranking_temperature=1.0,
            future_ranking_repeated_current_negative=True,
            future_ranking_score_mode="bogus",
        )


def test_future_ranking_same_task_negative_uses_anchor_target_without_donor_label() -> None:
    idm = _FakeFlowIdm()
    batch = {
        **_future_contrastive_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])),
        "task_index": torch.zeros(2, dtype=torch.long),
        "dataset_index": torch.tensor([10, 11]),
        "episode_index": torch.tensor([0, 1]),
    }

    losses = compute_idm_losses(
        idm,
        batch,
        future_ranking_weight=1.0,
        future_ranking_temperature=1.0,
        future_ranking_same_task_negative=True,
    )

    expected_loss = torch.log1p(torch.exp(torch.tensor(-1.0)))
    assert torch.allclose(losses["idm_future_ranking_loss"], expected_loss)
    assert torch.allclose(losses["idm_future_ranking_real_endpoint_mse"], torch.tensor(0.0))
    assert torch.allclose(losses["idm_future_ranking_best_negative_endpoint_mse"], torch.tensor(1.0))
    assert torch.allclose(losses["idm_future_ranking_mean_negative_endpoint_mse"], torch.tensor(1.0))
    assert torch.equal(losses["idm_future_ranking_rank_accuracy"], torch.tensor(1.0))
    assert torch.equal(losses["idm_future_ranking_negative_count"], torch.tensor(1.0))
    assert torch.equal(losses["idm_future_ranking_same_task_valid_fraction"], torch.tensor(1.0))
    assert torch.equal(losses["idm_future_ranking_same_task_no_donor_fraction"], torch.tensor(0.0))


def test_future_ranking_same_task_negative_no_donor_skips_cleanly() -> None:
    idm = _FakeFlowIdm()
    batch = {
        **_future_contrastive_batch(torch.tensor([0.0]), torch.tensor([0.0])),
        "task_index": torch.zeros(1, dtype=torch.long),
        "dataset_index": torch.tensor([10]),
        "episode_index": torch.tensor([0]),
    }

    losses = compute_idm_losses(
        idm,
        batch,
        future_ranking_weight=1.0,
        future_ranking_temperature=1.0,
        future_ranking_same_task_negative=True,
    )

    assert torch.equal(losses["idm_future_ranking_loss"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_negative_count"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_same_task_valid_fraction"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_same_task_no_donor_fraction"], torch.tensor(1.0))


def test_future_ranking_cross_attention_wan_latents_passes_candidate_visual_tokens() -> None:
    idm = _FakeCrossAttentionWanLatentFlowIdm()
    batch = _future_contrastive_latent_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))

    losses = compute_idm_losses(
        idm,
        batch,
        future_ranking_weight=1.0,
        future_ranking_temperature=1.0,
        future_ranking_shuffled_future_negative=True,
        future_ranking_zero_future_negative=True,
    )

    visual_token_values = idm.flow_head.visual_token_values
    assert torch.equal(losses["idm_future_ranking_negative_count"], torch.tensor(2.0))
    assert torch.isfinite(losses["idm_future_ranking_loss"])
    assert len(visual_token_values) == 3
    assert torch.equal(visual_token_values[0].squeeze(1), torch.tensor([0.0, 1.0]))
    assert torch.equal(visual_token_values[1].squeeze(1), torch.tensor([1.0, 0.0]))
    assert torch.equal(visual_token_values[2].squeeze(1), torch.tensor([0.0, 0.0]))


def test_future_ranking_wan_vae_latents_preserves_current_slice_where_possible() -> None:
    idm = _FakeWanLatentFlowIdm()
    batch = _future_contrastive_latent_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))

    candidates, singleton_indicator, same_task_metrics = train_lib._future_ranking_negative_candidates(
        batch,
        repeated_current=True,
        shuffled_future=True,
        noisy_future=False,
        zero_future=True,
        same_task_future=False,
        noise_std=1.0,
        min_same_episode_frame_gap=0,
    )
    for candidate in candidates:
        candidate_latents = candidate.wan_vae_latents
        assert candidate_latents is not None
        assert torch.equal(candidate_latents[:, :, 0], batch["wan_vae_latents"][:, :, 0])
    assert torch.equal(singleton_indicator, torch.tensor(0.0))
    assert torch.equal(same_task_metrics["same_task_valid_fraction"], torch.tensor(0.0))

    losses = compute_idm_losses(
        idm,
        batch,
        future_ranking_weight=1.0,
        future_ranking_temperature=1.0,
        future_ranking_repeated_current_negative=True,
        future_ranking_shuffled_future_negative=True,
        future_ranking_zero_future_negative=True,
    )

    assert torch.allclose(losses["idm_future_ranking_real_endpoint_mse"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_negative_count"], torch.tensor(3.0))
    assert torch.isfinite(losses["idm_future_ranking_loss"])


def test_future_ranking_same_task_wan_vae_too_short_skips_candidate() -> None:
    idm = _FakeWanLatentFlowIdm()
    idm.transition_encoder = _WanLatentLastScalarTransitionEncoder()
    batch = {
        **_future_contrastive_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])),
        "task_index": torch.zeros(2, dtype=torch.long),
        "dataset_index": torch.tensor([10, 11]),
        "episode_index": torch.tensor([0, 1]),
        "wan_vae_latents": torch.zeros((2, 1, 1, 1, 1)),
    }

    losses = compute_idm_losses(
        idm,
        batch,
        future_ranking_weight=1.0,
        future_ranking_temperature=1.0,
        future_ranking_same_task_negative=True,
    )

    assert torch.equal(losses["idm_future_ranking_loss"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_negative_count"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_same_task_valid_fraction"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_same_task_no_donor_fraction"], torch.tensor(0.0))
    assert torch.equal(losses["idm_future_ranking_same_task_wan_latent_too_short_fraction"], torch.tensor(1.0))


def test_future_ranking_history_conditioned_batch_runs_with_normalizers() -> None:
    idm = InverseDynamicsModel(
        ModelConfig(
            num_views=1,
            image_size=16,
            state_dim=2,
            action_dim=1,
            action_horizon=2,
            idm_history_length=1,
            num_future_frames=1,
            task_vocab_size=4,
            latent_dim=16,
            idm_arch="flow_transformer",
            idm_transformer_layers=1,
            idm_transformer_heads=4,
            idm_transformer_patch_size=16,
            idm_transformer_dropout=0.0,
            idm_transformer_ff_dim=32,
            idm_flow_sampling_steps=2,
        )
    )
    batch = {
        "current_images": torch.zeros(2, 1, 3, 16, 16),
        "future_images": torch.ones(2, 1, 1, 3, 16, 16) * 0.25,
        "state": torch.tensor([[12.0, 20.0], [8.0, 10.0]]),
        "task_id": torch.zeros(2, dtype=torch.long),
        "action_chunk": torch.tensor([[[2.0], [4.0]], [[6.0], [8.0]]]),
        "action_mask": torch.ones(2, 2),
        "prev_state_history": torch.tensor([[[10.0, 15.0]], [[14.0, 5.0]]]),
        "prev_action_history": torch.tensor([[[3.0]], [[7.0]]]),
        "history_mask": torch.ones(2, 1),
    }

    losses = compute_idm_losses(
        idm,
        batch,
        action_normalizer=ActionNormalizer(mean=torch.tensor([5.0]), std=torch.tensor([2.0])),
        state_normalizer=StateNormalizer(mean=torch.tensor([10.0, 10.0]), std=torch.tensor([2.0, 5.0])),
        future_ranking_weight=1.0,
        future_ranking_temperature=0.5,
        future_ranking_repeated_current_negative=True,
        future_ranking_shuffled_future_negative=True,
        future_ranking_zero_future_negative=True,
    )

    assert torch.isfinite(losses["idm_future_ranking_loss"])
    assert torch.equal(losses["idm_future_ranking_negative_count"], torch.tensor(3.0))


def test_same_task_batch_sampler_groups_tasks_and_is_deterministic() -> None:
    tasks = [0, 1, 0, 2, 1, 2, 0, 3, 1]
    dataset = _TaskMetadataDataset(tasks)

    first = list(train_lib.SameTaskBatchSampler(dataset, batch_size=4, seed=123))
    second = list(train_lib.SameTaskBatchSampler(dataset, batch_size=4, seed=123))

    assert first == second
    assert sorted(index for batch in first for index in batch) == list(range(len(tasks)))
    for task in (0, 1, 2):
        batches_with_task = [batch for batch in first if any(tasks[index] == task for index in batch)]
        assert len(batches_with_task) == 1
        task_positions = [position for position, index in enumerate(batches_with_task[0]) if tasks[index] == task]
        assert task_positions == list(range(min(task_positions), max(task_positions) + 1))


def test_same_task_batch_sampler_falls_back_to_task_id() -> None:
    tasks = [3, 4, 3, 4]
    dataset = _TaskMetadataDataset(tasks, key="task_id")

    batches = list(train_lib.SameTaskBatchSampler(dataset, batch_size=4, seed=7))

    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3]
    assert any({tasks[index] for index in batch} == {3, 4} for batch in batches)


def test_same_task_batch_sampler_raises_without_task_metadata() -> None:
    with pytest.raises(ValueError, match="task_index or task_id"):
        train_lib.SameTaskBatchSampler(_NoTaskMetadataDataset(), batch_size=2, seed=7)


def test_same_task_donor_sampler_excludes_self_prefers_different_episode_and_enforces_gap() -> None:
    batch = {
        "action_chunk": torch.zeros(4, 1, 1),
        "task_id": torch.tensor([9, 9, 9, 9]),
        "task_index": torch.tensor([1, 1, 1, 2]),
        "dataset_index": torch.tensor([10, 11, 12, 13]),
        "episode_index": torch.tensor([5, 5, 6, 6]),
        "frame_index": torch.tensor([0, 100, 0, 0]),
    }

    donors = train_lib.sample_same_task_donors(batch, min_same_episode_frame_gap=50)

    assert torch.equal(donors.donor_indices, torch.tensor([2, 2, 0, 0]))
    assert torch.equal(donors.has_donor, torch.tensor([True, True, True, False]))
    assert torch.equal(donors.used_different_episode, torch.tensor([True, True, True, False]))

    same_episode_batch = {
        "action_chunk": torch.zeros(2, 1, 1),
        "task_id": torch.zeros(2, dtype=torch.long),
        "dataset_index": torch.tensor([0, 1]),
        "episode_index": torch.tensor([0, 0]),
        "frame_index": torch.tensor([0, 4]),
    }
    no_gap_donors = train_lib.sample_same_task_donors(same_episode_batch, min_same_episode_frame_gap=5)
    gap_donors = train_lib.sample_same_task_donors(same_episode_batch, min_same_episode_frame_gap=4)

    assert torch.equal(no_gap_donors.has_donor, torch.tensor([False, False]))
    assert torch.equal(gap_donors.has_donor, torch.tensor([True, True]))
    assert torch.equal(gap_donors.donor_indices, torch.tensor([1, 0]))


def test_same_task_future_delta_donor_sampler_honors_state_and_action_delta_filters() -> None:
    batch = {
        "action_chunk": torch.tensor([0.0, 10.0, 0.0, 0.1, 4.0]).reshape(5, 1, 1),
        "action_mask": torch.ones(5, 1),
        "state": torch.tensor([[0.0], [0.3], [0.0], [0.2], [0.4]]),
        "task_index": torch.tensor([1, 1, 2, 1, 1]),
        "dataset_index": torch.tensor([10, 11, 12, 13, 14]),
        "episode_index": torch.tensor([0, 0, 1, 1, 1]),
        "frame_index": torch.tensor([0, 1, 0, 0, 3]),
    }

    donors = train_lib.sample_same_task_future_delta_donors(
        batch,
        min_same_episode_frame_gap=2,
        max_state_distance=0.5,
        min_action_delta_mse=1.0,
    )

    assert donors.has_donor[0]
    assert donors.donor_indices[0] == 4
    assert donors.used_different_episode[0]
    assert donors.state_distance[0] == pytest.approx(0.4)

    state_filtered = train_lib.sample_same_task_future_delta_donors(
        batch,
        min_same_episode_frame_gap=2,
        max_state_distance=0.05,
    )
    assert torch.equal(state_filtered.has_donor, torch.tensor([False, False, False, False, False]))
    assert torch.equal(
        state_filtered.max_state_distance_filtered,
        torch.tensor([True, True, False, True, True]),
    )

    action_filtered = train_lib.sample_same_task_future_delta_donors(
        batch,
        min_same_episode_frame_gap=2,
        min_action_delta_mse=1000.0,
    )
    assert torch.equal(action_filtered.has_donor, torch.tensor([False, False, False, False, False]))
    assert torch.equal(
        action_filtered.min_action_delta_filtered,
        torch.tensor([True, True, False, True, True]),
    )


def test_same_task_future_delta_singleton_returns_zero_and_no_donor_metric() -> None:
    idm = _FakeFlowIdm()
    batch = _future_contrastive_batch(torch.tensor([0.0]), torch.tensor([0.0]))

    losses = compute_idm_losses(
        idm,
        batch,
        same_task_future_delta_weight=1.0,
    )

    assert torch.equal(losses["idm_same_task_future_delta_loss"], torch.tensor(0.0))
    assert torch.equal(losses["idm_same_task_future_delta_donor_fraction"], torch.tensor(0.0))
    assert torch.equal(losses["idm_same_task_future_delta_no_donor_fraction"], torch.tensor(1.0))


def test_same_task_future_delta_uses_anchor_context_donor_future_and_shared_noisy_path() -> None:
    idm = _AnchorAwareFlowIdm()
    batch = {
        "current_images": torch.tensor([5.0, 9.0]).reshape(2, 1, 1, 1, 1),
        "future_images": torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1, 1, 1),
        "state": torch.tensor([[2.0], [3.0]]),
        "task_id": torch.zeros(2, dtype=torch.long),
        "task_index": torch.zeros(2, dtype=torch.long),
        "dataset_index": torch.tensor([0, 1]),
        "episode_index": torch.tensor([0, 1]),
        "action_chunk": torch.tensor([0.0, 1.0]).reshape(2, 1, 1),
        "action_mask": torch.ones(2, 1),
        "prev_state_history": torch.tensor([4.0, 6.0]).reshape(2, 1, 1),
        "prev_action_history": torch.tensor([7.0, 8.0]).reshape(2, 1, 1),
        "history_mask": torch.ones(2, 1),
    }

    losses = compute_idm_losses(
        idm,
        batch,
        same_task_future_delta_weight=1.0,
        same_task_future_delta_time_value=0.25,
        same_task_future_delta_max_state_distance=2.0,
        same_task_future_delta_min_action_delta_mse=0.5,
    )

    assert torch.allclose(losses["idm_same_task_future_delta_loss"], torch.tensor(0.0), atol=1e-5)
    assert torch.equal(losses["idm_same_task_future_delta_donor_fraction"], torch.tensor(1.0))
    assert torch.equal(losses["idm_same_task_future_delta_effective_donor_fraction"], torch.tensor(1.0))
    assert torch.equal(losses["idm_same_task_future_delta_no_donor_fraction"], torch.tensor(0.0))
    assert torch.equal(losses["idm_same_task_future_delta_state_distance"], torch.tensor(1.0))
    assert torch.equal(losses["idm_same_task_future_delta_min_action_delta_filtered_fraction"], torch.tensor(0.0))
    assert torch.equal(losses["idm_same_task_future_delta_max_state_distance_filtered_fraction"], torch.tensor(0.0))
    assert torch.allclose(losses["idm_same_task_future_delta_prediction_delta_mse"], torch.tensor(1.0), atol=2e-3)
    assert torch.allclose(losses["idm_same_task_future_delta_delta_cosine"], torch.tensor(1.0))
    assert len(idm.flow_head.noisy_actions) == 2
    assert torch.equal(idm.flow_head.noisy_actions[0], idm.flow_head.noisy_actions[1])
    assert len(idm.flow_head.history_values) == 2
    assert torch.equal(idm.flow_head.history_values[0], torch.tensor([[11.0], [14.0]]))
    assert torch.equal(idm.flow_head.history_values[0], idm.flow_head.history_values[1])


def test_same_task_wan_latent_swap_preserves_current_slice_and_swaps_future_only() -> None:
    latents = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3, 1, 1)
    donor_indices = torch.tensor([1, 0])

    swapped, valid = train_lib.swapped_same_task_wan_vae_future_latents(latents, donor_indices)

    expected = latents.clone()
    expected[:, :, 1:] = latents[donor_indices, :, 1:]
    assert torch.equal(swapped, expected)
    assert torch.equal(swapped[:, :, 0], latents[:, :, 0])
    assert torch.equal(valid, torch.tensor([True, True]))
    assert torch.equal(latents, torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3, 1, 1))


def test_same_task_wan_latent_swap_too_short_masks_samples() -> None:
    latents = torch.arange(2, dtype=torch.float32).reshape(2, 1, 1, 1, 1)

    swapped, valid = train_lib.swapped_same_task_wan_vae_future_latents(latents, torch.tensor([1, 0]))

    assert torch.equal(swapped, latents)
    assert torch.equal(valid, torch.tensor([False, False]))


def test_same_task_future_delta_uses_swapped_wan_latents_in_compute_losses() -> None:
    idm = _FakeWanLatentFlowIdm()
    recording_encoder = _RecordingWanLatentScalarTransitionEncoder()
    idm.transition_encoder = recording_encoder
    latents = torch.tensor([0.0, 2.0, 10.0, 20.0], dtype=torch.float32).reshape(2, 1, 2, 1, 1)
    batch = {
        "current_images": torch.zeros(2, 1, 3, 1, 1),
        "future_images": torch.zeros(2, 1, 1, 3, 1, 1),
        "state": torch.zeros(2, 1),
        "task_id": torch.zeros(2, dtype=torch.long),
        "task_index": torch.zeros(2, dtype=torch.long),
        "dataset_index": torch.tensor([0, 1]),
        "episode_index": torch.tensor([0, 1]),
        "action_chunk": torch.tensor([0.0, 8.0]).reshape(2, 1, 1),
        "action_mask": torch.ones(2, 1),
        "wan_vae_latents": latents,
    }

    losses = compute_idm_losses(
        idm,
        batch,
        same_task_future_delta_weight=1.0,
        same_task_future_delta_time_value=0.5,
    )

    assert torch.allclose(losses["idm_same_task_future_delta_loss"], torch.tensor(100.0), atol=1e-6)
    assert torch.equal(losses["idm_same_task_future_delta_wan_latent_too_short_fraction"], torch.tensor(0.0))
    assert len(recording_encoder.seen_latents) == 2
    assert torch.equal(recording_encoder.seen_latents[0], latents)
    assert torch.equal(recording_encoder.seen_latents[1][:, :, 0], latents[:, :, 0])
    assert torch.equal(recording_encoder.seen_latents[1][:, :, 1:], latents[torch.tensor([1, 0]), :, 1:])


def test_context_action_positive_weight_adds_supervised_flow_context_loss() -> None:
    idm = _FakeFlowIdm()
    batch = _future_contrastive_batch(torch.tensor([0.0, 0.0]), torch.tensor([2.0, 2.0]))

    losses = compute_idm_losses(
        idm,
        batch,
        context_action_loss_weight=1.0,
    )

    assert torch.allclose(losses["idm_context_action_loss"], torch.tensor(4.0))


def test_context_action_loss_uses_normalized_action_target() -> None:
    idm = _FakeFlowIdm()
    batch = _future_contrastive_batch(torch.tensor([0.0, 0.0]), torch.tensor([14.0, 8.0]))
    normalizer = ActionNormalizer(mean=torch.tensor([10.0]), std=torch.tensor([2.0]))

    compute_idm_losses(
        idm,
        batch,
        action_normalizer=normalizer,
        context_action_loss_weight=1.0,
    )

    assert idm.seen_context_action_targets
    assert torch.allclose(
        idm.seen_context_action_targets[-1],
        torch.tensor([[[2.0]], [[-1.0]]]),
    )


def test_context_action_weight_schedule_defaults_to_constant_and_validates() -> None:
    assert context_action_loss_weight_for_epoch(2.0, 0) == pytest.approx(2.0)
    assert context_action_loss_weight_for_epoch(2.0, 5) == pytest.approx(2.0)
    assert context_action_loss_weight_for_epoch(2.0, 0, warmup_epochs=2) == pytest.approx(2.0)
    assert context_action_loss_weight_for_epoch(2.0, 1, warmup_epochs=2) == pytest.approx(2.0)
    assert context_action_loss_weight_for_epoch(2.0, 2, warmup_epochs=2) == pytest.approx(0.0)
    assert context_action_loss_weight_for_epoch(2.0, 0, warmup_epochs=0) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="idm_context_action_warmup_epochs"):
        context_action_loss_weight_for_epoch(2.0, 0, warmup_epochs=-1)
    with pytest.raises(ValueError, match="epoch_index"):
        context_action_loss_weight_for_epoch(2.0, -1)
    with pytest.raises(ValueError, match="context_action_loss_weight"):
        context_action_loss_weight_for_epoch(-1.0, 0)


def test_future_ranking_weight_schedule_defaults_to_constant_and_validates() -> None:
    assert future_ranking_weight_for_epoch(2.0, 0) == pytest.approx(2.0)
    assert future_ranking_weight_for_epoch(2.0, 5) == pytest.approx(2.0)
    assert future_ranking_weight_for_epoch(2.0, 0, start_epoch=2) == pytest.approx(0.0)
    assert future_ranking_weight_for_epoch(2.0, 1, start_epoch=2) == pytest.approx(0.0)
    assert future_ranking_weight_for_epoch(2.0, 2, start_epoch=2) == pytest.approx(2.0)
    assert future_ranking_weight_for_epoch(2.0, 0, ramp_epochs=2) == pytest.approx(1.0)
    assert future_ranking_weight_for_epoch(2.0, 1, ramp_epochs=2) == pytest.approx(2.0)
    assert future_ranking_weight_for_epoch(2.0, 0, start_epoch=1, ramp_epochs=2) == pytest.approx(0.0)
    assert future_ranking_weight_for_epoch(2.0, 1, start_epoch=1, ramp_epochs=2) == pytest.approx(1.0)
    assert future_ranking_weight_for_epoch(2.0, 2, start_epoch=1, ramp_epochs=2) == pytest.approx(2.0)

    with pytest.raises(ValueError, match="idm_future_ranking_start_epoch"):
        future_ranking_weight_for_epoch(2.0, 0, start_epoch=-1)
    with pytest.raises(ValueError, match="idm_future_ranking_ramp_epochs"):
        future_ranking_weight_for_epoch(2.0, 0, ramp_epochs=-1)
    with pytest.raises(ValueError, match="epoch_index"):
        future_ranking_weight_for_epoch(2.0, -1)
    with pytest.raises(ValueError, match="idm_future_ranking_weight"):
        future_ranking_weight_for_epoch(-1.0, 0)


def test_train_idm_context_action_weight_contributes_to_loss_metric() -> None:
    idm = _FakeFlowIdm()
    loader = DataLoader(
        _future_contrastive_samples(torch.tensor([0.0, 0.0]), torch.tensor([2.0, 2.0])),
        batch_size=2,
    )
    optimizer = torch.optim.SGD(idm.parameters(), lr=0.0)
    config = TrainConfig(
        model=ModelConfig(idm_arch="flow_transformer"),
        idm_loss_weight=0.0,
        action_smoothness_weight=0.0,
        idm_context_action_loss_weight=2.0,
    )

    metrics = train_idm_one_epoch(idm, loader, optimizer, config, torch.device("cpu"))

    assert metrics["idm_context_action_loss"] == pytest.approx(4.0)
    assert metrics["idm_context_action_loss_weight_active"] == pytest.approx(2.0)
    assert metrics["loss"] == pytest.approx(8.0)


def test_train_idm_one_epoch_reports_endpoint_consistency_loss() -> None:
    idm = _FakeFlowIdm(endpoint_consistency_loss=1.25)
    loader = DataLoader(
        _future_contrastive_samples(torch.tensor([0.0, 0.0]), torch.tensor([2.0, 2.0])),
        batch_size=2,
    )
    optimizer = torch.optim.SGD(idm.parameters(), lr=0.0)
    config = TrainConfig(
        model=ModelConfig(idm_arch="flow_transformer"),
        idm_loss_weight=0.0,
        action_smoothness_weight=0.0,
    )

    metrics = train_idm_one_epoch(idm, loader, optimizer, config, torch.device("cpu"))

    assert metrics["idm_endpoint_consistency_loss"] == pytest.approx(1.25)


def test_train_idm_one_epoch_reports_zero_start_endpoint_loss() -> None:
    idm = _FakeFlowIdm(zero_start_endpoint_loss=0.75)
    loader = DataLoader(
        _future_contrastive_samples(torch.tensor([0.0, 0.0]), torch.tensor([2.0, 2.0])),
        batch_size=2,
    )
    optimizer = torch.optim.SGD(idm.parameters(), lr=0.0)
    config = TrainConfig(
        model=ModelConfig(idm_arch="flow_transformer"),
        idm_loss_weight=0.0,
        action_smoothness_weight=0.0,
    )

    metrics = train_idm_one_epoch(idm, loader, optimizer, config, torch.device("cpu"))

    assert metrics["idm_zero_start_endpoint_loss"] == pytest.approx(0.75)


def test_train_idm_one_epoch_reports_sampled_action_loss() -> None:
    idm = _FakeFlowIdm(sampled_action_loss=0.6)
    loader = DataLoader(
        _future_contrastive_samples(torch.tensor([0.0, 0.0]), torch.tensor([2.0, 2.0])),
        batch_size=2,
    )
    optimizer = torch.optim.SGD(idm.parameters(), lr=0.0)
    config = TrainConfig(
        model=ModelConfig(idm_arch="flow_transformer"),
        idm_loss_weight=0.0,
        action_smoothness_weight=0.0,
    )

    metrics = train_idm_one_epoch(idm, loader, optimizer, config, torch.device("cpu"))

    assert metrics["idm_sampled_action_loss"] == pytest.approx(0.6)


def test_run_idm_training_applies_and_reports_context_action_warmup_schedule(tmp_path, monkeypatch) -> None:
    dataset = _future_contrastive_samples(
        torch.tensor([0.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
    )
    seen_weights = []

    def fake_train_idm_one_epoch(
        idm,
        loader,
        optimizer,
        config,
        device,
        action_normalizer=None,
        state_normalizer=None,
        *,
        context_action_loss_weight=None,
        future_ranking_weight=None,
    ):
        del idm, loader, optimizer, config, device, action_normalizer, state_normalizer
        seen_weights.append(context_action_loss_weight)
        return {
            "loss": float(context_action_loss_weight or 0.0),
            "idm_loss": 0.0,
            "action_smoothness_loss": 0.0,
            "idm_context_action_loss": 1.0,
            "idm_future_contrastive_loss": 0.0,
            "idm_future_contrastive_real_endpoint_mse": 0.0,
            "idm_future_contrastive_corrupted_endpoint_mse": 0.0,
            "idm_future_contrastive_singleton_fraction": 0.0,
            "idm_context_action_loss_weight_active": float(context_action_loss_weight or 0.0),
            "idm_future_ranking_weight_active": float(future_ranking_weight or 0.0),
        }

    monkeypatch.setattr(train_lib, "create_dataset_with_optional_cache", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(train_lib, "create_idm_model", lambda config, device: _FakeFlowIdm())
    monkeypatch.setattr(train_lib, "train_idm_one_epoch", fake_train_idm_one_epoch)
    monkeypatch.setattr(train_lib, "evaluate_idm", lambda *args, **kwargs: {"idm_mse": 1.0, "idm_smooth_l1": 0.0})
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=1,
            synthetic_samples=len(dataset),
            action_horizon=1,
            num_future_frames=1,
        ),
        model=ModelConfig(
            num_views=1,
            image_size=1,
            state_dim=1,
            action_dim=1,
            action_horizon=1,
            num_future_frames=1,
            idm_arch="flow_transformer",
        ),
        output_dir=str(tmp_path),
        epochs=3,
        batch_size=2,
        eval_fraction=0.25,
        split_gap=0,
        normalize_actions=False,
        idm_context_action_loss_weight=2.0,
        device="cpu",
    )

    metrics = run_idm_training(config, idm_context_action_warmup_epochs=2)

    assert seen_weights == pytest.approx([2.0, 2.0, 0.0])
    assert [row["idm_context_action_loss_weight_active"] for row in metrics["history"]] == pytest.approx(
        [2.0, 2.0, 0.0]
    )
    assert [row["train_idm_context_action_loss_weight_active"] for row in metrics["history"]] == pytest.approx(
        [2.0, 2.0, 0.0]
    )
    assert [row["idm_future_ranking_weight_active"] for row in metrics["history"]] == pytest.approx([0.0, 0.0, 0.0])
    assert [row["train_idm_future_ranking_weight_active"] for row in metrics["history"]] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert metrics["idm_context_action_warmup_epochs"] == 2


def test_run_idm_training_applies_and_reports_future_ranking_schedule(tmp_path, monkeypatch) -> None:
    dataset = _future_contrastive_samples(
        torch.tensor([0.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
    )
    seen_weights = []

    def fake_train_idm_one_epoch(
        idm,
        loader,
        optimizer,
        config,
        device,
        action_normalizer=None,
        state_normalizer=None,
        *,
        context_action_loss_weight=None,
        future_ranking_weight=None,
    ):
        del idm, loader, optimizer, config, device, action_normalizer, state_normalizer
        seen_weights.append(future_ranking_weight)
        return {
            "loss": float(future_ranking_weight or 0.0),
            "idm_loss": 0.0,
            "action_smoothness_loss": 0.0,
            "idm_context_action_loss": 0.0,
            "idm_future_contrastive_loss": 0.0,
            "idm_future_contrastive_real_endpoint_mse": 0.0,
            "idm_future_contrastive_corrupted_endpoint_mse": 0.0,
            "idm_future_contrastive_singleton_fraction": 0.0,
            "idm_future_ranking_loss": 1.0,
            "idm_context_action_loss_weight_active": float(context_action_loss_weight or 0.0),
            "idm_future_ranking_weight_active": float(future_ranking_weight or 0.0),
        }

    monkeypatch.setattr(train_lib, "create_dataset_with_optional_cache", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(train_lib, "create_idm_model", lambda config, device: _FakeFlowIdm())
    monkeypatch.setattr(train_lib, "train_idm_one_epoch", fake_train_idm_one_epoch)
    monkeypatch.setattr(train_lib, "evaluate_idm", lambda *args, **kwargs: {"idm_mse": 1.0, "idm_smooth_l1": 0.0})
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=1,
            synthetic_samples=len(dataset),
            action_horizon=1,
            num_future_frames=1,
        ),
        model=ModelConfig(
            num_views=1,
            image_size=1,
            state_dim=1,
            action_dim=1,
            action_horizon=1,
            num_future_frames=1,
            idm_arch="flow_transformer",
        ),
        output_dir=str(tmp_path),
        epochs=3,
        batch_size=2,
        eval_fraction=0.25,
        split_gap=0,
        normalize_actions=False,
        idm_future_ranking_weight=2.0,
        idm_future_ranking_start_epoch=1,
        idm_future_ranking_ramp_epochs=2,
        idm_future_ranking_shuffled_future_negative=True,
        device="cpu",
    )

    metrics = run_idm_training(config)

    assert seen_weights == pytest.approx([0.0, 1.0, 2.0])
    assert [row["idm_future_ranking_weight_active"] for row in metrics["history"]] == pytest.approx([0.0, 1.0, 2.0])
    assert [row["train_idm_future_ranking_weight_active"] for row in metrics["history"]] == pytest.approx(
        [0.0, 1.0, 2.0]
    )
    assert metrics["idm_future_ranking_start_epoch"] == 1
    assert metrics["idm_future_ranking_ramp_epochs"] == 2


def test_train_idm_future_contrastive_weight_contributes_to_loss_metric() -> None:
    idm = _FakeFlowIdm()
    loader = DataLoader(
        _future_contrastive_samples(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])),
        batch_size=2,
    )
    optimizer = torch.optim.SGD(idm.parameters(), lr=0.0)
    config = TrainConfig(
        model=ModelConfig(idm_arch="flow_transformer"),
        idm_loss_weight=0.0,
        action_smoothness_weight=0.0,
        idm_future_contrastive_weight=2.0,
        idm_future_contrastive_margin=1.5,
    )

    metrics = train_idm_one_epoch(idm, loader, optimizer, config, torch.device("cpu"))

    assert metrics["idm_future_contrastive_loss"] == pytest.approx(0.5)
    assert metrics["idm_future_contrastive_corrupted_endpoint_mse"] == pytest.approx(1.0)
    assert metrics["idm_future_contrastive_singleton_fraction"] == pytest.approx(0.0)
    assert metrics["loss"] == pytest.approx(1.0)


def test_train_idm_future_ranking_weight_contributes_to_loss_metric() -> None:
    idm = _FakeFlowIdm()
    loader = DataLoader(
        _future_contrastive_samples(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])),
        batch_size=2,
    )
    optimizer = torch.optim.SGD(idm.parameters(), lr=0.0)
    config = TrainConfig(
        model=ModelConfig(idm_arch="flow_transformer"),
        idm_loss_weight=0.0,
        action_smoothness_weight=0.0,
        idm_future_ranking_weight=2.0,
        idm_future_ranking_temperature=1.0,
        idm_future_ranking_shuffled_future_negative=True,
    )

    metrics = train_idm_one_epoch(idm, loader, optimizer, config, torch.device("cpu"))

    expected_loss = torch.log1p(torch.exp(torch.tensor(-1.0))).item()
    assert metrics["idm_future_ranking_loss"] == pytest.approx(expected_loss)
    assert metrics["idm_future_ranking_real_endpoint_mse"] == pytest.approx(0.0)
    assert metrics["idm_future_ranking_best_negative_endpoint_mse"] == pytest.approx(1.0)
    assert metrics["idm_future_ranking_mean_negative_endpoint_mse"] == pytest.approx(1.0)
    assert metrics["idm_future_ranking_rank_accuracy"] == pytest.approx(1.0)
    assert metrics["idm_future_ranking_negative_count"] == pytest.approx(1.0)
    assert metrics["idm_future_ranking_weight_active"] == pytest.approx(2.0)
    assert metrics["loss"] == pytest.approx(2.0 * expected_loss)
    # Teacher mode (the default) leaves the per-epoch sampled-action metrics at zero.
    assert metrics["idm_future_ranking_real_sampled_action_mse"] == pytest.approx(0.0)
    assert metrics["idm_future_ranking_best_negative_sampled_action_mse"] == pytest.approx(0.0)
    assert metrics["idm_future_ranking_mean_negative_sampled_action_mse"] == pytest.approx(0.0)


def test_train_idm_future_ranking_sampled_action_mode_reports_sampled_epoch_metrics() -> None:
    idm = _FakeFlowIdm()
    loader = DataLoader(
        _future_contrastive_samples(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])),
        batch_size=2,
    )
    optimizer = torch.optim.SGD(idm.parameters(), lr=0.0)
    config = TrainConfig(
        model=ModelConfig(idm_arch="flow_transformer"),
        idm_loss_weight=0.0,
        action_smoothness_weight=0.0,
        idm_future_ranking_weight=2.0,
        idm_future_ranking_temperature=1.0,
        idm_future_ranking_shuffled_future_negative=True,
        idm_future_ranking_score_mode="sampled_action",
    )

    metrics = train_idm_one_epoch(idm, loader, optimizer, config, torch.device("cpu"))

    expected_loss = torch.log1p(torch.exp(torch.tensor(-1.0))).item()
    assert metrics["idm_future_ranking_loss"] == pytest.approx(expected_loss)
    # Sampled-action scoring populates the sampled metrics and zeroes the endpoint metrics.
    assert metrics["idm_future_ranking_real_sampled_action_mse"] == pytest.approx(0.0)
    assert metrics["idm_future_ranking_best_negative_sampled_action_mse"] == pytest.approx(1.0)
    assert metrics["idm_future_ranking_mean_negative_sampled_action_mse"] == pytest.approx(1.0)
    assert metrics["idm_future_ranking_real_endpoint_mse"] == pytest.approx(0.0)
    assert metrics["idm_future_ranking_best_negative_endpoint_mse"] == pytest.approx(0.0)
    assert metrics["idm_future_ranking_mean_negative_endpoint_mse"] == pytest.approx(0.0)
    assert metrics["idm_future_ranking_rank_accuracy"] == pytest.approx(1.0)
    assert metrics["idm_future_ranking_negative_count"] == pytest.approx(1.0)
    assert metrics["loss"] == pytest.approx(2.0 * expected_loss)


def test_train_idm_same_task_future_delta_weight_contributes_to_loss_metric() -> None:
    idm = _FakeFlowIdm()
    loader = DataLoader(
        _future_contrastive_samples(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 1.0])),
        batch_size=2,
    )
    optimizer = torch.optim.SGD(idm.parameters(), lr=0.0)
    config = TrainConfig(
        model=ModelConfig(idm_arch="flow_transformer"),
        idm_loss_weight=0.0,
        action_smoothness_weight=0.0,
        idm_same_task_future_delta_weight=2.0,
        idm_same_task_future_delta_time_value=0.5,
    )

    metrics = train_idm_one_epoch(idm, loader, optimizer, config, torch.device("cpu"))

    assert metrics["idm_same_task_future_delta_loss"] == pytest.approx(1.0)
    assert metrics["idm_same_task_future_delta_donor_fraction"] == pytest.approx(1.0)
    assert metrics["idm_same_task_future_delta_no_donor_fraction"] == pytest.approx(0.0)
    assert metrics["idm_same_task_future_delta_action_delta_mse"] == pytest.approx(1.0)
    assert metrics["idm_same_task_future_delta_effective_donor_fraction"] == pytest.approx(1.0)
    assert metrics["idm_same_task_future_delta_prediction_delta_mse"] == pytest.approx(0.0)
    assert metrics["idm_same_task_future_delta_delta_cosine"] == pytest.approx(0.0)
    assert metrics["idm_same_task_future_delta_weight_active"] == pytest.approx(2.0)
    assert metrics["loss"] == pytest.approx(2.0)


def test_train_idm_one_epoch_same_task_batching_produces_valid_same_task_fraction() -> None:
    samples = _future_contrastive_samples(torch.tensor([0.0, 1.0, 2.0, 3.0]), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    for index, sample in enumerate(samples):
        sample["task_index"] = torch.tensor(index // 2, dtype=torch.long)
        sample["dataset_index"] = torch.tensor(index, dtype=torch.long)
        sample["episode_index"] = torch.tensor(index % 2, dtype=torch.long)
    sampler = train_lib.SameTaskBatchSampler(samples, batch_size=4, seed=11)
    loader = DataLoader(samples, batch_sampler=sampler)
    idm = _FakeFlowIdm()
    optimizer = torch.optim.SGD(idm.parameters(), lr=0.0)
    config = TrainConfig(
        model=ModelConfig(idm_arch="flow_transformer"),
        idm_loss_weight=0.0,
        action_smoothness_weight=0.0,
        idm_future_ranking_weight=1.0,
        idm_future_ranking_temperature=1.0,
        idm_future_ranking_same_task_negative=True,
        idm_same_task_batching=True,
    )

    metrics = train_idm_one_epoch(idm, loader, optimizer, config, torch.device("cpu"))

    assert metrics["idm_future_ranking_same_task_valid_fraction"] == pytest.approx(1.0)
    assert metrics["idm_future_ranking_same_task_no_donor_fraction"] == pytest.approx(0.0)


def test_train_idm_future_ranking_active_weight_zero_skips_ranking_path() -> None:
    idm = _FakeFlowIdm(forbid_contrastive_path=True)
    loader = DataLoader(
        _future_contrastive_samples(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])),
        batch_size=2,
    )
    optimizer = torch.optim.SGD(idm.parameters(), lr=0.0)
    config = TrainConfig(
        model=ModelConfig(idm_arch="flow_transformer"),
        idm_loss_weight=0.0,
        action_smoothness_weight=0.0,
        idm_future_ranking_weight=2.0,
        idm_future_ranking_temperature=1.0,
        idm_future_ranking_shuffled_future_negative=True,
    )

    metrics = train_idm_one_epoch(
        idm,
        loader,
        optimizer,
        config,
        torch.device("cpu"),
        future_ranking_weight=0.0,
    )

    assert metrics["idm_future_ranking_loss"] == pytest.approx(0.0)
    assert metrics["idm_future_ranking_weight_active"] == pytest.approx(0.0)
    assert metrics["loss"] == pytest.approx(0.0)


def test_future_contrastive_batch_size_one_uses_current_repeated_corruption() -> None:
    idm = _FakeFlowIdm()
    batch = _future_contrastive_batch(torch.tensor([5.0]), torch.tensor([5.0]))

    losses = compute_idm_losses(
        idm,
        batch,
        future_contrastive_weight=1.0,
        future_contrastive_margin=10.0,
    )

    assert torch.allclose(losses["idm_future_contrastive_real_endpoint_mse"], torch.tensor(0.0))
    assert torch.allclose(losses["idm_future_contrastive_corrupted_endpoint_mse"], torch.tensor(9.0))
    assert torch.allclose(losses["idm_future_contrastive_loss"], torch.tensor(1.0))
    assert torch.equal(losses["idm_future_contrastive_singleton_fraction"], torch.tensor(1.0))


def test_future_contrastive_wan_vae_latent_corruption_rolls_only_future_slices() -> None:
    latents = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3, 1, 1)

    corrupted, singleton_indicator = train_lib._corrupt_wan_vae_latents_for_contrast(latents)

    expected = latents.clone()
    expected[:, :, 1:] = latents[:, :, 1:].roll(shifts=1, dims=0)
    assert torch.equal(corrupted, expected)
    assert torch.equal(corrupted[:, :, 0], latents[:, :, 0])
    assert torch.equal(latents, torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3, 1, 1))
    assert torch.equal(singleton_indicator, torch.tensor(0.0))


def test_future_contrastive_wan_vae_latent_singleton_replaces_future_with_current() -> None:
    latents = torch.tensor([[[[[1.0]], [[5.0]], [[7.0]]], [[[2.0]], [[6.0]], [[8.0]]]]])

    corrupted, singleton_indicator = train_lib._corrupt_wan_vae_latents_for_contrast(latents)

    assert torch.equal(corrupted[:, :, 0], latents[:, :, 0])
    assert torch.equal(corrupted[:, :, 1:], latents[:, :, :1].expand_as(latents[:, :, 1:]))
    assert torch.equal(singleton_indicator, torch.tensor(1.0))


@pytest.mark.parametrize(
    ("latents", "message"),
    [
        (torch.ones((1, 1, 2, 1)), r"rank 5.*\(B,C,T,H,W\)"),
        (torch.ones((1, 1, 1, 1, 1)), r"T >= 2"),
    ],
)
def test_future_contrastive_wan_vae_latent_corruption_invalid_shape_raises(
    latents: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        train_lib._corrupt_wan_vae_latents_for_contrast(latents)


def test_future_contrastive_refuses_non_flow_idm() -> None:
    class RegressionIdm(torch.nn.Module):
        uses_flow_matching = False

        def forward(self, current_images, future_images, state, task_id):
            del future_images, state, task_id
            return torch.zeros((current_images.shape[0], 1, 1), device=current_images.device)

    with pytest.raises(ValueError, match="flow_transformer"):
        TrainConfig(idm_future_contrastive_weight=0.1)

    with pytest.raises(ValueError, match="flow_transformer"):
        TrainConfig(idm_context_action_loss_weight=0.1)

    with pytest.raises(ValueError, match="flow_transformer"):
        TrainConfig(idm_same_task_future_delta_weight=0.1)

    with pytest.raises(ValueError, match="flow_transformer"):
        compute_idm_losses(
            RegressionIdm(),
            _future_contrastive_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])),
            context_action_loss_weight=1.0,
        )

    with pytest.raises(ValueError, match="flow_transformer"):
        compute_idm_losses(
            RegressionIdm(),
            _future_contrastive_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])),
            future_contrastive_weight=1.0,
        )

    with pytest.raises(ValueError, match="flow_transformer"):
        compute_idm_losses(
            RegressionIdm(),
            _future_contrastive_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])),
            future_ranking_weight=1.0,
            future_ranking_repeated_current_negative=True,
        )

    with pytest.raises(ValueError, match="flow_transformer"):
        compute_idm_losses(
            RegressionIdm(),
            _future_contrastive_batch(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])),
            same_task_future_delta_weight=1.0,
        )


def test_state_normalizer_forward_hook_normalizes_loaded_state_once() -> None:
    class RecordingIdm(torch.nn.Module):
        uses_flow_matching = False

        def __init__(self) -> None:
            super().__init__()
            self.seen_state: torch.Tensor | None = None

        def forward(self, current_images, future_images, state, task_id, *, sample_noise=None):
            del future_images, task_id, sample_noise
            self.seen_state = state.detach().cpu()
            return torch.zeros((current_images.shape[0], 2, 1), device=current_images.device)

    idm = RecordingIdm()
    normalizer = StateNormalizer(mean=torch.tensor([10.0, 20.0]), std=torch.tensor([2.0, 5.0]))
    attach_state_normalizer(idm, normalizer, normalize_forward=True)
    raw_state = torch.tensor([[12.0, 10.0], [8.0, 25.0]])
    expected_state = torch.tensor([[1.0, -2.0], [-1.0, 1.0]])

    routed_state = normalize_state_for_idm(idm, raw_state, normalizer)
    assert torch.equal(routed_state, raw_state)

    idm(
        torch.zeros(2, 1, 3, 8, 8),
        torch.zeros(2, 1, 1, 3, 8, 8),
        routed_state,
        torch.zeros(2, dtype=torch.long),
    )

    assert idm.seen_state is not None
    assert torch.allclose(idm.seen_state, expected_state)


def test_future_augmentation_adds_noise_and_replaces_frames() -> None:
    batch = {
        "current_images": torch.zeros((2, 1, 3, 4, 4)),
        "future_images": torch.ones((2, 3, 1, 3, 4, 4)),
    }

    augmented = apply_future_augmentation(batch, noise_std=0.0, frame_dropout=1.0)

    assert torch.allclose(augmented["future_images"], torch.zeros_like(batch["future_images"]))
    assert batch["future_images"].max() == 1.0


def test_current_conditioning_dropout_default_is_noop_and_does_not_mutate() -> None:
    current_images = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 2, 2)
    future_images = torch.ones((2, 2, 1, 3, 2, 2))
    latents = torch.arange(12, dtype=torch.float32).reshape(2, 1, 3, 2, 1)
    batch = {
        "current_images": current_images.clone(),
        "future_images": future_images.clone(),
        "wan_vae_latents": latents.clone(),
    }

    augmented = apply_current_conditioning_dropout(
        batch,
        current_frame_dropout=0.0,
        wan_vae_current_latent_dropout=0.0,
    )

    assert torch.equal(augmented["current_images"], current_images)
    assert torch.equal(augmented["future_images"], future_images)
    assert torch.equal(augmented["wan_vae_latents"], latents)
    assert torch.equal(batch["current_images"], current_images)
    assert torch.equal(batch["future_images"], future_images)
    assert torch.equal(batch["wan_vae_latents"], latents)


def test_current_conditioning_dropout_one_drops_only_current_conditioning() -> None:
    current_images = torch.ones((2, 1, 3, 2, 2))
    future_images = torch.arange(48, dtype=torch.float32).reshape(2, 2, 1, 3, 2, 2)
    latents = torch.arange(12, dtype=torch.float32).reshape(2, 1, 3, 2, 1)
    batch = {
        "current_images": current_images.clone(),
        "future_images": future_images.clone(),
        "wan_vae_latents": latents.clone(),
    }

    augmented = apply_current_conditioning_dropout(
        batch,
        current_frame_dropout=1.0,
        wan_vae_current_latent_dropout=1.0,
    )

    assert torch.equal(augmented["current_images"], torch.zeros_like(current_images))
    assert torch.equal(augmented["future_images"], future_images)
    assert torch.equal(augmented["wan_vae_latents"][:, :, 0], torch.zeros_like(latents[:, :, 0]))
    assert torch.equal(augmented["wan_vae_latents"][:, :, 1:], latents[:, :, 1:])
    assert torch.equal(batch["current_images"], current_images)
    assert torch.equal(batch["future_images"], future_images)
    assert torch.equal(batch["wan_vae_latents"], latents)


def test_current_conditioning_dropout_validation() -> None:
    with pytest.raises(ValueError, match="idm_current_frame_dropout"):
        TrainConfig(idm_current_frame_dropout=-0.1)
    with pytest.raises(ValueError, match="idm_current_frame_dropout"):
        TrainConfig(idm_current_frame_dropout=1.1)
    with pytest.raises(ValueError, match="idm_wan_vae_current_latent_dropout"):
        TrainConfig(idm_wan_vae_current_latent_dropout=-0.1)
    with pytest.raises(ValueError, match="idm_wan_vae_current_latent_dropout"):
        TrainConfig(idm_wan_vae_current_latent_dropout=1.1)


def test_wan_vae_latent_noise_default_is_noop_and_does_not_mutate() -> None:
    latents = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2, 1)
    batch = {
        "current_images": torch.zeros((2, 1, 3, 4, 4)),
        "future_images": torch.ones((2, 1, 1, 3, 4, 4)),
        "wan_vae_latents": latents.clone(),
    }

    augmented, stats = apply_wan_vae_latent_noise(batch, prob=0.0, s_min=0.5, s_max=1.0)

    assert torch.equal(augmented["wan_vae_latents"], latents)
    assert torch.equal(batch["wan_vae_latents"], latents)
    assert stats == {"augmented_count": 0.0, "sample_count": 2.0, "s_sum": 0.0}


def test_wan_vae_latent_noise_validation() -> None:
    with pytest.raises(ValueError, match="idm_wan_vae_latent_noise_prob"):
        TrainConfig(idm_wan_vae_latent_noise_prob=-0.1)
    with pytest.raises(ValueError, match="idm_wan_vae_latent_noise_s_min"):
        TrainConfig(idm_wan_vae_latent_noise_s_min=-0.1)
    with pytest.raises(ValueError, match="idm_wan_vae_latent_noise_s_max"):
        TrainConfig(idm_wan_vae_latent_noise_s_max=1.1)
    with pytest.raises(ValueError, match="idm_wan_vae_latent_noise_s_min.*<="):
        TrainConfig(idm_wan_vae_latent_noise_s_min=0.9, idm_wan_vae_latent_noise_s_max=0.2)
    with pytest.raises(ValueError, match="idm_wan_vae_latent_noise_time_mode"):
        TrainConfig(idm_wan_vae_latent_noise_time_mode="bogus")


def test_future_ranking_config_validation() -> None:
    with pytest.raises(ValueError, match="idm_future_ranking_weight"):
        TrainConfig(idm_future_ranking_weight=-0.1)
    with pytest.raises(ValueError, match="idm_future_ranking_start_epoch"):
        TrainConfig(idm_future_ranking_start_epoch=-1)
    with pytest.raises(ValueError, match="idm_future_ranking_ramp_epochs"):
        TrainConfig(idm_future_ranking_ramp_epochs=-1)
    with pytest.raises(ValueError, match="idm_future_ranking_temperature"):
        TrainConfig(idm_future_ranking_temperature=0.0)
    with pytest.raises(ValueError, match="idm_future_ranking_noise_std"):
        TrainConfig(idm_future_ranking_noise_std=-0.1)
    with pytest.raises(ValueError, match="flow_transformer"):
        TrainConfig(idm_future_ranking_weight=0.1, idm_future_ranking_repeated_current_negative=True)
    with pytest.raises(ValueError, match="at least one enabled negative"):
        TrainConfig(
            model=ModelConfig(idm_arch="flow_transformer"),
            idm_future_ranking_weight=0.1,
        )
    TrainConfig(
        model=ModelConfig(idm_arch="flow_transformer"),
        idm_future_ranking_weight=0.1,
        idm_future_ranking_same_task_negative=True,
    )


def test_same_task_future_delta_config_validation() -> None:
    with pytest.raises(ValueError, match="idm_same_task_future_delta_weight"):
        TrainConfig(
            model=ModelConfig(idm_arch="flow_transformer"),
            idm_same_task_future_delta_weight=-0.1,
        )
    with pytest.raises(ValueError, match="idm_same_task_future_delta_time_value"):
        TrainConfig(
            model=ModelConfig(idm_arch="flow_transformer"),
            idm_same_task_future_delta_time_value=1.1,
        )
    with pytest.raises(ValueError, match="idm_same_task_future_delta_time_value"):
        TrainConfig(
            model=ModelConfig(idm_arch="flow_transformer"),
            idm_same_task_future_delta_time_value=1.0,
        )
    with pytest.raises(ValueError, match="idm_same_task_future_delta_max_state_distance"):
        TrainConfig(
            model=ModelConfig(idm_arch="flow_transformer"),
            idm_same_task_future_delta_max_state_distance=-0.1,
        )
    with pytest.raises(ValueError, match="idm_same_task_future_delta_min_action_delta_mse"):
        TrainConfig(
            model=ModelConfig(idm_arch="flow_transformer"),
            idm_same_task_future_delta_min_action_delta_mse=-0.1,
        )
    with pytest.raises(ValueError, match="flow_transformer"):
        TrainConfig(idm_same_task_future_delta_weight=0.1)


def test_wan_vae_latent_noise_all_mode_corrupts_all_time_slices(monkeypatch) -> None:
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.zeros_like(tensor))
    latents = torch.ones((2, 1, 3, 2, 1))
    batch = {
        "current_images": torch.zeros((2, 1, 3, 4, 4)),
        "future_images": torch.ones((2, 1, 1, 3, 4, 4)),
        "wan_vae_latents": latents.clone(),
    }

    augmented, stats = apply_wan_vae_latent_noise(batch, prob=1.0, s_min=0.0, s_max=0.0)

    assert torch.equal(augmented["wan_vae_latents"], torch.zeros_like(latents))
    assert torch.equal(batch["wan_vae_latents"], latents)
    assert stats == {"augmented_count": 2.0, "sample_count": 2.0, "s_sum": 0.0}


def test_wan_vae_latent_noise_future_only_preserves_current_time_slice(monkeypatch) -> None:
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.zeros_like(tensor))
    latents = torch.ones((2, 1, 3, 2, 1))
    batch = {
        "current_images": torch.zeros((2, 1, 3, 4, 4)),
        "future_images": torch.ones((2, 1, 1, 3, 4, 4)),
        "wan_vae_latents": latents.clone(),
    }

    augmented, stats = apply_wan_vae_latent_noise(
        batch,
        prob=1.0,
        s_min=0.0,
        s_max=0.0,
        time_mode="future_only",
    )

    assert torch.equal(augmented["wan_vae_latents"][:, :, 0], latents[:, :, 0])
    assert torch.equal(augmented["wan_vae_latents"][:, :, 1:], torch.zeros_like(latents[:, :, 1:]))
    assert torch.equal(batch["wan_vae_latents"], latents)
    assert stats == {"augmented_count": 2.0, "sample_count": 2.0, "s_sum": 0.0}


def test_wan_vae_latent_noise_invalid_time_mode_raises() -> None:
    batch = {
        "current_images": torch.zeros((1, 1, 3, 4, 4)),
        "future_images": torch.ones((1, 1, 1, 3, 4, 4)),
        "wan_vae_latents": torch.ones((1, 1, 2, 1, 1)),
    }

    with pytest.raises(ValueError, match="idm_wan_vae_latent_noise_time_mode"):
        apply_wan_vae_latent_noise(batch, prob=1.0, s_min=0.0, s_max=0.0, time_mode="bogus")


def test_wan_vae_latent_noise_changes_only_wan_vae_latents() -> None:
    torch.manual_seed(123)
    batch = {
        "current_images": torch.zeros((2, 1, 3, 4, 4)),
        "future_images": torch.ones((2, 1, 1, 3, 4, 4)),
        "state": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "wan_vae_latents": torch.ones((2, 1, 2, 2, 1)),
    }
    original = {key: value.clone() for key, value in batch.items()}

    augmented, stats = apply_wan_vae_latent_noise(batch, prob=1.0, s_min=0.0, s_max=0.0)

    assert not torch.equal(augmented["wan_vae_latents"], original["wan_vae_latents"])
    assert torch.equal(batch["wan_vae_latents"], original["wan_vae_latents"])
    for key in ("current_images", "future_images", "state"):
        assert torch.equal(augmented[key], original[key])
        assert torch.equal(batch[key], original[key])
    assert stats == {"augmented_count": 2.0, "sample_count": 2.0, "s_sum": 0.0}


def test_train_idm_one_epoch_reports_wan_vae_latent_noise_metrics() -> None:
    idm = _RecordingWanLatentFlowIdm()
    samples = [
        {
            "current_images": torch.zeros(1, 3, 1, 1),
            "future_images": torch.zeros(1, 1, 3, 1, 1),
            "state": torch.zeros(1),
            "task_id": torch.tensor(0, dtype=torch.long),
            "action_chunk": torch.ones(1, 1),
            "action_mask": torch.ones(1),
            "wan_vae_latents": torch.ones(1, 1, 1, 1),
        }
        for _ in range(2)
    ]
    loader = DataLoader(samples, batch_size=2)
    optimizer = torch.optim.SGD(idm.parameters(), lr=0.0)
    config = TrainConfig(
        model=idm.config,
        idm_wan_vae_latent_noise_prob=1.0,
        idm_wan_vae_latent_noise_s_min=0.0,
        idm_wan_vae_latent_noise_s_max=0.0,
    )

    metrics = train_idm_one_epoch(idm, loader, optimizer, config, torch.device("cpu"))

    assert metrics["idm_wan_vae_latent_noise_fraction"] == pytest.approx(1.0)
    assert metrics["idm_wan_vae_latent_noise_s_mean"] == pytest.approx(0.0)
    assert idm.seen_latents
    assert not torch.equal(idm.seen_latents[-1], torch.ones_like(idm.seen_latents[-1]))


def test_train_idm_one_epoch_applies_current_conditioning_dropout_only() -> None:
    idm = _RecordingWanLatentFlowIdm()
    samples = [
        {
            "current_images": torch.ones(1, 3, 1, 1) * (index + 1),
            "future_images": torch.zeros(1, 1, 3, 1, 1),
            "state": torch.zeros(1),
            "task_id": torch.tensor(0, dtype=torch.long),
            "action_chunk": torch.ones(1, 1),
            "action_mask": torch.ones(1),
            "wan_vae_latents": torch.tensor([[[[10.0 + index]], [[20.0 + index]]]]),
        }
        for index in range(2)
    ]
    loader = DataLoader(samples, batch_size=2)
    optimizer = torch.optim.SGD(idm.parameters(), lr=0.0)
    config = TrainConfig(
        model=idm.config,
        idm_current_frame_dropout=1.0,
        idm_wan_vae_current_latent_dropout=1.0,
    )

    train_idm_one_epoch(idm, loader, optimizer, config, torch.device("cpu"))

    assert idm.seen_current_images
    assert idm.seen_latents
    seen_latents = idm.seen_latents[-1]
    assert torch.equal(idm.seen_current_images[-1], torch.zeros_like(idm.seen_current_images[-1]))
    assert torch.equal(seen_latents[:, :, 0], torch.zeros_like(seen_latents[:, :, 0]))
    assert torch.equal(seen_latents[:, :, 1], torch.tensor([[[[20.0]]], [[[21.0]]]]))


def test_evaluate_idm_uses_clean_wan_vae_latents() -> None:
    class RecordingEvalIdm(torch.nn.Module):
        uses_flow_matching = False

        def __init__(self) -> None:
            super().__init__()
            self.seen_latents: list[torch.Tensor] = []

        def forward(self, current_images, future_images, state, task_id, *, wan_vae_latents=None, sample_noise=None):
            del future_images, state, task_id, sample_noise
            if wan_vae_latents is not None:
                self.seen_latents.append(wan_vae_latents.detach().cpu())
            return torch.zeros((current_images.shape[0], 1, 1), device=current_images.device)

    clean_latents = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
    samples = [
        {
            "current_images": torch.zeros(1, 3, 1, 1),
            "future_images": torch.zeros(1, 1, 3, 1, 1),
            "state": torch.zeros(1),
            "task_id": torch.tensor(0, dtype=torch.long),
            "action_chunk": torch.zeros(1, 1),
            "action_mask": torch.ones(1),
            "wan_vae_latents": clean_latents[index],
        }
        for index in range(2)
    ]
    idm = RecordingEvalIdm()

    evaluate_idm(idm, DataLoader(samples, batch_size=2), torch.device("cpu"))

    assert len(idm.seen_latents) == 1
    assert torch.equal(idm.seen_latents[0], clean_latents)


def test_train_idm_refuses_cached_future_dir(tmp_path) -> None:
    with pytest.raises(ValueError, match="Generated/cached futures are for eval/ranking only"):
        train_idm_main(
            TrainIdmArgs(
                dataset_source="synthetic",
                output_dir=str(tmp_path / "idm"),
                cached_future_dir=str(tmp_path / "cache"),
                epochs=1,
                batch_size=4,
                image_size=32,
                synthetic_samples=8,
                action_horizon=4,
                device="cpu",
            )
        )


def test_train_idm_refuses_include_gt_futures_with_cache(tmp_path) -> None:
    with pytest.raises(ValueError, match="Generated/cached futures are for eval/ranking only"):
        train_idm_main(
            TrainIdmArgs(
                dataset_source="synthetic",
                output_dir=str(tmp_path / "idm"),
                include_gt_futures_with_cache=True,
                epochs=1,
                batch_size=4,
                image_size=32,
                synthetic_samples=8,
                action_horizon=4,
                device="cpu",
            )
        )


@pytest.mark.parametrize(
    ("cached_future_dir_name", "include_gt_futures_with_cache"),
    [
        ("cache", False),
        (None, True),
        ("cache", True),
    ],
)
def test_run_idm_training_refuses_cached_or_generated_futures(
    tmp_path,
    cached_future_dir_name,
    include_gt_futures_with_cache,
) -> None:
    cached_future_dir = tmp_path / cached_future_dir_name if cached_future_dir_name is not None else None
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=32,
            synthetic_samples=8,
            action_horizon=4,
        ),
        output_dir=str(tmp_path / "idm"),
        epochs=1,
        batch_size=4,
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="Generated/cached futures are for eval/ranking only; IDM training uses ground-truth dataset futures",
    ):
        run_idm_training(
            config,
            cached_future_dir=cached_future_dir,
            include_gt_futures_with_cache=include_gt_futures_with_cache,
        )


def test_synthetic_idm_training_can_stop_early(tmp_path) -> None:
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=32,
            synthetic_samples=16,
            action_horizon=8,
        ),
        output_dir=str(tmp_path),
        epochs=5,
        batch_size=4,
        device="cpu",
        early_stopping_patience=0,
        seed=11,
    )

    metrics = run_idm_training(config)

    assert metrics["stopped_early"]
    assert len(metrics["history"]) == 1


def test_synthetic_flow_transformer_idm_training_writes_artifacts(tmp_path) -> None:
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=32,
            synthetic_samples=12,
            num_future_frames=2,
            action_horizon=4,
        ),
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
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=4,
        device="cpu",
        seed=11,
    )

    metrics = run_idm_training(config)
    checkpoint = torch.load(tmp_path / "idm_checkpoint.pt", map_location="cpu", weights_only=False)

    assert metrics["training_target"] == "idm"
    assert metrics["model_config"]["idm_arch"] == "flow_transformer"
    assert metrics["model_config"]["idm_flow_visual_token_scope"] == "all"
    assert checkpoint["model_config"]["idm_flow_visual_token_scope"] == "all"
    assert metrics["final"]["idm_mse"] >= 0.0
    assert (tmp_path / "idm_checkpoint.pt").exists()


def test_run_idm_training_future_usage_gate_fallback_still_writes_best_checkpoint(tmp_path, monkeypatch) -> None:
    class CurrentOnlyTrainingFlowIdm(torch.nn.Module):
        uses_flow_matching = True

        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.config = config
            self.transition_encoder = _CurrentScalarTransitionEncoder()
            self.flow_head = _EndpointFlowHead()

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
            del future_images, state, task_id, sample_noise, action_mask
            if mode == "loss":
                if target_action is None:
                    raise ValueError("target_action is required")
                return {"loss": target_action.sum() * 0.0 + self.flow_head.scale * 0.0}
            return torch.zeros(
                (current_images.shape[0], self.config.action_horizon, self.config.action_dim),
                device=current_images.device,
            )

    def create_current_only_idm(config, device):
        return CurrentOnlyTrainingFlowIdm(config).to(device)

    monkeypatch.setattr(train_lib, "create_idm_model", create_current_only_idm)
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=16,
            synthetic_samples=12,
            action_horizon=1,
        ),
        model=ModelConfig(
            idm_arch="flow_transformer",
            latent_dim=16,
            idm_transformer_layers=1,
            idm_transformer_heads=4,
            idm_transformer_patch_size=16,
            idm_transformer_dropout=0.0,
            idm_flow_sampling_steps=2,
        ),
        output_dir=str(tmp_path),
        epochs=2,
        batch_size=4,
        eval_fraction=0.25,
        split_gap=0,
        normalize_actions=False,
        device="cpu",
        idm_future_usage_eval=True,
        seed=11,
    )

    metrics = run_idm_training(config)

    assert all(row["future_usage_gate_pass"] is False for row in metrics["history"])
    assert metrics["best_future_usage_gate_pass"] is False
    assert metrics["best"]["future_usage_gate_pass"] is False
    assert (tmp_path / "best_idm_checkpoint.pt").exists()


def test_run_idm_training_forwards_future_usage_score_mode(tmp_path, monkeypatch) -> None:
    class CurrentOnlyTrainingFlowIdm(torch.nn.Module):
        uses_flow_matching = True

        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.config = config
            self.transition_encoder = _CurrentScalarTransitionEncoder()
            self.flow_head = _EndpointFlowHead()

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
            del future_images, state, task_id, sample_noise, action_mask
            if mode == "loss":
                if target_action is None:
                    raise ValueError("target_action is required")
                return {"loss": target_action.sum() * 0.0 + self.flow_head.scale * 0.0}
            return torch.zeros(
                (current_images.shape[0], self.config.action_horizon, self.config.action_dim),
                device=current_images.device,
            )

    captured = {}

    def fake_future_usage(*args, **kwargs):
        del args
        captured["score_mode"] = kwargs.get("score_mode")
        return {
            "future_usage_gate_pass": True,
            "future_usage_gate_reasons": "",
            "future_usage_score_mode": kwargs.get("score_mode"),
        }

    monkeypatch.setattr(
        train_lib, "create_idm_model", lambda config, device: CurrentOnlyTrainingFlowIdm(config).to(device)
    )
    monkeypatch.setattr(train_lib, "evaluate_idm_future_usage", fake_future_usage)
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=16,
            synthetic_samples=12,
            action_horizon=1,
        ),
        model=ModelConfig(
            idm_arch="flow_transformer",
            latent_dim=16,
            idm_transformer_layers=1,
            idm_transformer_heads=4,
            idm_transformer_patch_size=16,
            idm_transformer_dropout=0.0,
            idm_flow_sampling_steps=2,
        ),
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=4,
        eval_fraction=0.25,
        split_gap=0,
        normalize_actions=False,
        device="cpu",
        idm_future_usage_eval=True,
        idm_future_usage_score_mode="sampled_action",
        seed=11,
    )

    run_idm_training(config)

    assert captured["score_mode"] == "sampled_action"


def test_run_idm_training_same_task_batching_produces_nonzero_same_task_fraction(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(train_lib, "create_idm_model", lambda config, device: _FakeFlowIdm())
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=16,
            synthetic_samples=20,
            action_horizon=1,
            task_vocab_size=5,
        ),
        model=ModelConfig(
            idm_arch="flow_transformer",
            latent_dim=16,
            idm_transformer_layers=1,
            idm_transformer_heads=4,
            idm_transformer_patch_size=16,
            idm_transformer_dropout=0.0,
            idm_flow_sampling_steps=2,
        ),
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=4,
        eval_fraction=0.2,
        split_gap=0,
        normalize_actions=False,
        device="cpu",
        idm_loss_weight=0.0,
        action_smoothness_weight=0.0,
        idm_future_ranking_weight=1.0,
        idm_future_ranking_temperature=1.0,
        idm_future_ranking_same_task_negative=True,
        idm_same_task_batching=True,
        seed=11,
    )

    metrics = run_idm_training(config)

    assert metrics["train_config"]["idm_same_task_batching"] is True
    assert metrics["final"]["train_idm_future_ranking_same_task_valid_fraction"] > 0.0
    assert (tmp_path / "idm_checkpoint.pt").exists()


def test_synthetic_flow_transformer_idm_training_with_history_writes_artifacts(tmp_path) -> None:
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=32,
            synthetic_samples=12,
            num_future_frames=2,
            action_horizon=4,
            idm_history_length=2,
        ),
        model=ModelConfig(
            idm_arch="flow_transformer",
            idm_history_length=2,
            latent_dim=64,
            idm_transformer_layers=1,
            idm_transformer_heads=4,
            idm_transformer_patch_size=16,
            idm_transformer_dropout=0.0,
            idm_flow_sampling_steps=2,
        ),
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=4,
        device="cpu",
        seed=11,
    )

    metrics = run_idm_training(config)
    checkpoint = torch.load(tmp_path / "idm_checkpoint.pt", map_location="cpu", weights_only=False)

    assert metrics["model_config"]["idm_history_length"] == 2
    assert checkpoint["train_config"]["dataset"]["idm_history_length"] == 2
    assert checkpoint["train_config"]["model"]["idm_history_length"] == 2
    assert metrics["final"]["idm_mse"] >= 0.0


def test_flow_transformer_eval_seed_is_deterministic(tmp_path) -> None:
    dataset = DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),
        image_size=32,
        synthetic_samples=12,
        num_future_frames=2,
        action_horizon=4,
    )
    config = TrainConfig(
        dataset=dataset,
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
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=4,
        device="cpu",
        seed=13,
    )
    run_idm_training(config)
    idm, _ = load_idm_checkpoint(tmp_path / "idm_checkpoint.pt", torch.device("cpu"))
    loader = DataLoader(create_dataset_with_optional_cache(dataset), batch_size=4, shuffle=False)

    first = evaluate_idm(idm, loader, torch.device("cpu"), flow_eval_seed=123)
    second = evaluate_idm(idm, loader, torch.device("cpu"), flow_eval_seed=123)

    assert first == second


def test_wan_vae_idm_checkpoint_excludes_external_encoder_and_reloads_with_fake(monkeypatch, tmp_path) -> None:
    external_encoder = _ExternalWanVaeEncoder()
    monkeypatch.setattr(
        "world_model.wan_vae_encoder.build_frozen_wan_vae_encoder",
        lambda config: external_encoder,
    )
    config = ModelConfig(
        num_views=1,
        num_future_frames=4,
        image_size=32,
        state_dim=4,
        action_dim=4,
        action_horizon=4,
        latent_dim=64,
        idm_arch="flow_transformer",
        idm_visual_encoder="wan_vae",
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        idm_transformer_dropout=0.0,
        idm_flow_sampling_steps=2,
    )
    idm = InverseDynamicsModel(config)

    idm_state = module_state_dict_for_checkpoint(idm)
    assert not any("_wan_encoder" in key or "wan_encoder" in key for key in idm_state)
    assert "external_weight" not in {key.rsplit(".", maxsplit=1)[-1] for key in idm_state}

    checkpoint_path = tmp_path / "wan_vae_idm.pt"
    save_idm_state_checkpoint(
        checkpoint_path,
        idm_state=idm_state,
        model_config=config,
        train_config=TrainConfig(model=config),
        metrics={},
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["idm"].keys() == idm_state.keys()
    assert not any("_wan_encoder" in key or "wan_encoder" in key for key in checkpoint["idm"])
    assert "external_weight" not in {key.rsplit(".", maxsplit=1)[-1] for key in checkpoint["idm"]}

    monkeypatch.setattr(
        "world_model.wan_vae_encoder.build_frozen_wan_vae_encoder",
        lambda config: FakeWanVaeEncoder(
            latent_channels=config.wan_vae_latent_channels,
            spatial_stride=config.wan_vae_spatial_stride,
        ),
    )
    loaded_idm, loaded_config = load_idm_checkpoint(checkpoint_path, torch.device("cpu"))

    assert loaded_config == config
    assert isinstance(loaded_idm.transition_encoder.wan_encoder, FakeWanVaeEncoder)


@pytest.fixture(scope="module")
def gt_idm_checkpoint(tmp_path_factory) -> Path:
    """Train a tiny ground-truth-only IDM once and return its checkpoint path.

    ``run_idm_training`` refuses cached/generated futures, so this checkpoint's
    recorded provenance is clean and is the baseline that the load-time guard
    must accept.
    """
    output_dir = tmp_path_factory.mktemp("gt_idm")
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            image_keys=("corner4.image",),
            image_size=32,
            synthetic_samples=8,
            action_horizon=4,
        ),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=4,
        device="cpu",
        seed=11,
    )
    run_idm_training(config)
    return output_dir / "idm_checkpoint.pt"


def _copy_idm_checkpoint(src: Path, dst: Path, mutate=None) -> Path:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    if mutate is not None:
        mutate(checkpoint)
    torch.save(checkpoint, dst)
    return dst


def test_load_idm_checkpoint_accepts_gt_only_checkpoint(gt_idm_checkpoint) -> None:
    idm, model_config = load_idm_checkpoint(gt_idm_checkpoint, torch.device("cpu"))

    assert isinstance(model_config, ModelConfig)
    assert isinstance(idm, torch.nn.Module)
    assert get_state_normalizer(idm, torch.device("cpu")) is not None


def test_load_idm_checkpoint_without_state_normalizer_keeps_legacy_behavior(gt_idm_checkpoint, tmp_path) -> None:
    legacy = _copy_idm_checkpoint(
        gt_idm_checkpoint,
        tmp_path / "legacy_without_state_normalizer.pt",
        lambda checkpoint: (
            checkpoint.pop("state_normalizer", None),
            checkpoint["metrics"].pop("state_normalizer", None),
        ),
    )

    idm, model_config = load_idm_checkpoint(legacy, torch.device("cpu"))

    assert isinstance(model_config, ModelConfig)
    assert isinstance(idm, torch.nn.Module)
    assert get_state_normalizer(idm, torch.device("cpu")) is None
    assert not state_normalizer_applies_in_forward(idm)


def test_load_idm_checkpoint_rejects_cached_future_dir_provenance(gt_idm_checkpoint, tmp_path) -> None:
    bad = _copy_idm_checkpoint(
        gt_idm_checkpoint,
        tmp_path / "cached_future.pt",
        lambda checkpoint: checkpoint["metrics"].__setitem__("cached_future_dir", str(tmp_path / "cache")),
    )

    with pytest.raises(ValueError, match="cached/generated futures"):
        load_idm_checkpoint(bad, torch.device("cpu"))


def test_load_idm_checkpoint_rejects_include_gt_futures_with_cache_provenance(gt_idm_checkpoint, tmp_path) -> None:
    bad = _copy_idm_checkpoint(
        gt_idm_checkpoint,
        tmp_path / "mixed_future.pt",
        lambda checkpoint: checkpoint["metrics"].__setitem__("include_gt_futures_with_cache", True),
    )

    with pytest.raises(ValueError, match="cached/generated futures"):
        load_idm_checkpoint(bad, torch.device("cpu"))


def test_load_idm_checkpoint_rejects_train_config_provenance(gt_idm_checkpoint, tmp_path) -> None:
    # Defense in depth also covers provenance recorded under train_config, not just metrics.
    bad = _copy_idm_checkpoint(
        gt_idm_checkpoint,
        tmp_path / "train_config_future.pt",
        lambda checkpoint: checkpoint["train_config"].__setitem__("cached_future_dir", str(tmp_path / "cache")),
    )

    with pytest.raises(ValueError, match="cached/generated futures"):
        load_idm_checkpoint(bad, torch.device("cpu"))


def test_load_idm_checkpoint_rejects_generated_target_source_provenance(gt_idm_checkpoint, tmp_path) -> None:
    bad = _copy_idm_checkpoint(
        gt_idm_checkpoint,
        tmp_path / "generated_target.pt",
        lambda checkpoint: checkpoint["train_config"].__setitem__("idm_target_source", "generated"),
    )

    with pytest.raises(ValueError, match="cached/generated futures"):
        load_idm_checkpoint(bad, torch.device("cpu"))


def test_load_idm_checkpoint_escape_hatch_allows_non_gt_checkpoint(gt_idm_checkpoint, tmp_path) -> None:
    bad = _copy_idm_checkpoint(
        gt_idm_checkpoint,
        tmp_path / "cached_future.pt",
        lambda checkpoint: checkpoint["metrics"].__setitem__("cached_future_dir", str(tmp_path / "cache")),
    )

    idm, model_config = load_idm_checkpoint(bad, torch.device("cpu"), allow_non_gt_futures=True)

    assert isinstance(model_config, ModelConfig)
    assert isinstance(idm, torch.nn.Module)


def test_evaluate_idm_weights_metrics_by_valid_action_elements() -> None:
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
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    metrics = evaluate_idm(ZeroIdm(), loader, torch.device("cpu"))

    assert metrics["idm_mse"] == torch.tensor((1.0 + 1.0 + 100.0) / 3.0).item()
    assert metrics["idm_smooth_l1"] == torch.tensor((0.5 + 0.5 + 9.5) / 3.0).item()
