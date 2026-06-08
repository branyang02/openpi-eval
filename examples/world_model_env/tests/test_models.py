from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from world_model.config import ModelConfig
from world_model.models import (
    ConvVideoWorldModel,
    CurrentOnlyTransitionEncoder,
    FlowActionTransformerHead,
    InverseDynamicsModel,
    ResidualTransitionEncoder,
    TransformerTransitionEncoder,
    WanVaeTransitionEncoder,
)
from world_model.train_lib import create_flow_sample_noise
from world_model.wan_vae_encoder import FakeWanVaeEncoder


class _ConstantTransitionEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, current_images, future_images, state):
        del future_images, state
        return torch.zeros(current_images.shape[0], self.latent_dim, device=current_images.device)


class _RecordingFlowHead(nn.Module):
    def __init__(self, predicted_velocity: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("predicted_velocity", predicted_velocity.clone())
        self.noisy_action: torch.Tensor | None = None
        self.time: torch.Tensor | None = None
        self.history_tokens: torch.Tensor | None = None
        self.visual_context_tokens: torch.Tensor | None = None
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, context, noisy_action, time, *, history_tokens=None, visual_context_tokens=None):
        del context
        self.noisy_action = noisy_action.detach().clone()
        self.time = time.detach().clone()
        self.calls.append((self.noisy_action, self.time))
        self.history_tokens = None if history_tokens is None else history_tokens.detach().clone()
        self.visual_context_tokens = (
            None if visual_context_tokens is None else visual_context_tokens.detach().clone()
        )
        return self.predicted_velocity.to(device=noisy_action.device, dtype=noisy_action.dtype)


class _ConstantVelocityFlowHead(nn.Module):
    """Flow head returning a constant velocity shaped like the queried action."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, context, noisy_action, time, *, history_tokens=None, visual_context_tokens=None):
        del context, history_tokens, visual_context_tokens
        self.calls.append((noisy_action.detach().clone(), time.detach().clone()))
        return torch.full_like(noisy_action, self.value)


class _ZeroRecordingFlowHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, tuple[int, ...] | None]] = []

    def forward(self, context, noisy_action, time, *, history_tokens=None, visual_context_tokens=None):
        del time
        self.calls.append(
            {
                "context": tuple(context.shape),
                "noisy_action": tuple(noisy_action.shape),
                "history_tokens": None if history_tokens is None else tuple(history_tokens.shape),
                "visual_context_tokens": (
                    None if visual_context_tokens is None else tuple(visual_context_tokens.shape)
                ),
            }
        )
        return torch.zeros_like(noisy_action)


class _ContextEchoFlowHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.history_tokens: list[torch.Tensor | None] = []

    def forward(self, context, noisy_action, time, *, history_tokens=None, visual_context_tokens=None):
        del time, visual_context_tokens
        self.history_tokens.append(None if history_tokens is None else history_tokens.detach().clone())
        velocity = torch.zeros_like(noisy_action)
        width = min(context.shape[-1], noisy_action.shape[-1])
        velocity[..., :width] = context[:, None, :width]
        return velocity


class _InputSummaryTransitionEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.calls: list[dict[str, float]] = []

    def forward(self, current_images, future_images, state, *, wan_vae_latents=None, return_tokens=False):
        batch_size = current_images.shape[0]
        current_signal = current_images.flatten(1).abs().sum(dim=1)
        state_signal = state.flatten(1).abs().sum(dim=1)
        if wan_vae_latents is None:
            future_signal = future_images.flatten(1).mean(dim=1)
            current_latent_signal = torch.zeros_like(future_signal)
        else:
            future_latents = wan_vae_latents[:, :, 1:]
            future_signal = future_latents.flatten(1).mean(dim=1)
            current_latent_signal = wan_vae_latents[:, :, :1].flatten(1).abs().sum(dim=1)
        context = torch.zeros(batch_size, self.latent_dim, device=current_images.device, dtype=current_images.dtype)
        context[:, 0] = future_signal
        context[:, 1] = current_signal + state_signal + current_latent_signal
        self.calls.append(
            {
                "current_sum": float(current_signal.sum().detach().cpu()),
                "state_sum": float(state_signal.sum().detach().cpu()),
                "current_latent_sum": float(current_latent_signal.sum().detach().cpu()),
                "future_signal": float(future_signal.sum().detach().cpu()),
            }
        )
        if return_tokens:
            return context, context[:, None, :]
        return context


class _RecordingWanEncoder:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.videos: list[torch.Tensor] = []

    def encode_videos(self, videos: torch.Tensor) -> torch.Tensor:
        self.videos.append(videos.detach().cpu())
        batch_size = videos.shape[0]
        latent_frames = (videos.shape[2] + 3) // 4
        latent_side = self.config.image_size // self.config.wan_vae_spatial_stride
        latents = torch.ones(
            batch_size,
            self.config.wan_vae_latent_channels,
            latent_frames,
            latent_side,
            latent_side,
            device=videos.device,
            dtype=videos.dtype,
        )
        if latent_frames > 1:
            latents[:, :, 1:] = 2.0
        return latents


def _flow_time_test_config(**overrides) -> ModelConfig:
    base = {
        "num_views": 1,
        "num_future_frames": 1,
        "image_size": 16,
        "state_dim": 2,
        "action_dim": 2,
        "action_horizon": 3,
        "latent_dim": 16,
        "idm_arch": "flow_transformer",
        "idm_transformer_layers": 1,
        "idm_transformer_heads": 4,
        "idm_transformer_patch_size": 16,
        "idm_transformer_dropout": 0.0,
        "idm_transformer_ff_dim": 32,
    }
    base.update(overrides)
    return ModelConfig(**base)


def _wan_token_test_config(**overrides) -> ModelConfig:
    base = {
        "num_views": 1,
        "num_future_frames": 4,
        "image_size": 32,
        "state_dim": 4,
        "action_dim": 4,
        "action_horizon": 4,
        "latent_dim": 64,
        "idm_arch": "flow_transformer",
        "idm_visual_encoder": "wan_vae",
        "idm_transformer_layers": 1,
        "idm_transformer_heads": 4,
        "idm_transformer_patch_size": 16,
        "idm_transformer_dropout": 0.0,
        "idm_transformer_ff_dim": 128,
        "idm_flow_sampling_steps": 2,
        "wan_vae_use_cached_latents": True,
    }
    base.update(overrides)
    return ModelConfig(**base)


def _patch_token_test_config(**overrides) -> ModelConfig:
    base = {
        "num_views": 2,
        "num_future_frames": 2,
        "image_size": 16,
        "state_dim": 3,
        "action_dim": 2,
        "action_horizon": 3,
        "latent_dim": 32,
        "idm_arch": "flow_transformer",
        "idm_visual_encoder": "patch",
        "idm_transformer_layers": 1,
        "idm_transformer_heads": 4,
        "idm_transformer_patch_size": 8,
        "idm_transformer_dropout": 0.0,
        "idm_transformer_ff_dim": 64,
        "idm_flow_sampling_steps": 2,
        "idm_flow_num_samples": 2,
    }
    base.update(overrides)
    return ModelConfig(**base)


def _wan_latents(config: ModelConfig, batch_size: int) -> torch.Tensor:
    latent_frames = (1 + config.num_future_frames + 3) // 4
    latent_side = config.image_size // config.wan_vae_spatial_stride
    return torch.rand(
        batch_size,
        config.wan_vae_latent_channels,
        latent_frames,
        latent_side,
        latent_side,
    )


def _flow_time_loss_inputs(config: ModelConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = 2
    current = torch.zeros(batch_size, config.num_views, 3, config.image_size, config.image_size)
    future = torch.zeros(
        batch_size,
        config.num_future_frames,
        config.num_views,
        3,
        config.image_size,
        config.image_size,
    )
    state = torch.zeros(batch_size, config.state_dim)
    action_mask = torch.ones(batch_size, config.action_horizon)
    return current, future, state, action_mask


def test_world_model_and_idm_shapes() -> None:
    config = ModelConfig(
        num_views=3,
        num_future_frames=2,
        image_size=32,
        state_dim=4,
        action_dim=4,
        action_horizon=8,
        task_vocab_size=32,
    )
    world_model = ConvVideoWorldModel(config)
    idm = InverseDynamicsModel(config)

    current = torch.rand(3, 3, 3, 32, 32)
    state = torch.rand(3, 4)
    task_id = torch.tensor([0, 1, 2], dtype=torch.long)

    predicted = world_model(current, state, task_id)
    action = idm(current, predicted, state, task_id)

    assert predicted.shape == (3, 2, 3, 3, 32, 32)
    assert action.shape == (3, 8, 4)
    assert float(predicted.min()) >= 0.0
    assert float(predicted.max()) <= 1.0
    assert isinstance(idm.encoder, ResidualTransitionEncoder)
    assert not any(isinstance(module, nn.Tanh) for module in idm.head)


def test_delta_idm_shape_and_architecture() -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=4,
        image_size=32,
        state_dim=4,
        action_dim=4,
        action_horizon=2,
        task_vocab_size=32,
        idm_arch="delta",
    )
    idm = InverseDynamicsModel(config)

    action = idm(
        torch.rand(3, 1, 3, 32, 32),
        torch.rand(3, 4, 1, 3, 32, 32),
        torch.rand(3, 4),
        torch.tensor([0, 1, 2], dtype=torch.long),
    )

    assert action.shape == (3, 2, 4)
    assert isinstance(idm.current_encoder, ResidualTransitionEncoder)
    assert isinstance(idm.future_encoder, ResidualTransitionEncoder)
    assert isinstance(idm.delta_encoder, ResidualTransitionEncoder)


def test_transformer_idm_shape_and_architecture() -> None:
    config = ModelConfig(
        num_views=2,
        num_future_frames=3,
        image_size=32,
        state_dim=4,
        action_dim=4,
        action_horizon=5,
        task_vocab_size=32,
        latent_dim=64,
        idm_arch="transformer",
        idm_transformer_layers=2,
        idm_transformer_heads=4,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
    )
    idm = InverseDynamicsModel(config)

    action = idm(
        torch.rand(3, 2, 3, 32, 32),
        torch.rand(3, 3, 2, 3, 32, 32),
        torch.rand(3, 4),
        torch.tensor([0, 1, 2], dtype=torch.long),
    )

    assert action.shape == (3, 5, 4)
    assert isinstance(idm.transition_encoder, TransformerTransitionEncoder)
    assert idm.transition_encoder.num_patches == 4


@pytest.mark.parametrize("arch", ["stacked", "delta", "transformer", "flow_transformer"])
def test_idm_rejects_temporal_state_tensor(arch) -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=2,
        image_size=32,
        state_dim=4,
        action_dim=4,
        action_horizon=3,
        latent_dim=64,
        idm_arch=arch,
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
        idm_flow_sampling_steps=2,
    )
    idm = InverseDynamicsModel(config)

    with pytest.raises(ValueError, match="state must have shape"):
        idm(
            torch.rand(2, 1, 3, 32, 32),
            torch.rand(2, 2, 1, 3, 32, 32),
            torch.rand(2, 2, 4),
            torch.tensor([0, 1], dtype=torch.long),
        )


def test_flow_transformer_idm_loss_and_sample_shape() -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=2,
        image_size=32,
        state_dim=4,
        action_dim=4,
        action_horizon=6,
        task_vocab_size=32,
        latent_dim=64,
        idm_arch="flow_transformer",
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
        idm_flow_sampling_steps=2,
        idm_flow_num_samples=3,
    )
    idm = InverseDynamicsModel(config)
    current = torch.rand(3, 1, 3, 32, 32)
    future = torch.rand(3, 2, 1, 3, 32, 32)
    state = torch.rand(3, 4)
    task_id = torch.tensor([0, 1, 2], dtype=torch.long)
    target_action = torch.rand(3, 6, 4)
    action_mask = torch.ones(3, 6)

    sampled = idm(current, future, state, task_id)
    loss = idm(
        current,
        future,
        state,
        task_id,
        target_action=target_action,
        action_mask=action_mask,
        mode="loss",
    )
    context_loss = idm.context_action_loss(current, future, state, target_action, action_mask)

    assert sampled.shape == (3, 6, 4)
    assert isinstance(loss, dict)
    assert loss["loss"].ndim == 0
    assert loss["flow_loss"].ndim == 0
    assert loss["endpoint_loss"].ndim == 0
    assert loss["predicted_velocity"].shape == (3, 6, 4)
    assert loss["endpoint_prediction"].shape == (3, 6, 4)
    assert context_loss["loss"].ndim == 0
    assert context_loss["predicted_action"].shape == (3, 6, 4)
    assert idm.uses_flow_matching
    assert idm.head is None
    assert config.idm_flow_num_samples == 3


def test_model_config_rejects_negative_flow_sample_noise_scale() -> None:
    with pytest.raises(ValueError, match="idm_flow_sample_noise_scale"):
        ModelConfig(idm_flow_sample_noise_scale=-0.1)


def test_model_config_rejects_negative_flow_zero_start_endpoint_loss_weight() -> None:
    with pytest.raises(ValueError, match="idm_flow_zero_start_endpoint_loss_weight"):
        ModelConfig(idm_flow_zero_start_endpoint_loss_weight=-0.1)


def test_model_config_rejects_negative_flow_sampled_action_loss_weight() -> None:
    with pytest.raises(ValueError, match="idm_flow_sampled_action_loss_weight"):
        ModelConfig(idm_flow_sampled_action_loss_weight=-0.1)


def test_create_flow_sample_noise_uses_configured_scale() -> None:
    config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=4,
        action_dim=2,
        action_horizon=3,
        num_future_frames=1,
        latent_dim=32,
        idm_arch="flow_transformer",
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        idm_transformer_patch_size=4,
        idm_flow_num_samples=2,
        idm_flow_sample_noise_scale=0.0,
    )
    idm = InverseDynamicsModel(config)
    generator = torch.Generator(device="cpu").manual_seed(123)

    zero_noise = create_flow_sample_noise(
        idm,
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=generator,
    )

    assert torch.count_nonzero(zero_noise) == 0

    scaled_config = dataclasses.replace(config, idm_flow_sample_noise_scale=0.25)
    scaled_idm = InverseDynamicsModel(scaled_config)
    scaled_generator = torch.Generator(device="cpu").manual_seed(123)
    expected_generator = torch.Generator(device="cpu").manual_seed(123)

    scaled_noise = create_flow_sample_noise(
        scaled_idm,
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=scaled_generator,
    )
    expected = torch.randn(
        4,
        scaled_config.action_horizon,
        scaled_config.action_dim,
        generator=expected_generator,
    ) * 0.25

    assert torch.allclose(scaled_noise, expected)


def test_flow_transformer_default_sample_noise_scale_zero_starts_from_zeros() -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=2,
        image_size=32,
        state_dim=4,
        action_dim=4,
        action_horizon=6,
        task_vocab_size=32,
        latent_dim=64,
        idm_arch="flow_transformer",
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
        idm_flow_sampling_steps=1,
        idm_flow_num_samples=2,
        idm_flow_sample_noise_scale=0.0,
    )
    idm = InverseDynamicsModel(config).eval()
    flow_head = _RecordingFlowHead(torch.zeros(6, 6, 4))
    idm.flow_head = flow_head

    with torch.no_grad():
        sampled = idm(
            torch.rand(3, 1, 3, 32, 32),
            torch.rand(3, 2, 1, 3, 32, 32),
            torch.rand(3, 4),
        )

    assert sampled.shape == (3, 6, 4)
    assert flow_head.noisy_action is not None
    assert torch.count_nonzero(flow_head.noisy_action) == 0


def test_idm_future_conditioning_config_validation() -> None:
    with pytest.raises(ValueError, match="current_only.*flow_transformer"):
        ModelConfig(idm_arch="stacked", idm_future_conditioning="current_only")

    with pytest.raises(ValueError, match="future_only.*flow_transformer"):
        ModelConfig(idm_arch="stacked", idm_future_conditioning="future_only")

    with pytest.raises(ValueError, match="current_only"):
        ModelConfig(
            idm_arch="flow_transformer",
            idm_future_conditioning="current_only",
            idm_flow_visual_token_conditioning=True,
        )

    config = ModelConfig(idm_arch="flow_transformer", idm_future_conditioning="future_only")
    assert config.idm_future_conditioning == "future_only"

    wan_token_config = ModelConfig(
        idm_arch="flow_transformer",
        idm_visual_encoder="wan_vae",
        idm_future_conditioning="future_only",
        idm_flow_visual_token_conditioning=True,
    )
    assert wan_token_config.idm_future_conditioning == "future_only"

    with pytest.raises(ValueError, match="idm_future_conditioning"):
        ModelConfig(idm_arch="flow_transformer", idm_future_conditioning="future")


def test_flow_visual_token_scope_config_validation() -> None:
    with pytest.raises(ValueError, match="idm_flow_visual_token_scope"):
        ModelConfig(idm_flow_visual_token_scope="future")

    with pytest.raises(ValueError, match="idm_flow_visual_token_scope='future_only'.*conditioning=True"):
        ModelConfig(
            idm_arch="flow_transformer",
            idm_visual_encoder="wan_vae",
            idm_flow_visual_token_scope="future_only",
        )

    with pytest.raises(ValueError, match="latent_frames=1"):
        ModelConfig(
            num_future_frames=1,
            idm_arch="flow_transformer",
            idm_visual_encoder="wan_vae",
            idm_flow_visual_token_conditioning=True,
            idm_flow_visual_token_scope="future_only",
        )


def test_flow_visual_token_representation_config_validation() -> None:
    assert ModelConfig().idm_flow_visual_token_representation == "encoded"

    with pytest.raises(ValueError, match="idm_flow_visual_token_representation"):
        ModelConfig(idm_flow_visual_token_representation="sideways")

    with pytest.raises(ValueError, match="future_delta.*conditioning=True"):
        ModelConfig(idm_flow_visual_token_representation="future_delta")

    patch_future_delta_config = ModelConfig(
        idm_arch="flow_transformer",
        idm_visual_encoder="patch",
        idm_flow_visual_token_conditioning=True,
        idm_flow_visual_token_scope="future_only",
        idm_flow_visual_token_representation="future_delta",
    )
    assert patch_future_delta_config.idm_flow_visual_token_representation == "future_delta"

    with pytest.raises(ValueError, match="idm_arch='flow_transformer'"):
        ModelConfig(
            num_future_frames=4,
            idm_arch="stacked",
            idm_visual_encoder="wan_vae",
            idm_flow_visual_token_conditioning=True,
            idm_flow_visual_token_scope="future_only",
            idm_flow_visual_token_representation="future_delta",
        )

    with pytest.raises(ValueError, match="scope='future_only'"):
        ModelConfig(
            num_future_frames=4,
            idm_arch="flow_transformer",
            idm_visual_encoder="wan_vae",
            idm_flow_visual_token_conditioning=True,
            idm_flow_visual_token_representation="future_delta",
        )

    with pytest.raises(ValueError, match="latent_frames=1"):
        ModelConfig(
            num_future_frames=1,
            idm_arch="flow_transformer",
            idm_visual_encoder="wan_vae",
            idm_flow_visual_token_conditioning=True,
            idm_flow_visual_token_scope="future_only",
            idm_flow_visual_token_representation="future_delta",
        )


def test_current_only_flow_transformer_ignores_future_images() -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=2,
        image_size=32,
        state_dim=4,
        action_dim=4,
        action_horizon=4,
        latent_dim=64,
        idm_arch="flow_transformer",
        idm_future_conditioning="current_only",
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
        idm_flow_sampling_steps=2,
    )
    idm = InverseDynamicsModel(config).eval()
    current = torch.rand(2, 1, 3, 32, 32)
    state = torch.rand(2, 4)
    future_a = torch.zeros(2, 2, 1, 3, 32, 32)
    future_b = torch.rand(2, 2, 1, 3, 32, 32)
    sample_noise = torch.rand(2, 4, 4)

    with torch.no_grad():
        action_a = idm(current, future_a, state, sample_noise=sample_noise)
        action_b = idm(current, future_b, state, sample_noise=sample_noise)

    assert isinstance(idm.transition_encoder, CurrentOnlyTransitionEncoder)
    assert torch.equal(action_a, action_b)


def test_current_only_flow_transformer_rejects_wan_vae_latents() -> None:
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
        idm_future_conditioning="current_only",
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
        idm_flow_sampling_steps=2,
        wan_vae_use_cached_latents=True,
    )
    idm = InverseDynamicsModel(config)

    with pytest.raises(ValueError, match="current_only"):
        idm(
            torch.rand(2, 1, 3, 32, 32),
            torch.rand(2, 4, 1, 3, 32, 32),
            torch.rand(2, 4),
            wan_vae_latents=torch.rand(2, 48, 2, 2, 2),
        )


def test_future_only_patch_flow_transformer_ignores_current_state_and_history_but_uses_future() -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=2,
        image_size=16,
        state_dim=4,
        action_dim=2,
        action_horizon=3,
        latent_dim=8,
        idm_arch="flow_transformer",
        idm_future_conditioning="future_only",
        idm_history_length=2,
        idm_transformer_layers=1,
        idm_transformer_heads=2,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
        idm_flow_sampling_steps=1,
    )
    idm = InverseDynamicsModel(config).eval()
    encoder = _InputSummaryTransitionEncoder(config.latent_dim)
    flow_head = _ContextEchoFlowHead()
    idm.transition_encoder = encoder
    idm.flow_head = flow_head
    batch_size = 2
    current_a = torch.rand(batch_size, 1, 3, 16, 16)
    current_b = torch.rand(batch_size, 1, 3, 16, 16) + 10.0
    future_a = torch.rand(batch_size, 2, 1, 3, 16, 16)
    future_b = future_a + 1.0
    state_a = torch.rand(batch_size, 4)
    state_b = torch.rand(batch_size, 4) + 10.0
    history_a = {
        "prev_state_history": torch.rand(batch_size, 2, 4),
        "prev_action_history": torch.rand(batch_size, 2, 2),
        "history_mask": torch.ones(batch_size, 2),
    }
    history_b = {
        "prev_state_history": torch.rand(batch_size, 2, 4) + 10.0,
        "prev_action_history": torch.rand(batch_size, 2, 2) + 10.0,
        "history_mask": torch.zeros(batch_size, 2),
    }
    sample_noise = torch.zeros(batch_size, config.action_horizon, config.action_dim)

    with torch.no_grad():
        action_a = idm(current_a, future_a, state_a, sample_noise=sample_noise, **history_a)
        action_b = idm(current_b, future_a, state_b, sample_noise=sample_noise, **history_b)
        action_without_history = idm(current_b, future_a, state_b, sample_noise=sample_noise)
        action_future_b = idm(current_b, future_b, state_b, sample_noise=sample_noise, **history_b)

    assert torch.allclose(action_a, action_b)
    assert torch.allclose(action_a, action_without_history)
    assert not torch.allclose(action_a, action_future_b)
    assert all(call["current_sum"] == pytest.approx(0.0) for call in encoder.calls)
    assert all(call["state_sum"] == pytest.approx(0.0) for call in encoder.calls)
    assert flow_head.history_tokens == [None, None, None, None]

    with pytest.raises(ValueError, match="IDM history conditioning requires"):
        idm(
            current_a,
            future_a,
            state_a,
            sample_noise=sample_noise,
            prev_state_history=history_a["prev_state_history"],
        )


def test_future_only_patch_flow_transformer_rejects_wan_vae_latents() -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=2,
        image_size=16,
        state_dim=4,
        action_dim=2,
        action_horizon=3,
        latent_dim=8,
        idm_arch="flow_transformer",
        idm_future_conditioning="future_only",
        idm_transformer_layers=1,
        idm_transformer_heads=2,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
        idm_flow_sampling_steps=1,
    )
    idm = InverseDynamicsModel(config)

    with pytest.raises(ValueError, match="not using idm_visual_encoder='wan_vae'"):
        idm(
            torch.rand(2, 1, 3, 16, 16),
            torch.rand(2, 2, 1, 3, 16, 16),
            torch.rand(2, 4),
            wan_vae_latents=torch.rand(2, 48, 1, 1, 1),
        )


def test_future_only_wan_vae_flow_transformer_ignores_current_state_history_and_current_latents() -> None:
    config = _wan_token_test_config(
        latent_dim=8,
        action_dim=2,
        action_horizon=3,
        idm_future_conditioning="future_only",
        idm_history_length=2,
        idm_transformer_heads=2,
        idm_transformer_ff_dim=16,
        idm_flow_sampling_steps=1,
        wan_vae_latent_channels=2,
    )
    idm = InverseDynamicsModel(config).eval()
    encoder = _InputSummaryTransitionEncoder(config.latent_dim)
    flow_head = _ContextEchoFlowHead()
    idm.transition_encoder = encoder
    idm.flow_head = flow_head
    batch_size = 2
    current_a = torch.rand(batch_size, 1, 3, config.image_size, config.image_size)
    current_b = torch.rand(batch_size, 1, 3, config.image_size, config.image_size) + 10.0
    future = torch.rand(batch_size, config.num_future_frames, 1, 3, config.image_size, config.image_size)
    state_a = torch.rand(batch_size, config.state_dim)
    state_b = torch.rand(batch_size, config.state_dim) + 10.0
    latents_a = _wan_latents(config, batch_size)
    latents_b = latents_a.clone()
    latents_b[:, :, 0] = latents_b[:, :, 0] + 100.0
    latents_future_b = latents_a.clone()
    latents_future_b[:, :, 1:] = latents_future_b[:, :, 1:] + 3.0
    history_a = {
        "prev_state_history": torch.rand(batch_size, 2, config.state_dim),
        "prev_action_history": torch.rand(batch_size, 2, config.action_dim),
        "history_mask": torch.ones(batch_size, 2),
    }
    history_b = {
        "prev_state_history": torch.rand(batch_size, 2, config.state_dim) + 10.0,
        "prev_action_history": torch.rand(batch_size, 2, config.action_dim) + 10.0,
        "history_mask": torch.zeros(batch_size, 2),
    }
    sample_noise = torch.zeros(batch_size, config.action_horizon, config.action_dim)

    with torch.no_grad():
        action_a = idm(
            current_a,
            future,
            state_a,
            sample_noise=sample_noise,
            wan_vae_latents=latents_a,
            **history_a,
        )
        action_b = idm(
            current_b,
            future,
            state_b,
            sample_noise=sample_noise,
            wan_vae_latents=latents_b,
            **history_b,
        )
        action_without_history = idm(
            current_b,
            future,
            state_b,
            sample_noise=sample_noise,
            wan_vae_latents=latents_b,
        )
        action_future_b = idm(
            current_b,
            future,
            state_b,
            sample_noise=sample_noise,
            wan_vae_latents=latents_future_b,
            **history_b,
        )

    assert torch.allclose(action_a, action_b)
    assert torch.allclose(action_a, action_without_history)
    assert not torch.allclose(action_a, action_future_b)
    assert all(call["current_sum"] == pytest.approx(0.0) for call in encoder.calls)
    assert all(call["state_sum"] == pytest.approx(0.0) for call in encoder.calls)
    assert all(call["current_latent_sum"] == pytest.approx(0.0) for call in encoder.calls)
    assert flow_head.history_tokens == [None, None, None, None]


def test_future_only_wan_vae_image_path_zeros_current_frame_before_encoding() -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=4,
        image_size=32,
        state_dim=4,
        action_dim=2,
        action_horizon=3,
        latent_dim=8,
        idm_arch="flow_transformer",
        idm_visual_encoder="wan_vae",
        idm_future_conditioning="future_only",
        idm_transformer_layers=1,
        idm_transformer_heads=2,
        idm_transformer_dropout=0.0,
        idm_transformer_ff_dim=16,
        idm_flow_sampling_steps=1,
        wan_vae_latent_channels=2,
    )
    wan_encoder = _RecordingWanEncoder(config)
    encoder = WanVaeTransitionEncoder(config, wan_encoder=wan_encoder).eval()
    current = torch.ones(2, 1, 3, 32, 32)
    future = torch.rand(2, 4, 1, 3, 32, 32)
    state = torch.rand(2, 4)

    with torch.no_grad():
        context = encoder(current, future, state)

    video = wan_encoder.videos[0]
    expected_future_video = future[:, :, 0].permute(0, 2, 1, 3, 4).mul(2.0).sub(1.0)
    assert context.shape == (2, config.latent_dim)
    assert torch.equal(video[:, :, 0], torch.full_like(video[:, :, 0], -1.0))
    assert torch.allclose(video[:, :, 1:], expected_future_video)


def test_flow_transformer_history_zero_has_no_history_parameters() -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=1,
        image_size=32,
        state_dim=4,
        action_dim=4,
        action_horizon=4,
        latent_dim=64,
        idm_arch="flow_transformer",
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
    )
    idm = InverseDynamicsModel(config)

    assert config.idm_history_length == 0
    assert not any(key.startswith("history_") for key in idm.state_dict())


@pytest.mark.parametrize("visual_encoder", ["patch", "wan_vae"])
def test_flow_transformer_idm_consumes_history_for_patch_and_wan_vae(visual_encoder: str) -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=4 if visual_encoder == "wan_vae" else 2,
        image_size=32,
        state_dim=4,
        action_dim=4,
        action_horizon=4,
        task_vocab_size=32,
        latent_dim=64,
        idm_arch="flow_transformer",
        idm_visual_encoder=visual_encoder,
        idm_history_length=2,
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
        idm_flow_sampling_steps=2,
        wan_vae_use_cached_latents=visual_encoder == "wan_vae",
    )
    idm = InverseDynamicsModel(config)
    batch_size = 3
    current = torch.rand(batch_size, 1, 3, 32, 32)
    future = torch.rand(batch_size, config.num_future_frames, 1, 3, 32, 32)
    state = torch.rand(batch_size, 4)
    target_action = torch.rand(batch_size, 4, 4)
    action_mask = torch.ones(batch_size, 4)
    prev_state_history = torch.rand(batch_size, 2, 4)
    prev_action_history = torch.rand(batch_size, 2, 4)
    history_mask = torch.ones(batch_size, 2)
    kwargs = {
        "prev_state_history": prev_state_history,
        "prev_action_history": prev_action_history,
        "history_mask": history_mask,
    }
    if visual_encoder == "wan_vae":
        kwargs["wan_vae_latents"] = torch.rand(batch_size, 48, 2, 2, 2)

    loss = idm(
        current,
        future,
        state,
        torch.tensor([0, 1, 2], dtype=torch.long),
        target_action=target_action,
        action_mask=action_mask,
        mode="loss",
        **kwargs,
    )
    sampled = idm(current, future, state, sample_noise=torch.randn(batch_size, 4, 4), **kwargs)

    assert loss["predicted_velocity"].shape == (batch_size, 4, 4)
    assert sampled.shape == (batch_size, 4, 4)


def test_flow_transformer_history_requires_explicit_batch_tensors() -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=1,
        image_size=32,
        state_dim=4,
        action_dim=4,
        action_horizon=4,
        latent_dim=64,
        idm_arch="flow_transformer",
        idm_history_length=2,
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
    )
    idm = InverseDynamicsModel(config)

    with pytest.raises(ValueError, match="IDM history conditioning requires"):
        idm(
            torch.rand(2, 1, 3, 32, 32),
            torch.rand(2, 1, 1, 3, 32, 32),
            torch.rand(2, 4),
            target_action=torch.rand(2, 4, 4),
            action_mask=torch.ones(2, 4),
            mode="loss",
        )


def test_flow_transformer_default_train_time_range_matches_unit_uniform() -> None:
    config = _flow_time_test_config()
    target_action = torch.arange(12, dtype=torch.float32).view(2, 3, 2) / 10.0
    idm = InverseDynamicsModel(config).eval()
    idm.transition_encoder = _ConstantTransitionEncoder(config.latent_dim)
    flow_head = _RecordingFlowHead(torch.zeros_like(target_action))
    idm.flow_head = flow_head
    current, future, state, action_mask = _flow_time_loss_inputs(config)

    torch.manual_seed(321)
    expected_noise = torch.randn_like(target_action)
    expected_time = torch.rand(target_action.shape[0])
    expected_noisy_action = (1.0 - expected_time.view(-1, 1, 1)) * expected_noise
    expected_noisy_action = expected_noisy_action + expected_time.view(-1, 1, 1) * target_action

    torch.manual_seed(321)
    idm.flow_matching_loss(current, future, state, target_action, action_mask)

    assert flow_head.time is not None
    assert flow_head.noisy_action is not None
    assert torch.allclose(flow_head.time, expected_time)
    assert torch.allclose(flow_head.noisy_action, expected_noisy_action)


@pytest.mark.parametrize("time_value", [0.0, 1.0], ids=["zero", "one"])
def test_flow_transformer_fixed_train_time_endpoints(monkeypatch, time_value: float) -> None:
    config = _flow_time_test_config(
        idm_flow_train_time_min=time_value,
        idm_flow_train_time_max=time_value,
    )
    target_action = torch.arange(12, dtype=torch.float32).view(2, 3, 2) / 10.0
    noise = torch.linspace(-0.5, 0.6, steps=12).view(2, 3, 2)
    target_velocity = target_action - noise
    idm = InverseDynamicsModel(config).eval()
    idm.transition_encoder = _ConstantTransitionEncoder(config.latent_dim)
    flow_head = _RecordingFlowHead(target_velocity)
    idm.flow_head = flow_head
    current, future, state, action_mask = _flow_time_loss_inputs(config)

    def fake_randn_like(tensor):
        assert tuple(tensor.shape) == tuple(noise.shape)
        return noise.to(device=tensor.device, dtype=tensor.dtype)

    monkeypatch.setattr(torch, "randn_like", fake_randn_like)
    loss = idm.flow_matching_loss(current, future, state, target_action, action_mask)

    assert flow_head.time is not None
    assert flow_head.noisy_action is not None
    assert torch.allclose(flow_head.time, torch.full((2,), time_value))
    if time_value == 0.0:
        assert torch.allclose(flow_head.noisy_action, noise)
    else:
        assert torch.allclose(flow_head.noisy_action, target_action)
    assert torch.allclose(loss["predicted_velocity"], target_velocity)
    assert torch.allclose(loss["flow_loss"], target_action.new_tensor(0.0))
    assert torch.allclose(loss["endpoint_loss"], target_action.new_tensor(0.0))
    assert torch.allclose(loss["endpoint_consistency_loss"], target_action.new_tensor(0.0))
    assert len(flow_head.calls) == 1


def test_flow_transformer_endpoint_consistency_loss_uses_second_noise_draw(monkeypatch) -> None:
    config = _flow_time_test_config(
        idm_flow_train_time_min=0.0,
        idm_flow_train_time_max=0.0,
        idm_flow_endpoint_consistency_loss_weight=0.25,
    )
    target_action = torch.zeros(2, 3, 2)
    noise = torch.full_like(target_action, 1.0)
    noise_2 = torch.full_like(target_action, 3.0)
    idm = InverseDynamicsModel(config).eval()
    idm.transition_encoder = _ConstantTransitionEncoder(config.latent_dim)
    flow_head = _RecordingFlowHead(torch.zeros_like(target_action))
    idm.flow_head = flow_head
    current, future, state, action_mask = _flow_time_loss_inputs(config)
    noise_draws = [noise, noise_2]

    def fake_randn_like(tensor):
        assert tuple(tensor.shape) == tuple(target_action.shape)
        return noise_draws.pop(0).to(device=tensor.device, dtype=tensor.dtype)

    monkeypatch.setattr(torch, "randn_like", fake_randn_like)

    loss = idm.flow_matching_loss(current, future, state, target_action, action_mask)

    assert len(noise_draws) == 0
    assert len(flow_head.calls) == 2
    assert torch.allclose(flow_head.calls[0][0], noise)
    assert torch.allclose(flow_head.calls[1][0], noise_2)
    assert torch.allclose(loss["flow_loss"], target_action.new_tensor(1.0))
    assert torch.allclose(loss["endpoint_loss"], target_action.new_tensor(1.0))
    assert torch.allclose(loss["endpoint_consistency_loss"], target_action.new_tensor(4.0))
    assert torch.allclose(loss["loss"], target_action.new_tensor(2.0))


def test_flow_transformer_zero_start_endpoint_loss_defaults_to_zero(monkeypatch) -> None:
    config = _flow_time_test_config(
        idm_flow_train_time_min=0.0,
        idm_flow_train_time_max=0.0,
    )
    target_action = torch.zeros(2, 3, 2)
    noise = torch.full_like(target_action, 1.0)
    idm = InverseDynamicsModel(config).eval()
    idm.transition_encoder = _ConstantTransitionEncoder(config.latent_dim)
    flow_head = _RecordingFlowHead(torch.full_like(target_action, 2.0))
    idm.flow_head = flow_head
    current, future, state, action_mask = _flow_time_loss_inputs(config)

    def fake_randn_like(tensor):
        return noise.to(device=tensor.device, dtype=tensor.dtype)

    monkeypatch.setattr(torch, "randn_like", fake_randn_like)
    loss = idm.flow_matching_loss(current, future, state, target_action, action_mask)

    # Default weight leaves behavior identical: no extra forward and a zero metric.
    assert len(flow_head.calls) == 1
    assert torch.allclose(loss["zero_start_endpoint_loss"], target_action.new_tensor(0.0))


def test_flow_transformer_zero_start_endpoint_loss_supervises_zero_start(monkeypatch) -> None:
    config = _flow_time_test_config(
        idm_flow_train_time_min=0.0,
        idm_flow_train_time_max=0.0,
        idm_flow_zero_start_endpoint_loss_weight=0.5,
    )
    target_action = torch.zeros(2, 3, 2)
    noise = torch.full_like(target_action, 1.0)
    idm = InverseDynamicsModel(config).eval()
    idm.transition_encoder = _ConstantTransitionEncoder(config.latent_dim)
    flow_head = _RecordingFlowHead(torch.full_like(target_action, 2.0))
    idm.flow_head = flow_head
    current, future, state, action_mask = _flow_time_loss_inputs(config)

    def fake_randn_like(tensor):
        return noise.to(device=tensor.device, dtype=tensor.dtype)

    monkeypatch.setattr(torch, "randn_like", fake_randn_like)
    loss = idm.flow_matching_loss(current, future, state, target_action, action_mask)

    # An extra deterministic forward runs from action zeros at time 0.
    assert len(flow_head.calls) == 2
    zero_noisy_action, zero_time = flow_head.calls[1]
    assert torch.allclose(zero_noisy_action, torch.zeros_like(target_action))
    assert torch.allclose(zero_time, torch.zeros(2))
    # Endpoint from a zero start equals the predicted velocity (2.0); target is 0 -> mse 4.0.
    assert torch.allclose(loss["zero_start_endpoint_loss"], target_action.new_tensor(4.0))
    # flow_loss = mse(velocity=2, target_velocity=target-noise=-1) = 9.0.
    assert torch.allclose(loss["flow_loss"], target_action.new_tensor(9.0))
    # total = flow_loss + weight * zero_start_endpoint_loss = 9.0 + 0.5 * 4.0 = 11.0.
    assert torch.allclose(loss["loss"], target_action.new_tensor(11.0))


def test_flow_transformer_sampled_action_loss_defaults_to_zero(monkeypatch) -> None:
    config = _flow_time_test_config(
        idm_flow_train_time_min=0.0,
        idm_flow_train_time_max=0.0,
        idm_flow_sampling_steps=2,
    )
    target_action = torch.zeros(2, 3, 2)
    noise = torch.full_like(target_action, 1.0)
    idm = InverseDynamicsModel(config).eval()
    idm.transition_encoder = _ConstantTransitionEncoder(config.latent_dim)
    flow_head = _ConstantVelocityFlowHead(2.0)
    idm.flow_head = flow_head
    current, future, state, action_mask = _flow_time_loss_inputs(config)

    def fake_randn_like(tensor):
        return noise.to(device=tensor.device, dtype=tensor.dtype)

    monkeypatch.setattr(torch, "randn_like", fake_randn_like)
    loss = idm.flow_matching_loss(current, future, state, target_action, action_mask)

    # Default weight skips the sampler rollout: only the single flow forward runs.
    assert len(flow_head.calls) == 1
    assert torch.allclose(loss["sampled_action_loss"], target_action.new_tensor(0.0))


def test_flow_transformer_sampled_action_loss_supervises_sampled_actions(monkeypatch) -> None:
    config = _flow_time_test_config(
        idm_flow_train_time_min=0.0,
        idm_flow_train_time_max=0.0,
        idm_flow_sampling_steps=2,
        idm_flow_sampled_action_loss_weight=0.5,
    )
    target_action = torch.zeros(2, 3, 2)
    noise = torch.full_like(target_action, 1.0)
    idm = InverseDynamicsModel(config).eval()
    idm.transition_encoder = _ConstantTransitionEncoder(config.latent_dim)
    flow_head = _ConstantVelocityFlowHead(2.0)
    idm.flow_head = flow_head
    current, future, state, action_mask = _flow_time_loss_inputs(config)

    def fake_randn_like(tensor):
        return noise.to(device=tensor.device, dtype=tensor.dtype)

    monkeypatch.setattr(torch, "randn_like", fake_randn_like)
    loss = idm.flow_matching_loss(current, future, state, target_action, action_mask)

    # One flow forward plus one per deterministic sampling step from a zero start.
    assert len(flow_head.calls) == 1 + config.idm_flow_sampling_steps
    first_sample_action, _ = flow_head.calls[1]
    # The sampler starts from explicit zeros even though noise_scale defaults to 1.0.
    assert torch.allclose(first_sample_action, torch.zeros_like(target_action))
    # Zero-noise rollout integrates a constant velocity of 2.0 to 2.0; target 0 -> mse 4.0.
    assert torch.allclose(loss["sampled_action_loss"], target_action.new_tensor(4.0))
    # flow_loss = mse(velocity=2, target_velocity=target-noise=-1) = 9.0.
    assert torch.allclose(loss["flow_loss"], target_action.new_tensor(9.0))
    # total = flow_loss + weight * sampled_action_loss = 9.0 + 0.5 * 4.0 = 11.0.
    assert torch.allclose(loss["loss"], target_action.new_tensor(11.0))


def test_flow_transformer_sampled_action_loss_uses_num_samples_shape(monkeypatch) -> None:
    config = _flow_time_test_config(
        idm_flow_train_time_min=0.0,
        idm_flow_train_time_max=0.0,
        idm_flow_sampling_steps=2,
        idm_flow_num_samples=2,
        idm_flow_sampled_action_loss_weight=0.5,
    )
    target_action = torch.zeros(2, 3, 2)
    noise = torch.full_like(target_action, 1.0)
    idm = InverseDynamicsModel(config).eval()
    idm.transition_encoder = _ConstantTransitionEncoder(config.latent_dim)
    flow_head = _ConstantVelocityFlowHead(2.0)
    idm.flow_head = flow_head
    current, future, state, action_mask = _flow_time_loss_inputs(config)

    def fake_randn_like(tensor):
        return noise.to(device=tensor.device, dtype=tensor.dtype)

    monkeypatch.setattr(torch, "randn_like", fake_randn_like)
    loss = idm.flow_matching_loss(current, future, state, target_action, action_mask)

    # The deterministic rollout covers (batch * idm_flow_num_samples) zero-noise trajectories.
    first_sample_action, _ = flow_head.calls[1]
    assert tuple(first_sample_action.shape) == (
        target_action.shape[0] * config.idm_flow_num_samples,
        config.action_horizon,
        config.action_dim,
    )
    assert torch.allclose(loss["sampled_action_loss"], target_action.new_tensor(4.0))


def test_flow_transformer_default_context_conditioning_matches_explicit_token_mode() -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=1,
        image_size=32,
        state_dim=4,
        action_dim=3,
        action_horizon=5,
        latent_dim=16,
        idm_arch="flow_transformer",
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
        idm_transformer_ff_dim=32,
    )
    token_config = dataclasses.replace(config, idm_flow_context_conditioning="token")

    torch.manual_seed(123)
    default_head = FlowActionTransformerHead(config).eval()
    torch.manual_seed(123)
    token_head = FlowActionTransformerHead(token_config).eval()
    context = torch.randn(2, 16)
    noisy_action = torch.randn(2, 5, 3)
    time = torch.tensor([0.25, 0.75])

    with torch.no_grad():
        default_velocity = default_head(context, noisy_action, time)
        token_velocity = token_head(context, noisy_action, time)

    assert config.idm_flow_context_conditioning == "token"
    assert default_head.state_dict().keys() == token_head.state_dict().keys()
    assert not hasattr(default_head, "context_action_projection")
    assert not hasattr(default_head, "context_velocity_head")
    assert torch.allclose(default_velocity, token_velocity)


def test_flow_transformer_additive_context_conditioning_changes_output_when_context_changes() -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=1,
        image_size=32,
        state_dim=4,
        action_dim=2,
        action_horizon=3,
        latent_dim=4,
        idm_arch="flow_transformer",
        idm_transformer_layers=1,
        idm_transformer_heads=2,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
        idm_transformer_ff_dim=8,
        idm_flow_context_conditioning="additive",
    )
    head = FlowActionTransformerHead(config).eval()
    for parameter in head.parameters():
        parameter.data.zero_()
    with torch.no_grad():
        head.context_velocity_head.weight[0, 0] = 1.0

    noisy_action = torch.zeros(2, 3, 2)
    time = torch.zeros(2)
    first_context = torch.zeros(2, 4)
    second_context = torch.zeros(2, 4)
    second_context[:, 0] = torch.tensor([1.0, 2.0])

    with torch.no_grad():
        first_velocity = head(first_context, noisy_action, time)
        second_velocity = head(second_context, noisy_action, time)

    assert torch.allclose(first_velocity, torch.zeros_like(first_velocity))
    assert torch.allclose(second_velocity[:, :, 0], second_context[:, :1].expand(-1, 3))
    assert torch.count_nonzero(second_velocity - first_velocity) > 0


def test_flow_transformer_rejects_invalid_visual_context_tokens() -> None:
    config = _flow_time_test_config()
    head = FlowActionTransformerHead(config)
    context = torch.randn(2, config.latent_dim)
    noisy_action = torch.randn(2, config.action_horizon, config.action_dim)
    time = torch.zeros(2)

    with pytest.raises(ValueError, match="visual_context_tokens"):
        head(context, noisy_action, time, visual_context_tokens=torch.randn(2, 1, config.latent_dim + 1))

    with pytest.raises(ValueError, match="visual_context_tokens must contain at least one token"):
        head(context, noisy_action, time, visual_context_tokens=torch.randn(2, 0, config.latent_dim))


def test_flow_transformer_visual_token_cross_attention_requires_and_uses_tokens() -> None:
    config = _flow_time_test_config(
        idm_visual_encoder="wan_vae",
        idm_flow_visual_token_conditioning=True,
        idm_flow_visual_token_conditioning_mode="cross_attention",
    )
    torch.manual_seed(123)
    head = FlowActionTransformerHead(config).eval()
    context = torch.randn(2, config.latent_dim)
    noisy_action = torch.randn(2, config.action_horizon, config.action_dim)
    time = torch.tensor([0.25, 0.75])
    first_visual_tokens = torch.zeros(2, 4, config.latent_dim)
    second_visual_tokens = torch.randn(2, 4, config.latent_dim)

    with pytest.raises(ValueError, match="visual_context_tokens are required"):
        head(context, noisy_action, time)

    with torch.no_grad():
        first_velocity = head(context, noisy_action, time, visual_context_tokens=first_visual_tokens)
        second_velocity = head(context, noisy_action, time, visual_context_tokens=second_visual_tokens)

    assert first_velocity.shape == (2, config.action_horizon, config.action_dim)
    assert second_velocity.shape == (2, config.action_horizon, config.action_dim)
    assert not torch.allclose(first_velocity, second_velocity)


def test_model_config_permits_patch_visual_token_conditioning_and_cross_attention() -> None:
    config = _patch_token_test_config(
        idm_flow_visual_token_conditioning=True,
        idm_flow_visual_token_conditioning_mode="cross_attention",
        idm_flow_visual_token_scope="future_only",
    )
    future_delta_config = dataclasses.replace(
        config,
        idm_flow_visual_token_representation="future_delta",
    )

    assert config.idm_visual_encoder == "patch"
    assert config.idm_flow_visual_token_conditioning is True
    assert config.idm_flow_visual_token_conditioning_mode == "cross_attention"
    assert config.idm_flow_visual_token_scope == "future_only"
    assert future_delta_config.idm_flow_visual_token_representation == "future_delta"


def test_patch_transition_encoder_returns_visual_tokens_for_scope_and_representation() -> None:
    config = _patch_token_test_config(idm_flow_visual_token_conditioning=True)
    future_scope_config = dataclasses.replace(config, idm_flow_visual_token_scope="future_only")
    future_delta_config = dataclasses.replace(
        future_scope_config,
        idm_flow_visual_token_representation="future_delta",
    )
    encoder = TransformerTransitionEncoder(config).eval()
    future_scope_encoder = TransformerTransitionEncoder(future_scope_config).eval()
    future_delta_encoder = TransformerTransitionEncoder(future_delta_config).eval()
    batch_size = 2
    current = torch.rand(batch_size, config.num_views, 3, config.image_size, config.image_size)
    future = torch.rand(
        batch_size,
        config.num_future_frames,
        config.num_views,
        3,
        config.image_size,
        config.image_size,
    )
    state = torch.rand(batch_size, config.state_dim)

    with torch.no_grad():
        context, visual_tokens = encoder(current, future, state, return_tokens=True)
        future_scope_context, future_scope_tokens = future_scope_encoder(
            current,
            future,
            state,
            return_tokens=True,
        )
        future_delta_context, future_delta_tokens = future_delta_encoder(
            current,
            future,
            state,
            return_tokens=True,
        )

    current_token_count = config.num_views * encoder.num_patches
    future_token_count = config.num_future_frames * config.num_views * encoder.num_patches
    assert context.shape == (batch_size, config.latent_dim)
    assert future_scope_context.shape == (batch_size, config.latent_dim)
    assert future_delta_context.shape == (batch_size, config.latent_dim)
    assert visual_tokens.shape == (batch_size, current_token_count + 2 * future_token_count, config.latent_dim)
    assert future_scope_tokens.shape == (batch_size, 2 * future_token_count, config.latent_dim)
    assert future_delta_tokens.shape == (batch_size, 2 * future_token_count, config.latent_dim)


def test_flow_transformer_patch_visual_token_conditioning_loss_and_sample_without_task_id() -> None:
    config = _patch_token_test_config(
        idm_flow_visual_token_conditioning=True,
        idm_flow_visual_token_conditioning_mode="cross_attention",
        idm_flow_visual_token_scope="future_only",
    )
    idm = InverseDynamicsModel(config).eval()
    flow_head = _ZeroRecordingFlowHead()
    idm.flow_head = flow_head
    batch_size = 2
    current = torch.rand(batch_size, config.num_views, 3, config.image_size, config.image_size)
    future = torch.rand(
        batch_size,
        config.num_future_frames,
        config.num_views,
        3,
        config.image_size,
        config.image_size,
    )
    state = torch.rand(batch_size, config.state_dim)
    target_action = torch.rand(batch_size, config.action_horizon, config.action_dim)
    action_mask = torch.ones(batch_size, config.action_horizon)
    sample_noise = torch.zeros(batch_size * config.idm_flow_num_samples, config.action_horizon, config.action_dim)

    loss = idm(
        current,
        future,
        state,
        target_action=target_action,
        action_mask=action_mask,
        mode="loss",
    )
    sampled = idm(
        current,
        future,
        state,
        sample_noise=sample_noise,
    )

    future_token_count = config.num_future_frames * config.num_views * idm.transition_encoder.num_patches
    expected_loss_tokens = (batch_size, 2 * future_token_count, config.latent_dim)
    expected_sample_tokens = (batch_size * config.idm_flow_num_samples, 2 * future_token_count, config.latent_dim)
    assert loss["predicted_velocity"].shape == (batch_size, config.action_horizon, config.action_dim)
    assert sampled.shape == (batch_size, config.action_horizon, config.action_dim)
    assert flow_head.calls[0]["visual_context_tokens"] == expected_loss_tokens
    assert [call["visual_context_tokens"] for call in flow_head.calls[1:]] == [expected_sample_tokens] * 2


def test_flow_transformer_patch_visual_token_conditioning_rejects_invalid_shapes_and_options() -> None:
    with pytest.raises(ValueError, match="scope='future_only'"):
        _patch_token_test_config(
            idm_flow_visual_token_conditioning=True,
            idm_flow_visual_token_representation="future_delta",
        )

    config = _patch_token_test_config(idm_flow_visual_token_conditioning=True)
    idm = InverseDynamicsModel(config)
    batch_size = 2
    current = torch.rand(batch_size, config.num_views, 3, config.image_size, config.image_size)
    future = torch.rand(
        batch_size,
        config.num_future_frames,
        config.num_views,
        3,
        config.image_size,
        config.image_size,
    )
    state = torch.rand(batch_size, config.state_dim)
    target_action = torch.rand(batch_size, config.action_horizon, config.action_dim)
    action_mask = torch.ones(batch_size, config.action_horizon)

    with pytest.raises(ValueError, match="wan_vae_latents.*not using idm_visual_encoder='wan_vae'"):
        idm(
            current,
            future,
            state,
            target_action=target_action,
            action_mask=action_mask,
            wan_vae_latents=torch.zeros(batch_size, 1, 1, 1, 1),
            mode="loss",
        )

    with pytest.raises(ValueError, match="future_images must have shape"):
        idm(
            current,
            future[:, :1],
            state,
            target_action=target_action,
            action_mask=action_mask,
            mode="loss",
        )


def test_wan_vae_transition_encoder_uses_frozen_latents() -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=4,
        image_size=32,
        state_dim=4,
        action_dim=4,
        action_horizon=4,
        task_vocab_size=32,
        latent_dim=64,
        idm_arch="flow_transformer",
        idm_visual_encoder="wan_vae",
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
        idm_flow_sampling_steps=2,
    )
    encoder = WanVaeTransitionEncoder(config, wan_encoder=FakeWanVaeEncoder())
    current = torch.rand(3, 1, 3, 32, 32)
    future = torch.rand(3, 4, 1, 3, 32, 32)
    state = torch.rand(3, 4)

    context = encoder(current, future, state)
    token_context, visual_tokens = encoder(current, future, state, return_tokens=True)
    future_scope_config = dataclasses.replace(
        config,
        idm_flow_visual_token_conditioning=True,
        idm_flow_visual_token_scope="future_only",
    )
    future_scope_encoder = WanVaeTransitionEncoder(future_scope_config, wan_encoder=FakeWanVaeEncoder())
    future_scope_context, future_scope_visual_tokens = future_scope_encoder(
        current,
        future,
        state,
        return_tokens=True,
    )
    future_token_count = (future_scope_encoder.latent_frames - 1) * future_scope_encoder.latent_side**2

    assert context.shape == (3, 64)
    assert token_context.shape == (3, 64)
    assert visual_tokens.shape == (3, encoder.num_latent_tokens, 64)
    assert encoder.num_latent_tokens == 8
    assert future_scope_context.shape == (3, 64)
    assert future_scope_visual_tokens.shape == (3, future_token_count, 64)
    assert future_scope_visual_tokens.shape[1] == encoder.num_latent_tokens - encoder.latent_side**2


def test_wan_vae_future_delta_visual_token_representation_counts() -> None:
    encoded_config = _wan_token_test_config(idm_flow_visual_token_conditioning=True)
    future_delta_config = dataclasses.replace(
        encoded_config,
        idm_flow_visual_token_scope="future_only",
        idm_flow_visual_token_representation="future_delta",
    )
    multi_future_delta_config = dataclasses.replace(future_delta_config, num_future_frames=8)
    batch_size = 2
    current = torch.rand(batch_size, 1, 3, encoded_config.image_size, encoded_config.image_size)
    state = torch.rand(batch_size, encoded_config.state_dim)

    encoded_encoder = WanVaeTransitionEncoder(encoded_config).eval()
    encoded_future = torch.rand(
        batch_size,
        encoded_config.num_future_frames,
        1,
        3,
        encoded_config.image_size,
        encoded_config.image_size,
    )
    with torch.no_grad():
        _, encoded_tokens = encoded_encoder(
            current,
            encoded_future,
            state,
            wan_vae_latents=_wan_latents(encoded_config, batch_size),
            return_tokens=True,
        )
    assert encoded_tokens.shape == (batch_size, encoded_encoder.num_latent_tokens, encoded_config.latent_dim)

    future_delta_encoder = WanVaeTransitionEncoder(future_delta_config).eval()
    with torch.no_grad():
        _, future_delta_tokens = future_delta_encoder(
            current,
            encoded_future,
            state,
            wan_vae_latents=_wan_latents(future_delta_config, batch_size),
            return_tokens=True,
        )
    future_token_count = (future_delta_encoder.latent_frames - 1) * future_delta_encoder.latent_side**2
    assert future_delta_tokens.shape == (batch_size, 2 * future_token_count, future_delta_config.latent_dim)

    multi_encoder = WanVaeTransitionEncoder(multi_future_delta_config).eval()
    multi_future = torch.rand(
        batch_size,
        multi_future_delta_config.num_future_frames,
        1,
        3,
        multi_future_delta_config.image_size,
        multi_future_delta_config.image_size,
    )
    with torch.no_grad():
        _, multi_tokens = multi_encoder(
            current,
            multi_future,
            state,
            wan_vae_latents=_wan_latents(multi_future_delta_config, batch_size),
            return_tokens=True,
        )
    multi_future_token_count = (multi_encoder.latent_frames - 1) * multi_encoder.latent_side**2
    assert multi_encoder.latent_frames == 3
    assert multi_tokens.shape == (batch_size, 2 * multi_future_token_count, multi_future_delta_config.latent_dim)


def test_wan_vae_future_delta_visual_token_representation_uses_current_and_future_latents() -> None:
    config = _wan_token_test_config(
        image_size=16,
        latent_dim=2,
        idm_transformer_heads=1,
        idm_transformer_ff_dim=8,
        wan_vae_latent_channels=2,
        idm_flow_visual_token_conditioning=True,
        idm_flow_visual_token_scope="future_only",
        idm_flow_visual_token_representation="future_delta",
    )
    encoder = WanVaeTransitionEncoder(config).eval()
    with torch.no_grad():
        encoder.latent_projection.weight.copy_(torch.eye(2))
        encoder.latent_projection.bias.zero_()
        encoder.transition_token_type_embedding.weight.zero_()

    current = torch.zeros(1, 1, 3, config.image_size, config.image_size)
    future = torch.zeros(1, config.num_future_frames, 1, 3, config.image_size, config.image_size)
    state = torch.zeros(1, config.state_dim)
    latents = torch.zeros(1, 2, 2, 1, 1)
    latents[:, :, 0, 0, 0] = torch.tensor([1.0, 2.0])
    latents[:, :, 1, 0, 0] = torch.tensor([4.0, 8.0])

    with torch.no_grad():
        _, tokens = encoder(current, future, state, wan_vae_latents=latents, return_tokens=True)

    current_changed = latents.clone()
    current_changed[:, :, 0, 0, 0] = torch.tensor([2.0, 3.0])
    future_changed = latents.clone()
    future_changed[:, :, 1, 0, 0] = torch.tensor([5.0, 10.0])
    with torch.no_grad():
        _, current_changed_tokens = encoder(
            current,
            future,
            state,
            wan_vae_latents=current_changed,
            return_tokens=True,
        )
        _, future_changed_tokens = encoder(
            current,
            future,
            state,
            wan_vae_latents=future_changed,
            return_tokens=True,
        )

    assert torch.allclose(tokens[:, :1], torch.tensor([[[4.0, 8.0]]]))
    assert torch.allclose(tokens[:, 1:], torch.tensor([[[3.0, 6.0]]]))
    assert torch.allclose(current_changed_tokens[:, :1], tokens[:, :1])
    assert not torch.allclose(current_changed_tokens[:, 1:], tokens[:, 1:])
    assert not torch.allclose(future_changed_tokens[:, :1], tokens[:, :1])
    assert not torch.allclose(future_changed_tokens[:, 1:], tokens[:, 1:])


def test_flow_transformer_wan_vae_default_does_not_use_visual_tokens_and_keeps_state_dict_keys() -> None:
    config = _wan_token_test_config()
    token_config = dataclasses.replace(config, idm_flow_visual_token_conditioning=True)
    idm = InverseDynamicsModel(config)
    token_idm = InverseDynamicsModel(token_config)
    assert list(idm.state_dict()) == list(token_idm.state_dict())

    batch_size = 2
    current = torch.rand(batch_size, 1, 3, config.image_size, config.image_size)
    future = torch.rand(batch_size, config.num_future_frames, 1, 3, config.image_size, config.image_size)
    state = torch.rand(batch_size, config.state_dim)
    target_action = torch.rand(batch_size, config.action_horizon, config.action_dim)
    action_mask = torch.ones(batch_size, config.action_horizon)
    flow_head = _RecordingFlowHead(torch.zeros_like(target_action))
    idm.flow_head = flow_head

    idm(
        current,
        future,
        state,
        target_action=target_action,
        action_mask=action_mask,
        wan_vae_latents=_wan_latents(config, batch_size),
        mode="loss",
    )

    assert config.idm_flow_visual_token_conditioning is False
    assert token_config.idm_flow_visual_token_conditioning_mode == "prefix"
    assert flow_head.visual_context_tokens is None


def test_flow_transformer_wan_vae_visual_token_conditioning_loss_and_sample_shape() -> None:
    config = _wan_token_test_config(idm_flow_visual_token_conditioning=True)
    idm = InverseDynamicsModel(config)
    flow_head = _ZeroRecordingFlowHead()
    idm.flow_head = flow_head
    batch_size = 2
    current = torch.rand(batch_size, 1, 3, config.image_size, config.image_size)
    future = torch.rand(batch_size, config.num_future_frames, 1, 3, config.image_size, config.image_size)
    state = torch.rand(batch_size, config.state_dim)
    target_action = torch.rand(batch_size, config.action_horizon, config.action_dim)
    action_mask = torch.ones(batch_size, config.action_horizon)
    wan_vae_latents = _wan_latents(config, batch_size)

    loss = idm(
        current,
        future,
        state,
        target_action=target_action,
        action_mask=action_mask,
        wan_vae_latents=wan_vae_latents,
        mode="loss",
    )
    sampled = idm(
        current,
        future,
        state,
        wan_vae_latents=wan_vae_latents,
        sample_noise=torch.zeros(batch_size, config.action_horizon, config.action_dim),
    )

    expected_tokens = (batch_size, idm.transition_encoder.num_latent_tokens, config.latent_dim)
    assert loss["predicted_velocity"].shape == (batch_size, config.action_horizon, config.action_dim)
    assert sampled.shape == (batch_size, config.action_horizon, config.action_dim)
    assert [call["visual_context_tokens"] for call in flow_head.calls] == [expected_tokens] * 3
    assert [call["history_tokens"] for call in flow_head.calls] == [None] * 3


def test_flow_transformer_wan_vae_cross_attention_future_scope_passes_only_future_tokens() -> None:
    config = _wan_token_test_config(
        idm_flow_visual_token_conditioning=True,
        idm_flow_visual_token_conditioning_mode="cross_attention",
        idm_flow_visual_token_scope="future_only",
    )
    idm = InverseDynamicsModel(config)
    flow_head = _ZeroRecordingFlowHead()
    idm.flow_head = flow_head
    batch_size = 2
    current = torch.rand(batch_size, 1, 3, config.image_size, config.image_size)
    future = torch.rand(batch_size, config.num_future_frames, 1, 3, config.image_size, config.image_size)
    state = torch.rand(batch_size, config.state_dim)
    target_action = torch.rand(batch_size, config.action_horizon, config.action_dim)
    action_mask = torch.ones(batch_size, config.action_horizon)
    wan_vae_latents = _wan_latents(config, batch_size)

    idm(
        current,
        future,
        state,
        target_action=target_action,
        action_mask=action_mask,
        wan_vae_latents=wan_vae_latents,
        mode="loss",
    )

    all_token_count = idm.transition_encoder.num_latent_tokens
    future_token_count = (idm.transition_encoder.latent_frames - 1) * idm.transition_encoder.latent_side**2
    assert future_token_count < all_token_count
    assert flow_head.calls[0]["visual_context_tokens"] == (batch_size, future_token_count, config.latent_dim)


def test_flow_transformer_wan_vae_future_delta_forwards_doubled_visual_token_count() -> None:
    config = _wan_token_test_config(
        idm_flow_visual_token_conditioning=True,
        idm_flow_visual_token_conditioning_mode="cross_attention",
        idm_flow_visual_token_scope="future_only",
        idm_flow_visual_token_representation="future_delta",
    )
    idm = InverseDynamicsModel(config)
    flow_head = _ZeroRecordingFlowHead()
    idm.flow_head = flow_head
    batch_size = 2
    current = torch.rand(batch_size, 1, 3, config.image_size, config.image_size)
    future = torch.rand(batch_size, config.num_future_frames, 1, 3, config.image_size, config.image_size)
    state = torch.rand(batch_size, config.state_dim)
    target_action = torch.rand(batch_size, config.action_horizon, config.action_dim)
    action_mask = torch.ones(batch_size, config.action_horizon)
    wan_vae_latents = _wan_latents(config, batch_size)

    idm(
        current,
        future,
        state,
        target_action=target_action,
        action_mask=action_mask,
        wan_vae_latents=wan_vae_latents,
        mode="loss",
    )

    future_token_count = (idm.transition_encoder.latent_frames - 1) * idm.transition_encoder.latent_side**2
    assert flow_head.calls[0]["visual_context_tokens"] == (batch_size, 2 * future_token_count, config.latent_dim)


def test_flow_transformer_wan_vae_visual_token_conditioning_supports_history() -> None:
    config = _wan_token_test_config(
        idm_flow_visual_token_conditioning=True,
        idm_flow_visual_token_conditioning_mode="cross_attention",
        idm_flow_num_samples=2,
        idm_history_length=2,
    )
    idm = InverseDynamicsModel(config)
    flow_head = _ZeroRecordingFlowHead()
    idm.flow_head = flow_head
    batch_size = 2
    current = torch.rand(batch_size, 1, 3, config.image_size, config.image_size)
    future = torch.rand(batch_size, config.num_future_frames, 1, 3, config.image_size, config.image_size)
    state = torch.rand(batch_size, config.state_dim)
    target_action = torch.rand(batch_size, config.action_horizon, config.action_dim)
    action_mask = torch.ones(batch_size, config.action_horizon)
    wan_vae_latents = _wan_latents(config, batch_size)
    history_kwargs = {
        "prev_state_history": torch.rand(batch_size, config.idm_history_length, config.state_dim),
        "prev_action_history": torch.rand(batch_size, config.idm_history_length, config.action_dim),
        "history_mask": torch.ones(batch_size, config.idm_history_length),
    }

    idm(
        current,
        future,
        state,
        target_action=target_action,
        action_mask=action_mask,
        wan_vae_latents=wan_vae_latents,
        mode="loss",
        **history_kwargs,
    )
    sampled = idm(
        current,
        future,
        state,
        wan_vae_latents=wan_vae_latents,
        sample_noise=torch.zeros(batch_size * config.idm_flow_num_samples, config.action_horizon, config.action_dim),
        **history_kwargs,
    )

    latent_tokens = idm.transition_encoder.num_latent_tokens
    assert sampled.shape == (batch_size, config.action_horizon, config.action_dim)
    assert flow_head.calls[0]["visual_context_tokens"] == (batch_size, latent_tokens, config.latent_dim)
    assert flow_head.calls[0]["history_tokens"] == (batch_size, config.idm_history_length, config.latent_dim)
    assert flow_head.calls[-1]["visual_context_tokens"] == (
        batch_size * config.idm_flow_num_samples,
        latent_tokens,
        config.latent_dim,
    )
    assert flow_head.calls[-1]["history_tokens"] == (
        batch_size * config.idm_flow_num_samples,
        config.idm_history_length,
        config.latent_dim,
    )


@pytest.mark.parametrize("spatial_stride", [0, -16])
def test_wan_vae_transition_encoder_rejects_non_positive_spatial_stride(spatial_stride: int) -> None:
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
        wan_vae_spatial_stride=spatial_stride,
    )

    with pytest.raises(ValueError, match="wan_vae_spatial_stride must be positive"):
        WanVaeTransitionEncoder(config, wan_encoder=FakeWanVaeEncoder())


def test_flow_transformer_idm_can_use_wan_vae_visual_encoder(monkeypatch) -> None:
    monkeypatch.setattr(
        "world_model.wan_vae_encoder.build_frozen_wan_vae_encoder",
        lambda config: FakeWanVaeEncoder(
            latent_channels=config.wan_vae_latent_channels,
            spatial_stride=config.wan_vae_spatial_stride,
        ),
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
        idm_flow_num_samples=2,
    )
    idm = InverseDynamicsModel(config)

    current = torch.rand(3, 1, 3, 32, 32)
    future = torch.rand(3, 4, 1, 3, 32, 32)
    state = torch.rand(3, 4)
    action = torch.rand(3, 4, 4)
    mask = torch.ones(3, 4)

    loss = idm(current, future, state, target_action=action, action_mask=mask, mode="loss")
    sampled = idm(current, future, state)

    assert loss["loss"].item() >= 0.0
    assert sampled.shape == (3, 4, 4)


def test_wan_vae_visual_encoder_rejects_legacy_idm_architecture() -> None:
    with pytest.raises(ValueError, match="Non-patch IDM visual encoders"):
        InverseDynamicsModel(ModelConfig(idm_arch="delta", idm_visual_encoder="wan_vae"))


def test_flow_visual_token_conditioning_allows_patch_encoder() -> None:
    config = ModelConfig(
        idm_arch="flow_transformer",
        idm_visual_encoder="patch",
        idm_flow_visual_token_conditioning=True,
    )

    assert config.idm_visual_encoder == "patch"
    assert config.idm_flow_visual_token_conditioning is True


def test_flow_visual_token_cross_attention_rejects_incompatible_settings() -> None:
    with pytest.raises(ValueError, match="cross_attention.*idm_flow_visual_token_conditioning=True"):
        ModelConfig(
            idm_arch="flow_transformer",
            idm_visual_encoder="wan_vae",
            idm_flow_visual_token_conditioning_mode="cross_attention",
        )

    with pytest.raises(ValueError, match="idm_arch='flow_transformer'"):
        ModelConfig(
            idm_arch="stacked",
            idm_visual_encoder="wan_vae",
            idm_flow_visual_token_conditioning=True,
            idm_flow_visual_token_conditioning_mode="cross_attention",
        )

    patch_config = ModelConfig(
        idm_arch="flow_transformer",
        idm_visual_encoder="patch",
        idm_flow_visual_token_conditioning=True,
        idm_flow_visual_token_conditioning_mode="cross_attention",
    )
    assert patch_config.idm_flow_visual_token_conditioning_mode == "cross_attention"

    with pytest.raises(ValueError, match="idm_flow_visual_token_conditioning_mode"):
        ModelConfig(idm_flow_visual_token_conditioning_mode="sideways")


def test_flow_transformer_accepts_explicit_sample_noise() -> None:
    config = ModelConfig(
        num_views=1,
        num_future_frames=2,
        image_size=32,
        state_dim=4,
        action_dim=4,
        action_horizon=6,
        task_vocab_size=32,
        latent_dim=64,
        idm_arch="flow_transformer",
        idm_transformer_layers=1,
        idm_transformer_heads=4,
        idm_transformer_patch_size=16,
        idm_transformer_dropout=0.0,
        idm_flow_sampling_steps=2,
        idm_flow_num_samples=3,
    )
    idm = InverseDynamicsModel(config).eval()
    current = torch.rand(2, 1, 3, 32, 32)
    future = torch.rand(2, 2, 1, 3, 32, 32)
    state = torch.rand(2, 4)
    sample_noise = torch.randn(6, 6, 4)

    with torch.no_grad():
        first = idm(current, future, state, sample_noise=sample_noise)
        torch.manual_seed(123)
        second = idm(current, future, state, sample_noise=sample_noise)

    assert first.shape == (2, 6, 4)
    assert torch.allclose(first, second)

    with pytest.raises(ValueError, match="sample_noise"):
        idm(current, future, state, sample_noise=torch.randn(2, 6, 4))


def test_idm_does_not_condition_on_task_id() -> None:
    current = torch.rand(2, 1, 3, 32, 32)
    future = torch.rand(2, 2, 1, 3, 32, 32)
    state = torch.rand(2, 4)
    first_task_id = torch.tensor([0, 1], dtype=torch.long)
    second_task_id = torch.tensor([17, 18], dtype=torch.long)

    for arch in ("stacked", "delta", "transformer", "flow_transformer"):
        config = ModelConfig(
            num_views=1,
            num_future_frames=2,
            image_size=32,
            state_dim=4,
            action_dim=4,
            action_horizon=3,
            task_vocab_size=32,
            latent_dim=64,
            idm_arch=arch,
            idm_transformer_layers=1,
            idm_transformer_heads=4,
            idm_transformer_patch_size=16,
            idm_transformer_dropout=0.0,
            idm_flow_sampling_steps=2,
            idm_flow_num_samples=2,
        )
        idm = InverseDynamicsModel(config).eval()

        with torch.no_grad():
            torch.manual_seed(100)
            first = idm(current, future, state, first_task_id)
            torch.manual_seed(100)
            second = idm(current, future, state, second_task_id)

        assert torch.allclose(first, second)


def test_transformer_idm_rejects_invalid_patch_size() -> None:
    config = ModelConfig(
        image_size=32,
        latent_dim=64,
        idm_arch="transformer",
        idm_transformer_heads=4,
        idm_transformer_patch_size=12,
    )

    with pytest.raises(ValueError, match="idm_transformer_patch_size"):
        InverseDynamicsModel(config)


def test_flow_transformer_rejects_invalid_num_samples() -> None:
    config = ModelConfig(
        image_size=32,
        latent_dim=64,
        idm_arch="flow_transformer",
        idm_transformer_heads=4,
        idm_transformer_patch_size=16,
        idm_flow_num_samples=0,
    )

    with pytest.raises(ValueError, match="idm_flow_num_samples"):
        InverseDynamicsModel(config)


def test_flow_transformer_rejects_invalid_time_scale_and_endpoint_weight() -> None:
    with pytest.raises(ValueError, match="idm_flow_time_scale"):
        InverseDynamicsModel(
            ModelConfig(
                image_size=32,
                latent_dim=64,
                idm_arch="flow_transformer",
                idm_transformer_heads=4,
                idm_transformer_patch_size=16,
                idm_flow_time_scale=0.0,
            )
        )

    with pytest.raises(ValueError, match="idm_flow_endpoint_loss_weight"):
        InverseDynamicsModel(
            ModelConfig(
                image_size=32,
                latent_dim=64,
                idm_arch="flow_transformer",
                idm_transformer_heads=4,
                idm_transformer_patch_size=16,
                idm_flow_endpoint_loss_weight=-0.1,
            )
        )

    with pytest.raises(ValueError, match="idm_flow_endpoint_consistency_loss_weight"):
        ModelConfig(idm_flow_endpoint_consistency_loss_weight=-0.1)

    with pytest.raises(ValueError, match="idm_flow_context_conditioning"):
        ModelConfig(idm_flow_context_conditioning="sideways")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"idm_flow_train_time_min": -0.1}, "idm_flow_train_time_min"),
        ({"idm_flow_train_time_max": 1.1}, "idm_flow_train_time_max"),
        (
            {"idm_flow_train_time_min": 0.8, "idm_flow_train_time_max": 0.2},
            "idm_flow_train_time_min must be <= idm_flow_train_time_max",
        ),
    ],
)
def test_flow_transformer_rejects_invalid_train_time_ranges(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        ModelConfig(**overrides)


def test_world_model_rejects_wrong_image_shape() -> None:
    config = ModelConfig(num_views=3, image_size=32, state_dim=4, action_dim=4, task_vocab_size=32)
    model = ConvVideoWorldModel(config)

    with pytest.raises(ValueError, match="current_images"):
        model(torch.rand(2, 2, 3, 32, 32), torch.rand(2, 4), torch.tensor([0, 1]))
