from __future__ import annotations

import math

import torch
from torch import nn

from world_model.config import ModelConfig


def _validate_images(images: torch.Tensor, *, num_views: int, image_size: int, name: str) -> None:
    expected = (num_views, 3, image_size, image_size)
    if images.ndim != 5 or tuple(images.shape[1:]) != expected:
        raise ValueError(f"{name} must have shape (B, {expected}), got {tuple(images.shape)}.")


def _validate_future_images(
    images: torch.Tensor,
    *,
    num_future_frames: int,
    num_views: int,
    image_size: int,
    name: str,
) -> None:
    expected = (num_future_frames, num_views, 3, image_size, image_size)
    if images.ndim != 6 or tuple(images.shape[1:]) != expected:
        raise ValueError(f"{name} must have shape (B, {expected}), got {tuple(images.shape)}.")


def _validate_action_chunk(action: torch.Tensor, *, action_horizon: int, action_dim: int, name: str) -> None:
    expected = (action_horizon, action_dim)
    if action.ndim != 3 or tuple(action.shape[1:]) != expected:
        raise ValueError(f"{name} must have shape (B, {expected}), got {tuple(action.shape)}.")


def _validate_state(state: torch.Tensor, *, state_dim: int, name: str) -> None:
    expected = (state_dim,)
    if state.ndim != 2 or tuple(state.shape[1:]) != expected:
        raise ValueError(f"{name} must have shape (B, {expected}), got {tuple(state.shape)}.")


def _validate_history(
    prev_state_history: torch.Tensor,
    prev_action_history: torch.Tensor,
    history_mask: torch.Tensor,
    *,
    history_length: int,
    state_dim: int,
    action_dim: int,
) -> None:
    expected_state = (history_length, state_dim)
    expected_action = (history_length, action_dim)
    if prev_state_history.ndim != 3 or tuple(prev_state_history.shape[1:]) != expected_state:
        raise ValueError(
            "prev_state_history must have shape " f"(B, {expected_state}), got {tuple(prev_state_history.shape)}."
        )
    if prev_action_history.ndim != 3 or tuple(prev_action_history.shape[1:]) != expected_action:
        raise ValueError(
            "prev_action_history must have shape " f"(B, {expected_action}), got {tuple(prev_action_history.shape)}."
        )
    if tuple(prev_action_history.shape[:2]) != tuple(prev_state_history.shape[:2]):
        raise ValueError(
            "prev_action_history batch/history dimensions must match prev_state_history: "
            f"{tuple(prev_action_history.shape[:2])} != {tuple(prev_state_history.shape[:2])}."
        )
    if history_mask.ndim != 2 or tuple(history_mask.shape) != tuple(prev_state_history.shape[:2]):
        raise ValueError(
            "history_mask must have shape " f"{tuple(prev_state_history.shape[:2])}, got {tuple(history_mask.shape)}."
        )


def _zero_wan_vae_current_latent_slice(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError(f"Wan VAE latents must have shape (B, C, T, H, W), got {tuple(latents.shape)}.")
    if latents.shape[2] < 1:
        raise ValueError(f"Wan VAE latents must contain at least one time step, got {tuple(latents.shape)}.")
    zeroed = latents.clone()
    zeroed[:, :, 0] = torch.zeros_like(zeroed[:, :, 0])
    return zeroed


def _masked_action_mse(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < predicted.ndim:
        mask = mask.unsqueeze(-1)
    squared = (predicted - target).square() * mask
    return squared.sum() / mask.expand_as(predicted).sum().clamp_min(1.0)


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
            nn.SiLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images + self.net(images)


class ResidualTransitionEncoder(nn.Module):
    """UniPi-style transition encoder for inverse dynamics from visual frame transitions."""

    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        channels = min(latent_dim, 128)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            ResidualConvBlock(channels),
            ResidualConvBlock(channels),
            ResidualConvBlock(channels),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, latent_dim),
            nn.SiLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class TransformerTransitionEncoder(nn.Module):
    """Patch-token inverse dynamics encoder over current, future, and delta visual transitions."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.image_size % config.idm_transformer_patch_size != 0:
            raise ValueError("image_size must be divisible by idm_transformer_patch_size.")
        if config.latent_dim % config.idm_transformer_heads != 0:
            raise ValueError("latent_dim must be divisible by idm_transformer_heads.")
        if config.idm_transformer_layers <= 0:
            raise ValueError("idm_transformer_layers must be positive.")
        if config.idm_transformer_heads <= 0:
            raise ValueError("idm_transformer_heads must be positive.")
        if config.idm_transformer_patch_size <= 0:
            raise ValueError("idm_transformer_patch_size must be positive.")
        if config.idm_transformer_dropout < 0.0:
            raise ValueError("idm_transformer_dropout must be non-negative.")

        self.config = config
        self.patch_embed = nn.Conv2d(
            3,
            config.latent_dim,
            kernel_size=config.idm_transformer_patch_size,
            stride=config.idm_transformer_patch_size,
        )
        patch_side = config.image_size // config.idm_transformer_patch_size
        self.num_patches = patch_side * patch_side
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))
        self.time_embedding = nn.Embedding(config.num_future_frames + 1, config.latent_dim)
        self.view_embedding = nn.Embedding(config.num_views, config.latent_dim)
        self.type_embedding = nn.Embedding(3, config.latent_dim)
        self.spatial_embedding = nn.Embedding(self.num_patches, config.latent_dim)
        self.state_projection = nn.Sequential(
            nn.Linear(config.state_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )
        self.input_norm = nn.LayerNorm(config.latent_dim)
        ff_dim = config.idm_transformer_ff_dim or config.latent_dim * 4
        layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.idm_transformer_heads,
            dim_feedforward=ff_dim,
            dropout=config.idm_transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.idm_transformer_layers)
        self.output_norm = nn.LayerNorm(config.latent_dim)

    def _patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        prefix = images.shape[:-3]
        flat_images = images.reshape(-1, 3, self.config.image_size, self.config.image_size)
        tokens = self.patch_embed(flat_images).flatten(2).transpose(1, 2)
        return tokens.reshape(*prefix, self.num_patches, self.config.latent_dim)

    def forward(
        self,
        current_images: torch.Tensor,
        future_images: torch.Tensor,
        state: torch.Tensor,
        *,
        return_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.config.idm_future_conditioning == "future_only":
            current_images = torch.zeros_like(current_images)
            state = torch.zeros_like(state)
        batch_size = current_images.shape[0]
        device = current_images.device
        spatial = self.spatial_embedding(torch.arange(self.num_patches, device=device))
        views = self.view_embedding(torch.arange(self.config.num_views, device=device))
        types = self.type_embedding(torch.arange(3, device=device))
        times = self.time_embedding(torch.arange(self.config.num_future_frames + 1, device=device))

        current_tokens = self._patch_tokens(current_images)
        current_tokens = current_tokens + spatial.view(1, 1, self.num_patches, -1)
        current_tokens = current_tokens + views.view(1, self.config.num_views, 1, -1)
        current_tokens = current_tokens + times[0].view(1, 1, 1, -1) + types[0].view(1, 1, 1, -1)
        current_tokens = current_tokens.reshape(batch_size, self.config.num_views * self.num_patches, -1)

        future_tokens = self._patch_tokens(future_images)
        future_tokens = future_tokens + spatial.view(1, 1, 1, self.num_patches, -1)
        future_tokens = future_tokens + views.view(1, 1, self.config.num_views, 1, -1)
        future_tokens = future_tokens + times[1:].view(1, self.config.num_future_frames, 1, 1, -1)
        future_tokens = future_tokens + types[1].view(1, 1, 1, 1, -1)
        future_tokens = future_tokens.reshape(
            batch_size,
            self.config.num_future_frames * self.config.num_views * self.num_patches,
            -1,
        )

        repeated_current = current_images.unsqueeze(1).expand_as(future_images)
        delta_tokens = self._patch_tokens(future_images - repeated_current)
        delta_tokens = delta_tokens + spatial.view(1, 1, 1, self.num_patches, -1)
        delta_tokens = delta_tokens + views.view(1, 1, self.config.num_views, 1, -1)
        delta_tokens = delta_tokens + times[1:].view(1, self.config.num_future_frames, 1, 1, -1)
        delta_tokens = delta_tokens + types[2].view(1, 1, 1, 1, -1)
        delta_tokens = delta_tokens.reshape(
            batch_size,
            self.config.num_future_frames * self.config.num_views * self.num_patches,
            -1,
        )

        tokens = torch.cat(
            [
                self.cls_token.expand(batch_size, -1, -1),
                self.state_projection(state).unsqueeze(1),
                current_tokens,
                future_tokens,
                delta_tokens,
            ],
            dim=1,
        )
        encoded = self.output_norm(self.encoder(self.input_norm(tokens)))
        context = encoded[:, 0]
        if return_tokens:
            if self.config.idm_flow_visual_token_representation == "future_delta":
                if self.config.idm_flow_visual_token_scope != "future_only":
                    raise ValueError(
                        "idm_flow_visual_token_representation='future_delta' requires "
                        "idm_flow_visual_token_scope='future_only'."
                    )
                visual_tokens = torch.cat([future_tokens, delta_tokens], dim=1)
            else:
                visual_tokens = encoded[:, 2:]
                if self.config.idm_flow_visual_token_scope == "future_only":
                    current_token_count = self.config.num_views * self.num_patches
                    visual_tokens = visual_tokens[:, current_token_count:]
            if visual_tokens.shape[1] <= 0:
                raise ValueError("Patch visual token conditioning produced no visual tokens.")
            return context, visual_tokens
        return context


class CurrentOnlyTransitionEncoder(nn.Module):
    """Patch-token flow context encoder over current images and state only."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.image_size % config.idm_transformer_patch_size != 0:
            raise ValueError("image_size must be divisible by idm_transformer_patch_size.")
        if config.latent_dim % config.idm_transformer_heads != 0:
            raise ValueError("latent_dim must be divisible by idm_transformer_heads.")
        if config.idm_transformer_layers <= 0:
            raise ValueError("idm_transformer_layers must be positive.")
        if config.idm_transformer_heads <= 0:
            raise ValueError("idm_transformer_heads must be positive.")
        if config.idm_transformer_patch_size <= 0:
            raise ValueError("idm_transformer_patch_size must be positive.")
        if config.idm_transformer_dropout < 0.0:
            raise ValueError("idm_transformer_dropout must be non-negative.")

        self.config = config
        self.patch_embed = nn.Conv2d(
            3,
            config.latent_dim,
            kernel_size=config.idm_transformer_patch_size,
            stride=config.idm_transformer_patch_size,
        )
        patch_side = config.image_size // config.idm_transformer_patch_size
        self.num_patches = patch_side * patch_side
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))
        self.view_embedding = nn.Embedding(config.num_views, config.latent_dim)
        self.spatial_embedding = nn.Embedding(self.num_patches, config.latent_dim)
        self.state_projection = nn.Sequential(
            nn.Linear(config.state_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )
        self.input_norm = nn.LayerNorm(config.latent_dim)
        ff_dim = config.idm_transformer_ff_dim or config.latent_dim * 4
        layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.idm_transformer_heads,
            dim_feedforward=ff_dim,
            dropout=config.idm_transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.idm_transformer_layers)
        self.output_norm = nn.LayerNorm(config.latent_dim)

    def _patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        flat_images = images.reshape(-1, 3, self.config.image_size, self.config.image_size)
        tokens = self.patch_embed(flat_images).flatten(2).transpose(1, 2)
        return tokens.reshape(images.shape[0], self.config.num_views, self.num_patches, self.config.latent_dim)

    def forward(self, current_images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = current_images.shape[0]
        device = current_images.device
        spatial = self.spatial_embedding(torch.arange(self.num_patches, device=device))
        views = self.view_embedding(torch.arange(self.config.num_views, device=device))

        current_tokens = self._patch_tokens(current_images)
        current_tokens = current_tokens + spatial.view(1, 1, self.num_patches, -1)
        current_tokens = current_tokens + views.view(1, self.config.num_views, 1, -1)
        current_tokens = current_tokens.reshape(batch_size, self.config.num_views * self.num_patches, -1)
        tokens = torch.cat(
            [
                self.cls_token.expand(batch_size, -1, -1),
                self.state_projection(state).unsqueeze(1),
                current_tokens,
            ],
            dim=1,
        )
        encoded = self.encoder(self.input_norm(tokens))
        return self.output_norm(encoded[:, 0])


class WanVaeTransitionEncoder(nn.Module):
    """Transition encoder over frozen Wan2.2 VAE video latents."""

    def __init__(self, config: ModelConfig, *, wan_encoder=None):
        super().__init__()
        if config.num_views != 1:
            raise ValueError("idm_visual_encoder='wan_vae' currently supports exactly one selected camera view.")
        if config.idm_arch != "flow_transformer":
            raise ValueError("idm_visual_encoder='wan_vae' is supported only with idm_arch='flow_transformer'.")
        if config.wan_vae_spatial_stride <= 0:
            raise ValueError("wan_vae_spatial_stride must be positive.")
        if config.image_size % config.wan_vae_spatial_stride != 0:
            raise ValueError("image_size must be divisible by wan_vae_spatial_stride.")
        if config.latent_dim % config.idm_transformer_heads != 0:
            raise ValueError("latent_dim must be divisible by idm_transformer_heads.")
        if config.idm_transformer_layers <= 0:
            raise ValueError("idm_transformer_layers must be positive.")
        if config.idm_transformer_heads <= 0:
            raise ValueError("idm_transformer_heads must be positive.")
        if config.wan_vae_latent_channels <= 0:
            raise ValueError("wan_vae_latent_channels must be positive.")

        self.config = config
        self.total_video_frames = 1 + config.num_future_frames
        self.latent_frames = (self.total_video_frames + 3) // 4
        self.latent_side = config.image_size // config.wan_vae_spatial_stride
        self.num_latent_tokens = self.latent_frames * self.latent_side * self.latent_side
        self.latent_projection = nn.Linear(config.wan_vae_latent_channels, config.latent_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))
        self.time_embedding = nn.Embedding(self.latent_frames, config.latent_dim)
        self.latent_type_embedding = nn.Embedding(2, config.latent_dim)
        self.spatial_embedding = nn.Embedding(self.latent_side * self.latent_side, config.latent_dim)
        if config.idm_flow_visual_token_representation == "future_delta":
            self.transition_token_type_embedding = nn.Embedding(2, config.latent_dim)
            nn.init.zeros_(self.transition_token_type_embedding.weight)
        self.state_projection = nn.Sequential(
            nn.Linear(config.state_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )
        self.input_norm = nn.LayerNorm(config.latent_dim)
        ff_dim = config.idm_transformer_ff_dim or config.latent_dim * 4
        layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.idm_transformer_heads,
            dim_feedforward=ff_dim,
            dropout=config.idm_transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.idm_transformer_layers)
        self.output_norm = nn.LayerNorm(config.latent_dim)
        if wan_encoder is None:
            if config.wan_vae_use_cached_latents:
                wan_encoder = None
            else:
                from world_model.wan_vae_encoder import build_frozen_wan_vae_encoder

                wan_encoder = build_frozen_wan_vae_encoder(config)
        object.__setattr__(self, "_wan_encoder", wan_encoder)

    @property
    def wan_encoder(self):
        return object.__getattribute__(self, "_wan_encoder")

    def _video_from_images(self, current_images: torch.Tensor, future_images: torch.Tensor) -> torch.Tensor:
        current = current_images[:, 0].unsqueeze(1)
        future = future_images[:, :, 0]
        video = torch.cat([current, future], dim=1)
        return video.permute(0, 2, 1, 3, 4).mul(2.0).sub(1.0).contiguous()

    def _expected_latent_shape(self, batch_size: int) -> tuple[int, int, int, int, int]:
        return (
            batch_size,
            self.config.wan_vae_latent_channels,
            self.latent_frames,
            self.latent_side,
            self.latent_side,
        )

    def _validate_latents(self, latents: torch.Tensor, *, batch_size: int) -> None:
        expected_shape = self._expected_latent_shape(batch_size)
        if tuple(latents.shape) != expected_shape:
            raise ValueError(f"Wan VAE latents must have shape {expected_shape}, got {tuple(latents.shape)}.")

    def _encode_images(self, current_images: torch.Tensor, future_images: torch.Tensor) -> torch.Tensor:
        if self.wan_encoder is None:
            raise ValueError(
                "wan_vae_latents are required because this IDM was configured with " "wan_vae_use_cached_latents=True."
            )
        video = self._video_from_images(current_images, future_images)
        with torch.no_grad():
            return self.wan_encoder.encode_videos(video)

    def _future_delta_visual_tokens(self, projected_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = projected_tokens.shape[0]
        spatial_token_count = self.latent_side * self.latent_side
        latent_tokens = projected_tokens.view(batch_size, self.latent_frames, spatial_token_count, -1)
        current_tokens = latent_tokens[:, :1]
        future_tokens = latent_tokens[:, 1:]
        delta_tokens = future_tokens - current_tokens.expand_as(future_tokens)
        future_tokens = future_tokens.reshape(batch_size, -1, self.config.latent_dim)
        delta_tokens = delta_tokens.reshape(batch_size, -1, self.config.latent_dim)
        visual_tokens = torch.cat([future_tokens, delta_tokens], dim=1)
        future_token_count = future_tokens.shape[1]
        type_ids = torch.cat(
            [
                torch.zeros(future_token_count, dtype=torch.long, device=projected_tokens.device),
                torch.ones(future_token_count, dtype=torch.long, device=projected_tokens.device),
            ]
        )
        return visual_tokens + self.transition_token_type_embedding(type_ids).unsqueeze(0)

    def forward(
        self,
        current_images: torch.Tensor,
        future_images: torch.Tensor,
        state: torch.Tensor,
        wan_vae_latents: torch.Tensor | None = None,
        *,
        return_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        future_only = self.config.idm_future_conditioning == "future_only"
        if future_only:
            current_images = torch.zeros_like(current_images)
            state = torch.zeros_like(state)
        if wan_vae_latents is None:
            if self.config.wan_vae_use_cached_latents:
                raise ValueError(
                    "wan_vae_latents are required because this IDM was configured with "
                    "wan_vae_use_cached_latents=True."
                )
            latents = self._encode_images(current_images, future_images)
        else:
            latents = wan_vae_latents
        self._validate_latents(latents, batch_size=current_images.shape[0])
        if future_only:
            latents = _zero_wan_vae_current_latent_slice(latents)

        batch_size = latents.shape[0]
        device = latents.device
        tokens = latents.permute(0, 2, 3, 4, 1).reshape(batch_size, self.num_latent_tokens, -1)
        projected_tokens = self.latent_projection(tokens.to(dtype=self.latent_projection.weight.dtype))
        tokens = projected_tokens
        time_ids = torch.arange(self.latent_frames, device=device).repeat_interleave(
            self.latent_side * self.latent_side
        )
        spatial_ids = torch.arange(self.latent_side * self.latent_side, device=device).repeat(self.latent_frames)
        type_ids = (time_ids > 0).long()
        tokens = tokens + self.time_embedding(time_ids).unsqueeze(0)
        tokens = tokens + self.latent_type_embedding(type_ids).unsqueeze(0)
        tokens = tokens + self.spatial_embedding(spatial_ids).unsqueeze(0)
        tokens = torch.cat(
            [
                self.cls_token.expand(batch_size, -1, -1),
                self.state_projection(state).unsqueeze(1),
                tokens,
            ],
            dim=1,
        )
        encoded = self.output_norm(self.encoder(self.input_norm(tokens)))
        context = encoded[:, 0]
        if return_tokens:
            if self.config.idm_flow_visual_token_representation == "future_delta":
                visual_tokens = self._future_delta_visual_tokens(projected_tokens)
            else:
                visual_tokens = encoded[:, 2:]
            if (
                self.config.idm_flow_visual_token_representation == "encoded"
                and self.config.idm_flow_visual_token_scope == "future_only"
            ):
                current_token_count = self.latent_side * self.latent_side
                visual_tokens = visual_tokens[:, current_token_count:]
            return context, visual_tokens
        return context


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, *, scale: float = 1.0):
        super().__init__()
        if dim < 2:
            raise ValueError("SinusoidalTimeEmbedding dim must be at least 2.")
        if scale <= 0.0:
            raise ValueError("SinusoidalTimeEmbedding scale must be positive.")
        self.dim = dim
        self.scale = scale

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        if half_dim == 1:
            frequencies = torch.ones(1, device=time.device, dtype=time.dtype)
        else:
            exponent = torch.arange(half_dim, device=time.device, dtype=time.dtype) / (half_dim - 1)
            frequencies = torch.exp(-math.log(10000.0) * exponent)
        angles = (time * self.scale).unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding


class FlowActionVisualCrossAttentionLayer(nn.Module):
    """Flow action layer with explicit action-token attention over visual memory."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        ff_dim = config.idm_transformer_ff_dim or config.latent_dim * 4
        self.self_attention = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.idm_transformer_heads,
            dim_feedforward=ff_dim,
            dropout=config.idm_transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_norm = nn.LayerNorm(config.latent_dim)
        self.visual_norm = nn.LayerNorm(config.latent_dim)
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=config.latent_dim,
            num_heads=config.idm_transformer_heads,
            dropout=config.idm_transformer_dropout,
            batch_first=True,
        )
        self.visual_dropout = nn.Dropout(config.idm_transformer_dropout)
        self.visual_output_norm = nn.LayerNorm(config.latent_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        action_start: int,
        visual_context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.self_attention(tokens)
        prefix_tokens = tokens[:, :action_start]
        action_tokens = tokens[:, action_start:]
        visual_tokens = self.visual_norm(visual_context_tokens)
        visual_update, _ = self.visual_attention(
            self.action_norm(action_tokens),
            visual_tokens,
            visual_tokens,
            need_weights=False,
        )
        action_tokens = self.visual_output_norm(action_tokens + self.visual_dropout(visual_update))
        return torch.cat([prefix_tokens, action_tokens], dim=1)


class FlowActionTransformerHead(nn.Module):
    """Flow-matching action decoder conditioned on visual transition context."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.idm_flow_sampling_steps <= 0:
            raise ValueError("idm_flow_sampling_steps must be positive.")
        if config.idm_flow_num_samples <= 0:
            raise ValueError("idm_flow_num_samples must be positive.")
        if config.idm_flow_sample_noise_scale < 0.0:
            raise ValueError("idm_flow_sample_noise_scale must be non-negative.")
        if config.idm_flow_time_scale <= 0.0:
            raise ValueError("idm_flow_time_scale must be positive.")
        if config.idm_flow_endpoint_loss_weight < 0.0:
            raise ValueError("idm_flow_endpoint_loss_weight must be non-negative.")
        if config.idm_flow_endpoint_consistency_loss_weight < 0.0:
            raise ValueError("idm_flow_endpoint_consistency_loss_weight must be non-negative.")
        if config.idm_flow_zero_start_endpoint_loss_weight < 0.0:
            raise ValueError("idm_flow_zero_start_endpoint_loss_weight must be non-negative.")
        if config.idm_flow_sampled_action_loss_weight < 0.0:
            raise ValueError("idm_flow_sampled_action_loss_weight must be non-negative.")
        if config.idm_flow_context_conditioning not in ("token", "additive"):
            raise ValueError("idm_flow_context_conditioning must be one of {'token', 'additive'}.")
        if config.idm_flow_visual_token_conditioning_mode not in ("prefix", "cross_attention"):
            raise ValueError("idm_flow_visual_token_conditioning_mode must be one of {'prefix', 'cross_attention'}.")
        self.config = config
        self.context_conditioning = config.idm_flow_context_conditioning
        self.visual_token_conditioning_mode = config.idm_flow_visual_token_conditioning_mode
        self.action_projection = nn.Linear(config.action_dim, config.latent_dim)
        self.horizon_embedding = nn.Embedding(config.action_horizon, config.latent_dim)
        self.time_projection = nn.Sequential(
            SinusoidalTimeEmbedding(config.latent_dim, scale=config.idm_flow_time_scale),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        ff_dim = config.idm_transformer_ff_dim or config.latent_dim * 4
        if self.visual_token_conditioning_mode == "cross_attention":
            self.visual_cross_attention_layers = nn.ModuleList(
                [FlowActionVisualCrossAttentionLayer(config) for _ in range(config.idm_transformer_layers)]
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=config.latent_dim,
                nhead=config.idm_transformer_heads,
                dim_feedforward=ff_dim,
                dropout=config.idm_transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.idm_transformer_layers)
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.output_norm = nn.LayerNorm(config.latent_dim)
        self.velocity_head = nn.Linear(config.latent_dim, config.action_dim)
        if self.context_conditioning == "additive":
            self.context_action_projection = nn.Linear(config.latent_dim, config.latent_dim)
            self.context_velocity_head = nn.Linear(config.latent_dim, config.action_dim)

    def forward(
        self,
        context: torch.Tensor,
        noisy_action: torch.Tensor,
        time: torch.Tensor,
        *,
        history_tokens: torch.Tensor | None = None,
        visual_context_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_action_chunk(
            noisy_action,
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
            name="noisy_action",
        )
        batch_size = noisy_action.shape[0]
        expected_context = (batch_size, self.config.latent_dim)
        if context.ndim != 2 or tuple(context.shape) != expected_context:
            raise ValueError(f"context must have shape {expected_context}, got {tuple(context.shape)}.")
        if history_tokens is not None:
            expected_history = (batch_size, self.config.idm_history_length, self.config.latent_dim)
            if history_tokens.ndim != 3 or tuple(history_tokens.shape) != expected_history:
                raise ValueError(
                    f"history_tokens must have shape {expected_history}, got {tuple(history_tokens.shape)}."
                )
        if visual_context_tokens is not None:
            if visual_context_tokens.ndim != 3:
                raise ValueError(
                    "visual_context_tokens must have shape "
                    f"(B, T, {self.config.latent_dim}), got {tuple(visual_context_tokens.shape)}."
                )
            if visual_context_tokens.shape[0] != batch_size or visual_context_tokens.shape[2] != self.config.latent_dim:
                raise ValueError(
                    "visual_context_tokens must have shape "
                    f"(B, T, {self.config.latent_dim}), got {tuple(visual_context_tokens.shape)}."
                )
            if visual_context_tokens.shape[1] <= 0:
                raise ValueError("visual_context_tokens must contain at least one token.")
        if time.ndim != 1 or tuple(time.shape) != (batch_size,):
            raise ValueError(f"time must have shape ({batch_size},), got {tuple(time.shape)}.")
        device = noisy_action.device
        action_tokens = self.action_projection(noisy_action)
        horizon = self.horizon_embedding(torch.arange(self.config.action_horizon, device=device))
        time_tokens = self.time_projection(time)
        action_tokens = action_tokens + horizon.view(1, self.config.action_horizon, -1)
        action_tokens = action_tokens + time_tokens.view(batch_size, 1, -1)
        conditioned_context = context + time_tokens
        if self.context_conditioning == "additive":
            action_tokens = action_tokens + self.context_action_projection(conditioned_context).unsqueeze(1)
        context_token = conditioned_context.unsqueeze(1)
        prefix_tokens = [context_token]
        if self.visual_token_conditioning_mode == "cross_attention":
            if visual_context_tokens is None:
                raise ValueError(
                    "visual_context_tokens are required when "
                    "idm_flow_visual_token_conditioning_mode='cross_attention'."
                )
        elif visual_context_tokens is not None:
            prefix_tokens.append(visual_context_tokens)
        if history_tokens is not None:
            prefix_tokens.append(history_tokens)
        action_start = sum(token.shape[1] for token in prefix_tokens)
        tokens = torch.cat([*prefix_tokens, action_tokens], dim=1)
        encoded = self.input_norm(tokens)
        if self.visual_token_conditioning_mode == "cross_attention":
            if visual_context_tokens is None:
                raise RuntimeError("visual_context_tokens were unexpectedly missing.")
            for layer in self.visual_cross_attention_layers:
                encoded = layer(
                    encoded,
                    action_start=action_start,
                    visual_context_tokens=visual_context_tokens,
                )
        else:
            encoded = self.encoder(encoded)
        velocity = self.velocity_head(self.output_norm(encoded[:, action_start:]))
        if self.context_conditioning == "additive":
            context_horizon_tokens = conditioned_context.unsqueeze(1) + horizon.view(1, self.config.action_horizon, -1)
            velocity = velocity + self.context_velocity_head(context_horizon_tokens)
        return velocity


class FlowContextActionHead(nn.Module):
    """Direct action decoder used only for auxiliary supervision of the flow context."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.horizon_embedding = nn.Embedding(config.action_horizon, config.latent_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.action_dim),
        )

    def forward(self, context: torch.Tensor, *, history_tokens: torch.Tensor | None = None) -> torch.Tensor:
        if context.ndim != 2 or tuple(context.shape[1:]) != (self.config.latent_dim,):
            expected = ("B", self.config.latent_dim)
            raise ValueError(f"context must have shape {expected}, got {tuple(context.shape)}.")
        if history_tokens is not None:
            expected_history = (context.shape[0], self.config.idm_history_length, self.config.latent_dim)
            if history_tokens.ndim != 3 or tuple(history_tokens.shape) != expected_history:
                raise ValueError(
                    f"history_tokens must have shape {expected_history}, got {tuple(history_tokens.shape)}."
                )
            context = context + history_tokens.mean(dim=1)
        horizon = self.horizon_embedding(torch.arange(self.config.action_horizon, device=context.device))
        tokens = context.unsqueeze(1) + horizon.view(1, self.config.action_horizon, -1)
        return self.net(tokens)


class ConvVideoWorldModel(nn.Module):
    """Small baseline world model: current multi-view image/state/task -> future multi-view image."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8.")
        self.config = config
        image_channels = config.num_views * 3
        output_channels = config.num_future_frames * config.num_views * 3
        self.encoder = ImageEncoder(image_channels, config.latent_dim)
        self.task_embedding = nn.Embedding(config.task_vocab_size, config.task_embed_dim)
        feature_side = config.image_size // 8
        self.condition = nn.Sequential(
            nn.Linear(config.latent_dim + config.state_dim + config.task_embed_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, 128 * feature_side * feature_side),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, current_images: torch.Tensor, state: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        _validate_images(
            current_images,
            num_views=self.config.num_views,
            image_size=self.config.image_size,
            name="current_images",
        )
        _validate_state(state, state_dim=self.config.state_dim, name="state")
        batch_size = current_images.shape[0]
        images = current_images.flatten(1, 2)
        latent = self.encoder(images)
        task = self.task_embedding(task_id)
        conditioned = self.condition(torch.cat([latent, state, task], dim=-1))
        features = conditioned.view(batch_size, 128, self.config.image_size // 8, self.config.image_size // 8)
        output = self.decoder(features)
        return output.view(
            batch_size,
            self.config.num_future_frames,
            self.config.num_views,
            3,
            self.config.image_size,
            self.config.image_size,
        )


class InverseDynamicsModel(nn.Module):
    """Predicts the action that connects current frames to target/generated future frames."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.uses_flow_matching = config.idm_arch == "flow_transformer"
        if config.idm_visual_encoder != "patch" and config.idm_arch != "flow_transformer":
            raise ValueError("Non-patch IDM visual encoders are supported only with idm_arch='flow_transformer'.")
        if config.idm_arch == "stacked":
            image_channels = config.num_views * 3 * (1 + config.num_future_frames)
            self.encoder = ResidualTransitionEncoder(image_channels, config.latent_dim)
            visual_dim = config.latent_dim
        elif config.idm_arch == "delta":
            current_channels = config.num_views * 3
            future_channels = config.num_views * 3 * config.num_future_frames
            self.current_encoder = ResidualTransitionEncoder(current_channels, config.latent_dim)
            self.future_encoder = ResidualTransitionEncoder(future_channels, config.latent_dim)
            self.delta_encoder = ResidualTransitionEncoder(future_channels, config.latent_dim)
            visual_dim = config.latent_dim * 3
        elif config.idm_arch == "transformer":
            self.transition_encoder = TransformerTransitionEncoder(config)
            visual_dim = config.latent_dim
        elif config.idm_arch == "flow_transformer":
            if config.idm_future_conditioning == "current_only":
                self.transition_encoder = CurrentOnlyTransitionEncoder(config)
            elif config.idm_visual_encoder == "patch":
                self.transition_encoder = TransformerTransitionEncoder(config)
            elif config.idm_visual_encoder == "wan_vae":
                self.transition_encoder = WanVaeTransitionEncoder(config)
            else:
                raise ValueError(f"Unknown IDM visual encoder: {config.idm_visual_encoder}")
            self.flow_head = FlowActionTransformerHead(config)
            self.context_action_head = FlowContextActionHead(config)
            if config.idm_history_length > 0:
                self.history_projection = nn.Linear(config.state_dim + config.action_dim, config.latent_dim)
                self.history_time_embedding = nn.Embedding(config.idm_history_length, config.latent_dim)
                self.history_norm = nn.LayerNorm(config.latent_dim)
            visual_dim = 0
        else:
            raise ValueError(f"Unknown IDM architecture: {config.idm_arch}")
        if config.idm_arch == "flow_transformer":
            self.head = None
        elif config.idm_arch == "transformer":
            head_input_dim = visual_dim
            self.head = self._build_regression_head(head_input_dim)
        else:
            head_input_dim = visual_dim + config.state_dim
            self.head = self._build_regression_head(head_input_dim)

    def _build_regression_head(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, self.config.latent_dim),
            nn.SiLU(),
            nn.Linear(self.config.latent_dim, self.config.latent_dim // 2),
            nn.SiLU(),
            nn.Linear(self.config.latent_dim // 2, self.config.action_horizon * self.config.action_dim),
        )

    def _transition_context(
        self,
        current_images: torch.Tensor,
        future_images: torch.Tensor,
        state: torch.Tensor,
        *,
        wan_vae_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.idm_future_conditioning == "current_only":
            if wan_vae_latents is not None:
                raise ValueError(
                    "wan_vae_latents were provided, but idm_future_conditioning='current_only' does not consume "
                    "future latents."
                )
            return self.transition_encoder(current_images, state)
        if self.config.idm_visual_encoder != "wan_vae" and wan_vae_latents is not None:
            raise ValueError("wan_vae_latents were provided, but this IDM is not using idm_visual_encoder='wan_vae'.")
        current_images, state, wan_vae_latents = self._future_only_inputs(
            current_images,
            state,
            wan_vae_latents,
        )
        if self.config.idm_visual_encoder == "wan_vae":
            return self.transition_encoder(current_images, future_images, state, wan_vae_latents=wan_vae_latents)
        return self.transition_encoder(current_images, future_images, state)

    def _transition_context_and_visual_tokens(
        self,
        current_images: torch.Tensor,
        future_images: torch.Tensor,
        state: torch.Tensor,
        *,
        wan_vae_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.config.idm_future_conditioning == "current_only":
            return self._transition_context(
                current_images,
                future_images,
                state,
                wan_vae_latents=wan_vae_latents,
            ), None
        if not self.config.idm_flow_visual_token_conditioning:
            return self._transition_context(
                current_images,
                future_images,
                state,
                wan_vae_latents=wan_vae_latents,
            ), None
        if self.config.idm_visual_encoder != "wan_vae" and wan_vae_latents is not None:
            raise ValueError("wan_vae_latents were provided, but this IDM is not using idm_visual_encoder='wan_vae'.")
        current_images, state, wan_vae_latents = self._future_only_inputs(
            current_images,
            state,
            wan_vae_latents,
        )
        if self.config.idm_visual_encoder == "wan_vae":
            return self.transition_encoder(
                current_images,
                future_images,
                state,
                wan_vae_latents=wan_vae_latents,
                return_tokens=True,
            )
        if self.config.idm_visual_encoder == "patch":
            return self.transition_encoder(
                current_images,
                future_images,
                state,
                return_tokens=True,
            )
        raise ValueError(f"Unknown IDM visual encoder: {self.config.idm_visual_encoder}")

    def _future_only_inputs(
        self,
        current_images: torch.Tensor,
        state: torch.Tensor,
        wan_vae_latents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.config.idm_future_conditioning != "future_only":
            return current_images, state, wan_vae_latents
        if wan_vae_latents is not None:
            wan_vae_latents = _zero_wan_vae_current_latent_slice(wan_vae_latents)
        return torch.zeros_like(current_images), torch.zeros_like(state), wan_vae_latents

    def _history_tokens(
        self,
        prev_state_history: torch.Tensor | None,
        prev_action_history: torch.Tensor | None,
        history_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        history_length = self.config.idm_history_length
        if self.config.idm_future_conditioning == "future_only":
            if history_length == 0:
                if prev_state_history is not None or prev_action_history is not None or history_mask is not None:
                    raise ValueError("IDM history tensors were provided, but model idm_history_length is 0.")
                return None
            if prev_state_history is None and prev_action_history is None and history_mask is None:
                return None
            if prev_state_history is None or prev_action_history is None or history_mask is None:
                raise ValueError(
                    "IDM history conditioning requires prev_state_history, prev_action_history, and history_mask "
                    f"because idm_history_length={history_length}."
                )
            _validate_history(
                prev_state_history,
                prev_action_history,
                history_mask,
                history_length=history_length,
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
            )
            return None
        if history_length == 0:
            if prev_state_history is not None or prev_action_history is not None or history_mask is not None:
                raise ValueError("IDM history tensors were provided, but model idm_history_length is 0.")
            return None
        if prev_state_history is None or prev_action_history is None or history_mask is None:
            raise ValueError(
                "IDM history conditioning requires prev_state_history, prev_action_history, and history_mask "
                f"because idm_history_length={history_length}."
            )
        _validate_history(
            prev_state_history,
            prev_action_history,
            history_mask,
            history_length=history_length,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
        )
        values = torch.cat([prev_state_history, prev_action_history], dim=-1)
        projection = self.history_projection
        tokens = projection(values.to(dtype=projection.weight.dtype))
        positions = torch.arange(history_length, device=tokens.device)
        tokens = tokens + self.history_time_embedding(positions).unsqueeze(0)
        mask = history_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        return self.history_norm(tokens * mask) * mask

    def forward(
        self,
        current_images: torch.Tensor,
        future_images: torch.Tensor,
        state: torch.Tensor,
        task_id: torch.Tensor | None = None,
        *,
        wan_vae_latents: torch.Tensor | None = None,
        prev_state_history: torch.Tensor | None = None,
        prev_action_history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        target_action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        sample_noise: torch.Tensor | None = None,
        mode: str = "sample",
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del task_id
        _validate_images(
            current_images,
            num_views=self.config.num_views,
            image_size=self.config.image_size,
            name="current_images",
        )
        _validate_future_images(
            future_images,
            num_future_frames=self.config.num_future_frames,
            num_views=self.config.num_views,
            image_size=self.config.image_size,
            name="future_images",
        )
        _validate_state(state, state_dim=self.config.state_dim, name="state")
        if self.config.idm_arch != "flow_transformer" and (
            prev_state_history is not None or prev_action_history is not None or history_mask is not None
        ):
            raise ValueError("IDM history conditioning is supported only with idm_arch='flow_transformer'.")
        if self.config.idm_arch == "flow_transformer":
            if mode == "sample":
                return self.sample_action(
                    current_images,
                    future_images,
                    state,
                    sample_noise=sample_noise,
                    wan_vae_latents=wan_vae_latents,
                    prev_state_history=prev_state_history,
                    prev_action_history=prev_action_history,
                    history_mask=history_mask,
                )
            if mode == "loss":
                if target_action is None or action_mask is None:
                    raise ValueError("target_action and action_mask are required for flow_transformer loss mode.")
                return self.flow_matching_loss(
                    current_images,
                    future_images,
                    state,
                    target_action,
                    action_mask,
                    wan_vae_latents=wan_vae_latents,
                    prev_state_history=prev_state_history,
                    prev_action_history=prev_action_history,
                    history_mask=history_mask,
                )
            raise ValueError(f"Unknown IDM forward mode: {mode}")
        if mode != "sample":
            raise ValueError(f"Mode {mode!r} is only supported by flow_transformer.")

        batch_size = current_images.shape[0]
        current = current_images.flatten(1, 2)
        future = future_images.flatten(1, 3)
        if self.config.idm_arch == "stacked":
            images = torch.cat([current, future], dim=1)
            latent = self.encoder(images)
            head_input = torch.cat([latent, state], dim=-1)
        elif self.config.idm_arch == "delta":
            repeated_current = current_images.unsqueeze(1).expand_as(future_images).flatten(1, 3)
            delta = future - repeated_current
            latent = torch.cat(
                [
                    self.current_encoder(current),
                    self.future_encoder(future),
                    self.delta_encoder(delta),
                ],
                dim=-1,
            )
            head_input = torch.cat([latent, state], dim=-1)
        elif self.config.idm_arch == "transformer":
            head_input = self.transition_encoder(current_images, future_images, state)
        else:
            raise ValueError(f"Unknown IDM architecture: {self.config.idm_arch}")
        if self.head is None:
            raise RuntimeError("Regression head is not initialized for flow_transformer.")
        action = self.head(head_input)
        return action.view(batch_size, self.config.action_horizon, self.config.action_dim)

    def context_action_loss(
        self,
        current_images: torch.Tensor,
        future_images: torch.Tensor,
        state: torch.Tensor,
        target_action: torch.Tensor,
        action_mask: torch.Tensor,
        wan_vae_latents: torch.Tensor | None = None,
        prev_state_history: torch.Tensor | None = None,
        prev_action_history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.config.idm_arch != "flow_transformer":
            raise ValueError("Context-to-action IDM loss is supported only for flow_transformer IDMs.")
        _validate_images(
            current_images,
            num_views=self.config.num_views,
            image_size=self.config.image_size,
            name="current_images",
        )
        _validate_future_images(
            future_images,
            num_future_frames=self.config.num_future_frames,
            num_views=self.config.num_views,
            image_size=self.config.image_size,
            name="future_images",
        )
        _validate_state(state, state_dim=self.config.state_dim, name="state")
        _validate_action_chunk(
            target_action,
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
            name="target_action",
        )
        if action_mask.ndim != 2 or tuple(action_mask.shape) != tuple(target_action.shape[:2]):
            raise ValueError(
                f"action_mask must have shape {tuple(target_action.shape[:2])}, got {tuple(action_mask.shape)}."
            )
        context = self._transition_context(current_images, future_images, state, wan_vae_latents=wan_vae_latents)
        history_tokens = self._history_tokens(prev_state_history, prev_action_history, history_mask)
        predicted_action = self.context_action_head(context, history_tokens=history_tokens)
        loss = _masked_action_mse(predicted_action, target_action, action_mask)
        return {
            "loss": loss,
            "predicted_action": predicted_action,
        }

    def flow_matching_loss(
        self,
        current_images: torch.Tensor,
        future_images: torch.Tensor,
        state: torch.Tensor,
        target_action: torch.Tensor,
        action_mask: torch.Tensor,
        wan_vae_latents: torch.Tensor | None = None,
        prev_state_history: torch.Tensor | None = None,
        prev_action_history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _validate_action_chunk(
            target_action,
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
            name="target_action",
        )
        if action_mask.ndim != 2 or tuple(action_mask.shape) != tuple(target_action.shape[:2]):
            raise ValueError(
                f"action_mask must have shape {tuple(target_action.shape[:2])}, got {tuple(action_mask.shape)}."
            )
        context, visual_context_tokens = self._transition_context_and_visual_tokens(
            current_images,
            future_images,
            state,
            wan_vae_latents=wan_vae_latents,
        )
        history_tokens = self._history_tokens(prev_state_history, prev_action_history, history_mask)
        noise = torch.randn_like(target_action)
        time_min = self.config.idm_flow_train_time_min
        time_max = self.config.idm_flow_train_time_max
        if time_min == time_max:
            time = torch.full(
                (target_action.shape[0],),
                time_min,
                device=target_action.device,
                dtype=target_action.dtype,
            )
        else:
            time = torch.rand(target_action.shape[0], device=target_action.device, dtype=target_action.dtype)
            if time_min != 0.0 or time_max != 1.0:
                time = time * (time_max - time_min) + time_min
        time_view = time.view(-1, 1, 1)
        noisy_action = (1.0 - time_view) * noise + time_view * target_action
        target_velocity = target_action - noise
        flow_head_kwargs = {}
        if visual_context_tokens is not None:
            flow_head_kwargs["visual_context_tokens"] = visual_context_tokens
        if history_tokens is not None:
            flow_head_kwargs["history_tokens"] = history_tokens
        predicted_velocity = self.flow_head(context, noisy_action, time, **flow_head_kwargs)
        flow_loss = _masked_action_mse(predicted_velocity, target_velocity, action_mask)
        endpoint_prediction = noisy_action + (1.0 - time_view) * predicted_velocity
        endpoint_loss = _masked_action_mse(endpoint_prediction, target_action, action_mask)
        endpoint_consistency_loss = flow_loss.new_zeros(())
        consistency_weight = self.config.idm_flow_endpoint_consistency_loss_weight
        if consistency_weight > 0.0:
            noise_2 = torch.randn_like(target_action)
            noisy_action_2 = (1.0 - time_view) * noise_2 + time_view * target_action
            predicted_velocity_2 = self.flow_head(context, noisy_action_2, time, **flow_head_kwargs)
            endpoint_prediction_2 = noisy_action_2 + (1.0 - time_view) * predicted_velocity_2
            endpoint_consistency_loss = _masked_action_mse(
                endpoint_prediction,
                endpoint_prediction_2,
                action_mask,
            )
        zero_start_endpoint_loss = flow_loss.new_zeros(())
        zero_start_weight = self.config.idm_flow_zero_start_endpoint_loss_weight
        if zero_start_weight > 0.0:
            zero_start_action = torch.zeros_like(target_action)
            zero_start_time = torch.zeros(
                target_action.shape[0],
                device=target_action.device,
                dtype=target_action.dtype,
            )
            zero_start_velocity = self.flow_head(context, zero_start_action, zero_start_time, **flow_head_kwargs)
            # Endpoint estimate from the deterministic zero start (action=0, t=0); (1 - t) = 1 there.
            zero_start_endpoint_prediction = zero_start_action + zero_start_velocity
            zero_start_endpoint_loss = _masked_action_mse(
                zero_start_endpoint_prediction,
                target_action,
                action_mask,
            )
        sampled_action_loss = flow_loss.new_zeros(())
        sampled_action_weight = self.config.idm_flow_sampled_action_loss_weight
        if sampled_action_weight > 0.0:
            # Supervise the deterministic 16-step sampler itself (not just the t=0 endpoint).
            # Use explicit zero noise so the rollout is deterministic even when the configured
            # idm_flow_sample_noise_scale is nonzero; the shape must match sample_action's
            # (batch * idm_flow_num_samples, action_horizon, action_dim) noise contract.
            num_samples = self.config.idm_flow_num_samples
            zero_sample_noise = target_action.new_zeros(
                target_action.shape[0] * num_samples,
                self.config.action_horizon,
                self.config.action_dim,
            )
            sampled_action = self.sample_action(
                current_images,
                future_images,
                state,
                sample_noise=zero_sample_noise,
                wan_vae_latents=wan_vae_latents,
                prev_state_history=prev_state_history,
                prev_action_history=prev_action_history,
                history_mask=history_mask,
            )
            sampled_action_loss = _masked_action_mse(sampled_action, target_action, action_mask)
        loss = (
            flow_loss
            + self.config.idm_flow_endpoint_loss_weight * endpoint_loss
            + consistency_weight * endpoint_consistency_loss
            + zero_start_weight * zero_start_endpoint_loss
            + sampled_action_weight * sampled_action_loss
        )
        return {
            "loss": loss,
            "flow_loss": flow_loss,
            "endpoint_loss": endpoint_loss,
            "endpoint_consistency_loss": endpoint_consistency_loss,
            "zero_start_endpoint_loss": zero_start_endpoint_loss,
            "sampled_action_loss": sampled_action_loss,
            "predicted_velocity": predicted_velocity,
            "endpoint_prediction": endpoint_prediction,
        }

    def sample_action(
        self,
        current_images: torch.Tensor,
        future_images: torch.Tensor,
        state: torch.Tensor,
        *,
        sample_noise: torch.Tensor | None = None,
        wan_vae_latents: torch.Tensor | None = None,
        prev_state_history: torch.Tensor | None = None,
        prev_action_history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = current_images.shape[0]
        num_samples = self.config.idm_flow_num_samples
        expected_noise_shape = (batch_size * num_samples, self.config.action_horizon, self.config.action_dim)
        if sample_noise is None:
            noise_scale = self.config.idm_flow_sample_noise_scale
            if noise_scale == 0.0:
                action = torch.zeros(
                    expected_noise_shape,
                    device=current_images.device,
                    dtype=current_images.dtype,
                )
            else:
                action = (
                    torch.randn(
                        expected_noise_shape,
                        device=current_images.device,
                        dtype=current_images.dtype,
                    )
                    * noise_scale
                )
        else:
            _validate_action_chunk(
                sample_noise,
                action_horizon=self.config.action_horizon,
                action_dim=self.config.action_dim,
                name="sample_noise",
            )
            if tuple(sample_noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"sample_noise must have shape {expected_noise_shape}, got {tuple(sample_noise.shape)}."
                )
            action = sample_noise.to(device=current_images.device, dtype=current_images.dtype)
        context, visual_context_tokens = self._transition_context_and_visual_tokens(
            current_images,
            future_images,
            state,
            wan_vae_latents=wan_vae_latents,
        )
        context = context.repeat_interleave(num_samples, dim=0)
        if visual_context_tokens is not None:
            visual_context_tokens = visual_context_tokens.repeat_interleave(num_samples, dim=0)
        history_tokens = self._history_tokens(prev_state_history, prev_action_history, history_mask)
        if history_tokens is not None:
            history_tokens = history_tokens.repeat_interleave(num_samples, dim=0)
        step_size = 1.0 / self.config.idm_flow_sampling_steps
        for step in range(self.config.idm_flow_sampling_steps):
            time = torch.full(
                (batch_size * num_samples,),
                (step + 0.5) * step_size,
                device=current_images.device,
                dtype=current_images.dtype,
            )
            flow_head_kwargs = {}
            if visual_context_tokens is not None:
                flow_head_kwargs["visual_context_tokens"] = visual_context_tokens
            if history_tokens is not None:
                flow_head_kwargs["history_tokens"] = history_tokens
            velocity = self.flow_head(context, action, time, **flow_head_kwargs)
            action = action + velocity * step_size
        action = action.view(batch_size, num_samples, self.config.action_horizon, self.config.action_dim)
        return action.mean(dim=1)
