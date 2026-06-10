from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import Dataset

FUTURE_LEAKAGE_KEYS = frozenset(
    {
        "future_image",
        "future_images",
        "future_frame",
        "future_frames",
        "future_latent",
        "future_latents",
        "future_wan_latents",
        "next_image",
        "next_images",
        "next_latents",
        "target_image",
        "target_images",
        "target_latents",
        "wan_future_latents",
    }
)
ConditioningMode = Literal["wan_prefix", "wan_prefix_state"]
TimestepConditioningMode = Literal["additive", "film"]
TimestepEmbeddingStyle = Literal["diffusion", "pi05"]
DecoderArch = Literal["encoder", "context_cross_attention", "suffix_prefix_cache", "joint_softmax_prefix_cache"]
# pi0.5 sin/cos posemb periods (OpenPI uses these for the action-flow timestep embedding).
_PI05_TIME_EMBEDDING_MIN_PERIOD = 4e-3
_PI05_TIME_EMBEDDING_MAX_PERIOD = 4.0
_VALID_DECODER_ARCHES = (
    "encoder",
    "context_cross_attention",
    "suffix_prefix_cache",
    "joint_softmax_prefix_cache",
)
_PREFIX_CACHE_DECODER_ARCHES = ("suffix_prefix_cache", "joint_softmax_prefix_cache")


def _validate_3d(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape (B, T, D), got {tuple(tensor.shape)}.")


def _validate_2d(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape (B, D), got {tuple(tensor.shape)}.")


def _fit_vector_width(vector: torch.Tensor, width: int) -> torch.Tensor:
    if vector.numel() >= width:
        return vector[:width]
    return torch.nn.functional.pad(vector, (0, width - vector.numel()))


def _model_device_and_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters(), None)
    if parameter is None:
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype


class SinusoidalTimeEmbedding(nn.Module):
    """Scalar-time sin/cos embedding.

    ``style="diffusion"`` keeps the legacy geometric ``max_period``-based frequencies (slow over
    ``t in [0, 1]``). ``style="pi05"`` matches OpenPI's pi0.5 ``posemb_sincos`` with
    ``min_period=4e-3``/``max_period=4.0``, giving frequencies that change quickly over ``[0, 1]``.
    """

    def __init__(self, dim: int, *, style: TimestepEmbeddingStyle = "diffusion", max_period: float = 10_000.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        self.dim = dim
        self.style = _normalize_timestep_embedding_style(style)
        if self.style == "pi05":
            if dim % 2 != 0:
                raise ValueError(f"timestep_embedding_style='pi05' requires an even dim, got {dim}.")
            fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float32)
            period = (
                _PI05_TIME_EMBEDDING_MIN_PERIOD
                * (_PI05_TIME_EMBEDDING_MAX_PERIOD / _PI05_TIME_EMBEDDING_MIN_PERIOD) ** fraction
            )
            self.register_buffer("frequencies", 2.0 * math.pi / period, persistent=False)
            return
        half_dim = dim // 2
        if half_dim > 0:
            exponent = -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32)
            exponent = exponent / max(half_dim - 1, 1)
            self.register_buffer("frequencies", torch.exp(exponent), persistent=False)
        else:
            self.register_buffer("frequencies", torch.empty(0), persistent=False)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError(f"time must have shape (B,), got {tuple(time.shape)}.")
        if self.style == "pi05":
            frequencies = self.frequencies.to(device=time.device, dtype=torch.float32)
            angles = time.to(dtype=torch.float32)[:, None] * frequencies[None]
            embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            return embedding.to(dtype=time.dtype)
        frequencies = self.frequencies.to(device=time.device, dtype=time.dtype)
        angles = time[:, None] * frequencies[None]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding


def _normalize_conditioning_mode(mode: str) -> ConditioningMode:
    if mode not in {"wan_prefix", "wan_prefix_state"}:
        raise ValueError(f"conditioning_mode must be 'wan_prefix' or 'wan_prefix_state', got {mode!r}.")
    return mode  # type: ignore[return-value]


def _normalize_timestep_conditioning(mode: str) -> TimestepConditioningMode:
    if mode not in {"additive", "film"}:
        raise ValueError(f"timestep_conditioning must be 'additive' or 'film', got {mode!r}.")
    return mode  # type: ignore[return-value]


def _normalize_timestep_embedding_style(style: str) -> TimestepEmbeddingStyle:
    if style not in {"diffusion", "pi05"}:
        raise ValueError(f"timestep_embedding_style must be 'diffusion' or 'pi05', got {style!r}.")
    return style  # type: ignore[return-value]


def _normalize_decoder_arch(arch: str) -> DecoderArch:
    if arch not in _VALID_DECODER_ARCHES:
        valid = ", ".join(repr(valid_arch) for valid_arch in _VALID_DECODER_ARCHES)
        raise ValueError(f"decoder_arch must be one of {valid}, got {arch!r}.")
    return arch  # type: ignore[return-value]


@dataclasses.dataclass(frozen=True)
class WanActionPrefixMemory:
    """Action-expert prefix K/V memory derived from Wan features, not native Wan attention KV."""

    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    batch_size: int
    prefix_length: int


@dataclasses.dataclass(frozen=True)
class ActionDenoisingContext:
    """Reusable action-expert context derived from Wan prefix tokens, not native Wan attention KV."""

    decoder_arch: DecoderArch
    batch_size: int
    context_tokens: tuple[torch.Tensor, ...] = ()
    encoded_context: torch.Tensor | None = None
    prefix_memory: WanActionPrefixMemory | None = None


class _CachedPrefixCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads, got {hidden_dim} and {num_heads}.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = tokens.shape
        return tokens.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

    def encode_memory(self, memory_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split_heads(self.key_projection(memory_tokens)), self._split_heads(
            self.value_projection(memory_tokens)
        )

    def forward(
        self, query_tokens: torch.Tensor, memory_keys: torch.Tensor, memory_values: torch.Tensor
    ) -> torch.Tensor:
        query = self._split_heads(self.query_projection(query_tokens))
        scores = torch.matmul(query, memory_keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, memory_values)
        attended = (
            attended.transpose(1, 2).contiguous().view(query_tokens.shape[0], query_tokens.shape[1], self.hidden_dim)
        )
        return self.output_projection(attended)


class _CachedJointPrefixSuffixAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads, got {hidden_dim} and {num_heads}.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = tokens.shape
        return tokens.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, _, token_count, _ = tokens.shape
        return tokens.transpose(1, 2).contiguous().view(batch_size, token_count, self.hidden_dim)

    def encode_memory(self, memory_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split_heads(self.key_projection(memory_tokens)), self._split_heads(
            self.value_projection(memory_tokens)
        )

    def forward(
        self, query_tokens: torch.Tensor, prefix_keys: torch.Tensor, prefix_values: torch.Tensor
    ) -> torch.Tensor:
        query = self._split_heads(self.query_projection(query_tokens))
        suffix_keys = self._split_heads(self.key_projection(query_tokens))
        suffix_values = self._split_heads(self.value_projection(query_tokens))
        keys = torch.cat([prefix_keys, suffix_keys], dim=-2)
        values = torch.cat([prefix_values, suffix_values], dim=-2)
        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, values)
        return self.output_projection(self._merge_heads(attended))


class _WanSuffixPrefixCacheLayer(nn.Module):
    def __init__(self, *, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.prefix_self_norm = nn.LayerNorm(hidden_dim)
        self.prefix_self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.prefix_self_dropout = nn.Dropout(dropout)
        self.prefix_ff_norm = nn.LayerNorm(hidden_dim)
        self.prefix_feedforward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.prefix_ff_dropout = nn.Dropout(dropout)

        self.action_self_norm = nn.LayerNorm(hidden_dim)
        self.action_self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.action_self_dropout = nn.Dropout(dropout)
        self.action_cross_norm = nn.LayerNorm(hidden_dim)
        self.action_cross_attention = _CachedPrefixCrossAttention(hidden_dim, num_heads, dropout)
        self.action_cross_dropout = nn.Dropout(dropout)
        self.action_ff_norm = nn.LayerNorm(hidden_dim)
        self.action_feedforward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.action_ff_dropout = nn.Dropout(dropout)

    def encode_prefix(self, prefix_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_prefix = self.prefix_self_norm(prefix_tokens)
        attended_prefix, _ = self.prefix_self_attention(normalized_prefix, normalized_prefix, normalized_prefix)
        prefix_tokens = prefix_tokens + self.prefix_self_dropout(attended_prefix)
        prefix_tokens = prefix_tokens + self.prefix_ff_dropout(
            self.prefix_feedforward(self.prefix_ff_norm(prefix_tokens))
        )
        memory_keys, memory_values = self.action_cross_attention.encode_memory(prefix_tokens)
        return prefix_tokens, memory_keys, memory_values

    def forward(
        self, action_tokens: torch.Tensor, memory_keys: torch.Tensor, memory_values: torch.Tensor
    ) -> torch.Tensor:
        normalized_actions = self.action_self_norm(action_tokens)
        attended_actions, _ = self.action_self_attention(normalized_actions, normalized_actions, normalized_actions)
        action_tokens = action_tokens + self.action_self_dropout(attended_actions)
        cross_attended = self.action_cross_attention(self.action_cross_norm(action_tokens), memory_keys, memory_values)
        action_tokens = action_tokens + self.action_cross_dropout(cross_attended)
        return action_tokens + self.action_ff_dropout(self.action_feedforward(self.action_ff_norm(action_tokens)))


class _WanJointSoftmaxPrefixCacheLayer(nn.Module):
    def __init__(self, *, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.prefix_self_norm = nn.LayerNorm(hidden_dim)
        self.prefix_self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.prefix_self_dropout = nn.Dropout(dropout)
        self.prefix_ff_norm = nn.LayerNorm(hidden_dim)
        self.prefix_feedforward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.prefix_ff_dropout = nn.Dropout(dropout)

        self.action_joint_norm = nn.LayerNorm(hidden_dim)
        self.action_joint_attention = _CachedJointPrefixSuffixAttention(hidden_dim, num_heads, dropout)
        self.action_joint_dropout = nn.Dropout(dropout)
        self.action_ff_norm = nn.LayerNorm(hidden_dim)
        self.action_feedforward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.action_ff_dropout = nn.Dropout(dropout)

    def encode_prefix(self, prefix_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_prefix = self.prefix_self_norm(prefix_tokens)
        attended_prefix, _ = self.prefix_self_attention(normalized_prefix, normalized_prefix, normalized_prefix)
        prefix_tokens = prefix_tokens + self.prefix_self_dropout(attended_prefix)
        prefix_tokens = prefix_tokens + self.prefix_ff_dropout(
            self.prefix_feedforward(self.prefix_ff_norm(prefix_tokens))
        )
        memory_keys, memory_values = self.action_joint_attention.encode_memory(self.action_joint_norm(prefix_tokens))
        return prefix_tokens, memory_keys, memory_values

    def forward(
        self, action_tokens: torch.Tensor, memory_keys: torch.Tensor, memory_values: torch.Tensor
    ) -> torch.Tensor:
        attended_actions = self.action_joint_attention(
            self.action_joint_norm(action_tokens), memory_keys, memory_values
        )
        action_tokens = action_tokens + self.action_joint_dropout(attended_actions)
        return action_tokens + self.action_ff_dropout(self.action_feedforward(self.action_ff_norm(action_tokens)))


class _WanSuffixPrefixCacheDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        joint_softmax: bool = False,
    ) -> None:
        super().__init__()
        layer_class = _WanJointSoftmaxPrefixCacheLayer if joint_softmax else _WanSuffixPrefixCacheLayer
        self.layers = nn.ModuleList(
            [
                layer_class(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def encode_prefix_memory(self, context_tokens: list[torch.Tensor]) -> WanActionPrefixMemory:
        prefix_tokens = torch.cat(context_tokens, dim=1)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for layer in self.layers:
            prefix_tokens, layer_keys, layer_values = layer.encode_prefix(prefix_tokens)
            keys.append(layer_keys)
            values.append(layer_values)
        return WanActionPrefixMemory(
            keys=tuple(keys),
            values=tuple(values),
            batch_size=prefix_tokens.shape[0],
            prefix_length=prefix_tokens.shape[1],
        )

    def forward(self, action_tokens: torch.Tensor, memory: WanActionPrefixMemory) -> torch.Tensor:
        if len(memory.keys) != len(self.layers) or len(memory.values) != len(self.layers):
            raise ValueError(
                f"memory must contain {len(self.layers)} key/value layers, got "
                f"{len(memory.keys)} keys and {len(memory.values)} values."
            )
        if memory.batch_size != action_tokens.shape[0]:
            raise ValueError(f"memory batch size must be {action_tokens.shape[0]}, got {memory.batch_size}.")
        for layer, memory_keys, memory_values in zip(self.layers, memory.keys, memory.values, strict=True):
            action_tokens = layer(action_tokens, memory_keys, memory_values)
        return action_tokens


class WanPi05ActionExpert(nn.Module):
    """Compact pi0.5-style action flow over Wan-derived prefix tokens."""

    def __init__(
        self,
        *,
        prefix_dim: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int | None = None,
        dropout: float = 0.0,
        conditioning_mode: ConditioningMode = "wan_prefix_state",
        timestep_conditioning: TimestepConditioningMode = "additive",
        timestep_embedding_style: TimestepEmbeddingStyle = "diffusion",
        decoder_arch: DecoderArch = "encoder",
    ) -> None:
        super().__init__()
        if prefix_dim <= 0:
            raise ValueError(f"prefix_dim must be positive, got {prefix_dim}.")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}.")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads, got {hidden_dim} and {num_heads}.")
        if dropout < 0.0:
            raise ValueError(f"dropout must be non-negative, got {dropout}.")

        self.prefix_dim = prefix_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim
        self.conditioning_mode = _normalize_conditioning_mode(conditioning_mode)
        self.condition_on_state = self.conditioning_mode == "wan_prefix_state"
        self.timestep_conditioning = _normalize_timestep_conditioning(timestep_conditioning)
        self.timestep_embedding_style = _normalize_timestep_embedding_style(timestep_embedding_style)
        self.decoder_arch = _normalize_decoder_arch(decoder_arch)

        self.prefix_projection = nn.Sequential(
            nn.Linear(prefix_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        if self.condition_on_state:
            self.state_projection = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.state_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        else:
            self.state_projection = None
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.horizon_embedding = nn.Embedding(action_horizon, hidden_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim, style=self.timestep_embedding_style),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prefix_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.action_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        feedforward_dim = ff_dim or hidden_dim * 4
        if self.decoder_arch == "encoder":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        elif self.decoder_arch == "context_cross_attention":
            context_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(context_layer, num_layers=num_layers)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        else:
            self.suffix_prefix_decoder = _WanSuffixPrefixCacheDecoder(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=feedforward_dim,
                dropout=dropout,
                joint_softmax=self.decoder_arch == "joint_softmax_prefix_cache",
            )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, action_dim)
        if self.timestep_conditioning == "film":
            self.timestep_film = nn.Linear(hidden_dim, hidden_dim * 2)

    def _validate_prefix_state_inputs(self, prefix_tokens: torch.Tensor, state: torch.Tensor) -> tuple[int, int]:
        _validate_3d(prefix_tokens, name="prefix_tokens")
        _validate_2d(state, name="state")
        batch_size, prefix_count, prefix_dim = prefix_tokens.shape
        if prefix_count <= 0:
            raise ValueError("prefix_tokens must contain at least one token.")
        if prefix_dim != self.prefix_dim:
            raise ValueError(f"prefix_tokens last dim must be {self.prefix_dim}, got {prefix_dim}.")
        if tuple(state.shape) != (batch_size, self.state_dim):
            raise ValueError(f"state must have shape ({batch_size}, {self.state_dim}), got {tuple(state.shape)}.")
        return batch_size, prefix_count

    def _validate_action_time_inputs(
        self,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        *,
        batch_size: int,
    ) -> None:
        _validate_3d(noisy_actions, name="noisy_actions")
        expected_action_shape = (batch_size, self.action_horizon, self.action_dim)
        if tuple(noisy_actions.shape) != expected_action_shape:
            raise ValueError(
                f"noisy_actions must have shape {expected_action_shape}, got {tuple(noisy_actions.shape)}."
            )
        if time.ndim != 1 or tuple(time.shape) != (batch_size,):
            raise ValueError(f"time must have shape ({batch_size},), got {tuple(time.shape)}.")

    def _build_context_tokens(self, prefix_tokens: torch.Tensor, state: torch.Tensor) -> list[torch.Tensor]:
        prefix_tokens = self.prefix_projection(prefix_tokens) + self.prefix_type_embedding
        context_tokens = [prefix_tokens]
        if self.condition_on_state:
            if self.state_projection is None:
                raise RuntimeError("state_projection is missing while condition_on_state=True.")
            state_token = self.state_projection(state).unsqueeze(1) + self.state_type_embedding
            context_tokens.append(state_token)
        return context_tokens

    def _build_action_tokens(
        self, noisy_actions: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = noisy_actions.shape[0]
        device = noisy_actions.device
        action_tokens = self.action_projection(noisy_actions) + self.action_type_embedding
        horizon = self.horizon_embedding(torch.arange(self.action_horizon, device=device))
        action_tokens = action_tokens + horizon.view(1, self.action_horizon, self.hidden_dim)
        time_features = self.time_mlp(time)
        if self.timestep_conditioning == "additive":
            action_tokens = action_tokens + time_features.view(batch_size, 1, self.hidden_dim)
        return action_tokens, time_features

    def _project_action_output(
        self,
        action_output: torch.Tensor,
        time_features: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        action_output = self.output_norm(action_output)
        if self.timestep_conditioning == "film":
            timestep_film = getattr(self, "timestep_film", None)
            if timestep_film is None:
                raise RuntimeError("timestep_film is missing while timestep_conditioning='film'.")
            scale, shift = timestep_film(time_features).chunk(2, dim=-1)
            action_output = action_output * (1.0 + scale.tanh().view(batch_size, 1, self.hidden_dim))
            action_output = action_output + shift.view(batch_size, 1, self.hidden_dim)
        return self.velocity_head(action_output)

    def _encode_with_prefix_encoder(
        self,
        context_tokens: list[torch.Tensor],
        action_tokens: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        context_len = sum(token.shape[1] for token in context_tokens)
        tokens = torch.cat([*context_tokens, action_tokens], dim=1)
        attention_mask = torch.zeros(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=device)
        attention_mask[:context_len, context_len:] = True
        encoder = getattr(self, "encoder", None)
        if encoder is None:
            raise RuntimeError("encoder is missing while decoder_arch='encoder'.")
        encoded = encoder(tokens, mask=attention_mask)
        return encoded[:, context_len:]

    def _decode_with_context_cross_attention(
        self,
        context_tokens: list[torch.Tensor],
        action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        encoded_context = self._encode_context_for_cross_attention(context_tokens)
        return self._decode_with_encoded_context(encoded_context, action_tokens)

    def _encode_context_for_cross_attention(self, context_tokens: Sequence[torch.Tensor]) -> torch.Tensor:
        context_encoder = getattr(self, "context_encoder", None)
        if context_encoder is None:
            raise RuntimeError("context encoder module is missing while decoder_arch='context_cross_attention'.")
        context = torch.cat(context_tokens, dim=1)
        return context_encoder(context)

    def _decode_with_encoded_context(self, encoded_context: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        decoder = getattr(self, "decoder", None)
        if decoder is None:
            raise RuntimeError("decoder module is missing while decoder_arch='context_cross_attention'.")
        return decoder(action_tokens, encoded_context)

    def _decode_with_suffix_prefix_cache(
        self,
        memory: WanActionPrefixMemory,
        action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        suffix_prefix_decoder = getattr(self, "suffix_prefix_decoder", None)
        if suffix_prefix_decoder is None:
            raise RuntimeError(f"suffix-prefix cache modules are missing while decoder_arch={self.decoder_arch!r}.")
        return suffix_prefix_decoder(action_tokens, memory)

    def _encode_prefix_memory_from_context(self, context_tokens: Sequence[torch.Tensor]) -> WanActionPrefixMemory:
        if self.decoder_arch not in _PREFIX_CACHE_DECODER_ARCHES:
            valid = " or ".join(repr(arch) for arch in _PREFIX_CACHE_DECODER_ARCHES)
            raise RuntimeError(f"encode_prefix_memory is only available for decoder_arch={valid}.")
        suffix_prefix_decoder = getattr(self, "suffix_prefix_decoder", None)
        if suffix_prefix_decoder is None:
            raise RuntimeError(f"suffix-prefix cache modules are missing while decoder_arch={self.decoder_arch!r}.")
        return suffix_prefix_decoder.encode_prefix_memory(list(context_tokens))

    def encode_prefix_memory(self, prefix_tokens: torch.Tensor, state: torch.Tensor) -> WanActionPrefixMemory:
        batch_size, prefix_count = self._validate_prefix_state_inputs(prefix_tokens, state)
        del batch_size, prefix_count
        return self._encode_prefix_memory_from_context(self._build_context_tokens(prefix_tokens, state))

    def prepare_action_context(self, prefix_tokens: torch.Tensor, state: torch.Tensor) -> ActionDenoisingContext:
        """Prepare reusable action-expert context once for a flow-denoising chunk."""
        batch_size, prefix_count = self._validate_prefix_state_inputs(prefix_tokens, state)
        del prefix_count
        context_tokens = tuple(self._build_context_tokens(prefix_tokens, state))
        if self.decoder_arch == "encoder":
            return ActionDenoisingContext(
                decoder_arch=self.decoder_arch,
                batch_size=batch_size,
                context_tokens=context_tokens,
            )
        if self.decoder_arch == "context_cross_attention":
            return ActionDenoisingContext(
                decoder_arch=self.decoder_arch,
                batch_size=batch_size,
                encoded_context=self._encode_context_for_cross_attention(context_tokens),
            )
        return ActionDenoisingContext(
            decoder_arch=self.decoder_arch,
            batch_size=batch_size,
            prefix_memory=self._encode_prefix_memory_from_context(context_tokens),
        )

    def forward_with_action_context(
        self,
        context: ActionDenoisingContext,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if context.decoder_arch != self.decoder_arch:
            raise ValueError(
                f"action context was prepared for decoder_arch={context.decoder_arch!r}, "
                f"but this model uses decoder_arch={self.decoder_arch!r}."
            )
        self._validate_action_time_inputs(noisy_actions, time, batch_size=context.batch_size)
        device = noisy_actions.device
        action_tokens, time_features = self._build_action_tokens(noisy_actions, time)
        if self.decoder_arch == "encoder":
            if not context.context_tokens:
                raise ValueError("encoder action context is missing context_tokens.")
            action_output = self._encode_with_prefix_encoder(list(context.context_tokens), action_tokens, device=device)
        elif self.decoder_arch == "context_cross_attention":
            if context.encoded_context is None:
                raise ValueError("context_cross_attention action context is missing encoded_context.")
            action_output = self._decode_with_encoded_context(context.encoded_context, action_tokens)
        else:
            if context.prefix_memory is None:
                raise ValueError(f"{self.decoder_arch!r} action context is missing prefix_memory.")
            action_output = self._decode_with_suffix_prefix_cache(context.prefix_memory, action_tokens)
        return self._project_action_output(action_output, time_features, batch_size=context.batch_size)

    def forward_with_prefix_memory(
        self,
        memory: WanActionPrefixMemory,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if self.decoder_arch not in _PREFIX_CACHE_DECODER_ARCHES:
            valid = " or ".join(repr(arch) for arch in _PREFIX_CACHE_DECODER_ARCHES)
            raise RuntimeError(f"forward_with_prefix_memory is only available for decoder_arch={valid}.")
        context = ActionDenoisingContext(
            decoder_arch=self.decoder_arch,
            batch_size=memory.batch_size,
            prefix_memory=memory,
        )
        return self.forward_with_action_context(context, noisy_actions, time)

    def forward(
        self,
        prefix_tokens: torch.Tensor,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        context = self.prepare_action_context(prefix_tokens, state)
        return self.forward_with_action_context(context, noisy_actions, time)


def sample_pi05_time(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample pi0.5 train times with t=1 as noise and t=0 as data."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    uniform = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    return uniform.pow(1.0 / 1.5) * 0.999 + 0.001


def pi05_flow_targets(
    actions: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_3d(actions, name="actions")
    if tuple(noise.shape) != tuple(actions.shape):
        raise ValueError(f"noise must have shape {tuple(actions.shape)}, got {tuple(noise.shape)}.")
    if time.ndim != 1 or tuple(time.shape) != (actions.shape[0],):
        raise ValueError(f"time must have shape ({actions.shape[0]},), got {tuple(time.shape)}.")
    time_view = time.view(-1, 1, 1)
    noisy_actions = time_view * noise + (1.0 - time_view) * actions
    target_velocity = noise - actions
    return noisy_actions, target_velocity


def masked_action_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor | None = None,
    *,
    action_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    numerator, count = masked_action_mse_per_sample_parts(
        predicted,
        target,
        action_mask,
        action_weights=action_weights,
    )
    return numerator.sum() / count.sum().clamp_min(1.0)


def masked_action_mse_per_sample(
    predicted: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor | None = None,
    *,
    action_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    numerator, count = masked_action_mse_per_sample_parts(
        predicted,
        target,
        action_mask,
        action_weights=action_weights,
    )
    return numerator / count.clamp_min(1.0)


def masked_action_mse_per_sample_parts(
    predicted: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor | None = None,
    *,
    action_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(predicted.shape) != tuple(target.shape):
        raise ValueError(
            f"predicted and target shapes must match, got {tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    squared = (predicted - target).square()
    if action_weights is not None:
        if action_weights.ndim != 1 or action_weights.shape[0] != predicted.shape[-1]:
            raise ValueError(
                f"action_weights must have shape ({predicted.shape[-1]},), got {tuple(action_weights.shape)}."
            )
        weights = action_weights.to(device=predicted.device, dtype=predicted.dtype).view(1, 1, -1)
        squared = squared * weights
    if action_mask is None:
        count = torch.full(
            (predicted.shape[0],),
            predicted.shape[1] * predicted.shape[2],
            device=predicted.device,
            dtype=predicted.dtype,
        )
        return squared.sum(dim=(1, 2)), count
    if action_mask.ndim != 2 or tuple(action_mask.shape) != tuple(predicted.shape[:2]):
        raise ValueError(f"action_mask must have shape {tuple(predicted.shape[:2])}, got {tuple(action_mask.shape)}.")
    mask = action_mask.to(device=predicted.device, dtype=predicted.dtype).unsqueeze(-1)
    squared = squared * mask
    return squared.sum(dim=(1, 2)), mask.expand_as(predicted).sum(dim=(1, 2))


def flow_matching_loss(
    model: WanPi05ActionExpert,
    prefix_tokens: torch.Tensor,
    state: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    *,
    action_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    _, numerator, count = flow_matching_loss_per_sample_parts(
        model,
        prefix_tokens,
        state,
        actions,
        action_mask,
        generator=generator,
        action_weights=action_weights,
    )
    return numerator.sum() / count.sum().clamp_min(1.0)


def flow_matching_loss_per_sample(
    model: WanPi05ActionExpert,
    prefix_tokens: torch.Tensor,
    state: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    *,
    action_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per_sample_loss, _, _ = flow_matching_loss_per_sample_parts(
        model,
        prefix_tokens,
        state,
        actions,
        action_mask,
        generator=generator,
        action_weights=action_weights,
    )
    return per_sample_loss


def flow_matching_loss_per_sample_parts(
    model: WanPi05ActionExpert,
    prefix_tokens: torch.Tensor,
    state: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    *,
    action_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
    time = sample_pi05_time(actions.shape[0], device=actions.device, dtype=actions.dtype, generator=generator)
    noisy_actions, target_velocity = pi05_flow_targets(actions, noise, time)
    predicted_velocity = model(prefix_tokens, state, noisy_actions, time)
    numerator, count = masked_action_mse_per_sample_parts(
        predicted_velocity,
        target_velocity,
        action_mask,
        action_weights=action_weights,
    )
    return numerator / count.clamp_min(1.0), numerator, count


@torch.no_grad()
def sample_actions(
    model: WanPi05ActionExpert,
    prefix_tokens: torch.Tensor,
    state: torch.Tensor,
    *,
    num_steps: int = 16,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    _validate_3d(prefix_tokens, name="prefix_tokens")
    batch_size = prefix_tokens.shape[0]
    expected_shape = (batch_size, model.action_horizon, model.action_dim)
    if noise is None:
        actions = torch.randn(
            expected_shape, device=prefix_tokens.device, dtype=prefix_tokens.dtype, generator=generator
        )
    else:
        if tuple(noise.shape) != expected_shape:
            raise ValueError(f"noise must have shape {expected_shape}, got {tuple(noise.shape)}.")
        actions = noise.to(device=prefix_tokens.device, dtype=prefix_tokens.dtype).clone()

    step_size = 1.0 / num_steps
    was_training = model.training
    model.eval()
    try:
        context = model.prepare_action_context(prefix_tokens, state)
        for step in range(num_steps):
            time_value = 1.0 - step * step_size
            time = torch.full((batch_size,), time_value, device=actions.device, dtype=actions.dtype)
            velocity = model.forward_with_action_context(context, actions, time)
            actions = actions - velocity * step_size
        return actions
    finally:
        model.train(was_training)


@dataclasses.dataclass(frozen=True)
class LoadedWanPi05ActionExpert:
    model: WanPi05ActionExpert
    checkpoint_path: Path
    action_normalization: Mapping[str, Any]
    action_norm_mean: torch.Tensor | None = None
    action_norm_std: torch.Tensor | None = None
    action_norm_by_task: Mapping[str, tuple[torch.Tensor, torch.Tensor]] = dataclasses.field(default_factory=dict)
    args: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    metrics: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    wan_action_mode: str | None = None


@dataclasses.dataclass(frozen=True)
class CachedWanPrefixSample:
    prefix_tokens: torch.Tensor
    state: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor
    task: str
    metadata: Mapping[str, Any]


def _forbidden_key_paths(value: Any, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in FUTURE_LEAKAGE_KEYS:
                paths.append(path)
            paths.extend(_forbidden_key_paths(child, prefix=path))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, child in enumerate(value):
            paths.extend(_forbidden_key_paths(child, prefix=f"{prefix}[{index}]"))
    return paths


def _load_pt_row(path: Path) -> Mapping[str, Any]:
    row = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(row, Mapping):
        raise ValueError(f"Cache row {path} must be a mapping, got {type(row).__name__}.")
    forbidden_paths = _forbidden_key_paths(row)
    if forbidden_paths:
        joined = ", ".join(sorted(forbidden_paths))
        raise ValueError(f"Cached Wan prefix rows must be current-only; found future leakage keys: {joined}.")
    missing = [key for key in ("prefix_tokens", "state", "actions") if key not in row]
    if missing:
        raise ValueError(f"Cache row {path} is missing required key(s): {', '.join(missing)}.")
    return row


def _optional_wan_action_mode(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"wan_action_mode in {context} must be a non-empty string, got {value!r}.")
    return value


def wan_action_mode_from_row(row: Mapping[str, Any], row_path: str | Path) -> str | None:
    metadata = row.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"metadata in {row_path} must be a mapping when present.")
    metadata_mode = _optional_wan_action_mode(
        metadata.get("wan_action_mode"),
        context=f"metadata in {row_path}",
    )
    row_mode = _optional_wan_action_mode(row.get("wan_action_mode"), context=str(row_path))
    if metadata_mode is not None and row_mode is not None and metadata_mode != row_mode:
        raise ValueError(
            f"wan_action_mode in {row_path} disagrees between row ({row_mode}) and metadata ({metadata_mode})."
        )
    return metadata_mode if metadata_mode is not None else row_mode


def find_prefix_cache_rows(cache_path: str | Path) -> list[Path]:
    path = Path(cache_path)
    if path.is_file():
        return [path] if path.suffix == ".pt" else []
    if not path.exists():
        return []
    return sorted(child for child in path.iterdir() if child.is_file() and child.suffix == ".pt")


class CachedWanPrefixActionDataset(Dataset[dict[str, Any]]):
    """Dataset for `.pt` rows containing current-image/task prefix tokens and action chunks."""

    def __init__(self, cache_path: str | Path) -> None:
        self.cache_path = Path(cache_path)
        self.cache_paths = find_prefix_cache_rows(self.cache_path)
        if not self.cache_paths:
            raise FileNotFoundError(f"No .pt cached Wan prefix rows found at {self.cache_path}.")

    def __len__(self) -> int:
        return len(self.cache_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_path = self.cache_paths[index]
        row = _load_pt_row(row_path)
        prefix_tokens = torch.as_tensor(row["prefix_tokens"], dtype=torch.float32)
        state = torch.as_tensor(row["state"], dtype=torch.float32)
        actions = torch.as_tensor(row["actions"], dtype=torch.float32)
        if prefix_tokens.ndim != 2:
            raise ValueError(f"prefix_tokens in {row_path} must have shape (N, D), got {tuple(prefix_tokens.shape)}.")
        if state.ndim != 1:
            raise ValueError(f"state in {row_path} must have shape (D,), got {tuple(state.shape)}.")
        if actions.ndim != 2:
            raise ValueError(f"actions in {row_path} must have shape (H, A), got {tuple(actions.shape)}.")
        if "action_mask" in row and row["action_mask"] is not None:
            action_mask = torch.as_tensor(row["action_mask"], dtype=torch.float32)
            if tuple(action_mask.shape) != tuple(actions.shape[:1]):
                raise ValueError(
                    f"action_mask in {row_path} must have shape ({actions.shape[0]},), got {tuple(action_mask.shape)}."
                )
        else:
            action_mask = torch.ones(actions.shape[0], dtype=torch.float32)

        metadata = row.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise ValueError(f"metadata in {row_path} must be a mapping when present.")
        wan_action_mode = wan_action_mode_from_row(row, row_path)
        task = row.get("task", metadata.get("task", ""))
        return {
            "prefix_tokens": prefix_tokens,
            "state": state,
            "actions": actions,
            "action_mask": action_mask,
            "task": str(task),
            "metadata": dict(metadata),
            "wan_action_mode": wan_action_mode,
            "cache_path": str(row_path),
        }


def load_pi05_wan_prefix_cache_row(row_path: str | Path) -> dict[str, Any]:
    path = Path(row_path)
    if not path.is_file():
        raise FileNotFoundError(f"Wan prefix cache row does not exist: {path}.")
    dataset = CachedWanPrefixActionDataset(path)
    return dataset[0]


def load_cached_prefix_dataset(
    cache_path: str | Path,
    *,
    real_wan_prefix_cache: bool = False,
) -> CachedWanPrefixActionDataset:
    if find_prefix_cache_rows(cache_path):
        return CachedWanPrefixActionDataset(cache_path)
    if real_wan_prefix_cache:
        raise FileNotFoundError(
            "Real Wan prefix caching was requested, but no cached .pt prefix rows were found at "
            f"{Path(cache_path)}. This scaffold does not run Wan2.2 5B inference; create the prefix cache "
            "offline from current image + task text only, or use the fake cache path for a smoke test."
        )
    raise FileNotFoundError(f"No .pt cached Wan prefix rows found at {Path(cache_path)}.")


def write_fake_prefix_cache(
    cache_dir: str | Path,
    *,
    num_rows: int = 16,
    prefix_tokens: int = 4,
    prefix_dim: int = 32,
    state_dim: int = 8,
    action_horizon: int = 6,
    action_dim: int = 4,
    seed: int = 0,
) -> list[Path]:
    if num_rows <= 0:
        raise ValueError(f"num_rows must be positive, got {num_rows}.")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    paths: list[Path] = []
    horizon_ramp = torch.linspace(-0.5, 0.5, action_horizon).unsqueeze(-1)
    for row_index in range(num_rows):
        task_id = row_index % 3
        prefix = torch.randn(prefix_tokens, prefix_dim, generator=generator) + task_id * 0.05
        state = torch.randn(state_dim, generator=generator)
        prefix_signal = _fit_vector_width(prefix.mean(dim=0), action_dim)
        state_signal = _fit_vector_width(state, action_dim)
        actions = 0.25 * prefix_signal.view(1, -1) + 0.5 * state_signal.view(1, -1)
        actions = actions + 0.1 * horizon_ramp + task_id * 0.03
        row = {
            "prefix_tokens": prefix,
            "state": state,
            "actions": actions.to(dtype=torch.float32),
            "action_mask": torch.ones(action_horizon, dtype=torch.float32),
            "task": f"fake_task_{task_id}",
            "metadata": {
                "cache_kind": "fake_current_only_prefix",
                "row_index": row_index,
                "task": f"fake_task_{task_id}",
            },
        }
        path = cache_dir / f"row_{row_index:05d}.pt"
        torch.save(row, path)
        paths.append(path)
    manifest = {
        "cache_kind": "fake_current_only_prefix",
        "num_rows": num_rows,
        "contains_future_images": False,
        "contains_future_latents": False,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return paths


def _require_mapping(value: Any, *, name: str, checkpoint_path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} in {checkpoint_path} must be a mapping, got {type(value).__name__}.")
    return value


def _checkpoint_wan_action_mode(checkpoint: Mapping[str, Any], checkpoint_path: Path) -> str | None:
    modes: dict[str, str] = {}
    args = checkpoint.get("args", {})
    if args is not None:
        args = _require_mapping(args, name="args", checkpoint_path=checkpoint_path)
        mode = _optional_wan_action_mode(args.get("wan_action_mode"), context=f"args in {checkpoint_path}")
        if mode is not None:
            modes["args"] = mode
    metrics = checkpoint.get("metrics", {})
    if metrics is not None:
        metrics = _require_mapping(metrics, name="metrics", checkpoint_path=checkpoint_path)
        mode = _optional_wan_action_mode(metrics.get("wan_action_mode"), context=f"metrics in {checkpoint_path}")
        if mode is not None:
            modes["metrics"] = mode
    top_level_mode = _optional_wan_action_mode(checkpoint.get("wan_action_mode"), context=str(checkpoint_path))
    if top_level_mode is not None:
        modes["checkpoint"] = top_level_mode
    if len(set(modes.values())) > 1:
        details = ", ".join(f"{source}={mode}" for source, mode in sorted(modes.items()))
        raise ValueError(f"wan_action_mode metadata in {checkpoint_path} is inconsistent: {details}.")
    if not modes:
        return None
    return next(iter(modes.values()))


def _normalization_tensor(
    normalization: Mapping[str, Any],
    *,
    key: str,
    action_dim: int,
    checkpoint_path: Path,
) -> torch.Tensor:
    if key not in normalization:
        raise ValueError(f"action_normalization in {checkpoint_path} is enabled but missing {key!r}.")
    tensor = torch.as_tensor(normalization[key], dtype=torch.float32)
    if tensor.ndim != 1 or tensor.shape[0] != action_dim:
        raise ValueError(
            f"action_normalization[{key!r}] in {checkpoint_path} must have shape ({action_dim},), "
            f"got {tuple(tensor.shape)}."
        )
    return tensor.detach().cpu()


def _load_action_normalization(
    checkpoint: Mapping[str, Any],
    *,
    action_dim: int,
    checkpoint_path: Path,
) -> tuple[
    Mapping[str, Any], torch.Tensor | None, torch.Tensor | None, Mapping[str, tuple[torch.Tensor, torch.Tensor]]
]:
    raw_normalization = checkpoint.get("action_normalization", {"enabled": False})
    normalization = _require_mapping(
        raw_normalization,
        name="action_normalization",
        checkpoint_path=checkpoint_path,
    )
    enabled = normalization.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"action_normalization['enabled'] in {checkpoint_path} must be a bool, got {enabled!r}.")
    if not enabled:
        return dict(normalization), None, None, {}
    scope = normalization.get("scope", "global")
    if scope not in {"global", "per_task"}:
        raise ValueError(f"action_normalization['scope'] in {checkpoint_path} must be 'global' or 'per_task'.")
    if scope == "per_task":
        raw_tasks = normalization.get("tasks")
        if not isinstance(raw_tasks, Mapping) or not raw_tasks:
            raise ValueError(
                f"action_normalization['tasks'] in {checkpoint_path} must be a non-empty mapping for per-task stats."
            )
        task_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for task, raw_stats in raw_tasks.items():
            stats = _require_mapping(
                raw_stats, name=f"action_normalization['tasks'][{task!r}]", checkpoint_path=checkpoint_path
            )
            mean = _normalization_tensor(stats, key="mean", action_dim=action_dim, checkpoint_path=checkpoint_path)
            std = _normalization_tensor(stats, key="std", action_dim=action_dim, checkpoint_path=checkpoint_path)
            if bool((std <= 0).any()):
                raise ValueError(
                    f"action_normalization['tasks'][{task!r}]['std'] in {checkpoint_path} must be strictly positive."
                )
            task_stats[str(task)] = (mean, std)
        return dict(normalization), None, None, task_stats
    mean = _normalization_tensor(normalization, key="mean", action_dim=action_dim, checkpoint_path=checkpoint_path)
    std = _normalization_tensor(normalization, key="std", action_dim=action_dim, checkpoint_path=checkpoint_path)
    if bool((std <= 0).any()):
        raise ValueError(f"action_normalization['std'] in {checkpoint_path} must be strictly positive.")
    return dict(normalization), mean, std, {}


def load_wan_pi05_action_expert_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> LoadedWanPi05ActionExpert:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = _require_mapping(checkpoint, name="checkpoint", checkpoint_path=path)
    if "model_kwargs" not in checkpoint:
        raise ValueError(f"Checkpoint {path} is missing required key 'model_kwargs'.")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {path} is missing required key 'model_state_dict'.")
    model_kwargs = dict(_require_mapping(checkpoint["model_kwargs"], name="model_kwargs", checkpoint_path=path))
    state_dict = _require_mapping(checkpoint["model_state_dict"], name="model_state_dict", checkpoint_path=path)
    model = WanPi05ActionExpert(**model_kwargs).to(torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    normalization, action_norm_mean, action_norm_std, action_norm_by_task = _load_action_normalization(
        checkpoint,
        action_dim=model.action_dim,
        checkpoint_path=path,
    )
    args = checkpoint.get("args", {})
    metrics = checkpoint.get("metrics", {})
    return LoadedWanPi05ActionExpert(
        model=model,
        checkpoint_path=path,
        action_normalization=normalization,
        action_norm_mean=action_norm_mean,
        action_norm_std=action_norm_std,
        action_norm_by_task=action_norm_by_task,
        args=dict(_require_mapping(args, name="args", checkpoint_path=path)) if args is not None else {},
        metrics=dict(_require_mapping(metrics, name="metrics", checkpoint_path=path)) if metrics is not None else {},
        wan_action_mode=_checkpoint_wan_action_mode(checkpoint, path),
    )


def _denormalize_actions(
    actions: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or std is None:
        return actions
    mean = mean.to(device=actions.device, dtype=actions.dtype).view(1, 1, -1)
    std = std.to(device=actions.device, dtype=actions.dtype).view(1, 1, -1)
    return actions * std + mean


def _normalize_prediction_tasks(tasks: str | Sequence[str] | None, *, batch_size: int) -> list[str] | None:
    if tasks is None:
        return None
    if isinstance(tasks, str):
        if batch_size != 1:
            raise ValueError(f"A single task string can only be used with batch size 1, got {batch_size}.")
        return [tasks]
    normalized = [str(task) for task in tasks]
    if len(normalized) != batch_size:
        raise ValueError(f"tasks length {len(normalized)} must match batch size {batch_size}.")
    return normalized


def _denormalize_actions_by_task(
    actions: torch.Tensor,
    task_stats: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    tasks: Sequence[str] | None,
    *,
    checkpoint_path: Path,
) -> torch.Tensor:
    if tasks is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} uses per-task action normalization; pass task labels for prediction."
        )
    missing = sorted({task for task in tasks if task not in task_stats})
    if missing:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing action normalization stats for task(s): {missing}.")
    means = torch.stack([task_stats[task][0] for task in tasks]).to(device=actions.device, dtype=actions.dtype)
    stds = torch.stack([task_stats[task][1] for task in tasks]).to(device=actions.device, dtype=actions.dtype)
    return actions * stds.unsqueeze(1) + means.unsqueeze(1)


@torch.no_grad()
def predict_denormalized_action_chunk(
    loaded: LoadedWanPi05ActionExpert,
    prefix_tokens: torch.Tensor,
    state: torch.Tensor,
    *,
    num_steps: int = 16,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    tasks: str | Sequence[str] | None = None,
) -> torch.Tensor:
    device, dtype = _model_device_and_dtype(loaded.model)
    prefix_tokens = torch.as_tensor(prefix_tokens, device=device, dtype=dtype)
    state = torch.as_tensor(state, device=device, dtype=dtype)
    batched = True
    if prefix_tokens.ndim == 2 and state.ndim == 1:
        prefix_tokens = prefix_tokens.unsqueeze(0)
        state = state.unsqueeze(0)
        batched = False
    elif prefix_tokens.ndim != 3 or state.ndim != 2:
        raise ValueError(
            "prefix_tokens/state must be either unbatched shapes (T, D)/(D,) or batched shapes (B, T, D)/(B, D); "
            f"got {tuple(prefix_tokens.shape)} and {tuple(state.shape)}."
        )
    normalized_tasks = _normalize_prediction_tasks(tasks, batch_size=prefix_tokens.shape[0])
    if noise is not None:
        noise = torch.as_tensor(noise, device=device, dtype=dtype)
        if not batched and noise.ndim == 2:
            noise = noise.unsqueeze(0)
    sampled = sample_actions(
        loaded.model,
        prefix_tokens,
        state,
        num_steps=num_steps,
        noise=noise,
        generator=generator,
    )
    if loaded.action_norm_by_task:
        denormalized = _denormalize_actions_by_task(
            sampled,
            loaded.action_norm_by_task,
            normalized_tasks,
            checkpoint_path=loaded.checkpoint_path,
        )
    else:
        denormalized = _denormalize_actions(sampled, loaded.action_norm_mean, loaded.action_norm_std)
    return denormalized if batched else denormalized[0]


__all__ = [
    "CachedWanPrefixActionDataset",
    "CachedWanPrefixSample",
    "FUTURE_LEAKAGE_KEYS",
    "LoadedWanPi05ActionExpert",
    "SinusoidalTimeEmbedding",
    "ActionDenoisingContext",
    "WanActionPrefixMemory",
    "WanPi05ActionExpert",
    "find_prefix_cache_rows",
    "flow_matching_loss",
    "flow_matching_loss_per_sample",
    "flow_matching_loss_per_sample_parts",
    "load_cached_prefix_dataset",
    "load_pi05_wan_prefix_cache_row",
    "load_wan_pi05_action_expert_checkpoint",
    "masked_action_mse",
    "masked_action_mse_per_sample",
    "masked_action_mse_per_sample_parts",
    "pi05_flow_targets",
    "predict_denormalized_action_chunk",
    "sample_actions",
    "sample_pi05_time",
    "wan_action_mode_from_row",
    "write_fake_prefix_cache",
]
