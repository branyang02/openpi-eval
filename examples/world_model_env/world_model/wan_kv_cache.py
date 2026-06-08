"""Experimental Wan KV-cache data contracts.

This module is a narrow scaffold for testing true key/value cache ideas with
toy modules. It is not wired into Wan serving or training.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator, Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import torch

WAN_KV_SCAFFOLD_KIND = "experimental_true_wan_kv_scaffold"
DIFFSYNTH_WAN_KV_EXPOSURE_NOTE = (
    "DiffSynth Wan does not expose KV directly: SelfAttention/CrossAttention compute Q/K/V inside forward "
    "but do not return or accept KV caches. Real use requires cache-aware SelfAttention/CrossAttention wrappers."
)


class WanKVLayout(StrEnum):
    """Supported projected/head-split KV tensor layouts."""

    DIFFSYNTH_PROJECTED = "bsd"
    FLASH_ATTN = "bsnd"
    TORCH_SDPA = "bnsd"


class WanAttentionKind(StrEnum):
    SELF = "self"
    CROSS = "cross"


class WanKVDependency(StrEnum):
    CURRENT_IMAGE = "current_image"
    TEXT = "text"
    TIMESTEP = "timestep"
    NOISE_LATENTS = "noise_latents"
    FUTURE_LATENTS = "future_latents"
    ACTION_TOKENS = "action_tokens"


class WanKVCacheLifetime(StrEnum):
    STATIC_CURRENT_IMAGE_TEXT = "static_current_image_text"
    DYNAMIC_TIMESTEP_NOISE_OR_ACTION = "dynamic_timestep_noise_or_action"


STATIC_CURRENT_IMAGE_TEXT_DEPENDENCIES = frozenset(
    {
        WanKVDependency.CURRENT_IMAGE,
        WanKVDependency.TEXT,
    }
)
TIMESTEP_NOISE_DEPENDENCIES = frozenset(
    {
        WanKVDependency.TIMESTEP,
        WanKVDependency.NOISE_LATENTS,
        WanKVDependency.FUTURE_LATENTS,
    }
)
DYNAMIC_DEPENDENCIES = frozenset(
    {
        *TIMESTEP_NOISE_DEPENDENCIES,
        WanKVDependency.ACTION_TOKENS,
    }
)


def wan_kv_scaffold_metadata() -> dict[str, Any]:
    """Return metadata that marks this file as an experimental, non-integrated scaffold."""

    return {
        "cache_kind": WAN_KV_SCAFFOLD_KIND,
        "serving_integration": False,
        "training_integration": False,
        "diffsynth_wan_exposes_kv_directly": False,
        "requires_cache_aware_attention_wrappers": True,
        "note": DIFFSYNTH_WAN_KV_EXPOSURE_NOTE,
    }


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _coerce_layout(layout: WanKVLayout | str) -> WanKVLayout:
    try:
        return WanKVLayout(layout)
    except ValueError as exc:
        values = ", ".join(layout.value for layout in WanKVLayout)
        raise ValueError(f"layout must be one of {values}, got {layout!r}.") from exc


def _coerce_attention_kind(kind: WanAttentionKind | str) -> WanAttentionKind:
    try:
        return WanAttentionKind(kind)
    except ValueError as exc:
        values = ", ".join(kind.value for kind in WanAttentionKind)
        raise ValueError(f"attention_kind must be one of {values}, got {kind!r}.") from exc


def _coerce_dependency(dependency: WanKVDependency | str) -> WanKVDependency:
    try:
        return WanKVDependency(dependency)
    except ValueError as exc:
        values = ", ".join(dependency.value for dependency in WanKVDependency)
        raise ValueError(f"Wan KV dependency must be one of {values}, got {dependency!r}.") from exc


def normalize_kv_dependencies(
    dependencies: Iterable[WanKVDependency | str] | WanKVDependency | str,
) -> frozenset[WanKVDependency]:
    """Normalize dependency tags used by pure static/dynamic cache classifiers."""

    if isinstance(dependencies, WanKVDependency | str):
        dependencies = (dependencies,)
    normalized = frozenset(_coerce_dependency(dependency) for dependency in dependencies)
    if not normalized:
        raise ValueError("KV dependencies must contain at least one dependency tag.")
    return normalized


@dataclasses.dataclass(frozen=True)
class WanKVShape:
    batch_size: int
    token_count: int
    model_dim: int
    num_heads: int
    head_dim: int
    layout: WanKVLayout = WanKVLayout.DIFFSYNTH_PROJECTED

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", _coerce_layout(self.layout))
        batch_size = _positive_int(self.batch_size, name="batch_size")
        token_count = _positive_int(self.token_count, name="token_count")
        model_dim = _positive_int(self.model_dim, name="model_dim")
        num_heads = _positive_int(self.num_heads, name="num_heads")
        head_dim = _positive_int(self.head_dim, name="head_dim")
        if model_dim != num_heads * head_dim:
            raise ValueError(
                f"model_dim must equal num_heads * head_dim, got {model_dim} != {num_heads} * {head_dim}."
            )
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "token_count", token_count)
        object.__setattr__(self, "model_dim", model_dim)
        object.__setattr__(self, "num_heads", num_heads)
        object.__setattr__(self, "head_dim", head_dim)


def _check_expected_dimension(actual: int, expected: int | None, *, name: str, label: str) -> None:
    if expected is None:
        return
    expected = _positive_int(expected, name=label)
    if actual != expected:
        raise ValueError(f"{name} {label} {actual} must match expected {expected}.")


def validate_kv_tensors(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    layout: WanKVLayout | str = WanKVLayout.DIFFSYNTH_PROJECTED,
    batch_size: int | None = None,
    token_count: int | None = None,
    model_dim: int | None = None,
    head_dim: int | None = None,
    name: str = "kv",
) -> WanKVShape:
    """Validate Wan-style K/V tensor shape and head consistency.

    DiffSynth Wan computes projected attention tensors as ``(B, tokens, dim)``
    before its attention helper splits them into either ``(B, tokens, heads,
    head_dim)`` or ``(B, heads, tokens, head_dim)``. This scaffold supports all
    three layouts so toy tests can inspect either side of that reshape.
    """

    layout = _coerce_layout(layout)
    num_heads = _positive_int(num_heads, name="num_heads")
    if model_dim is not None and head_dim is not None and model_dim != num_heads * head_dim:
        raise ValueError(
            f"model_dim must equal num_heads * head_dim, got {model_dim} != {num_heads} * {head_dim}."
        )
    if key.shape != value.shape:
        raise ValueError(f"{name} key shape {tuple(key.shape)} must match value shape {tuple(value.shape)}.")

    if layout == WanKVLayout.DIFFSYNTH_PROJECTED:
        if key.ndim != 3:
            raise ValueError(f"{name} tensors with layout 'bsd' must have shape (B, tokens, dim), got {tuple(key.shape)}.")
        actual_batch_size, actual_token_count, actual_model_dim = key.shape
        if actual_model_dim % num_heads != 0:
            raise ValueError(
                f"{name} model_dim {actual_model_dim} must be divisible by num_heads {num_heads}."
            )
        actual_head_dim = actual_model_dim // num_heads
    elif layout == WanKVLayout.FLASH_ATTN:
        if key.ndim != 4:
            raise ValueError(
                f"{name} tensors with layout 'bsnd' must have shape (B, tokens, heads, head_dim), "
                f"got {tuple(key.shape)}."
            )
        actual_batch_size, actual_token_count, actual_num_heads, actual_head_dim = key.shape
        if actual_num_heads != num_heads:
            raise ValueError(f"{name} heads {actual_num_heads} must match num_heads {num_heads}.")
        actual_model_dim = actual_num_heads * actual_head_dim
    else:
        if key.ndim != 4:
            raise ValueError(
                f"{name} tensors with layout 'bnsd' must have shape (B, heads, tokens, head_dim), "
                f"got {tuple(key.shape)}."
            )
        actual_batch_size, actual_num_heads, actual_token_count, actual_head_dim = key.shape
        if actual_num_heads != num_heads:
            raise ValueError(f"{name} heads {actual_num_heads} must match num_heads {num_heads}.")
        actual_model_dim = actual_num_heads * actual_head_dim

    shape = WanKVShape(
        batch_size=int(actual_batch_size),
        token_count=int(actual_token_count),
        model_dim=int(actual_model_dim),
        num_heads=num_heads,
        head_dim=int(actual_head_dim),
        layout=layout,
    )
    _check_expected_dimension(shape.batch_size, batch_size, name=name, label="batch_size")
    _check_expected_dimension(shape.token_count, token_count, name=name, label="token_count")
    _check_expected_dimension(shape.model_dim, model_dim, name=name, label="model_dim")
    _check_expected_dimension(shape.head_dim, head_dim, name=name, label="head_dim")
    return shape


@dataclasses.dataclass(frozen=True)
class WanAttentionKVCache:
    key: torch.Tensor
    value: torch.Tensor
    attention_kind: WanAttentionKind | str
    dependencies: Iterable[WanKVDependency | str] | WanKVDependency | str
    num_heads: int
    layout: WanKVLayout | str = WanKVLayout.DIFFSYNTH_PROJECTED
    layer_index: int | None = None
    name: str = ""
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    shape: WanKVShape = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        attention_kind = _coerce_attention_kind(self.attention_kind)
        dependencies = normalize_kv_dependencies(self.dependencies)
        layout = _coerce_layout(self.layout)
        object.__setattr__(self, "attention_kind", attention_kind)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "layout", layout)
        if self.layer_index is not None:
            object.__setattr__(self, "layer_index", _non_negative_int(self.layer_index, name="layer_index"))
        shape = validate_kv_tensors(
            self.key,
            self.value,
            num_heads=self.num_heads,
            layout=layout,
            name=f"{attention_kind.value}_attention",
        )
        object.__setattr__(self, "num_heads", shape.num_heads)
        object.__setattr__(self, "shape", shape)

    @property
    def lifetime(self) -> WanKVCacheLifetime:
        return classify_kv_cache_dependencies(self.dependencies)

    @property
    def is_static_current_image_text(self) -> bool:
        return is_static_current_image_text_cache(self)

    @property
    def is_timestep_noise_dependent(self) -> bool:
        return is_timestep_noise_dependent_cache(self)


@dataclasses.dataclass(frozen=True)
class WanLayerKVCache:
    layer_index: int
    self_attention: WanAttentionKVCache | None = None
    cross_attention: WanAttentionKVCache | None = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        layer_index = _non_negative_int(self.layer_index, name="layer_index")
        object.__setattr__(self, "layer_index", layer_index)
        if self.self_attention is None and self.cross_attention is None:
            raise ValueError("WanLayerKVCache must contain self_attention and/or cross_attention KV.")
        if self.self_attention is not None:
            self._validate_attention_cache(self.self_attention, expected_kind=WanAttentionKind.SELF)
        if self.cross_attention is not None:
            self._validate_attention_cache(self.cross_attention, expected_kind=WanAttentionKind.CROSS)
        self._validate_shape_consistency()

    def _validate_attention_cache(self, cache: WanAttentionKVCache, *, expected_kind: WanAttentionKind) -> None:
        if cache.attention_kind != expected_kind:
            raise ValueError(
                f"{expected_kind.value}_attention must be a {expected_kind.value} attention KV cache, "
                f"got {cache.attention_kind.value}."
            )
        if cache.layer_index is not None and cache.layer_index != self.layer_index:
            raise ValueError(
                f"{expected_kind.value}_attention layer_index {cache.layer_index} must match layer {self.layer_index}."
            )

    def _validate_shape_consistency(self) -> None:
        shapes = [cache.shape for cache in self.iter_attention_caches()]
        first = shapes[0]
        for shape in shapes[1:]:
            if shape.batch_size != first.batch_size:
                raise ValueError(
                    f"layer {self.layer_index} attention caches must share batch_size, got "
                    f"{first.batch_size} and {shape.batch_size}."
                )
            if shape.model_dim != first.model_dim or shape.num_heads != first.num_heads or shape.head_dim != first.head_dim:
                raise ValueError(
                    f"layer {self.layer_index} attention caches must share model/head dimensions, got "
                    f"{first} and {shape}."
                )

    def iter_attention_caches(self) -> Iterator[WanAttentionKVCache]:
        if self.self_attention is not None:
            yield self.self_attention
        if self.cross_attention is not None:
            yield self.cross_attention


@dataclasses.dataclass(frozen=True)
class WanPrefixKVCache:
    layers: Sequence[WanLayerKVCache]
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=wan_kv_scaffold_metadata)

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("WanPrefixKVCache must contain at least one layer cache.")
        seen: set[int] = set()
        for layer in layers:
            if layer.layer_index in seen:
                raise ValueError(f"WanPrefixKVCache layer indices must be unique, duplicate {layer.layer_index}.")
            seen.add(layer.layer_index)
        self._validate_shape_consistency(layers)

        metadata = dict(self.metadata)
        metadata.update(wan_kv_scaffold_metadata())
        metadata["layer_count"] = len(layers)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "metadata", metadata)

    def _validate_shape_consistency(self, layers: tuple[WanLayerKVCache, ...]) -> None:
        first = next(layers[0].iter_attention_caches()).shape
        for layer in layers[1:]:
            for cache in layer.iter_attention_caches():
                shape = cache.shape
                if shape.batch_size != first.batch_size:
                    raise ValueError(
                        f"all prefix KV layers must share batch_size, got {first.batch_size} and {shape.batch_size}."
                    )
                if shape.model_dim != first.model_dim or shape.num_heads != first.num_heads or shape.head_dim != first.head_dim:
                    raise ValueError(f"all prefix KV layers must share model/head dimensions, got {first} and {shape}.")

    def iter_attention_caches(self) -> Iterator[WanAttentionKVCache]:
        for layer in self.layers:
            yield from layer.iter_attention_caches()

    @property
    def batch_size(self) -> int:
        return next(self.iter_attention_caches()).shape.batch_size

    @property
    def model_dim(self) -> int:
        return next(self.iter_attention_caches()).shape.model_dim

    @property
    def num_heads(self) -> int:
        return next(self.iter_attention_caches()).shape.num_heads


@runtime_checkable
class WanAttentionKVBackend(Protocol):
    """Minimal backend hooks a Wan-like attention module must expose for this scaffold.

    Real Wan attention implementations commonly compute Q/K/V projections inside
    ``forward``. A cache-aware wrapper has to split that behavior into a prefix
    K/V emission path and a dynamic query/action path that consumes cached K/V.
    """

    def emit_prefix_kv(
        self,
        *,
        prefix_tokens: torch.Tensor,
        context_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | WanAttentionKVCache:
        """Return K/V tensors for a current-image/text prefix pass."""
        ...

    def forward_with_cached_kv(
        self,
        *,
        query_tokens: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
    ) -> torch.Tensor:
        """Consume cached prefix K/V plus dynamic action/query tokens."""
        ...


@runtime_checkable
class WanCacheAwareAttentionWrapper(Protocol):
    """Attention-level contract required before true Wan K/V reuse can be integrated."""

    @property
    def attention_kind(self) -> WanAttentionKind:
        """Report whether this wrapper represents self- or cross-attention."""
        ...

    @property
    def dependencies(self) -> frozenset[WanKVDependency]:
        """Report cache dependency tags for emitted K/V."""
        ...

    @property
    def layout(self) -> WanKVLayout:
        """Report the K/V tensor layout emitted by this wrapper."""
        ...

    @property
    def num_heads(self) -> int:
        """Report the number of attention heads used to validate K/V tensors."""
        ...

    def emit_prefix_kv(
        self,
        *,
        prefix_tokens: torch.Tensor,
        context_tokens: torch.Tensor | None = None,
    ) -> WanAttentionKVCache:
        """Emit validated K/V for a prefix/current image+text pass."""
        ...

    def forward_with_attention_cache(
        self,
        *,
        query_tokens: torch.Tensor,
        prefix_attention_cache: WanAttentionKVCache,
    ) -> torch.Tensor:
        """Consume cached prefix K/V plus dynamic action/query tokens."""
        ...


def _validate_3d_token_tensor(
    tokens: torch.Tensor,
    *,
    name: str,
    batch_size: int | None = None,
    model_dim: int | None = None,
) -> tuple[int, int, int]:
    if tokens.ndim != 3:
        raise ValueError(f"{name} must have shape (B, tokens, dim), got {tuple(tokens.shape)}.")
    actual_batch_size, actual_token_count, actual_model_dim = tokens.shape
    if actual_token_count <= 0:
        raise ValueError(f"{name} must contain at least one token, got {actual_token_count}.")
    if batch_size is not None and actual_batch_size != batch_size:
        raise ValueError(f"{name} batch size {actual_batch_size} must match expected {batch_size}.")
    if model_dim is not None and actual_model_dim != model_dim:
        raise ValueError(f"{name} model dim {actual_model_dim} must match expected {model_dim}.")
    return int(actual_batch_size), int(actual_token_count), int(actual_model_dim)


def validate_query_tokens_against_attention_cache(
    query_tokens: torch.Tensor,
    prefix_attention_cache: WanAttentionKVCache,
    *,
    name: str = "query_tokens",
) -> None:
    _validate_3d_token_tensor(
        query_tokens,
        name=name,
        batch_size=prefix_attention_cache.shape.batch_size,
        model_dim=prefix_attention_cache.shape.model_dim,
    )


def _cache_token_count_for_prefix(
    *,
    attention_kind: WanAttentionKind,
    prefix_token_count: int,
    context_token_count: int | None,
) -> int:
    if attention_kind == WanAttentionKind.CROSS and context_token_count is not None:
        return context_token_count
    return prefix_token_count


@dataclasses.dataclass(frozen=True)
class WanCacheAwareAttentionAdapter:
    """Pure adapter for testing what a cache-aware Wan attention wrapper must do."""

    module: Any
    attention_kind: WanAttentionKind | str
    dependencies: Iterable[WanKVDependency | str] | WanKVDependency | str
    num_heads: int
    layout: WanKVLayout | str = WanKVLayout.DIFFSYNTH_PROJECTED
    layer_index: int | None = None
    name: str = ""
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    prefix_kv_method_name: str = "emit_prefix_kv"
    cached_forward_method_name: str = "forward_with_cached_kv"

    def __post_init__(self) -> None:
        attention_kind = _coerce_attention_kind(self.attention_kind)
        dependencies = normalize_kv_dependencies(self.dependencies)
        layout = _coerce_layout(self.layout)
        num_heads = _positive_int(self.num_heads, name="num_heads")
        layer_index = None
        if self.layer_index is not None:
            layer_index = _non_negative_int(self.layer_index, name="layer_index")

        object.__setattr__(self, "attention_kind", attention_kind)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "num_heads", num_heads)
        object.__setattr__(self, "layer_index", layer_index)

        self._validate_backend_method(self.prefix_kv_method_name, role="prefix K/V emission")
        self._validate_backend_method(self.cached_forward_method_name, role="cached K/V consumption")

    def _validate_backend_method(self, method_name: str, *, role: str) -> None:
        if not isinstance(method_name, str) or not method_name:
            raise ValueError(f"{role} method name must be a non-empty string, got {method_name!r}.")
        method = getattr(self.module, method_name, None)
        if not callable(method):
            raise TypeError(
                f"cache-aware attention module must expose {method_name}(...), the {role} hook required by "
                "WanAttentionKVBackend."
            )

    @property
    def cache_dependency_tags(self) -> tuple[WanKVDependency, ...]:
        return tuple(sorted(self.dependencies, key=lambda dependency: dependency.value))

    def describe_cache_contract(self) -> dict[str, Any]:
        """Report the explicit cache contract exposed by this adapter."""

        return {
            **wan_kv_scaffold_metadata(),
            "adapter": type(self).__name__,
            "source_module_type": type(self.module).__name__,
            "name": self.name,
            "attention_kind": self.attention_kind.value,
            "cache_dependency_tags": tuple(dependency.value for dependency in self.cache_dependency_tags),
            "layout": self.layout.value,
            "num_heads": self.num_heads,
            "layer_index": self.layer_index,
            "emits_prefix_kv": True,
            "consumes_prefix_kv_with_dynamic_query_tokens": True,
            "metadata": dict(self.metadata),
        }

    def emit_prefix_kv(
        self,
        *,
        prefix_tokens: torch.Tensor,
        context_tokens: torch.Tensor | None = None,
    ) -> WanAttentionKVCache:
        """Run the backend's prefix pass and return a validated ``WanAttentionKVCache``."""

        batch_size, prefix_token_count, model_dim = _validate_3d_token_tensor(prefix_tokens, name="prefix_tokens")
        context_token_count = None
        if context_tokens is not None:
            _, context_token_count, _ = _validate_3d_token_tensor(
                context_tokens,
                name="context_tokens",
                batch_size=batch_size,
                model_dim=model_dim,
            )

        method = getattr(self.module, self.prefix_kv_method_name)
        result = method(prefix_tokens=prefix_tokens, context_tokens=context_tokens)
        expected_token_count = _cache_token_count_for_prefix(
            attention_kind=self.attention_kind,
            prefix_token_count=prefix_token_count,
            context_token_count=context_token_count,
        )
        cache = self._coerce_prefix_kv_result(result)
        self._validate_adapter_cache(
            cache,
            expected_batch_size=batch_size,
            expected_token_count=expected_token_count,
            expected_model_dim=model_dim,
        )
        return cache

    def _coerce_prefix_kv_result(self, result: Any) -> WanAttentionKVCache:
        if isinstance(result, WanAttentionKVCache):
            return result
        if not isinstance(result, Sequence) or len(result) != 2:
            raise TypeError(
                f"{self.prefix_kv_method_name}(...) must return WanAttentionKVCache or (key, value), "
                f"got {type(result).__name__}."
            )
        key, value = result
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError(f"{self.prefix_kv_method_name}(...) must return tensor K/V values.")
        return WanAttentionKVCache(
            key=key,
            value=value,
            attention_kind=self.attention_kind,
            dependencies=self.dependencies,
            num_heads=self.num_heads,
            layout=self.layout,
            layer_index=self.layer_index,
            name=self.name,
            metadata={
                **wan_kv_scaffold_metadata(),
                "adapter": type(self).__name__,
                "source_module_type": type(self.module).__name__,
                **dict(self.metadata),
            },
        )

    def _validate_adapter_cache(
        self,
        cache: WanAttentionKVCache,
        *,
        expected_batch_size: int | None = None,
        expected_token_count: int | None = None,
        expected_model_dim: int | None = None,
    ) -> None:
        if cache.attention_kind != self.attention_kind:
            raise ValueError(
                f"emitted cache attention_kind {cache.attention_kind.value} must match adapter "
                f"{self.attention_kind.value}."
            )
        if cache.dependencies != self.dependencies:
            expected = sorted(dependency.value for dependency in self.dependencies)
            actual = sorted(dependency.value for dependency in cache.dependencies)
            raise ValueError(f"emitted cache dependencies {actual} must match adapter dependencies {expected}.")
        if cache.layout != self.layout:
            raise ValueError(f"emitted cache layout {cache.layout.value} must match adapter layout {self.layout.value}.")
        if cache.num_heads != self.num_heads:
            raise ValueError(f"emitted cache num_heads {cache.num_heads} must match adapter {self.num_heads}.")
        if self.layer_index is not None and cache.layer_index != self.layer_index:
            raise ValueError(f"emitted cache layer_index {cache.layer_index} must match adapter {self.layer_index}.")
        _check_expected_dimension(
            cache.shape.batch_size,
            expected_batch_size,
            name="emitted cache",
            label="batch_size",
        )
        _check_expected_dimension(
            cache.shape.token_count,
            expected_token_count,
            name="emitted cache",
            label="token_count",
        )
        _check_expected_dimension(
            cache.shape.model_dim,
            expected_model_dim,
            name="emitted cache",
            label="model_dim",
        )

    def forward_with_attention_cache(
        self,
        *,
        query_tokens: torch.Tensor,
        prefix_attention_cache: WanAttentionKVCache,
    ) -> torch.Tensor:
        """Consume one attention cache with dynamic query/action tokens."""

        self._validate_adapter_cache(prefix_attention_cache)
        validate_query_tokens_against_attention_cache(query_tokens, prefix_attention_cache)
        method = getattr(self.module, self.cached_forward_method_name)
        return method(
            query_tokens=query_tokens,
            prefix_key=prefix_attention_cache.key,
            prefix_value=prefix_attention_cache.value,
        )

    def forward_with_prefix_kv(
        self,
        *,
        action_tokens: torch.Tensor,
        prefix_cache: WanPrefixKVCache,
    ) -> torch.Tensor:
        """WanActionContextKVConsumer-compatible entry point for dynamic action tokens."""

        validate_action_tokens_against_prefix_cache(action_tokens, prefix_cache)
        prefix_attention_cache = select_attention_cache_for_wrapper(prefix_cache, wrapper=self)
        return self.forward_with_attention_cache(
            query_tokens=action_tokens,
            prefix_attention_cache=prefix_attention_cache,
        )


def adapt_wan_attention_module(
    module: Any,
    *,
    attention_kind: WanAttentionKind | str,
    dependencies: Iterable[WanKVDependency | str] | WanKVDependency | str,
    num_heads: int,
    layout: WanKVLayout | str = WanKVLayout.DIFFSYNTH_PROJECTED,
    layer_index: int | None = None,
    name: str = "",
    metadata: Mapping[str, Any] | None = None,
    prefix_kv_method_name: str = "emit_prefix_kv",
    cached_forward_method_name: str = "forward_with_cached_kv",
) -> WanCacheAwareAttentionAdapter:
    """Create a pure cache-aware adapter around a toy Wan-like attention module."""

    return WanCacheAwareAttentionAdapter(
        module=module,
        attention_kind=attention_kind,
        dependencies=dependencies,
        num_heads=num_heads,
        layout=layout,
        layer_index=layer_index,
        name=name,
        metadata={} if metadata is None else metadata,
        prefix_kv_method_name=prefix_kv_method_name,
        cached_forward_method_name=cached_forward_method_name,
    )


def select_attention_cache_for_wrapper(
    prefix_cache: WanPrefixKVCache,
    *,
    wrapper: WanCacheAwareAttentionWrapper,
) -> WanAttentionKVCache:
    """Select the prefix attention cache matching a cache-aware attention wrapper."""

    matches = tuple(
        cache
        for layer in prefix_cache.layers
        for cache in layer.iter_attention_caches()
        if cache.attention_kind == wrapper.attention_kind
        and (getattr(wrapper, "layer_index", None) is None or layer.layer_index == getattr(wrapper, "layer_index"))
    )
    if not matches:
        layer_index = getattr(wrapper, "layer_index", None)
        layer_label = "any layer" if layer_index is None else f"layer {layer_index}"
        raise ValueError(f"prefix cache does not contain {wrapper.attention_kind.value}-attention KV for {layer_label}.")
    if len(matches) > 1:
        raise ValueError(
            f"prefix cache contains {len(matches)} {wrapper.attention_kind.value}-attention KV entries; "
            "set wrapper.layer_index to select one."
        )
    return matches[0]


def is_static_current_image_text_dependencies(
    dependencies: Iterable[WanKVDependency | str] | WanKVDependency | str,
) -> bool:
    dependencies = normalize_kv_dependencies(dependencies)
    return dependencies <= STATIC_CURRENT_IMAGE_TEXT_DEPENDENCIES


def is_timestep_noise_dependent_dependencies(
    dependencies: Iterable[WanKVDependency | str] | WanKVDependency | str,
) -> bool:
    dependencies = normalize_kv_dependencies(dependencies)
    return bool(dependencies & TIMESTEP_NOISE_DEPENDENCIES)


def classify_kv_cache_dependencies(
    dependencies: Iterable[WanKVDependency | str] | WanKVDependency | str,
) -> WanKVCacheLifetime:
    dependencies = normalize_kv_dependencies(dependencies)
    if dependencies <= STATIC_CURRENT_IMAGE_TEXT_DEPENDENCIES:
        return WanKVCacheLifetime.STATIC_CURRENT_IMAGE_TEXT
    return WanKVCacheLifetime.DYNAMIC_TIMESTEP_NOISE_OR_ACTION


def is_static_current_image_text_cache(cache: WanAttentionKVCache) -> bool:
    return is_static_current_image_text_dependencies(cache.dependencies)


def is_timestep_noise_dependent_cache(cache: WanAttentionKVCache) -> bool:
    return is_timestep_noise_dependent_dependencies(cache.dependencies)


def static_current_image_text_attention_slots(prefix_cache: WanPrefixKVCache) -> tuple[tuple[int, WanAttentionKind], ...]:
    return tuple(
        (cache.layer_index if cache.layer_index is not None else layer.layer_index, cache.attention_kind)
        for layer in prefix_cache.layers
        for cache in layer.iter_attention_caches()
        if cache.is_static_current_image_text
    )


def dynamic_attention_slots(prefix_cache: WanPrefixKVCache) -> tuple[tuple[int, WanAttentionKind], ...]:
    return tuple(
        (cache.layer_index if cache.layer_index is not None else layer.layer_index, cache.attention_kind)
        for layer in prefix_cache.layers
        for cache in layer.iter_attention_caches()
        if not cache.is_static_current_image_text
    )


def validate_action_tokens_against_prefix_cache(action_tokens: torch.Tensor, prefix_cache: WanPrefixKVCache) -> None:
    if action_tokens.ndim != 3:
        raise ValueError(f"action_tokens must have shape (B, action_tokens, dim), got {tuple(action_tokens.shape)}.")
    batch_size, token_count, model_dim = action_tokens.shape
    if batch_size != prefix_cache.batch_size:
        raise ValueError(f"action_tokens batch size {batch_size} must match prefix KV batch size {prefix_cache.batch_size}.")
    if token_count <= 0:
        raise ValueError(f"action_tokens must contain at least one token, got {token_count}.")
    if model_dim != prefix_cache.model_dim:
        raise ValueError(f"action_tokens model dim {model_dim} must match prefix KV model dim {prefix_cache.model_dim}.")


@runtime_checkable
class WanActionContextKVConsumer(Protocol):
    """Tiny interface for action modules that can reuse prefix K/V and append dynamic tokens."""

    def forward_with_prefix_kv(
        self,
        *,
        action_tokens: torch.Tensor,
        prefix_cache: WanPrefixKVCache,
    ) -> torch.Tensor:
        """Consume cached current-image/text prefix K/V plus per-step action tokens."""


def apply_cache_aware_action_context(
    consumer: WanActionContextKVConsumer,
    *,
    prefix_cache: WanPrefixKVCache,
    action_tokens: torch.Tensor,
) -> torch.Tensor:
    """Validate action tokens and call a cache-aware action-context consumer."""

    validate_action_tokens_against_prefix_cache(action_tokens, prefix_cache)
    method = getattr(consumer, "forward_with_prefix_kv", None)
    if method is None:
        raise TypeError("consumer must expose forward_with_prefix_kv(action_tokens=..., prefix_cache=...).")
    return method(action_tokens=action_tokens, prefix_cache=prefix_cache)


__all__ = [
    "DIFFSYNTH_WAN_KV_EXPOSURE_NOTE",
    "DYNAMIC_DEPENDENCIES",
    "STATIC_CURRENT_IMAGE_TEXT_DEPENDENCIES",
    "TIMESTEP_NOISE_DEPENDENCIES",
    "WAN_KV_SCAFFOLD_KIND",
    "WanActionContextKVConsumer",
    "WanAttentionKVBackend",
    "WanAttentionKVCache",
    "WanAttentionKind",
    "WanCacheAwareAttentionAdapter",
    "WanCacheAwareAttentionWrapper",
    "WanKVCacheLifetime",
    "WanKVDependency",
    "WanKVLayout",
    "WanKVShape",
    "WanLayerKVCache",
    "WanPrefixKVCache",
    "adapt_wan_attention_module",
    "apply_cache_aware_action_context",
    "classify_kv_cache_dependencies",
    "dynamic_attention_slots",
    "is_static_current_image_text_cache",
    "is_static_current_image_text_dependencies",
    "is_timestep_noise_dependent_cache",
    "is_timestep_noise_dependent_dependencies",
    "normalize_kv_dependencies",
    "select_attention_cache_for_wrapper",
    "static_current_image_text_attention_slots",
    "validate_action_tokens_against_prefix_cache",
    "validate_kv_tensors",
    "validate_query_tokens_against_attention_cache",
    "wan_kv_scaffold_metadata",
]
