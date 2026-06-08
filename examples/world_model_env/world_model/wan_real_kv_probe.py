"""Small Wan-like projected K/V probe.

This module is an experimental scaffold for fake or locally constructed
Wan-style attention modules. It does not patch DiffSynth and does not claim
production true-KV action conditioning.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch

from .wan_kv_cache import (
    DYNAMIC_DEPENDENCIES,
    STATIC_CURRENT_IMAGE_TEXT_DEPENDENCIES,
    WanAttentionKind,
    WanAttentionKVCache,
    WanCacheAwareAttentionAdapter,
    WanKVDependency,
    WanKVLayout,
    adapt_wan_attention_module,
    classify_kv_cache_dependencies,
    normalize_kv_dependencies,
    wan_kv_scaffold_metadata,
)

WAN_SELF_ATTENTION_DYNAMIC_DEPENDENCIES = frozenset(
    {
        WanKVDependency.TIMESTEP,
        WanKVDependency.NOISE_LATENTS,
        WanKVDependency.FUTURE_LATENTS,
    }
)
WAN_CROSS_ATTENTION_TEXT_DEPENDENCIES = frozenset({WanKVDependency.TEXT})
WAN_CROSS_ATTENTION_IMAGE_DEPENDENCIES = frozenset({WanKVDependency.CURRENT_IMAGE})


def _coerce_attention_kind(kind: WanAttentionKind | str | None, *, module: Any) -> WanAttentionKind:
    if kind is not None:
        return WanAttentionKind(kind)
    class_name = type(module).__name__.lower()
    if "cross" in class_name:
        return WanAttentionKind.CROSS
    if "self" in class_name:
        return WanAttentionKind.SELF
    raise ValueError(
        "attention_kind is required when the module class name does not contain 'Self' or 'Cross'."
    )


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _module_num_heads(module: Any, num_heads: int | None) -> int:
    if num_heads is not None:
        return _positive_int(num_heads, name="num_heads")
    if not hasattr(module, "num_heads"):
        raise ValueError("num_heads must be provided for modules without a num_heads attribute.")
    return _positive_int(getattr(module, "num_heads"), name="num_heads")


def _module_dim(module: Any) -> int | None:
    value = getattr(module, "dim", None)
    if value is None:
        return None
    return _positive_int(value, name="dim")


def _require_callable(module: Any, name: str, *, role: str) -> Callable[..., torch.Tensor]:
    value = getattr(module, name, None)
    if not callable(value):
        raise TypeError(f"{type(module).__name__} must expose callable .{name}(...) for {role}.")
    return value


def _maybe_call(module: Any, name: str, tensor: torch.Tensor) -> torch.Tensor:
    value = getattr(module, name, None)
    if callable(value):
        return value(tensor)
    return tensor


def _apply_optional_rope(
    module: Any,
    tensor: torch.Tensor,
    *,
    freqs: torch.Tensor | None,
    num_heads: int,
    rope_apply_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] | None,
) -> torch.Tensor:
    if freqs is None:
        return tensor
    if rope_apply_fn is not None:
        return rope_apply_fn(tensor, freqs, num_heads)
    module_rope = getattr(module, "rope_apply", None)
    if callable(module_rope):
        return module_rope(tensor, freqs, num_heads)
    raise ValueError("freqs were provided but no rope_apply_fn or module.rope_apply(...) is available.")


def _default_dependencies(attention_kind: WanAttentionKind) -> frozenset[WanKVDependency]:
    if attention_kind == WanAttentionKind.CROSS:
        return WAN_CROSS_ATTENTION_TEXT_DEPENDENCIES
    return WAN_SELF_ATTENTION_DYNAMIC_DEPENDENCIES


def _dependency_values(dependencies: Iterable[WanKVDependency]) -> tuple[str, ...]:
    return tuple(dependency.value for dependency in sorted(dependencies, key=lambda item: item.value))


@dataclasses.dataclass(frozen=True)
class WanProjectionCandidate:
    """A projected attention tensor and the dependencies that govern reuse."""

    name: str
    role: str
    cacheable_as_kv: bool
    dependencies: frozenset[WanKVDependency]
    source: str
    note: str

    @property
    def cache_lifetime(self) -> str:
        return classify_kv_cache_dependencies(self.dependencies).value

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "cacheable_as_kv": self.cacheable_as_kv,
            "dependencies": _dependency_values(self.dependencies),
            "cache_lifetime": self.cache_lifetime,
            "source": self.source,
            "note": self.note,
        }


def describe_wan_attention_module(
    module: Any,
    *,
    attention_kind: WanAttentionKind | str | None = None,
    num_heads: int | None = None,
) -> dict[str, Any]:
    """Describe cacheable projection surfaces on a Wan-like attention module."""

    kind = _coerce_attention_kind(attention_kind, module=module)
    resolved_num_heads = _module_num_heads(module, num_heads)
    dim = _module_dim(module)
    has_image_input = bool(getattr(module, "has_image_input", False))
    if kind == WanAttentionKind.SELF:
        candidates = (
            WanProjectionCandidate(
                name="q",
                role="query",
                cacheable_as_kv=False,
                dependencies=WAN_SELF_ATTENTION_DYNAMIC_DEPENDENCIES,
                source="x hidden states plus rotary frequencies",
                note="Query is needed for the current attention call, not a reusable K/V entry.",
            ),
            WanProjectionCandidate(
                name="k",
                role="key",
                cacheable_as_kv=True,
                dependencies=WAN_SELF_ATTENTION_DYNAMIC_DEPENDENCIES,
                source="x hidden states plus rotary frequencies",
                note="Self-attention K is reusable only for the exact hidden states, timestep/noise state, and positions.",
            ),
            WanProjectionCandidate(
                name="v",
                role="value",
                cacheable_as_kv=True,
                dependencies=WAN_SELF_ATTENTION_DYNAMIC_DEPENDENCIES,
                source="x hidden states",
                note="Self-attention V is dynamic because Wan block hidden states change with denoising state.",
            ),
        )
    else:
        candidates = (
            WanProjectionCandidate(
                name="q",
                role="query",
                cacheable_as_kv=False,
                dependencies=WAN_SELF_ATTENTION_DYNAMIC_DEPENDENCIES,
                source="x hidden states",
                note="Cross-attention query changes with the denoising hidden state.",
            ),
            WanProjectionCandidate(
                name="k",
                role="key",
                cacheable_as_kv=True,
                dependencies=WAN_CROSS_ATTENTION_TEXT_DEPENDENCIES,
                source="text/context tokens",
                note="Text K is static for the same prompt context if a wrapper exposes it.",
            ),
            WanProjectionCandidate(
                name="v",
                role="value",
                cacheable_as_kv=True,
                dependencies=WAN_CROSS_ATTENTION_TEXT_DEPENDENCIES,
                source="text/context tokens",
                note="Text V is static for the same prompt context if a wrapper exposes it.",
            ),
        )
        if has_image_input:
            candidates = (
                *candidates,
                WanProjectionCandidate(
                    name="k_img",
                    role="key",
                    cacheable_as_kv=True,
                    dependencies=WAN_CROSS_ATTENTION_IMAGE_DEPENDENCIES,
                    source="image context prefix tokens",
                    note="Image-context K is static for the same current-image context if exposed separately.",
                ),
                WanProjectionCandidate(
                    name="v_img",
                    role="value",
                    cacheable_as_kv=True,
                    dependencies=WAN_CROSS_ATTENTION_IMAGE_DEPENDENCIES,
                    source="image context prefix tokens",
                    note="Image-context V is static for the same current-image context if exposed separately.",
                ),
            )
    cacheable = tuple(candidate for candidate in candidates if candidate.cacheable_as_kv)
    return {
        **wan_kv_scaffold_metadata(),
        "module_type": type(module).__name__,
        "attention_kind": kind.value,
        "num_heads": resolved_num_heads,
        "dim": dim,
        "has_image_input": has_image_input,
        "projected_layout": WanKVLayout.DIFFSYNTH_PROJECTED.value,
        "production_true_kv_action_conditioning": False,
        "projection_candidates": tuple(candidate.as_dict() for candidate in candidates),
        "cacheable_projection_names": tuple(candidate.name for candidate in cacheable),
        "static_cacheable_projection_names": tuple(
            candidate.name
            for candidate in cacheable
            if candidate.dependencies <= STATIC_CURRENT_IMAGE_TEXT_DEPENDENCIES
        ),
        "dynamic_cacheable_projection_names": tuple(
            candidate.name for candidate in cacheable if bool(candidate.dependencies & DYNAMIC_DEPENDENCIES)
        ),
    }


def _cross_context_tokens(
    module: Any,
    context_tokens: torch.Tensor,
    *,
    image_context_token_count: int,
) -> torch.Tensor:
    if not bool(getattr(module, "has_image_input", False)):
        return context_tokens
    if context_tokens.shape[1] <= image_context_token_count:
        raise ValueError(
            "context_tokens must include text tokens after the image prefix when module.has_image_input is true."
        )
    return context_tokens[:, image_context_token_count:]


def emit_projected_wan_kv(
    module: Any,
    *,
    attention_kind: WanAttentionKind | str | None = None,
    prefix_tokens: torch.Tensor | None = None,
    context_tokens: torch.Tensor | None = None,
    freqs: torch.Tensor | None = None,
    rope_apply_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] | None = None,
    dependencies: Iterable[WanKVDependency | str] | WanKVDependency | str | None = None,
    num_heads: int | None = None,
    layout: WanKVLayout | str = WanKVLayout.DIFFSYNTH_PROJECTED,
    layer_index: int | None = None,
    name: str = "",
    metadata: Mapping[str, Any] | None = None,
    image_context_token_count: int = 257,
) -> WanAttentionKVCache:
    """Emit projected K/V from a Wan-like module's own projection layers."""

    kind = _coerce_attention_kind(attention_kind, module=module)
    resolved_num_heads = _module_num_heads(module, num_heads)
    resolved_dependencies = (
        _default_dependencies(kind)
        if dependencies is None
        else normalize_kv_dependencies(dependencies)
    )

    if kind == WanAttentionKind.SELF:
        if prefix_tokens is None:
            raise ValueError("prefix_tokens are required to emit self-attention K/V.")
        key_input = prefix_tokens
        value_input = prefix_tokens
        key = _maybe_call(module, "norm_k", _require_callable(module, "k", role="self-attention key")(key_input))
        key = _apply_optional_rope(
            module,
            key,
            freqs=freqs,
            num_heads=resolved_num_heads,
            rope_apply_fn=rope_apply_fn,
        )
        value = _require_callable(module, "v", role="self-attention value")(value_input)
    else:
        if context_tokens is None:
            raise ValueError("context_tokens are required to emit cross-attention K/V.")
        text_context = _cross_context_tokens(
            module,
            context_tokens,
            image_context_token_count=image_context_token_count,
        )
        key = _maybe_call(module, "norm_k", _require_callable(module, "k", role="cross-attention key")(text_context))
        value = _require_callable(module, "v", role="cross-attention value")(text_context)

    return WanAttentionKVCache(
        key=key,
        value=value,
        attention_kind=kind,
        dependencies=resolved_dependencies,
        num_heads=resolved_num_heads,
        layout=layout,
        layer_index=layer_index,
        name=name,
        metadata={
            **wan_kv_scaffold_metadata(),
            "probe": "emit_projected_wan_kv",
            "source_module_type": type(module).__name__,
            "production_true_kv_action_conditioning": False,
            **({} if metadata is None else dict(metadata)),
        },
    )


@dataclasses.dataclass(frozen=True)
class _ProjectedKVBackend:
    module: Any
    attention_kind: WanAttentionKind
    dependencies: frozenset[WanKVDependency]
    num_heads: int
    layout: WanKVLayout | str
    layer_index: int | None
    name: str
    metadata: Mapping[str, Any]
    rope_apply_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] | None = None

    def emit_prefix_kv(
        self,
        *,
        prefix_tokens: torch.Tensor,
        context_tokens: torch.Tensor | None = None,
    ) -> WanAttentionKVCache:
        return emit_projected_wan_kv(
            self.module,
            attention_kind=self.attention_kind,
            prefix_tokens=prefix_tokens,
            context_tokens=context_tokens,
            rope_apply_fn=self.rope_apply_fn,
            dependencies=self.dependencies,
            num_heads=self.num_heads,
            layout=self.layout,
            layer_index=self.layer_index,
            name=self.name,
            metadata=self.metadata,
        )

    def forward_with_cached_kv(
        self,
        *,
        query_tokens: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
    ) -> torch.Tensor:
        forward = getattr(self.module, "forward_with_cached_kv", None)
        if callable(forward):
            return forward(query_tokens=query_tokens, prefix_key=prefix_key, prefix_value=prefix_value)
        raise NotImplementedError(
            "Projected K/V emission is available, but cached attention forward is not implemented for this module."
        )


def adapt_wan_projection_module(
    module: Any,
    *,
    attention_kind: WanAttentionKind | str | None = None,
    dependencies: Iterable[WanKVDependency | str] | WanKVDependency | str | None = None,
    num_heads: int | None = None,
    layout: WanKVLayout | str = WanKVLayout.DIFFSYNTH_PROJECTED,
    layer_index: int | None = None,
    name: str = "",
    metadata: Mapping[str, Any] | None = None,
    rope_apply_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] | None = None,
) -> WanCacheAwareAttentionAdapter:
    """Wrap a fake Wan-like projection module with the existing KV adapter contract."""

    kind = _coerce_attention_kind(attention_kind, module=module)
    resolved_num_heads = _module_num_heads(module, num_heads)
    resolved_dependencies = (
        _default_dependencies(kind)
        if dependencies is None
        else normalize_kv_dependencies(dependencies)
    )
    backend = _ProjectedKVBackend(
        module=module,
        attention_kind=kind,
        dependencies=resolved_dependencies,
        num_heads=resolved_num_heads,
        layout=layout,
        layer_index=layer_index,
        name=name,
        metadata={} if metadata is None else metadata,
        rope_apply_fn=rope_apply_fn,
    )
    return adapt_wan_attention_module(
        backend,
        attention_kind=kind,
        dependencies=resolved_dependencies,
        num_heads=resolved_num_heads,
        layout=layout,
        layer_index=layer_index,
        name=name,
        metadata={
            "probe": "adapt_wan_projection_module",
            "source_module_type": type(module).__name__,
            **({} if metadata is None else dict(metadata)),
        },
    )


__all__ = [
    "WAN_CROSS_ATTENTION_IMAGE_DEPENDENCIES",
    "WAN_CROSS_ATTENTION_TEXT_DEPENDENCIES",
    "WAN_SELF_ATTENTION_DYNAMIC_DEPENDENCIES",
    "WanProjectionCandidate",
    "adapt_wan_projection_module",
    "describe_wan_attention_module",
    "emit_projected_wan_kv",
]
