from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

WORLD_MODEL_ENV_DIR = Path(__file__).resolve().parents[1]
if str(WORLD_MODEL_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_ENV_DIR))

from world_model.wan_kv_cache import (  # noqa: E402
    WanAttentionKind,
    WanAttentionKVCache,
    WanKVCacheLifetime,
    WanKVDependency,
    WanKVLayout,
    WanLayerKVCache,
    WanPrefixKVCache,
    adapt_wan_attention_module,
    apply_cache_aware_action_context,
    classify_kv_cache_dependencies,
    dynamic_attention_slots,
    is_static_current_image_text_dependencies,
    is_timestep_noise_dependent_dependencies,
    static_current_image_text_attention_slots,
    validate_action_tokens_against_prefix_cache,
    validate_kv_tensors,
    wan_kv_scaffold_metadata,
)
from world_model.wan_real_kv_probe import (  # noqa: E402
    adapt_wan_projection_module,
    describe_wan_attention_module,
    emit_projected_wan_kv,
)


def _projected_cache(
    *,
    layer_index: int,
    attention_kind: WanAttentionKind,
    dependencies: tuple[WanKVDependency, ...],
    batch_size: int = 2,
    token_count: int = 3,
    model_dim: int = 12,
    num_heads: int = 3,
    offset: float = 0.0,
) -> WanAttentionKVCache:
    values = torch.arange(batch_size * token_count * model_dim, dtype=torch.float32)
    key = values.reshape(batch_size, token_count, model_dim) + offset
    value = key + 100.0
    return WanAttentionKVCache(
        key=key,
        value=value,
        attention_kind=attention_kind,
        dependencies=dependencies,
        num_heads=num_heads,
        layout=WanKVLayout.DIFFSYNTH_PROJECTED,
        layer_index=layer_index,
    )


def test_projected_wan_kv_shape_validation_matches_diffsynth_layout() -> None:
    key = torch.zeros(2, 5, 12)
    value = torch.ones(2, 5, 12)

    shape = validate_kv_tensors(
        key,
        value,
        num_heads=3,
        layout=WanKVLayout.DIFFSYNTH_PROJECTED,
        batch_size=2,
        token_count=5,
        model_dim=12,
        head_dim=4,
    )

    assert shape.batch_size == 2
    assert shape.token_count == 5
    assert shape.model_dim == 12
    assert shape.num_heads == 3
    assert shape.head_dim == 4
    assert shape.layout == WanKVLayout.DIFFSYNTH_PROJECTED

    sdpa_shape = validate_kv_tensors(
        torch.zeros(2, 3, 5, 4),
        torch.ones(2, 3, 5, 4),
        num_heads=3,
        layout=WanKVLayout.TORCH_SDPA,
        token_count=5,
        model_dim=12,
    )
    assert sdpa_shape.head_dim == 4
    assert sdpa_shape.layout == WanKVLayout.TORCH_SDPA


def test_kv_shape_validation_rejects_mismatched_batch_tokens_and_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        validate_kv_tensors(torch.zeros(2, 5, 10), torch.zeros(2, 5, 10), num_heads=3)

    with pytest.raises(ValueError, match="token_count"):
        validate_kv_tensors(torch.zeros(2, 5, 12), torch.zeros(2, 5, 12), num_heads=3, token_count=4)

    with pytest.raises(ValueError, match="batch_size"):
        validate_kv_tensors(torch.zeros(2, 5, 12), torch.zeros(2, 5, 12), num_heads=3, batch_size=1)

    with pytest.raises(ValueError, match="heads"):
        validate_kv_tensors(
            torch.zeros(2, 5, 4, 3),
            torch.zeros(2, 5, 4, 3),
            num_heads=3,
            layout=WanKVLayout.FLASH_ATTN,
        )


def test_layer_prefix_containers_record_metadata_and_static_dynamic_slots() -> None:
    self_cache = _projected_cache(
        layer_index=0,
        attention_kind=WanAttentionKind.SELF,
        dependencies=(WanKVDependency.CURRENT_IMAGE, WanKVDependency.TEXT),
        token_count=2,
    )
    cross_cache = _projected_cache(
        layer_index=0,
        attention_kind=WanAttentionKind.CROSS,
        dependencies=(WanKVDependency.TEXT,),
        token_count=4,
        offset=1000.0,
    )
    dynamic_self_cache = _projected_cache(
        layer_index=1,
        attention_kind=WanAttentionKind.SELF,
        dependencies=(WanKVDependency.TIMESTEP, WanKVDependency.NOISE_LATENTS),
        token_count=2,
        offset=2000.0,
    )

    prefix_cache = WanPrefixKVCache(
        layers=(
            WanLayerKVCache(layer_index=0, self_attention=self_cache, cross_attention=cross_cache),
            WanLayerKVCache(layer_index=1, self_attention=dynamic_self_cache),
        )
    )

    assert prefix_cache.metadata["diffsynth_wan_exposes_kv_directly"] is False
    assert prefix_cache.metadata["requires_cache_aware_attention_wrappers"] is True
    assert prefix_cache.metadata["serving_integration"] is False
    assert "does not expose" in prefix_cache.metadata["note"]
    assert static_current_image_text_attention_slots(prefix_cache) == (
        (0, WanAttentionKind.SELF),
        (0, WanAttentionKind.CROSS),
    )
    assert dynamic_attention_slots(prefix_cache) == ((1, WanAttentionKind.SELF),)


def test_layer_container_rejects_wrong_attention_kind_and_inconsistent_dims() -> None:
    self_cache = _projected_cache(
        layer_index=0,
        attention_kind=WanAttentionKind.SELF,
        dependencies=(WanKVDependency.TEXT,),
    )
    cross_cache_wrong_dim = _projected_cache(
        layer_index=0,
        attention_kind=WanAttentionKind.CROSS,
        dependencies=(WanKVDependency.TEXT,),
        model_dim=16,
        num_heads=4,
    )

    with pytest.raises(ValueError, match="cross_attention must be a cross"):
        WanLayerKVCache(layer_index=0, cross_attention=self_cache)

    with pytest.raises(ValueError, match="model/head dimensions"):
        WanLayerKVCache(layer_index=0, self_attention=self_cache, cross_attention=cross_cache_wrong_dim)


def test_static_dynamic_dependency_classification() -> None:
    assert is_static_current_image_text_dependencies((WanKVDependency.CURRENT_IMAGE, WanKVDependency.TEXT))
    assert is_static_current_image_text_dependencies("text")
    assert classify_kv_cache_dependencies((WanKVDependency.TEXT,)) == WanKVCacheLifetime.STATIC_CURRENT_IMAGE_TEXT

    assert is_timestep_noise_dependent_dependencies((WanKVDependency.TIMESTEP,))
    assert is_timestep_noise_dependent_dependencies((WanKVDependency.NOISE_LATENTS,))
    assert (
        classify_kv_cache_dependencies((WanKVDependency.TEXT, WanKVDependency.TIMESTEP))
        == WanKVCacheLifetime.DYNAMIC_TIMESTEP_NOISE_OR_ACTION
    )
    assert classify_kv_cache_dependencies((WanKVDependency.ACTION_TOKENS,)) == (
        WanKVCacheLifetime.DYNAMIC_TIMESTEP_NOISE_OR_ACTION
    )
    assert not is_timestep_noise_dependent_dependencies((WanKVDependency.ACTION_TOKENS,))


class _ToyActionContextConsumer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_prefix_key_ids: list[int] = []

    def forward_with_prefix_kv(
        self,
        *,
        action_tokens: torch.Tensor,
        prefix_cache: WanPrefixKVCache,
    ) -> torch.Tensor:
        prefix = prefix_cache.layers[0].self_attention
        if prefix is None:
            raise AssertionError("toy consumer expects self-attention prefix KV")
        self.seen_prefix_key_ids.append(id(prefix.key))
        dynamic_action_context = action_tokens + prefix.value.mean(dim=1, keepdim=True)
        return torch.cat([prefix.value, dynamic_action_context], dim=1)


class _FakeWanSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_prefix_key_ids: list[int] = []
        self.prefix_projection_call_count = 0
        self.cached_forward_call_count = 0

    def _compute_qkv(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tokens + 1.0, tokens + 2.0, tokens + 3.0

    def emit_prefix_kv(
        self,
        *,
        prefix_tokens: torch.Tensor,
        context_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context_tokens is not None:
            raise AssertionError("toy self-attention should emit K/V from prefix tokens, not context tokens")
        self.prefix_projection_call_count += 1
        _, key, value = self._compute_qkv(prefix_tokens)
        return key, value

    def forward_with_cached_kv(
        self,
        *,
        query_tokens: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
    ) -> torch.Tensor:
        self.cached_forward_call_count += 1
        self.seen_prefix_key_ids.append(id(prefix_key))
        dynamic_query, _, dynamic_value = self._compute_qkv(query_tokens)
        dynamic_context = dynamic_query + dynamic_value + prefix_value.mean(dim=1, keepdim=True)
        return torch.cat([prefix_value, dynamic_context], dim=1)


class _FakeProjectedSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 12
        self.num_heads = 3
        self.k = nn.Linear(12, 12, bias=False)
        self.v = nn.Linear(12, 12, bias=False)
        with torch.no_grad():
            self.k.weight.copy_(2.0 * torch.eye(12))
            self.v.weight.copy_(3.0 * torch.eye(12))

    def norm_k(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + 5.0


class _FakeProjectedCrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 12
        self.num_heads = 3
        self.has_image_input = False
        self.k = nn.Linear(12, 12, bias=False)
        self.v = nn.Linear(12, 12, bias=False)
        with torch.no_grad():
            self.k.weight.copy_(4.0 * torch.eye(12))
            self.v.weight.copy_(6.0 * torch.eye(12))

    def norm_k(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens - 7.0


def test_action_context_interface_reuses_prefix_kv_and_allows_dynamic_action_tokens() -> None:
    prefix = _projected_cache(
        layer_index=0,
        attention_kind=WanAttentionKind.SELF,
        dependencies=(WanKVDependency.CURRENT_IMAGE, WanKVDependency.TEXT),
        token_count=2,
    )
    prefix_cache = WanPrefixKVCache(layers=(WanLayerKVCache(layer_index=0, self_attention=prefix),))
    consumer = _ToyActionContextConsumer()
    action_tokens_a = torch.zeros(2, 3, 12)
    action_tokens_b = torch.ones(2, 3, 12)

    output_a = apply_cache_aware_action_context(
        consumer,
        prefix_cache=prefix_cache,
        action_tokens=action_tokens_a,
    )
    output_b = apply_cache_aware_action_context(
        consumer,
        prefix_cache=prefix_cache,
        action_tokens=action_tokens_b,
    )

    assert consumer.seen_prefix_key_ids == [id(prefix.key), id(prefix.key)]
    assert output_a.shape == (2, 5, 12)
    assert torch.equal(output_a[:, :2], prefix.value)
    assert torch.equal(output_b[:, :2], prefix.value)
    assert not torch.equal(output_a[:, 2:], output_b[:, 2:])


def test_fake_wan_self_attention_adapter_emits_validated_prefix_kv_contract() -> None:
    module = _FakeWanSelfAttention()
    wrapper = adapt_wan_attention_module(
        module,
        attention_kind=WanAttentionKind.SELF,
        dependencies=(WanKVDependency.CURRENT_IMAGE, WanKVDependency.TEXT),
        num_heads=3,
        layout=WanKVLayout.DIFFSYNTH_PROJECTED,
        layer_index=0,
        name="toy_wan_self_attention",
    )
    prefix_tokens = torch.arange(2 * 4 * 12, dtype=torch.float32).reshape(2, 4, 12)

    cache = wrapper.emit_prefix_kv(prefix_tokens=prefix_tokens)
    contract = wrapper.describe_cache_contract()

    assert isinstance(cache, WanAttentionKVCache)
    assert module.prefix_projection_call_count == 1
    assert torch.equal(cache.key, prefix_tokens + 2.0)
    assert torch.equal(cache.value, prefix_tokens + 3.0)
    assert cache.attention_kind == WanAttentionKind.SELF
    assert cache.dependencies == frozenset((WanKVDependency.CURRENT_IMAGE, WanKVDependency.TEXT))
    assert cache.shape.batch_size == 2
    assert cache.shape.token_count == 4
    assert cache.shape.model_dim == 12
    assert cache.shape.num_heads == 3
    assert cache.shape.head_dim == 4
    assert cache.shape.layout == WanKVLayout.DIFFSYNTH_PROJECTED
    assert contract["attention_kind"] == "self"
    assert contract["cache_dependency_tags"] == ("current_image", "text")
    assert contract["serving_integration"] is False
    assert contract["diffsynth_wan_exposes_kv_directly"] is False
    assert contract["requires_cache_aware_attention_wrappers"] is True


def test_fake_wan_self_attention_adapter_reuses_prefix_kv_with_dynamic_action_tokens() -> None:
    module = _FakeWanSelfAttention()
    wrapper = adapt_wan_attention_module(
        module,
        attention_kind=WanAttentionKind.SELF,
        dependencies=(WanKVDependency.CURRENT_IMAGE, WanKVDependency.TEXT),
        num_heads=3,
        layout=WanKVLayout.DIFFSYNTH_PROJECTED,
        layer_index=0,
    )
    prefix_tokens = torch.zeros(2, 4, 12)
    prefix = wrapper.emit_prefix_kv(prefix_tokens=prefix_tokens)
    prefix_cache = WanPrefixKVCache(layers=(WanLayerKVCache(layer_index=0, self_attention=prefix),))
    action_tokens_a = torch.zeros(2, 2, 12)
    action_tokens_b = torch.ones(2, 2, 12)

    output_a = apply_cache_aware_action_context(
        wrapper,
        prefix_cache=prefix_cache,
        action_tokens=action_tokens_a,
    )
    output_b = apply_cache_aware_action_context(
        wrapper,
        prefix_cache=prefix_cache,
        action_tokens=action_tokens_b,
    )

    assert module.cached_forward_call_count == 2
    assert module.seen_prefix_key_ids == [id(prefix.key), id(prefix.key)]
    assert output_a.shape == (2, 6, 12)
    assert torch.equal(output_a[:, :4], prefix.value)
    assert torch.equal(output_b[:, :4], prefix.value)
    assert not torch.equal(output_a[:, 4:], output_b[:, 4:])


def test_projected_self_attention_probe_emits_dynamic_validated_kv() -> None:
    module = _FakeProjectedSelfAttention()
    prefix_tokens = torch.arange(2 * 4 * 12, dtype=torch.float32).reshape(2, 4, 12)

    description = describe_wan_attention_module(module, attention_kind=WanAttentionKind.SELF)
    cache = emit_projected_wan_kv(
        module,
        attention_kind=WanAttentionKind.SELF,
        prefix_tokens=prefix_tokens,
        layer_index=2,
        name="fake_projected_self",
    )

    assert description["attention_kind"] == "self"
    assert description["cacheable_projection_names"] == ("k", "v")
    assert description["dynamic_cacheable_projection_names"] == ("k", "v")
    assert description["static_cacheable_projection_names"] == ()
    assert cache.attention_kind == WanAttentionKind.SELF
    assert cache.dependencies == frozenset(
        (WanKVDependency.TIMESTEP, WanKVDependency.NOISE_LATENTS, WanKVDependency.FUTURE_LATENTS)
    )
    assert cache.lifetime == WanKVCacheLifetime.DYNAMIC_TIMESTEP_NOISE_OR_ACTION
    assert cache.shape.batch_size == 2
    assert cache.shape.token_count == 4
    assert cache.shape.model_dim == 12
    assert torch.equal(cache.key, (prefix_tokens * 2.0) + 5.0)
    assert torch.equal(cache.value, prefix_tokens * 3.0)
    assert cache.metadata["production_true_kv_action_conditioning"] is False


def test_projected_cross_attention_probe_emits_static_text_kv() -> None:
    module = _FakeProjectedCrossAttention()
    context_tokens = torch.arange(2 * 5 * 12, dtype=torch.float32).reshape(2, 5, 12)

    description = describe_wan_attention_module(module, attention_kind=WanAttentionKind.CROSS)
    cache = emit_projected_wan_kv(
        module,
        attention_kind=WanAttentionKind.CROSS,
        context_tokens=context_tokens,
        dependencies=(WanKVDependency.TEXT,),
        layer_index=1,
        name="fake_projected_cross",
    )

    assert description["attention_kind"] == "cross"
    assert description["cacheable_projection_names"] == ("k", "v")
    assert description["static_cacheable_projection_names"] == ("k", "v")
    assert description["dynamic_cacheable_projection_names"] == ()
    assert cache.attention_kind == WanAttentionKind.CROSS
    assert cache.dependencies == frozenset((WanKVDependency.TEXT,))
    assert cache.lifetime == WanKVCacheLifetime.STATIC_CURRENT_IMAGE_TEXT
    assert cache.shape.token_count == 5
    assert torch.equal(cache.key, (context_tokens * 4.0) - 7.0)
    assert torch.equal(cache.value, context_tokens * 6.0)


def test_projected_probe_adapter_uses_existing_cache_contract_and_fails_loudly_without_cached_forward() -> None:
    module = _FakeProjectedSelfAttention()
    wrapper = adapt_wan_projection_module(
        module,
        attention_kind=WanAttentionKind.SELF,
        dependencies=(WanKVDependency.CURRENT_IMAGE, WanKVDependency.TEXT),
        layer_index=0,
        name="fake_projected_adapter",
    )
    prefix_tokens = torch.ones(2, 3, 12)
    cache = wrapper.emit_prefix_kv(prefix_tokens=prefix_tokens)

    assert cache.dependencies == frozenset((WanKVDependency.CURRENT_IMAGE, WanKVDependency.TEXT))
    assert cache.shape.token_count == 3
    assert wrapper.describe_cache_contract()["source_module_type"] == "_ProjectedKVBackend"
    with pytest.raises(NotImplementedError, match="cached attention forward is not implemented"):
        wrapper.forward_with_attention_cache(
            query_tokens=torch.ones(2, 1, 12),
            prefix_attention_cache=cache,
        )


def test_action_context_validation_rejects_bad_dynamic_action_tokens() -> None:
    prefix = _projected_cache(
        layer_index=0,
        attention_kind=WanAttentionKind.SELF,
        dependencies=(WanKVDependency.CURRENT_IMAGE, WanKVDependency.TEXT),
    )
    prefix_cache = WanPrefixKVCache(layers=(WanLayerKVCache(layer_index=0, self_attention=prefix),))

    with pytest.raises(ValueError, match="batch size"):
        validate_action_tokens_against_prefix_cache(torch.zeros(3, 2, 12), prefix_cache)

    with pytest.raises(ValueError, match="model dim"):
        validate_action_tokens_against_prefix_cache(torch.zeros(2, 2, 8), prefix_cache)


def test_scaffold_metadata_explicitly_disclaims_diffsynth_integration() -> None:
    metadata = wan_kv_scaffold_metadata()

    assert metadata["training_integration"] is False
    assert metadata["serving_integration"] is False
    assert metadata["diffsynth_wan_exposes_kv_directly"] is False
    assert metadata["requires_cache_aware_attention_wrappers"] is True
    assert "cache-aware SelfAttention/CrossAttention wrappers" in metadata["note"]
