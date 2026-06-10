from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from world_model.wan_dit_prefix_encoder import (
    DIFFSYNTH_WAN22_TI2V_DIT_SOURCE,
    WAN_DIT_HIDDEN_POOL_DESCRIPTION,
    WAN_DIT_HIDDEN_POOL_TOKEN_POOL,
    FrozenDiffSynthWanDiTCurrentPrefixEncoder,
    WanDiTHiddenFeatureExtractor,
    WanDiTRandomSmokeArgs,
    run_wan_dit_random_feature_smoke,
    wan_dit_future_latent_noise_seed,
)


class _TinyWanBlock(nn.Module):
    def __init__(self, dim: int, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        _t_mod: torch.Tensor | None,
        _freqs: torch.Tensor | None,
    ) -> torch.Tensor:
        context_signal = context.mean(dim=1, keepdim=True)
        return self.proj(x + context_signal) + (self.layer + 1) * 0.01


class _TinyWanDiT(nn.Module):
    def __init__(self, *, dim: int = 8, num_layers: int = 4, seperated_timestep: bool = False) -> None:
        super().__init__()
        self.seperated_timestep = seperated_timestep
        self.blocks = nn.ModuleList([_TinyWanBlock(dim, layer) for layer in range(num_layers)])


def _fake_model_fn(
    *,
    dit: _TinyWanDiT,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    fuse_vae_embedding_in_latents: bool,
    **_kwargs: Any,
) -> torch.Tensor:
    if not fuse_vae_embedding_in_latents:
        raise AssertionError("Wan2.2 TI2V feature extraction should use fused current-image latents.")
    if timestep.ndim != 1:
        raise AssertionError("timestep should be a rank-1 tensor.")
    batch_size, channels, frames, height, width = latents.shape
    x = latents.flatten(2).transpose(1, 2).contiguous()
    for block in dit.blocks:
        x = block(x, context, None, None)
    return x.transpose(1, 2).reshape(batch_size, channels, frames, height, width)


def _fake_separated_timestep_model_fn(
    *,
    dit: _TinyWanDiT,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    fuse_vae_embedding_in_latents: bool,
    **_kwargs: Any,
) -> torch.Tensor:
    if not fuse_vae_embedding_in_latents:
        raise AssertionError("Wan2.2 TI2V feature extraction should use fused current-image latents.")
    batch_size, channels, frames, height, width = latents.shape
    spatial_tokens = height * width // 4
    separated_timestep = torch.concat(
        [
            torch.zeros((1, spatial_tokens), dtype=latents.dtype, device=latents.device),
            torch.ones((frames - 1, spatial_tokens), dtype=latents.dtype, device=latents.device) * timestep,
        ]
    ).flatten()
    if separated_timestep.shape != (frames * spatial_tokens,):
        raise AssertionError(f"unexpected separated timestep shape {tuple(separated_timestep.shape)}.")
    x = latents.flatten(2).transpose(1, 2).contiguous()
    for block in dit.blocks:
        x = block(x, context, None, None)
    return x.transpose(1, 2).reshape(batch_size, channels, frames, height, width)


def test_wan_dit_hidden_extractor_hooks_selected_layers_and_pools() -> None:
    dit = _TinyWanDiT(dim=8, num_layers=4)
    extractor = WanDiTHiddenFeatureExtractor(
        dit,
        selected_layers=(0, 2, 3),
        prefix_dim=4,
        model_fn=_fake_model_fn,
    )
    generator = torch.Generator().manual_seed(13)
    latents = torch.randn(2, 8, 1, 2, 2, generator=generator)
    context = torch.randn(2, 5, 8, generator=generator)
    timestep = torch.full((2,), 500.0)

    result = extractor.extract(latents=latents, timestep=timestep, context=context)

    assert result.prefix_tokens.shape == (2, 3, 4)
    assert result.denoised_latents.shape == latents.shape
    assert result.captured_layers == (0, 2, 3)
    assert result.metadata["source"] == DIFFSYNTH_WAN22_TI2V_DIT_SOURCE
    assert result.metadata["hidden_pool"] == WAN_DIT_HIDDEN_POOL_DESCRIPTION
    assert result.metadata["tokens_per_layer"] == 1
    assert result.metadata["prefix_token_count"] == 3
    assert result.metadata["raw_hidden_dim"] == 8
    assert result.metadata["prefix_dim"] == 4
    assert not dit.blocks[0]._forward_hooks
    assert not dit.blocks[2]._forward_hooks
    assert torch.isfinite(result.prefix_tokens).all()


def test_wan_dit_hidden_extractor_token_pool_returns_multiple_tokens_per_layer() -> None:
    dit = _TinyWanDiT(dim=8, num_layers=4)
    extractor = WanDiTHiddenFeatureExtractor(
        dit,
        selected_layers=(0, 2),
        prefix_dim=4,
        hidden_pool=WAN_DIT_HIDDEN_POOL_TOKEN_POOL,
        tokens_per_layer=4,
        model_fn=_fake_model_fn,
    )
    generator = torch.Generator().manual_seed(17)
    latents = torch.randn(2, 8, 1, 4, 4, generator=generator)
    context = torch.randn(2, 5, 8, generator=generator)
    timestep = torch.full((1,), 500.0)

    result = extractor.extract(latents=latents, timestep=timestep, context=context)

    assert result.prefix_tokens.shape == (2, 8, 4)
    assert result.captured_layers == (0, 2)
    assert result.metadata["hidden_pool"] == WAN_DIT_HIDDEN_POOL_TOKEN_POOL
    assert result.metadata["tokens_per_layer"] == 4
    assert result.metadata["prefix_token_count"] == 8
    assert torch.isfinite(result.prefix_tokens).all()


def test_wan_dit_hidden_extractor_marks_single_frame_effective_timestep_zero() -> None:
    dit = _TinyWanDiT(dim=8, num_layers=2, seperated_timestep=True)
    extractor = WanDiTHiddenFeatureExtractor(dit, selected_layers=(1,), model_fn=_fake_model_fn)
    latents = torch.randn(1, 8, 1, 2, 2)
    context = torch.randn(1, 3, 8)
    timestep = torch.full((1,), 500.0)

    result = extractor.extract(latents=latents, timestep=timestep, context=context)

    assert result.metadata["timestep_applies_to_future_latents_only"] is True
    assert result.metadata["future_latent_frames"] == 0
    assert result.metadata["effective_timestep"] == 0.0


def test_wan_dit_hidden_extractor_is_frozen_and_no_grad() -> None:
    dit = _TinyWanDiT(dim=8, num_layers=2)
    extractor = WanDiTHiddenFeatureExtractor(dit, selected_layers=(1,), model_fn=_fake_model_fn)
    latents = torch.randn(1, 8, 1, 2, 2, requires_grad=True)
    context = torch.randn(1, 3, 8, requires_grad=True)
    timestep = torch.full((1,), 250.0)

    tokens = extractor.extract_hidden_tokens(latents=latents, timestep=timestep, context=context)

    assert tokens.shape == (1, 1, 8)
    assert tokens.requires_grad is False
    assert all(parameter.requires_grad is False for parameter in dit.parameters())


def test_wan_dit_hidden_extractor_validates_layer_indices() -> None:
    dit = _TinyWanDiT(dim=8, num_layers=2)

    with pytest.raises(ValueError, match="out of range"):
        WanDiTHiddenFeatureExtractor(dit, selected_layers=(2,), model_fn=_fake_model_fn)

    with pytest.raises(ValueError, match="duplicates"):
        WanDiTHiddenFeatureExtractor(dit, selected_layers=(1, 1), model_fn=_fake_model_fn)


def test_wan_dit_hidden_extractor_validates_tokens_per_layer() -> None:
    dit = _TinyWanDiT(dim=8, num_layers=2)

    with pytest.raises(ValueError, match="tokens_per_layer must be positive"):
        WanDiTHiddenFeatureExtractor(
            dit,
            selected_layers=(1,),
            hidden_pool=WAN_DIT_HIDDEN_POOL_TOKEN_POOL,
            tokens_per_layer=0,
            model_fn=_fake_model_fn,
        )


def test_wan_dit_hidden_extractor_removes_hooks_when_model_fn_fails() -> None:
    dit = _TinyWanDiT(dim=8, num_layers=2)

    def failing_model_fn(**_kwargs: Any) -> torch.Tensor:
        raise RuntimeError("boom")

    extractor = WanDiTHiddenFeatureExtractor(dit, selected_layers=(0, 1), model_fn=failing_model_fn)
    latents = torch.randn(1, 8, 1, 2, 2)
    context = torch.randn(1, 3, 8)
    timestep = torch.full((1,), 500.0)

    with pytest.raises(RuntimeError, match="boom"):
        extractor.extract(latents=latents, timestep=timestep, context=context)

    assert not dit.blocks[0]._forward_hooks
    assert not dit.blocks[1]._forward_hooks


def test_frozen_wan_dit_prefix_encoder_signature_is_current_only() -> None:
    parameters = inspect.signature(FrozenDiffSynthWanDiTCurrentPrefixEncoder.encode_prefix).parameters

    assert set(parameters) == {"self", "current_images", "prompts", "sample_indices", "future_latents"}
    assert "future_images" not in parameters
    assert "actions" not in parameters


def test_future_latent_noise_seed_is_keyed_by_sample_index() -> None:
    seed_for_index_5 = wan_dit_future_latent_noise_seed(123, 5)

    assert seed_for_index_5 == wan_dit_future_latent_noise_seed(123, 5)
    assert seed_for_index_5 != wan_dit_future_latent_noise_seed(123, 6)
    assert seed_for_index_5 != wan_dit_future_latent_noise_seed(124, 5)


def test_noise_future_latents_are_stable_by_sample_index_across_batch_order() -> None:
    encoder = object.__new__(FrozenDiffSynthWanDiTCurrentPrefixEncoder)
    encoder.num_latent_frames = 3
    encoder.future_latent_fill = "noise"
    encoder.future_latent_seed = 123

    first_frame_latents = torch.zeros(3, 2, 1, 2, 2)
    ordered = encoder._build_current_only_latents(first_frame_latents, sample_indices=[7, 8, 9])
    shuffled = encoder._build_current_only_latents(first_frame_latents, sample_indices=[9, 7, 8])

    assert torch.allclose(ordered[0, :, 1:], shuffled[1, :, 1:])
    assert torch.allclose(ordered[1, :, 1:], shuffled[2, :, 1:])
    assert torch.allclose(ordered[2, :, 1:], shuffled[0, :, 1:])
    assert not torch.allclose(ordered[0, :, 1:], ordered[1, :, 1:])


def test_noise_future_latents_keep_legacy_batch_position_behavior_without_sample_indices() -> None:
    encoder = object.__new__(FrozenDiffSynthWanDiTCurrentPrefixEncoder)
    encoder.num_latent_frames = 3
    encoder.future_latent_fill = "noise"
    encoder.future_latent_seed = 123

    first_frame_latents = torch.zeros(2, 2, 1, 2, 2)
    first = encoder._build_current_only_latents(first_frame_latents)
    second = encoder._build_current_only_latents(first_frame_latents)

    assert torch.allclose(first, second)


def test_explicit_future_latents_fill_future_slots_and_skip_noise() -> None:
    encoder = object.__new__(FrozenDiffSynthWanDiTCurrentPrefixEncoder)
    encoder.num_latent_frames = 3
    encoder.future_latent_fill = "noise"
    encoder.future_latent_seed = 123

    first_frame_latents = torch.zeros(2, 4, 1, 2, 2)
    future_latents = torch.arange(2 * 4 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 4, 2, 2, 2)

    latents = encoder._build_current_only_latents(first_frame_latents, future_latents=future_latents)

    assert torch.allclose(latents[:, :, :1], first_frame_latents)
    assert torch.allclose(latents[:, :, 1:], future_latents)


def test_explicit_future_latents_validate_shape() -> None:
    encoder = object.__new__(FrozenDiffSynthWanDiTCurrentPrefixEncoder)
    encoder.num_latent_frames = 3
    encoder.future_latent_fill = "zeros"
    encoder.future_latent_seed = 0
    first_frame_latents = torch.zeros(2, 4, 1, 2, 2)

    with pytest.raises(ValueError, match="future_latents must have shape"):
        encoder._build_current_only_latents(first_frame_latents, future_latents=torch.zeros(2, 4, 1, 2, 2))

    encoder.num_latent_frames = 1
    with pytest.raises(ValueError, match="future_latents require num_latent_frames > 1"):
        encoder._build_current_only_latents(first_frame_latents, future_latents=torch.zeros(2, 4, 0, 2, 2))


def test_frozen_wan_dit_prefix_encoder_uses_shared_timestep_for_ti2v_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = object.__new__(FrozenDiffSynthWanDiTCurrentPrefixEncoder)
    encoder.dtype = "float32"
    encoder.timestep = 500.0
    encoder.num_latent_frames = 1
    encoder.future_latent_fill = "zeros"
    encoder.future_latent_seed = 0

    dit = _TinyWanDiT(dim=8, num_layers=1, seperated_timestep=True)
    extractor = WanDiTHiddenFeatureExtractor(
        dit,
        selected_layers=(0,),
        prefix_dim=4,
        model_fn=_fake_separated_timestep_model_fn,
    )
    first_frame_latents = torch.randn(2, 8, 1, 4, 4)
    text_context = torch.randn(2, 3, 8)

    monkeypatch.setattr(encoder, "_load_pipeline", lambda _device: (object(), extractor))
    monkeypatch.setattr(encoder, "_encode_first_frame_latents", lambda _pipe, _current_images: first_frame_latents)
    monkeypatch.setattr(encoder, "_encode_text_context", lambda _pipe, _prompts: text_context)

    prefix_tokens = encoder.encode_prefix(torch.rand(2, 3, 32, 32), ["pick", "place"])

    assert prefix_tokens.shape == (2, 1, 4)
    assert prefix_tokens.dtype == torch.float32


def test_missing_real_wan_dit_paths_fail_clearly(tmp_path: Path) -> None:
    repo_dir = tmp_path / "DiffSynth-Studio"
    (repo_dir / "diffsynth").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Wan2.2 DiT checkpoint shard"):
        FrozenDiffSynthWanDiTCurrentPrefixEncoder(
            repo_dir=repo_dir,
            checkpoint_dir=tmp_path / "missing-wan2.2-ti2v-5b",
            prefix_dim=8,
        )


def test_wan_dit_random_smoke_is_guarded_when_paths_are_missing(tmp_path: Path) -> None:
    result = run_wan_dit_random_feature_smoke(
        WanDiTRandomSmokeArgs(
            repo_dir=str(tmp_path / "missing-diffsynth"),
            checkpoint_dir=str(tmp_path / "missing-wan"),
            require_gpu=False,
        )
    )

    assert result["skipped"] is True
    assert "DiffSynth-Studio repo" in result["reason"]


@pytest.mark.skipif(
    os.environ.get("RUN_WAN_DIT_REAL_SMOKE") != "1",
    reason="Set RUN_WAN_DIT_REAL_SMOKE=1 to load the local Wan2.2 TI2V-5B DiT.",
)
def test_real_wan_dit_random_feature_smoke_opt_in() -> None:
    result = run_wan_dit_random_feature_smoke(WanDiTRandomSmokeArgs())
    if result.get("skipped"):
        pytest.skip(str(result["reason"]))

    assert result["skipped"] is False
    assert result["prefix_shape"] == [1, 3, 3072]
    assert result["denoised_latent_shape"] == [1, 48, 1, 2, 2]
    assert result["timestep_shape"] == [1]
    assert result["effective_timestep"] == 0.0
