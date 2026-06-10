from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cache_wan_vae_latents import Args as CacheArgs
from cache_wan_vae_latents import precompute_wan_vae_latents
from world_model.config import DatasetConfig, ModelConfig, TrainConfig
from world_model.data import CachedWanVaeLatentDataset, SyntheticMetaWorldFramePairDataset
from world_model.models import WanVaeTransitionEncoder
from world_model.train_lib import run_idm_training
from world_model.wan_vae_encoder import FakeWanVaeEncoder


def _cache_args(cache_dir: Path, **overrides) -> CacheArgs:
    values = {
        "dataset_source": "synthetic",
        "image_keys": ("corner4.image",),
        "output_dir": str(cache_dir),
        "synthetic_samples": 8,
        "image_size": 16,
        "frame_delta": 1,
        "num_future_frames": 4,
        "action_horizon": 4,
        "batch_size": 2,
        "device": "cpu",
        "wan_vae_checkpoint_path": "fake-wan-vae.ckpt",
        "wan_vae_dtype": "float32",
        "wan_vae_latent_channels": 48,
        "wan_vae_spatial_stride": 16,
    }
    values.update(overrides)
    return CacheArgs(**values)


def _dataset_config(**overrides) -> DatasetConfig:
    values = {
        "source": "synthetic",
        "image_keys": ("corner4.image",),
        "image_size": 16,
        "frame_delta": 1,
        "num_future_frames": 4,
        "action_horizon": 4,
        "synthetic_samples": 8,
    }
    values.update(overrides)
    return DatasetConfig(**values)


def _model_config(**overrides) -> ModelConfig:
    values = {
        "num_views": 1,
        "num_future_frames": 4,
        "image_size": 16,
        "state_dim": 4,
        "action_dim": 4,
        "action_horizon": 4,
        "latent_dim": 32,
        "idm_arch": "flow_transformer",
        "idm_visual_encoder": "wan_vae",
        "idm_transformer_layers": 1,
        "idm_transformer_heads": 4,
        "idm_transformer_dropout": 0.0,
        "idm_transformer_ff_dim": 64,
        "idm_flow_sampling_steps": 2,
        "wan_vae_checkpoint_path": "fake-wan-vae.ckpt",
        "wan_vae_dtype": "float32",
        "wan_vae_latent_channels": 48,
        "wan_vae_spatial_stride": 16,
        "wan_vae_use_cached_latents": True,
    }
    values.update(overrides)
    return ModelConfig(**values)


class _ForbiddenWanVaeEncoder:
    def encode_videos(self, videos: torch.Tensor) -> torch.Tensor:
        del videos
        raise AssertionError("cached Wan VAE latents should avoid encoder calls")


def _write_cache(cache_dir: Path) -> None:
    precompute_wan_vae_latents(
        _cache_args(cache_dir),
        wan_encoder=FakeWanVaeEncoder(latent_channels=48, spatial_stride=16),
    )


def test_wan_vae_latent_cache_records_idm_history_length_and_accepts_match(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    precompute_wan_vae_latents(
        _cache_args(cache_dir, idm_history_length=2),
        wan_encoder=FakeWanVaeEncoder(latent_channels=48, spatial_stride=16),
    )
    metadata = json.loads((cache_dir / "config.json").read_text())

    assert metadata["idm_history_length"] == 2
    assert metadata["dataset_config"]["idm_history_length"] == 2

    dataset = CachedWanVaeLatentDataset(
        SyntheticMetaWorldFramePairDataset(_dataset_config(idm_history_length=2)),
        cache_dir,
        model_config=_model_config(idm_history_length=2),
    )

    assert len(dataset) == 8
    assert dataset[0]["prev_state_history"].shape == (2, 4)


def test_wan_vae_latent_cache_rejects_legacy_no_history_metadata_for_history_dataset(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir)
    metadata = json.loads((cache_dir / "config.json").read_text())
    metadata.pop("idm_history_length", None)
    metadata["dataset_config"].pop("idm_history_length", None)
    (cache_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    zero_history_dataset = CachedWanVaeLatentDataset(
        SyntheticMetaWorldFramePairDataset(_dataset_config()),
        cache_dir,
        model_config=_model_config(),
    )
    assert len(zero_history_dataset) == 8

    with pytest.raises(ValueError, match="metadata mismatch.*idm_history_length"):
        CachedWanVaeLatentDataset(
            SyntheticMetaWorldFramePairDataset(_dataset_config(idm_history_length=2)),
            cache_dir,
            model_config=_model_config(idm_history_length=2),
        )


def test_wan_vae_latent_cache_metadata_mismatch_raises(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir)
    dataset = SyntheticMetaWorldFramePairDataset(_dataset_config())

    with pytest.raises(ValueError, match="metadata mismatch.*wan_vae_checkpoint_path"):
        CachedWanVaeLatentDataset(
            dataset,
            cache_dir,
            model_config=_model_config(wan_vae_checkpoint_path="different.ckpt"),
        )

    mismatched_dataset = SyntheticMetaWorldFramePairDataset(_dataset_config(frame_delta=2))
    with pytest.raises(ValueError, match="metadata mismatch.*frame_delta"):
        CachedWanVaeLatentDataset(mismatched_dataset, cache_dir, model_config=_model_config())


def test_wan_vae_latent_cache_hit_preserves_real_futures_and_skips_encoder(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir)
    base_dataset = SyntheticMetaWorldFramePairDataset(_dataset_config())
    cached_dataset = CachedWanVaeLatentDataset(base_dataset, cache_dir, model_config=_model_config())

    base_item = base_dataset[0]
    cached_item = cached_dataset[0]

    assert torch.allclose(cached_item["current_images"], base_item["current_images"])
    assert torch.allclose(cached_item["future_images"], base_item["future_images"])
    assert cached_item["wan_vae_latents"].shape == (48, 2, 1, 1)

    encoder = WanVaeTransitionEncoder(_model_config(), wan_encoder=_ForbiddenWanVaeEncoder())
    context = encoder(
        cached_item["current_images"].unsqueeze(0),
        cached_item["future_images"].unsqueeze(0),
        cached_item["state"].unsqueeze(0),
        wan_vae_latents=cached_item["wan_vae_latents"].unsqueeze(0),
    )

    assert context.shape == (1, 32)


def test_wan_vae_cached_latent_idm_training_smoke(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "idm"
    _write_cache(cache_dir)

    config = TrainConfig(
        dataset=_dataset_config(),
        model=_model_config(),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=2,
        normalize_actions=False,
        device="cpu",
        seed=11,
    )

    metrics = run_idm_training(config, wan_vae_latent_cache_dir=cache_dir)

    assert metrics["training_target"] == "idm"
    assert metrics["cached_future_dir"] is None
    assert metrics["wan_vae_latent_cache_dir"] == str(cache_dir)
    assert metrics["model_config"]["wan_vae_use_cached_latents"]
    assert metrics["final"]["idm_mse"] >= 0.0
    assert (output_dir / "idm_checkpoint.pt").exists()

    written = json.loads((output_dir / "metrics.json").read_text())
    assert written["wan_vae_latent_cache_dir"] == str(cache_dir)
