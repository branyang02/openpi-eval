"""Checkpoint metadata consistency for the training entrypoints.

These tests guard against a regression where a saved checkpoint's top-level
``model_config`` reflected the dataset-resolved model dimensions while
``train_config['model']`` still carried the stale ``ModelConfig`` defaults that
were passed in before resolution. That mismatch is confusing for downstream
rank/eval/debug tooling, so newly trained checkpoints must serialize the two
consistently while preserving ``train_config['dataset']``.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from world_model.config import DatasetConfig, ModelConfig, TrainConfig
from world_model.train_lib import (
    assert_train_config_model_matches,
    run_idm_training,
    run_training,
)


def _non_default_dataset() -> DatasetConfig:
    # Every dimension below resolves to a value that differs from the
    # ``ModelConfig`` defaults (num_views=3, image_size=64, action_horizon=32,
    # num_future_frames=1) so a stale ``train_config['model']`` is detectable.
    return DatasetConfig(
        source="synthetic",
        image_keys=("corner4.image",),  # -> num_views=1
        image_size=32,
        synthetic_samples=16,
        num_future_frames=4,
        action_horizon=4,
    )


def test_idm_checkpoint_train_config_model_matches_resolved_model_config(tmp_path) -> None:
    dataset = _non_default_dataset()
    config = TrainConfig(
        dataset=dataset,
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=4,
        device="cpu",
        seed=11,
    )

    run_idm_training(config)

    for name in ("idm_checkpoint.pt", "best_idm_checkpoint.pt"):
        checkpoint = torch.load(tmp_path / name, map_location="cpu", weights_only=False)
        model_config = checkpoint["model_config"]

        # Top-level model_config carries the dataset-resolved dims, not defaults.
        assert model_config["num_views"] == 1, name
        assert model_config["image_size"] == 32, name
        assert model_config["action_horizon"] == 4, name
        assert model_config["num_future_frames"] == 4, name

        # train_config['model'] must match the resolved model_config exactly.
        assert checkpoint["train_config"]["model"] == model_config, name

        # train_config['dataset'] must be preserved unchanged.
        assert checkpoint["train_config"]["dataset"] == dataclasses.asdict(dataset), name


def test_run_training_checkpoint_train_config_model_matches_model_config(tmp_path) -> None:
    dataset = _non_default_dataset()
    config = TrainConfig(
        dataset=dataset,
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=4,
        device="cpu",
        seed=11,
    )

    run_training(config)

    checkpoint = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]

    assert model_config["num_views"] == 1
    assert model_config["action_horizon"] == 4
    assert model_config["num_future_frames"] == 4
    assert checkpoint["train_config"]["model"] == model_config
    assert checkpoint["train_config"]["dataset"] == dataclasses.asdict(dataset)


def test_assert_train_config_model_matches_raises_on_stale_model() -> None:
    # Resolved dims (what the model was actually built with)...
    resolved = ModelConfig(num_views=1, image_size=32, action_horizon=4, num_future_frames=4)
    # ...vs a TrainConfig still carrying ModelConfig() defaults.
    stale = TrainConfig()

    with pytest.raises(ValueError, match="train_config.model"):
        assert_train_config_model_matches(stale, resolved)


def test_assert_train_config_model_matches_accepts_consistent_config() -> None:
    resolved = ModelConfig(num_views=1, image_size=32, action_horizon=4, num_future_frames=4)
    consistent = dataclasses.replace(TrainConfig(), model=resolved)

    assert_train_config_model_matches(consistent, resolved)  # must not raise
