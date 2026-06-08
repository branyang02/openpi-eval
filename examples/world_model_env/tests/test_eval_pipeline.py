from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import pytest
import torch

from cache_future_rollouts import Args as CacheArgs
from cache_future_rollouts import main as cache_main
from eval_pipeline import Args, main
from world_model.config import DatasetConfig, ModelConfig, TrainConfig
from world_model.train_lib import run_idm_training


def _train_idm_checkpoint(
    output_dir: Path,
    *,
    image_size: int,
    num_future_frames: int,
    action_horizon: int,
    frame_delta: int = 1,
) -> Path:
    """Train a tiny IDM checkpoint whose model_config carries the requested dims."""
    run_idm_training(
        TrainConfig(
            dataset=DatasetConfig(
                source="synthetic",
                image_keys=("corner4.image",),
                image_size=image_size,
                frame_delta=frame_delta,
                max_samples=16,
                synthetic_samples=16,
                num_future_frames=num_future_frames,
                action_horizon=action_horizon,
            ),
            output_dir=str(output_dir),
            epochs=1,
            batch_size=4,
            device="cpu",
            seed=11,
        )
    )
    return output_dir / "idm_checkpoint.pt"


def test_eval_pipeline_rejects_idm_frame_delta_mismatch(tmp_path) -> None:
    checkpoint = _train_idm_checkpoint(
        tmp_path / "train",
        image_size=32,
        num_future_frames=4,
        action_horizon=8,
        frame_delta=1,
    )

    with pytest.raises(ValueError, match="frame_delta=1"):
        main(
            Args(
                idm_checkpoint=str(checkpoint),
                dataset_source="synthetic",
                output_dir=str(tmp_path / "eval"),
                max_samples=16,
                synthetic_samples=16,
                image_size=32,
                frame_delta=2,
                num_future_frames=4,
                action_horizon=8,
                batch_size=4,
                device="cpu",
            )
        )


def test_eval_pipeline_accepts_matching_idm_frame_delta(tmp_path) -> None:
    checkpoint = _train_idm_checkpoint(
        tmp_path / "train",
        image_size=32,
        num_future_frames=4,
        action_horizon=8,
        frame_delta=1,
    )
    output_dir = tmp_path / "eval"

    main(
        Args(
            idm_checkpoint=str(checkpoint),
            dataset_source="synthetic",
            output_dir=str(output_dir),
            max_samples=16,
            synthetic_samples=16,
            image_size=32,
            frame_delta=1,
            num_future_frames=4,
            action_horizon=8,
            batch_size=4,
            device="cpu",
        )
    )

    output = json.loads((output_dir / "pipeline_eval.json").read_text())
    assert output["dataset_config"]["frame_delta"] == 1


def test_eval_pipeline_runs_without_trained_models(tmp_path) -> None:
    main(
        Args(
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            max_samples=16,
            synthetic_samples=16,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
            batch_size=4,
            device="cpu",
        )
    )

    output = json.loads((tmp_path / "pipeline_eval.json").read_text())

    assert output["future_source"] == "dataset_future"
    assert output["idm_source"] == "untrained"
    assert output["flow_eval_seed"] is None
    assert output["metrics"]["idm_mse"] >= 0.0
    assert output["metrics"]["idm_smooth_l1"] >= 0.0
    assert (tmp_path / "current_frame.png").exists()
    assert (tmp_path / "dataset_future_future_debug.mp4").exists()
    assert output["visual_debug"]["future_debug_video"] == str(tmp_path / "dataset_future_future_debug.mp4")
    assert len(list(iio.imiter(tmp_path / "dataset_future_future_debug.mp4"))) == 5


def test_eval_pipeline_uses_checkpoint_dims_for_dataset(tmp_path) -> None:
    checkpoint = _train_idm_checkpoint(
        tmp_path / "train",
        image_size=48,
        num_future_frames=2,
        action_horizon=6,
    )
    output_dir = tmp_path / "eval"

    # The CLI dims below are deliberately stale (different from the checkpoint) to prove
    # the checkpoint's trained dimensions — not these args — drive dataset construction.
    main(
        Args(
            idm_checkpoint=str(checkpoint),
            dataset_source="synthetic",
            output_dir=str(output_dir),
            max_samples=16,
            synthetic_samples=16,
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
            batch_size=4,
            device="cpu",
        )
    )

    output = json.loads((output_dir / "pipeline_eval.json").read_text())

    assert output["idm_source"] == "checkpoint"
    assert output["dataset_config"]["image_size"] == 48
    assert output["dataset_config"]["num_future_frames"] == 2
    assert output["dataset_config"]["action_horizon"] == 6
    assert output["model_config"]["image_size"] == 48
    assert output["model_config"]["num_future_frames"] == 2
    assert output["model_config"]["action_horizon"] == 6
    # The materialized sample must have the checkpoint's spatial/temporal shape.
    assert output["visual_debug"]["current_images_shape"] == [1, 3, 48, 48]
    assert output["visual_debug"]["future_images_shape"] == [2, 1, 3, 48, 48]
    assert output["metrics"]["idm_mse"] >= 0.0
    # current frame + one frame per future view => num_future_frames + 1 frames.
    assert len(list(iio.imiter(output_dir / "dataset_future_future_debug.mp4"))) == 3


def test_eval_pipeline_uses_checkpoint_history_length_for_dataset(tmp_path, monkeypatch) -> None:
    class FakeIdm(torch.nn.Module):
        def forward(self):
            raise AssertionError("evaluate_idm is monkeypatched")

    model_config = ModelConfig(
        num_views=1,
        image_size=8,
        state_dim=4,
        action_dim=1,
        action_horizon=2,
        num_future_frames=1,
        idm_arch="flow_transformer",
        idm_history_length=2,
    )
    captured_configs: list[DatasetConfig] = []

    def fake_create_dataset(config):
        captured_configs.append(config)
        return [object()]

    monkeypatch.setattr("eval_pipeline.load_idm_checkpoint", lambda path, device: (FakeIdm().to(device), model_config))
    monkeypatch.setattr("eval_pipeline.enforce_idm_frame_delta_contract", lambda path, frame_delta: None)
    monkeypatch.setattr("eval_pipeline.create_dataset", fake_create_dataset)
    monkeypatch.setattr(
        "eval_pipeline.evaluate_idm",
        lambda idm, loader, device, *, flow_eval_seed=None: {"idm_mse": 0.0, "idm_smooth_l1": 0.0},
    )
    monkeypatch.setattr("eval_pipeline.write_visual_debug_sample", lambda *args, **kwargs: {})

    main(
        Args(
            idm_checkpoint="fake.pt",
            dataset_source="synthetic",
            output_dir=str(tmp_path),
            image_size=32,
            num_future_frames=4,
            action_horizon=8,
            batch_size=1,
            device="cpu",
        )
    )

    output = json.loads((tmp_path / "pipeline_eval.json").read_text())

    assert captured_configs[0].idm_history_length == 2
    assert output["dataset_config"]["idm_history_length"] == 2


def test_eval_pipeline_checkpoint_dims_drive_cached_future(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    # The cache is built at the checkpoint's (non-default) dims.
    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(cache_dir),
            max_samples=8,
            synthetic_samples=8,
            image_size=48,
            frame_delta=1,
            num_future_frames=2,
            action_horizon=6,
            seed=7,
        )
    )
    checkpoint = _train_idm_checkpoint(
        tmp_path / "train",
        image_size=48,
        num_future_frames=2,
        action_horizon=6,
    )
    output_dir = tmp_path / "eval"

    # With stale CLI dims, the base dataset would mismatch the cache config and fail
    # CachedFutureDataset validation; the checkpoint dims must drive construction so the
    # cache (frame_delta/seed match) is accepted.
    main(
        Args(
            future_source="cached",
            cached_future_dir=str(cache_dir),
            idm_checkpoint=str(checkpoint),
            dataset_source="synthetic",
            output_dir=str(output_dir),
            max_samples=8,
            synthetic_samples=8,
            image_size=32,
            frame_delta=1,
            num_future_frames=4,
            action_horizon=8,
            batch_size=4,
            device="cpu",
            seed=7,
        )
    )

    output = json.loads((output_dir / "pipeline_eval.json").read_text())

    assert output["future_source"] == "cached"
    assert output["idm_source"] == "checkpoint"
    assert output["dataset_config"]["image_size"] == 48
    assert output["dataset_config"]["num_future_frames"] == 2
    assert output["dataset_config"]["action_horizon"] == 6
    assert output["metrics"]["idm_mse"] >= 0.0
