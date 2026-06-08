from __future__ import annotations

import json
import types
from collections.abc import Mapping
from typing import Any

import pytest
import torch

from inspect_alignment import Args, inspect_alignment, main
from world_model.config import DatasetConfig
from world_model.data import sample_to_training_item


def _raw_sample(config: DatasetConfig, *, bad_image_frames: bool = False) -> dict[str, Any]:
    temporal_frames = config.num_future_frames + (2 if bad_image_frames else 1)
    sample: dict[str, Any] = {
        config.state_key: torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),
        config.action_key: torch.tensor(
            [[float(step), float(step + 1)] for step in range(config.action_horizon)],
            dtype=torch.float32,
        ),
        f"{config.action_key}_is_pad": torch.tensor(
            [False] * max(config.action_horizon - 1, 0) + [True],
            dtype=torch.bool,
        ),
        config.task_key: "push the button",
        "episode_index": torch.tensor(7),
        "frame_index": torch.tensor(12),
    }
    for key_index, key in enumerate(config.image_keys):
        sample[key] = torch.full(
            (temporal_frames, config.image_size, config.image_size, 3),
            fill_value=key_index,
            dtype=torch.uint8,
        )
        image_pad = [False] * temporal_frames
        if key_index == 0 and temporal_frames > 1:
            image_pad[-1] = True
        sample[f"{key}_is_pad"] = torch.tensor(image_pad, dtype=torch.bool)
    return sample


class _FakeLeRobotDataset:
    metadata = types.SimpleNamespace(fps=20)

    def __init__(self, config: DatasetConfig, *, bad_image_frames: bool = False):
        self.config = config
        self.bad_image_frames = bad_image_frames

    def __len__(self) -> int:
        return 5

    def raw_sample(self, index: int) -> Mapping[str, Any]:
        del index
        return _raw_sample(self.config, bad_image_frames=self.bad_image_frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return sample_to_training_item(self.raw_sample(index), self.config)

    def task_text(self, index: int) -> str:
        del index
        return "fallback task text"


def test_inspect_alignment_summarizes_fake_lerobot_delta_sample() -> None:
    config = DatasetConfig(
        source="lerobot",
        repo_id="fake/metaworld",
        image_keys=("corner.image", "corner4.image"),
        image_size=8,
        frame_delta=3,
        num_future_frames=2,
        action_horizon=4,
    )
    summary = inspect_alignment(_FakeLeRobotDataset(config), config, sample_index=2)

    assert summary["selected_sample_index"] == 2
    assert summary["sample"]["task"] == "push the button"
    assert summary["sample"]["episode_index"] == 7
    assert summary["sample"]["frame_index"] == 12
    assert summary["action_mask"] == {
        "present": True,
        "length": 4,
        "valid_count": 3,
        "invalid_count": 1,
        "valid_fraction": 0.75,
        "invalid_indices": [3],
    }
    assert summary["future_image_mask"] == {
        "present": True,
        "length": 2,
        "valid_count": 1,
        "invalid_count": 1,
        "valid_fraction": 0.5,
        "invalid_indices": [1],
    }
    assert summary["current_future_frame_offset_contract"]["all_source_frame_offsets"] == [0, 3, 6]
    assert summary["current_future_frame_offset_contract"]["future_source_frame_offsets"] == [3, 6]
    assert summary["current_future_frame_offset_contract"]["delta_timestamps_seconds_by_image_key"][
        "corner.image"
    ] == pytest.approx([0.0, 0.15, 0.3])
    assert summary["action_horizon_contract"]["source_action_offsets"] == [0, 1, 2, 3]
    assert summary["action_horizon_contract"]["delta_timestamps_seconds"] == pytest.approx([0.0, 0.05, 0.1, 0.15])
    assert summary["shapes"]["future_images"] == [2, 2, 3, 8, 8]
    assert summary["raw_shapes"]["actions"] == [4, 2]


def test_main_builds_dataset_config_and_writes_json(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, DatasetConfig] = {}

    def fake_create_dataset(config: DatasetConfig) -> _FakeLeRobotDataset:
        captured["config"] = config
        return _FakeLeRobotDataset(config)

    monkeypatch.setattr("inspect_alignment.create_dataset", fake_create_dataset)
    output_path = tmp_path / "alignment.json"

    main(
        Args(
            dataset_source="lerobot",
            repo_id="fake/repo",
            image_keys=("corner.image",),
            image_size=8,
            frame_delta=2,
            num_future_frames=1,
            action_horizon=3,
            sample_index=1,
            max_samples=4,
            output_json=str(output_path),
        )
    )

    payload = json.loads(output_path.read_text())
    assert captured["config"] == DatasetConfig(
        source="lerobot",
        repo_id="fake/repo",
        image_keys=("corner.image",),
        image_size=8,
        frame_delta=2,
        num_future_frames=1,
        action_horizon=3,
        max_samples=4,
        synthetic_samples=8,
    )
    assert payload["selected_sample_index"] == 1
    assert payload["current_future_frame_offset_contract"]["all_source_frame_offsets"] == [0, 2]
    assert payload["action_horizon_contract"]["source_action_offsets"] == [0, 1, 2]


def test_inspect_alignment_fails_loudly_on_malformed_transformed_sample() -> None:
    config = DatasetConfig(
        source="synthetic",
        image_keys=("corner.image",),
        image_size=8,
        num_future_frames=1,
        action_horizon=2,
    )

    class BadTransformedDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            del index
            return {
                "current_images": torch.zeros((1, 3, 8, 8)),
                "future_images": torch.zeros((1, 1, 3, 8, 8)),
                "state": torch.zeros(4),
                "action_chunk": torch.zeros((2, 1)),
                "action_mask": torch.ones((2, 1)),
                "task_id": torch.tensor(0),
            }

    with pytest.raises(ValueError, match=r"action_mask must have shape \(2,\), got shape \(2, 1\)"):
        inspect_alignment(BadTransformedDataset(), config, sample_index=0)


def test_inspect_alignment_fails_loudly_on_malformed_raw_delta_sample() -> None:
    config = DatasetConfig(
        source="lerobot",
        image_keys=("corner.image",),
        image_size=8,
        num_future_frames=1,
        action_horizon=2,
    )

    with pytest.raises(ValueError, match=r"corner\.image must have 2 temporal frames"):
        inspect_alignment(_FakeLeRobotDataset(config, bad_image_frames=True), config, sample_index=0)
