from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset

from world_model.config import DatasetConfig, TrainConfig
from world_model.train_lib import effective_training_split_gap, split_dataset


class SequentialDataset(Dataset):
    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return index


class EpisodeDataset(SequentialDataset):
    def __init__(self, episode_ids: list[int]):
        super().__init__(len(episode_ids))
        self.episode_ids = episode_ids

    def raw_sample(self, index: int) -> dict[str, torch.Tensor]:
        return {"episode_index": torch.tensor(self.episode_ids[index])}


def _indices(dataset: Dataset) -> list[int]:
    assert isinstance(dataset, Subset)
    return [int(index) for index in dataset.indices]


def test_contiguous_split_drops_gap_to_prevent_adjacent_window_leakage() -> None:
    train, eval_ = split_dataset(SequentialDataset(20), eval_fraction=0.25, seed=123, split_gap=1)

    train_indices = _indices(train)
    eval_indices = _indices(eval_)

    assert train_indices == list(range(14))
    assert eval_indices == list(range(15, 20))
    assert 14 not in train_indices
    assert 14 not in eval_indices
    assert all(abs(train_index - eval_index) > 1 for train_index in train_indices for eval_index in eval_indices)


def test_contiguous_split_gap_is_configurable_and_sizes_remain_sane() -> None:
    train, eval_ = split_dataset(SequentialDataset(20), eval_fraction=0.25, seed=123, split_gap=3)

    train_indices = _indices(train)
    eval_indices = _indices(eval_)

    assert train_indices == list(range(12))
    assert eval_indices == list(range(15, 20))
    assert set(range(12, 15)).isdisjoint(train_indices)
    assert set(range(12, 15)).isdisjoint(eval_indices)
    assert len(train_indices) == 12
    assert len(eval_indices) == 5


def test_episode_aware_split_holds_out_whole_episodes_without_per_frame_leakage() -> None:
    dataset = EpisodeDataset([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)

    train, eval_ = split_dataset(dataset, eval_fraction=0.25, seed=123, split_gap=3)

    assert _indices(train) == list(range(15))
    assert _indices(eval_) == list(range(15, 20))
    assert {dataset.episode_ids[index] for index in _indices(train)} == {0, 1, 2}
    assert {dataset.episode_ids[index] for index in _indices(eval_)} == {3}


def test_effective_training_split_gap_covers_real_dataset_temporal_window() -> None:
    config = TrainConfig(
        dataset=DatasetConfig(
            source="lerobot",
            frame_delta=4,
            num_future_frames=2,
            action_horizon=5,
        ),
        split_gap=1,
    )

    assert effective_training_split_gap(config) == 8


def test_effective_training_split_gap_keeps_synthetic_explicit_gap() -> None:
    config = TrainConfig(
        dataset=DatasetConfig(
            source="synthetic",
            frame_delta=4,
            num_future_frames=2,
            action_horizon=5,
        ),
        split_gap=1,
    )

    assert effective_training_split_gap(config) == 1
