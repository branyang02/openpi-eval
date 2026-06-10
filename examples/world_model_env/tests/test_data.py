from __future__ import annotations

import sys
import types

import pytest
import torch

from world_model.config import DatasetConfig
from world_model.data import (
    MetaWorldFramePairDataset,
    PromptFromLeRobotTask,
    SyntheticMetaWorldFramePairDataset,
    build_delta_timestamps,
    expected_wan_selected_frame_indices,
    expected_wan_source_frame_offsets,
    image_to_chw_float,
    repair_lerobot_episode_data_index,
    sample_to_training_item,
    split_frame_pair,
    stable_task_id,
    temporal_image_stack,
    validate_raw_wan_frame_delta,
    validate_wan_selected_frame_indices,
)


def _install_fake_lerobot(monkeypatch: pytest.MonkeyPatch, episode_ids: list[int]) -> None:
    class FakeHfDataset:
        def __init__(self, selected_episode_ids: list[int]):
            self.selected_episode_ids = selected_episode_ids

        def __getitem__(self, key):
            if key != "episode_index":
                raise KeyError(key)
            return [torch.tensor(episode) for episode in self.selected_episode_ids]

    class FakeLeRobotDataset:
        def __init__(self, repo_id, *, episodes, delta_timestamps):
            del repo_id, delta_timestamps
            allowed_episodes = None if episodes is None else set(episodes)
            self.episode_ids = [
                episode for episode in episode_ids if allowed_episodes is None or episode in allowed_episodes
            ]
            self.hf_dataset = FakeHfDataset(self.episode_ids)
            self.episode_data_index = {
                "from": torch.zeros(max(len(self.episode_ids), 1), dtype=torch.long),
                "to": torch.zeros(max(len(self.episode_ids), 1), dtype=torch.long),
            }

        def __len__(self):
            return len(self.episode_ids)

        def __getitem__(self, index):
            return {
                "corner4.image": torch.zeros((2, 16, 16, 3), dtype=torch.uint8),
                "observation.state": torch.zeros((1, 4), dtype=torch.float32),
                "actions": torch.zeros((4, 4), dtype=torch.float32),
                "task": "fake task",
                "task_index": torch.tensor(0),
                "episode_index": torch.tensor(self.episode_ids[index]),
                "frame_index": torch.tensor(index),
            }

    class FakeLeRobotDatasetMetadata:
        fps = 80
        tasks = {0: "fake task"}

        def __init__(self, repo_id):
            del repo_id

    lerobot_module = types.ModuleType("lerobot")
    common_module = types.ModuleType("lerobot.common")
    datasets_module = types.ModuleType("lerobot.common.datasets")
    lerobot_dataset_module = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
    lerobot_dataset_module.LeRobotDataset = FakeLeRobotDataset
    lerobot_dataset_module.LeRobotDatasetMetadata = FakeLeRobotDatasetMetadata
    datasets_module.lerobot_dataset = lerobot_dataset_module
    common_module.datasets = datasets_module
    lerobot_module.common = common_module
    monkeypatch.setitem(sys.modules, "lerobot", lerobot_module)
    monkeypatch.setitem(sys.modules, "lerobot.common", common_module)
    monkeypatch.setitem(sys.modules, "lerobot.common.datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "lerobot.common.datasets.lerobot_dataset", lerobot_dataset_module)


def test_build_delta_timestamps_matches_openpi_lerobot_style() -> None:
    config = DatasetConfig(
        image_keys=("corner4.image",),
        frame_delta=2,
        num_future_frames=3,
        action_horizon=4,
    )

    dts = build_delta_timestamps(config, fps=80)

    assert dts["corner4.image"] == [0.0, 2 / 80, 4 / 80, 6 / 80]
    assert dts["observation.state"] == [0.0]
    assert dts["actions"] == [0.0, 1 / 80, 2 / 80, 3 / 80]


def test_build_delta_timestamps_adds_explicit_idm_history_window() -> None:
    config = DatasetConfig(
        image_keys=("corner4.image",),
        frame_delta=2,
        num_future_frames=1,
        action_horizon=3,
        idm_history_length=2,
    )

    dts = build_delta_timestamps(config, fps=10)

    assert dts["corner4.image"] == [0.0, 0.2]
    assert dts["observation.state"] == [-0.2, -0.1, 0.0]
    assert dts["actions"] == [-0.2, -0.1, 0.0, 0.1, 0.2]


def test_wan_temporal_helpers_separate_video_indices_from_dataset_offsets() -> None:
    assert expected_wan_selected_frame_indices(frame_delta=4, num_future_frames=3) == [1, 2, 3]
    assert expected_wan_selected_frame_indices(
        frame_delta=4,
        num_future_frames=3,
        strategy="source_offsets",
    ) == [4, 8, 12]
    assert expected_wan_source_frame_offsets(frame_delta=4, num_future_frames=3) == [4, 8, 12]

    validate_wan_selected_frame_indices(
        [1, 2, 3],
        frame_delta=4,
        num_future_frames=3,
        context="test cache",
    )
    with pytest.raises(ValueError, match=r"generated-video frame contract \[1, 2, 3\]"):
        validate_wan_selected_frame_indices(
            [1, 4, 8],
            frame_delta=4,
            num_future_frames=3,
            context="test cache",
        )
    validate_wan_selected_frame_indices(
        [4, 8, 12],
        frame_delta=4,
        num_future_frames=3,
        strategy="source_offsets",
        context="dense Wan cache",
    )
    with pytest.raises(ValueError, match="Raw Wan2.2 only supports frame_delta=1"):
        validate_raw_wan_frame_delta(4, context="unit test")


def test_prompt_from_lerobot_task_uses_metadata_task_mapping() -> None:
    transform = PromptFromLeRobotTask({3: "put the banana in the bowl"})

    output = transform({"task_index": torch.tensor(3)})

    assert output["task"] == "put the banana in the bowl"


def test_repair_lerobot_episode_data_index_expands_original_episode_ids() -> None:
    class FakeHfDataset:
        def __getitem__(self, key):
            assert key == "episode_index"
            return [torch.tensor(16), torch.tensor(16), torch.tensor(18)]

    class FakeLeRobotDataset:
        hf_dataset = FakeHfDataset()
        episode_data_index = {"from": torch.tensor([0, 2]), "to": torch.tensor([2, 3])}

    dataset = FakeLeRobotDataset()

    repair_lerobot_episode_data_index(dataset, [16, 18])

    assert int(dataset.episode_data_index["from"][16]) == 0
    assert int(dataset.episode_data_index["to"][16]) == 2
    assert int(dataset.episode_data_index["from"][18]) == 2
    assert int(dataset.episode_data_index["to"][18]) == 3


def test_lerobot_samples_per_episode_balances_requested_episodes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_lerobot(monkeypatch, [10] * 6 + [20] * 4 + [30] * 5)
    dataset = MetaWorldFramePairDataset(
        DatasetConfig(
            source="lerobot",
            image_keys=("corner4.image",),
            episodes=(10, 20, 30),
            frame_delta=1,
            num_future_frames=1,
            action_horizon=2,
            samples_per_episode=2,
            prompt_from_task=False,
        )
    )

    samples = [dataset.raw_sample(index) for index in range(len(dataset))]

    assert len(dataset) == 6
    assert [int(sample["episode_index"]) for sample in samples] == [10, 10, 20, 20, 30, 30]
    assert [int(sample["frame_index"]) for sample in samples] == [0, 4, 6, 8, 10, 13]


def test_lerobot_samples_per_episode_rejects_short_nonterminal_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lerobot(monkeypatch, [10] * 3)

    with pytest.raises(ValueError, match="valid non-terminal window count"):
        MetaWorldFramePairDataset(
            DatasetConfig(
                source="lerobot",
                image_keys=("corner4.image",),
                episodes=(10,),
                frame_delta=1,
                num_future_frames=1,
                action_horizon=3,
                samples_per_episode=2,
                prompt_from_task=False,
            )
        )


def test_dataset_config_rejects_samples_per_episode_with_max_samples() -> None:
    with pytest.raises(ValueError, match="samples_per_episode.*max_samples"):
        DatasetConfig(source="lerobot", max_samples=10, samples_per_episode=2)


def test_synthetic_dataset_shapes() -> None:
    dataset = SyntheticMetaWorldFramePairDataset(
        DatasetConfig(image_size=32, synthetic_samples=4, num_future_frames=2, action_horizon=8)
    )
    sample = dataset[0]

    assert sample["current_images"].shape == (3, 3, 32, 32)
    assert sample["future_images"].shape == (2, 3, 3, 32, 32)
    assert sample["future_image_mask"].shape == (2,)
    assert sample["state"].shape == (4,)
    assert sample["action_chunk"].shape == (8, 4)
    assert sample["action_mask"].shape == (8,)
    assert sample["task_id"].dtype == torch.long
    assert int(sample["dataset_index"]) == 0
    assert int(sample["episode_index"]) == 0
    assert int(sample["frame_index"]) == 0
    assert int(sample["task_index"]) == int(sample["task_id"])
    assert dataset.task_text(0) == "synthetic metaworld task 0"


def test_synthetic_dataset_history_shapes() -> None:
    dataset = SyntheticMetaWorldFramePairDataset(
        DatasetConfig(
            image_size=32,
            synthetic_samples=4,
            num_future_frames=2,
            action_horizon=8,
            idm_history_length=3,
        )
    )
    sample = dataset[0]

    assert sample["prev_state_history"].shape == (3, 4)
    assert sample["prev_action_history"].shape == (3, 4)
    assert torch.allclose(sample["history_mask"], torch.ones(3))


def test_image_pair_parsing_from_delta_timestamp_tensor() -> None:
    pair = torch.zeros((2, 16, 16, 3), dtype=torch.uint8)
    current, target = split_frame_pair(pair)

    assert current.shape == (16, 16, 3)
    assert target.shape == (16, 16, 3)


def test_image_to_chw_float_resizes_and_normalizes() -> None:
    image = torch.full((16, 16, 3), 255, dtype=torch.uint8)
    tensor = image_to_chw_float(image, image_size=32)

    assert tensor.shape == (3, 32, 32)
    assert torch.allclose(tensor.max(), torch.tensor(1.0))


def test_temporal_image_stack_accepts_chw_float() -> None:
    images = torch.ones((2, 3, 16, 16), dtype=torch.float32)
    tensor = temporal_image_stack(images, image_size=16)

    assert tensor.shape == (2, 3, 16, 16)


def test_sample_to_training_item_matches_metaworld_dataset_contract() -> None:
    config = DatasetConfig(image_size=16, num_future_frames=2, action_horizon=4)
    sample = {
        "corner.image": torch.zeros((3, 16, 16, 3), dtype=torch.uint8),
        "corner.image_is_pad": torch.tensor([False, False, True]),
        "corner4.image": torch.zeros((3, 16, 16, 3), dtype=torch.uint8),
        "corner4.image_is_pad": torch.tensor([False, False, False]),
        "gripperPOV.image": torch.zeros((3, 16, 16, 3), dtype=torch.uint8),
        "gripperPOV.image_is_pad": torch.tensor([False, False, False]),
        "observation.state": torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),
        "actions": torch.tensor(
            [
                [0.5, 0.0, -0.5, 1.0],
                [0.4, 0.0, -0.4, 1.0],
                [0.3, 0.0, -0.3, 1.0],
                [0.2, 0.0, -0.2, 1.0],
            ],
            dtype=torch.float32,
        ),
        "actions_is_pad": torch.tensor([False, False, True, True]),
        "task": "open the drawer",
        "dataset_index": torch.tensor(42),
        "episode_index": torch.tensor(7),
        "frame_index": torch.tensor(123),
        "task_index": torch.tensor(3),
    }

    item = sample_to_training_item(sample, config)

    assert item["current_images"].shape == (3, 3, 16, 16)
    assert item["future_images"].shape == (2, 3, 3, 16, 16)
    assert torch.allclose(item["future_image_mask"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(item["state"], torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert item["action_chunk"].shape == (4, 4)
    assert torch.allclose(item["action_mask"], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert 0 <= int(item["task_id"]) < config.task_vocab_size
    assert int(item["dataset_index"]) == 42
    assert int(item["episode_index"]) == 7
    assert int(item["frame_index"]) == 123
    assert int(item["task_index"]) == 3


def test_lerobot_dataset_item_preserves_batch_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_lerobot(monkeypatch, [10, 10, 20])
    dataset = MetaWorldFramePairDataset(
        DatasetConfig(
            source="lerobot",
            image_keys=("corner4.image",),
            image_size=16,
            action_horizon=4,
            prompt_from_task=True,
        )
    )

    item = dataset[1]

    assert int(item["dataset_index"]) == 1
    assert int(item["episode_index"]) == 10
    assert int(item["frame_index"]) == 1
    assert int(item["task_index"]) == 0


def test_sample_to_training_item_splits_idm_history_from_lerobot_temporal_window() -> None:
    config = DatasetConfig(
        image_size=16,
        num_future_frames=1,
        action_horizon=3,
        idm_history_length=2,
        image_keys=("corner4.image",),
    )
    sample = {
        "corner4.image": torch.zeros((2, 16, 16, 3), dtype=torch.uint8),
        "corner4.image_is_pad": torch.tensor([False, False]),
        "observation.state": torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
            ],
            dtype=torch.float32,
        ),
        "observation.state_is_pad": torch.tensor([False, True, False]),
        "actions": torch.tensor(
            [
                [0.0, 0.1, 0.2, 0.3],
                [1.0, 1.1, 1.2, 1.3],
                [2.0, 2.1, 2.2, 2.3],
                [3.0, 3.1, 3.2, 3.3],
                [4.0, 4.1, 4.2, 4.3],
            ],
            dtype=torch.float32,
        ),
        "actions_is_pad": torch.tensor([False, False, False, True, False]),
        "task": "open the drawer",
    }

    item = sample_to_training_item(sample, config)

    assert torch.allclose(item["state"], torch.tensor([0.9, 1.0, 1.1, 1.2]))
    assert torch.allclose(item["prev_state_history"], sample["observation.state"][:2])
    assert torch.allclose(item["prev_action_history"], sample["actions"][:2])
    assert torch.allclose(item["action_chunk"], sample["actions"][2:])
    assert torch.allclose(item["history_mask"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(item["action_mask"], torch.tensor([1.0, 0.0, 1.0]))


def test_uses_plural_actions_key() -> None:
    config = DatasetConfig(image_size=16, action_horizon=4)
    sample = {
        "corner.image": torch.zeros((2, 16, 16, 3), dtype=torch.uint8),
        "corner4.image": torch.zeros((2, 16, 16, 3), dtype=torch.uint8),
        "gripperPOV.image": torch.zeros((2, 16, 16, 3), dtype=torch.uint8),
        "observation.state": torch.zeros((1, 4), dtype=torch.float32),
        "action": torch.zeros((4, 4), dtype=torch.float32),
        "task": "open the drawer",
    }

    with pytest.raises(KeyError, match="actions"):
        sample_to_training_item(sample, config)


def test_sample_to_training_item_fails_on_missing_key() -> None:
    config = DatasetConfig(image_size=16)

    with pytest.raises(KeyError, match="corner.image"):
        sample_to_training_item({}, config)


def test_stable_task_id_is_deterministic() -> None:
    first = stable_task_id("open the drawer", 4096)
    second = stable_task_id("open the drawer", 4096)

    assert first == second
    assert first != stable_task_id("close the drawer", 4096)
