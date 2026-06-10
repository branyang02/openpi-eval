from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, SupportsIndex, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from world_model.config import DatasetConfig, FutureFrameStrategy, validate_future_frame_strategy


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


T_co = TypeVar("T_co", covariant=True)


class RandomAccessDataset(Protocol[T_co]):
    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class DataTransform(Protocol):
    def __call__(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError


class TransformedDataset(Dataset):
    """Small local equivalent of OpenPI's transformed LeRobot dataset wrapper."""

    def __init__(self, dataset: RandomAccessDataset[Mapping[str, Any]], transforms: tuple[DataTransform, ...]):
        self._dataset = dataset
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: SupportsIndex) -> Mapping[str, Any]:
        data = self._dataset[index]
        for transform in self._transforms:
            data = transform(data)
        return data


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask:
    """Extracts task text from LeRobot metadata using the sample task index."""

    tasks: Mapping[int, str]
    task_key: str = "task"

    def __call__(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        if "task_index" not in data:
            raise ValueError('Cannot extract task text without "task_index".')
        task_index = int(_as_numpy(data["task_index"]).reshape(-1)[0])
        if task_index not in self.tasks:
            raise ValueError(f"task_index={task_index} not found in LeRobot task metadata.")
        return {**data, self.task_key: self.tasks[task_index]}


def build_delta_timestamps(config: DatasetConfig, fps: int | float) -> dict[str, list[float]]:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    future_dts = [config.frame_delta * offset / fps for offset in range(config.num_future_frames + 1)]
    if config.idm_history_length < 0:
        raise ValueError(f"idm_history_length must be non-negative, got {config.idm_history_length}.")
    history_dts = [offset / fps for offset in range(-config.idm_history_length, 0)]
    action_dts = [*history_dts, *[offset / fps for offset in range(config.action_horizon)]]
    delta_timestamps = {key: future_dts for key in config.image_keys}
    delta_timestamps[config.state_key] = [*history_dts, 0.0]
    delta_timestamps[config.action_key] = action_dts
    return delta_timestamps


def image_to_chw_float(image: Any, image_size: int) -> torch.Tensor:
    array = _as_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"Expected RGB image with 3 dimensions, got shape {array.shape}.")

    if array.shape[0] == 3:
        chw = array
    elif array.shape[-1] == 3:
        chw = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"Expected RGB image in CHW or HWC layout, got shape {array.shape}.")

    if not chw.flags.writeable:
        chw = chw.copy()
    tensor = torch.as_tensor(chw, dtype=torch.float32)
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    tensor = tensor.clamp(0.0, 1.0)
    if tuple(tensor.shape[-2:]) != (image_size, image_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )[0]
    return tensor


def temporal_image_stack(value: Any, image_size: int) -> torch.Tensor:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Expected at least one image in temporal image sequence.")
        return torch.stack([image_to_chw_float(item, image_size) for item in value], dim=0)

    array = _as_numpy(value)
    if array.ndim == 3:
        return image_to_chw_float(array, image_size).unsqueeze(0)
    if array.ndim != 4:
        raise ValueError(f"Expected temporal image stack with 4 dimensions, got shape {array.shape}.")
    if array.shape[1] == 3:
        return torch.stack([image_to_chw_float(frame, image_size) for frame in array], dim=0)
    if array.shape[-1] == 3:
        return torch.stack([image_to_chw_float(frame, image_size) for frame in array], dim=0)
    raise ValueError(f"Expected temporal image stack in TCHW or THWC layout, got shape {array.shape}.")


def split_frame_pair(value: Any) -> tuple[Any, Any]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[0], value[1]

    array = _as_numpy(value)
    if array.ndim == 4 and array.shape[0] == 2:
        return array[0], array[1]
    raise ValueError(
        "Expected a two-frame tensor/list from LeRobot delta_timestamps. " f"Got value with shape {array.shape}."
    )


def first_vector(value: Any, *, name: str) -> torch.Tensor:
    array = _as_numpy(value).astype(np.float32)
    if array.ndim == 1:
        return torch.from_numpy(array)
    if array.ndim >= 2:
        flattened = array.reshape(-1, array.shape[-1])
        return torch.from_numpy(flattened[0])
    raise ValueError(f"Expected vector-like value for {name}, got shape {array.shape}.")


def temporal_vectors(value: Any, *, name: str, horizon: int) -> torch.Tensor:
    array = _as_numpy(value).astype(np.float32)
    if array.ndim == 1:
        raise ValueError(f"{name} must be a temporal action chunk, got single vector shape {array.shape}.")
    if array.ndim != 2:
        array = array.reshape(-1, array.shape[-1])
    if array.shape[0] != horizon:
        raise ValueError(f"{name} must have horizon {horizon}, got shape {array.shape}.")
    return torch.from_numpy(array)


def temporal_vector_sequence(value: Any, *, name: str, length: int) -> torch.Tensor:
    if length <= 0:
        raise ValueError(f"{name} temporal sequence length must be positive, got {length}.")
    array = _as_numpy(value).astype(np.float32)
    if array.ndim == 1:
        if length != 1:
            raise ValueError(f"{name} must have temporal length {length}, got single vector shape {array.shape}.")
        return torch.from_numpy(array.reshape(1, -1))
    if array.ndim != 2:
        array = array.reshape(-1, array.shape[-1])
    if array.shape[0] != length:
        raise ValueError(f"{name} must have temporal length {length}, got shape {array.shape}.")
    return torch.from_numpy(array)


def valid_mask(sample: Mapping[str, Any], key: str, length: int, *, skip_first: bool = False) -> torch.Tensor:
    pad_key = f"{key}_is_pad"
    if pad_key not in sample:
        return torch.ones(length, dtype=torch.float32)
    pad = torch.as_tensor(_as_numpy(sample[pad_key]), dtype=torch.bool).flatten()
    if skip_first:
        pad = pad[1:]
    if pad.numel() != length:
        raise ValueError(f"{pad_key} must have length {length}, got {pad.numel()}.")
    return (~pad).to(torch.float32)


def optional_scalar_long(sample: Mapping[str, Any], key: str) -> torch.Tensor | None:
    if key not in sample:
        return None
    tensor = torch.as_tensor(_as_numpy(sample[key])).reshape(-1)
    if tensor.numel() == 0:
        return None
    return tensor[0].to(dtype=torch.long)


def stable_task_id(task: Any, task_vocab_size: int) -> int:
    if task_vocab_size <= 0:
        raise ValueError("task_vocab_size must be positive.")
    if isinstance(task, (list, tuple)):
        if len(task) != 1:
            raise ValueError(f"Expected one task string, got {len(task)} entries.")
        task = task[0]
    task_text = str(task)
    digest = hashlib.sha256(task_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % task_vocab_size


WAN_FUTURE_SOURCES = {"wan2_2", "wan_lora"}
WAN_VAE_LATENT_CACHE_VERSION = 1
GENERATED_WAN_LATENT_CACHE_VERSION = 1
GENERATED_WAN_LATENT_CACHE_SCHEMA = "generated_wan_latents"


def expected_wan_source_frame_offsets(frame_delta: int, num_future_frames: int) -> list[int]:
    """Dataset source-frame offsets represented by Wan futures: delta, 2delta, ..."""

    if frame_delta <= 0:
        raise ValueError(f"frame_delta must be positive, got {frame_delta}.")
    if num_future_frames <= 0:
        raise ValueError(f"num_future_frames must be positive, got {num_future_frames}.")
    return [frame_delta * offset for offset in range(1, num_future_frames + 1)]


def expected_wan_selected_frame_indices(
    frame_delta: int,
    num_future_frames: int,
    *,
    strategy: FutureFrameStrategy = "first",
) -> list[int]:
    """Generated-video indices selected after skipping Wan's conditioning frame 0."""

    normalized_strategy = validate_future_frame_strategy(strategy)
    if normalized_strategy == "source_offsets":
        return expected_wan_source_frame_offsets(frame_delta, num_future_frames)
    if frame_delta <= 0:
        raise ValueError(f"frame_delta must be positive, got {frame_delta}.")
    if num_future_frames <= 0:
        raise ValueError(f"num_future_frames must be positive, got {num_future_frames}.")
    return list(range(1, num_future_frames + 1))


def _future_frame_strategy_from_metadata(
    metadata: Mapping[str, Any],
    *,
    context: str,
    default: FutureFrameStrategy = "first",
) -> FutureFrameStrategy:
    value = metadata.get("future_frame_strategy", default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{context} future_frame_strategy must be a string, got {type(value)}.")
    return validate_future_frame_strategy(value)


def validate_raw_wan_frame_delta(frame_delta: int, *, context: str) -> None:
    """Reject raw Wan2.2 use when the IDM expects subsampled dataset futures."""

    if frame_delta <= 0:
        raise ValueError(f"frame_delta must be positive, got {frame_delta}.")
    if frame_delta != 1:
        raise ValueError(
            f"Raw Wan2.2 only supports frame_delta=1 for {context}. It is not finetuned on the subsampled "
            "MetaWorld Wan export, so generated-video indices [1..K] would be native video steps "
            f"instead of dataset source offsets {expected_wan_source_frame_offsets(frame_delta, 2)[:2]}... "
            "Use future_source='wan_lora' with a LoRA finetuned for this frame_delta, or rerun with "
            "--frame-delta 1."
        )


def _normalize_selected_frame_indices(value: Any, *, context: str) -> list[int]:
    if value is None:
        raise ValueError(f"{context} is missing selected generated-video frame indices.")
    if not isinstance(value, list | tuple):
        raise ValueError(f"{context} selected generated-video frame indices must be a list, got {type(value)}.")
    try:
        return [int(index) for index in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} selected generated-video frame indices must be integers: {value!r}.") from error


def validate_wan_selected_frame_indices(
    selected_frame_indices: Any,
    *,
    frame_delta: int,
    num_future_frames: int,
    strategy: FutureFrameStrategy = "first",
    context: str,
) -> list[int]:
    """Return normalized Wan frame indices after checking the dataset temporal contract."""

    normalized_strategy = validate_future_frame_strategy(strategy)
    expected = expected_wan_selected_frame_indices(
        frame_delta,
        num_future_frames,
        strategy=normalized_strategy,
    )
    normalized = _normalize_selected_frame_indices(selected_frame_indices, context=context)
    if normalized != expected:
        raise ValueError(
            f"{context} selected generated-video frame indices {normalized} do not match the requested "
            f"generated-video frame contract {expected} for future_frame_strategy={normalized_strategy!r}, "
            f"frame_delta={frame_delta}, num_future_frames={num_future_frames}."
        )
    return normalized


def _validate_wan_source_temporal_metadata(
    metadata: Mapping[str, Any],
    *,
    frame_delta: int,
    num_future_frames: int,
    context: str,
) -> None:
    """Validate dataset cadence metadata stored separately from Wan video-frame indices."""

    expected_offsets = expected_wan_source_frame_offsets(frame_delta, num_future_frames)
    recorded_frame_delta = metadata.get("dataset_frame_delta", metadata.get("frame_delta"))
    recorded_num_future_frames = metadata.get("num_future_frames")
    recorded_offsets = metadata.get("source_frame_offsets")

    if recorded_frame_delta is None:
        raise ValueError(f"{context} is missing dataset_frame_delta metadata.")
    if int(recorded_frame_delta) != frame_delta:
        raise ValueError(
            f"{context} dataset_frame_delta={recorded_frame_delta} does not match requested "
            f"frame_delta={frame_delta}."
        )
    if recorded_num_future_frames is not None and int(recorded_num_future_frames) != num_future_frames:
        raise ValueError(
            f"{context} num_future_frames={recorded_num_future_frames} does not match requested "
            f"num_future_frames={num_future_frames}."
        )
    if recorded_offsets is None:
        raise ValueError(f"{context} is missing source_frame_offsets metadata.")
    try:
        normalized_offsets = [int(offset) for offset in recorded_offsets]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} source_frame_offsets must be integers: {recorded_offsets!r}.") from error
    if normalized_offsets != expected_offsets:
        raise ValueError(
            f"{context} source_frame_offsets {normalized_offsets} do not match expected dataset offsets "
            f"{expected_offsets} for frame_delta={frame_delta}, num_future_frames={num_future_frames}."
        )


def validate_cached_future_temporal_contract(
    *,
    cache_dir: Path,
    cache_config: Mapping[str, Any],
    rows: list[dict[str, Any]],
    frame_delta: int,
    num_future_frames: int,
) -> None:
    """Reject Wan caches whose selected generated-video frames do not match dataset futures."""

    source = cache_config.get("future_source")
    temporal_config = cache_config.get("future_frame_selection")
    has_manifest_temporal_metadata = any(row.get("selected_frame_indices") is not None for row in rows)
    has_config_temporal_metadata = temporal_config is not None
    if source not in WAN_FUTURE_SOURCES and not has_manifest_temporal_metadata and not has_config_temporal_metadata:
        return

    label = str(cache_dir)
    if source in WAN_FUTURE_SOURCES and not isinstance(temporal_config, dict) and not has_manifest_temporal_metadata:
        raise ValueError(
            f"Cache '{label}' is a Wan cache but neither config.json nor manifest.jsonl records "
            "selected generated-video frame indices. "
            "Regenerate the cache so selected generated-video frame indices are explicit."
        )
    config_strategy: FutureFrameStrategy = "first"
    if isinstance(temporal_config, dict):
        config_strategy = _future_frame_strategy_from_metadata(
            temporal_config,
            context=f"Cache '{label}' config future_frame_selection",
        )
        _validate_wan_source_temporal_metadata(
            temporal_config,
            frame_delta=frame_delta,
            num_future_frames=num_future_frames,
            context=f"Cache '{label}' config future_frame_selection",
        )
        config_indices = temporal_config.get("selected_frame_indices")
        if config_indices is not None:
            validate_wan_selected_frame_indices(
                config_indices,
                frame_delta=frame_delta,
                num_future_frames=num_future_frames,
                strategy=config_strategy,
                context=f"Cache '{label}' config future_frame_selection",
            )

    bad_rows: list[tuple[int, list[int], str, list[int]]] = []
    bad_metadata_rows: list[int] = []
    missing_rows: list[int] = []
    for row in rows:
        dataset_index = int(row["dataset_index"])
        if row.get("selected_frame_indices") is None:
            missing_rows.append(dataset_index)
            continue
        row_context = f"Cache '{label}' manifest row dataset_index={dataset_index}"
        row_strategy = _future_frame_strategy_from_metadata(
            row,
            context=row_context,
            default=config_strategy,
        )
        try:
            validate_wan_selected_frame_indices(
                row.get("selected_frame_indices"),
                frame_delta=frame_delta,
                num_future_frames=num_future_frames,
                strategy=row_strategy,
                context=row_context,
            )
        except ValueError:
            row_indices = _normalize_selected_frame_indices(
                row.get("selected_frame_indices"),
                context=row_context,
            )
            expected = expected_wan_selected_frame_indices(
                frame_delta,
                num_future_frames,
                strategy=row_strategy,
            )
            bad_rows.append((dataset_index, row_indices, row_strategy, expected))
        if not isinstance(temporal_config, dict) or row.get("source_frame_offsets") is not None:
            try:
                _validate_wan_source_temporal_metadata(
                    row,
                    frame_delta=frame_delta,
                    num_future_frames=num_future_frames,
                    context=f"Cache '{label}' manifest row dataset_index={dataset_index}",
                )
            except ValueError:
                bad_metadata_rows.append(dataset_index)

    if missing_rows:
        raise ValueError(
            f"Cache '{label}' manifest is missing selected generated-video frame indices for dataset_index values "
            f"{missing_rows}. Regenerate the cache; cannot infer temporal metadata."
        )
    if bad_rows:
        preview = bad_rows[:5]
        expected_contracts = {tuple(expected) for _, _, _, expected in bad_rows}
        expected_summary: list[int] | str
        if len(expected_contracts) == 1:
            expected_summary = list(next(iter(expected_contracts)))
        else:
            expected_summary = "mixed strategies; see mismatched rows"
        raise ValueError(
            f"Cache '{label}' manifest selected generated-video frame indices do not match the requested "
            f"generated-video frame contract {expected_summary} for frame_delta={frame_delta}, "
            f"num_future_frames={num_future_frames}. Mismatched rows: {preview}."
        )
    if bad_metadata_rows:
        raise ValueError(
            f"Cache '{label}' manifest source-frame temporal metadata does not match the requested "
            f"frame_delta={frame_delta}, num_future_frames={num_future_frames}. "
            f"Mismatched dataset_index values: {bad_metadata_rows[:5]}."
        )


@dataclasses.dataclass(frozen=True)
class BatchSpec:
    num_views: int
    num_future_frames: int
    state_dim: int
    action_dim: int
    action_horizon: int
    idm_history_length: int
    task_vocab_size: int


class MetaWorldFramePairDataset(Dataset):
    """Strict LeRobot-backed frame-pair dataset for MetaWorld WM/IDM training."""

    def __init__(self, config: DatasetConfig):
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        except ImportError as exc:
            raise ImportError("Install examples/world_model_env with `uv sync` before loading LeRobot data.") from exc

        self.config = config
        metadata = LeRobotDatasetMetadata(config.repo_id)
        self.metadata = metadata
        delta_timestamps = build_delta_timestamps(config, metadata.fps)
        episodes = list(config.episodes) if config.episodes is not None else None
        dataset = LeRobotDataset(config.repo_id, episodes=episodes, delta_timestamps=delta_timestamps)
        repair_lerobot_episode_data_index(dataset, episodes)
        balanced_indices = (
            balanced_lerobot_frame_indices(dataset, config) if config.samples_per_episode is not None else None
        )
        if config.prompt_from_task:
            dataset = TransformedDataset(dataset, (PromptFromLeRobotTask(metadata.tasks, task_key=config.task_key),))
        self.dataset = dataset
        self._indices = balanced_indices
        if self._indices is not None:
            self._length = len(self._indices)
        else:
            self._length = (
                len(self.dataset) if config.max_samples is None else min(config.max_samples, len(self.dataset))
            )

    def __len__(self) -> int:
        return self._length

    def raw_sample(self, index: int) -> Mapping[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        dataset_index = self._indices[index] if self._indices is not None else index
        sample = self.dataset[dataset_index]
        if not isinstance(sample, Mapping):
            return sample
        return {**sample, "dataset_index": sample.get("dataset_index", dataset_index)}

    def task_text(self, index: int) -> str:
        return str(extract_task_text(self.raw_sample(index), self.config.task_key))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.raw_sample(index)
        return sample_to_training_item(sample, self.config)


def repair_lerobot_episode_data_index(dataset: Any, episodes: list[int] | None) -> None:
    """Make LeRobot delta lookups work for filtered nonzero episode ids."""

    if episodes is None:
        return
    if not episodes:
        raise ValueError("episodes must not be empty.")
    if len(set(episodes)) != len(episodes):
        raise ValueError(f"episodes must be unique, got {episodes}.")
    if min(episodes) < 0:
        raise ValueError(f"episodes must be non-negative, got {episodes}.")
    if not hasattr(dataset, "hf_dataset") or not hasattr(dataset, "episode_data_index"):
        raise TypeError("Expected a LeRobotDataset-like object with hf_dataset and episode_data_index.")

    raw_episode_values = dataset.hf_dataset["episode_index"]
    actual_episodes = [int(_as_numpy(value).reshape(-1)[0]) for value in raw_episode_values]
    if not actual_episodes:
        raise ValueError(f"No frames found for episodes={episodes}.")
    expected = set(episodes)
    actual = set(actual_episodes)
    if actual != expected:
        raise ValueError(f"Requested episodes {sorted(expected)}, but dataset contains {sorted(actual)}.")

    original = dataset.episode_data_index
    starts = torch.zeros(max(expected) + 1, dtype=original["from"].dtype, device=original["from"].device)
    stops = torch.zeros_like(starts)
    seen: set[int] = set()
    cursor = 0
    while cursor < len(actual_episodes):
        episode = actual_episodes[cursor]
        if episode in seen:
            raise ValueError(f"Episode {episode} appears in multiple non-contiguous ranges.")
        start = cursor
        while cursor < len(actual_episodes) and actual_episodes[cursor] == episode:
            cursor += 1
        starts[episode] = start
        stops[episode] = cursor
        seen.add(episode)
    dataset.episode_data_index = {"from": starts, "to": stops}


def balanced_lerobot_frame_indices(dataset: Any, config: DatasetConfig) -> list[int]:
    """Return deterministic, evenly spaced non-terminal frame indices per episode."""

    samples_per_episode = config.samples_per_episode
    if samples_per_episode is None:
        raise ValueError("samples_per_episode must be set before balanced LeRobot sampling.")
    if samples_per_episode <= 0:
        raise ValueError(f"samples_per_episode must be positive, got {samples_per_episode}.")
    if not hasattr(dataset, "hf_dataset"):
        raise TypeError("Expected a LeRobotDataset-like object with hf_dataset.")
    required_future_offset = config.frame_delta * config.num_future_frames
    required_action_offset = max(config.action_horizon - 1, 0)
    required_offset = max(required_future_offset, required_action_offset)

    raw_episode_values = dataset.hf_dataset["episode_index"]
    groups: dict[int, list[int]] = {}
    order: list[int] = []
    current_episode: int | None = None
    closed_episodes: set[int] = set()
    for frame_index, value in enumerate(raw_episode_values):
        episode = int(_as_numpy(value).reshape(-1)[0])
        if episode != current_episode:
            if episode in closed_episodes:
                raise ValueError(f"Episode {episode} appears in multiple non-contiguous ranges.")
            if current_episode is not None:
                closed_episodes.add(current_episode)
            current_episode = episode
            order.append(episode)
            groups[episode] = []
        groups[episode].append(frame_index)

    if not order:
        raise ValueError("Cannot apply samples_per_episode to an empty LeRobot dataset.")

    selected: list[int] = []
    for episode in order:
        episode_indices = groups[episode]
        valid_episode_indices = episode_indices[: max(len(episode_indices) - required_offset, 0)]
        if len(valid_episode_indices) < samples_per_episode:
            raise ValueError(
                f"samples_per_episode={samples_per_episode} exceeds episode {episode} valid non-terminal "
                f"window count {len(valid_episode_indices)} (episode length {len(episode_indices)}, "
                f"required future/action offset {required_offset})."
            )
        positions = np.linspace(0, len(valid_episode_indices) - 1, num=samples_per_episode, dtype=np.int64)
        selected.extend(valid_episode_indices[int(position)] for position in positions)
    return selected


def extract_task_text(sample: Mapping[str, Any], task_key: str) -> str:
    if task_key not in sample:
        raise KeyError(f'Sample is missing task key "{task_key}".')
    task = sample[task_key]
    if isinstance(task, (list, tuple)):
        if len(task) != 1:
            raise ValueError(f"Expected one task string, got {len(task)} entries.")
        task = task[0]
    return str(task)


class SyntheticMetaWorldFramePairDataset(Dataset):
    """Deterministic synthetic dataset for tests and pipeline smoke runs."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.num_views = len(config.image_keys)
        self.state_dim = 4
        self.action_dim = 4
        self._length = config.synthetic_samples

    def __len__(self) -> int:
        return self._length

    def task_text(self, index: int) -> str:
        return f"synthetic metaworld task {index % min(self.config.task_vocab_size, 16)}"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.config.seed + index)
        size = self.config.image_size
        current = torch.rand((self.num_views, 3, size, size), generator=generator)
        state = torch.rand(self.state_dim, generator=generator) * 2.0 - 1.0
        base_action = torch.tanh(
            torch.stack([state[0], state[1], state.mean(), state.norm() / math.sqrt(self.state_dim)])
        )
        time = torch.linspace(0.0, 1.0, self.config.action_horizon)
        action_chunk = torch.stack(
            [
                base_action[0] * torch.cos(time),
                base_action[1] * torch.sin(time + 0.5),
                base_action[2].repeat(self.config.action_horizon),
                base_action[3].repeat(self.config.action_horizon),
            ],
            dim=-1,
        ).clamp(-1.0, 1.0)
        future_frames = []
        for offset in range(1, self.config.num_future_frames + 1):
            action = action_chunk[min(offset * self.config.frame_delta, self.config.action_horizon - 1)]
            shift = action[:2].mul(3).round().to(torch.int64)
            target = torch.roll(current, shifts=(int(shift[0]), int(shift[1])), dims=(-2, -1))
            target = (0.9 * target + 0.1 * torch.sigmoid(action.mean())).clamp(0.0, 1.0)
            future_frames.append(target)
        synthetic_task_count = min(self.config.task_vocab_size, 16)
        task_index = index % synthetic_task_count
        task_id = torch.tensor(task_index, dtype=torch.long)

        item = {
            "current_images": current,
            "future_images": torch.stack(future_frames, dim=0),
            "future_image_mask": torch.ones(self.config.num_future_frames, dtype=torch.float32),
            "state": state,
            "action_chunk": action_chunk,
            "action_mask": torch.ones(self.config.action_horizon, dtype=torch.float32),
            "task_id": task_id,
            "dataset_index": torch.tensor(index, dtype=torch.long),
            "episode_index": torch.tensor(index // synthetic_task_count, dtype=torch.long),
            "frame_index": torch.tensor(index % synthetic_task_count, dtype=torch.long),
            "task_index": torch.tensor(task_index, dtype=torch.long),
        }
        if self.config.idm_history_length > 0:
            history_time = torch.linspace(
                -1.0,
                -1.0 / self.config.idm_history_length,
                self.config.idm_history_length,
            )
            prev_states = torch.stack(
                [(state + 0.05 * step).clamp(-1.0, 1.0) for step in history_time],
                dim=0,
            )
            prev_actions = torch.stack(
                [
                    torch.stack(
                        [
                            base_action[0] * torch.cos(step),
                            base_action[1] * torch.sin(step + 0.5),
                            base_action[2],
                            base_action[3],
                        ]
                    )
                    for step in history_time
                ],
                dim=0,
            ).clamp(-1.0, 1.0)
            item.update(
                {
                    "prev_state_history": prev_states,
                    "prev_action_history": prev_actions,
                    "history_mask": torch.ones(self.config.idm_history_length, dtype=torch.float32),
                }
            )
        return item


class CachedFutureDataset(Dataset):
    """Dataset wrapper that replaces ground-truth future frames with cached generated futures."""

    def __init__(self, base_dataset: Dataset, cache_dir: str | Path):
        self.base_dataset = base_dataset
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cached future manifest not found: {manifest_path}")
        self.rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"Cached future manifest is empty: {manifest_path}")
        self._validate_config()

    def __len__(self) -> int:
        return len(self.rows)

    def task_text(self, index: int) -> str:
        row = self.rows[index]
        dataset_index = int(row["dataset_index"])
        if hasattr(self.base_dataset, "task_text"):
            return self.base_dataset.task_text(dataset_index)
        return f"metaworld task id {int(self.base_dataset[dataset_index]['task_id'])}"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        dataset_index = int(row["dataset_index"])
        item = dict(self.base_dataset[dataset_index])
        future_path = self._resolve_cache_path(row["future_tensor"])
        if not future_path.exists():
            raise FileNotFoundError(f"Cached future tensor not found: {future_path}")
        future_images = torch.load(future_path, map_location="cpu", weights_only=True)
        if not isinstance(future_images, torch.Tensor):
            raise TypeError(f"Cached future tensor must be a torch.Tensor, got {type(future_images)}.")
        if tuple(future_images.shape) != tuple(item["future_images"].shape):
            raise ValueError(
                "Cached future tensor shape does not match dataset future_images shape: "
                f"{tuple(future_images.shape)} != {tuple(item['future_images'].shape)}"
            )
        item["future_images"] = future_images.to(dtype=item["future_images"].dtype)
        return item

    def _resolve_cache_path(self, relative_path: str) -> Path:
        path = (self.cache_dir / relative_path).resolve()
        if not path.is_relative_to(self.cache_dir):
            raise ValueError(f"Cached future path escapes cache directory: {relative_path}")
        return path

    def _validate_config(self) -> None:
        config_path = self.cache_dir / "config.json"
        base_config = getattr(self.base_dataset, "config", None)
        if not config_path.exists():
            if base_config is None:
                return
            raise FileNotFoundError(f"Cached future config not found: {config_path}")
        full_cache_config = json.loads(config_path.read_text())
        if not isinstance(full_cache_config, dict):
            raise ValueError(f"Cached future config must be a JSON object: {config_path}")
        cache_config = full_cache_config.get("dataset_config", {})
        if base_config is None:
            frame_delta = cache_config.get("frame_delta")
            num_future_frames = cache_config.get("num_future_frames")
            if frame_delta is None or num_future_frames is None:
                has_manifest_temporal_metadata = any(row.get("selected_frame_indices") is not None for row in self.rows)
                if (
                    full_cache_config.get("future_source") in WAN_FUTURE_SOURCES
                    or full_cache_config.get("future_frame_selection") is not None
                    or has_manifest_temporal_metadata
                ):
                    raise ValueError(
                        f"Cached future config {config_path} is missing dataset_config.frame_delta or "
                        "dataset_config.num_future_frames, so Wan temporal metadata cannot be validated."
                    )
                return
            validate_cached_future_temporal_contract(
                cache_dir=self.cache_dir,
                cache_config=full_cache_config,
                rows=self.rows,
                frame_delta=int(frame_delta),
                num_future_frames=int(num_future_frames),
            )
            return
        expected = dataclasses.asdict(base_config)
        for key in (
            "source",
            "repo_id",
            "image_keys",
            "state_key",
            "action_key",
            "task_key",
            "frame_delta",
            "num_future_frames",
            "action_horizon",
            "image_size",
            "episodes",
            "samples_per_episode",
            "seed",
        ):
            cache_value = _json_normalized(cache_config.get(key))
            expected_value = _json_normalized(expected.get(key))
            if cache_value != expected_value:
                raise ValueError(
                    "Cached future dataset_config does not match base dataset config for "
                    f"{key}: {cache_config.get(key)!r} != {expected.get(key)!r}"
                )
        validate_cached_future_temporal_contract(
            cache_dir=self.cache_dir,
            cache_config=full_cache_config,
            rows=self.rows,
            frame_delta=int(base_config.frame_delta),
            num_future_frames=int(base_config.num_future_frames),
        )


class MixedFutureDataset(Dataset):
    """Dataset wrapper that exposes ground-truth samples followed by cached-future samples."""

    def __init__(self, base_dataset: Dataset, cache_dir: str | Path):
        self.base_dataset = base_dataset
        self.cached_dataset = CachedFutureDataset(base_dataset, cache_dir)

    def __len__(self) -> int:
        return len(self.base_dataset) + len(self.cached_dataset)

    def task_text(self, index: int) -> str:
        if index < len(self.base_dataset):
            if hasattr(self.base_dataset, "task_text"):
                return self.base_dataset.task_text(index)
            return f"metaworld task id {int(self.base_dataset[index]['task_id'])}"
        return self.cached_dataset.task_text(index - len(self.base_dataset))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < len(self.base_dataset):
            return self.base_dataset[index]
        return self.cached_dataset[index - len(self.base_dataset)]


def _metadata_mismatch_message(
    *,
    cache_dir: Path,
    field: str,
    cached: Any,
    requested: Any,
) -> str:
    return (
        f"Wan VAE latent cache metadata mismatch for {field} in {cache_dir}: "
        f"cached {field}={cached!r}, requested {field}={requested!r}."
    )


def _require_metadata_match(
    metadata: Mapping[str, Any],
    *,
    cache_dir: Path,
    field: str,
    requested: Any,
) -> None:
    cached = metadata.get(field)
    cached_normalized = _json_normalized(cached)
    requested_normalized = _json_normalized(requested)
    if cached_normalized != requested_normalized:
        raise ValueError(
            _metadata_mismatch_message(
                cache_dir=cache_dir,
                field=field,
                cached=cached,
                requested=requested,
            )
        )


def _cached_idm_history_length(metadata: Mapping[str, Any]) -> Any:
    cached = metadata.get("idm_history_length")
    if cached is not None:
        return cached
    dataset_metadata = metadata.get("dataset_config")
    if isinstance(dataset_metadata, Mapping):
        return dataset_metadata.get("idm_history_length")
    return None


def _require_wan_vae_idm_history_length_match(
    metadata: Mapping[str, Any],
    *,
    cache_dir: Path,
    requested: int,
) -> None:
    cached = _cached_idm_history_length(metadata)
    if cached is None:
        if requested == 0:
            return
        raise ValueError(
            _metadata_mismatch_message(
                cache_dir=cache_dir,
                field="idm_history_length",
                cached=cached,
                requested=requested,
            )
        )
    try:
        cached_int = int(cached)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Wan VAE latent cache metadata idm_history_length must be an integer in {cache_dir}: {cached!r}."
        ) from error
    if cached_int != requested:
        raise ValueError(
            _metadata_mismatch_message(
                cache_dir=cache_dir,
                field="idm_history_length",
                cached=cached,
                requested=requested,
            )
        )


def wan_vae_latent_cache_metadata(
    *,
    dataset_config: DatasetConfig,
    wan_vae_checkpoint_path: str | None,
    wan_vae_dtype: str,
    wan_vae_latent_channels: int,
    wan_vae_spatial_stride: int,
    num_samples: int,
) -> dict[str, Any]:
    dataset_dict = dataclasses.asdict(dataset_config)
    return {
        "version": WAN_VAE_LATENT_CACHE_VERSION,
        "repo_id": dataset_config.repo_id,
        "episodes": _json_normalized(dataset_config.episodes),
        "samples_per_episode": dataset_config.samples_per_episode,
        "max_samples": dataset_config.max_samples,
        "frame_delta": dataset_config.frame_delta,
        "num_future_frames": dataset_config.num_future_frames,
        "idm_history_length": dataset_config.idm_history_length,
        "image_size": dataset_config.image_size,
        "image_keys": list(dataset_config.image_keys),
        "wan_vae_checkpoint_path": wan_vae_checkpoint_path,
        "wan_vae_dtype": wan_vae_dtype,
        "wan_vae_latent_channels": wan_vae_latent_channels,
        "wan_vae_spatial_stride": wan_vae_spatial_stride,
        "num_samples": num_samples,
        "dataset_config": dataset_dict,
    }


def validate_wan_vae_latent_cache_metadata(
    metadata: Mapping[str, Any],
    *,
    cache_dir: Path,
    dataset_config: DatasetConfig,
    model_config: Any,
) -> None:
    for field, requested in (
        ("version", WAN_VAE_LATENT_CACHE_VERSION),
        ("repo_id", dataset_config.repo_id),
        ("episodes", dataset_config.episodes),
        ("samples_per_episode", dataset_config.samples_per_episode),
        ("max_samples", dataset_config.max_samples),
        ("frame_delta", dataset_config.frame_delta),
        ("num_future_frames", dataset_config.num_future_frames),
        ("image_size", dataset_config.image_size),
        ("image_keys", list(dataset_config.image_keys)),
        ("wan_vae_checkpoint_path", getattr(model_config, "wan_vae_checkpoint_path")),
        ("wan_vae_dtype", getattr(model_config, "wan_vae_dtype")),
        ("wan_vae_latent_channels", getattr(model_config, "wan_vae_latent_channels")),
        ("wan_vae_spatial_stride", getattr(model_config, "wan_vae_spatial_stride")),
    ):
        _require_metadata_match(metadata, cache_dir=cache_dir, field=field, requested=requested)
    _require_wan_vae_idm_history_length_match(
        metadata,
        cache_dir=cache_dir,
        requested=dataset_config.idm_history_length,
    )


class CachedWanVaeLatentDataset(Dataset):
    """Dataset wrapper that adds precomputed real current+future Wan VAE latents."""

    def __init__(self, base_dataset: Dataset, cache_dir: str | Path, *, model_config: Any):
        self.base_dataset = base_dataset
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        config_path = self.cache_dir / "config.json"
        manifest_path = self.cache_dir / "manifest.jsonl"
        if not config_path.exists():
            raise FileNotFoundError(f"Wan VAE latent cache config not found: {config_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Wan VAE latent cache manifest not found: {manifest_path}")
        self.metadata = json.loads(config_path.read_text())
        if not isinstance(self.metadata, dict):
            raise ValueError(f"Wan VAE latent cache config must be a JSON object: {config_path}")
        self.rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"Wan VAE latent cache manifest is empty: {manifest_path}")
        base_config = getattr(base_dataset, "config", None)
        if base_config is None:
            raise ValueError("CachedWanVaeLatentDataset requires a base dataset with a DatasetConfig.")
        validate_wan_vae_latent_cache_metadata(
            self.metadata,
            cache_dir=self.cache_dir,
            dataset_config=base_config,
            model_config=model_config,
        )
        self._validate_rows()

    def __len__(self) -> int:
        return len(self.rows)

    def task_text(self, index: int) -> str:
        dataset_index = int(self.rows[index]["dataset_index"])
        if hasattr(self.base_dataset, "task_text"):
            return self.base_dataset.task_text(dataset_index)
        return f"metaworld task id {int(self.base_dataset[dataset_index]['task_id'])}"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        dataset_index = int(row["dataset_index"])
        item = dict(self.base_dataset[dataset_index])
        latent_path = self._resolve_cache_path(row["latent_tensor"])
        if not latent_path.exists():
            raise FileNotFoundError(f"Cached Wan VAE latent tensor not found: {latent_path}")
        latents = torch.load(latent_path, map_location="cpu", weights_only=True)
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"Cached Wan VAE latents must be a torch.Tensor, got {type(latents)}.")
        expected = tuple(row["latent_shape"])
        if tuple(latents.shape) != expected:
            raise ValueError(
                f"Cached Wan VAE latent tensor shape does not match manifest for dataset_index={dataset_index}: "
                f"{tuple(latents.shape)} != {expected}."
            )
        item["wan_vae_latents"] = latents.to(dtype=torch.float32)
        return item

    def _resolve_cache_path(self, relative_path: str) -> Path:
        path = (self.cache_dir / relative_path).resolve()
        if not path.is_relative_to(self.cache_dir):
            raise ValueError(f"Cached Wan VAE latent path escapes cache directory: {relative_path}")
        return path

    def _validate_rows(self) -> None:
        expected_num_samples = int(self.metadata["num_samples"])
        if expected_num_samples != len(self.rows):
            raise ValueError(
                f"Wan VAE latent cache metadata num_samples={expected_num_samples} does not match "
                f"manifest row count {len(self.rows)} in {self.cache_dir}."
            )
        seen: set[int] = set()
        for row in self.rows:
            if "dataset_index" not in row or "latent_tensor" not in row or "latent_shape" not in row:
                raise ValueError(f"Wan VAE latent cache manifest row is missing required fields: {row}")
            dataset_index = int(row["dataset_index"])
            if dataset_index in seen:
                raise ValueError(f"Wan VAE latent cache has duplicate dataset_index={dataset_index}.")
            if dataset_index < 0 or dataset_index >= len(self.base_dataset):
                raise ValueError(
                    f"Wan VAE latent cache dataset_index={dataset_index} is outside base dataset length "
                    f"{len(self.base_dataset)}."
                )
            seen.add(dataset_index)


def _generated_wan_metadata_mismatch_message(
    *,
    cache_dir: Path,
    field: str,
    cached: Any,
    requested: Any,
) -> str:
    return (
        f"Generated Wan latent cache metadata mismatch for {field} in {cache_dir}: "
        f"cached {field}={cached!r}, requested {field}={requested!r}."
    )


def _require_generated_wan_metadata_match(
    metadata: Mapping[str, Any],
    *,
    cache_dir: Path,
    field: str,
    requested: Any,
) -> None:
    cached = metadata.get(field)
    cached_normalized = _json_normalized(cached)
    requested_normalized = _json_normalized(requested)
    if cached_normalized != requested_normalized:
        raise ValueError(
            _generated_wan_metadata_mismatch_message(
                cache_dir=cache_dir,
                field=field,
                cached=cached,
                requested=requested,
            )
        )


def _require_generated_wan_idm_history_length_match(
    metadata: Mapping[str, Any],
    *,
    cache_dir: Path,
    requested: int,
) -> None:
    cached = _cached_idm_history_length(metadata)
    if cached is None:
        if requested == 0:
            return
        raise ValueError(
            _generated_wan_metadata_mismatch_message(
                cache_dir=cache_dir,
                field="idm_history_length",
                cached=cached,
                requested=requested,
            )
        )
    try:
        cached_int = int(cached)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Generated Wan latent cache metadata idm_history_length must be an integer in {cache_dir}: {cached!r}."
        ) from error
    if cached_int != requested:
        raise ValueError(
            _generated_wan_metadata_mismatch_message(
                cache_dir=cache_dir,
                field="idm_history_length",
                cached=cached,
                requested=requested,
            )
        )


def _normalize_generated_wan_generator_metadata(generator_metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(generator_metadata, Mapping):
        raise ValueError(
            "Generated Wan latent cache generator metadata must be a mapping, " f"got {type(generator_metadata)}."
        )
    normalized = _json_normalized(dict(generator_metadata))
    if not isinstance(normalized, dict):
        raise ValueError("Generated Wan latent cache generator metadata must normalize to a JSON object.")
    return normalized


def generated_wan_latent_cache_metadata(
    *,
    dataset_config: DatasetConfig,
    wan_vae_latent_channels: int,
    wan_vae_spatial_stride: int,
    generator_metadata: Mapping[str, Any],
    num_samples: int,
) -> dict[str, Any]:
    dataset_dict = dataclasses.asdict(dataset_config)
    return {
        "cache_schema": GENERATED_WAN_LATENT_CACHE_SCHEMA,
        "version": GENERATED_WAN_LATENT_CACHE_VERSION,
        "repo_id": dataset_config.repo_id,
        "episodes": _json_normalized(dataset_config.episodes),
        "samples_per_episode": dataset_config.samples_per_episode,
        "max_samples": dataset_config.max_samples,
        "frame_delta": dataset_config.frame_delta,
        "num_future_frames": dataset_config.num_future_frames,
        "idm_history_length": dataset_config.idm_history_length,
        "image_size": dataset_config.image_size,
        "image_keys": list(dataset_config.image_keys),
        "wan_vae_latent_channels": wan_vae_latent_channels,
        "wan_vae_spatial_stride": wan_vae_spatial_stride,
        "generator": _normalize_generated_wan_generator_metadata(generator_metadata),
        "num_samples": num_samples,
        "dataset_config": dataset_dict,
    }


def validate_generated_wan_latent_cache_metadata(
    metadata: Mapping[str, Any],
    *,
    cache_dir: Path,
    dataset_config: DatasetConfig,
    model_config: Any,
    generator_metadata: Mapping[str, Any],
) -> None:
    if getattr(model_config, "idm_arch", None) != "flow_transformer":
        raise ValueError("GeneratedWanLatentDataset requires model_config.idm_arch='flow_transformer'.")
    if getattr(model_config, "idm_visual_encoder", None) != "wan_vae":
        raise ValueError("GeneratedWanLatentDataset requires model_config.idm_visual_encoder='wan_vae'.")
    if not isinstance(metadata.get("generator"), Mapping):
        raise ValueError(f"Generated Wan latent cache metadata generator must be a JSON object in {cache_dir}.")

    for field, requested in (
        ("cache_schema", GENERATED_WAN_LATENT_CACHE_SCHEMA),
        ("version", GENERATED_WAN_LATENT_CACHE_VERSION),
        ("repo_id", dataset_config.repo_id),
        ("episodes", dataset_config.episodes),
        ("samples_per_episode", dataset_config.samples_per_episode),
        ("max_samples", dataset_config.max_samples),
        ("frame_delta", dataset_config.frame_delta),
        ("num_future_frames", dataset_config.num_future_frames),
        ("image_size", dataset_config.image_size),
        ("image_keys", list(dataset_config.image_keys)),
        ("wan_vae_latent_channels", getattr(model_config, "wan_vae_latent_channels")),
        ("wan_vae_spatial_stride", getattr(model_config, "wan_vae_spatial_stride")),
        ("generator", _normalize_generated_wan_generator_metadata(generator_metadata)),
    ):
        _require_generated_wan_metadata_match(metadata, cache_dir=cache_dir, field=field, requested=requested)
    _require_generated_wan_idm_history_length_match(
        metadata,
        cache_dir=cache_dir,
        requested=dataset_config.idm_history_length,
    )


def _normalize_generated_wan_latent_shape(value: Any, *, context: str) -> tuple[int, int, int, int]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{context} latent_shape must be a list with rank 4 (C,T,H,W), got {type(value)}.")
    if len(value) != 4:
        raise ValueError(f"{context} latent_shape must have rank 4 (C,T,H,W), got {value!r}.")
    try:
        shape = tuple(int(dim) for dim in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} latent_shape entries must be integers: {value!r}.") from error
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"{context} latent_shape entries must be positive, got {shape}.")
    return shape


class GeneratedWanLatentDataset(Dataset):
    """Dataset wrapper that adds generated Wan latents while preserving real dataset futures."""

    def __init__(
        self,
        base_dataset: Dataset,
        cache_dir: str | Path,
        model_config: Any,
        *,
        generator_metadata: Mapping[str, Any],
    ):
        self.base_dataset = base_dataset
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.model_config = model_config
        config_path = self.cache_dir / "config.json"
        manifest_path = self.cache_dir / "manifest.jsonl"
        if not config_path.exists():
            raise FileNotFoundError(f"Generated Wan latent cache config not found: {config_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Generated Wan latent cache manifest not found: {manifest_path}")
        self.metadata = json.loads(config_path.read_text())
        if not isinstance(self.metadata, dict):
            raise ValueError(f"Generated Wan latent cache config must be a JSON object: {config_path}")
        self.rows = self._read_manifest(manifest_path)
        if not self.rows:
            raise ValueError(f"Generated Wan latent cache manifest is empty: {manifest_path}")

        base_config = getattr(base_dataset, "config", None)
        if base_config is None:
            raise ValueError("GeneratedWanLatentDataset requires a base dataset with a DatasetConfig.")
        validate_generated_wan_latent_cache_metadata(
            self.metadata,
            cache_dir=self.cache_dir,
            dataset_config=base_config,
            model_config=model_config,
            generator_metadata=generator_metadata,
        )
        self._expected_latent_shape = self._expected_model_latent_shape(base_config)
        self._row_latent_shapes: list[tuple[int, int, int, int]] = []
        self._validate_rows()

    def __len__(self) -> int:
        return len(self.rows)

    def task_text(self, index: int) -> str:
        dataset_index = int(self.rows[index]["dataset_index"])
        if hasattr(self.base_dataset, "task_text"):
            return self.base_dataset.task_text(dataset_index)
        return f"metaworld task id {int(self.base_dataset[dataset_index]['task_id'])}"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        dataset_index = int(row["dataset_index"])
        item = dict(self.base_dataset[dataset_index])
        latent_path = self._resolve_cache_path(row["latent_tensor"])
        if not latent_path.exists():
            raise FileNotFoundError(f"Generated Wan latent tensor not found: {latent_path}")
        latents = torch.load(latent_path, map_location="cpu", weights_only=True)
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"Generated Wan latents must be a torch.Tensor, got {type(latents)}.")
        if latents.ndim != 4:
            raise ValueError(
                "Generated Wan latent tensor must have rank 4 (C,T,H,W) for "
                f"dataset_index={dataset_index}, got shape {tuple(latents.shape)}."
            )
        expected = self._row_latent_shapes[index]
        if tuple(latents.shape) != expected:
            raise ValueError(
                "Generated Wan latent tensor shape does not match manifest for "
                f"dataset_index={dataset_index}: {tuple(latents.shape)} != {expected}."
            )
        item["wan_vae_latents"] = latents.to(dtype=torch.float32)
        return item

    def _read_manifest(self, manifest_path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Generated Wan latent cache manifest row {line_number} must be a JSON object: {row!r}"
                )
            rows.append(row)
        return rows

    def _resolve_cache_path(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Generated Wan latent path must be a non-empty string, got {relative_path!r}.")
        path = (self.cache_dir / relative_path).resolve()
        if not path.is_relative_to(self.cache_dir):
            raise ValueError(f"Generated Wan latent path escapes cache directory: {relative_path}")
        return path

    def _expected_model_latent_shape(self, dataset_config: DatasetConfig) -> tuple[int, int, int, int]:
        stride = int(getattr(self.model_config, "wan_vae_spatial_stride"))
        image_size = int(dataset_config.image_size)
        if stride <= 0:
            raise ValueError(f"wan_vae_spatial_stride must be positive, got {stride}.")
        if image_size % stride != 0:
            raise ValueError(f"image_size={image_size} must be divisible by wan_vae_spatial_stride={stride}.")
        total_video_frames = 1 + int(dataset_config.num_future_frames)
        latent_frames = (total_video_frames + 3) // 4
        latent_side = image_size // stride
        return (
            int(getattr(self.model_config, "wan_vae_latent_channels")),
            latent_frames,
            latent_side,
            latent_side,
        )

    def _validate_rows(self) -> None:
        if "num_samples" not in self.metadata:
            raise ValueError(f"Generated Wan latent cache metadata is missing num_samples in {self.cache_dir}.")
        expected_num_samples = int(self.metadata["num_samples"])
        if expected_num_samples != len(self.rows):
            raise ValueError(
                f"Generated Wan latent cache metadata num_samples={expected_num_samples} does not match "
                f"manifest row count {len(self.rows)} in {self.cache_dir}."
            )

        expected_generator_metadata = _normalize_generated_wan_generator_metadata(self.metadata["generator"])
        seen: set[int] = set()
        for row_number, row in enumerate(self.rows, start=1):
            if "dataset_index" not in row or "latent_tensor" not in row or "latent_shape" not in row:
                raise ValueError(f"Generated Wan latent cache manifest row is missing required fields: {row}")
            try:
                dataset_index = int(row["dataset_index"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Generated Wan latent cache manifest row {row_number} has invalid dataset_index: {row!r}"
                ) from error
            if dataset_index in seen:
                raise ValueError(f"Generated Wan latent cache has duplicate dataset_index={dataset_index}.")
            if dataset_index < 0 or dataset_index >= len(self.base_dataset):
                raise ValueError(
                    f"Generated Wan latent cache dataset_index={dataset_index} is outside base dataset length "
                    f"{len(self.base_dataset)}."
                )
            if "generator_metadata" not in row:
                raise ValueError(
                    "Generated Wan latent cache manifest row "
                    f"{row_number} dataset_index={dataset_index} is missing generator_metadata."
                )
            if not isinstance(row["generator_metadata"], Mapping):
                raise ValueError(
                    "Generated Wan latent cache manifest row "
                    f"{row_number} dataset_index={dataset_index} generator_metadata must be a mapping, "
                    f"got {type(row['generator_metadata'])}."
                )
            row_generator_metadata = _normalize_generated_wan_generator_metadata(row["generator_metadata"])
            for key, expected_value in expected_generator_metadata.items():
                row_value = row_generator_metadata.get(key)
                if row_value != expected_value:
                    raise ValueError(
                        "Generated Wan latent cache manifest row generator_metadata mismatch for "
                        f"row {row_number} dataset_index={dataset_index} key={key!r}: "
                        f"row value {row_value!r} != config value {expected_value!r}."
                    )
            self._resolve_cache_path(row["latent_tensor"])
            manifest_shape = _normalize_generated_wan_latent_shape(
                row["latent_shape"],
                context=f"Generated Wan latent cache manifest row {row_number}",
            )
            if manifest_shape != self._expected_latent_shape:
                raise ValueError(
                    "Generated Wan latent cache manifest latent_shape does not match expected IDM latent shape "
                    f"for dataset_index={dataset_index}: {manifest_shape} != {self._expected_latent_shape}."
                )
            self._row_latent_shapes.append(manifest_shape)
            seen.add(dataset_index)


def _json_normalized(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_normalized(item) for item in value]
    if isinstance(value, list):
        return [_json_normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_normalized(item) for key, item in value.items()}
    return value


def sample_to_training_item(sample: Mapping[str, Any], config: DatasetConfig) -> dict[str, torch.Tensor]:
    missing = [
        key for key in [*config.image_keys, config.state_key, config.action_key, config.task_key] if key not in sample
    ]
    if missing:
        raise KeyError(f"Sample is missing required key(s): {missing}")

    current_images = []
    per_view_future_images = []
    per_view_future_masks = []
    for key in config.image_keys:
        frames = temporal_image_stack(sample[key], config.image_size)
        expected_frames = config.num_future_frames + 1
        if frames.shape[0] != expected_frames:
            raise ValueError(f"{key} must have {expected_frames} temporal frames, got shape {tuple(frames.shape)}.")
        current_images.append(frames[0])
        per_view_future_images.append(frames[1:])
        per_view_future_masks.append(valid_mask(sample, key, config.num_future_frames, skip_first=True))

    if config.idm_history_length > 0:
        state_sequence = temporal_vector_sequence(
            sample[config.state_key],
            name=config.state_key,
            length=config.idm_history_length + 1,
        )
        action_sequence = temporal_vector_sequence(
            sample[config.action_key],
            name=config.action_key,
            length=config.idm_history_length + config.action_horizon,
        )
        state = state_sequence[-1]
        prev_state_history = state_sequence[: config.idm_history_length]
        prev_action_history = action_sequence[: config.idm_history_length]
        action_chunk = action_sequence[config.idm_history_length :]
        state_history_mask = valid_mask(sample, config.state_key, config.idm_history_length + 1)[
            : config.idm_history_length
        ]
        action_valid = valid_mask(sample, config.action_key, config.idm_history_length + config.action_horizon)
        history_mask = state_history_mask * action_valid[: config.idm_history_length]
        action_mask = action_valid[config.idm_history_length :]
    else:
        state = first_vector(sample[config.state_key], name=config.state_key)
        action_chunk = temporal_vectors(
            sample[config.action_key], name=config.action_key, horizon=config.action_horizon
        )
        prev_state_history = None
        prev_action_history = None
        history_mask = None
        action_mask = valid_mask(sample, config.action_key, config.action_horizon)
    future_mask = torch.stack(per_view_future_masks, dim=0).amin(dim=0)

    item = {
        "current_images": torch.stack(current_images, dim=0),
        "future_images": torch.stack(per_view_future_images, dim=1),
        "future_image_mask": future_mask,
        "state": state,
        "action_chunk": action_chunk,
        "action_mask": action_mask,
        "task_id": torch.tensor(
            stable_task_id(extract_task_text(sample, config.task_key), config.task_vocab_size),
            dtype=torch.long,
        ),
    }
    for metadata_key in ("dataset_index", "episode_index", "frame_index", "task_index"):
        metadata_value = optional_scalar_long(sample, metadata_key)
        if metadata_value is not None:
            item[metadata_key] = metadata_value
    if config.idm_history_length > 0:
        if prev_state_history is None or prev_action_history is None or history_mask is None:
            raise RuntimeError("IDM history tensors were not built despite idm_history_length > 0.")
        item.update(
            {
                "prev_state_history": prev_state_history,
                "prev_action_history": prev_action_history,
                "history_mask": history_mask,
            }
        )
    return item


def create_dataset(config: DatasetConfig) -> Dataset:
    if config.source == "synthetic":
        return SyntheticMetaWorldFramePairDataset(config)
    if config.source == "lerobot":
        return MetaWorldFramePairDataset(config)
    raise ValueError(f"Unknown dataset source: {config.source}")


def infer_batch_spec(dataset: Dataset, task_vocab_size: int) -> BatchSpec:
    sample = dataset[0]
    history_length = int(sample["prev_state_history"].shape[0]) if "prev_state_history" in sample else 0
    return BatchSpec(
        num_views=int(sample["current_images"].shape[0]),
        num_future_frames=int(sample["future_images"].shape[0]),
        state_dim=int(sample["state"].shape[-1]),
        action_dim=int(sample["action_chunk"].shape[-1]),
        action_horizon=int(sample["action_chunk"].shape[0]),
        idm_history_length=history_length,
        task_vocab_size=task_vocab_size,
    )
