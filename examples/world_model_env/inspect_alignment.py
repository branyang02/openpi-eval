from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from world_model.config import DatasetConfig, DatasetSource
from world_model.data import build_delta_timestamps, create_dataset, sample_to_training_item


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_source: DatasetSource = "lerobot"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_keys: tuple[str, ...] = ("corner.image", "corner4.image", "gripperPOV.image")
    state_key: str = "observation.state"
    action_key: str = "actions"
    task_key: str = "task"
    prompt_from_task: bool = True
    image_size: int = 64
    frame_delta: int = 4
    num_future_frames: int = 1
    action_horizon: int = 32
    sample_index: int = 0
    max_samples: int | None = None
    samples_per_episode: int | None = None
    episodes: tuple[int, ...] | None = None
    synthetic_samples: int = 8
    seed: int = 7
    output_json: str | None = None


def _shape(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        return [len(value), *_shape(value[0])]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(dim) for dim in shape]
    return list(np.asarray(value).shape)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return [_json_scalar(item) for item in value]
        return _json_scalar(value[0])
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value.detach().cpu().tolist()
        return value.detach().cpu().reshape(-1)[0].item()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    return value


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _require_shape(sample: Mapping[str, Any], key: str, expected: tuple[int, ...]) -> torch.Tensor:
    if key not in sample:
        raise KeyError(f'Sample is missing required key "{key}".')
    tensor = torch.as_tensor(sample[key])
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{key} must have shape {expected}, got shape {tuple(tensor.shape)}.")
    return tensor


def _require_vector(sample: Mapping[str, Any], key: str, expected_length: int) -> torch.Tensor:
    if key not in sample:
        raise KeyError(f'Sample is missing required key "{key}".')
    tensor = torch.as_tensor(sample[key])
    if tensor.ndim != 1 or tensor.shape[0] != expected_length:
        raise ValueError(f"{key} must have shape ({expected_length},), got shape {tuple(tensor.shape)}.")
    return tensor


def _mask_summary(mask: torch.Tensor, *, name: str) -> dict[str, Any]:
    numeric = mask.to(torch.float32)
    if not torch.isfinite(numeric).all():
        raise ValueError(f"{name} must contain only finite values.")
    if not torch.logical_or(numeric == 0.0, numeric == 1.0).all():
        raise ValueError(f"{name} must be a binary 0/1 mask, got values {numeric.detach().cpu().tolist()}.")
    valid_count = int(numeric.sum().item())
    length = int(numeric.numel())
    invalid_indices = (numeric == 0.0).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    return {
        "present": True,
        "length": length,
        "valid_count": valid_count,
        "invalid_count": length - valid_count,
        "valid_fraction": valid_count / length if length else None,
        "invalid_indices": [int(index) for index in invalid_indices],
    }


def _validate_training_item(item: Mapping[str, Any], config: DatasetConfig) -> dict[str, Any]:
    num_views = len(config.image_keys)
    image_shape = (3, config.image_size, config.image_size)
    current = _require_shape(item, "current_images", (num_views, *image_shape))
    future = _require_shape(item, "future_images", (config.num_future_frames, num_views, *image_shape))

    if "state" not in item:
        raise KeyError('Sample is missing required key "state".')
    state = torch.as_tensor(item["state"])
    if state.ndim != 1:
        raise ValueError(f"state must be a rank-1 vector, got shape {tuple(state.shape)}.")

    if "action_chunk" not in item:
        raise KeyError('Sample is missing required key "action_chunk".')
    action_chunk = torch.as_tensor(item["action_chunk"])
    if action_chunk.ndim != 2 or action_chunk.shape[0] != config.action_horizon:
        raise ValueError(
            "action_chunk must have shape "
            f"({config.action_horizon}, action_dim), got shape {tuple(action_chunk.shape)}."
        )

    action_mask = _require_vector(item, "action_mask", config.action_horizon)
    future_image_mask = None
    if "future_image_mask" in item:
        future_image_mask = _require_vector(item, "future_image_mask", config.num_future_frames)

    if "task_id" in item:
        task_id = torch.as_tensor(item["task_id"])
        if task_id.ndim != 0:
            raise ValueError(f"task_id must be scalar, got shape {tuple(task_id.shape)}.")

    summary = {
        "shapes": {
            "current_images": list(current.shape),
            "future_images": list(future.shape),
            "state": list(state.shape),
            "action_chunk": list(action_chunk.shape),
            "action_mask": list(action_mask.shape),
        },
        "action_mask": _mask_summary(action_mask, name="action_mask"),
        "future_image_mask": {"present": False},
    }
    if future_image_mask is not None:
        summary["shapes"]["future_image_mask"] = list(future_image_mask.shape)
        summary["future_image_mask"] = _mask_summary(future_image_mask, name="future_image_mask")
    return summary


def _raw_sample(dataset: Any, sample_index: int) -> Mapping[str, Any] | None:
    raw_sample = getattr(dataset, "raw_sample", None)
    if not callable(raw_sample):
        return None
    return _require_mapping(raw_sample(sample_index), name="raw_sample")


def _dataset_fps(dataset: Any) -> float | None:
    metadata = getattr(dataset, "metadata", None)
    fps = getattr(metadata, "fps", None)
    if fps is None:
        return None
    return float(fps)


def _sample_metadata(
    *,
    dataset: Any,
    raw_sample: Mapping[str, Any] | None,
    sample_index: int,
    config: DatasetConfig,
) -> dict[str, Any]:
    task = None
    if raw_sample is not None and config.task_key in raw_sample:
        task = str(_json_scalar(raw_sample[config.task_key]))
    elif callable(getattr(dataset, "task_text", None)):
        task = str(dataset.task_text(sample_index))

    metadata: dict[str, Any] = {
        "selected_sample_index": sample_index,
        "task": task,
        "episode_index": None,
        "frame_index": None,
    }
    if raw_sample is not None:
        if "episode_index" in raw_sample:
            metadata["episode_index"] = _json_scalar(raw_sample["episode_index"])
        if "frame_index" in raw_sample:
            metadata["frame_index"] = _json_scalar(raw_sample["frame_index"])
    return metadata


def _temporal_contract(config: DatasetConfig, fps: float | None) -> dict[str, Any]:
    future_offsets = [config.frame_delta * offset for offset in range(1, config.num_future_frames + 1)]
    all_frame_offsets = [0, *future_offsets]
    action_offsets = list(range(config.action_horizon))
    delta_timestamps = build_delta_timestamps(config, fps) if fps is not None else None

    frame_contract: dict[str, Any] = {
        "image_keys": list(config.image_keys),
        "frame_delta": config.frame_delta,
        "num_future_frames": config.num_future_frames,
        "current_source_frame_offset": 0,
        "future_source_frame_offsets": future_offsets,
        "all_source_frame_offsets": all_frame_offsets,
        "delta_timestamps_seconds_by_image_key": None,
    }
    action_contract: dict[str, Any] = {
        "action_key": config.action_key,
        "action_horizon": config.action_horizon,
        "source_action_offsets": action_offsets,
        "delta_timestamps_seconds": None,
    }
    if delta_timestamps is not None:
        frame_contract["delta_timestamps_seconds_by_image_key"] = {
            key: list(delta_timestamps[key]) for key in config.image_keys
        }
        action_contract["delta_timestamps_seconds"] = list(delta_timestamps[config.action_key])
    return {
        "current_future_frame_offset_contract": frame_contract,
        "action_horizon_contract": action_contract,
    }


def inspect_alignment(dataset: Any, config: DatasetConfig, sample_index: int) -> dict[str, Any]:
    dataset_length = len(dataset)
    if dataset_length <= 0:
        raise ValueError("Cannot inspect an empty dataset.")
    if not 0 <= sample_index < dataset_length:
        raise IndexError(f"sample_index={sample_index} is outside dataset length {dataset_length}.")

    raw = _raw_sample(dataset, sample_index)
    if raw is not None:
        converted = sample_to_training_item(raw, config)
        _validate_training_item(converted, config)

    item = _require_mapping(dataset[sample_index], name="dataset sample")
    item_summary = _validate_training_item(item, config)
    fps = _dataset_fps(dataset)
    contracts = _temporal_contract(config, fps)

    return {
        "dataset": {
            "source": config.source,
            "repo_id": config.repo_id,
            "length": dataset_length,
            "fps": fps,
        },
        "selected_sample_index": sample_index,
        "sample": _sample_metadata(dataset=dataset, raw_sample=raw, sample_index=sample_index, config=config),
        "action_mask": item_summary["action_mask"],
        "future_image_mask": item_summary["future_image_mask"],
        **contracts,
        "shapes": item_summary["shapes"],
        "raw_shapes": {
            key: _shape(raw[key])
            for key in [*config.image_keys, config.state_key, config.action_key]
            if raw is not None and key in raw
        },
    }


def main(args: Args) -> None:
    config = DatasetConfig(
        source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=args.image_keys,
        state_key=args.state_key,
        action_key=args.action_key,
        task_key=args.task_key,
        prompt_from_task=args.prompt_from_task,
        image_size=args.image_size,
        frame_delta=args.frame_delta,
        num_future_frames=args.num_future_frames,
        action_horizon=args.action_horizon,
        max_samples=args.max_samples,
        samples_per_episode=args.samples_per_episode,
        episodes=args.episodes,
        synthetic_samples=args.synthetic_samples,
        seed=args.seed,
    )
    dataset = create_dataset(config)
    summary = inspect_alignment(dataset, config, args.sample_index)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json is None:
        print(payload, end="")
        return

    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload)


if __name__ == "__main__":
    main(tyro.cli(Args))
