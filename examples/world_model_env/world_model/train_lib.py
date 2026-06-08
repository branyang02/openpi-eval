from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import random
import warnings
from collections import defaultdict
from collections.abc import Hashable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from tqdm import tqdm

from world_model.config import (
    IdmFutureRankingScoreMode,
    IdmTargetSource,
    ModelConfig,
    TrainConfig,
    WanVaeLatentNoiseTimeMode,
)
from world_model.data import (
    CachedFutureDataset,
    CachedWanVaeLatentDataset,
    MixedFutureDataset,
    create_dataset,
    infer_batch_spec,
)
from world_model.models import ConvVideoWorldModel, InverseDynamicsModel

IdmPredictionMode = Literal["sample", "context_action"]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_eval_size(length: int, eval_fraction: float) -> int:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1.")
    if length <= 1:
        raise ValueError("Dataset is too small to create non-empty train/eval splits.")
    eval_size = max(1, int(round(length * eval_fraction)))
    if length - eval_size <= 0:
        raise ValueError("Dataset is too small to create a non-empty train split.")
    return eval_size


def _scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().reshape(-1)[0])
    return int(np.asarray(value).reshape(-1)[0])


def _sample_episode_id(sample: Any) -> int | None:
    if not isinstance(sample, Mapping):
        return None
    for key in ("episode_index", "episode_id", "episode"):
        if key in sample:
            return _scalar_int(sample[key])
    return None


def _hf_episode_ids(dataset: Dataset) -> list[int] | None:
    candidate = getattr(dataset, "dataset", dataset)
    candidate = getattr(candidate, "_dataset", candidate)
    hf_dataset = getattr(candidate, "hf_dataset", None)
    if hf_dataset is None:
        return None
    try:
        raw_episode_ids = hf_dataset["episode_index"]
    except (KeyError, TypeError):
        return None
    return [_scalar_int(value) for value in raw_episode_ids[: len(dataset)]]


def _dataset_episode_ids(dataset: Dataset) -> list[int] | None:
    episode_ids = _hf_episode_ids(dataset)
    if episode_ids is not None:
        return episode_ids if len(set(episode_ids)) > 1 else None

    raw_sample = getattr(dataset, "raw_sample", None)
    if raw_sample is None:
        return None

    episode_ids: list[int] = []
    for index in range(len(dataset)):
        episode_id = _sample_episode_id(raw_sample(index))
        if episode_id is None:
            return None
        episode_ids.append(episode_id)
    return episode_ids if len(set(episode_ids)) > 1 else None


def _episode_split_indices(episode_ids: list[int], eval_size: int) -> tuple[list[int], list[int]]:
    episode_order = sorted(set(episode_ids), key=episode_ids.index)
    eval_episodes: set[int] = set()
    eval_count = 0
    for episode_id in reversed(episode_order):
        if len(eval_episodes) >= len(episode_order) - 1:
            break
        eval_episodes.add(episode_id)
        eval_count += episode_ids.count(episode_id)
        if eval_count >= eval_size:
            break

    train_indices = [index for index, episode_id in enumerate(episode_ids) if episode_id not in eval_episodes]
    eval_indices = [index for index, episode_id in enumerate(episode_ids) if episode_id in eval_episodes]
    if not train_indices or not eval_indices:
        raise ValueError("Episode-aware split could not create non-empty train/eval splits.")
    return train_indices, eval_indices


def _contiguous_split_indices(length: int, eval_size: int, split_gap: int) -> tuple[list[int], list[int]]:
    if split_gap < 0:
        raise ValueError("split_gap must be non-negative.")
    eval_start = length - eval_size
    train_end = eval_start - split_gap
    if train_end <= 0:
        raise ValueError(
            "Dataset is too small to create non-empty train/eval splits with "
            f"split_gap={split_gap} and eval_size={eval_size}."
        )
    return list(range(train_end)), list(range(eval_start, length))


def split_dataset(dataset: Dataset, eval_fraction: float, seed: int, split_gap: int = 1) -> tuple[Dataset, Dataset]:
    """Split sequential frame-pair datasets without random adjacent leakage.

    If episode ids are exposed via ``raw_sample(...)[\"episode_index\"]`` (or
    similar), hold out whole tail episodes. Otherwise, hold out the tail block
    and drop ``split_gap`` samples between train and eval.
    """

    del seed  # Kept for API compatibility with the old random split.
    eval_size = _split_eval_size(len(dataset), eval_fraction)
    episode_ids = _dataset_episode_ids(dataset)
    if episode_ids is not None:
        train_indices, eval_indices = _episode_split_indices(episode_ids, eval_size)
    else:
        train_indices, eval_indices = _contiguous_split_indices(len(dataset), eval_size, split_gap)
    return Subset(dataset, train_indices), Subset(dataset, eval_indices)


def _task_identity_from_sample(sample: Any) -> Hashable | None:
    if not isinstance(sample, Mapping):
        return None
    for key in ("task_index", "task_id"):
        if key in sample:
            return _scalar_int(sample[key])
    return None


def _sample_task_identity(dataset: Dataset, index: int) -> Hashable:
    if isinstance(dataset, Subset):
        subset_index = dataset.indices[index]
        return _sample_task_identity(dataset.dataset, int(subset_index))

    if hasattr(dataset, "rows") and hasattr(dataset, "base_dataset"):
        row = dataset.rows[index]
        if isinstance(row, Mapping) and "dataset_index" in row:
            return _sample_task_identity(dataset.base_dataset, int(row["dataset_index"]))

    if hasattr(dataset, "base_dataset") and hasattr(dataset, "cached_dataset"):
        base_dataset = dataset.base_dataset
        if index < len(base_dataset):
            return _sample_task_identity(base_dataset, index)
        return _sample_task_identity(dataset.cached_dataset, index - len(base_dataset))

    raw_sample = getattr(dataset, "raw_sample", None)
    if callable(raw_sample):
        task = _task_identity_from_sample(raw_sample(index))
        if task is not None:
            return task

    task = _task_identity_from_sample(dataset[index])
    if task is None:
        raise ValueError(
            "IDM same-task batching requires each training sample to expose task_index or task_id. "
            f"Neither key was found for dataset index {index} from {type(dataset).__name__}."
        )
    return task


class SameTaskBatchSampler(Sampler[list[int]]):
    """Yield batches made from same-task chunks while preserving every sample."""

    def __init__(self, dataset: Dataset, *, batch_size: int, seed: int):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

        task_groups: dict[Hashable, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            task_groups[_sample_task_identity(dataset, index)].append(index)
        if not task_groups:
            raise ValueError("IDM same-task batching requires a non-empty training dataset.")
        if max(len(indices) for indices in task_groups.values()) < 2:
            raise ValueError(
                "IDM same-task batching requires at least one task with two or more training samples; "
                "all task groups were singletons."
            )
        self._task_groups = dict(task_groups)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}.")
        self.epoch = int(epoch)

    def __iter__(self):
        yield from self._build_batches()

    def __len__(self) -> int:
        return len(self._build_batches())

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        task_keys = list(self._task_groups)
        rng.shuffle(task_keys)

        chunks: list[list[int]] = []
        for task in task_keys:
            indices = list(self._task_groups[task])
            rng.shuffle(indices)
            chunks.extend(indices[start : start + self.batch_size] for start in range(0, len(indices), self.batch_size))
        rng.shuffle(chunks)

        batches: list[list[int]] = []
        current: list[int] = []
        for chunk in chunks:
            if current and len(current) + len(chunk) > self.batch_size:
                batches.append(current)
                current = []
            current.extend(chunk)
            if len(current) == self.batch_size:
                batches.append(current)
                current = []
        if current:
            batches.append(current)
        return batches


def create_idm_train_loader(
    dataset: Dataset,
    config: TrainConfig,
    *,
    pin_memory: bool,
) -> DataLoader:
    if not config.idm_same_task_batching:
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )

    # Singleton task groups are still emitted and may land in mixed or leftover
    # batches; they simply produce no same-task donor for that sample. The
    # training loop calls set_epoch because DataLoader does not do it for custom
    # batch samplers outside DistributedSampler.
    batch_sampler = SameTaskBatchSampler(dataset, batch_size=config.batch_size, seed=config.seed)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )


def effective_training_split_gap(config: TrainConfig) -> int:
    """Return the split gap needed by fallback contiguous splits for this temporal window."""

    if config.dataset.source == "synthetic":
        return config.split_gap
    future_window = int(config.dataset.frame_delta) * int(config.dataset.num_future_frames)
    action_window = int(config.dataset.idm_history_length) + int(config.dataset.action_horizon)
    return max(int(config.split_gap), future_window, action_window)


def create_models(config: ModelConfig, device: torch.device) -> tuple[ConvVideoWorldModel, InverseDynamicsModel]:
    return ConvVideoWorldModel(config).to(device), InverseDynamicsModel(config).to(device)


def create_idm_model(config: ModelConfig, device: torch.device) -> InverseDynamicsModel:
    return InverseDynamicsModel(config).to(device)


def create_dataset_with_optional_cache(
    dataset_config,
    cached_future_dir: str | Path | None = None,
    *,
    include_gt_futures_with_cache: bool = False,
) -> Dataset:
    dataset = create_dataset(dataset_config)
    if cached_future_dir is None:
        if include_gt_futures_with_cache:
            raise ValueError("include_gt_futures_with_cache requires cached_future_dir.")
        return dataset
    if include_gt_futures_with_cache:
        return MixedFutureDataset(dataset, cached_future_dir)
    return CachedFutureDataset(dataset, cached_future_dir)


def maybe_data_parallel(
    world_model: ConvVideoWorldModel,
    idm: InverseDynamicsModel,
    *,
    enabled: bool,
) -> tuple[nn.Module, nn.Module]:
    if not enabled:
        return world_model, idm
    if torch.cuda.device_count() < 2:
        raise RuntimeError("--data-parallel requires at least two CUDA devices.")
    return nn.DataParallel(world_model), nn.DataParallel(idm)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in unwrap_model(model).parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in unwrap_model(model).parameters() if parameter.requires_grad)


def idm_uses_flow_matching(idm: nn.Module) -> bool:
    return bool(getattr(unwrap_model(idm), "uses_flow_matching", False))


def _validate_flow_num_samples(num_samples: int) -> int:
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise ValueError(f"flow_num_samples must be a positive int, got {num_samples!r}.")
    if num_samples <= 0:
        raise ValueError(f"flow_num_samples must be positive, got {num_samples}.")
    return int(num_samples)


def _validate_flow_sample_noise_scale(noise_scale: float) -> float:
    if isinstance(noise_scale, bool) or not isinstance(noise_scale, int | float):
        raise ValueError(f"flow_noise_scale must be a non-negative float, got {noise_scale!r}.")
    resolved = float(noise_scale)
    if resolved < 0.0:
        raise ValueError(f"flow_noise_scale must be non-negative, got {resolved}.")
    return resolved


def resolve_flow_num_samples(idm: nn.Module, num_samples: int | None = None) -> int | None:
    if num_samples is not None:
        resolved = _validate_flow_num_samples(num_samples)
        if not idm_uses_flow_matching(idm):
            raise ValueError("flow_num_samples override requires a flow-matching IDM.")
        return resolved
    if not idm_uses_flow_matching(idm):
        return None
    return int(unwrap_model(idm).config.idm_flow_num_samples)


def resolve_flow_sample_noise_scale(idm: nn.Module, noise_scale: float | None = None) -> float | None:
    if noise_scale is not None:
        resolved = _validate_flow_sample_noise_scale(noise_scale)
        if not idm_uses_flow_matching(idm):
            raise ValueError("flow_noise_scale override requires a flow-matching IDM.")
        return resolved
    if not idm_uses_flow_matching(idm):
        return None
    return float(unwrap_model(idm).config.idm_flow_sample_noise_scale)


@contextlib.contextmanager
def temporary_flow_sampling_config(
    idm: nn.Module,
    *,
    num_samples: int | None = None,
    noise_scale: float | None = None,
) -> Iterator[tuple[int | None, float | None]]:
    resolved_num_samples = resolve_flow_num_samples(idm, num_samples)
    resolved_noise_scale = resolve_flow_sample_noise_scale(idm, noise_scale)
    if num_samples is None and noise_scale is None:
        yield resolved_num_samples, resolved_noise_scale
        return

    module = unwrap_model(idm)
    flow_head = getattr(module, "flow_head", None)
    if flow_head is None or not hasattr(flow_head, "config"):
        raise ValueError("flow sampling override requires a flow-matching IDM with flow_head.config.")

    original_config = module.config
    original_flow_head_config = flow_head.config
    replacements = {}
    if num_samples is not None:
        replacements["idm_flow_num_samples"] = resolved_num_samples
    if noise_scale is not None:
        replacements["idm_flow_sample_noise_scale"] = resolved_noise_scale
    try:
        override_config = dataclasses.replace(original_config, **replacements)
        override_flow_head_config = dataclasses.replace(original_flow_head_config, **replacements)
    except TypeError as error:
        raise ValueError("flow sampling override requires dataclass model and flow_head configs.") from error

    module.config = override_config
    flow_head.config = override_flow_head_config
    try:
        yield resolved_num_samples, resolved_noise_scale
    finally:
        module.config = original_config
        flow_head.config = original_flow_head_config


@contextlib.contextmanager
def temporary_flow_num_samples(idm: nn.Module, num_samples: int) -> Iterator[int]:
    with temporary_flow_sampling_config(idm, num_samples=num_samples) as (resolved, _):
        if resolved is None:
            raise ValueError("flow_num_samples override requires a flow-matching IDM.")
        yield resolved


def create_flow_sample_noise(
    idm: nn.Module,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    config = unwrap_model(idm).config
    shape = (
        batch_size * config.idm_flow_num_samples,
        config.action_horizon,
        config.action_dim,
    )
    noise_scale = config.idm_flow_sample_noise_scale
    if noise_scale == 0.0:
        return torch.zeros(shape, device=device, dtype=dtype)
    return torch.randn(*shape, device=device, dtype=dtype, generator=generator) * noise_scale


@dataclasses.dataclass(frozen=True)
class ActionNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def to(self, device: torch.device) -> ActionNormalizer:
        return ActionNormalizer(mean=self.mean.to(device), std=self.std.to(device))

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        mean, std = self._broadcast(action)
        return (action - mean) / std

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        mean, std = self._broadcast(action)
        return action * std + mean

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": [float(value) for value in self.mean.detach().cpu().tolist()],
            "std": [float(value) for value in self.std.detach().cpu().tolist()],
        }

    def _broadcast(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean.to(action.device, dtype=action.dtype)
        std = self.std.to(action.device, dtype=action.dtype)
        while mean.ndim < action.ndim:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean, std


@dataclasses.dataclass(frozen=True)
class StateNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def to(self, device: torch.device) -> StateNormalizer:
        return StateNormalizer(mean=self.mean.to(device), std=self.std.to(device))

    def normalize(self, state: torch.Tensor) -> torch.Tensor:
        mean, std = self._broadcast(state)
        return (state - mean) / std

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": [float(value) for value in self.mean.detach().cpu().tolist()],
            "std": [float(value) for value in self.std.detach().cpu().tolist()],
        }

    def _broadcast(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean.to(state.device, dtype=state.dtype)
        std = self.std.to(state.device, dtype=state.dtype)
        while mean.ndim < state.ndim:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean, std


def action_normalizer_from_dict(data: dict[str, Any], device: torch.device) -> ActionNormalizer:
    return ActionNormalizer(
        mean=torch.tensor(data["mean"], dtype=torch.float32, device=device),
        std=torch.tensor(data["std"], dtype=torch.float32, device=device),
    )


def state_normalizer_from_dict(data: dict[str, Any], device: torch.device) -> StateNormalizer:
    return StateNormalizer(
        mean=torch.tensor(data["mean"], dtype=torch.float32, device=device),
        std=torch.tensor(data["std"], dtype=torch.float32, device=device),
    )


def attach_action_normalizer(model: nn.Module, normalizer: ActionNormalizer | None) -> None:
    unwrap_model(model)._action_normalizer = normalizer


def get_action_normalizer(model: nn.Module, device: torch.device | None = None) -> ActionNormalizer | None:
    normalizer = getattr(unwrap_model(model), "_action_normalizer", None)
    if normalizer is None or device is None:
        return normalizer
    return normalizer.to(device)


def _state_normalizer_forward_pre_hook(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    normalizer = getattr(module, "_state_normalizer", None)
    if normalizer is None:
        return args, kwargs
    if len(args) >= 3:
        normalized_args = list(args)
        normalized_args[2] = normalizer.normalize(normalized_args[2])
        return tuple(normalized_args), kwargs
    if "state" in kwargs:
        normalized_kwargs = dict(kwargs)
        normalized_kwargs["state"] = normalizer.normalize(normalized_kwargs["state"])
        return args, normalized_kwargs
    return args, kwargs


def attach_state_normalizer(
    model: nn.Module,
    normalizer: StateNormalizer | None,
    *,
    normalize_forward: bool = False,
) -> None:
    module = unwrap_model(model)
    handle = getattr(module, "_state_normalizer_forward_hook_handle", None)
    if handle is not None:
        handle.remove()
        module._state_normalizer_forward_hook_handle = None
    module._state_normalizer = normalizer
    module._state_normalizer_applies_in_forward = False
    if normalizer is not None and normalize_forward:
        module._state_normalizer_forward_hook_handle = module.register_forward_pre_hook(
            _state_normalizer_forward_pre_hook,
            with_kwargs=True,
        )
        module._state_normalizer_applies_in_forward = True


def get_state_normalizer(model: nn.Module, device: torch.device | None = None) -> StateNormalizer | None:
    normalizer = getattr(unwrap_model(model), "_state_normalizer", None)
    if normalizer is None or device is None:
        return normalizer
    return normalizer.to(device)


def state_normalizer_applies_in_forward(model: nn.Module) -> bool:
    return bool(getattr(unwrap_model(model), "_state_normalizer_applies_in_forward", False))


def normalize_state_for_idm(
    idm: nn.Module,
    state: torch.Tensor,
    state_normalizer: StateNormalizer | None,
) -> torch.Tensor:
    if state_normalizer is None or state_normalizer_applies_in_forward(idm):
        return state
    return state_normalizer.normalize(state)


def idm_history_kwargs(
    batch: dict[str, torch.Tensor],
    *,
    idm: nn.Module | None = None,
    action_normalizer: ActionNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
) -> dict[str, torch.Tensor]:
    history_keys = ("prev_state_history", "prev_action_history", "history_mask")
    present = [key in batch for key in history_keys]
    if not any(present):
        return {}
    if not all(present):
        missing = [key for key, is_present in zip(history_keys, present, strict=True) if not is_present]
        raise ValueError(f"IDM history batch is missing required key(s): {missing}.")
    prev_state_history = batch["prev_state_history"]
    prev_action_history = batch["prev_action_history"]
    if state_normalizer is not None and (idm is None or not state_normalizer_applies_in_forward(idm)):
        prev_state_history = state_normalizer.normalize(prev_state_history)
    if action_normalizer is not None:
        prev_action_history = action_normalizer.normalize(prev_action_history)
    return {
        "prev_state_history": prev_state_history,
        "prev_action_history": prev_action_history,
        "history_mask": batch["history_mask"],
    }


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def compute_losses(
    world_model: nn.Module,
    idm: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    idm_target_source: IdmTargetSource,
) -> dict[str, torch.Tensor]:
    predicted_images = world_model(batch["current_images"], batch["state"], batch["task_id"])
    wm_loss = masked_mse(predicted_images, batch["future_images"], batch["future_image_mask"])

    if idm_target_source == "ground_truth":
        idm_target_images = batch["future_images"]
    elif idm_target_source == "generated":
        idm_target_images = predicted_images.detach()
    else:
        raise ValueError(f"Unknown idm_target_source: {idm_target_source}")

    if idm_uses_flow_matching(idm):
        flow_kwargs = idm_history_kwargs(batch, idm=idm)
        flow_outputs = idm(
            batch["current_images"],
            idm_target_images,
            batch["state"],
            batch["task_id"],
            target_action=batch["action_chunk"],
            action_mask=batch["action_mask"],
            mode="loss",
            **flow_kwargs,
        )
        idm_loss = flow_outputs["loss"]
        action_smoothness_loss = idm_loss.new_zeros(())
    else:
        predicted_action = idm(
            batch["current_images"],
            idm_target_images,
            batch["state"],
            batch["task_id"],
            **idm_history_kwargs(batch, idm=idm),
        )
        idm_loss = masked_smooth_l1(predicted_action, batch["action_chunk"], batch["action_mask"])
        action_smoothness_loss = action_smoothness(predicted_action, batch["action_mask"])
    return {
        "wm_loss": wm_loss,
        "idm_loss": idm_loss,
        "action_smoothness_loss": action_smoothness_loss,
        "predicted_images": predicted_images,
    }


def masked_mse_sum_and_count(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    while mask.ndim < predicted.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=predicted.device, dtype=predicted.dtype)
    squared = (predicted - target).square() * mask
    return squared.sum(), mask.expand_as(predicted).sum()


def masked_mse(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    total, count = masked_mse_sum_and_count(predicted, target, mask)
    return total / count.clamp_min(1.0)


def masked_smooth_l1_sum_and_count(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    while mask.ndim < predicted.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=predicted.device, dtype=predicted.dtype)
    loss = F.smooth_l1_loss(predicted, target, reduction="none") * mask
    return loss.sum(), mask.expand_as(predicted).sum()


def masked_smooth_l1(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    total, count = masked_smooth_l1_sum_and_count(predicted, target, mask)
    return total / count.clamp_min(1.0)


def action_smoothness(action: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if action.shape[1] < 2:
        return action.new_zeros(())
    delta = action[:, 1:] - action[:, :-1]
    delta_mask = mask[:, 1:] * mask[:, :-1]
    while delta_mask.ndim < delta.ndim:
        delta_mask = delta_mask.unsqueeze(-1)
    return (delta.square() * delta_mask).sum() / delta_mask.expand_as(delta).sum().clamp_min(1.0)


@dataclasses.dataclass(frozen=True)
class SameTaskDonorSample:
    donor_indices: torch.Tensor
    has_donor: torch.Tensor
    used_different_episode: torch.Tensor
    state_distance: torch.Tensor
    max_state_distance_filtered: torch.Tensor
    min_action_delta_filtered: torch.Tensor


@dataclasses.dataclass(frozen=True)
class FutureRankingNegativeCandidate:
    name: str
    future_images: torch.Tensor
    wan_vae_latents: torch.Tensor | None
    valid_mask: torch.Tensor


def _batch_scalar_long(batch: dict[str, torch.Tensor], key: str, *, batch_size: int) -> torch.Tensor | None:
    if key not in batch:
        return None
    value = batch[key]
    if value.ndim == 0:
        value = value.reshape(1)
    value = value.reshape(value.shape[0], -1)[:, 0].to(dtype=torch.long)
    if value.shape[0] != batch_size:
        raise ValueError(f"{key} must have batch dimension {batch_size}, got {value.shape[0]}.")
    return value


def _first_candidate_after_anchor(candidate_mask: torch.Tensor, anchor_index: int) -> int:
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        raise ValueError("candidate_mask must contain at least one valid candidate.")
    after_anchor = candidate_indices[candidate_indices > anchor_index]
    if after_anchor.numel() > 0:
        return int(after_anchor[0])
    return int(candidate_indices[0])


def sample_same_task_donors(
    batch: dict[str, torch.Tensor],
    *,
    min_same_episode_frame_gap: int,
) -> SameTaskDonorSample:
    if min_same_episode_frame_gap < 0:
        raise ValueError(f"min_same_episode_frame_gap must be non-negative, got {min_same_episode_frame_gap}.")
    batch_size = int(batch["action_chunk"].shape[0])
    device = batch["action_chunk"].device
    task = _batch_scalar_long(
        batch,
        "task_index" if "task_index" in batch else "task_id",
        batch_size=batch_size,
    )
    if task is None:
        raise KeyError("Same-task donor sampling requires task_index or task_id in the batch.")
    dataset_index = _batch_scalar_long(batch, "dataset_index", batch_size=batch_size)
    episode_index = _batch_scalar_long(batch, "episode_index", batch_size=batch_size)
    frame_index = _batch_scalar_long(batch, "frame_index", batch_size=batch_size)
    positions = torch.arange(batch_size, device=device, dtype=torch.long)

    donor_indices = torch.zeros(batch_size, device=device, dtype=torch.long)
    has_donor = torch.zeros(batch_size, device=device, dtype=torch.bool)
    used_different_episode = torch.zeros(batch_size, device=device, dtype=torch.bool)
    for anchor_index in range(batch_size):
        same_task = task == task[anchor_index]
        if dataset_index is None:
            not_self = positions != anchor_index
        else:
            not_self = dataset_index != dataset_index[anchor_index]
        candidates = same_task & not_self
        selected_mask = candidates
        selected_different_episode = False

        if episode_index is not None:
            different_episode = candidates & (episode_index != episode_index[anchor_index])
            if bool(different_episode.any()):
                selected_mask = different_episode
                selected_different_episode = True
            else:
                selected_mask = candidates & (episode_index == episode_index[anchor_index])
                if frame_index is not None:
                    frame_gap = (frame_index - frame_index[anchor_index]).abs()
                    selected_mask = selected_mask & (frame_gap >= min_same_episode_frame_gap)

        if bool(selected_mask.any()):
            donor_indices[anchor_index] = _first_candidate_after_anchor(selected_mask, anchor_index)
            has_donor[anchor_index] = True
            used_different_episode[anchor_index] = selected_different_episode

    return SameTaskDonorSample(
        donor_indices=donor_indices,
        has_donor=has_donor,
        used_different_episode=used_different_episode,
        state_distance=torch.zeros(batch_size, device=device, dtype=batch["action_chunk"].dtype),
        max_state_distance_filtered=torch.zeros(batch_size, device=device, dtype=torch.bool),
        min_action_delta_filtered=torch.zeros(batch_size, device=device, dtype=torch.bool),
    )


def _pairwise_state_distance(state: torch.Tensor) -> torch.Tensor:
    flat_state = state.reshape(state.shape[0], -1).to(dtype=torch.float32)
    if flat_state.shape[1] == 0:
        return flat_state.new_zeros((flat_state.shape[0], flat_state.shape[0]))
    return torch.cdist(flat_state, flat_state, p=2) / math.sqrt(float(flat_state.shape[1]))


def _pairwise_action_delta_mse(action: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    if action.ndim < 2:
        raise ValueError(f"action must have batch and time dimensions, got shape {tuple(action.shape)}.")
    if action_mask.shape[:2] != action.shape[:2]:
        raise ValueError(
            "action_mask must match the first two action dimensions, "
            f"got action shape {tuple(action.shape)} and mask shape {tuple(action_mask.shape)}."
        )
    diff = action.unsqueeze(1) - action.unsqueeze(0)
    mask = action_mask.unsqueeze(1) * action_mask.unsqueeze(0)
    while mask.ndim < diff.ndim:
        mask = mask.unsqueeze(-1)
    weighted = diff.square() * mask
    denominator = mask.expand_as(diff).sum(dim=tuple(range(2, diff.ndim))).clamp_min(1.0)
    return weighted.sum(dim=tuple(range(2, diff.ndim))) / denominator


def sample_same_task_future_delta_donors(
    batch: dict[str, torch.Tensor],
    *,
    min_same_episode_frame_gap: int,
    state: torch.Tensor | None = None,
    max_state_distance: float | None = None,
    action: torch.Tensor | None = None,
    min_action_delta_mse: float = 0.0,
) -> SameTaskDonorSample:
    if min_same_episode_frame_gap < 0:
        raise ValueError(f"min_same_episode_frame_gap must be non-negative, got {min_same_episode_frame_gap}.")
    if max_state_distance is not None and max_state_distance < 0.0:
        raise ValueError(f"max_state_distance must be non-negative or None, got {max_state_distance}.")
    if min_action_delta_mse < 0.0:
        raise ValueError(f"min_action_delta_mse must be non-negative, got {min_action_delta_mse}.")

    batch_size = int(batch["action_chunk"].shape[0])
    device = batch["action_chunk"].device
    dtype = batch["action_chunk"].dtype
    task = _batch_scalar_long(
        batch,
        "task_index" if "task_index" in batch else "task_id",
        batch_size=batch_size,
    )
    if task is None:
        raise KeyError("Same-task future/action-delta donor sampling requires task_index or task_id in the batch.")
    dataset_index = _batch_scalar_long(batch, "dataset_index", batch_size=batch_size)
    episode_index = _batch_scalar_long(batch, "episode_index", batch_size=batch_size)
    frame_index = _batch_scalar_long(batch, "frame_index", batch_size=batch_size)
    positions = torch.arange(batch_size, device=device, dtype=torch.long)

    state_reference = state if state is not None else batch.get("state")
    state_distances = None if state_reference is None else _pairwise_state_distance(state_reference).to(device=device)
    action_reference = action if action is not None else batch["action_chunk"]
    action_delta_mses = _pairwise_action_delta_mse(action_reference, batch["action_mask"]).to(device=device)

    donor_indices = torch.zeros(batch_size, device=device, dtype=torch.long)
    has_donor = torch.zeros(batch_size, device=device, dtype=torch.bool)
    used_different_episode = torch.zeros(batch_size, device=device, dtype=torch.bool)
    state_distance = torch.zeros(batch_size, device=device, dtype=dtype)
    max_state_distance_filtered = torch.zeros(batch_size, device=device, dtype=torch.bool)
    min_action_delta_filtered = torch.zeros(batch_size, device=device, dtype=torch.bool)

    for anchor_index in range(batch_size):
        same_task = task == task[anchor_index]
        if dataset_index is None:
            not_self = positions != anchor_index
        else:
            not_self = dataset_index != dataset_index[anchor_index]
        candidates = same_task & not_self
        selected_mask = candidates

        if episode_index is not None:
            different_episode = candidates & (episode_index != episode_index[anchor_index])
            if bool(different_episode.any()):
                selected_mask = different_episode
            else:
                selected_mask = candidates & (episode_index == episode_index[anchor_index])
                if frame_index is not None:
                    frame_gap = (frame_index - frame_index[anchor_index]).abs()
                    selected_mask = selected_mask & (frame_gap >= min_same_episode_frame_gap)

        filtered_mask = selected_mask
        if max_state_distance is not None and state_distances is not None:
            state_mask = state_distances[anchor_index].to(dtype=torch.float32) <= float(max_state_distance)
            state_filtered_mask = filtered_mask & state_mask
            max_state_distance_filtered[anchor_index] = bool(filtered_mask.any()) and not bool(
                state_filtered_mask.any()
            )
            filtered_mask = state_filtered_mask

        if min_action_delta_mse > 0.0:
            action_mask = action_delta_mses[anchor_index].to(dtype=torch.float32) >= float(min_action_delta_mse)
            action_filtered_mask = filtered_mask & action_mask
            min_action_delta_filtered[anchor_index] = bool(filtered_mask.any()) and not bool(action_filtered_mask.any())
            filtered_mask = action_filtered_mask

        if bool(filtered_mask.any()):
            donor_index = _first_candidate_after_anchor(filtered_mask, anchor_index)
            donor_indices[anchor_index] = donor_index
            has_donor[anchor_index] = True
            if episode_index is not None:
                used_different_episode[anchor_index] = bool(episode_index[donor_index] != episode_index[anchor_index])
            if state_distances is not None:
                state_distance[anchor_index] = state_distances[anchor_index, donor_index].to(dtype=dtype)

    return SameTaskDonorSample(
        donor_indices=donor_indices,
        has_donor=has_donor,
        used_different_episode=used_different_episode,
        state_distance=state_distance,
        max_state_distance_filtered=max_state_distance_filtered,
        min_action_delta_filtered=min_action_delta_filtered,
    )


def _corrupt_future_images_for_contrast(
    current_images: torch.Tensor,
    future_images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(future_images.shape[0])
    if batch_size <= 1:
        repeated_current = current_images.unsqueeze(1).expand_as(future_images)
        return repeated_current, future_images.new_tensor(1.0)
    return future_images.roll(shifts=1, dims=0), future_images.new_tensor(0.0)


def _corrupt_wan_vae_latents_for_contrast(wan_vae_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if wan_vae_latents.ndim != 5:
        raise ValueError(
            "Future contrastive Wan VAE latent corruption requires rank 5 latents "
            f"with shape (B,C,T,H,W), got shape {tuple(wan_vae_latents.shape)}."
        )
    if wan_vae_latents.shape[2] < 2:
        raise ValueError(
            "Future contrastive Wan VAE latent corruption requires at least two latent time steps "
            f"(T >= 2), got shape {tuple(wan_vae_latents.shape)}."
        )

    batch_size = int(wan_vae_latents.shape[0])
    corrupted = wan_vae_latents.clone()
    if batch_size <= 1:
        current_latent = wan_vae_latents[:, :, :1]
        corrupted[:, :, 1:] = current_latent.expand_as(wan_vae_latents[:, :, 1:])
        return corrupted, wan_vae_latents.new_tensor(1.0)
    corrupted[:, :, 1:] = wan_vae_latents[:, :, 1:].roll(shifts=1, dims=0)
    return corrupted, wan_vae_latents.new_tensor(0.0)


def _wan_vae_future_time_mask(wan_vae_latents: torch.Tensor, *, context: str) -> torch.Tensor:
    if wan_vae_latents.ndim != 5:
        raise ValueError(
            f"{context} Wan VAE latent corruption requires rank 5 latents "
            f"with shape (B,C,T,H,W), got shape {tuple(wan_vae_latents.shape)}."
        )
    if wan_vae_latents.shape[2] < 1:
        raise ValueError(
            f"{context} Wan VAE latent corruption requires at least one latent time step, "
            f"got shape {tuple(wan_vae_latents.shape)}."
        )
    time_mask = torch.ones(
        (1, 1, wan_vae_latents.shape[2], 1, 1),
        device=wan_vae_latents.device,
        dtype=torch.bool,
    )
    if wan_vae_latents.shape[2] >= 2:
        time_mask[:, :, :1] = False
    return time_mask


def _replace_wan_vae_future_latents(
    wan_vae_latents: torch.Tensor,
    replacement: torch.Tensor,
    *,
    context: str,
) -> torch.Tensor:
    time_mask = _wan_vae_future_time_mask(wan_vae_latents, context=context)
    return torch.where(time_mask.expand_as(wan_vae_latents), replacement, wan_vae_latents)


def _repeat_current_wan_vae_future_latents(wan_vae_latents: torch.Tensor) -> torch.Tensor:
    if wan_vae_latents.ndim != 5:
        raise ValueError(
            "Future ranking Wan VAE repeated-current corruption requires rank 5 latents "
            f"with shape (B,C,T,H,W), got shape {tuple(wan_vae_latents.shape)}."
        )
    if wan_vae_latents.shape[2] >= 2:
        corrupted = wan_vae_latents.clone()
        corrupted[:, :, 1:] = wan_vae_latents[:, :, :1].expand_as(wan_vae_latents[:, :, 1:])
        return corrupted
    return torch.zeros_like(wan_vae_latents)


def _shuffle_wan_vae_future_latents(wan_vae_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if wan_vae_latents.ndim != 5:
        raise ValueError(
            "Future ranking Wan VAE shuffled-future corruption requires rank 5 latents "
            f"with shape (B,C,T,H,W), got shape {tuple(wan_vae_latents.shape)}."
        )
    batch_size = int(wan_vae_latents.shape[0])
    if batch_size <= 1:
        return _repeat_current_wan_vae_future_latents(wan_vae_latents), wan_vae_latents.new_tensor(1.0)
    if wan_vae_latents.shape[2] >= 2:
        corrupted = wan_vae_latents.clone()
        corrupted[:, :, 1:] = wan_vae_latents[:, :, 1:].roll(shifts=1, dims=0)
        return corrupted, wan_vae_latents.new_tensor(0.0)
    return wan_vae_latents.roll(shifts=1, dims=0), wan_vae_latents.new_tensor(0.0)


def _same_task_future_ranking_negative_candidate(
    batch: dict[str, torch.Tensor],
    *,
    min_same_episode_frame_gap: int,
) -> tuple[FutureRankingNegativeCandidate, dict[str, torch.Tensor]]:
    donor_sample = sample_same_task_donors(
        batch,
        min_same_episode_frame_gap=min_same_episode_frame_gap,
    )
    donor_indices = donor_sample.donor_indices
    has_donor = donor_sample.has_donor
    real_wan_vae_latents = batch.get("wan_vae_latents")
    swapped_wan_vae_latents = None
    latent_valid = torch.ones_like(has_donor)
    if real_wan_vae_latents is not None:
        swapped_wan_vae_latents, latent_valid = swapped_same_task_wan_vae_future_latents(
            real_wan_vae_latents,
            donor_indices,
        )

    # This negative is label-free: we keep the anchor action target and only swap in a
    # plausible same-task future. Samples without a valid donor/latent are masked out
    # of this candidate instead of falling back to an easier synthetic corruption.
    valid_mask = has_donor & latent_valid
    metrics = {
        "same_task_valid_fraction": valid_mask.to(dtype=batch["future_images"].dtype).mean(),
        "same_task_no_donor_fraction": (~has_donor).to(dtype=batch["future_images"].dtype).mean(),
        "same_task_different_episode_fraction": donor_sample.used_different_episode.to(
            dtype=batch["future_images"].dtype
        ).mean(),
        "same_task_wan_latent_too_short_fraction": (~latent_valid).to(dtype=batch["future_images"].dtype).mean(),
    }
    return (
        FutureRankingNegativeCandidate(
            name="same_task_future",
            future_images=batch["future_images"][donor_indices],
            wan_vae_latents=swapped_wan_vae_latents,
            valid_mask=valid_mask,
        ),
        metrics,
    )


def _future_ranking_negative_candidates(
    batch: dict[str, torch.Tensor],
    *,
    repeated_current: bool,
    shuffled_future: bool,
    noisy_future: bool,
    zero_future: bool,
    same_task_future: bool,
    noise_std: float,
    min_same_episode_frame_gap: int,
) -> tuple[list[FutureRankingNegativeCandidate], torch.Tensor, dict[str, torch.Tensor]]:
    future_images = batch["future_images"]
    real_wan_vae_latents = batch.get("wan_vae_latents")
    singleton_indicator = future_images.new_tensor(0.0)
    candidates: list[FutureRankingNegativeCandidate] = []
    batch_size = int(future_images.shape[0])
    all_valid = torch.ones(batch_size, device=future_images.device, dtype=torch.bool)
    same_task_metrics = {
        "same_task_valid_fraction": future_images.new_zeros(()),
        "same_task_no_donor_fraction": future_images.new_zeros(()),
        "same_task_different_episode_fraction": future_images.new_zeros(()),
        "same_task_wan_latent_too_short_fraction": future_images.new_zeros(()),
    }

    if real_wan_vae_latents is None:
        if repeated_current:
            candidates.append(
                FutureRankingNegativeCandidate(
                    name="repeated_current",
                    future_images=batch["current_images"].unsqueeze(1).expand_as(future_images),
                    wan_vae_latents=None,
                    valid_mask=all_valid,
                )
            )
        if shuffled_future:
            if batch_size <= 1:
                shuffled_images = batch["current_images"].unsqueeze(1).expand_as(future_images)
                singleton_indicator = future_images.new_tensor(1.0)
            else:
                shuffled_images = future_images.roll(shifts=1, dims=0)
            candidates.append(
                FutureRankingNegativeCandidate(
                    name="shuffled_future",
                    future_images=shuffled_images,
                    wan_vae_latents=None,
                    valid_mask=all_valid,
                )
            )
        if noisy_future:
            noisy_images = (future_images + torch.randn_like(future_images) * noise_std).clamp(0.0, 1.0)
            candidates.append(
                FutureRankingNegativeCandidate(
                    name="noisy_future",
                    future_images=noisy_images,
                    wan_vae_latents=None,
                    valid_mask=all_valid,
                )
            )
        if zero_future:
            candidates.append(
                FutureRankingNegativeCandidate(
                    name="zero_future",
                    future_images=torch.zeros_like(future_images),
                    wan_vae_latents=None,
                    valid_mask=all_valid,
                )
            )
        if same_task_future:
            same_task_candidate, same_task_metrics = _same_task_future_ranking_negative_candidate(
                batch,
                min_same_episode_frame_gap=min_same_episode_frame_gap,
            )
            candidates.append(same_task_candidate)
        return candidates, singleton_indicator, same_task_metrics

    if repeated_current:
        candidates.append(
            FutureRankingNegativeCandidate(
                name="repeated_current",
                future_images=future_images,
                wan_vae_latents=_repeat_current_wan_vae_future_latents(real_wan_vae_latents),
                valid_mask=all_valid,
            )
        )
    if shuffled_future:
        shuffled_latents, singleton_indicator = _shuffle_wan_vae_future_latents(real_wan_vae_latents)
        candidates.append(
            FutureRankingNegativeCandidate(
                name="shuffled_future",
                future_images=future_images,
                wan_vae_latents=shuffled_latents,
                valid_mask=all_valid,
            )
        )
    if noisy_future:
        replacement = real_wan_vae_latents + torch.randn_like(real_wan_vae_latents) * noise_std
        candidates.append(
            FutureRankingNegativeCandidate(
                name="noisy_future",
                future_images=future_images,
                wan_vae_latents=_replace_wan_vae_future_latents(
                    real_wan_vae_latents,
                    replacement,
                    context="Future ranking",
                ),
                valid_mask=all_valid,
            )
        )
    if zero_future:
        candidates.append(
            FutureRankingNegativeCandidate(
                name="zero_future",
                future_images=future_images,
                wan_vae_latents=_replace_wan_vae_future_latents(
                    real_wan_vae_latents,
                    torch.zeros_like(real_wan_vae_latents),
                    context="Future ranking",
                ),
                valid_mask=all_valid,
            )
        )
    if same_task_future:
        same_task_candidate, same_task_metrics = _same_task_future_ranking_negative_candidate(
            batch,
            min_same_episode_frame_gap=min_same_episode_frame_gap,
        )
        candidates.append(same_task_candidate)
    return candidates, singleton_indicator, same_task_metrics


def _flow_teacher_forced_endpoint_prediction(
    idm: nn.Module,
    current_images: torch.Tensor,
    future_images: torch.Tensor,
    state: torch.Tensor,
    target_action: torch.Tensor,
    noise: torch.Tensor,
    *,
    wan_vae_latents: torch.Tensor | None = None,
    prev_state_history: torch.Tensor | None = None,
    prev_action_history: torch.Tensor | None = None,
    history_mask: torch.Tensor | None = None,
    time_value: float = 0.5,
) -> torch.Tensor:
    flow_idm = unwrap_model(idm)
    context, visual_context_tokens = _flow_transition_context_and_visual_tokens(
        flow_idm,
        current_images,
        future_images,
        state,
        wan_vae_latents=wan_vae_latents,
    )
    if prev_state_history is None and prev_action_history is None and history_mask is None:
        history_tokens = None
    else:
        history_tokens = flow_idm._history_tokens(prev_state_history, prev_action_history, history_mask)
    time = torch.full(
        (target_action.shape[0],),
        time_value,
        device=target_action.device,
        dtype=target_action.dtype,
    )
    time_view = time.view(-1, 1, 1)
    noisy_action = (1.0 - time_view) * noise + time_view * target_action
    flow_head_kwargs = {}
    if visual_context_tokens is not None:
        flow_head_kwargs["visual_context_tokens"] = visual_context_tokens
    if history_tokens is not None:
        flow_head_kwargs["history_tokens"] = history_tokens
    predicted_velocity = flow_idm.flow_head(context, noisy_action, time, **flow_head_kwargs)
    return noisy_action + (1.0 - time_view) * predicted_velocity


def _flow_teacher_forced_endpoint_mse_per_sample(
    idm: nn.Module,
    current_images: torch.Tensor,
    future_images: torch.Tensor,
    state: torch.Tensor,
    target_action: torch.Tensor,
    action_mask: torch.Tensor,
    noise: torch.Tensor,
    *,
    wan_vae_latents: torch.Tensor | None = None,
    prev_state_history: torch.Tensor | None = None,
    prev_action_history: torch.Tensor | None = None,
    history_mask: torch.Tensor | None = None,
    time_value: float = 0.5,
) -> torch.Tensor:
    endpoint_prediction = _flow_teacher_forced_endpoint_prediction(
        idm,
        current_images,
        future_images,
        state,
        target_action,
        noise,
        wan_vae_latents=wan_vae_latents,
        prev_state_history=prev_state_history,
        prev_action_history=prev_action_history,
        history_mask=history_mask,
        time_value=time_value,
    )
    mask = action_mask
    while mask.ndim < endpoint_prediction.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=endpoint_prediction.device, dtype=endpoint_prediction.dtype)
    squared = (endpoint_prediction - target_action).square() * mask
    per_sample_total = squared.flatten(1).sum(dim=1)
    per_sample_count = mask.expand_as(endpoint_prediction).flatten(1).sum(dim=1).clamp_min(1.0)
    return per_sample_total / per_sample_count


def _flow_sampled_action_prediction(
    idm: nn.Module,
    current_images: torch.Tensor,
    future_images: torch.Tensor,
    state: torch.Tensor,
    target_action: torch.Tensor,
    *,
    wan_vae_latents: torch.Tensor | None = None,
    prev_state_history: torch.Tensor | None = None,
    prev_action_history: torch.Tensor | None = None,
    history_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deterministic flow-sampler action for a transition (mirrors the endpoint predictor's API).

    Unlike the teacher-forced endpoint estimate (a single denoising step from a shared noise
    sample), this runs the full multi-step sampler. Explicit zero ``sample_noise`` keeps the
    rollout deterministic even when ``idm_flow_sample_noise_scale`` is nonzero; its shape must
    match ``sample_action``'s ``(batch * idm_flow_num_samples, action_horizon, action_dim)``
    contract. ``target_action`` is used only as a device/dtype reference for that zero noise.
    """
    flow_idm = unwrap_model(idm)
    num_samples = flow_idm.config.idm_flow_num_samples
    zero_sample_noise = target_action.new_zeros(
        current_images.shape[0] * num_samples,
        flow_idm.config.action_horizon,
        flow_idm.config.action_dim,
    )
    return flow_idm.sample_action(
        current_images,
        future_images,
        state,
        sample_noise=zero_sample_noise,
        wan_vae_latents=wan_vae_latents,
        prev_state_history=prev_state_history,
        prev_action_history=prev_action_history,
        history_mask=history_mask,
    )


def _flow_sampled_action_mse_per_sample(
    idm: nn.Module,
    current_images: torch.Tensor,
    future_images: torch.Tensor,
    state: torch.Tensor,
    target_action: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    wan_vae_latents: torch.Tensor | None = None,
    prev_state_history: torch.Tensor | None = None,
    prev_action_history: torch.Tensor | None = None,
    history_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample masked MSE between the deterministic flow sampler output and the target action."""
    sampled_action = _flow_sampled_action_prediction(
        idm,
        current_images,
        future_images,
        state,
        target_action,
        wan_vae_latents=wan_vae_latents,
        prev_state_history=prev_state_history,
        prev_action_history=prev_action_history,
        history_mask=history_mask,
    )
    mask = action_mask
    while mask.ndim < sampled_action.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=sampled_action.device, dtype=sampled_action.dtype)
    squared = (sampled_action - target_action).square() * mask
    per_sample_total = squared.flatten(1).sum(dim=1)
    per_sample_count = mask.expand_as(sampled_action).flatten(1).sum(dim=1).clamp_min(1.0)
    return per_sample_total / per_sample_count


def _flow_transition_context_and_visual_tokens(
    flow_idm: nn.Module,
    current_images: torch.Tensor,
    future_images: torch.Tensor,
    state: torch.Tensor,
    *,
    wan_vae_latents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    transition_with_tokens = getattr(flow_idm, "_transition_context_and_visual_tokens", None)
    if callable(transition_with_tokens):
        return transition_with_tokens(
            current_images,
            future_images,
            state,
            wan_vae_latents=wan_vae_latents,
        )

    config = getattr(flow_idm, "config", None)
    if bool(getattr(config, "idm_flow_visual_token_conditioning", False)):
        if getattr(config, "idm_visual_encoder", None) != "wan_vae":
            raise ValueError("idm_flow_visual_token_conditioning is supported only with idm_visual_encoder='wan_vae'.")
        return flow_idm.transition_encoder(
            current_images,
            future_images,
            state,
            wan_vae_latents=wan_vae_latents,
            return_tokens=True,
        )

    if wan_vae_latents is None:
        return flow_idm.transition_encoder(current_images, future_images, state), None
    return (
        flow_idm.transition_encoder(
            current_images,
            future_images,
            state,
            wan_vae_latents=wan_vae_latents,
        ),
        None,
    )


def compute_future_contrastive_loss(
    idm: nn.Module,
    batch: dict[str, torch.Tensor],
    target_action: torch.Tensor,
    idm_state: torch.Tensor,
    *,
    action_normalizer: ActionNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
    margin: float,
) -> dict[str, torch.Tensor]:
    if not idm_uses_flow_matching(idm):
        raise ValueError("Future contrastive IDM loss is supported only for flow_transformer IDMs.")

    real_wan_vae_latents = batch.get("wan_vae_latents")
    history_kwargs = idm_history_kwargs(
        batch,
        idm=idm,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
    )
    corrupted_wan_vae_latents = None
    if real_wan_vae_latents is None:
        corrupted_future_images, singleton_indicator = _corrupt_future_images_for_contrast(
            batch["current_images"],
            batch["future_images"],
        )
    else:
        corrupted_future_images = batch["future_images"]
        corrupted_wan_vae_latents, singleton_indicator = _corrupt_wan_vae_latents_for_contrast(real_wan_vae_latents)
    shared_noise = torch.randn_like(target_action)
    real_mse = _flow_teacher_forced_endpoint_mse_per_sample(
        idm,
        batch["current_images"],
        batch["future_images"],
        idm_state,
        target_action,
        batch["action_mask"],
        shared_noise,
        wan_vae_latents=real_wan_vae_latents,
        **history_kwargs,
    )
    corrupted_mse = _flow_teacher_forced_endpoint_mse_per_sample(
        idm,
        batch["current_images"],
        corrupted_future_images,
        idm_state,
        target_action,
        batch["action_mask"],
        shared_noise,
        wan_vae_latents=corrupted_wan_vae_latents,
        **history_kwargs,
    )
    ranking_loss = F.relu(real_mse - corrupted_mse + margin).mean()
    return {
        "idm_future_contrastive_loss": ranking_loss,
        "idm_future_contrastive_real_endpoint_mse": real_mse.mean(),
        "idm_future_contrastive_corrupted_endpoint_mse": corrupted_mse.mean(),
        "idm_future_contrastive_singleton_fraction": singleton_indicator,
    }


def compute_future_ranking_loss(
    idm: nn.Module,
    batch: dict[str, torch.Tensor],
    target_action: torch.Tensor,
    idm_state: torch.Tensor,
    *,
    action_normalizer: ActionNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
    temperature: float,
    noise_std: float,
    repeated_current_negative: bool,
    shuffled_future_negative: bool,
    noisy_future_negative: bool,
    zero_future_negative: bool,
    same_task_negative: bool,
    min_same_episode_frame_gap: int,
    score_mode: IdmFutureRankingScoreMode = "teacher_forced_endpoint",
) -> dict[str, torch.Tensor]:
    if not idm_uses_flow_matching(idm):
        raise ValueError("Future ranking IDM loss is supported only for flow_transformer IDMs.")
    if temperature <= 0.0:
        raise ValueError(f"future_ranking_temperature must be positive, got {temperature}.")
    if noise_std < 0.0:
        raise ValueError(f"future_ranking_noise_std must be non-negative, got {noise_std}.")
    if score_mode not in ("teacher_forced_endpoint", "sampled_action"):
        raise ValueError(
            "future_ranking_score_mode must be one of {'teacher_forced_endpoint', 'sampled_action'}, "
            f"got {score_mode!r}."
        )

    if min_same_episode_frame_gap < 0:
        raise ValueError(
            "future_ranking_same_task min_same_episode_frame_gap must be non-negative, "
            f"got {min_same_episode_frame_gap}."
        )

    negative_candidates, singleton_indicator, same_task_metrics = _future_ranking_negative_candidates(
        batch,
        repeated_current=repeated_current_negative,
        shuffled_future=shuffled_future_negative,
        noisy_future=noisy_future_negative,
        zero_future=zero_future_negative,
        same_task_future=same_task_negative,
        noise_std=noise_std,
        min_same_episode_frame_gap=min_same_episode_frame_gap,
    )
    if not negative_candidates:
        raise ValueError("Future ranking IDM loss requires at least one enabled negative candidate.")

    real_wan_vae_latents = batch.get("wan_vae_latents")
    history_kwargs = idm_history_kwargs(
        batch,
        idm=idm,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
    )
    # Both score modes rank the same real future against the same negative candidates with the
    # same CE/rank logic; only the per-sample MSE scorer differs. The endpoint scorer shares one
    # noise draw across real and candidates for a fair single-step comparison, while the sampled
    # scorer is deterministic (explicit zero sample noise).
    if score_mode == "sampled_action":

        def _score(future_images: torch.Tensor, candidate_wan_vae_latents: torch.Tensor | None) -> torch.Tensor:
            return _flow_sampled_action_mse_per_sample(
                idm,
                batch["current_images"],
                future_images,
                idm_state,
                target_action,
                batch["action_mask"],
                wan_vae_latents=candidate_wan_vae_latents,
                **history_kwargs,
            )
    else:
        shared_noise = torch.randn_like(target_action)

        def _score(future_images: torch.Tensor, candidate_wan_vae_latents: torch.Tensor | None) -> torch.Tensor:
            return _flow_teacher_forced_endpoint_mse_per_sample(
                idm,
                batch["current_images"],
                future_images,
                idm_state,
                target_action,
                batch["action_mask"],
                shared_noise,
                wan_vae_latents=candidate_wan_vae_latents,
                **history_kwargs,
            )

    real_mse = _score(batch["future_images"], real_wan_vae_latents)
    candidate_mses = []
    for candidate in negative_candidates:
        if bool(candidate.valid_mask.any()):
            candidate_mses.append(_score(candidate.future_images, candidate.wan_vae_latents))
        else:
            candidate_mses.append(real_mse.new_zeros(real_mse.shape))
    negative_mses = torch.stack(candidate_mses, dim=1)
    candidate_valid = torch.stack(
        [candidate.valid_mask.to(device=real_mse.device) for candidate in negative_candidates],
        dim=1,
    )
    valid_sample = candidate_valid.any(dim=1)
    valid_negative_count = candidate_valid.to(dtype=real_mse.dtype).sum(dim=1)
    if bool(valid_sample.any()):
        invalid_logit = torch.finfo(real_mse.dtype).min
        negative_logits = -negative_mses / temperature
        negative_logits = negative_logits.masked_fill(~candidate_valid, invalid_logit)
        logits = torch.cat([-real_mse.unsqueeze(1) / temperature, negative_logits], dim=1)
        labels = torch.zeros(real_mse.shape[0], device=real_mse.device, dtype=torch.long)
        ranking_loss = F.cross_entropy(logits[valid_sample], labels[valid_sample])
        masked_negative_mses = negative_mses.masked_fill(~candidate_valid, torch.inf)
        best_negative_mse = masked_negative_mses.min(dim=1).values
        mean_negative_mse = (negative_mses * candidate_valid.to(dtype=negative_mses.dtype)).sum(dim=1) / (
            valid_negative_count.clamp_min(1.0)
        )
        rank_accuracy = (real_mse[valid_sample] < best_negative_mse[valid_sample]).to(dtype=real_mse.dtype).mean()
        real_score_mse = real_mse[valid_sample].mean()
        best_negative_score_mse = best_negative_mse[valid_sample].mean()
        mean_negative_score_mse = mean_negative_mse[valid_sample].mean()
    else:
        ranking_loss = real_mse.new_zeros(())
        rank_accuracy = real_mse.new_zeros(())
        real_score_mse = real_mse.new_zeros(())
        best_negative_score_mse = real_mse.new_zeros(())
        mean_negative_score_mse = real_mse.new_zeros(())
    # Route the scored MSEs to the active mode's metric keys; the inactive mode's keys stay zero so
    # downstream accumulation always sees every key regardless of score_mode.
    zero = real_mse.new_zeros(())
    if score_mode == "sampled_action":
        real_endpoint_mse = best_negative_endpoint_mse = mean_negative_endpoint_mse = zero
        real_sampled_action_mse = real_score_mse
        best_negative_sampled_action_mse = best_negative_score_mse
        mean_negative_sampled_action_mse = mean_negative_score_mse
    else:
        real_endpoint_mse = real_score_mse
        best_negative_endpoint_mse = best_negative_score_mse
        mean_negative_endpoint_mse = mean_negative_score_mse
        real_sampled_action_mse = best_negative_sampled_action_mse = mean_negative_sampled_action_mse = zero
    return {
        "idm_future_ranking_loss": ranking_loss,
        "idm_future_ranking_real_endpoint_mse": real_endpoint_mse,
        "idm_future_ranking_best_negative_endpoint_mse": best_negative_endpoint_mse,
        "idm_future_ranking_mean_negative_endpoint_mse": mean_negative_endpoint_mse,
        "idm_future_ranking_real_sampled_action_mse": real_sampled_action_mse,
        "idm_future_ranking_best_negative_sampled_action_mse": best_negative_sampled_action_mse,
        "idm_future_ranking_mean_negative_sampled_action_mse": mean_negative_sampled_action_mse,
        "idm_future_ranking_rank_accuracy": rank_accuracy,
        "idm_future_ranking_negative_count": valid_negative_count.mean(),
        "idm_future_ranking_singleton_fraction": singleton_indicator,
        "idm_future_ranking_same_task_valid_fraction": same_task_metrics["same_task_valid_fraction"],
        "idm_future_ranking_same_task_no_donor_fraction": same_task_metrics["same_task_no_donor_fraction"],
        "idm_future_ranking_same_task_different_episode_fraction": same_task_metrics[
            "same_task_different_episode_fraction"
        ],
        "idm_future_ranking_same_task_wan_latent_too_short_fraction": same_task_metrics[
            "same_task_wan_latent_too_short_fraction"
        ],
    }


def swapped_same_task_wan_vae_future_latents(
    wan_vae_latents: torch.Tensor,
    donor_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if wan_vae_latents.ndim != 5:
        raise ValueError(
            "Same-task future/action-delta Wan VAE latent swap requires rank 5 latents "
            f"with shape (B,C,T,H,W), got shape {tuple(wan_vae_latents.shape)}."
        )
    if donor_indices.ndim != 1 or donor_indices.shape[0] != wan_vae_latents.shape[0]:
        raise ValueError(
            f"donor_indices must have shape ({wan_vae_latents.shape[0]},), got {tuple(donor_indices.shape)}."
        )
    if wan_vae_latents.shape[2] < 2:
        return wan_vae_latents.clone(), torch.zeros(
            wan_vae_latents.shape[0],
            device=wan_vae_latents.device,
            dtype=torch.bool,
        )
    swapped = wan_vae_latents.clone()
    swapped[:, :, 1:] = wan_vae_latents[donor_indices, :, 1:]
    return swapped, torch.ones(wan_vae_latents.shape[0], device=wan_vae_latents.device, dtype=torch.bool)


def _zero_same_task_future_delta_losses(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.new_zeros(())
    return {
        "idm_same_task_future_delta_loss": zero,
        "idm_same_task_future_delta_donor_fraction": zero,
        "idm_same_task_future_delta_no_donor_fraction": zero,
        "idm_same_task_future_delta_different_episode_fraction": zero,
        "idm_same_task_future_delta_action_delta_mse": zero,
        "idm_same_task_future_delta_valid_action_fraction": zero,
        "idm_same_task_future_delta_wan_latent_too_short_fraction": zero,
        "idm_same_task_future_delta_state_distance": zero,
        "idm_same_task_future_delta_effective_donor_fraction": zero,
        "idm_same_task_future_delta_min_action_delta_filtered_fraction": zero,
        "idm_same_task_future_delta_max_state_distance_filtered_fraction": zero,
        "idm_same_task_future_delta_prediction_delta_mse": zero,
        "idm_same_task_future_delta_delta_cosine": zero,
    }


def compute_same_task_future_delta_loss(
    idm: nn.Module,
    batch: dict[str, torch.Tensor],
    target_action: torch.Tensor,
    idm_state: torch.Tensor,
    *,
    action_normalizer: ActionNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
    time_value: float,
    min_same_episode_frame_gap: int,
    max_state_distance: float | None = None,
    min_action_delta_mse: float = 0.0,
) -> dict[str, torch.Tensor]:
    if not idm_uses_flow_matching(idm):
        raise ValueError("Same-task future/action-delta IDM loss is supported only for flow_transformer IDMs.")
    if not 0.0 <= time_value < 1.0:
        raise ValueError(f"same_task_future_delta time_value must be in [0, 1), got {time_value}.")

    if max_state_distance is not None and max_state_distance < 0.0:
        raise ValueError(
            f"same_task_future_delta max_state_distance must be non-negative or None, got {max_state_distance}."
        )
    if min_action_delta_mse < 0.0:
        raise ValueError(
            f"same_task_future_delta min_action_delta_mse must be non-negative, got {min_action_delta_mse}."
        )

    donor_distance_state = idm_state
    if state_normalizer is not None:
        donor_distance_state = state_normalizer.normalize(batch["state"])

    donor_sample = sample_same_task_future_delta_donors(
        batch,
        min_same_episode_frame_gap=min_same_episode_frame_gap,
        state=donor_distance_state,
        max_state_distance=max_state_distance,
        action=target_action,
        min_action_delta_mse=min_action_delta_mse,
    )
    donor_indices = donor_sample.donor_indices
    has_donor = donor_sample.has_donor
    donor_action = target_action[donor_indices]
    donor_action_mask = batch["action_mask"][donor_indices]
    target_delta = donor_action - target_action
    combined_action_mask = batch["action_mask"] * donor_action_mask

    real_wan_vae_latents = batch.get("wan_vae_latents")
    swapped_wan_vae_latents = None
    latent_valid = torch.ones_like(has_donor)
    if real_wan_vae_latents is not None:
        swapped_wan_vae_latents, latent_valid = swapped_same_task_wan_vae_future_latents(
            real_wan_vae_latents,
            donor_indices,
        )

    effective_donor = has_donor & latent_valid
    valid_action_mask = combined_action_mask * effective_donor.to(dtype=combined_action_mask.dtype).unsqueeze(-1)
    valid_action_count = valid_action_mask.sum()

    donor_fraction = has_donor.to(dtype=target_action.dtype).mean()
    effective_donor_fraction = effective_donor.to(dtype=target_action.dtype).mean()
    no_donor_fraction = (~has_donor).to(dtype=target_action.dtype).mean()
    different_episode_fraction = donor_sample.used_different_episode.to(dtype=target_action.dtype).mean()
    latent_too_short_fraction = (~latent_valid).to(dtype=target_action.dtype).mean()
    state_distance = (donor_sample.state_distance * effective_donor.to(dtype=target_action.dtype)).sum() / (
        effective_donor.to(dtype=target_action.dtype).sum().clamp_min(1.0)
    )
    min_action_delta_filtered_fraction = donor_sample.min_action_delta_filtered.to(dtype=target_action.dtype).mean()
    max_state_distance_filtered_fraction = donor_sample.max_state_distance_filtered.to(dtype=target_action.dtype).mean()
    action_delta_mse = (target_delta.square() * valid_action_mask.unsqueeze(-1)).sum() / (
        valid_action_mask.unsqueeze(-1).expand_as(target_delta).sum().clamp_min(1.0)
    )
    valid_action_fraction = valid_action_mask.to(dtype=target_action.dtype).mean()

    if not bool(effective_donor.any()) or float(valid_action_count.detach().cpu()) <= 0.0:
        return {
            **_zero_same_task_future_delta_losses(target_action),
            "idm_same_task_future_delta_donor_fraction": donor_fraction,
            "idm_same_task_future_delta_no_donor_fraction": no_donor_fraction,
            "idm_same_task_future_delta_different_episode_fraction": different_episode_fraction,
            "idm_same_task_future_delta_action_delta_mse": action_delta_mse,
            "idm_same_task_future_delta_valid_action_fraction": valid_action_fraction,
            "idm_same_task_future_delta_wan_latent_too_short_fraction": latent_too_short_fraction,
            "idm_same_task_future_delta_state_distance": state_distance,
            "idm_same_task_future_delta_effective_donor_fraction": effective_donor_fraction,
            "idm_same_task_future_delta_min_action_delta_filtered_fraction": min_action_delta_filtered_fraction,
            "idm_same_task_future_delta_max_state_distance_filtered_fraction": max_state_distance_filtered_fraction,
        }

    history_kwargs = idm_history_kwargs(
        batch,
        idm=idm,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
    )
    shared_noise = torch.randn_like(target_action)
    anchor_endpoint = _flow_teacher_forced_endpoint_prediction(
        idm,
        batch["current_images"],
        batch["future_images"],
        idm_state,
        target_action,
        shared_noise,
        wan_vae_latents=real_wan_vae_latents,
        time_value=time_value,
        **history_kwargs,
    )
    swapped_endpoint = _flow_teacher_forced_endpoint_prediction(
        idm,
        batch["current_images"],
        batch["future_images"][donor_indices],
        idm_state,
        target_action,
        shared_noise,
        wan_vae_latents=swapped_wan_vae_latents,
        time_value=time_value,
        **history_kwargs,
    )

    prediction_delta = swapped_endpoint - anchor_endpoint.detach()
    mask = valid_action_mask
    while mask.ndim < prediction_delta.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=prediction_delta.device, dtype=prediction_delta.dtype)
    expanded_mask = mask.expand_as(prediction_delta)
    expanded_target_delta = target_delta.expand_as(prediction_delta)
    loss = ((prediction_delta - expanded_target_delta).square() * expanded_mask).sum() / expanded_mask.sum().clamp_min(
        1.0
    )
    prediction_delta_mse = (prediction_delta.square() * mask).sum() / mask.expand_as(prediction_delta).sum().clamp_min(
        1.0
    )
    masked_prediction_delta = prediction_delta * expanded_mask
    masked_target_delta = expanded_target_delta * expanded_mask
    delta_cosine = (masked_prediction_delta * masked_target_delta).sum() / (
        masked_prediction_delta.square().sum().sqrt() * masked_target_delta.square().sum().sqrt()
    ).clamp_min(1e-12)
    delta_cosine = delta_cosine.clamp(min=-1.0, max=1.0)
    return {
        "idm_same_task_future_delta_loss": loss,
        "idm_same_task_future_delta_donor_fraction": donor_fraction,
        "idm_same_task_future_delta_effective_donor_fraction": effective_donor_fraction,
        "idm_same_task_future_delta_no_donor_fraction": no_donor_fraction,
        "idm_same_task_future_delta_different_episode_fraction": different_episode_fraction,
        "idm_same_task_future_delta_action_delta_mse": action_delta_mse,
        "idm_same_task_future_delta_valid_action_fraction": valid_action_fraction,
        "idm_same_task_future_delta_wan_latent_too_short_fraction": latent_too_short_fraction,
        "idm_same_task_future_delta_state_distance": state_distance,
        "idm_same_task_future_delta_min_action_delta_filtered_fraction": min_action_delta_filtered_fraction,
        "idm_same_task_future_delta_max_state_distance_filtered_fraction": max_state_distance_filtered_fraction,
        "idm_same_task_future_delta_prediction_delta_mse": prediction_delta_mse,
        "idm_same_task_future_delta_delta_cosine": delta_cosine,
    }


def compute_context_action_loss(
    idm: nn.Module,
    batch: dict[str, torch.Tensor],
    target_action: torch.Tensor,
    idm_state: torch.Tensor,
    *,
    action_normalizer: ActionNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
) -> dict[str, torch.Tensor]:
    if not idm_uses_flow_matching(idm):
        raise ValueError("Context-to-action IDM loss is supported only for flow_transformer IDMs.")

    kwargs = {}
    if "wan_vae_latents" in batch:
        kwargs["wan_vae_latents"] = batch["wan_vae_latents"]
    kwargs.update(
        idm_history_kwargs(
            batch,
            idm=idm,
            action_normalizer=action_normalizer,
            state_normalizer=state_normalizer,
        )
    )
    outputs = unwrap_model(idm).context_action_loss(
        batch["current_images"],
        batch["future_images"],
        idm_state,
        target_action,
        batch["action_mask"],
        **kwargs,
    )
    return {
        "idm_context_action_loss": outputs["loss"],
    }


def compute_idm_losses(
    idm: nn.Module,
    batch: dict[str, torch.Tensor],
    action_normalizer: ActionNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
    *,
    context_action_loss_weight: float = 0.0,
    future_contrastive_weight: float = 0.0,
    future_contrastive_margin: float = 0.1,
    future_ranking_weight: float = 0.0,
    future_ranking_temperature: float = 0.1,
    future_ranking_noise_std: float = 1.0,
    future_ranking_repeated_current_negative: bool = False,
    future_ranking_shuffled_future_negative: bool = False,
    future_ranking_noisy_future_negative: bool = False,
    future_ranking_zero_future_negative: bool = False,
    future_ranking_same_task_negative: bool = False,
    future_ranking_score_mode: IdmFutureRankingScoreMode = "teacher_forced_endpoint",
    same_task_future_delta_weight: float = 0.0,
    same_task_future_delta_time_value: float = 0.5,
    same_task_future_delta_min_frame_gap: int = 0,
    same_task_future_delta_max_state_distance: float | None = None,
    same_task_future_delta_min_action_delta_mse: float = 0.0,
) -> dict[str, torch.Tensor]:
    if context_action_loss_weight < 0.0:
        raise ValueError(f"context_action_loss_weight must be non-negative, got {context_action_loss_weight}.")
    if future_contrastive_weight < 0.0:
        raise ValueError(f"future_contrastive_weight must be non-negative, got {future_contrastive_weight}.")
    if future_contrastive_margin < 0.0:
        raise ValueError(f"future_contrastive_margin must be non-negative, got {future_contrastive_margin}.")
    if future_ranking_weight < 0.0:
        raise ValueError(f"future_ranking_weight must be non-negative, got {future_ranking_weight}.")
    if future_ranking_temperature <= 0.0:
        raise ValueError(f"future_ranking_temperature must be positive, got {future_ranking_temperature}.")
    if future_ranking_noise_std < 0.0:
        raise ValueError(f"future_ranking_noise_std must be non-negative, got {future_ranking_noise_std}.")
    if same_task_future_delta_weight < 0.0:
        raise ValueError(f"same_task_future_delta_weight must be non-negative, got {same_task_future_delta_weight}.")
    if not 0.0 <= same_task_future_delta_time_value < 1.0:
        raise ValueError(
            f"same_task_future_delta_time_value must be in [0, 1), got {same_task_future_delta_time_value}."
        )
    if same_task_future_delta_min_frame_gap < 0:
        raise ValueError(
            f"same_task_future_delta_min_frame_gap must be non-negative, got {same_task_future_delta_min_frame_gap}."
        )
    if same_task_future_delta_max_state_distance is not None and same_task_future_delta_max_state_distance < 0.0:
        raise ValueError(
            "same_task_future_delta_max_state_distance must be non-negative or None, "
            f"got {same_task_future_delta_max_state_distance}."
        )
    if same_task_future_delta_min_action_delta_mse < 0.0:
        raise ValueError(
            "same_task_future_delta_min_action_delta_mse must be non-negative, "
            f"got {same_task_future_delta_min_action_delta_mse}."
        )
    if action_normalizer is None:
        target_action = batch["action_chunk"]
    else:
        target_action = action_normalizer.normalize(batch["action_chunk"])
    idm_state = normalize_state_for_idm(idm, batch["state"], state_normalizer)
    if idm_uses_flow_matching(idm):
        flow_kwargs = idm_history_kwargs(
            batch,
            idm=idm,
            action_normalizer=action_normalizer,
            state_normalizer=state_normalizer,
        )
        if "wan_vae_latents" in batch:
            flow_kwargs["wan_vae_latents"] = batch["wan_vae_latents"]
        flow_outputs = idm(
            batch["current_images"],
            batch["future_images"],
            idm_state,
            batch["task_id"],
            target_action=target_action,
            action_mask=batch["action_mask"],
            mode="loss",
            **flow_kwargs,
        )
        idm_loss = flow_outputs["loss"]
        action_smoothness_loss = idm_loss.new_zeros(())
        endpoint_consistency_loss = flow_outputs.get("endpoint_consistency_loss", idm_loss.new_zeros(()))
        zero_start_endpoint_loss = flow_outputs.get("zero_start_endpoint_loss", idm_loss.new_zeros(()))
        sampled_action_loss = flow_outputs.get("sampled_action_loss", idm_loss.new_zeros(()))
        context_action_losses = {
            "idm_context_action_loss": idm_loss.new_zeros(()),
        }
        if context_action_loss_weight > 0.0:
            context_action_losses = compute_context_action_loss(
                idm,
                batch,
                target_action,
                idm_state,
                action_normalizer=action_normalizer,
                state_normalizer=state_normalizer,
            )
        contrastive_losses = {
            "idm_future_contrastive_loss": idm_loss.new_zeros(()),
            "idm_future_contrastive_real_endpoint_mse": idm_loss.new_zeros(()),
            "idm_future_contrastive_corrupted_endpoint_mse": idm_loss.new_zeros(()),
            "idm_future_contrastive_singleton_fraction": idm_loss.new_zeros(()),
        }
        ranking_losses = {
            "idm_future_ranking_loss": idm_loss.new_zeros(()),
            "idm_future_ranking_real_endpoint_mse": idm_loss.new_zeros(()),
            "idm_future_ranking_best_negative_endpoint_mse": idm_loss.new_zeros(()),
            "idm_future_ranking_mean_negative_endpoint_mse": idm_loss.new_zeros(()),
            "idm_future_ranking_real_sampled_action_mse": idm_loss.new_zeros(()),
            "idm_future_ranking_best_negative_sampled_action_mse": idm_loss.new_zeros(()),
            "idm_future_ranking_mean_negative_sampled_action_mse": idm_loss.new_zeros(()),
            "idm_future_ranking_rank_accuracy": idm_loss.new_zeros(()),
            "idm_future_ranking_negative_count": idm_loss.new_zeros(()),
            "idm_future_ranking_singleton_fraction": idm_loss.new_zeros(()),
            "idm_future_ranking_same_task_valid_fraction": idm_loss.new_zeros(()),
            "idm_future_ranking_same_task_no_donor_fraction": idm_loss.new_zeros(()),
            "idm_future_ranking_same_task_different_episode_fraction": idm_loss.new_zeros(()),
            "idm_future_ranking_same_task_wan_latent_too_short_fraction": idm_loss.new_zeros(()),
        }
        same_task_future_delta_losses = _zero_same_task_future_delta_losses(idm_loss)
        if future_contrastive_weight > 0.0:
            contrastive_losses = compute_future_contrastive_loss(
                idm,
                batch,
                target_action,
                idm_state,
                action_normalizer=action_normalizer,
                state_normalizer=state_normalizer,
                margin=future_contrastive_margin,
            )
        if future_ranking_weight > 0.0:
            ranking_losses = compute_future_ranking_loss(
                idm,
                batch,
                target_action,
                idm_state,
                action_normalizer=action_normalizer,
                state_normalizer=state_normalizer,
                temperature=future_ranking_temperature,
                noise_std=future_ranking_noise_std,
                repeated_current_negative=future_ranking_repeated_current_negative,
                shuffled_future_negative=future_ranking_shuffled_future_negative,
                noisy_future_negative=future_ranking_noisy_future_negative,
                zero_future_negative=future_ranking_zero_future_negative,
                same_task_negative=future_ranking_same_task_negative,
                min_same_episode_frame_gap=same_task_future_delta_min_frame_gap,
                score_mode=future_ranking_score_mode,
            )
        if same_task_future_delta_weight > 0.0:
            same_task_future_delta_losses = compute_same_task_future_delta_loss(
                idm,
                batch,
                target_action,
                idm_state,
                action_normalizer=action_normalizer,
                state_normalizer=state_normalizer,
                time_value=same_task_future_delta_time_value,
                min_same_episode_frame_gap=same_task_future_delta_min_frame_gap,
                max_state_distance=same_task_future_delta_max_state_distance,
                min_action_delta_mse=same_task_future_delta_min_action_delta_mse,
            )
    else:
        if context_action_loss_weight > 0.0:
            raise ValueError("Context-to-action IDM loss is supported only for flow_transformer IDMs.")
        if future_contrastive_weight > 0.0:
            raise ValueError("Future contrastive IDM loss is supported only for flow_transformer IDMs.")
        if future_ranking_weight > 0.0:
            raise ValueError("Future ranking IDM loss is supported only for flow_transformer IDMs.")
        if same_task_future_delta_weight > 0.0:
            raise ValueError("Same-task future/action-delta IDM loss is supported only for flow_transformer IDMs.")
        model_action = idm(
            batch["current_images"],
            batch["future_images"],
            idm_state,
            batch["task_id"],
            **idm_history_kwargs(
                batch,
                idm=idm,
                action_normalizer=action_normalizer,
                state_normalizer=state_normalizer,
            ),
        )
        idm_loss = masked_smooth_l1(model_action, target_action, batch["action_mask"])
        if action_normalizer is None:
            predicted_action = model_action
        else:
            action_normalizer = action_normalizer.to(model_action.device)
            predicted_action = action_normalizer.denormalize(model_action)
        action_smoothness_loss = action_smoothness(predicted_action, batch["action_mask"])
        contrastive_losses = {
            "idm_future_contrastive_loss": idm_loss.new_zeros(()),
            "idm_future_contrastive_real_endpoint_mse": idm_loss.new_zeros(()),
            "idm_future_contrastive_corrupted_endpoint_mse": idm_loss.new_zeros(()),
            "idm_future_contrastive_singleton_fraction": idm_loss.new_zeros(()),
        }
        ranking_losses = {
            "idm_future_ranking_loss": idm_loss.new_zeros(()),
            "idm_future_ranking_real_endpoint_mse": idm_loss.new_zeros(()),
            "idm_future_ranking_best_negative_endpoint_mse": idm_loss.new_zeros(()),
            "idm_future_ranking_mean_negative_endpoint_mse": idm_loss.new_zeros(()),
            "idm_future_ranking_real_sampled_action_mse": idm_loss.new_zeros(()),
            "idm_future_ranking_best_negative_sampled_action_mse": idm_loss.new_zeros(()),
            "idm_future_ranking_mean_negative_sampled_action_mse": idm_loss.new_zeros(()),
            "idm_future_ranking_rank_accuracy": idm_loss.new_zeros(()),
            "idm_future_ranking_negative_count": idm_loss.new_zeros(()),
            "idm_future_ranking_singleton_fraction": idm_loss.new_zeros(()),
            "idm_future_ranking_same_task_valid_fraction": idm_loss.new_zeros(()),
            "idm_future_ranking_same_task_no_donor_fraction": idm_loss.new_zeros(()),
            "idm_future_ranking_same_task_different_episode_fraction": idm_loss.new_zeros(()),
            "idm_future_ranking_same_task_wan_latent_too_short_fraction": idm_loss.new_zeros(()),
        }
        same_task_future_delta_losses = _zero_same_task_future_delta_losses(idm_loss)
        context_action_losses = {
            "idm_context_action_loss": idm_loss.new_zeros(()),
        }
        endpoint_consistency_loss = idm_loss.new_zeros(())
        zero_start_endpoint_loss = idm_loss.new_zeros(())
        sampled_action_loss = idm_loss.new_zeros(())
    return {
        "idm_loss": idm_loss,
        "action_smoothness_loss": action_smoothness_loss,
        "idm_endpoint_consistency_loss": endpoint_consistency_loss,
        "idm_zero_start_endpoint_loss": zero_start_endpoint_loss,
        "idm_sampled_action_loss": sampled_action_loss,
        **context_action_losses,
        **contrastive_losses,
        **ranking_losses,
        **same_task_future_delta_losses,
    }


def context_action_loss_weight_for_epoch(
    base_weight: float,
    epoch_index: int,
    warmup_epochs: int | None = None,
) -> float:
    if base_weight < 0.0:
        raise ValueError(f"context_action_loss_weight must be non-negative, got {base_weight}.")
    if epoch_index < 0:
        raise ValueError(f"epoch_index must be non-negative, got {epoch_index}.")
    if warmup_epochs is None:
        return float(base_weight)
    if warmup_epochs < 0:
        raise ValueError(f"idm_context_action_warmup_epochs must be non-negative, got {warmup_epochs}.")
    return float(base_weight) if epoch_index < warmup_epochs else 0.0


def future_ranking_weight_for_epoch(
    base_weight: float,
    epoch_index: int,
    start_epoch: int | None = None,
    ramp_epochs: int = 0,
) -> float:
    if base_weight < 0.0:
        raise ValueError(f"idm_future_ranking_weight must be non-negative, got {base_weight}.")
    if epoch_index < 0:
        raise ValueError(f"epoch_index must be non-negative, got {epoch_index}.")
    if start_epoch is not None and start_epoch < 0:
        raise ValueError(f"idm_future_ranking_start_epoch must be non-negative, got {start_epoch}.")
    if ramp_epochs < 0:
        raise ValueError(f"idm_future_ranking_ramp_epochs must be non-negative, got {ramp_epochs}.")
    if start_epoch is None and ramp_epochs == 0:
        return float(base_weight)

    effective_start_epoch = 0 if start_epoch is None else start_epoch
    if epoch_index < effective_start_epoch:
        return 0.0
    if ramp_epochs == 0:
        return float(base_weight)
    progress = min((epoch_index - effective_start_epoch + 1) / ramp_epochs, 1.0)
    return float(base_weight) * progress


def apply_state_dropout(batch: dict[str, torch.Tensor], dropout: float) -> dict[str, torch.Tensor]:
    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"idm_state_dropout must be between 0 and 1, got {dropout}.")
    if dropout == 0.0:
        return batch
    state = batch["state"]
    keep_mask = (torch.rand(state.shape[0], 1, device=state.device) >= dropout).to(dtype=state.dtype)
    return {**batch, "state": state * keep_mask}


def apply_future_augmentation(
    batch: dict[str, torch.Tensor], noise_std: float, frame_dropout: float
) -> dict[str, torch.Tensor]:
    if noise_std < 0.0:
        raise ValueError(f"idm_future_noise_std must be non-negative, got {noise_std}.")
    if not 0.0 <= frame_dropout <= 1.0:
        raise ValueError(f"idm_future_frame_dropout must be between 0 and 1, got {frame_dropout}.")
    if noise_std == 0.0 and frame_dropout == 0.0:
        return batch
    if "wan_vae_latents" in batch:
        raise ValueError("idm_future_noise_std/idm_future_frame_dropout are not supported with cached Wan VAE latents.")

    future_images = batch["future_images"]
    if frame_dropout > 0.0:
        replacement = batch["current_images"].unsqueeze(1).expand_as(future_images)
        drop_mask = (
            torch.rand(
                future_images.shape[0],
                future_images.shape[1],
                1,
                1,
                1,
                1,
                device=future_images.device,
            )
            < frame_dropout
        )
        future_images = torch.where(drop_mask, replacement, future_images)
    if noise_std > 0.0:
        future_images = (future_images + torch.randn_like(future_images) * noise_std).clamp(0.0, 1.0)
    return {**batch, "future_images": future_images}


def apply_current_conditioning_dropout(
    batch: dict[str, torch.Tensor],
    current_frame_dropout: float,
    wan_vae_current_latent_dropout: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= current_frame_dropout <= 1.0:
        raise ValueError(f"idm_current_frame_dropout must be in [0, 1], got {current_frame_dropout}.")
    if not 0.0 <= wan_vae_current_latent_dropout <= 1.0:
        raise ValueError(f"idm_wan_vae_current_latent_dropout must be in [0, 1], got {wan_vae_current_latent_dropout}.")
    if current_frame_dropout == 0.0 and wan_vae_current_latent_dropout == 0.0:
        return batch

    augmented = dict(batch)
    if current_frame_dropout > 0.0:
        current_images = batch["current_images"]
        sample_mask = (
            torch.rand(
                (current_images.shape[0], *([1] * (current_images.ndim - 1))),
                device=current_images.device,
            )
            < current_frame_dropout
        )
        augmented["current_images"] = torch.where(sample_mask, torch.zeros_like(current_images), current_images)

    if wan_vae_current_latent_dropout > 0.0 and "wan_vae_latents" in batch:
        latents = batch["wan_vae_latents"]
        if latents.ndim != 5:
            raise ValueError(
                "idm_wan_vae_current_latent_dropout requires batched Wan VAE latents "
                f"with rank 5 shape (B,C,T,H,W), got shape {tuple(latents.shape)}."
            )
        if latents.shape[2] < 1:
            raise ValueError(
                "idm_wan_vae_current_latent_dropout requires at least one latent time step, "
                f"got shape {tuple(latents.shape)}."
            )
        sample_mask = torch.rand((latents.shape[0], 1, 1, 1), device=latents.device) < wan_vae_current_latent_dropout
        dropped_latents = latents.clone()
        dropped_latents[:, :, 0] = torch.where(
            sample_mask,
            torch.zeros_like(dropped_latents[:, :, 0]),
            dropped_latents[:, :, 0],
        )
        augmented["wan_vae_latents"] = dropped_latents

    return augmented


def apply_wan_vae_latent_noise(
    batch: dict[str, torch.Tensor],
    prob: float,
    s_min: float,
    s_max: float,
    time_mode: WanVaeLatentNoiseTimeMode = "all",
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"idm_wan_vae_latent_noise_prob must be in [0, 1], got {prob}.")
    if not 0.0 <= s_min <= 1.0:
        raise ValueError(f"idm_wan_vae_latent_noise_s_min must be in [0, 1], got {s_min}.")
    if not 0.0 <= s_max <= 1.0:
        raise ValueError(f"idm_wan_vae_latent_noise_s_max must be in [0, 1], got {s_max}.")
    if s_min > s_max:
        raise ValueError(
            f"idm_wan_vae_latent_noise_s_min must be <= idm_wan_vae_latent_noise_s_max, got {s_min} > {s_max}."
        )
    if time_mode not in ("all", "future_only"):
        raise ValueError(
            f"idm_wan_vae_latent_noise_time_mode must be one of {{'all', 'future_only'}}, got {time_mode!r}."
        )
    stats = {
        "augmented_count": 0.0,
        "sample_count": 0.0,
        "s_sum": 0.0,
    }
    if "wan_vae_latents" not in batch:
        return batch, stats
    latents = batch["wan_vae_latents"]
    batch_size = int(latents.shape[0])
    stats["sample_count"] = float(batch_size)
    if prob == 0.0:
        return batch, stats
    if time_mode == "future_only":
        if latents.ndim != 5:
            raise ValueError(
                "idm_wan_vae_latent_noise_time_mode='future_only' requires batched Wan VAE latents "
                f"with rank 5 shape (B,C,T,H,W), got shape {tuple(latents.shape)}."
            )
        if latents.shape[2] < 2:
            raise ValueError(
                "idm_wan_vae_latent_noise_time_mode='future_only' requires at least two latent time steps "
                f"(T >= 2), got shape {tuple(latents.shape)}."
            )

    sample_shape = (batch_size, *([1] * (latents.ndim - 1)))
    augment_mask = torch.rand(sample_shape, device=latents.device) < prob
    sampled_s = torch.empty(sample_shape, device=latents.device, dtype=latents.dtype).uniform_(s_min, s_max)
    noise = torch.randn_like(latents)
    augmented_latents = (1.0 - sampled_s) * noise + sampled_s * latents
    if time_mode == "future_only":
        time_mask = torch.arange(latents.shape[2], device=latents.device).view(1, 1, -1, 1, 1) >= 1
        latents = torch.where(augment_mask & time_mask, augmented_latents, latents)
    else:
        latents = torch.where(augment_mask, augmented_latents, latents)

    augmented_count = float(augment_mask.sum().detach().cpu())
    stats["augmented_count"] = augmented_count
    if augmented_count > 0.0:
        stats["s_sum"] = float(sampled_s.masked_select(augment_mask).sum().detach().cpu())
    return {**batch, "wan_vae_latents": latents}, stats


def train_idm_one_epoch(
    idm: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
    action_normalizer: ActionNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
    *,
    context_action_loss_weight: float | None = None,
    future_ranking_weight: float | None = None,
) -> dict[str, float]:
    idm.train()
    active_context_action_loss_weight = (
        config.idm_context_action_loss_weight if context_action_loss_weight is None else context_action_loss_weight
    )
    active_future_ranking_weight = (
        config.idm_future_ranking_weight if future_ranking_weight is None else future_ranking_weight
    )
    if active_context_action_loss_weight < 0.0:
        raise ValueError(f"context_action_loss_weight must be non-negative, got {active_context_action_loss_weight}.")
    if active_future_ranking_weight < 0.0:
        raise ValueError(f"future_ranking_weight must be non-negative, got {active_future_ranking_weight}.")
    same_task_future_delta_min_frame_gap = max(
        int(config.dataset.frame_delta) * int(config.dataset.num_future_frames),
        int(config.dataset.action_horizon) + int(config.dataset.idm_history_length),
    )
    totals = {
        "loss": 0.0,
        "idm_loss": 0.0,
        "action_smoothness_loss": 0.0,
        "idm_endpoint_consistency_loss": 0.0,
        "idm_zero_start_endpoint_loss": 0.0,
        "idm_sampled_action_loss": 0.0,
        "idm_context_action_loss": 0.0,
        "idm_future_contrastive_loss": 0.0,
        "idm_future_contrastive_real_endpoint_mse": 0.0,
        "idm_future_contrastive_corrupted_endpoint_mse": 0.0,
        "idm_future_contrastive_singleton_fraction": 0.0,
        "idm_future_ranking_loss": 0.0,
        "idm_future_ranking_real_endpoint_mse": 0.0,
        "idm_future_ranking_best_negative_endpoint_mse": 0.0,
        "idm_future_ranking_mean_negative_endpoint_mse": 0.0,
        "idm_future_ranking_real_sampled_action_mse": 0.0,
        "idm_future_ranking_best_negative_sampled_action_mse": 0.0,
        "idm_future_ranking_mean_negative_sampled_action_mse": 0.0,
        "idm_future_ranking_rank_accuracy": 0.0,
        "idm_future_ranking_negative_count": 0.0,
        "idm_future_ranking_singleton_fraction": 0.0,
        "idm_future_ranking_same_task_valid_fraction": 0.0,
        "idm_future_ranking_same_task_no_donor_fraction": 0.0,
        "idm_future_ranking_same_task_different_episode_fraction": 0.0,
        "idm_future_ranking_same_task_wan_latent_too_short_fraction": 0.0,
        "idm_same_task_future_delta_loss": 0.0,
        "idm_same_task_future_delta_donor_fraction": 0.0,
        "idm_same_task_future_delta_no_donor_fraction": 0.0,
        "idm_same_task_future_delta_different_episode_fraction": 0.0,
        "idm_same_task_future_delta_action_delta_mse": 0.0,
        "idm_same_task_future_delta_valid_action_fraction": 0.0,
        "idm_same_task_future_delta_wan_latent_too_short_fraction": 0.0,
        "idm_same_task_future_delta_state_distance": 0.0,
        "idm_same_task_future_delta_effective_donor_fraction": 0.0,
        "idm_same_task_future_delta_min_action_delta_filtered_fraction": 0.0,
        "idm_same_task_future_delta_max_state_distance_filtered_fraction": 0.0,
        "idm_same_task_future_delta_prediction_delta_mse": 0.0,
        "idm_same_task_future_delta_delta_cosine": 0.0,
    }
    count = 0
    for batch in tqdm(loader, desc="train-idm", leave=False):
        batch = to_device(batch, device)
        batch = apply_state_dropout(batch, config.idm_state_dropout)
        batch = apply_future_augmentation(batch, config.idm_future_noise_std, config.idm_future_frame_dropout)
        batch, latent_noise_stats = apply_wan_vae_latent_noise(
            batch,
            config.idm_wan_vae_latent_noise_prob,
            config.idm_wan_vae_latent_noise_s_min,
            config.idm_wan_vae_latent_noise_s_max,
            config.idm_wan_vae_latent_noise_time_mode,
        )
        batch = apply_current_conditioning_dropout(
            batch,
            config.idm_current_frame_dropout,
            config.idm_wan_vae_current_latent_dropout,
        )
        losses = compute_idm_losses(
            idm,
            batch,
            action_normalizer,
            state_normalizer,
            context_action_loss_weight=active_context_action_loss_weight,
            future_contrastive_weight=config.idm_future_contrastive_weight,
            future_contrastive_margin=config.idm_future_contrastive_margin,
            future_ranking_weight=active_future_ranking_weight,
            future_ranking_temperature=config.idm_future_ranking_temperature,
            future_ranking_noise_std=config.idm_future_ranking_noise_std,
            future_ranking_repeated_current_negative=config.idm_future_ranking_repeated_current_negative,
            future_ranking_shuffled_future_negative=config.idm_future_ranking_shuffled_future_negative,
            future_ranking_noisy_future_negative=config.idm_future_ranking_noisy_future_negative,
            future_ranking_zero_future_negative=config.idm_future_ranking_zero_future_negative,
            future_ranking_same_task_negative=config.idm_future_ranking_same_task_negative,
            future_ranking_score_mode=config.idm_future_ranking_score_mode,
            same_task_future_delta_weight=config.idm_same_task_future_delta_weight,
            same_task_future_delta_time_value=config.idm_same_task_future_delta_time_value,
            same_task_future_delta_min_frame_gap=same_task_future_delta_min_frame_gap,
            same_task_future_delta_max_state_distance=config.idm_same_task_future_delta_max_state_distance,
            same_task_future_delta_min_action_delta_mse=config.idm_same_task_future_delta_min_action_delta_mse,
        )
        loss = (
            config.idm_loss_weight * losses["idm_loss"]
            + config.action_smoothness_weight * losses["action_smoothness_loss"]
            + active_context_action_loss_weight * losses["idm_context_action_loss"]
            + config.idm_future_contrastive_weight * losses["idm_future_contrastive_loss"]
            + active_future_ranking_weight * losses["idm_future_ranking_loss"]
            + config.idm_same_task_future_delta_weight * losses["idm_same_task_future_delta_loss"]
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(idm.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = int(batch["current_images"].shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["idm_loss"] += float(losses["idm_loss"].detach().cpu()) * batch_size
        totals["action_smoothness_loss"] += float(losses["action_smoothness_loss"].detach().cpu()) * batch_size
        totals["idm_endpoint_consistency_loss"] += (
            float(losses["idm_endpoint_consistency_loss"].detach().cpu()) * batch_size
        )
        totals["idm_zero_start_endpoint_loss"] += (
            float(losses["idm_zero_start_endpoint_loss"].detach().cpu()) * batch_size
        )
        totals["idm_sampled_action_loss"] += float(losses["idm_sampled_action_loss"].detach().cpu()) * batch_size
        totals["idm_context_action_loss"] += float(losses["idm_context_action_loss"].detach().cpu()) * batch_size
        totals["idm_future_contrastive_loss"] += (
            float(losses["idm_future_contrastive_loss"].detach().cpu()) * batch_size
        )
        totals["idm_future_contrastive_real_endpoint_mse"] += (
            float(losses["idm_future_contrastive_real_endpoint_mse"].detach().cpu()) * batch_size
        )
        totals["idm_future_contrastive_corrupted_endpoint_mse"] += (
            float(losses["idm_future_contrastive_corrupted_endpoint_mse"].detach().cpu()) * batch_size
        )
        totals["idm_future_contrastive_singleton_fraction"] += (
            float(losses["idm_future_contrastive_singleton_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_future_ranking_loss"] += float(losses["idm_future_ranking_loss"].detach().cpu()) * batch_size
        totals["idm_future_ranking_real_endpoint_mse"] += (
            float(losses["idm_future_ranking_real_endpoint_mse"].detach().cpu()) * batch_size
        )
        totals["idm_future_ranking_best_negative_endpoint_mse"] += (
            float(losses["idm_future_ranking_best_negative_endpoint_mse"].detach().cpu()) * batch_size
        )
        totals["idm_future_ranking_mean_negative_endpoint_mse"] += (
            float(losses["idm_future_ranking_mean_negative_endpoint_mse"].detach().cpu()) * batch_size
        )
        totals["idm_future_ranking_real_sampled_action_mse"] += (
            float(losses["idm_future_ranking_real_sampled_action_mse"].detach().cpu()) * batch_size
        )
        totals["idm_future_ranking_best_negative_sampled_action_mse"] += (
            float(losses["idm_future_ranking_best_negative_sampled_action_mse"].detach().cpu()) * batch_size
        )
        totals["idm_future_ranking_mean_negative_sampled_action_mse"] += (
            float(losses["idm_future_ranking_mean_negative_sampled_action_mse"].detach().cpu()) * batch_size
        )
        totals["idm_future_ranking_rank_accuracy"] += (
            float(losses["idm_future_ranking_rank_accuracy"].detach().cpu()) * batch_size
        )
        totals["idm_future_ranking_negative_count"] += (
            float(losses["idm_future_ranking_negative_count"].detach().cpu()) * batch_size
        )
        totals["idm_future_ranking_singleton_fraction"] += (
            float(losses["idm_future_ranking_singleton_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_future_ranking_same_task_valid_fraction"] += (
            float(losses["idm_future_ranking_same_task_valid_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_future_ranking_same_task_no_donor_fraction"] += (
            float(losses["idm_future_ranking_same_task_no_donor_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_future_ranking_same_task_different_episode_fraction"] += (
            float(losses["idm_future_ranking_same_task_different_episode_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_future_ranking_same_task_wan_latent_too_short_fraction"] += (
            float(losses["idm_future_ranking_same_task_wan_latent_too_short_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_same_task_future_delta_loss"] += (
            float(losses["idm_same_task_future_delta_loss"].detach().cpu()) * batch_size
        )
        totals["idm_same_task_future_delta_donor_fraction"] += (
            float(losses["idm_same_task_future_delta_donor_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_same_task_future_delta_no_donor_fraction"] += (
            float(losses["idm_same_task_future_delta_no_donor_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_same_task_future_delta_different_episode_fraction"] += (
            float(losses["idm_same_task_future_delta_different_episode_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_same_task_future_delta_action_delta_mse"] += (
            float(losses["idm_same_task_future_delta_action_delta_mse"].detach().cpu()) * batch_size
        )
        totals["idm_same_task_future_delta_valid_action_fraction"] += (
            float(losses["idm_same_task_future_delta_valid_action_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_same_task_future_delta_wan_latent_too_short_fraction"] += (
            float(losses["idm_same_task_future_delta_wan_latent_too_short_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_same_task_future_delta_state_distance"] += (
            float(losses["idm_same_task_future_delta_state_distance"].detach().cpu()) * batch_size
        )
        totals["idm_same_task_future_delta_effective_donor_fraction"] += (
            float(losses["idm_same_task_future_delta_effective_donor_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_same_task_future_delta_min_action_delta_filtered_fraction"] += (
            float(losses["idm_same_task_future_delta_min_action_delta_filtered_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_same_task_future_delta_max_state_distance_filtered_fraction"] += (
            float(losses["idm_same_task_future_delta_max_state_distance_filtered_fraction"].detach().cpu()) * batch_size
        )
        totals["idm_same_task_future_delta_prediction_delta_mse"] += (
            float(losses["idm_same_task_future_delta_prediction_delta_mse"].detach().cpu()) * batch_size
        )
        totals["idm_same_task_future_delta_delta_cosine"] += (
            float(losses["idm_same_task_future_delta_delta_cosine"].detach().cpu()) * batch_size
        )
        totals.setdefault("idm_wan_vae_latent_noise_augmented_count", 0.0)
        totals.setdefault("idm_wan_vae_latent_noise_sample_count", 0.0)
        totals.setdefault("idm_wan_vae_latent_noise_s_sum", 0.0)
        totals["idm_wan_vae_latent_noise_augmented_count"] += latent_noise_stats["augmented_count"]
        totals["idm_wan_vae_latent_noise_sample_count"] += latent_noise_stats["sample_count"]
        totals["idm_wan_vae_latent_noise_s_sum"] += latent_noise_stats["s_sum"]
        count += batch_size

    latent_augmented_count = totals.pop("idm_wan_vae_latent_noise_augmented_count", 0.0)
    latent_sample_count = totals.pop("idm_wan_vae_latent_noise_sample_count", 0.0)
    latent_s_sum = totals.pop("idm_wan_vae_latent_noise_s_sum", 0.0)
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    metrics["idm_context_action_loss_weight_active"] = float(active_context_action_loss_weight)
    metrics["idm_future_ranking_weight_active"] = float(active_future_ranking_weight)
    metrics["idm_same_task_future_delta_weight_active"] = float(config.idm_same_task_future_delta_weight)
    metrics["idm_wan_vae_latent_noise_fraction"] = (
        latent_augmented_count / latent_sample_count if latent_sample_count > 0.0 else 0.0
    )
    metrics["idm_wan_vae_latent_noise_s_mean"] = (
        latent_s_sum / latent_augmented_count if latent_augmented_count > 0.0 else 0.0
    )
    return metrics


@torch.no_grad()
def _evaluate_context_action_prediction(
    idm: nn.Module,
    current_images: torch.Tensor,
    future_images: torch.Tensor,
    state: torch.Tensor,
    *,
    wan_vae_latents: torch.Tensor | None = None,
    prev_state_history: torch.Tensor | None = None,
    prev_action_history: torch.Tensor | None = None,
    history_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not idm_uses_flow_matching(idm):
        raise ValueError("prediction_mode='context_action' is supported only for flow_transformer IDMs.")
    flow_idm = unwrap_model(idm)
    transition_context = getattr(flow_idm, "_transition_context", None)
    history_tokens_fn = getattr(flow_idm, "_history_tokens", None)
    context_action_head = getattr(flow_idm, "context_action_head", None)
    if not callable(transition_context) or not callable(history_tokens_fn) or not callable(context_action_head):
        raise ValueError(
            "prediction_mode='context_action' requires a flow_transformer IDM with "
            "_transition_context, _history_tokens, and context_action_head."
        )
    context = transition_context(
        current_images,
        future_images,
        state,
        wan_vae_latents=wan_vae_latents,
    )
    history_tokens = history_tokens_fn(prev_state_history, prev_action_history, history_mask)
    return context_action_head(context, history_tokens=history_tokens)


@torch.no_grad()
def evaluate_idm(
    idm: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_normalizer: ActionNormalizer | None = None,
    flow_eval_seed: int | None = 0,
    state_normalizer: StateNormalizer | None = None,
    flow_num_samples: int | None = None,
    flow_noise_scale: float | None = None,
    prediction_mode: IdmPredictionMode = "sample",
) -> dict[str, float | int | None]:
    if prediction_mode not in ("sample", "context_action"):
        raise ValueError(f"prediction_mode must be one of {{'sample', 'context_action'}}, got {prediction_mode!r}.")
    if prediction_mode == "context_action" and not idm_uses_flow_matching(idm):
        raise ValueError("prediction_mode='context_action' is supported only for flow_transformer IDMs.")
    idm.eval()
    if action_normalizer is None:
        action_normalizer = get_action_normalizer(idm, device)
    elif action_normalizer is not None:
        action_normalizer = action_normalizer.to(device)
    if state_normalizer is None:
        state_normalizer = get_state_normalizer(idm, device)
    elif state_normalizer is not None:
        state_normalizer = state_normalizer.to(device)
    totals = {
        "idm_mse_sum": 0.0,
        "idm_mse_count": 0.0,
        "idm_smooth_l1_sum": 0.0,
        "idm_smooth_l1_count": 0.0,
    }
    flow_generator = None
    if idm_uses_flow_matching(idm) and flow_eval_seed is not None:
        flow_generator = torch.Generator(device=device).manual_seed(flow_eval_seed)
    with temporary_flow_sampling_config(
        idm,
        num_samples=flow_num_samples,
        noise_scale=flow_noise_scale,
    ) as (effective_flow_num_samples, effective_flow_noise_scale):
        for batch in tqdm(loader, desc="eval-idm", leave=False):
            batch = to_device(batch, device)
            sample_noise = None
            if prediction_mode == "sample" and idm_uses_flow_matching(idm):
                sample_noise = create_flow_sample_noise(
                    idm,
                    batch_size=int(batch["current_images"].shape[0]),
                    device=device,
                    dtype=batch["current_images"].dtype,
                    generator=flow_generator,
                )
            idm_state = normalize_state_for_idm(idm, batch["state"], state_normalizer)
            wan_vae_kwargs = {"wan_vae_latents": batch["wan_vae_latents"]} if "wan_vae_latents" in batch else {}
            history_kwargs = idm_history_kwargs(
                batch,
                idm=idm,
                action_normalizer=action_normalizer,
                state_normalizer=state_normalizer,
            )
            if prediction_mode == "sample":
                model_action = idm(
                    batch["current_images"],
                    batch["future_images"],
                    idm_state,
                    batch["task_id"],
                    sample_noise=sample_noise,
                    **wan_vae_kwargs,
                    **history_kwargs,
                )
            else:
                model_action = _evaluate_context_action_prediction(
                    idm,
                    batch["current_images"],
                    batch["future_images"],
                    idm_state,
                    **wan_vae_kwargs,
                    **history_kwargs,
                )
            predicted_action = (
                model_action if action_normalizer is None else action_normalizer.denormalize(model_action)
            )
            mse_sum, mse_count = masked_mse_sum_and_count(
                predicted_action,
                batch["action_chunk"],
                batch["action_mask"],
            )
            smooth_l1_sum, smooth_l1_count = masked_smooth_l1_sum_and_count(
                predicted_action,
                batch["action_chunk"],
                batch["action_mask"],
            )
            totals["idm_mse_sum"] += float(mse_sum.detach().cpu())
            totals["idm_mse_count"] += float(mse_count.detach().cpu())
            totals["idm_smooth_l1_sum"] += float(smooth_l1_sum.detach().cpu())
            totals["idm_smooth_l1_count"] += float(smooth_l1_count.detach().cpu())

    return {
        "idm_mse": totals["idm_mse_sum"] / max(totals["idm_mse_count"], 1.0),
        "idm_smooth_l1": totals["idm_smooth_l1_sum"] / max(totals["idm_smooth_l1_count"], 1.0),
        "flow_num_samples": effective_flow_num_samples,
        "flow_noise_scale": effective_flow_noise_scale,
    }


def _masked_mse_per_sample(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    while mask.ndim < predicted.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=predicted.device, dtype=predicted.dtype)
    squared = (predicted - target).square() * mask
    per_sample_total = squared.flatten(1).sum(dim=1)
    per_sample_count = mask.expand_as(predicted).flatten(1).sum(dim=1).clamp_min(1.0)
    return per_sample_total / per_sample_count


def _future_usage_gate_reasons(
    *,
    rank_accuracy: float,
    rank_accuracy_min: float,
    gap: float,
    gap_min: float,
    degradation: float,
    degradation_min: float,
    output_delta_mse: float,
    output_delta_mse_min: float,
    num_ranked_samples: int,
) -> list[str]:
    reasons = []
    if num_ranked_samples <= 0:
        reasons.append("no_ranked_samples")
    if rank_accuracy < rank_accuracy_min:
        reasons.append("rank_accuracy")
    if gap < gap_min:
        reasons.append("gap")
    if degradation < degradation_min:
        reasons.append("current_repeated_degradation")
    if output_delta_mse < output_delta_mse_min:
        reasons.append("current_repeated_output_delta_mse")
    return reasons


@torch.no_grad()
def evaluate_idm_future_usage(
    idm: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_normalizer: ActionNormalizer | None = None,
    flow_eval_seed: int | None = 0,
    state_normalizer: StateNormalizer | None = None,
    flow_num_samples: int | None = None,
    flow_noise_scale: float | None = None,
    *,
    rank_accuracy_min: float = 0.55,
    gap_min: float = 0.0,
    degradation_min: float = 1e-4,
    output_delta_mse_min: float = 1e-4,
    score_mode: IdmFutureRankingScoreMode = "teacher_forced_endpoint",
) -> dict[str, float | int | bool | str]:
    if not idm_uses_flow_matching(idm):
        raise ValueError("Future-usage eval is supported only for flow_transformer IDMs.")
    if score_mode not in ("teacher_forced_endpoint", "sampled_action"):
        raise ValueError(
            "future_usage_score_mode must be one of {'teacher_forced_endpoint', 'sampled_action'}, "
            f"got {score_mode!r}."
        )
    effective_flow_num_samples = resolve_flow_num_samples(idm, flow_num_samples)
    effective_flow_noise_scale = resolve_flow_sample_noise_scale(idm, flow_noise_scale)
    if not 0.0 <= rank_accuracy_min <= 1.0:
        raise ValueError(f"rank_accuracy_min must be in [0, 1], got {rank_accuracy_min}.")
    for name, value in (
        ("gap_min", gap_min),
        ("degradation_min", degradation_min),
        ("output_delta_mse_min", output_delta_mse_min),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}.")

    idm.eval()
    if action_normalizer is None:
        action_normalizer = get_action_normalizer(idm, device)
    elif action_normalizer is not None:
        action_normalizer = action_normalizer.to(device)
    if state_normalizer is None:
        state_normalizer = get_state_normalizer(idm, device)
    elif state_normalizer is not None:
        state_normalizer = state_normalizer.to(device)

    seed = 0 if flow_eval_seed is None else int(flow_eval_seed)
    noise_generator = torch.Generator(device=device).manual_seed(seed)
    totals = {
        "real_endpoint_mse": 0.0,
        "current_repeated_endpoint_mse": 0.0,
        "current_repeated_degradation": 0.0,
        "current_repeated_output_delta_mse": 0.0,
        "rank_accuracy": 0.0,
        "real_vs_best_negative_gap": 0.0,
        "gap_norm": 0.0,
    }
    ranked_count = 0
    current_repeated_count = 0
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )

    for batch_index, batch in enumerate(tqdm(loader, desc="eval-idm-future-usage", leave=False)):
        batch = to_device(batch, device)
        if action_normalizer is None:
            target_action = batch["action_chunk"]
        else:
            target_action = action_normalizer.normalize(batch["action_chunk"])
        idm_state = normalize_state_for_idm(idm, batch["state"], state_normalizer)
        history_kwargs = idm_history_kwargs(
            batch,
            idm=idm,
            action_normalizer=action_normalizer,
            state_normalizer=state_normalizer,
        )
        # Both score modes rank the same real future against the same negative candidates with the
        # same gate logic; only the per-future prediction differs. The endpoint scorer shares one
        # noise draw across the real future and every candidate for a fair single-step comparison;
        # the sampled scorer runs the full sampler deterministically (explicit zero sample noise)
        # and needs no shared noise.
        if score_mode == "sampled_action":

            def predict(future_images: torch.Tensor, candidate_wan_vae_latents: torch.Tensor | None) -> torch.Tensor:
                return _flow_sampled_action_prediction(
                    idm,
                    batch["current_images"],
                    future_images,
                    idm_state,
                    target_action,
                    wan_vae_latents=candidate_wan_vae_latents,
                    **history_kwargs,
                )
        else:
            shared_noise = torch.randn(
                target_action.shape,
                device=target_action.device,
                dtype=target_action.dtype,
                generator=noise_generator,
            )

            def predict(future_images: torch.Tensor, candidate_wan_vae_latents: torch.Tensor | None) -> torch.Tensor:
                return _flow_teacher_forced_endpoint_prediction(
                    idm,
                    batch["current_images"],
                    future_images,
                    idm_state,
                    target_action,
                    shared_noise,
                    wan_vae_latents=candidate_wan_vae_latents,
                    time_value=0.5,
                    **history_kwargs,
                )

        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed + batch_index)
            negative_candidates, _, _ = _future_ranking_negative_candidates(
                batch,
                repeated_current=True,
                shuffled_future=True,
                noisy_future=True,
                zero_future=True,
                same_task_future=False,
                noise_std=1.0,
                min_same_episode_frame_gap=0,
            )
        real_endpoint = predict(batch["future_images"], batch.get("wan_vae_latents"))
        real_mse = _masked_mse_per_sample(real_endpoint, target_action, batch["action_mask"])

        candidate_mses = []
        candidate_valid = []
        current_repeated_endpoint = None
        current_repeated_mse = None
        current_repeated_valid = None
        for candidate in negative_candidates:
            candidate_endpoint = predict(candidate.future_images, candidate.wan_vae_latents)
            candidate_mse = _masked_mse_per_sample(candidate_endpoint, target_action, batch["action_mask"])
            candidate_mses.append(candidate_mse)
            candidate_valid.append(candidate.valid_mask.to(device=real_mse.device))
            if candidate.name == "repeated_current":
                current_repeated_endpoint = candidate_endpoint
                current_repeated_mse = candidate_mse
                current_repeated_valid = candidate.valid_mask.to(device=real_mse.device)

        if candidate_mses:
            negative_mses = torch.stack(candidate_mses, dim=1)
            valid_mask = torch.stack(candidate_valid, dim=1)
            valid_sample = valid_mask.any(dim=1)
            if bool(valid_sample.any()):
                best_negative_mse = negative_mses.masked_fill(~valid_mask, torch.inf).min(dim=1).values
                gap = best_negative_mse - real_mse
                batch_count = int(valid_sample.sum().detach().cpu())
                totals["real_endpoint_mse"] += float(real_mse[valid_sample].sum().detach().cpu())
                totals["rank_accuracy"] += float(
                    (real_mse[valid_sample] < best_negative_mse[valid_sample]).sum().detach().cpu()
                )
                totals["real_vs_best_negative_gap"] += float(gap[valid_sample].sum().detach().cpu())
                gap_norm = gap / (real_mse.abs() + best_negative_mse.abs()).clamp_min(1e-8)
                totals["gap_norm"] += float(gap_norm[valid_sample].sum().detach().cpu())
                ranked_count += batch_count

        if (
            current_repeated_endpoint is not None
            and current_repeated_mse is not None
            and current_repeated_valid is not None
            and bool(current_repeated_valid.any())
        ):
            output_delta_mse = _masked_mse_per_sample(
                current_repeated_endpoint,
                real_endpoint,
                batch["action_mask"],
            )
            valid = current_repeated_valid
            batch_count = int(valid.sum().detach().cpu())
            degradation = current_repeated_mse - real_mse
            totals["current_repeated_endpoint_mse"] += float(current_repeated_mse[valid].sum().detach().cpu())
            totals["current_repeated_degradation"] += float(degradation[valid].sum().detach().cpu())
            totals["current_repeated_output_delta_mse"] += float(output_delta_mse[valid].sum().detach().cpu())
            current_repeated_count += batch_count

    real_endpoint_mse = totals["real_endpoint_mse"] / max(ranked_count, 1)
    current_repeated_endpoint_mse = totals["current_repeated_endpoint_mse"] / max(current_repeated_count, 1)
    current_repeated_degradation = totals["current_repeated_degradation"] / max(current_repeated_count, 1)
    current_repeated_output_delta_mse = totals["current_repeated_output_delta_mse"] / max(current_repeated_count, 1)
    rank_accuracy = totals["rank_accuracy"] / max(ranked_count, 1)
    real_vs_best_negative_gap = totals["real_vs_best_negative_gap"] / max(ranked_count, 1)
    gap_norm = totals["gap_norm"] / max(ranked_count, 1)
    reasons = _future_usage_gate_reasons(
        rank_accuracy=rank_accuracy,
        rank_accuracy_min=rank_accuracy_min,
        gap=real_vs_best_negative_gap,
        gap_min=gap_min,
        degradation=current_repeated_degradation,
        degradation_min=degradation_min,
        output_delta_mse=current_repeated_output_delta_mse,
        output_delta_mse_min=output_delta_mse_min,
        num_ranked_samples=ranked_count,
    )
    return {
        "future_usage_real_endpoint_mse": real_endpoint_mse,
        "future_usage_current_repeated_endpoint_mse": current_repeated_endpoint_mse,
        "future_usage_current_repeated_degradation": current_repeated_degradation,
        "future_usage_current_repeated_output_delta_mse": current_repeated_output_delta_mse,
        "future_usage_rank_accuracy": rank_accuracy,
        "future_usage_real_vs_best_negative_gap": real_vs_best_negative_gap,
        "future_usage_gap_norm": gap_norm,
        "future_usage_num_ranked_samples": ranked_count,
        "future_usage_gate_pass": not reasons,
        "future_usage_gate_reasons": ",".join(reasons),
        "future_usage_score_mode": score_mode,
        "flow_num_samples": effective_flow_num_samples,
        "flow_noise_scale": effective_flow_noise_scale,
    }


def is_better_idm_checkpoint_row(
    candidate: Mapping[str, Any],
    best: Mapping[str, Any] | None,
    *,
    future_usage_eval: bool,
    min_delta: float = 0.0,
) -> bool:
    if best is None:
        return True
    if future_usage_eval:
        candidate_passed = bool(candidate.get("future_usage_gate_pass", False))
        best_passed = bool(best.get("future_usage_gate_pass", False))
        if candidate_passed and not best_passed:
            return True
        if best_passed and not candidate_passed:
            return False
    return float(candidate["idm_mse"]) < float(best["idm_mse"]) - min_delta


def train_one_epoch(
    world_model: nn.Module,
    idm: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    world_model.train()
    idm.train()
    totals = {"loss": 0.0, "wm_loss": 0.0, "idm_loss": 0.0, "action_smoothness_loss": 0.0}
    count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = to_device(batch, device)
        losses = compute_losses(world_model, idm, batch, idm_target_source=config.idm_target_source)
        loss = (
            config.wm_loss_weight * losses["wm_loss"]
            + config.idm_loss_weight * losses["idm_loss"]
            + config.action_smoothness_weight * losses["action_smoothness_loss"]
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([*world_model.parameters(), *idm.parameters()], max_norm=1.0)
        optimizer.step()

        batch_size = int(batch["current_images"].shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["wm_loss"] += float(losses["wm_loss"].detach().cpu()) * batch_size
        totals["idm_loss"] += float(losses["idm_loss"].detach().cpu()) * batch_size
        totals["action_smoothness_loss"] += float(losses["action_smoothness_loss"].detach().cpu()) * batch_size
        count += batch_size

    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    world_model: nn.Module,
    idm: nn.Module,
    loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    world_model.eval()
    idm.eval()
    totals = {"wm_mse": 0.0, "idm_mse": 0.0, "idm_generated_mse": 0.0}
    count = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = to_device(batch, device)
        losses = compute_losses(world_model, idm, batch, idm_target_source="ground_truth")
        generated_losses = compute_losses(world_model, idm, batch, idm_target_source="generated")
        batch_size = int(batch["current_images"].shape[0])
        totals["wm_mse"] += float(losses["wm_loss"].detach().cpu()) * batch_size
        totals["idm_mse"] += float(losses["idm_loss"].detach().cpu()) * batch_size
        totals["idm_generated_mse"] += float(generated_losses["idm_loss"].detach().cpu()) * batch_size
        count += batch_size

    wm_mse = totals["wm_mse"] / max(count, 1)
    idm_mse = totals["idm_mse"] / max(count, 1)
    idm_generated_mse = totals["idm_generated_mse"] / max(count, 1)
    psnr = 10.0 * np.log10(1.0 / max(wm_mse, 1e-12))
    return {
        "wm_mse": wm_mse,
        "wm_psnr": float(psnr),
        "idm_mse": idm_mse,
        "idm_generated_mse": idm_generated_mse,
    }


def image_grid(batch: dict[str, torch.Tensor], predicted_images: torch.Tensor) -> Image.Image:
    current = batch["current_images"][0].detach().cpu()
    target = batch["future_images"][0, 0].detach().cpu()
    predicted = predicted_images[0, 0].detach().cpu().clamp(0.0, 1.0)
    rows = [current, target, predicted]
    tiles = []
    for row in rows:
        views = []
        for view in row:
            array = (view.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            views.append(array)
        tiles.append(np.concatenate(views, axis=1))
    grid = np.concatenate(tiles, axis=0)
    return Image.fromarray(grid)


@torch.no_grad()
def save_prediction_grid(world_model: nn.Module, loader: DataLoader, output_path: Path, device: torch.device) -> None:
    world_model.eval()
    batch = next(iter(loader))
    device_batch = to_device(batch, device)
    predicted_images = world_model(device_batch["current_images"], device_batch["state"], device_batch["task_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_grid(batch, predicted_images).save(output_path)


def assert_train_config_model_matches(train_config: TrainConfig, model_config: ModelConfig) -> None:
    """Fail loudly when ``train_config.model`` disagrees with ``model_config``.

    The training entrypoints resolve model dimensions (num_views, image_size,
    action_horizon, num_future_frames, ...) from the dataset before building the
    model. The persisted ``train_config.model`` must equal that resolved
    ``model_config`` so rank/eval/debug tooling reads a single, consistent set of
    dims. We raise rather than silently reconciling the two so an inconsistent
    checkpoint is never written.
    """
    train_model = dataclasses.asdict(train_config.model)
    resolved_model = dataclasses.asdict(model_config)
    if train_model != resolved_model:
        mismatched = {
            key: (train_model.get(key), value) for key, value in resolved_model.items() if train_model.get(key) != value
        }
        raise ValueError(
            "train_config.model is inconsistent with the model_config used to build the model; "
            f"mismatched fields (train_config.model vs model_config): {mismatched}"
        )


def save_checkpoint(
    output_path: Path,
    *,
    world_model: nn.Module,
    idm: nn.Module,
    model_config: ModelConfig,
    train_config: TrainConfig,
    metrics: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "world_model": unwrap_model(world_model).state_dict(),
            "idm": unwrap_model(idm).state_dict(),
            "model_config": dataclasses.asdict(model_config),
            "train_config": dataclasses.asdict(train_config),
            "metrics": metrics,
        },
        output_path,
    )


def save_idm_checkpoint(
    output_path: Path,
    *,
    idm: nn.Module,
    model_config: ModelConfig,
    train_config: TrainConfig,
    metrics: dict[str, Any],
    action_normalizer: ActionNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
) -> None:
    save_idm_state_checkpoint(
        output_path,
        idm_state=module_state_dict_for_checkpoint(idm),
        model_config=model_config,
        train_config=train_config,
        metrics=metrics,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
    )


def module_state_dict_for_checkpoint(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in unwrap_model(model).state_dict().items()}


def save_idm_state_checkpoint(
    output_path: Path,
    *,
    idm_state: dict[str, torch.Tensor],
    model_config: ModelConfig,
    train_config: TrainConfig,
    metrics: dict[str, Any],
    action_normalizer: ActionNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "idm": idm_state,
            "model_config": dataclasses.asdict(model_config),
            "train_config": dataclasses.asdict(train_config),
            "metrics": metrics,
            "checkpoint_type": "idm",
            "action_normalizer": action_normalizer.to_dict() if action_normalizer is not None else None,
            "state_normalizer": state_normalizer.to_dict() if state_normalizer is not None else None,
        },
        output_path,
    )


def _write_jsonl_row(output_path: Path, row: Mapping[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _build_idm_training_metrics(
    *,
    history: list[dict[str, Any]],
    best_row: Mapping[str, Any] | None,
    model_config: ModelConfig,
    train_config: TrainConfig,
    idm_context_action_warmup_epochs: int | None,
    device: torch.device,
    idm: nn.Module,
    cached_future_dir: str | Path | None,
    wan_vae_latent_cache_dir: str | Path | None,
    include_gt_futures_with_cache: bool,
    action_normalizer: ActionNormalizer | None,
    state_normalizer: StateNormalizer,
    stopped_early: bool,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "history": history,
        "final": history[-1] if history else {},
        "model_config": dataclasses.asdict(model_config),
        "train_config": dataclasses.asdict(train_config),
        "idm_context_action_warmup_epochs": idm_context_action_warmup_epochs,
        "idm_future_ranking_start_epoch": train_config.idm_future_ranking_start_epoch,
        "idm_future_ranking_ramp_epochs": train_config.idm_future_ranking_ramp_epochs,
        "device": str(device),
        "cuda_device_count": torch.cuda.device_count(),
        "idm_parameter_count": count_parameters(idm),
        "idm_trainable_parameter_count": count_trainable_parameters(idm),
        "training_target": "idm",
        "cached_future_dir": str(cached_future_dir) if cached_future_dir is not None else None,
        "wan_vae_latent_cache_dir": str(wan_vae_latent_cache_dir) if wan_vae_latent_cache_dir is not None else None,
        "include_gt_futures_with_cache": include_gt_futures_with_cache,
        "action_normalizer": action_normalizer.to_dict() if action_normalizer is not None else None,
        "state_normalizer": state_normalizer.to_dict(),
        "best": dict(best_row) if best_row is not None else {},
        "best_future_usage_gate_pass": (
            bool(best_row.get("future_usage_gate_pass", False))
            if train_config.idm_future_usage_eval and best_row is not None
            else None
        ),
        "best_future_usage_gate_reasons": (
            str(best_row.get("future_usage_gate_reasons", ""))
            if train_config.idm_future_usage_eval and best_row is not None
            else None
        ),
        "stopped_early": stopped_early,
        "final_checkpoint": str(output_dir / "idm_checkpoint.pt"),
        "best_checkpoint": str(output_dir / "best_idm_checkpoint.pt") if best_row is not None else None,
    }


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[ConvVideoWorldModel, InverseDynamicsModel, ModelConfig]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    world_model, idm = create_models(model_config, device)
    world_model.load_state_dict(checkpoint["world_model"])
    idm.load_state_dict(checkpoint["idm"])
    return world_model, idm, model_config


def idm_checkpoint_non_gt_future_provenance(checkpoint: Any) -> str | None:
    """Return a reason string if ``checkpoint`` records that its IDM was trained on
    cached/generated futures, else ``None``.

    A controller IDM must learn to decode actions from ground-truth dataset
    futures; cached/generated Wan futures are for caching, ranking, and
    evaluation only. ``run_idm_training`` already refuses cached/generated
    futures at train time and records the provenance below, so a truthy
    ``cached_future_dir`` or a true ``include_gt_futures_with_cache`` marks a
    checkpoint that should not be used as a controller. Both the top-level
    ``train_config`` and the ``metrics`` dict (plus the ``train_config`` nested
    inside ``metrics``) are scanned so older and newer checkpoint layouts are
    covered.
    """
    if not isinstance(checkpoint, dict):
        return None
    candidates: list[tuple[str, dict]] = []
    train_config = checkpoint.get("train_config")
    if isinstance(train_config, dict):
        candidates.append(("train_config", train_config))
    metrics = checkpoint.get("metrics")
    if isinstance(metrics, dict):
        candidates.append(("metrics", metrics))
        nested = metrics.get("train_config")
        if isinstance(nested, dict):
            candidates.append(("metrics.train_config", nested))
    for source, mapping in candidates:
        cached_future_dir = mapping.get("cached_future_dir")
        if cached_future_dir:
            return f"{source}.cached_future_dir={cached_future_dir!r}"
        if mapping.get("include_gt_futures_with_cache"):
            return f"{source}.include_gt_futures_with_cache=True"
        if mapping.get("idm_target_source") == "generated":
            return f"{source}.idm_target_source='generated'"
    return None


def load_idm_training_frame_delta(idm_checkpoint: str | Path) -> int | None:
    """Return the dataset ``frame_delta`` the IDM was trained on, or ``None`` if unrecorded."""

    checkpoint = torch.load(idm_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return None
    train_config = checkpoint.get("train_config")
    if not isinstance(train_config, dict):
        return None
    dataset_config = train_config.get("dataset")
    if not isinstance(dataset_config, dict):
        return None
    frame_delta = dataset_config.get("frame_delta")
    return None if frame_delta is None else int(frame_delta)


def enforce_idm_frame_delta_contract(idm_checkpoint: str | Path, requested_frame_delta: int) -> int | None:
    """Reject checkpoint use when recorded IDM training ``frame_delta`` mismatches the request."""

    training_frame_delta = load_idm_training_frame_delta(idm_checkpoint)
    if training_frame_delta is None:
        warnings.warn(
            f"IDM checkpoint {str(idm_checkpoint)!r} does not record its training frame_delta; "
            f"cannot verify requested --frame-delta={requested_frame_delta}. Results may be invalid "
            "if this checkpoint was trained with a different temporal gap.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if training_frame_delta != requested_frame_delta:
        raise ValueError(
            f"--frame-delta={requested_frame_delta} disagrees with the IDM training dataset "
            f"frame_delta={training_frame_delta} recorded in {idm_checkpoint}. The IDM decodes "
            "actions across the temporal gap it was trained on; evaluating with a different "
            "--frame-delta feeds futures the IDM was never trained to read. Re-run with "
            f"--frame-delta {training_frame_delta}, or use an IDM trained for "
            f"frame_delta={requested_frame_delta}."
        )
    return training_frame_delta


def load_idm_checkpoint(
    path: str | Path,
    device: torch.device,
    *,
    allow_non_gt_futures: bool = False,
    use_cached_wan_vae_latents: bool = False,
) -> tuple[InverseDynamicsModel, ModelConfig]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # Defense in depth: even though IDM training refuses cached/generated futures,
    # reject loading a checkpoint whose recorded provenance shows it slipped
    # through (or predates that guard). ``allow_non_gt_futures`` is an explicit
    # escape hatch for historical inspection/tests only -- normal use stays safe.
    if not allow_non_gt_futures:
        reason = idm_checkpoint_non_gt_future_provenance(checkpoint)
        if reason is not None:
            raise ValueError(
                f"Refusing to load IDM checkpoint {str(path)!r} that was trained on "
                f"cached/generated futures ({reason}). Cached/generated Wan futures are for "
                "caching/ranking/evaluation only; a controller IDM must be trained on "
                "ground-truth dataset futures. Pass allow_non_gt_futures=True only for "
                "explicit historical inspection or tests."
            )
    model_config = ModelConfig(**checkpoint["model_config"])
    if use_cached_wan_vae_latents:
        model_config = dataclasses.replace(model_config, wan_vae_use_cached_latents=True)
    idm = create_idm_model(model_config, device)
    idm.load_state_dict(checkpoint["idm"])
    if checkpoint.get("action_normalizer") is not None:
        attach_action_normalizer(idm, action_normalizer_from_dict(checkpoint["action_normalizer"], device))
    if checkpoint.get("state_normalizer") is not None:
        attach_state_normalizer(
            idm,
            state_normalizer_from_dict(checkpoint["state_normalizer"], device),
            normalize_forward=True,
        )
    return idm, model_config


@torch.no_grad()
def compute_action_normalizer(dataset: Dataset, batch_size: int, num_workers: int) -> ActionNormalizer:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total = None
    total_square = None
    count = 0
    for batch in loader:
        action = batch["action_chunk"].to(torch.float32)
        mask = batch["action_mask"].to(torch.bool)
        values = action[mask]
        if values.numel() == 0:
            continue
        batch_total = values.sum(dim=0)
        batch_total_square = values.square().sum(dim=0)
        total = batch_total if total is None else total + batch_total
        total_square = batch_total_square if total_square is None else total_square + batch_total_square
        count += int(values.shape[0])
    if total is None or total_square is None or count == 0:
        raise ValueError("Cannot compute action normalizer from an empty action set.")
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(1e-8)
    std = variance.sqrt().clamp_min(1e-4)
    return ActionNormalizer(mean=mean, std=std)


@torch.no_grad()
def compute_state_normalizer(dataset: Dataset, batch_size: int, num_workers: int) -> StateNormalizer:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total = None
    total_square = None
    count = 0
    for batch in loader:
        state = batch["state"].to(torch.float32)
        values = state.reshape(-1, state.shape[-1])
        if values.numel() == 0:
            continue
        batch_total = values.sum(dim=0)
        batch_total_square = values.square().sum(dim=0)
        total = batch_total if total is None else total + batch_total
        total_square = batch_total_square if total_square is None else total_square + batch_total_square
        count += int(values.shape[0])
    if total is None or total_square is None or count == 0:
        raise ValueError("Cannot compute state normalizer from an empty state set.")
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(1e-8)
    std = variance.sqrt().clamp_min(1e-4)
    return StateNormalizer(mean=mean, std=std)


def run_training(config: TrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    dataset = create_dataset(config.dataset)
    spec = infer_batch_spec(dataset, task_vocab_size=config.dataset.task_vocab_size)
    model_config = dataclasses.replace(
        config.model,
        num_views=spec.num_views,
        image_size=config.dataset.image_size,
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        action_horizon=spec.action_horizon,
        idm_history_length=spec.idm_history_length,
        num_future_frames=spec.num_future_frames,
        task_vocab_size=spec.task_vocab_size,
    )
    # Persist the dataset-resolved model dims: every artifact below records
    # config.model, so it must equal the model_config used to build the model.
    config = dataclasses.replace(config, model=model_config, split_gap=effective_training_split_gap(config))
    assert_train_config_model_matches(config, model_config)

    train_dataset, eval_dataset = split_dataset(dataset, config.eval_fraction, config.seed, config.split_gap)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    world_model, idm = create_models(model_config, device)
    world_model, idm = maybe_data_parallel(world_model, idm, enabled=config.data_parallel)
    optimizer = torch.optim.AdamW(
        [*world_model.parameters(), *idm.parameters()],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = []
    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(world_model, idm, train_loader, optimizer, config, device)
        eval_metrics = evaluate(world_model, idm, eval_loader, config, device)
        row = {"epoch": epoch + 1, **{f"train_{key}": value for key, value in train_metrics.items()}, **eval_metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "history": history,
        "final": history[-1] if history else {},
        "model_config": dataclasses.asdict(model_config),
        "train_config": dataclasses.asdict(config),
        "device": str(device),
        "cuda_device_count": torch.cuda.device_count(),
        "world_model_parameter_count": count_parameters(world_model),
        "world_model_trainable_parameter_count": count_trainable_parameters(world_model),
        "idm_parameter_count": count_parameters(idm),
        "idm_trainable_parameter_count": count_trainable_parameters(idm),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_checkpoint(
        output_dir / "checkpoint.pt",
        world_model=world_model,
        idm=idm,
        model_config=model_config,
        train_config=config,
        metrics=metrics,
    )
    save_prediction_grid(world_model, eval_loader, output_dir / "prediction_grid.png", device)
    return metrics


def run_idm_training(
    config: TrainConfig,
    *,
    cached_future_dir: str | Path | None = None,
    include_gt_futures_with_cache: bool = False,
    wan_vae_latent_cache_dir: str | Path | None = None,
    idm_context_action_warmup_epochs: int | None = None,
) -> dict[str, Any]:
    if cached_future_dir is not None or include_gt_futures_with_cache:
        raise ValueError(
            "Generated/cached futures are for eval/ranking only; IDM training uses ground-truth dataset futures."
        )
    if wan_vae_latent_cache_dir is not None and config.model.idm_visual_encoder != "wan_vae":
        raise ValueError("wan_vae_latent_cache_dir requires idm_visual_encoder='wan_vae'.")
    context_action_loss_weight_for_epoch(
        config.idm_context_action_loss_weight,
        0,
        idm_context_action_warmup_epochs,
    )
    future_ranking_weight_for_epoch(
        config.idm_future_ranking_weight,
        0,
        config.idm_future_ranking_start_epoch,
        config.idm_future_ranking_ramp_epochs,
    )
    seed_everything(config.seed)
    device = resolve_device(config.device)
    dataset = create_dataset_with_optional_cache(
        config.dataset,
        cached_future_dir,
        include_gt_futures_with_cache=include_gt_futures_with_cache,
    )
    spec = infer_batch_spec(dataset, task_vocab_size=config.dataset.task_vocab_size)
    model_config = dataclasses.replace(
        config.model,
        num_views=spec.num_views,
        image_size=config.dataset.image_size,
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        action_horizon=spec.action_horizon,
        idm_history_length=spec.idm_history_length,
        num_future_frames=spec.num_future_frames,
        task_vocab_size=spec.task_vocab_size,
        wan_vae_use_cached_latents=wan_vae_latent_cache_dir is not None,
    )
    if wan_vae_latent_cache_dir is not None:
        dataset = CachedWanVaeLatentDataset(dataset, wan_vae_latent_cache_dir, model_config=model_config)
    # Persist the dataset-resolved model dims: every artifact below records
    # config.model, so it must equal the model_config used to build the model.
    config = dataclasses.replace(config, model=model_config, split_gap=effective_training_split_gap(config))
    assert_train_config_model_matches(config, model_config)

    train_dataset, eval_dataset = split_dataset(dataset, config.eval_fraction, config.seed, config.split_gap)
    pin_memory = device.type == "cuda"
    train_loader = create_idm_train_loader(train_dataset, config, pin_memory=pin_memory)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    action_normalizer = (
        compute_action_normalizer(train_dataset, config.batch_size, config.num_workers)
        if config.normalize_actions
        else None
    )
    state_normalizer = compute_state_normalizer(train_dataset, config.batch_size, config.num_workers)

    idm = create_idm_model(model_config, device)
    if config.data_parallel:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("--data-parallel requires at least two CUDA devices.")
        idm = nn.DataParallel(idm)
    attach_action_normalizer(idm, action_normalizer.to(device) if action_normalizer is not None else None)
    attach_state_normalizer(idm, state_normalizer.to(device), normalize_forward=False)
    optimizer = torch.optim.AdamW(idm.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_stream_path = output_dir / "metrics.jsonl"
    metrics_stream_path.write_text("", encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    stopped_early = False
    for epoch in range(config.epochs):
        set_epoch = getattr(train_loader.batch_sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        active_context_action_loss_weight = context_action_loss_weight_for_epoch(
            config.idm_context_action_loss_weight,
            epoch,
            idm_context_action_warmup_epochs,
        )
        active_future_ranking_weight = future_ranking_weight_for_epoch(
            config.idm_future_ranking_weight,
            epoch,
            config.idm_future_ranking_start_epoch,
            config.idm_future_ranking_ramp_epochs,
        )
        train_metrics = train_idm_one_epoch(
            idm,
            train_loader,
            optimizer,
            config,
            device,
            action_normalizer,
            state_normalizer,
            context_action_loss_weight=active_context_action_loss_weight,
            future_ranking_weight=active_future_ranking_weight,
        )
        eval_metrics = evaluate_idm(idm, eval_loader, device, action_normalizer, state_normalizer=state_normalizer)
        if config.idm_future_usage_eval:
            eval_metrics.update(
                evaluate_idm_future_usage(
                    idm,
                    eval_loader,
                    device,
                    action_normalizer,
                    state_normalizer=state_normalizer,
                    rank_accuracy_min=config.idm_future_usage_rank_accuracy_min,
                    gap_min=config.idm_future_usage_gap_min,
                    degradation_min=config.idm_future_usage_degradation_min,
                    output_delta_mse_min=config.idm_future_usage_output_delta_mse_min,
                    score_mode=config.idm_future_usage_score_mode,
                )
            )
        row = {
            "epoch": epoch + 1,
            "idm_context_action_loss_weight_active": active_context_action_loss_weight,
            "idm_future_ranking_weight_active": active_future_ranking_weight,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **eval_metrics,
        }
        history.append(row)
        _write_jsonl_row(metrics_stream_path, row)
        improved = is_better_idm_checkpoint_row(
            row,
            best_row,
            future_usage_eval=config.idm_future_usage_eval,
            min_delta=config.early_stopping_min_delta,
        )
        if improved:
            best_row = dict(row)
            best_state = module_state_dict_for_checkpoint(idm)
            progress_metrics = _build_idm_training_metrics(
                history=history,
                best_row=best_row,
                model_config=model_config,
                train_config=config,
                idm_context_action_warmup_epochs=idm_context_action_warmup_epochs,
                device=device,
                idm=idm,
                cached_future_dir=cached_future_dir,
                wan_vae_latent_cache_dir=wan_vae_latent_cache_dir,
                include_gt_futures_with_cache=include_gt_futures_with_cache,
                action_normalizer=action_normalizer,
                state_normalizer=state_normalizer,
                stopped_early=False,
                output_dir=output_dir,
            )
            save_idm_state_checkpoint(
                output_dir / "best_idm_checkpoint.pt",
                idm_state=best_state,
                model_config=model_config,
                train_config=config,
                metrics=progress_metrics,
                action_normalizer=action_normalizer,
                state_normalizer=state_normalizer,
            )
        print(json.dumps(row, sort_keys=True))
        if (
            config.early_stopping_patience is not None
            and best_row is not None
            and row["epoch"] - int(best_row["epoch"]) >= config.early_stopping_patience
        ):
            stopped_early = True
            break

    metrics = _build_idm_training_metrics(
        history=history,
        best_row=best_row,
        model_config=model_config,
        train_config=config,
        idm_context_action_warmup_epochs=idm_context_action_warmup_epochs,
        device=device,
        idm=idm,
        cached_future_dir=cached_future_dir,
        wan_vae_latent_cache_dir=wan_vae_latent_cache_dir,
        include_gt_futures_with_cache=include_gt_futures_with_cache,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
        stopped_early=stopped_early,
        output_dir=output_dir,
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_idm_checkpoint(
        output_dir / "idm_checkpoint.pt",
        idm=idm,
        model_config=model_config,
        train_config=config,
        metrics=metrics,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
    )
    if best_state is not None:
        save_idm_state_checkpoint(
            output_dir / "best_idm_checkpoint.pt",
            idm_state=best_state,
            model_config=model_config,
            train_config=config,
            metrics=metrics,
            action_normalizer=action_normalizer,
            state_normalizer=state_normalizer,
        )
    return metrics
