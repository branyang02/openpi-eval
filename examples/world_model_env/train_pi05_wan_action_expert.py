from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import tyro
from torch.utils.data import DataLoader, Dataset, Subset

from world_model.pi05_wan_action_expert import (
    WanPi05ActionExpert,
    find_prefix_cache_rows,
    flow_matching_loss_per_sample_parts,
    load_cached_prefix_dataset,
    sample_actions,
    write_fake_prefix_cache,
)

ActionLossWeighting = Literal["none", "original_scale", "clipped_original_scale", "normalized_original_scale"]
ActionLossAggregation = Literal["mean", "task_balanced", "task_cvar"]
ActionNormalizationScope = Literal["global", "per_task"]


@dataclasses.dataclass
class Args:
    cache_path: str = "output/pi05_wan_prefix_cache"
    output_dir: str = "output/pi05_wan_action_expert"
    eval_cache_path: str | None = None
    init_checkpoint: str | None = None
    fake_cache: bool = False
    real_wan_prefix_cache: bool = False
    fake_cache_rows: int = 32
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.0
    conditioning_mode: Literal["wan_prefix", "wan_prefix_state"] = "wan_prefix_state"
    timestep_conditioning: Literal["additive", "film"] = "additive"
    timestep_embedding_style: Literal["diffusion", "pi05"] = "diffusion"
    decoder_arch: Literal[
        "encoder",
        "context_cross_attention",
        "suffix_prefix_cache",
        "joint_softmax_prefix_cache",
    ] = "encoder"
    val_fraction: float = 0.25
    sample_steps: int = 8
    normalize_actions: bool = False
    action_normalization_scope: ActionNormalizationScope = "global"
    action_loss_weighting: ActionLossWeighting = "none"
    action_loss_weight_max: float = 4.0
    action_loss_aggregation: ActionLossAggregation = "mean"
    task_cvar_fraction: float = 0.25
    task_cvar_weight: float = 0.5
    task_cvar_start_weight: float | None = None
    task_cvar_warmup_epochs: int = 0
    eval_random_samples: int = 0
    eval_random_seed: int = 10_007
    device: str = "auto"
    seed: int = 7


_ACTION_NORM_EPS = 1e-6


def _normalize_action_loss_weighting(weighting: str) -> ActionLossWeighting:
    if weighting not in {"none", "original_scale", "clipped_original_scale", "normalized_original_scale"}:
        raise ValueError(
            "action_loss_weighting must be 'none', 'original_scale', 'clipped_original_scale', "
            f"or 'normalized_original_scale', got {weighting!r}."
        )
    return weighting  # type: ignore[return-value]


def _normalize_action_loss_aggregation(aggregation: str) -> ActionLossAggregation:
    if aggregation not in {"mean", "task_balanced", "task_cvar"}:
        raise ValueError(
            "action_loss_aggregation must be 'mean', 'task_balanced', or 'task_cvar', " f"got {aggregation!r}."
        )
    return aggregation  # type: ignore[return-value]


def _validate_task_cvar_settings(
    *,
    fraction: float,
    weight: float,
    start_weight: float | None = None,
    warmup_epochs: int = 0,
) -> None:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"task_cvar_fraction must be in (0, 1], got {fraction}.")
    if weight < 0.0:
        raise ValueError(f"task_cvar_weight must be nonnegative, got {weight}.")
    if start_weight is not None and start_weight < 0.0:
        raise ValueError(f"task_cvar_start_weight must be nonnegative when provided, got {start_weight}.")
    if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int) or warmup_epochs < 0:
        raise ValueError(f"task_cvar_warmup_epochs must be a nonnegative integer, got {warmup_epochs!r}.")


def _action_loss_weight_source(weighting: ActionLossWeighting) -> str:
    if weighting == "original_scale":
        return "action_norm_std_squared"
    if weighting == "clipped_original_scale":
        return "action_norm_std_squared_clipped"
    if weighting == "normalized_original_scale":
        return "action_norm_std_squared_mean_normalized"
    return "ones"


def _requires_action_normalization(weighting: ActionLossWeighting) -> bool:
    return weighting in {"original_scale", "clipped_original_scale", "normalized_original_scale"}


def _resolve_action_loss_weights(
    *,
    weighting: ActionLossWeighting,
    normalize_actions: bool,
    action_norm_std: torch.Tensor | None,
    action_dim: int,
    device: torch.device,
    action_loss_weight_max: float,
) -> torch.Tensor:
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}.")
    if action_loss_weight_max <= 0.0:
        raise ValueError(f"action_loss_weight_max must be positive, got {action_loss_weight_max}.")
    if weighting == "none":
        return torch.ones(action_dim, device=device, dtype=torch.float32)
    if not normalize_actions or action_norm_std is None:
        raise ValueError(
            f"action_loss_weighting={weighting!r} requires normalize_actions=True so action_norm_std is available."
        )
    if action_norm_std.ndim != 1 or action_norm_std.shape[0] != action_dim:
        raise ValueError(f"action_norm_std must have shape ({action_dim},), got {tuple(action_norm_std.shape)}.")
    weights = action_norm_std.to(device=device, dtype=torch.float32).square()
    if weighting == "clipped_original_scale":
        return weights.clamp(max=float(action_loss_weight_max))
    if weighting == "normalized_original_scale":
        return weights / weights.mean().clamp_min(_ACTION_NORM_EPS)
    return weights


def _float_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu()]


def _task_mean_losses(loss_numerator: torch.Tensor, loss_count: torch.Tensor, tasks: Sequence[str]) -> torch.Tensor:
    if loss_numerator.ndim != 1:
        raise ValueError(f"loss_numerator must have shape (B,), got {tuple(loss_numerator.shape)}.")
    if tuple(loss_count.shape) != tuple(loss_numerator.shape):
        raise ValueError(f"loss_count must have shape {tuple(loss_numerator.shape)}, got {tuple(loss_count.shape)}.")
    if len(tasks) != loss_numerator.shape[0]:
        raise ValueError(
            "Task-balanced action loss requires exactly one task label per per-sample loss; "
            f"got {len(tasks)} task label(s) for {loss_numerator.shape[0]} loss value(s)."
        )
    empty_task_indices = [index for index, task in enumerate(tasks) if not isinstance(task, str) or not task]
    if empty_task_indices:
        raise ValueError(
            "Task-tail action loss requires non-empty task labels; "
            f"empty/non-string label(s) at batch position(s): {empty_task_indices}."
        )
    task_means = []
    for task in sorted(set(tasks)):
        indices = [index for index, value in enumerate(tasks) if value == task]
        task_indices = torch.tensor(indices, device=loss_numerator.device, dtype=torch.long)
        task_numerator = loss_numerator.index_select(0, task_indices).sum()
        task_count = loss_count.index_select(0, task_indices).sum()
        if float(task_count.detach().cpu()) <= 0.0:
            raise ValueError(f"Task-tail action loss cannot aggregate task {task!r} with zero valid elements.")
        task_means.append(task_numerator / task_count)
    if not task_means:
        raise ValueError("Task-balanced action loss requires at least one task label.")
    return torch.stack(task_means)


def _aggregate_action_loss(
    per_sample_loss: torch.Tensor,
    loss_numerator: torch.Tensor,
    loss_count: torch.Tensor,
    *,
    tasks: Sequence[str],
    aggregation: ActionLossAggregation,
    task_cvar_fraction: float,
    task_cvar_weight: float,
) -> torch.Tensor:
    aggregation = _normalize_action_loss_aggregation(aggregation)
    _validate_task_cvar_settings(fraction=task_cvar_fraction, weight=task_cvar_weight)
    if tuple(loss_numerator.shape) != tuple(per_sample_loss.shape) or tuple(loss_count.shape) != tuple(
        per_sample_loss.shape
    ):
        raise ValueError("per-sample loss, numerator, and count tensors must all have shape (B,).")
    if aggregation == "mean":
        return loss_numerator.sum() / loss_count.sum().clamp_min(1.0)
    task_means = _task_mean_losses(loss_numerator, loss_count, tasks)
    base = task_means.mean()
    if aggregation == "task_balanced":
        return base
    topk_count = max(1, math.ceil(task_means.shape[0] * task_cvar_fraction))
    tail = torch.topk(task_means, k=topk_count).values.mean()
    return base + task_cvar_weight * tail


def _effective_task_cvar_weight(
    epoch: int,
    *,
    start_weight: float | None,
    final_weight: float,
    warmup_epochs: int,
) -> float:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"epoch must be a nonnegative integer, got {epoch!r}.")
    _validate_task_cvar_settings(
        fraction=1.0,
        weight=final_weight,
        start_weight=start_weight,
        warmup_epochs=warmup_epochs,
    )
    if start_weight is None or warmup_epochs == 0:
        return float(final_weight)
    progress = min(float(epoch) / float(warmup_epochs), 1.0)
    return float(start_weight + (final_weight - start_weight) * progress)


def _normalize_action_normalization_scope(scope: str) -> ActionNormalizationScope:
    if scope not in {"global", "per_task"}:
        raise ValueError(f"action_normalization_scope must be 'global' or 'per_task', got {scope!r}.")
    return scope  # type: ignore[return-value]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _row_wan_action_mode(row: Mapping[str, Any], row_path: Path) -> str | None:
    metadata = row.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"metadata in {row_path} must be a mapping when present.")
    value = metadata.get("wan_action_mode", row.get("wan_action_mode"))
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"wan_action_mode in {row_path} must be a non-empty string, got {value!r}.")
    return value


def _resolve_consistent_wan_action_mode(cache_paths: list[Path]) -> str | None:
    modes: dict[str, list[Path]] = {}
    missing_paths: list[Path] = []
    for row_path in cache_paths:
        row = torch.load(row_path, map_location="cpu", weights_only=False)
        if not isinstance(row, Mapping):
            raise ValueError(f"Cache row {row_path} must be a mapping, got {type(row).__name__}.")
        mode = _row_wan_action_mode(row, row_path)
        if mode is None:
            missing_paths.append(row_path)
        else:
            modes.setdefault(mode, []).append(row_path)

    if missing_paths and modes:
        present = ", ".join(sorted(modes))
        missing = ", ".join(str(path) for path in missing_paths[:5])
        if len(missing_paths) > 5:
            missing += f", ... ({len(missing_paths)} total)"
        raise ValueError(
            "Wan prefix cache rows must all agree on wan_action_mode; "
            f"found labelled mode(s) {present} but missing wan_action_mode in: {missing}."
        )
    if len(modes) > 1:
        details = "; ".join(
            f"{mode}: {paths[0]}{' ...' if len(paths) > 1 else ''}" for mode, paths in sorted(modes.items())
        )
        raise ValueError(f"Wan prefix cache rows disagree on wan_action_mode: {details}.")
    if not modes:
        return None
    return next(iter(modes))


def _collate_prefix_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prefix_tokens": torch.stack([row["prefix_tokens"] for row in rows]),
        "state": torch.stack([row["state"] for row in rows]),
        "actions": torch.stack([row["actions"] for row in rows]),
        "action_mask": torch.stack([row["action_mask"] for row in rows]),
        "task": [str(row["task"]) for row in rows],
    }


def _split_dataset(dataset: Dataset[Any], *, val_fraction: float, seed: int) -> tuple[Dataset[Any], Dataset[Any]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")
    if len(dataset) == 1:
        return dataset, dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_count = max(1, min(len(dataset) - 1, round(len(dataset) * val_fraction)))
    return Subset(dataset, indices[val_count:]), Subset(dataset, indices[:val_count])


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _mean_action(train_loader: DataLoader[dict[str, torch.Tensor]], device: torch.device) -> torch.Tensor:
    total: torch.Tensor | None = None
    count: torch.Tensor | None = None
    for batch in train_loader:
        batch = _move_batch(batch, device)
        actions = batch["actions"]
        mask = batch["action_mask"].to(dtype=actions.dtype).unsqueeze(-1)
        batch_total = (actions * mask).sum(dim=0)
        batch_count = mask.sum(dim=0)
        total = batch_total if total is None else total + batch_total
        count = batch_count if count is None else count + batch_count
    if total is None or count is None:
        raise ValueError("Cannot compute a mean-action baseline from an empty train loader.")
    return total / count.clamp_min(1.0)


def _action_normalization_stats(
    train_loader: DataLoader[dict[str, torch.Tensor]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    total: torch.Tensor | None = None
    square_total: torch.Tensor | None = None
    count: torch.Tensor | None = None
    for batch in train_loader:
        batch = _move_batch(batch, device)
        actions = batch["actions"]
        mask = batch["action_mask"].to(dtype=actions.dtype).unsqueeze(-1).expand_as(actions)
        batch_total = (actions * mask).sum(dim=(0, 1))
        batch_square_total = (actions.square() * mask).sum(dim=(0, 1))
        batch_count = mask.sum(dim=(0, 1))
        total = batch_total if total is None else total + batch_total
        square_total = batch_square_total if square_total is None else square_total + batch_square_total
        count = batch_count if count is None else count + batch_count
    if total is None or square_total is None or count is None:
        raise ValueError("Cannot compute action normalization stats from an empty train loader.")
    if bool((count <= 0).any().detach().cpu()):
        raise ValueError("Cannot normalize actions because at least one action dimension has no valid train values.")
    mean = total / count
    variance = (square_total / count - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(_ACTION_NORM_EPS)
    return mean, std


def _action_normalization_stats_by_task(
    train_loader: DataLoader[dict[str, Any]], device: torch.device
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    totals: dict[str, torch.Tensor] = {}
    square_totals: dict[str, torch.Tensor] = {}
    counts: dict[str, torch.Tensor] = {}
    for batch in train_loader:
        batch = _move_batch(batch, device)
        actions = batch["actions"]
        tasks = batch["task"]
        mask = batch["action_mask"].to(dtype=actions.dtype).unsqueeze(-1).expand_as(actions)
        for batch_index, task in enumerate(tasks):
            task_actions = actions[batch_index]
            task_mask = mask[batch_index]
            task_total = (task_actions * task_mask).sum(dim=0)
            task_square_total = (task_actions.square() * task_mask).sum(dim=0)
            task_count = task_mask.sum(dim=0)
            totals[task] = task_total if task not in totals else totals[task] + task_total
            square_totals[task] = (
                task_square_total if task not in square_totals else square_totals[task] + task_square_total
            )
            counts[task] = task_count if task not in counts else counts[task] + task_count
    if not totals:
        raise ValueError("Cannot compute per-task action normalization stats from an empty train loader.")
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for task in sorted(totals):
        count = counts[task]
        if bool((count <= 0).any().detach().cpu()):
            raise ValueError(f"Cannot normalize actions for task {task!r} because one action dimension has no values.")
        mean = totals[task] / count
        variance = (square_totals[task] / count - mean.square()).clamp_min(0.0)
        stats[task] = (mean, variance.sqrt().clamp_min(_ACTION_NORM_EPS))
    return stats


def _normalize_actions(actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(device=actions.device, dtype=actions.dtype).view(1, 1, -1)
    std = std.to(device=actions.device, dtype=actions.dtype).view(1, 1, -1)
    return (actions - mean) / std


def _task_stats_for_batch(
    task_stats: Mapping[str, tuple[torch.Tensor, torch.Tensor]], tasks: Sequence[str], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    missing_tasks = sorted({task for task in tasks if task not in task_stats})
    if missing_tasks:
        raise ValueError(f"Missing per-task action normalization stats for task(s): {missing_tasks}.")
    means = torch.stack([task_stats[task][0] for task in tasks]).to(device=device)
    stds = torch.stack([task_stats[task][1] for task in tasks]).to(device=device)
    return means, stds


def _normalize_actions_by_task(
    actions: torch.Tensor, task_stats: Mapping[str, tuple[torch.Tensor, torch.Tensor]], tasks: Sequence[str]
) -> torch.Tensor:
    means, stds = _task_stats_for_batch(task_stats, tasks, device=actions.device)
    means = means.to(dtype=actions.dtype).unsqueeze(1)
    stds = stds.to(dtype=actions.dtype).unsqueeze(1)
    return (actions - means) / stds


def _unnormalize_actions(actions: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None) -> torch.Tensor:
    if mean is None or std is None:
        return actions
    mean = mean.to(device=actions.device, dtype=actions.dtype).view(1, 1, -1)
    std = std.to(device=actions.device, dtype=actions.dtype).view(1, 1, -1)
    return actions * std + mean


def _unnormalize_actions_by_task(
    actions: torch.Tensor, task_stats: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None, tasks: Sequence[str]
) -> torch.Tensor:
    if task_stats is None:
        return actions
    means, stds = _task_stats_for_batch(task_stats, tasks, device=actions.device)
    means = means.to(dtype=actions.dtype).unsqueeze(1)
    stds = stds.to(dtype=actions.dtype).unsqueeze(1)
    return actions * stds + means


def _masked_squared_error_sums(
    predicted: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(predicted.shape) != tuple(target.shape):
        raise ValueError(
            f"predicted and target shapes must match, got {tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    if action_mask.ndim != 2 or tuple(action_mask.shape) != tuple(predicted.shape[:2]):
        raise ValueError(f"action_mask must have shape {tuple(predicted.shape[:2])}, got {tuple(action_mask.shape)}.")
    mask = action_mask.to(device=predicted.device, dtype=predicted.dtype).unsqueeze(-1).expand_as(predicted)
    squared_error = (predicted - target).square() * mask
    return squared_error.sum(dim=(0, 1)), mask.sum(dim=(0, 1))


def _mse_from_sums(squared_error: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    return squared_error.sum() / count.sum().clamp_min(1.0)


def _per_dim_mse_from_sums(squared_error: torch.Tensor, count: torch.Tensor) -> list[float]:
    per_dim = squared_error / count.clamp_min(1.0)
    return [float(value) for value in per_dim.detach().cpu()]


@torch.no_grad()
def _evaluate(
    model: WanPi05ActionExpert,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    mean_action: torch.Tensor,
    sample_steps: int,
    device: torch.device,
    action_norm_mean: torch.Tensor | None = None,
    action_norm_std: torch.Tensor | None = None,
    action_norm_by_task: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    eval_random_samples: int = 0,
    eval_random_seed: int = 10_007,
) -> dict[str, Any]:
    if eval_random_samples < 0:
        raise ValueError(f"eval_random_samples must be non-negative, got {eval_random_samples}.")
    model.eval()
    model_squared_error: torch.Tensor | None = None
    model_count: torch.Tensor | None = None
    baseline_squared_error: torch.Tensor | None = None
    baseline_count: torch.Tensor | None = None
    random_squared_error = torch.zeros(eval_random_samples, device=device, dtype=torch.float64)
    random_count = torch.zeros(eval_random_samples, device=device, dtype=torch.float64)
    random_generator = _make_torch_generator(device, eval_random_seed) if eval_random_samples > 0 else None
    count = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        actions = batch["actions"]
        zero_noise = torch.zeros_like(actions)
        predicted_model_space = sample_actions(
            model,
            batch["prefix_tokens"],
            batch["state"],
            num_steps=sample_steps,
            noise=zero_noise,
        )
        if action_norm_by_task is not None:
            predicted = _unnormalize_actions_by_task(predicted_model_space, action_norm_by_task, batch["task"])
        else:
            predicted = _unnormalize_actions(predicted_model_space, action_norm_mean, action_norm_std)
        baseline = mean_action.to(device=device, dtype=actions.dtype).unsqueeze(0).expand_as(actions)
        batch_model_squared_error, batch_model_count = _masked_squared_error_sums(
            predicted, actions, batch["action_mask"]
        )
        batch_baseline_squared_error, batch_baseline_count = _masked_squared_error_sums(
            baseline, actions, batch["action_mask"]
        )
        model_squared_error = (
            batch_model_squared_error
            if model_squared_error is None
            else model_squared_error + batch_model_squared_error
        )
        model_count = batch_model_count if model_count is None else model_count + batch_model_count
        baseline_squared_error = (
            batch_baseline_squared_error
            if baseline_squared_error is None
            else baseline_squared_error + batch_baseline_squared_error
        )
        baseline_count = batch_baseline_count if baseline_count is None else baseline_count + batch_baseline_count
        for sample_index in range(eval_random_samples):
            random_noise = torch.randn(
                actions.shape, device=actions.device, dtype=actions.dtype, generator=random_generator
            )
            random_predicted_model_space = sample_actions(
                model,
                batch["prefix_tokens"],
                batch["state"],
                num_steps=sample_steps,
                noise=random_noise,
            )
            if action_norm_by_task is not None:
                random_predicted = _unnormalize_actions_by_task(
                    random_predicted_model_space, action_norm_by_task, batch["task"]
                )
            else:
                random_predicted = _unnormalize_actions(random_predicted_model_space, action_norm_mean, action_norm_std)
            batch_random_squared_error, batch_random_count = _masked_squared_error_sums(
                random_predicted, actions, batch["action_mask"]
            )
            random_squared_error[sample_index] += batch_random_squared_error.sum().to(dtype=torch.float64)
            random_count[sample_index] += batch_random_count.sum().to(dtype=torch.float64)
        batch_size = actions.shape[0]
        count += batch_size
    if (
        count == 0
        or model_squared_error is None
        or model_count is None
        or baseline_squared_error is None
        or baseline_count is None
    ):
        raise ValueError("Cannot evaluate on an empty validation loader.")
    zero_noise_mse = _mse_from_sums(model_squared_error, model_count)
    mean_action_mse = _mse_from_sums(baseline_squared_error, baseline_count)
    metrics: dict[str, Any] = {
        "model_sample_mse": float(zero_noise_mse.detach().cpu()),
        "model_zero_noise_mse": float(zero_noise_mse.detach().cpu()),
        "mean_action_mse": float(mean_action_mse.detach().cpu()),
        "model_zero_noise_mse_per_action_dim": _per_dim_mse_from_sums(model_squared_error, model_count),
        "mean_action_mse_per_action_dim": _per_dim_mse_from_sums(baseline_squared_error, baseline_count),
    }
    if eval_random_samples > 0:
        random_mses = random_squared_error / random_count.clamp_min(1.0)
        metrics["model_random_noise_mse_mean"] = float(random_mses.mean().detach().cpu())
        metrics["model_random_noise_mse_std"] = float(random_mses.std(unbiased=False).detach().cpu())
    return metrics


def _build_model(sample: dict[str, Any], args: Args) -> WanPi05ActionExpert:
    prefix_tokens = sample["prefix_tokens"]
    state = sample["state"]
    actions = sample["actions"]
    return WanPi05ActionExpert(
        prefix_dim=prefix_tokens.shape[-1],
        state_dim=state.shape[-1],
        action_dim=actions.shape[-1],
        action_horizon=actions.shape[-2],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        conditioning_mode=args.conditioning_mode,
        timestep_conditioning=args.timestep_conditioning,
        timestep_embedding_style=args.timestep_embedding_style,
        decoder_arch=args.decoder_arch,
    )


_INIT_CHECKPOINT_MODEL_KWARGS = (
    "prefix_dim",
    "state_dim",
    "action_dim",
    "action_horizon",
    "hidden_dim",
    "num_layers",
    "num_heads",
    "dropout",
    "conditioning_mode",
    "timestep_conditioning",
    "timestep_embedding_style",
    "decoder_arch",
)

# Keys that older checkpoints may not carry. A missing key is read as the value below
# (the behavior in effect before the key existed) instead of being rejected as missing;
# explicit values are still compared and must match.
_INIT_CHECKPOINT_MODEL_KWARGS_DEFAULTS = {
    "timestep_embedding_style": "diffusion",
}


def _expected_model_kwargs(model: WanPi05ActionExpert, args: Args) -> dict[str, Any]:
    return {
        "prefix_dim": model.prefix_dim,
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "action_horizon": model.action_horizon,
        "hidden_dim": model.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "conditioning_mode": model.conditioning_mode,
        "timestep_conditioning": model.timestep_conditioning,
        "timestep_embedding_style": model.timestep_embedding_style,
        "decoder_arch": model.decoder_arch,
    }


def _require_checkpoint_mapping(value: Any, *, checkpoint_path: Path, key: str | None = None) -> Mapping[str, Any]:
    label = f"{key!r} in {checkpoint_path}" if key is not None else str(checkpoint_path)
    if not isinstance(value, Mapping):
        raise ValueError(f"Init checkpoint {label} must be a mapping, got {type(value).__name__}.")
    return value


def _validate_init_checkpoint_model_kwargs(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    expected: Mapping[str, Any],
) -> None:
    if "model_kwargs" not in checkpoint:
        raise ValueError(f"Init checkpoint {checkpoint_path} is missing required key 'model_kwargs'.")
    model_kwargs = _require_checkpoint_mapping(
        checkpoint["model_kwargs"],
        checkpoint_path=checkpoint_path,
        key="model_kwargs",
    )
    missing = [
        key
        for key in _INIT_CHECKPOINT_MODEL_KWARGS
        if key not in model_kwargs and key not in _INIT_CHECKPOINT_MODEL_KWARGS_DEFAULTS
    ]
    if missing:
        raise ValueError(f"Init checkpoint {checkpoint_path} model_kwargs is missing required key(s): {missing}.")
    mismatches = {}
    for key in _INIT_CHECKPOINT_MODEL_KWARGS:
        checkpoint_value = model_kwargs[key] if key in model_kwargs else _INIT_CHECKPOINT_MODEL_KWARGS_DEFAULTS[key]
        if checkpoint_value != expected[key]:
            mismatches[key] = {"checkpoint": checkpoint_value, "current": expected[key]}
    if mismatches:
        raise ValueError(f"Init checkpoint {checkpoint_path} architecture/model_kwargs mismatch: {mismatches}.")


def _allclose_or_mismatch(
    checkpoint_value: Any, current_value: torch.Tensor, *, field: str, checkpoint_path: Path
) -> None:
    checkpoint_tensor = torch.as_tensor(checkpoint_value).detach().cpu().to(dtype=torch.float32)
    current_tensor = current_value.detach().cpu().to(dtype=torch.float32)
    if tuple(checkpoint_tensor.shape) != tuple(current_tensor.shape) or not torch.allclose(
        checkpoint_tensor,
        current_tensor,
    ):
        raise ValueError(
            f"Init checkpoint {checkpoint_path} action_normalization mismatch for {field}: "
            f"checkpoint shape/value {tuple(checkpoint_tensor.shape)} {checkpoint_tensor.tolist()} != "
            f"current shape/value {tuple(current_tensor.shape)} {current_tensor.tolist()}."
        )


def _validate_init_checkpoint_action_normalization(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    normalize_actions: bool,
    action_normalization_scope: ActionNormalizationScope,
    action_norm_mean: torch.Tensor | None,
    action_norm_std: torch.Tensor | None,
    action_norm_by_task: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None,
) -> None:
    if "action_normalization" not in checkpoint:
        return
    metadata = _require_checkpoint_mapping(
        checkpoint["action_normalization"],
        checkpoint_path=checkpoint_path,
        key="action_normalization",
    )
    checkpoint_enabled = bool(metadata.get("enabled"))
    if checkpoint_enabled != bool(normalize_actions):
        raise ValueError(
            f"Init checkpoint {checkpoint_path} action_normalization.enabled mismatch: "
            f"checkpoint={checkpoint_enabled}, current={bool(normalize_actions)}."
        )
    checkpoint_scope = metadata.get("scope", "global")
    if checkpoint_scope != action_normalization_scope:
        raise ValueError(
            f"Init checkpoint {checkpoint_path} action_normalization.scope mismatch: "
            f"checkpoint={checkpoint_scope!r}, current={action_normalization_scope!r}."
        )
    if not normalize_actions:
        return
    if action_normalization_scope == "global":
        if action_norm_mean is None or action_norm_std is None:
            raise ValueError("Internal error: global action normalization stats were not computed before init load.")
        for field, current_value in (("mean", action_norm_mean), ("std", action_norm_std)):
            if field not in metadata:
                raise ValueError(f"Init checkpoint {checkpoint_path} action_normalization is missing {field!r}.")
            _allclose_or_mismatch(metadata[field], current_value, field=field, checkpoint_path=checkpoint_path)
        return
    if action_norm_by_task is None:
        raise ValueError("Internal error: per-task action normalization stats were not computed before init load.")
    tasks_metadata = _require_checkpoint_mapping(
        metadata.get("tasks"),
        checkpoint_path=checkpoint_path,
        key="action_normalization.tasks",
    )
    checkpoint_tasks = set(tasks_metadata)
    current_tasks = set(action_norm_by_task)
    if checkpoint_tasks != current_tasks:
        raise ValueError(
            f"Init checkpoint {checkpoint_path} action_normalization task keys mismatch: "
            f"checkpoint={sorted(checkpoint_tasks)}, current={sorted(current_tasks)}."
        )
    for task in sorted(current_tasks):
        task_metadata = _require_checkpoint_mapping(
            tasks_metadata[task],
            checkpoint_path=checkpoint_path,
            key=f"action_normalization.tasks[{task!r}]",
        )
        mean, std = action_norm_by_task[task]
        for field, current_value in (("mean", mean), ("std", std)):
            if field not in task_metadata:
                raise ValueError(
                    f"Init checkpoint {checkpoint_path} action_normalization for task {task!r} is missing {field!r}."
                )
            _allclose_or_mismatch(
                task_metadata[field],
                current_value,
                field=f"tasks[{task!r}].{field}",
                checkpoint_path=checkpoint_path,
            )


def _load_init_checkpoint_strict(
    model: WanPi05ActionExpert,
    *,
    args: Args,
    device: torch.device,
    action_normalization_scope: ActionNormalizationScope,
    action_norm_mean: torch.Tensor | None,
    action_norm_std: torch.Tensor | None,
    action_norm_by_task: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None,
) -> str | None:
    if args.init_checkpoint is None:
        return None
    checkpoint_path = Path(args.init_checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise ValueError(f"Init checkpoint does not exist or is not a file: {checkpoint_path}.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint = _require_checkpoint_mapping(checkpoint, checkpoint_path=checkpoint_path)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Init checkpoint {checkpoint_path} is missing required key 'model_state_dict'.")
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError(
            f"Init checkpoint {checkpoint_path} model_state_dict must be a mapping, got {type(state_dict).__name__}."
        )
    _validate_init_checkpoint_model_kwargs(
        checkpoint,
        checkpoint_path=checkpoint_path,
        expected=_expected_model_kwargs(model, args),
    )
    _validate_init_checkpoint_action_normalization(
        checkpoint,
        checkpoint_path=checkpoint_path,
        normalize_actions=args.normalize_actions,
        action_normalization_scope=action_normalization_scope,
        action_norm_mean=action_norm_mean,
        action_norm_std=action_norm_std,
        action_norm_by_task=action_norm_by_task,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Init checkpoint {checkpoint_path} model_state_dict strict load failed: {error}") from error
    return str(checkpoint_path)


def run_train_eval(args: Args) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    cache_path = Path(args.cache_path)
    eval_cache_path = Path(args.eval_cache_path) if args.eval_cache_path is not None else None
    action_loss_weighting = _normalize_action_loss_weighting(args.action_loss_weighting)
    action_loss_aggregation = _normalize_action_loss_aggregation(args.action_loss_aggregation)
    _validate_task_cvar_settings(
        fraction=args.task_cvar_fraction,
        weight=args.task_cvar_weight,
        start_weight=args.task_cvar_start_weight,
        warmup_epochs=args.task_cvar_warmup_epochs,
    )
    action_normalization_scope = _normalize_action_normalization_scope(args.action_normalization_scope)
    if _requires_action_normalization(action_loss_weighting) and not args.normalize_actions:
        raise ValueError(
            f"action_loss_weighting={action_loss_weighting!r} requires normalize_actions=True; "
            "enable --normalize-actions or use --action-loss-weighting none."
        )
    if action_normalization_scope == "per_task" and not args.normalize_actions:
        raise ValueError(
            "action_normalization_scope='per_task' requires normalize_actions=True; "
            "enable --normalize-actions or use --action-normalization-scope global."
        )
    if args.fake_cache and not find_prefix_cache_rows(cache_path):
        write_fake_prefix_cache(cache_path, num_rows=args.fake_cache_rows, seed=args.seed)

    dataset = load_cached_prefix_dataset(cache_path, real_wan_prefix_cache=args.real_wan_prefix_cache)
    action_mode_cache_paths = list(dataset.cache_paths)
    if eval_cache_path is None:
        train_dataset, val_dataset = _split_dataset(dataset, val_fraction=args.val_fraction, seed=args.seed)
    else:
        train_dataset = dataset
        eval_dataset = load_cached_prefix_dataset(eval_cache_path, real_wan_prefix_cache=args.real_wan_prefix_cache)
        action_mode_cache_paths.extend(eval_dataset.cache_paths)
        val_dataset = eval_dataset
    wan_action_mode = _resolve_consistent_wan_action_mode(action_mode_cache_paths)
    train_shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_prefix_batch,
        generator=train_shuffle_generator,
    )
    eval_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=_collate_prefix_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=_collate_prefix_batch)

    device = _resolve_device(args.device)
    action_norm_mean: torch.Tensor | None = None
    action_norm_std: torch.Tensor | None = None
    action_norm_by_task: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
    if args.normalize_actions:
        if action_normalization_scope == "per_task":
            action_norm_by_task = _action_normalization_stats_by_task(eval_train_loader, device)
        else:
            action_norm_mean, action_norm_std = _action_normalization_stats(eval_train_loader, device)
    model = _build_model(dataset[0], args).to(device)
    init_checkpoint_path = _load_init_checkpoint_strict(
        model,
        args=args,
        device=device,
        action_normalization_scope=action_normalization_scope,
        action_norm_mean=action_norm_mean,
        action_norm_std=action_norm_std,
        action_norm_by_task=action_norm_by_task,
    )
    action_loss_weights = _resolve_action_loss_weights(
        weighting=action_loss_weighting,
        normalize_actions=args.normalize_actions,
        action_norm_std=action_norm_std,
        action_dim=model.action_dim,
        device=device,
        action_loss_weight_max=args.action_loss_weight_max,
    )
    active_action_loss_weights = action_loss_weights if action_loss_weighting != "none" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    flow_generator = _make_torch_generator(device, args.seed + 1)

    last_train_loss = 0.0
    last_effective_task_cvar_weight = _effective_task_cvar_weight(
        0,
        start_weight=args.task_cvar_start_weight,
        final_weight=args.task_cvar_weight,
        warmup_epochs=args.task_cvar_warmup_epochs,
    )
    for epoch in range(args.epochs):
        effective_task_cvar_weight = _effective_task_cvar_weight(
            epoch,
            start_weight=args.task_cvar_start_weight,
            final_weight=args.task_cvar_weight,
            warmup_epochs=args.task_cvar_warmup_epochs,
        )
        last_effective_task_cvar_weight = effective_task_cvar_weight
        model.train()
        loss_total = 0.0
        sample_count = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            actions = batch["actions"]
            if action_norm_by_task is not None:
                actions = _normalize_actions_by_task(actions, action_norm_by_task, batch["task"])
            elif action_norm_mean is not None and action_norm_std is not None:
                actions = _normalize_actions(actions, action_norm_mean, action_norm_std)
            per_sample_loss, loss_numerator, loss_count = flow_matching_loss_per_sample_parts(
                model,
                batch["prefix_tokens"],
                batch["state"],
                actions,
                batch["action_mask"],
                generator=flow_generator,
                action_weights=active_action_loss_weights,
            )
            loss = _aggregate_action_loss(
                per_sample_loss,
                loss_numerator,
                loss_count,
                tasks=batch["task"],
                aggregation=action_loss_aggregation,
                task_cvar_fraction=args.task_cvar_fraction,
                task_cvar_weight=effective_task_cvar_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = batch["actions"].shape[0]
            loss_total += float(loss.detach().cpu()) * batch_size
            sample_count += batch_size
        last_train_loss = loss_total / max(sample_count, 1)

    mean_action = _mean_action(eval_train_loader, device)
    eval_metrics = _evaluate(
        model,
        val_loader,
        mean_action=mean_action,
        sample_steps=args.sample_steps,
        device=device,
        action_norm_mean=action_norm_mean,
        action_norm_std=action_norm_std,
        action_norm_by_task=action_norm_by_task,
        eval_random_samples=args.eval_random_samples,
        eval_random_seed=args.eval_random_seed,
    )
    metrics: dict[str, Any] = {
        "train_loss": last_train_loss,
        "val_model_sample_mse": eval_metrics["model_sample_mse"],
        "val_model_zero_noise_mse": eval_metrics["model_zero_noise_mse"],
        "val_mean_action_mse": eval_metrics["mean_action_mse"],
        "val_model_zero_noise_mse_per_action_dim": eval_metrics["model_zero_noise_mse_per_action_dim"],
        "val_mean_action_mse_per_action_dim": eval_metrics["mean_action_mse_per_action_dim"],
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "cache_path": str(cache_path),
        "init_checkpoint": init_checkpoint_path,
        "device": str(device),
        "normalize_actions": args.normalize_actions,
        "action_normalization_scope": action_normalization_scope,
        "action_loss_weighting": action_loss_weighting,
        "action_loss_weights": _float_list(action_loss_weights),
        "action_loss_weights_source": _action_loss_weight_source(action_loss_weighting),
        "action_loss_weight_max": args.action_loss_weight_max,
        "action_loss_weights_mean": float(action_loss_weights.mean().detach().cpu()),
        "action_loss_aggregation": action_loss_aggregation,
        "task_cvar_fraction": args.task_cvar_fraction,
        "task_cvar_weight": args.task_cvar_weight,
        "task_cvar_start_weight": args.task_cvar_start_weight,
        "task_cvar_warmup_epochs": args.task_cvar_warmup_epochs,
        "task_cvar_schedule_enabled": args.task_cvar_start_weight is not None,
        "task_cvar_final_effective_weight": last_effective_task_cvar_weight,
        "conditioning_mode": args.conditioning_mode,
        "timestep_conditioning": args.timestep_conditioning,
        "timestep_embedding_style": model.timestep_embedding_style,
        "decoder_arch": model.decoder_arch,
    }
    if wan_action_mode is not None:
        metrics["wan_action_mode"] = wan_action_mode
    if args.normalize_actions and action_norm_mean is not None and action_norm_std is not None:
        metrics["action_normalization_mean"] = [float(value) for value in action_norm_mean.detach().cpu()]
        metrics["action_normalization_std"] = [float(value) for value in action_norm_std.detach().cpu()]
        metrics["action_normalization_eps"] = _ACTION_NORM_EPS
    if args.normalize_actions and action_norm_by_task is not None:
        metrics["action_normalization_task_count"] = len(action_norm_by_task)
        metrics["action_normalization_tasks"] = sorted(action_norm_by_task)
        metrics["action_normalization_eps"] = _ACTION_NORM_EPS
    if args.eval_random_samples > 0:
        metrics["eval_random_samples"] = args.eval_random_samples
        metrics["eval_random_seed"] = args.eval_random_seed
        metrics["val_model_random_noise_mse_mean"] = eval_metrics["model_random_noise_mse_mean"]
        metrics["val_model_random_noise_mse_std"] = eval_metrics["model_random_noise_mse_std"]
    if eval_cache_path is not None:
        metrics["eval_cache_path"] = str(eval_cache_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_normalization: dict[str, Any] = {"enabled": args.normalize_actions, "scope": action_normalization_scope}
    if args.normalize_actions and action_norm_mean is not None and action_norm_std is not None:
        action_normalization.update(
            {
                "mean": action_norm_mean.detach().cpu(),
                "std": action_norm_std.detach().cpu(),
                "eps": _ACTION_NORM_EPS,
            }
        )
    if args.normalize_actions and action_norm_by_task is not None:
        action_normalization.update(
            {
                "tasks": {
                    task: {
                        "mean": mean.detach().cpu(),
                        "std": std.detach().cpu(),
                    }
                    for task, (mean, std) in sorted(action_norm_by_task.items())
                },
                "eps": _ACTION_NORM_EPS,
            }
        )
    checkpoint_args = dataclasses.asdict(args)
    checkpoint_args["init_checkpoint"] = init_checkpoint_path
    if wan_action_mode is not None:
        checkpoint_args["wan_action_mode"] = wan_action_mode
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": {
            "prefix_dim": model.prefix_dim,
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
            "action_horizon": model.action_horizon,
            "hidden_dim": model.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "conditioning_mode": model.conditioning_mode,
            "timestep_conditioning": model.timestep_conditioning,
            "timestep_embedding_style": model.timestep_embedding_style,
            "decoder_arch": model.decoder_arch,
        },
        "args": checkpoint_args,
        "metrics": metrics,
        "init_checkpoint": init_checkpoint_path,
        "action_normalization": action_normalization,
        "action_loss": {
            "weighting": action_loss_weighting,
            "weights": action_loss_weights.detach().cpu(),
            "weights_source": _action_loss_weight_source(action_loss_weighting),
            "weight_max": args.action_loss_weight_max,
            "weights_mean": float(action_loss_weights.mean().detach().cpu()),
            "aggregation": action_loss_aggregation,
            "task_cvar_fraction": args.task_cvar_fraction,
            "task_cvar_weight": args.task_cvar_weight,
            "task_cvar_start_weight": args.task_cvar_start_weight,
            "task_cvar_warmup_epochs": args.task_cvar_warmup_epochs,
            "task_cvar_schedule_enabled": args.task_cvar_start_weight is not None,
            "task_cvar_final_effective_weight": last_effective_task_cvar_weight,
        },
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main(args: Args) -> None:
    metrics = run_train_eval(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
