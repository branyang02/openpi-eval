from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader

from world_model.pi05_wan_action_expert import (
    LoadedWanPi05ActionExpert,
    load_cached_prefix_dataset,
    load_wan_pi05_action_expert_checkpoint,
    predict_denormalized_action_chunk,
)


@dataclasses.dataclass
class Args:
    checkpoint: str
    cache_path: str
    output_dir: str = "output/pi05_wan_action_expert_eval"
    output_json: str | None = None
    sample_steps: int = 16
    batch_size: int = 16
    device: str = "auto"
    zero_noise: bool = True
    flow_seed: int | None = 0


_FINGERPRINT_DATASET_CONFIG_KEYS = (
    "source",
    "repo_id",
    "image_keys",
    "state_key",
    "action_key",
    "task_key",
    "frame_delta",
    "action_horizon",
    "image_size",
    "max_samples",
    "samples_per_episode",
    "episodes",
    "seed",
)


def _json_normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _dataset_fingerprint(dataset_config: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(dataset_config):
        config = dataclasses.asdict(dataset_config)
    elif isinstance(dataset_config, Mapping):
        config = dict(dataset_config)
    else:
        raise TypeError(f"dataset_config must be a dataclass or mapping, got {type(dataset_config).__name__}.")
    return {
        "dataset_config": {
            key: _json_normalized(config[key]) for key in _FINGERPRINT_DATASET_CONFIG_KEYS if key in config
        }
    }


def _build_sample_fingerprints(
    dataset_config: Any,
    *,
    num_samples: int | None = None,
) -> dict[str, Any]:
    dataset_fingerprint = _dataset_fingerprint(dataset_config)
    sample_fingerprint: dict[str, Any] = {"dataset_fingerprint": dataset_fingerprint}
    if num_samples is not None:
        sample_fingerprint["num_samples"] = int(num_samples)
    return {
        "dataset_fingerprint": dataset_fingerprint,
        "sample_fingerprint": sample_fingerprint,
    }


def _cache_config_path(cache_path: str | Path) -> Path:
    path = Path(cache_path).expanduser()
    if path.is_file():
        return path.parent / "config.json"
    return path / "config.json"


def _load_prefix_cache_dataset_config(cache_path: str | Path) -> dict[str, Any] | None:
    config_path = _cache_config_path(cache_path)
    if not config_path.exists():
        return None
    try:
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Wan prefix cache config is invalid JSON: {config_path}") from error
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Wan prefix cache config must be a JSON object: {config_path}")
    dataset_config = metadata.get("dataset_config")
    if not isinstance(dataset_config, Mapping):
        raise ValueError(f"Wan prefix cache config must contain a dataset_config JSON object: {config_path}")
    return dict(dataset_config)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _synchronize_device_for_timing(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _model_device_and_dtype(loaded: LoadedWanPi05ActionExpert) -> tuple[torch.device, torch.dtype]:
    parameter = next(loaded.model.parameters(), None)
    if parameter is None:
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype


def _collate_prefix_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prefix_tokens": torch.stack([row["prefix_tokens"] for row in rows]),
        "state": torch.stack([row["state"] for row in rows]),
        "actions": torch.stack([row["actions"] for row in rows]),
        "action_mask": torch.stack([row["action_mask"] for row in rows]),
        "task": [str(row["task"]) for row in rows],
    }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _action_mode_from_sample(sample: Mapping[str, Any]) -> str | None:
    value = sample.get("wan_action_mode")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"wan_action_mode must be a non-empty string when present, got {value!r}.")
    return value


def _resolve_wan_action_mode(
    *,
    loaded: LoadedWanPi05ActionExpert,
    dataset: Any,
    checkpoint_path: str,
    cache_path: str,
) -> str | None:
    row_modes: dict[str, int] = {}
    missing = 0
    for index in range(len(dataset)):
        mode = _action_mode_from_sample(dataset[index])
        if mode is None:
            missing += 1
        else:
            row_modes[mode] = row_modes.get(mode, 0) + 1

    if missing and row_modes:
        present = ", ".join(sorted(row_modes))
        raise ValueError(
            "Wan prefix cache rows must all agree on wan_action_mode; "
            f"found labelled mode(s) {present} but {missing} row(s) are missing wan_action_mode in {cache_path}."
        )
    if len(row_modes) > 1:
        details = ", ".join(f"{mode} ({count})" for mode, count in sorted(row_modes.items()))
        raise ValueError(f"Wan prefix cache rows disagree on wan_action_mode in {cache_path}: {details}.")

    row_mode = next(iter(row_modes), None)
    checkpoint_mode = loaded.wan_action_mode
    if checkpoint_mode is not None and row_mode is not None and checkpoint_mode != row_mode:
        raise ValueError(
            "wan_action_mode disagrees between checkpoint and prefix cache rows: "
            f"{checkpoint_path} has {checkpoint_mode!r}, {cache_path} has {row_mode!r}."
        )
    return checkpoint_mode if checkpoint_mode is not None else row_mode


def _validate_eval_args(args: Args) -> None:
    if args.sample_steps <= 0:
        raise ValueError(f"sample_steps must be positive, got {args.sample_steps}.")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")
    if not args.zero_noise and args.flow_seed is None:
        raise ValueError("flow_seed must be set when zero_noise=False so random-noise eval is reproducible.")


def _validate_prediction_shapes(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    loaded: LoadedWanPi05ActionExpert,
) -> None:
    if tuple(predicted.shape) != tuple(target.shape):
        raise ValueError(
            f"Predicted and target action shapes must match, got {tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    expected_action_shape = (loaded.model.action_horizon, loaded.model.action_dim)
    if tuple(target.shape[1:]) != expected_action_shape:
        raise ValueError(
            f"Cached actions must have shape (B, {expected_action_shape[0]}, {expected_action_shape[1]}), "
            f"got {tuple(target.shape)}."
        )
    if action_mask.ndim != 2 or tuple(action_mask.shape) != tuple(target.shape[:2]):
        raise ValueError(f"action_mask must have shape {tuple(target.shape[:2])}, got {tuple(action_mask.shape)}.")


def _masked_metric_sums(
    predicted: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if tuple(predicted.shape) != tuple(target.shape):
        raise ValueError(
            f"predicted and target shapes must match, got {tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    if action_mask.ndim != 2 or tuple(action_mask.shape) != tuple(predicted.shape[:2]):
        raise ValueError(f"action_mask must have shape {tuple(predicted.shape[:2])}, got {tuple(action_mask.shape)}.")
    predicted = predicted.to(dtype=torch.float64)
    target = target.to(device=predicted.device, dtype=torch.float64)
    mask = action_mask.to(device=predicted.device, dtype=torch.float64).unsqueeze(-1).expand_as(predicted)
    squared_error = (predicted - target).square() * mask
    smooth_l1 = F.smooth_l1_loss(predicted, target, reduction="none") * mask
    return (
        squared_error.sum(dim=(0, 1)),
        mask.sum(dim=(0, 1)),
        smooth_l1.sum(),
        mask.sum(),
        action_mask.to(device=predicted.device, dtype=torch.float64).sum(),
    )


def _metrics_from_sums(
    *,
    squared_error_per_dim: torch.Tensor,
    count_per_dim: torch.Tensor,
    smooth_l1_sum: torch.Tensor,
    smooth_l1_count: torch.Tensor,
) -> dict[str, float | list[float]]:
    if bool((count_per_dim <= 0).any().detach().cpu()) or float(smooth_l1_count.detach().cpu()) <= 0.0:
        raise ValueError("Cannot compute dataset action metrics because no valid action steps were found.")
    per_dim_mse = squared_error_per_dim / count_per_dim
    return {
        "dataset_action_mse": float((squared_error_per_dim.sum() / count_per_dim.sum()).detach().cpu()),
        "dataset_action_smooth_l1": float((smooth_l1_sum / smooth_l1_count).detach().cpu()),
        "dataset_action_mse_per_action_dim": [float(value) for value in per_dim_mse.detach().cpu()],
    }


def _empirical_mean_action(loader: Iterable[dict[str, torch.Tensor]], device: torch.device) -> torch.Tensor:
    total: torch.Tensor | None = None
    count: torch.Tensor | None = None
    for batch in loader:
        batch = _move_batch(batch, device)
        actions = batch["actions"].to(dtype=torch.float64)
        mask = batch["action_mask"].to(dtype=torch.float64).unsqueeze(-1).expand_as(actions)
        batch_total = (actions * mask).sum(dim=(0, 1))
        batch_count = mask.sum(dim=(0, 1))
        total = batch_total if total is None else total + batch_total
        count = batch_count if count is None else count + batch_count
    if total is None or count is None or bool((count <= 0).any().detach().cpu()):
        raise ValueError("Cannot compute a mean-action baseline from an empty valid action set.")
    return total / count


@torch.no_grad()
def evaluate_loaded_action_expert(
    loaded: LoadedWanPi05ActionExpert,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    sample_steps: int,
    zero_noise: bool,
    flow_seed: int | None,
    device: torch.device,
) -> dict[str, Any]:
    mean_action = _empirical_mean_action(loader, device)
    model_device, model_dtype = _model_device_and_dtype(loaded)
    generator = None
    if not zero_noise:
        if flow_seed is None:
            raise ValueError("flow_seed must be set when zero_noise=False so random-noise eval is reproducible.")
        generator = torch.Generator(device=model_device).manual_seed(flow_seed)

    model_squared_error: torch.Tensor | None = None
    model_count: torch.Tensor | None = None
    model_smooth_l1_sum: torch.Tensor | None = None
    model_smooth_l1_count: torch.Tensor | None = None
    baseline_squared_error: torch.Tensor | None = None
    baseline_count: torch.Tensor | None = None
    baseline_smooth_l1_sum: torch.Tensor | None = None
    baseline_smooth_l1_count: torch.Tensor | None = None
    num_samples = 0
    num_valid_action_steps = torch.zeros((), device=device, dtype=torch.float64)

    for batch in loader:
        batch = _move_batch(batch, device)
        actions = batch["actions"].to(dtype=torch.float32)
        noise = None
        if zero_noise:
            noise = torch.zeros(
                actions.shape,
                device=model_device,
                dtype=model_dtype,
            )
        predicted = predict_denormalized_action_chunk(
            loaded,
            batch["prefix_tokens"],
            batch["state"],
            num_steps=sample_steps,
            noise=noise,
            generator=generator,
            tasks=batch["task"],
        ).to(device=device, dtype=torch.float32)
        _validate_prediction_shapes(
            predicted=predicted,
            target=actions,
            action_mask=batch["action_mask"],
            loaded=loaded,
        )

        batch_model_squared, batch_model_count, batch_model_l1, batch_model_l1_count, batch_valid_steps = (
            _masked_metric_sums(predicted, actions, batch["action_mask"])
        )
        baseline = mean_action.to(device=device, dtype=actions.dtype).view(1, 1, -1).expand_as(actions)
        batch_baseline_squared, batch_baseline_count, batch_baseline_l1, batch_baseline_l1_count, _ = (
            _masked_metric_sums(baseline, actions, batch["action_mask"])
        )

        model_squared_error = (
            batch_model_squared if model_squared_error is None else model_squared_error + batch_model_squared
        )
        model_count = batch_model_count if model_count is None else model_count + batch_model_count
        model_smooth_l1_sum = batch_model_l1 if model_smooth_l1_sum is None else model_smooth_l1_sum + batch_model_l1
        model_smooth_l1_count = (
            batch_model_l1_count if model_smooth_l1_count is None else model_smooth_l1_count + batch_model_l1_count
        )
        baseline_squared_error = (
            batch_baseline_squared
            if baseline_squared_error is None
            else baseline_squared_error + batch_baseline_squared
        )
        baseline_count = batch_baseline_count if baseline_count is None else baseline_count + batch_baseline_count
        baseline_smooth_l1_sum = (
            batch_baseline_l1 if baseline_smooth_l1_sum is None else baseline_smooth_l1_sum + batch_baseline_l1
        )
        baseline_smooth_l1_count = (
            batch_baseline_l1_count
            if baseline_smooth_l1_count is None
            else baseline_smooth_l1_count + batch_baseline_l1_count
        )
        num_samples += int(actions.shape[0])
        num_valid_action_steps = num_valid_action_steps + batch_valid_steps

    if (
        num_samples == 0
        or model_squared_error is None
        or model_count is None
        or model_smooth_l1_sum is None
        or model_smooth_l1_count is None
        or baseline_squared_error is None
        or baseline_count is None
        or baseline_smooth_l1_sum is None
        or baseline_smooth_l1_count is None
    ):
        raise ValueError("Cannot evaluate on an empty prefix cache.")

    metrics = _metrics_from_sums(
        squared_error_per_dim=model_squared_error,
        count_per_dim=model_count,
        smooth_l1_sum=model_smooth_l1_sum,
        smooth_l1_count=model_smooth_l1_count,
    )
    baseline_metrics = _metrics_from_sums(
        squared_error_per_dim=baseline_squared_error,
        count_per_dim=baseline_count,
        smooth_l1_sum=baseline_smooth_l1_sum,
        smooth_l1_count=baseline_smooth_l1_count,
    )
    baseline_metrics["mean_action"] = [float(value) for value in mean_action.detach().cpu()]
    return {
        **metrics,
        "mean_action_baseline": baseline_metrics,
        "num_samples": num_samples,
        "num_valid_action_steps": int(num_valid_action_steps.detach().cpu().item()),
    }


def _output_json_path(args: Args) -> Path:
    if args.output_json is not None:
        return Path(args.output_json)
    return Path(args.output_dir) / "eval_metrics.json"


def main(args: Args) -> dict[str, Any]:
    _validate_eval_args(args)
    device = _resolve_device(args.device)
    loaded = load_wan_pi05_action_expert_checkpoint(args.checkpoint, device=device)
    dataset = load_cached_prefix_dataset(args.cache_path)
    cache_dataset_config = _load_prefix_cache_dataset_config(args.cache_path)
    wan_action_mode = _resolve_wan_action_mode(
        loaded=loaded,
        dataset=dataset,
        checkpoint_path=args.checkpoint,
        cache_path=args.cache_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_prefix_batch,
    )
    _synchronize_device_for_timing(device)
    eval_start = time.perf_counter()
    metrics = evaluate_loaded_action_expert(
        loaded,
        loader,
        sample_steps=args.sample_steps,
        zero_noise=args.zero_noise,
        flow_seed=args.flow_seed,
        device=device,
    )
    _synchronize_device_for_timing(device)
    eval_elapsed_ms = (time.perf_counter() - eval_start) * 1000.0

    output: dict[str, Any] = {
        **metrics,
        "checkpoint": str(Path(args.checkpoint)),
        "cache_path": str(Path(args.cache_path)),
        "sample_steps": args.sample_steps,
        "zero_noise": args.zero_noise,
        "flow_seed": None if args.zero_noise else args.flow_seed,
        "metric_family": "dataset_action_mse",
        "eval_elapsed_ms": eval_elapsed_ms,
        "eval_ms_per_sample": eval_elapsed_ms / max(int(metrics["num_samples"]), 1),
        "eval_device": str(device),
    }
    if cache_dataset_config is not None:
        output.update(
            _build_sample_fingerprints(
                cache_dataset_config,
                num_samples=int(metrics["num_samples"]),
            )
        )
    if wan_action_mode is not None:
        output["wan_action_mode"] = wan_action_mode

    output_json = _output_json_path(args)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return output


if __name__ == "__main__":
    main(tyro.cli(Args))
