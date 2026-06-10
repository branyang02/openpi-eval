from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import tyro
from torch.utils.data import Dataset

from world_model.config import DatasetConfig, DatasetSource
from world_model.data import CachedFutureDataset, create_dataset

MOTION_METRIC_KEYS = (
    "current_to_first_future",
    "current_to_all_futures",
    "adjacent_future_delta",
)


@dataclasses.dataclass
class Args:
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_keys: tuple[str, ...] = ("corner4.image",)
    output_dir: str = "output/future_motion_diagnostics"
    cached_future_dir: str | None = None
    episodes: tuple[int, ...] | None = None
    max_samples: int | None = None
    samples_per_episode: int | None = None
    synthetic_samples: int = 8
    image_size: int = 64
    frame_delta: int = 4
    num_future_frames: int = 1
    action_horizon: int = 32
    seed: int = 7
    write_markdown: bool = False


def _as_float_tensor(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().to(dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values.")
    return tensor


def _future_mask(item: Mapping[str, Any], num_future_frames: int) -> torch.Tensor:
    if "future_image_mask" not in item:
        return torch.ones(num_future_frames, dtype=torch.bool)
    mask = torch.as_tensor(item["future_image_mask"]).detach().cpu().flatten().to(dtype=torch.bool)
    if mask.numel() != num_future_frames:
        raise ValueError(f"future_image_mask length {mask.numel()} does not match {num_future_frames}.")
    return mask


def _item_dataset_index(item: Mapping[str, Any], fallback: int) -> int:
    if "dataset_index" not in item:
        return fallback
    value = torch.as_tensor(item["dataset_index"]).detach().cpu().reshape(-1)
    if value.numel() == 0:
        return fallback
    return int(value[0].item())


def _mae_delta(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    absolute = (left - right).abs()
    return {
        "mae": float(absolute.mean().item()),
        "num_elements": int(absolute.numel()),
    }


def _assign_metric(row: dict[str, Any], key: str, metric: dict[str, float | int] | None) -> None:
    row[f"{key}_mae"] = None if metric is None else float(metric["mae"])
    row[f"{key}_num_elements"] = 0 if metric is None else int(metric["num_elements"])


def motion_metrics_for_item(
    item: Mapping[str, Any],
    *,
    source: str,
    sample_index: int,
    dataset_index: int | None = None,
) -> dict[str, Any]:
    """Compute per-sample future-motion MAE metrics for one dataset item.

    ``current_images`` must have shape ``(V, C, H, W)`` and ``future_images`` must
    have shape ``(K, V, C, H, W)``. Metrics are masked by ``future_image_mask``:
    current-to-first uses future slot 0, current-to-all uses all valid future
    slots, and adjacent-future delta uses valid adjacent future pairs.
    """

    current = _as_float_tensor(item["current_images"], name="current_images")
    futures = _as_float_tensor(item["future_images"], name="future_images")
    if current.ndim != 4:
        raise ValueError(f"current_images must have shape (V, C, H, W), got {tuple(current.shape)}.")
    if futures.ndim != 5:
        raise ValueError(f"future_images must have shape (K, V, C, H, W), got {tuple(futures.shape)}.")
    if tuple(futures.shape[1:]) != tuple(current.shape):
        raise ValueError(
            "future_images shape after the future axis must match current_images: "
            f"{tuple(futures.shape[1:])} != {tuple(current.shape)}."
        )

    mask = _future_mask(item, int(futures.shape[0]))
    valid_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    row: dict[str, Any] = {
        "source": source,
        "sample_index": sample_index,
        "dataset_index": _item_dataset_index(item, sample_index) if dataset_index is None else dataset_index,
        "current_shape": list(current.shape),
        "future_shape": list(futures.shape),
        "valid_future_indices": [int(index) for index in valid_indices],
        "num_valid_future_frames": int(mask.sum().item()),
    }

    first_metric = _mae_delta(futures[0], current) if bool(mask[0]) else None
    all_metric = _mae_delta(futures[mask], current.unsqueeze(0).expand_as(futures[mask])) if bool(mask.any()) else None

    adjacent_pair_mask = mask[1:] & mask[:-1]
    if bool(adjacent_pair_mask.any()):
        adjacent_metric = _mae_delta(futures[1:][adjacent_pair_mask], futures[:-1][adjacent_pair_mask])
    else:
        adjacent_metric = None

    _assign_metric(row, "current_to_first_future", first_metric)
    _assign_metric(row, "current_to_all_futures", all_metric)
    _assign_metric(row, "adjacent_future_delta", adjacent_metric)
    return row


def _aggregate_one_metric(per_sample: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    value_key = f"{key}_mae"
    count_key = f"{key}_num_elements"
    values = [
        (float(row[value_key]), int(row[count_key]))
        for row in per_sample
        if row.get(value_key) is not None and int(row.get(count_key, 0)) > 0
    ]
    if not values:
        return {
            "mae": None,
            "sample_mean": None,
            "sample_std": None,
            "sample_min": None,
            "sample_max": None,
            "num_samples": 0,
            "num_elements": 0,
        }

    sample_values = torch.tensor([value for value, _count in values], dtype=torch.float64)
    total_elements = sum(count for _value, count in values)
    weighted_mae = sum(value * count for value, count in values) / total_elements
    return {
        "mae": float(weighted_mae),
        "sample_mean": float(sample_values.mean().item()),
        "sample_std": float(sample_values.std(unbiased=False).item()),
        "sample_min": float(sample_values.min().item()),
        "sample_max": float(sample_values.max().item()),
        "num_samples": len(values),
        "num_elements": total_elements,
    }


def aggregate_motion_metrics(per_sample: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "num_samples": len(per_sample),
        "num_samples_with_valid_futures": sum(int(row["num_valid_future_frames"]) > 0 for row in per_sample),
        "metrics": {key: _aggregate_one_metric(per_sample, key) for key in MOTION_METRIC_KEYS},
    }


def diagnose_dataset(
    dataset: Dataset,
    *,
    source: str,
    cache_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    for sample_index in range(len(dataset)):
        row = cache_rows[sample_index] if cache_rows is not None else None
        dataset_index = int(row["dataset_index"]) if row is not None else None
        metrics = motion_metrics_for_item(
            dataset[sample_index],
            source=source,
            sample_index=sample_index,
            dataset_index=dataset_index,
        )
        if row is not None:
            metrics["cache_index"] = sample_index
            metrics["cache_row_source"] = row.get("source", "unknown")
            if row.get("generation_seed") is not None:
                metrics["generation_seed"] = int(row["generation_seed"])
        per_sample.append(metrics)

    return {
        "source": source,
        "num_total_samples": len(dataset),
        "per_sample": per_sample,
        "aggregates": aggregate_motion_metrics(per_sample),
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def markdown_summary(output: Mapping[str, Any]) -> str:
    lines = [
        "# Future Motion Diagnostics",
        "",
        f"- Frame delta: `{output['dataset_config']['frame_delta']}`",
        f"- Future frames: `{output['dataset_config']['num_future_frames']}`",
        "",
        "| Source | Samples | current_to_first_future | current_to_all_futures | adjacent_future_delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ("gt", "cached"):
        report = output.get(key)
        if report is None:
            continue
        metrics = report["aggregates"]["metrics"]
        lines.append(
            "| "
            f"{key} | "
            f"{report['aggregates']['num_samples']} | "
            f"{_format_metric(metrics['current_to_first_future']['mae'])} | "
            f"{_format_metric(metrics['current_to_all_futures']['mae'])} | "
            f"{_format_metric(metrics['adjacent_future_delta']['mae'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_dataset_config(args: Args) -> DatasetConfig:
    return DatasetConfig(
        source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=args.image_keys,
        image_size=args.image_size,
        frame_delta=args.frame_delta,
        num_future_frames=args.num_future_frames,
        action_horizon=args.action_horizon,
        max_samples=args.max_samples,
        samples_per_episode=args.samples_per_episode,
        synthetic_samples=args.synthetic_samples,
        episodes=args.episodes,
        seed=args.seed,
    )


def main(args: Args) -> None:
    dataset_config = _build_dataset_config(args)
    base_dataset = create_dataset(dataset_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output: dict[str, Any] = {
        "dataset_config": dataclasses.asdict(dataset_config),
        "cached_future_dir": args.cached_future_dir,
        "gt": diagnose_dataset(base_dataset, source="gt"),
    }

    if args.cached_future_dir is not None:
        cached_dataset = CachedFutureDataset(base_dataset, args.cached_future_dir)
        output["cached"] = diagnose_dataset(
            cached_dataset,
            source="cached",
            cache_rows=cached_dataset.rows,
        )
    else:
        output["cached"] = None

    json_path = output_dir / "future_motion_metrics.json"
    output["output_json"] = str(json_path)
    if args.write_markdown:
        markdown_path = output_dir / "future_motion_summary.md"
        output["markdown_summary"] = str(markdown_path)
        markdown_path.write_text(markdown_summary(output))
    else:
        output["markdown_summary"] = None

    json_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
