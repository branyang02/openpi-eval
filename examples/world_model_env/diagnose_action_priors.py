from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import tyro

BASELINE_NAMES = (
    "global_mean",
    "task_mean",
    "task_frame_mean",
    "nearest_state",
    "nearest_state_same_task",
)
REQUIRED_ROW_KEYS = ("prefix_tokens", "state", "actions", "action_mask", "task", "metadata")
ScalarKey = bool | int | float | str
TaskKey = tuple[str, ScalarKey]


@dataclasses.dataclass
class Args:
    train_cache_path: str
    eval_cache_path: str
    output_dir: str = "output/action_prior_diagnostics"


@dataclasses.dataclass(frozen=True)
class CacheRow:
    path: Path
    state: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor
    task: str
    metadata: Mapping[str, Any]
    task_keys: tuple[TaskKey, ...]
    frame_key: ScalarKey | None


@dataclasses.dataclass
class _MetricAccumulator:
    squared_error: torch.Tensor
    count: torch.Tensor
    valid_action_steps: float = 0.0
    num_eval_rows: int = 0

    @classmethod
    def create(cls, action_dim: int) -> _MetricAccumulator:
        return cls(
            squared_error=torch.zeros(action_dim, dtype=torch.float64),
            count=torch.zeros(action_dim, dtype=torch.float64),
        )

    def update(self, prediction: torch.Tensor, target: torch.Tensor, action_mask: torch.Tensor) -> None:
        if tuple(prediction.shape) != tuple(target.shape):
            raise ValueError(
                f"prediction shape {tuple(prediction.shape)} must match target shape {tuple(target.shape)}."
            )
        if action_mask.ndim != 1 or int(action_mask.shape[0]) != int(target.shape[0]):
            raise ValueError(
                f"action_mask must have shape ({target.shape[0]},), got {tuple(action_mask.shape)}."
            )

        prediction = prediction.detach().cpu().to(dtype=torch.float64)
        target = target.detach().cpu().to(dtype=torch.float64)
        weights = action_mask.detach().cpu().to(dtype=torch.float64).view(-1, 1).expand_as(target)
        squared = (prediction - target).square() * weights
        self.squared_error += squared.sum(dim=0)
        self.count += weights.sum(dim=0)
        self.valid_action_steps += float(action_mask.detach().cpu().to(dtype=torch.float64).sum().item())
        self.num_eval_rows += 1

    def to_metrics(self) -> dict[str, Any]:
        total_count = self.count.sum()
        mse = None
        if float(total_count.item()) > 0.0:
            mse = float((self.squared_error.sum() / total_count).item())
        per_dim_mse = [
            None if float(count.item()) == 0.0 else float((squared_error / count).item())
            for squared_error, count in zip(self.squared_error, self.count, strict=True)
        ]
        return {
            "mse": mse,
            "per_action_dim_mse": per_dim_mse,
            "num_eval_rows": self.num_eval_rows,
            "num_valid_action_steps": _maybe_int(self.valid_action_steps),
            "num_valid_action_elements": _maybe_int(float(total_count.item())),
        }


def _maybe_int(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _as_float_tensor(value: Any, *, name: str, path: Path) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().to(dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} in {path} contains non-finite values.")
    return tensor


def _scalar_key(value: Any) -> ScalarKey | None:
    if value is None:
        return None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value) if value.is_integer() else value
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() == 0:
            return None
        return _scalar_key(flat[0].item())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if not value:
            return None
        return _scalar_key(value[0])
    return str(value)


def _task_text(row_task: Any, metadata: Mapping[str, Any]) -> str:
    for value in (metadata.get("task"), row_task):
        if value is None:
            continue
        scalar = _scalar_key(value)
        text = str(value if scalar is None else scalar).strip()
        if text:
            return text
    return ""


def _task_keys(row_task: Any, metadata: Mapping[str, Any]) -> tuple[TaskKey, ...]:
    keys: list[TaskKey] = []
    task_index = _scalar_key(metadata.get("task_index"))
    if task_index is not None:
        keys.append(("task_index", task_index))
    task_text = _task_text(row_task, metadata)
    if task_text:
        keys.append(("task", task_text))

    deduped: list[TaskKey] = []
    seen: set[TaskKey] = set()
    for key in keys:
        if key in seen:
            continue
        deduped.append(key)
        seen.add(key)
    return tuple(deduped)


def _frame_key(metadata: Mapping[str, Any]) -> ScalarKey | None:
    return _scalar_key(metadata.get("frame_index"))


def find_cache_rows(cache_path: str | Path) -> list[Path]:
    path = Path(cache_path)
    if path.is_file():
        return [path] if path.suffix == ".pt" else []
    if not path.exists():
        return []
    return sorted(child for child in path.iterdir() if child.is_file() and child.suffix == ".pt")


def _load_row(path: Path) -> CacheRow:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Cache row {path} must be a mapping, got {type(raw).__name__}.")
    missing = [key for key in REQUIRED_ROW_KEYS if key not in raw]
    if missing:
        raise ValueError(f"Cache row {path} is missing required key(s): {', '.join(missing)}.")

    prefix_tokens = _as_float_tensor(raw["prefix_tokens"], name="prefix_tokens", path=path)
    state = _as_float_tensor(raw["state"], name="state", path=path)
    actions = _as_float_tensor(raw["actions"], name="actions", path=path)
    action_mask = _as_float_tensor(raw["action_mask"], name="action_mask", path=path).flatten()
    if prefix_tokens.ndim != 2:
        raise ValueError(f"prefix_tokens in {path} must have shape (N, D), got {tuple(prefix_tokens.shape)}.")
    if state.ndim != 1:
        raise ValueError(f"state in {path} must have shape (D,), got {tuple(state.shape)}.")
    if actions.ndim != 2:
        raise ValueError(f"actions in {path} must have shape (H, A), got {tuple(actions.shape)}.")
    if int(action_mask.shape[0]) != int(actions.shape[0]):
        raise ValueError(f"action_mask in {path} must have shape ({actions.shape[0]},), got {tuple(action_mask.shape)}.")
    if bool((action_mask < 0).any()):
        raise ValueError(f"action_mask in {path} contains negative values.")

    metadata = raw["metadata"]
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"metadata in {path} must be a mapping, got {type(metadata).__name__}.")
    task = str(raw["task"])
    metadata_dict = dict(metadata)
    return CacheRow(
        path=path,
        state=state,
        actions=actions,
        action_mask=action_mask,
        task=task,
        metadata=metadata_dict,
        task_keys=_task_keys(task, metadata_dict),
        frame_key=_frame_key(metadata_dict),
    )


def load_cache_rows(cache_path: str | Path) -> list[CacheRow]:
    row_paths = find_cache_rows(cache_path)
    if not row_paths:
        raise FileNotFoundError(f"No .pt cached prefix rows found at {Path(cache_path)}.")
    return [_load_row(path) for path in row_paths]


def _validate_compatible_rows(train_rows: Sequence[CacheRow], eval_rows: Sequence[CacheRow]) -> tuple[int, int, int]:
    if not train_rows:
        raise ValueError("Cannot diagnose action priors with an empty train cache.")
    if not eval_rows:
        raise ValueError("Cannot diagnose action priors with an empty eval cache.")

    action_horizon, action_dim = train_rows[0].actions.shape
    state_dim = int(train_rows[0].state.shape[0])
    for split, rows in (("train", train_rows), ("eval", eval_rows)):
        for row in rows:
            if tuple(row.actions.shape) != (action_horizon, action_dim):
                raise ValueError(
                    f"{split} row {row.path} has action shape {tuple(row.actions.shape)}; "
                    f"expected ({action_horizon}, {action_dim})."
                )
            if tuple(row.action_mask.shape) != (action_horizon,):
                raise ValueError(
                    f"{split} row {row.path} has action_mask shape {tuple(row.action_mask.shape)}; "
                    f"expected ({action_horizon},)."
                )
            if tuple(row.state.shape) != (state_dim,):
                raise ValueError(
                    f"{split} row {row.path} has state shape {tuple(row.state.shape)}; expected ({state_dim},)."
                )
    return int(action_horizon), int(action_dim), state_dim


def _action_total_and_count(rows: Sequence[CacheRow], indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    if not indices:
        raise ValueError("Cannot compute an action mean from an empty row group.")
    action_horizon, action_dim = rows[indices[0]].actions.shape
    total = torch.zeros((action_horizon, action_dim), dtype=torch.float64)
    count = torch.zeros((action_horizon, action_dim), dtype=torch.float64)
    for index in indices:
        row = rows[index]
        actions = row.actions.to(dtype=torch.float64)
        weights = row.action_mask.to(dtype=torch.float64).view(-1, 1).expand_as(actions)
        total += actions * weights
        count += weights
    return total, count


def _mean_from_total_count(total: torch.Tensor, count: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    fallback = fallback.to(dtype=torch.float64)
    raw_mean = total / count.clamp_min(1.0)
    return torch.where(count > 0.0, raw_mean, fallback.expand_as(raw_mean))


def _global_mean_action(rows: Sequence[CacheRow]) -> torch.Tensor:
    total, count = _action_total_and_count(rows, list(range(len(rows))))
    dim_total = total.sum(dim=0)
    dim_count = count.sum(dim=0)
    if bool((dim_count <= 0.0).any()):
        raise ValueError("Cannot compute global_mean because no valid train actions were found.")
    fallback = (dim_total / dim_count).view(1, -1)
    return _mean_from_total_count(total, count, fallback)


def _mean_action_for_indices(
    rows: Sequence[CacheRow],
    indices: Sequence[int],
    *,
    fallback: torch.Tensor,
) -> torch.Tensor:
    total, count = _action_total_and_count(rows, indices)
    return _mean_from_total_count(total, count, fallback)


def _build_index_groups(
    rows: Sequence[CacheRow],
) -> tuple[dict[TaskKey, list[int]], dict[tuple[TaskKey, ScalarKey], list[int]]]:
    task_indices: dict[TaskKey, list[int]] = {}
    task_frame_indices: dict[tuple[TaskKey, ScalarKey], list[int]] = {}
    for index, row in enumerate(rows):
        for task_key in row.task_keys:
            task_indices.setdefault(task_key, []).append(index)
            if row.frame_key is not None:
                task_frame_indices.setdefault((task_key, row.frame_key), []).append(index)
    return task_indices, task_frame_indices


def _first_existing_task_key(
    candidates: Sequence[TaskKey],
    mapping: Mapping[TaskKey, Any],
) -> TaskKey | None:
    for key in candidates:
        if key in mapping:
            return key
    return None


def _first_existing_task_frame_key(
    task_keys: Sequence[TaskKey],
    frame_key: ScalarKey | None,
    mapping: Mapping[tuple[TaskKey, ScalarKey], Any],
) -> tuple[TaskKey, ScalarKey] | None:
    if frame_key is None:
        return None
    for task_key in task_keys:
        key = (task_key, frame_key)
        if key in mapping:
            return key
    return None


def _nearest_index(
    state: torch.Tensor,
    train_states: torch.Tensor,
    *,
    candidate_indices: Sequence[int] | None = None,
) -> int:
    state = state.to(dtype=torch.float64).view(1, -1)
    if candidate_indices is None:
        distances = (train_states - state).square().sum(dim=1)
        return int(torch.argmin(distances).item())
    if not candidate_indices:
        raise ValueError("candidate_indices must not be empty.")
    indices = torch.tensor(list(candidate_indices), dtype=torch.long)
    distances = (train_states.index_select(0, indices) - state).square().sum(dim=1)
    return int(indices[int(torch.argmin(distances).item())].item())


def compute_action_prior_metrics(train_rows: Sequence[CacheRow], eval_rows: Sequence[CacheRow]) -> dict[str, Any]:
    action_horizon, action_dim, state_dim = _validate_compatible_rows(train_rows, eval_rows)
    global_mean = _global_mean_action(train_rows)
    task_indices, task_frame_indices = _build_index_groups(train_rows)
    task_means = {
        key: _mean_action_for_indices(train_rows, indices, fallback=global_mean)
        for key, indices in task_indices.items()
    }
    task_frame_means = {
        key: _mean_action_for_indices(train_rows, indices, fallback=task_means.get(key[0], global_mean))
        for key, indices in task_frame_indices.items()
    }
    train_states = torch.stack([row.state.to(dtype=torch.float64) for row in train_rows])
    accumulators = {name: _MetricAccumulator.create(action_dim) for name in BASELINE_NAMES}
    task_mean_source_counts = {"task_mean": 0, "global_mean": 0}
    task_frame_source_counts = {"task_frame_mean": 0, "task_mean": 0, "global_mean": 0}
    nearest_same_task_source_counts = {"same_task": 0, "nearest_state": 0}
    frame_available_count = 0

    for row in eval_rows:
        target = row.actions.to(dtype=torch.float64)
        mask = row.action_mask.to(dtype=torch.float64)

        accumulators["global_mean"].update(global_mean, target, mask)

        task_key = _first_existing_task_key(row.task_keys, task_means)
        if task_key is None:
            task_mean_source_counts["global_mean"] += 1
            task_prediction = global_mean
        else:
            task_mean_source_counts["task_mean"] += 1
            task_prediction = task_means[task_key]
        accumulators["task_mean"].update(task_prediction, target, mask)

        if row.frame_key is not None:
            frame_available_count += 1
        task_frame_key = _first_existing_task_frame_key(row.task_keys, row.frame_key, task_frame_means)
        if task_frame_key is not None:
            task_frame_source_counts["task_frame_mean"] += 1
            task_frame_prediction = task_frame_means[task_frame_key]
        elif task_key is not None:
            task_frame_source_counts["task_mean"] += 1
            task_frame_prediction = task_means[task_key]
        else:
            task_frame_source_counts["global_mean"] += 1
            task_frame_prediction = global_mean
        accumulators["task_frame_mean"].update(task_frame_prediction, target, mask)

        nearest_global_index = _nearest_index(row.state, train_states)
        accumulators["nearest_state"].update(train_rows[nearest_global_index].actions, target, mask)

        same_task_key = _first_existing_task_key(row.task_keys, task_indices)
        if same_task_key is None:
            nearest_same_task_source_counts["nearest_state"] += 1
            nearest_same_task_index = nearest_global_index
        else:
            nearest_same_task_source_counts["same_task"] += 1
            nearest_same_task_index = _nearest_index(
                row.state,
                train_states,
                candidate_indices=task_indices[same_task_key],
            )
        accumulators["nearest_state_same_task"].update(train_rows[nearest_same_task_index].actions, target, mask)

    num_eval_rows = len(eval_rows)
    baselines = {name: accumulators[name].to_metrics() for name in BASELINE_NAMES}
    baselines["task_mean"]["prediction_source_counts"] = task_mean_source_counts
    baselines["task_mean"]["fallback_fraction"] = task_mean_source_counts["global_mean"] / num_eval_rows
    baselines["task_frame_mean"]["prediction_source_counts"] = task_frame_source_counts
    baselines["task_frame_mean"]["coverage_count"] = task_frame_source_counts["task_frame_mean"]
    baselines["task_frame_mean"]["coverage_denominator"] = num_eval_rows
    baselines["task_frame_mean"]["coverage_fraction"] = task_frame_source_counts["task_frame_mean"] / num_eval_rows
    baselines["task_frame_mean"]["frame_available_count"] = frame_available_count
    baselines["task_frame_mean"]["frame_available_fraction"] = frame_available_count / num_eval_rows
    baselines["nearest_state_same_task"]["prediction_source_counts"] = nearest_same_task_source_counts
    baselines["nearest_state_same_task"]["fallback_fraction"] = (
        nearest_same_task_source_counts["nearest_state"] / num_eval_rows
    )

    return {
        "num_train_rows": len(train_rows),
        "num_eval_rows": num_eval_rows,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "num_train_task_keys": len(task_indices),
        "num_train_task_frame_keys": len(task_frame_indices),
        "baseline_order": list(BASELINE_NAMES),
        "baselines": baselines,
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _format_fraction(numerator: int | float, denominator: int | float) -> str:
    denominator = float(denominator)
    if denominator == 0.0:
        return "n/a"
    return f"{float(numerator) / denominator:.3f}"


def _format_per_dim(values: Sequence[Any]) -> str:
    return "[" + ", ".join(_format_metric(value) for value in values) + "]"


def _baseline_note(name: str, metrics: Mapping[str, Any]) -> str:
    if name == "task_mean":
        counts = metrics["prediction_source_counts"]
        return f"global fallback {counts['global_mean']}/{metrics['num_eval_rows']}"
    if name == "task_frame_mean":
        counts = metrics["prediction_source_counts"]
        return (
            "exact "
            f"{metrics['coverage_count']}/{metrics['coverage_denominator']} "
            f"({_format_fraction(metrics['coverage_count'], metrics['coverage_denominator'])}); "
            f"task/global fallback {counts['task_mean']}/{counts['global_mean']}"
        )
    if name == "nearest_state_same_task":
        counts = metrics["prediction_source_counts"]
        return f"same task {counts['same_task']}/{metrics['num_eval_rows']}; global fallback {counts['nearest_state']}"
    return ""


def markdown_summary(report: Mapping[str, Any]) -> str:
    lines = [
        "# Action Prior Diagnostics",
        "",
        f"- Train rows: `{report['num_train_rows']}`",
        f"- Eval rows: `{report['num_eval_rows']}`",
        f"- Action shape: `({report['action_horizon']}, {report['action_dim']})`",
        "",
        "| Baseline | MSE | Per-action-dim MSE | Notes |",
        "| --- | ---: | --- | --- |",
    ]
    baselines = report["baselines"]
    for name in report["baseline_order"]:
        metrics = baselines[name]
        lines.append(
            "| "
            f"{name} | "
            f"{_format_metric(metrics['mse'])} | "
            f"{_format_per_dim(metrics['per_action_dim_mse'])} | "
            f"{_baseline_note(name, metrics)} |"
        )
    lines.append("")
    return "\n".join(lines)


def diagnose_action_priors(
    *,
    train_cache_path: str | Path,
    eval_cache_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    train_rows = load_cache_rows(train_cache_path)
    eval_rows = load_cache_rows(eval_cache_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "train_cache_path": str(train_cache_path),
        "eval_cache_path": str(eval_cache_path),
        **compute_action_prior_metrics(train_rows, eval_rows),
    }
    json_path = output_dir / "action_prior_metrics.json"
    markdown_path = output_dir / "action_prior_summary.md"
    report["output_json"] = str(json_path)
    report["markdown_summary"] = str(markdown_path)
    markdown_path.write_text(markdown_summary(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(args: Args) -> None:
    report = diagnose_action_priors(
        train_cache_path=args.train_cache_path,
        eval_cache_path=args.eval_cache_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
