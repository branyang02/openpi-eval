from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import tyro

_DATASET_METADATA_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("source", ("source", "dataset_source")),
    ("repo_id", ("repo_id", "repo")),
    ("image_keys", ("image_keys",)),
    ("episodes", ("episodes",)),
    ("samples_per_episode", ("samples_per_episode",)),
    ("frame_delta", ("frame_delta",)),
    ("num_future_frames", ("num_future_frames",)),
    ("action_horizon", ("action_horizon",)),
    ("image_size", ("image_size",)),
)
_DENOISE_METADATA_FIELDS = (
    "denoise_mode",
    "denoise_fraction",
    "denoise_steps_run",
    "completed_denoise_steps",
    "num_inference_steps",
    "stop_after_steps",
)
_MISSING = object()


@dataclasses.dataclass
class Args:
    reference_cache_dir: str = "output/wan_vae_latent_cache"
    generated_cache_dirs: tuple[str, ...] = ()
    output_dir: str = "output/wan_latent_gap"
    require_identical_dataset_indices: bool = True


@dataclasses.dataclass(frozen=True)
class _Cache:
    cache_dir: Path
    config: dict[str, Any]
    rows_by_index: dict[int, dict[str, Any]]
    label: str


@dataclasses.dataclass
class _MetricAccumulator:
    num_samples: int = 0
    total_elements: int = 0
    sum_squared_error: float = 0.0
    sum_absolute_error: float = 0.0
    reference_sum: float = 0.0
    reference_sum_squared: float = 0.0
    cosine_similarity_sum: float = 0.0
    cosine_distance_sum: float = 0.0
    reference_norm_sum: float = 0.0
    generated_norm_sum: float = 0.0
    delta_norm_sum: float = 0.0
    generated_to_reference_norm_ratio_sum: float = 0.0
    generated_to_reference_norm_ratio_count: int = 0
    per_time_squared_error: list[float] = dataclasses.field(default_factory=list)
    per_time_element_count: list[int] = dataclasses.field(default_factory=list)

    def add(self, reference: torch.Tensor, generated: torch.Tensor) -> None:
        diff = generated - reference
        squared = diff.square()
        self.num_samples += 1
        self.total_elements += int(squared.numel())
        self.sum_squared_error += float(squared.sum().item())
        self.sum_absolute_error += float(diff.abs().sum().item())
        self.reference_sum += float(reference.sum().item())
        self.reference_sum_squared += float(reference.square().sum().item())

        reference_flat = reference.reshape(-1)
        generated_flat = generated.reshape(-1)
        delta_flat = diff.reshape(-1)
        reference_norm = float(torch.linalg.vector_norm(reference_flat).item())
        generated_norm = float(torch.linalg.vector_norm(generated_flat).item())
        delta_norm = float(torch.linalg.vector_norm(delta_flat).item())
        cosine_similarity = _cosine_similarity(reference_flat, generated_flat, reference_norm, generated_norm)
        self.cosine_similarity_sum += cosine_similarity
        self.cosine_distance_sum += 1.0 - cosine_similarity
        self.reference_norm_sum += reference_norm
        self.generated_norm_sum += generated_norm
        self.delta_norm_sum += delta_norm
        if reference_norm > 0.0:
            self.generated_to_reference_norm_ratio_sum += generated_norm / reference_norm
            self.generated_to_reference_norm_ratio_count += 1

        time_steps = int(squared.shape[1])
        while len(self.per_time_squared_error) < time_steps:
            self.per_time_squared_error.append(0.0)
            self.per_time_element_count.append(0)
        for time_index in range(time_steps):
            time_slice = squared[:, time_index, :, :]
            self.per_time_squared_error[time_index] += float(time_slice.sum().item())
            self.per_time_element_count[time_index] += int(time_slice.numel())

    def finalize(self) -> dict[str, Any]:
        if self.num_samples <= 0 or self.total_elements <= 0:
            raise ValueError("Cannot finalize Wan latent cache comparison without any matched samples.")

        latent_mse = self.sum_squared_error / self.total_elements
        reference_mean = self.reference_sum / self.total_elements
        reference_variance = max(
            0.0,
            self.reference_sum_squared / self.total_elements - reference_mean * reference_mean,
        )
        ratio_count = self.generated_to_reference_norm_ratio_count
        return {
            "num_samples": self.num_samples,
            "latent_mse": latent_mse,
            "latent_mae": self.sum_absolute_error / self.total_elements,
            "latent_rmse": math.sqrt(latent_mse),
            "normalized_mse_vs_reference_variance": (
                latent_mse / reference_variance if reference_variance > 0.0 else None
            ),
            "reference_variance": reference_variance,
            "cosine_similarity_mean": self.cosine_similarity_sum / self.num_samples,
            "cosine_distance_mean": self.cosine_distance_sum / self.num_samples,
            "reference_norm_mean": self.reference_norm_sum / self.num_samples,
            "generated_norm_mean": self.generated_norm_sum / self.num_samples,
            "delta_norm_mean": self.delta_norm_sum / self.num_samples,
            "generated_to_reference_norm_ratio_mean": (
                self.generated_to_reference_norm_ratio_sum / ratio_count if ratio_count else None
            ),
            "generated_to_reference_norm_ratio_valid_samples": ratio_count,
            "per_time_mse": [
                squared_error / count if count else None
                for squared_error, count in zip(
                    self.per_time_squared_error,
                    self.per_time_element_count,
                    strict=True,
                )
            ],
        }


def _json_normalized(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_normalized(item) for item in value]
    if isinstance(value, list):
        return [_json_normalized(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_normalized(item) for key, item in value.items()}
    return value


def _cosine_similarity(
    reference_flat: torch.Tensor,
    generated_flat: torch.Tensor,
    reference_norm: float,
    generated_norm: float,
) -> float:
    if reference_norm == 0.0 and generated_norm == 0.0:
        return 1.0
    if reference_norm == 0.0 or generated_norm == 0.0:
        return 0.0
    similarity = float(torch.dot(reference_flat, generated_flat).item()) / (reference_norm * generated_norm)
    return min(1.0, max(-1.0, similarity))


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} cache config not found: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} cache config must be a JSON object: {path}")
    return value


def _load_manifest_by_index(cache_dir: Path, *, label: str) -> dict[int, dict[str, Any]]:
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{label} cache manifest not found: {manifest_path}")

    rows_by_index: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{label} cache manifest row {line_number} must be a JSON object: {row!r}")
        if "dataset_index" not in row:
            raise ValueError(f"{label} cache manifest row {line_number} is missing dataset_index: {row}")
        try:
            dataset_index = int(row["dataset_index"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} cache manifest row {line_number} has invalid dataset_index: {row!r}") from error
        if dataset_index in rows_by_index:
            raise ValueError(f"{label} cache manifest has duplicate dataset_index={dataset_index}.")
        if "latent_tensor" not in row or "latent_shape" not in row:
            raise ValueError(f"{label} cache manifest row is missing required latent fields: {row}")
        rows_by_index[dataset_index] = row

    if not rows_by_index:
        raise ValueError(f"{label} cache manifest is empty: {manifest_path}")
    return rows_by_index


def _load_cache(cache_dir: str | Path, *, label: str) -> _Cache:
    resolved_cache_dir = Path(cache_dir).expanduser().resolve()
    config = _read_json_object(resolved_cache_dir / "config.json", label=label)
    rows_by_index = _load_manifest_by_index(resolved_cache_dir, label=label)
    return _Cache(cache_dir=resolved_cache_dir, config=config, rows_by_index=rows_by_index, label=label)


def _manifest_shape(row: Mapping[str, Any], *, label: str, dataset_index: int) -> tuple[int, int, int, int]:
    value = row["latent_shape"]
    if not isinstance(value, list | tuple):
        raise ValueError(
            f"{label} cache manifest latent_shape for dataset_index={dataset_index} must be a rank-4 list, "
            f"got {type(value).__name__}."
        )
    if len(value) != 4:
        raise ValueError(
            f"{label} cache manifest latent_shape for dataset_index={dataset_index} must be rank 4 (C,T,H,W), "
            f"got {value!r}."
        )
    try:
        shape = tuple(int(dim) for dim in value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label} cache manifest latent_shape for dataset_index={dataset_index} must contain integers: {value!r}"
        ) from error
    if any(dim <= 0 for dim in shape):
        raise ValueError(
            f"{label} cache manifest latent_shape for dataset_index={dataset_index} must be positive, got {shape}."
        )
    return shape


def _resolve_cache_path(cache: _Cache, relative_path: Any, *, dataset_index: int) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(
            f"{cache.label} cache latent path for dataset_index={dataset_index} must be a non-empty string, "
            f"got {relative_path!r}."
        )
    path = (cache.cache_dir / relative_path).resolve()
    if not path.is_relative_to(cache.cache_dir):
        raise ValueError(
            f"{cache.label} cache latent path escapes cache directory for dataset_index={dataset_index}: "
            f"{relative_path}"
        )
    return path


def _load_rank4_latent(cache: _Cache, dataset_index: int) -> torch.Tensor:
    row = cache.rows_by_index[dataset_index]
    expected_shape = _manifest_shape(row, label=cache.label, dataset_index=dataset_index)
    latent_path = _resolve_cache_path(cache, row["latent_tensor"], dataset_index=dataset_index)
    if not latent_path.exists():
        raise FileNotFoundError(
            f"{cache.label} cache latent tensor not found for dataset_index={dataset_index}: {latent_path}"
        )
    latent = torch.load(latent_path, map_location="cpu", weights_only=True)
    if not isinstance(latent, torch.Tensor):
        raise TypeError(
            f"{cache.label} cache latent payload for dataset_index={dataset_index} must be a torch.Tensor, "
            f"got {type(latent).__name__}."
        )
    if latent.ndim != 4:
        raise ValueError(
            f"{cache.label} cache latent tensor for dataset_index={dataset_index} must have rank 4 (C,T,H,W), "
            f"got shape {tuple(latent.shape)}."
        )
    actual_shape = tuple(int(dim) for dim in latent.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{cache.label} cache latent tensor shape does not match manifest for dataset_index={dataset_index}: "
            f"{actual_shape} != {expected_shape}."
        )
    return latent.detach().cpu().to(dtype=torch.float64)


def _metadata_value(config: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    containers: list[Mapping[str, Any]] = [config]
    dataset_config = config.get("dataset_config")
    if isinstance(dataset_config, Mapping):
        containers.append(dataset_config)

    for container in containers:
        for alias in aliases:
            if alias in container:
                return container[alias]
    return _MISSING


def _validate_dataset_metadata(reference: _Cache, generated: _Cache) -> None:
    for field, aliases in _DATASET_METADATA_FIELDS:
        reference_value = _metadata_value(reference.config, aliases)
        generated_value = _metadata_value(generated.config, aliases)
        if reference_value is _MISSING or generated_value is _MISSING:
            continue
        if _json_normalized(reference_value) != _json_normalized(generated_value):
            raise ValueError(
                "Dataset metadata mismatch for "
                f"{field!r} between reference cache {reference.cache_dir} and generated cache {generated.cache_dir}: "
                f"reference {field}={reference_value!r}, generated {field}={generated_value!r}."
            )


def _format_index_preview(indices: list[int], *, limit: int = 20) -> str:
    if len(indices) <= limit:
        return str(indices)
    return f"{indices[:limit]} ... ({len(indices)} total)"


def _matched_dataset_indices(
    reference: _Cache,
    generated: _Cache,
    *,
    require_identical_dataset_indices: bool,
) -> list[int]:
    reference_indices = set(reference.rows_by_index)
    generated_indices = set(generated.rows_by_index)
    missing = sorted(reference_indices - generated_indices)
    extra = sorted(generated_indices - reference_indices)
    if require_identical_dataset_indices and (missing or extra):
        details = []
        if missing:
            details.append(f"missing dataset_index values {_format_index_preview(missing)}")
        if extra:
            details.append(f"extra dataset_index values {_format_index_preview(extra)}")
        raise ValueError(
            f"Generated cache {generated.cache_dir} dataset_index set does not match reference cache "
            f"{reference.cache_dir}: {'; '.join(details)}."
        )

    matched = sorted(reference_indices & generated_indices)
    if not matched:
        raise ValueError(f"Generated cache {generated.cache_dir} has no dataset_index values in common with reference.")
    return matched


def _unique_json_values(values: list[Any]) -> list[Any]:
    unique_by_key: dict[str, Any] = {}
    for value in values:
        normalized = _json_normalized(value)
        key = json.dumps(normalized, sort_keys=True)
        if key not in unique_by_key:
            unique_by_key[key] = normalized
    return list(unique_by_key.values())


def _extract_denoise_metadata(generated: _Cache, matched_indices: list[int]) -> dict[str, Any]:
    generator_config = generated.config.get("generator")
    if not isinstance(generator_config, Mapping):
        generator_config = {}

    result: dict[str, Any] = {}
    for field in _DENOISE_METADATA_FIELDS:
        if field in generator_config:
            result[field] = _json_normalized(generator_config[field])
            continue

        row_values: list[Any] = []
        for dataset_index in matched_indices:
            row_metadata = generated.rows_by_index[dataset_index].get("generator_metadata")
            if isinstance(row_metadata, Mapping) and field in row_metadata:
                row_values.append(row_metadata[field])
        if not row_values:
            continue
        unique_values = _unique_json_values(row_values)
        result[field] = unique_values[0] if len(unique_values) == 1 else {"unique_values": unique_values}
    return result


def _compare_generated_cache(reference: _Cache, generated: _Cache, *, args: Args) -> dict[str, Any]:
    _validate_dataset_metadata(reference, generated)
    matched_indices = _matched_dataset_indices(
        reference,
        generated,
        require_identical_dataset_indices=args.require_identical_dataset_indices,
    )

    accumulator = _MetricAccumulator()
    for dataset_index in matched_indices:
        reference_latent = _load_rank4_latent(reference, dataset_index)
        generated_latent = _load_rank4_latent(generated, dataset_index)
        if tuple(reference_latent.shape) != tuple(generated_latent.shape):
            raise ValueError(
                "Latent tensor shape mismatch between reference and generated cache for "
                f"dataset_index={dataset_index}: reference shape {tuple(reference_latent.shape)} != "
                f"generated shape {tuple(generated_latent.shape)}."
            )
        accumulator.add(reference_latent, generated_latent)

    metrics = accumulator.finalize()
    metrics.update(
        {
            "cache_dir": str(generated.cache_dir),
            "cache_name": generated.cache_dir.name,
            "dataset_index_min": min(matched_indices),
            "dataset_index_max": max(matched_indices),
            "denoise_metadata": _extract_denoise_metadata(generated, matched_indices),
        }
    )
    return metrics


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0.0 else "-inf"
        return f"{value:.6g}"
    return str(value)


def _format_markdown_list(values: list[Any]) -> str:
    return "[" + ", ".join(_format_markdown_value(value) for value in values) + "]"


def _format_denoise_metadata(metadata: Mapping[str, Any]) -> str:
    if not metadata:
        return "n/a"
    parts: list[str] = []
    mode = metadata.get("denoise_mode")
    if mode is not None:
        parts.append(str(mode))

    steps = metadata.get("denoise_steps_run", metadata.get("completed_denoise_steps"))
    num_inference_steps = metadata.get("num_inference_steps")
    if steps is not None and num_inference_steps is not None:
        parts.append(f"steps {steps}/{num_inference_steps}")
    elif steps is not None:
        parts.append(f"steps {steps}")

    denoise_fraction = metadata.get("denoise_fraction")
    if denoise_fraction is not None:
        parts.append(f"frac {_format_markdown_value(denoise_fraction)}")

    if "stop_after_steps" in metadata and metadata["stop_after_steps"] is not None:
        parts.append(f"stop {metadata['stop_after_steps']}")
    return ", ".join(parts) if parts else "n/a"


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _summary_to_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Wan Latent Gap Summary",
        "",
        f"Reference cache: `{payload['reference_cache_dir']}`",
        "",
        "| generated_cache | samples | mse | mae | rmse | mse/ref_var | cos_sim | cos_dist | ref_norm | gen_norm | delta_norm | gen/ref_norm | per_time_mse | denoise |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for summary in payload["generated_cache_summaries"]:
        denoise_metadata = summary.get("denoise_metadata", {})
        row = [
            f"`{_escape_markdown_cell(summary['cache_name'])}`",
            _format_markdown_value(summary["num_samples"]),
            _format_markdown_value(summary["latent_mse"]),
            _format_markdown_value(summary["latent_mae"]),
            _format_markdown_value(summary["latent_rmse"]),
            _format_markdown_value(summary["normalized_mse_vs_reference_variance"]),
            _format_markdown_value(summary["cosine_similarity_mean"]),
            _format_markdown_value(summary["cosine_distance_mean"]),
            _format_markdown_value(summary["reference_norm_mean"]),
            _format_markdown_value(summary["generated_norm_mean"]),
            _format_markdown_value(summary["delta_norm_mean"]),
            _format_markdown_value(summary["generated_to_reference_norm_ratio_mean"]),
            _format_markdown_list(summary["per_time_mse"]),
            _escape_markdown_cell(_format_denoise_metadata(denoise_metadata)),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def compare_wan_latent_caches(args: Args) -> dict[str, Any]:
    if not args.generated_cache_dirs:
        raise ValueError("At least one generated Wan latent cache directory is required.")

    reference = _load_cache(args.reference_cache_dir, label="Reference Wan VAE latent")
    summaries = [
        _compare_generated_cache(reference, _load_cache(cache_dir, label="Generated Wan latent"), args=args)
        for cache_dir in args.generated_cache_dirs
    ]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "reference_cache_dir": str(reference.cache_dir),
        "generated_cache_dirs": [str(Path(cache_dir).expanduser().resolve()) for cache_dir in args.generated_cache_dirs],
        "output_dir": str(output_dir),
        "require_identical_dataset_indices": args.require_identical_dataset_indices,
        "summary_json_path": str(output_dir / "latent_gap_summary.json"),
        "summary_markdown_path": str(output_dir / "latent_gap_summary.md"),
        "generated_cache_summaries": summaries,
    }
    (output_dir / "latent_gap_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_dir / "latent_gap_summary.md").write_text(_summary_to_markdown(payload))
    return payload


def main(args: Args) -> None:
    payload = compare_wan_latent_caches(args)
    print(
        json.dumps(
            {
                "summary_json_path": payload["summary_json_path"],
                "summary_markdown_path": payload["summary_markdown_path"],
                "num_generated_caches": len(payload["generated_cache_summaries"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
