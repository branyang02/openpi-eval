from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import tyro


@dataclasses.dataclass
class Args:
    base_cache_path: str
    target_cache_path: str
    output_dir: str
    alpha: float
    overwrite: bool = False


_ALIGNMENT_KEYS = ("dataset_index", "episode_index", "frame_index", "task_index", "source_dataset_index")


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be a number in [0, 1], got {alpha!r}.")
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}.")
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(value)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Manifest row {index} in {path} must be a JSON object.")
    return [dict(row) for row in rows]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _load_pt_row(cache_dir: Path, manifest_row: Mapping[str, Any]) -> dict[str, Any]:
    row_file = manifest_row.get("row_file")
    if not isinstance(row_file, str) or not row_file:
        raise ValueError(f"Manifest row is missing a string row_file: {manifest_row}")
    row = torch.load(cache_dir / row_file, map_location="cpu", weights_only=False)
    if not isinstance(row, Mapping):
        raise ValueError(f"{cache_dir / row_file} must contain a mapping.")
    return dict(row)


def _require_matching_manifest_rows(
    base_row: Mapping[str, Any],
    target_row: Mapping[str, Any],
    *,
    row_index: int,
) -> None:
    for key in _ALIGNMENT_KEYS:
        if key in base_row or key in target_row:
            if base_row.get(key) != target_row.get(key):
                raise ValueError(
                    f"Manifest row {row_index} is not aligned on {key}: "
                    f"{base_row.get(key)!r} != {target_row.get(key)!r}."
                )
    if base_row.get("prefix_shape") != target_row.get("prefix_shape"):
        raise ValueError(
            f"Manifest row {row_index} prefix_shape mismatch: "
            f"{base_row.get('prefix_shape')!r} != {target_row.get('prefix_shape')!r}."
        )


def _require_matching_sample_rows(
    base_sample: Mapping[str, Any],
    target_sample: Mapping[str, Any],
    *,
    row_index: int,
) -> None:
    base_prefix = torch.as_tensor(base_sample["prefix_tokens"], dtype=torch.float32)
    target_prefix = torch.as_tensor(target_sample["prefix_tokens"], dtype=torch.float32)
    if tuple(base_prefix.shape) != tuple(target_prefix.shape):
        raise ValueError(
            f"Sample row {row_index} prefix token shape mismatch: "
            f"{tuple(base_prefix.shape)} != {tuple(target_prefix.shape)}."
        )
    for key in ("state", "actions", "action_mask"):
        if key in base_sample or key in target_sample:
            base_value = torch.as_tensor(base_sample[key])
            target_value = torch.as_tensor(target_sample[key])
            if tuple(base_value.shape) != tuple(target_value.shape):
                raise ValueError(
                    f"Sample row {row_index} {key} shape mismatch: "
                    f"{tuple(base_value.shape)} != {tuple(target_value.shape)}."
                )
    if str(base_sample.get("task", "")) != str(target_sample.get("task", "")):
        raise ValueError(
            f"Sample row {row_index} task mismatch: {base_sample.get('task')!r} != {target_sample.get('task')!r}."
        )


def interpolate_prefix_caches(args: Args) -> dict[str, Any]:
    alpha = _validate_alpha(args.alpha)
    base_cache = Path(args.base_cache_path).expanduser().resolve()
    target_cache = Path(args.target_cache_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    base_manifest = _load_manifest(base_cache / "manifest.jsonl")
    target_manifest = _load_manifest(target_cache / "manifest.jsonl")
    if len(base_manifest) != len(target_manifest):
        raise ValueError(
            f"Cache row counts differ: {base_cache} has {len(base_manifest)}, "
            f"{target_cache} has {len(target_manifest)}."
        )

    interpolation_metadata = {
        "alpha": alpha,
        "base_cache_path": str(base_cache),
        "target_cache_path": str(target_cache),
        "formula": "(1 - alpha) * base_prefix_tokens + alpha * target_prefix_tokens",
    }
    manifest_rows: list[dict[str, Any]] = []
    output_samples: list[tuple[str, dict[str, Any]]] = []
    for row_index, (base_manifest_row, target_manifest_row) in enumerate(zip(base_manifest, target_manifest)):
        _require_matching_manifest_rows(base_manifest_row, target_manifest_row, row_index=row_index)
        base_sample = _load_pt_row(base_cache, base_manifest_row)
        target_sample = _load_pt_row(target_cache, target_manifest_row)
        _require_matching_sample_rows(base_sample, target_sample, row_index=row_index)

        base_prefix = torch.as_tensor(base_sample["prefix_tokens"], dtype=torch.float32)
        target_prefix = torch.as_tensor(target_sample["prefix_tokens"], dtype=torch.float32)
        blended_prefix = (1.0 - alpha) * base_prefix + alpha * target_prefix

        output_row_file = f"sample_{row_index:06d}.pt"
        output_sample = dict(base_sample)
        output_sample["prefix_tokens"] = blended_prefix
        sample_metadata = dict(output_sample.get("metadata") or {})
        sample_metadata.update(
            {
                "source": "interpolated_wan_prefix_tokens",
                "prefix_interpolation": interpolation_metadata,
                "contains_future_ground_truth_latents": alpha == 0.0,
            }
        )
        output_sample["metadata"] = sample_metadata
        output_samples.append((output_row_file, output_sample))

        output_manifest_row = dict(base_manifest_row)
        output_manifest_row.update(
            {
                "row_file": output_row_file,
                "source": "interpolated_wan_prefix_tokens",
                "prefix_interpolation": interpolation_metadata,
                "contains_future_ground_truth_latents": alpha == 0.0,
            }
        )
        manifest_rows.append(output_manifest_row)

    config = _load_json_object(base_cache / "config.json")
    config.update(
        {
            "cache_kind": "pi05_wan_prefix_interpolation",
            "num_samples": len(manifest_rows),
            "prefix_interpolation": interpolation_metadata,
            "contains_future_ground_truth_latents": alpha == 0.0,
        }
    )
    result = {
        "output_dir": str(output_dir),
        "num_samples": len(manifest_rows),
        "prefix_interpolation": interpolation_metadata,
    }
    tmp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    try:
        tmp_dir.mkdir(parents=True)
        for row_file, sample in output_samples:
            torch.save(sample, tmp_dir / row_file)
        _write_json(tmp_dir / "config.json", config)
        _write_jsonl(tmp_dir / "manifest.jsonl", manifest_rows)
        _write_json(tmp_dir / "interpolation_summary.json", result)
        tmp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return result


def main() -> None:
    result = interpolate_prefix_caches(tyro.cli(Args))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
