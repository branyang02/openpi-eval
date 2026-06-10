from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import torch
import tyro
from PIL import Image, ImageDraw

from world_model.config import DatasetConfig, DatasetSource
from world_model.data import CachedFutureDataset, create_dataset
from world_model.media import chw_float_to_uint8
from world_model.wan22 import conditioning_frame_mae, read_video_frames

WAN_DECODE_SOURCES = {"wan2_2", "wan_lora"}


@dataclasses.dataclass
class Args:
    cache_dir: str = "output/future_cache"
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_key: str = "corner4.image"
    output_dir: str | None = None
    episodes: tuple[int, ...] | None = None
    max_samples: int | None = None
    samples_per_episode: int | None = None
    synthetic_samples: int = 8
    image_size: int = 64
    frame_delta: int = 1
    num_future_frames: int = 4
    action_horizon: int = 8
    seed: int = 7
    visual_samples: int = 4
    visual_indices: tuple[int, ...] | None = None
    visual_view_index: int = 0
    visual_tile_size: int = 96


def _psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _valid_future_frames(mask: torch.Tensor, num_future_frames: int) -> torch.Tensor:
    valid_frames = mask.detach().cpu().to(dtype=torch.bool).flatten()
    if valid_frames.numel() != num_future_frames:
        raise ValueError(f"future_image_mask length {valid_frames.numel()} does not match {num_future_frames}")
    return valid_frames


def _sample_future_error(
    ground_truth: torch.Tensor,
    cached: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any] | None:
    if tuple(ground_truth.shape) != tuple(cached.shape):
        raise ValueError(f"Future shape mismatch: {tuple(cached.shape)} != {tuple(ground_truth.shape)}")

    valid_frames = _valid_future_frames(mask, int(ground_truth.shape[0]))
    if not bool(valid_frames.any()):
        return None

    diff = cached.detach().cpu()[valid_frames] - ground_truth.detach().cpu()[valid_frames]
    squared = diff.square()
    absolute = diff.abs()
    mse = float(squared.mean().item())
    mae = float(absolute.mean().item())
    return {
        "num_pixels": int(diff.numel()),
        "future_shape": list(cached.shape),
        "future_mse": mse,
        "future_mae": mae,
        "future_psnr": _psnr_from_mse(mse),
        "max_abs_error": float(absolute.max().item()),
        "valid_future_frames": int(valid_frames.sum().item()),
    }


def _normalize_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list | tuple):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def _resolve_cache_path(cache_dir: Path, value: Any, *, field_name: str) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value:
        return None, f"missing_{field_name}"
    root = cache_dir.resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        return None, f"{field_name}_escapes_cache_dir"
    if not path.exists():
        return None, f"missing_{field_name}_file"
    return path, None


def _sample_wan_decode_proof(
    *,
    cache_dir: Path,
    row: dict[str, Any],
    cached_future: torch.Tensor,
    image_size: int,
) -> dict[str, Any] | None:
    source = row.get("source")
    if source not in WAN_DECODE_SOURCES and row.get("selected_frame_indices") is None:
        return None

    future_shape = [int(dim) for dim in cached_future.shape]
    manifest_future_shape = row.get("future_shape")
    selected_frame_indices = _normalize_int_list(row.get("selected_frame_indices"))
    issues: list[str] = []
    proof: dict[str, Any] = {
        "source": source,
        "future_tensor_shape": future_shape,
        "manifest_future_shape": manifest_future_shape,
        "future_shape_matches_manifest": manifest_future_shape is None or manifest_future_shape == future_shape,
        "selected_frame_indices": selected_frame_indices,
        "manifest_total_video_frames": row.get("total_video_frames"),
        "video": row.get("video"),
        "current_image": row.get("current_image"),
    }

    if manifest_future_shape is not None and manifest_future_shape != future_shape:
        issues.append("future_shape_mismatch")

    if selected_frame_indices is None:
        issues.append("missing_or_invalid_selected_frame_indices")
    else:
        proof["selected_frame_count"] = len(selected_frame_indices)
        proof["selected_frame_count_matches_future"] = len(selected_frame_indices) == future_shape[0]
        if len(selected_frame_indices) != future_shape[0]:
            issues.append("selected_frame_count_mismatch")

    video_path, video_issue = _resolve_cache_path(cache_dir, row.get("video"), field_name="video")
    current_path, current_issue = _resolve_cache_path(cache_dir, row.get("current_image"), field_name="current_image")
    issues.extend(issue for issue in (video_issue, current_issue) if issue is not None)
    if video_path is None or current_path is None:
        proof["issues"] = issues
        proof["passed"] = False
        return proof

    try:
        frames = read_video_frames(video_path)
    except Exception as exc:  # pragma: no cover - exact backend errors vary by imageio/ffmpeg install.
        proof["video_read_error"] = f"{type(exc).__name__}: {exc}"
        issues.append("video_read_failed")
        proof["issues"] = issues
        proof["passed"] = False
        return proof

    decoded_frame_count = len(frames)
    proof["decoded_video_frame_count"] = decoded_frame_count
    manifest_total_frames = row.get("total_video_frames")
    if manifest_total_frames is not None:
        proof["video_frame_count_matches_manifest"] = int(manifest_total_frames) == decoded_frame_count
        if int(manifest_total_frames) != decoded_frame_count:
            issues.append("video_frame_count_mismatch")

    if selected_frame_indices is not None:
        selected_indices_within_video = all(0 <= index < decoded_frame_count for index in selected_frame_indices)
        proof["selected_indices_within_video"] = selected_indices_within_video
        if not selected_indices_within_video:
            issues.append("selected_frame_index_out_of_bounds")

    if decoded_frame_count == 0:
        issues.append("video_has_no_frames")
    else:
        with Image.open(current_path) as current_image:
            proof["conditioning_frame_mae"] = conditioning_frame_mae(
                frames[0],
                current_image,
                image_size=image_size,
            )

    proof["issues"] = issues
    proof["passed"] = not issues
    return proof


def _summarize_wan_decode_proofs(per_sample: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows_with_proofs = [row for row in per_sample if "wan_decode_proof" in row]
    if not rows_with_proofs:
        return None

    proofs = [row["wan_decode_proof"] for row in rows_with_proofs]
    conditioning_maes = [
        float(proof["conditioning_frame_mae"]) for proof in proofs if proof.get("conditioning_frame_mae") is not None
    ]
    frame_counts = sorted(
        {
            int(proof["decoded_video_frame_count"])
            for proof in proofs
            if proof.get("decoded_video_frame_count") is not None
        }
    )
    selected_indices = sorted(
        {tuple(proof["selected_frame_indices"]) for proof in proofs if proof.get("selected_frame_indices") is not None}
    )
    return {
        "num_samples": len(proofs),
        "all_samples_passed": all(bool(proof.get("passed")) for proof in proofs),
        "decoded_video_frame_counts": frame_counts,
        "selected_frame_indices": [list(indices) for indices in selected_indices],
        "max_conditioning_frame_mae": max(conditioning_maes) if conditioning_maes else None,
        "issues": [
            {
                "cache_index": int(row["cache_index"]),
                "dataset_index": int(row["dataset_index"]),
                "issues": row["wan_decode_proof"].get("issues", []),
            }
            for row in rows_with_proofs
            if row["wan_decode_proof"].get("issues")
        ],
    }


def _make_labeled_tile(image: torch.Tensor, label: str, tile_size: int) -> Image.Image:
    label_height = 18
    canvas = Image.new("RGB", (tile_size, tile_size + label_height), (250, 250, 248))
    tile = Image.fromarray(chw_float_to_uint8(image)).resize((tile_size, tile_size))
    canvas.paste(tile, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    max_text_width = tile_size - 8
    text = label
    if draw.textlength(text) > max_text_width:
        for length in range(len(label), 0, -1):
            candidate = f"{label[:length]}..."
            if draw.textlength(candidate) <= max_text_width:
                text = candidate
                break
    draw.text((4, 3), text, fill=(30, 32, 36))
    return canvas


def _hstack(images: list[Image.Image], *, gap: int, background: tuple[int, int, int]) -> Image.Image:
    width = sum(image.width for image in images) + gap * max(len(images) - 1, 0)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), background)
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    return canvas


def _vstack(images: list[Image.Image], *, gap: int, background: tuple[int, int, int]) -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * max(len(images) - 1, 0)
    canvas = Image.new("RGB", (width, height), background)
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height + gap
    return canvas


def write_contact_sheet(
    *,
    base_dataset,
    cached_dataset: CachedFutureDataset,
    path: Path,
    cache_indices: list[int],
    view_index: int,
    tile_size: int,
) -> None:
    if not cache_indices:
        return

    rows: list[Image.Image] = []
    background = (244, 244, 241)
    for cache_index in cache_indices:
        if not 0 <= cache_index < len(cached_dataset):
            raise IndexError(
                f"visual cache index {cache_index} is out of range for cache length {len(cached_dataset)}."
            )
        row = cached_dataset.rows[cache_index]
        dataset_index = int(row["dataset_index"])
        ground_truth_item = base_dataset[dataset_index]
        cached_item = cached_dataset[cache_index]

        num_views = int(ground_truth_item["current_images"].shape[0])
        if not 0 <= view_index < num_views:
            raise ValueError(f"visual_view_index must be in [0, {num_views}), got {view_index}.")

        current = ground_truth_item["current_images"][view_index]
        ground_truth_futures = ground_truth_item["future_images"][:, view_index]
        cached_futures = cached_item["future_images"][:, view_index]

        gt_tiles = [_make_labeled_tile(current, f"s{dataset_index} current", tile_size)]
        gt_tiles.extend(
            _make_labeled_tile(frame, f"gt t+{offset + 1}", tile_size)
            for offset, frame in enumerate(ground_truth_futures)
        )
        cached_tiles = [_make_labeled_tile(current, f"s{dataset_index} current", tile_size)]
        cached_tiles.extend(
            _make_labeled_tile(frame, f"cache t+{offset + 1}", tile_size) for offset, frame in enumerate(cached_futures)
        )
        rows.append(_hstack(gt_tiles, gap=4, background=background))
        rows.append(_hstack(cached_tiles, gap=4, background=background))

    path.parent.mkdir(parents=True, exist_ok=True)
    _vstack(rows, gap=8, background=background).save(path)


def main(args: Args) -> None:
    dataset_config = DatasetConfig(
        source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=(args.image_key,),
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
    base_dataset = create_dataset(dataset_config)
    cached_dataset = CachedFutureDataset(base_dataset, args.cache_dir)

    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(args.cache_dir) / "quality_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample = []
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_pixels = 0
    max_abs_error = 0.0
    skipped_samples: list[dict[str, object]] = []

    for cache_index, row in enumerate(cached_dataset.rows):
        dataset_index = int(row["dataset_index"])
        ground_truth_item = base_dataset[dataset_index]
        valid_frames = _valid_future_frames(
            ground_truth_item["future_image_mask"],
            int(ground_truth_item["future_images"].shape[0]),
        )
        if not bool(valid_frames.any()):
            skipped_samples.append(
                {
                    "cache_index": cache_index,
                    "dataset_index": dataset_index,
                    "source": row.get("source", "unknown"),
                    "reason": "no_valid_future_frames",
                }
            )
            continue

        cached_item = cached_dataset[cache_index]
        sample_metrics = _sample_future_error(
            ground_truth=ground_truth_item["future_images"],
            cached=cached_item["future_images"],
            mask=ground_truth_item["future_image_mask"],
        )
        if sample_metrics is None:
            skipped_samples.append(
                {
                    "cache_index": cache_index,
                    "dataset_index": dataset_index,
                    "source": row.get("source", "unknown"),
                    "reason": "no_valid_future_frames",
                }
            )
            continue
        sample_pixels = int(sample_metrics["num_pixels"])
        total_pixels += sample_pixels
        total_squared_error += float(sample_metrics["future_mse"]) * sample_pixels
        total_absolute_error += float(sample_metrics["future_mae"]) * sample_pixels
        max_abs_error = max(max_abs_error, float(sample_metrics["max_abs_error"]))
        per_sample_row = {
            "cache_index": cache_index,
            "dataset_index": dataset_index,
            "source": row.get("source", "unknown"),
            **sample_metrics,
        }
        wan_decode_proof = _sample_wan_decode_proof(
            cache_dir=Path(args.cache_dir),
            row=row,
            cached_future=cached_item["future_images"],
            image_size=args.image_size,
        )
        if wan_decode_proof is not None:
            per_sample_row["wan_decode_proof"] = wan_decode_proof
        per_sample.append(per_sample_row)

    if total_pixels == 0:
        if skipped_samples:
            skipped_cache_indices = [sample["cache_index"] for sample in skipped_samples]
            raise ValueError(
                f"All {len(skipped_samples)} cached future samples were skipped because none had valid future frames. "
                f"Skipped cache indices: {skipped_cache_indices}"
            )
        raise ValueError("No cached future pixels were available for evaluation.")

    future_mse = total_squared_error / total_pixels
    future_mae = total_absolute_error / total_pixels
    wan_decode_validation = _summarize_wan_decode_proofs(per_sample)
    per_sample_path = output_dir / "per_sample_metrics.jsonl"
    per_sample_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in per_sample) + "\n")

    contact_sheet_path = output_dir / "future_cache_contact_sheet.png"
    visual_indices = (
        list(args.visual_indices)
        if args.visual_indices is not None
        else [int(row["cache_index"]) for row in per_sample[: max(args.visual_samples, 0)]]
    )
    write_contact_sheet(
        base_dataset=base_dataset,
        cached_dataset=cached_dataset,
        path=contact_sheet_path,
        cache_indices=visual_indices,
        view_index=args.visual_view_index,
        tile_size=args.visual_tile_size,
    )

    output = {
        "cache_dir": str(args.cache_dir),
        "dataset_config": dataclasses.asdict(dataset_config),
        "num_samples": len(per_sample),
        "num_total_samples": len(cached_dataset),
        "num_skipped_samples": len(skipped_samples),
        "skipped_samples": skipped_samples,
        "future_mse": future_mse,
        "future_mae": future_mae,
        "future_psnr": _psnr_from_mse(future_mse),
        "max_abs_error": max_abs_error,
        "per_sample_metrics": str(per_sample_path),
        "contact_sheet": str(contact_sheet_path) if visual_indices else None,
        "visual_indices": visual_indices,
        "wan_decode_validation": wan_decode_validation,
    }
    (output_dir / "future_cache_metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
