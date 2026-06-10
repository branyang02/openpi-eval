from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from compare_wan_latent_caches import Args, compare_wan_latent_caches


def _dataset_metadata(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "source": "synthetic",
        "repo_id": "brandonyang/metaworld_ml45",
        "image_keys": ["corner4.image"],
        "episodes": [1, 3],
        "samples_per_episode": 2,
        "frame_delta": 1,
        "num_future_frames": 4,
        "action_horizon": 4,
        "image_size": 16,
    }
    values.update(overrides)
    return values


def _reference_config(dataset_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "repo_id": dataset_metadata["repo_id"],
        "episodes": dataset_metadata["episodes"],
        "samples_per_episode": dataset_metadata["samples_per_episode"],
        "frame_delta": dataset_metadata["frame_delta"],
        "num_future_frames": dataset_metadata["num_future_frames"],
        "image_size": dataset_metadata["image_size"],
        "image_keys": dataset_metadata["image_keys"],
        "wan_vae_checkpoint_path": "fake-wan-vae.ckpt",
        "wan_vae_dtype": "float32",
        "wan_vae_latent_channels": 1,
        "wan_vae_spatial_stride": 16,
        "num_samples": 0,
        "dataset_config": dataset_metadata,
    }


def _generated_config(dataset_metadata: dict[str, Any], generator_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_schema": "generated_wan_latents",
        "version": 1,
        "repo_id": dataset_metadata["repo_id"],
        "episodes": dataset_metadata["episodes"],
        "samples_per_episode": dataset_metadata["samples_per_episode"],
        "frame_delta": dataset_metadata["frame_delta"],
        "num_future_frames": dataset_metadata["num_future_frames"],
        "image_size": dataset_metadata["image_size"],
        "image_keys": dataset_metadata["image_keys"],
        "wan_vae_latent_channels": 1,
        "wan_vae_spatial_stride": 16,
        "generator": generator_metadata,
        "num_samples": 0,
        "dataset_config": dataset_metadata,
    }


def _generator_metadata(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "source": "diffsynth_wan_lora",
        "denoise_mode": "partial",
        "denoise_fraction": 0.5,
        "denoise_steps_run": 1,
        "num_inference_steps": 2,
        "stop_after_steps": 1,
    }
    values.update(overrides)
    return values


def _write_cache(
    cache_dir: Path,
    *,
    tensors_by_index: dict[int, torch.Tensor],
    cache_kind: str,
    dataset_metadata: dict[str, Any] | None = None,
    generator_metadata: dict[str, Any] | None = None,
) -> None:
    dataset_metadata = dataset_metadata or _dataset_metadata()
    generator_metadata = generator_metadata or _generator_metadata()
    config = (
        _reference_config(dataset_metadata)
        if cache_kind == "reference"
        else _generated_config(dataset_metadata, generator_metadata)
    )
    config["num_samples"] = len(tensors_by_index)

    rows = []
    for dataset_index, tensor in tensors_by_index.items():
        relative_path = f"latents/sample_{dataset_index:06d}.pt"
        row: dict[str, Any] = {
            "dataset_index": dataset_index,
            "latent_tensor": relative_path,
            "latent_shape": list(tensor.shape),
        }
        if cache_kind == "generated":
            row["generator_metadata"] = {
                **generator_metadata,
                "completed_denoise_steps": generator_metadata.get("denoise_steps_run"),
            }
        rows.append(row)

        tensor_path = cache_dir / relative_path
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, tensor_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (cache_dir / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def _summary_for(tmp_path: Path, reference: Path, generated: Path) -> dict[str, Any]:
    payload = compare_wan_latent_caches(
        Args(
            reference_cache_dir=str(reference),
            generated_cache_dirs=(str(generated),),
            output_dir=str(tmp_path / "summary"),
        )
    )
    return payload["generated_cache_summaries"][0]


def test_compare_wan_latent_caches_computes_exact_metrics(tmp_path) -> None:
    reference = tmp_path / "reference"
    generated = tmp_path / "generated"
    reference_tensors = {
        0: torch.tensor([[[[0.0, 1.0]], [[2.0, 3.0]]]]),
        1: torch.tensor([[[[4.0, 5.0]], [[6.0, 7.0]]]]),
    }
    generated_tensors = {
        0: torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]),
        1: torch.tensor([[[[3.0, 4.0]], [[5.0, 6.0]]]]),
    }
    _write_cache(reference, tensors_by_index=reference_tensors, cache_kind="reference")
    _write_cache(generated, tensors_by_index=generated_tensors, cache_kind="generated")

    summary = _summary_for(tmp_path, reference, generated)

    cosine_0 = 20.0 / math.sqrt(14.0 * 30.0)
    cosine_1 = 104.0 / math.sqrt(126.0 * 86.0)
    assert summary["num_samples"] == 2
    assert summary["latent_mse"] == pytest.approx(1.0)
    assert summary["latent_mae"] == pytest.approx(1.0)
    assert summary["latent_rmse"] == pytest.approx(1.0)
    assert summary["reference_variance"] == pytest.approx(5.25)
    assert summary["normalized_mse_vs_reference_variance"] == pytest.approx(1.0 / 5.25)
    assert summary["cosine_similarity_mean"] == pytest.approx((cosine_0 + cosine_1) / 2.0)
    assert summary["cosine_distance_mean"] == pytest.approx((2.0 - cosine_0 - cosine_1) / 2.0)
    assert summary["reference_norm_mean"] == pytest.approx((math.sqrt(14.0) + math.sqrt(126.0)) / 2.0)
    assert summary["generated_norm_mean"] == pytest.approx((math.sqrt(30.0) + math.sqrt(86.0)) / 2.0)
    assert summary["delta_norm_mean"] == pytest.approx(2.0)
    assert summary["generated_to_reference_norm_ratio_mean"] == pytest.approx(
        (math.sqrt(30.0 / 14.0) + math.sqrt(86.0 / 126.0)) / 2.0
    )
    assert summary["per_time_mse"] == pytest.approx([1.0, 1.0])
    assert summary["denoise_metadata"] == {
        "completed_denoise_steps": 1,
        "denoise_fraction": 0.5,
        "denoise_mode": "partial",
        "denoise_steps_run": 1,
        "num_inference_steps": 2,
        "stop_after_steps": 1,
    }
    assert (tmp_path / "summary" / "latent_gap_summary.json").exists()
    assert (tmp_path / "summary" / "latent_gap_summary.md").exists()


def test_compare_wan_latent_caches_rejects_dataset_index_mismatch(tmp_path) -> None:
    reference = tmp_path / "reference"
    generated = tmp_path / "generated"
    tensor = torch.zeros((1, 2, 1, 2))
    _write_cache(reference, tensors_by_index={0: tensor, 1: tensor}, cache_kind="reference")
    _write_cache(generated, tensors_by_index={0: tensor, 2: tensor}, cache_kind="generated")

    with pytest.raises(ValueError, match="dataset_index set.*missing dataset_index values \\[1\\].*extra"):
        _summary_for(tmp_path, reference, generated)


def test_compare_wan_latent_caches_rejects_shape_mismatch(tmp_path) -> None:
    reference = tmp_path / "reference"
    generated = tmp_path / "generated"
    _write_cache(reference, tensors_by_index={0: torch.zeros((1, 2, 1, 2))}, cache_kind="reference")
    _write_cache(generated, tensors_by_index={0: torch.zeros((1, 2, 1, 3))}, cache_kind="generated")

    with pytest.raises(ValueError, match="Latent tensor shape mismatch.*dataset_index=0"):
        _summary_for(tmp_path, reference, generated)


def test_compare_wan_latent_caches_rejects_dataset_metadata_mismatch(tmp_path) -> None:
    reference = tmp_path / "reference"
    generated = tmp_path / "generated"
    tensor = torch.zeros((1, 2, 1, 2))
    _write_cache(reference, tensors_by_index={0: tensor}, cache_kind="reference")
    _write_cache(
        generated,
        tensors_by_index={0: tensor},
        cache_kind="generated",
        dataset_metadata=_dataset_metadata(frame_delta=2),
    )

    with pytest.raises(ValueError, match="Dataset metadata mismatch.*frame_delta"):
        _summary_for(tmp_path, reference, generated)
