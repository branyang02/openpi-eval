from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from world_model.config import DatasetConfig, ModelConfig
from world_model.data import (
    GeneratedWanLatentDataset,
    SyntheticMetaWorldFramePairDataset,
    generated_wan_latent_cache_metadata,
)


def _dataset_config(**overrides: Any) -> DatasetConfig:
    values = {
        "source": "synthetic",
        "repo_id": "brandonyang/generated-wan-test",
        "image_keys": ("corner4.image",),
        "image_size": 16,
        "frame_delta": 1,
        "num_future_frames": 4,
        "action_horizon": 4,
        "synthetic_samples": 5,
        "episodes": (3, 7),
    }
    values.update(overrides)
    return DatasetConfig(**values)


def _model_config(**overrides: Any) -> ModelConfig:
    values = {
        "num_views": 1,
        "num_future_frames": 4,
        "image_size": 16,
        "state_dim": 4,
        "action_dim": 4,
        "action_horizon": 4,
        "latent_dim": 32,
        "idm_arch": "flow_transformer",
        "idm_visual_encoder": "wan_vae",
        "idm_transformer_layers": 1,
        "idm_transformer_heads": 4,
        "wan_vae_latent_channels": 8,
        "wan_vae_spatial_stride": 8,
        "wan_vae_use_cached_latents": True,
    }
    values.update(overrides)
    return ModelConfig(**values)


def _generator_metadata(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "source": "wan_lora",
        "checkpoint": "Wan2.2-TI2V-5B",
        "lora": ("metaworld-rank8.safetensors",),
        "seed": 123,
        "num_inference_steps": 8,
        "stop_after_steps": 4,
    }
    values.update(overrides)
    return values


def _latent_shape(dataset_config: DatasetConfig, model_config: ModelConfig) -> tuple[int, int, int, int]:
    latent_frames = (1 + dataset_config.num_future_frames + 3) // 4
    latent_side = dataset_config.image_size // model_config.wan_vae_spatial_stride
    return (
        model_config.wan_vae_latent_channels,
        latent_frames,
        latent_side,
        latent_side,
    )


def _tensor(shape: tuple[int, int, int, int], offset: int = 0) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    return (torch.arange(numel, dtype=torch.float32).reshape(shape) + offset).to(dtype=torch.float16)


def _write_cache(
    cache_dir: Path,
    *,
    dataset_config: DatasetConfig | None = None,
    model_config: ModelConfig | None = None,
    generator_metadata: dict[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
    tensors: dict[str, Any] | None = None,
    metadata_overrides: dict[str, Any] | None = None,
    add_row_generator_metadata: bool = True,
) -> list[dict[str, Any]]:
    dataset_config = dataset_config or _dataset_config()
    model_config = model_config or _model_config()
    generator_metadata = generator_metadata or _generator_metadata()
    shape = _latent_shape(dataset_config, model_config)
    if rows is None:
        rows = [
            {
                "dataset_index": 1,
                "latent_tensor": "latents/sample_000001.pt",
                "latent_shape": list(shape),
            },
            {
                "dataset_index": 3,
                "latent_tensor": "latents/sample_000003.pt",
                "latent_shape": list(shape),
            },
        ]
    if tensors is None:
        tensors = {
            "latents/sample_000001.pt": _tensor(shape, offset=1),
            "latents/sample_000003.pt": _tensor(shape, offset=3),
        }

    metadata = generated_wan_latent_cache_metadata(
        dataset_config=dataset_config,
        wan_vae_latent_channels=model_config.wan_vae_latent_channels,
        wan_vae_spatial_stride=model_config.wan_vae_spatial_stride,
        generator_metadata=generator_metadata,
        num_samples=len(rows),
    )
    if metadata_overrides is not None:
        metadata.update(metadata_overrides)
    if add_row_generator_metadata:
        rows = [
            {
                **row,
                "generator_metadata": row.get("generator_metadata", metadata["generator"]),
            }
            for row in rows
        ]

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (cache_dir / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for relative_path, value in tensors.items():
        path = cache_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
    return rows


def test_generated_wan_latent_cache_happy_path_preserves_base_item_and_task_text(tmp_path) -> None:
    dataset_config = _dataset_config()
    model_config = _model_config()
    generator_metadata = _generator_metadata()
    rows = _write_cache(
        tmp_path,
        dataset_config=dataset_config,
        model_config=model_config,
        generator_metadata=generator_metadata,
    )
    base_dataset = SyntheticMetaWorldFramePairDataset(dataset_config)
    dataset = GeneratedWanLatentDataset(
        base_dataset,
        tmp_path,
        model_config,
        generator_metadata=generator_metadata,
    )

    item = dataset[0]
    base_item = base_dataset[1]
    expected_latents = torch.load(tmp_path / rows[0]["latent_tensor"], map_location="cpu", weights_only=True)

    assert len(dataset) == 2
    assert dataset.task_text(0) == base_dataset.task_text(1)
    assert torch.allclose(item["current_images"], base_item["current_images"])
    assert torch.allclose(item["future_images"], base_item["future_images"])
    assert torch.allclose(item["state"], base_item["state"])
    assert torch.equal(item["task_id"], base_item["task_id"])
    assert torch.allclose(item["action_chunk"], base_item["action_chunk"])
    assert torch.allclose(item["action_mask"], base_item["action_mask"])
    assert item["wan_vae_latents"].dtype == torch.float32
    assert torch.allclose(item["wan_vae_latents"], expected_latents.to(dtype=torch.float32))


def test_generated_wan_latent_cache_records_idm_history_length_and_accepts_match(tmp_path) -> None:
    dataset_config = _dataset_config(idm_history_length=2)
    model_config = _model_config(idm_history_length=2)
    generator_metadata = _generator_metadata()
    _write_cache(
        tmp_path,
        dataset_config=dataset_config,
        model_config=model_config,
        generator_metadata=generator_metadata,
    )
    metadata = json.loads((tmp_path / "config.json").read_text())

    assert metadata["idm_history_length"] == 2
    assert metadata["dataset_config"]["idm_history_length"] == 2

    dataset = GeneratedWanLatentDataset(
        SyntheticMetaWorldFramePairDataset(dataset_config),
        tmp_path,
        model_config,
        generator_metadata=generator_metadata,
    )

    assert len(dataset) == 2
    assert dataset[0]["prev_action_history"].shape == (2, 4)


def test_generated_wan_latent_cache_rejects_legacy_no_history_metadata_for_history_dataset(tmp_path) -> None:
    dataset_config = _dataset_config()
    model_config = _model_config()
    generator_metadata = _generator_metadata()
    _write_cache(
        tmp_path,
        dataset_config=dataset_config,
        model_config=model_config,
        generator_metadata=generator_metadata,
    )
    metadata = json.loads((tmp_path / "config.json").read_text())
    metadata.pop("idm_history_length", None)
    metadata["dataset_config"].pop("idm_history_length", None)
    (tmp_path / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    zero_history_dataset = GeneratedWanLatentDataset(
        SyntheticMetaWorldFramePairDataset(dataset_config),
        tmp_path,
        model_config,
        generator_metadata=generator_metadata,
    )
    assert len(zero_history_dataset) == 2

    with pytest.raises(ValueError, match="metadata mismatch.*idm_history_length"):
        GeneratedWanLatentDataset(
            SyntheticMetaWorldFramePairDataset(_dataset_config(idm_history_length=2)),
            tmp_path,
            _model_config(idm_history_length=2),
            generator_metadata=generator_metadata,
        )


def test_generated_wan_latent_cache_rejects_metadata_mismatch(tmp_path) -> None:
    dataset_config = _dataset_config()
    model_config = _model_config()
    generator_metadata = _generator_metadata()
    _write_cache(
        tmp_path,
        dataset_config=dataset_config,
        model_config=model_config,
        generator_metadata=generator_metadata,
    )

    mismatched_base = SyntheticMetaWorldFramePairDataset(_dataset_config(frame_delta=2))
    with pytest.raises(ValueError, match="Generated Wan latent cache metadata mismatch.*frame_delta"):
        GeneratedWanLatentDataset(
            mismatched_base,
            tmp_path,
            model_config,
            generator_metadata=generator_metadata,
        )

    base_dataset = SyntheticMetaWorldFramePairDataset(dataset_config)
    with pytest.raises(ValueError, match="Generated Wan latent cache metadata mismatch.*generator"):
        GeneratedWanLatentDataset(
            base_dataset,
            tmp_path,
            model_config,
            generator_metadata=_generator_metadata(seed=999),
        )


def test_generated_wan_latent_cache_allows_row_generator_metadata_extra_fields(tmp_path) -> None:
    dataset_config = _dataset_config()
    model_config = _model_config()
    generator_metadata = _generator_metadata()
    shape = _latent_shape(dataset_config, model_config)
    rows = [
        {
            "dataset_index": 1,
            "latent_tensor": "latents/sample_000001.pt",
            "latent_shape": list(shape),
            "generator_metadata": {
                **generator_metadata,
                "row_seed": 1001,
                "row_latent_shape": list(shape),
            },
        },
    ]
    _write_cache(
        tmp_path,
        dataset_config=dataset_config,
        model_config=model_config,
        generator_metadata=generator_metadata,
        rows=rows,
        tensors={"latents/sample_000001.pt": _tensor(shape, offset=1)},
    )

    dataset = GeneratedWanLatentDataset(
        SyntheticMetaWorldFramePairDataset(dataset_config),
        tmp_path,
        model_config,
        generator_metadata=generator_metadata,
    )

    assert len(dataset) == 1


def test_generated_wan_latent_cache_rejects_row_generator_metadata_mismatch(tmp_path) -> None:
    dataset_config = _dataset_config()
    model_config = _model_config()
    generator_metadata = _generator_metadata()
    shape = _latent_shape(dataset_config, model_config)
    rows = [
        {
            "dataset_index": 1,
            "latent_tensor": "latents/sample_000001.pt",
            "latent_shape": list(shape),
            "generator_metadata": {
                **generator_metadata,
                "stop_after_steps": None,
                "row_seed": 1001,
            },
        },
    ]
    _write_cache(
        tmp_path,
        dataset_config=dataset_config,
        model_config=model_config,
        generator_metadata=generator_metadata,
        rows=rows,
        tensors={"latents/sample_000001.pt": _tensor(shape, offset=1)},
    )

    with pytest.raises(ValueError, match="generator_metadata mismatch.*dataset_index=1.*stop_after_steps"):
        GeneratedWanLatentDataset(
            SyntheticMetaWorldFramePairDataset(dataset_config),
            tmp_path,
            model_config,
            generator_metadata=generator_metadata,
        )


def test_generated_wan_latent_cache_rejects_missing_row_generator_metadata(tmp_path) -> None:
    dataset_config = _dataset_config()
    model_config = _model_config()
    shape = _latent_shape(dataset_config, model_config)
    _write_cache(
        tmp_path,
        dataset_config=dataset_config,
        model_config=model_config,
        rows=[
            {
                "dataset_index": 1,
                "latent_tensor": "latents/sample_000001.pt",
                "latent_shape": list(shape),
            },
        ],
        tensors={"latents/sample_000001.pt": _tensor(shape, offset=1)},
        add_row_generator_metadata=False,
    )

    with pytest.raises(ValueError, match="dataset_index=1.*missing generator_metadata"):
        GeneratedWanLatentDataset(
            SyntheticMetaWorldFramePairDataset(dataset_config),
            tmp_path,
            model_config,
            generator_metadata=_generator_metadata(),
        )


def test_generated_wan_latent_cache_rejects_missing_and_empty_manifest(tmp_path) -> None:
    dataset_config = _dataset_config()
    model_config = _model_config()
    generator_metadata = _generator_metadata()
    metadata = generated_wan_latent_cache_metadata(
        dataset_config=dataset_config,
        wan_vae_latent_channels=model_config.wan_vae_latent_channels,
        wan_vae_spatial_stride=model_config.wan_vae_spatial_stride,
        generator_metadata=generator_metadata,
        num_samples=0,
    )
    (tmp_path / "config.json").write_text(json.dumps(metadata) + "\n")
    base_dataset = SyntheticMetaWorldFramePairDataset(dataset_config)

    with pytest.raises(FileNotFoundError, match="Generated Wan latent cache manifest not found"):
        GeneratedWanLatentDataset(
            base_dataset,
            tmp_path,
            model_config,
            generator_metadata=generator_metadata,
        )

    (tmp_path / "manifest.jsonl").write_text("")
    with pytest.raises(ValueError, match="Generated Wan latent cache manifest is empty"):
        GeneratedWanLatentDataset(
            base_dataset,
            tmp_path,
            model_config,
            generator_metadata=generator_metadata,
        )


def test_generated_wan_latent_cache_rejects_path_escape(tmp_path) -> None:
    dataset_config = _dataset_config()
    model_config = _model_config()
    shape = _latent_shape(dataset_config, model_config)
    _write_cache(
        tmp_path,
        dataset_config=dataset_config,
        model_config=model_config,
        rows=[
            {
                "dataset_index": 0,
                "latent_tensor": "../escape.pt",
                "latent_shape": list(shape),
            }
        ],
        tensors={},
    )

    with pytest.raises(ValueError, match="Generated Wan latent path escapes cache directory"):
        GeneratedWanLatentDataset(
            SyntheticMetaWorldFramePairDataset(dataset_config),
            tmp_path,
            model_config,
            generator_metadata=_generator_metadata(),
        )


def test_generated_wan_latent_cache_rejects_duplicate_and_out_of_range_rows(tmp_path) -> None:
    dataset_config = _dataset_config()
    model_config = _model_config()
    shape = _latent_shape(dataset_config, model_config)
    duplicate_rows = [
        {"dataset_index": 0, "latent_tensor": "latents/a.pt", "latent_shape": list(shape)},
        {"dataset_index": 0, "latent_tensor": "latents/b.pt", "latent_shape": list(shape)},
    ]
    _write_cache(tmp_path / "duplicate", dataset_config=dataset_config, model_config=model_config, rows=duplicate_rows)

    with pytest.raises(ValueError, match="duplicate dataset_index=0"):
        GeneratedWanLatentDataset(
            SyntheticMetaWorldFramePairDataset(dataset_config),
            tmp_path / "duplicate",
            model_config,
            generator_metadata=_generator_metadata(),
        )

    out_of_range_rows = [
        {"dataset_index": 5, "latent_tensor": "latents/a.pt", "latent_shape": list(shape)},
    ]
    _write_cache(
        tmp_path / "out_of_range", dataset_config=dataset_config, model_config=model_config, rows=out_of_range_rows
    )

    with pytest.raises(ValueError, match="dataset_index=5 is outside base dataset length"):
        GeneratedWanLatentDataset(
            SyntheticMetaWorldFramePairDataset(dataset_config),
            tmp_path / "out_of_range",
            model_config,
            generator_metadata=_generator_metadata(),
        )


def test_generated_wan_latent_cache_rejects_non_tensor_payload(tmp_path) -> None:
    dataset_config = _dataset_config()
    model_config = _model_config()
    shape = _latent_shape(dataset_config, model_config)
    rows = [
        {
            "dataset_index": 0,
            "latent_tensor": "latents/not_tensor.pt",
            "latent_shape": list(shape),
        }
    ]
    _write_cache(
        tmp_path,
        dataset_config=dataset_config,
        model_config=model_config,
        rows=rows,
        tensors={"latents/not_tensor.pt": {"not": "a tensor"}},
    )
    dataset = GeneratedWanLatentDataset(
        SyntheticMetaWorldFramePairDataset(dataset_config),
        tmp_path,
        model_config,
        generator_metadata=_generator_metadata(),
    )

    with pytest.raises(TypeError, match="Generated Wan latents must be a torch.Tensor"):
        dataset[0]


def test_generated_wan_latent_cache_rejects_tensor_shape_mismatch(tmp_path) -> None:
    dataset_config = _dataset_config()
    model_config = _model_config()
    shape = _latent_shape(dataset_config, model_config)
    rows = [
        {
            "dataset_index": 0,
            "latent_tensor": "latents/wrong_shape.pt",
            "latent_shape": list(shape),
        }
    ]
    _write_cache(
        tmp_path,
        dataset_config=dataset_config,
        model_config=model_config,
        rows=rows,
        tensors={"latents/wrong_shape.pt": torch.zeros((shape[0], shape[1] + 1, shape[2], shape[3]))},
    )
    dataset = GeneratedWanLatentDataset(
        SyntheticMetaWorldFramePairDataset(dataset_config),
        tmp_path,
        model_config,
        generator_metadata=_generator_metadata(),
    )

    with pytest.raises(ValueError, match="Generated Wan latent tensor shape does not match manifest"):
        dataset[0]


def test_generated_wan_latent_cache_rejects_non_rank4_tensor(tmp_path) -> None:
    dataset_config = _dataset_config()
    model_config = _model_config()
    shape = _latent_shape(dataset_config, model_config)
    rows = [
        {
            "dataset_index": 0,
            "latent_tensor": "latents/rank5.pt",
            "latent_shape": list(shape),
        }
    ]
    _write_cache(
        tmp_path,
        dataset_config=dataset_config,
        model_config=model_config,
        rows=rows,
        tensors={"latents/rank5.pt": torch.zeros((1, *shape))},
    )
    dataset = GeneratedWanLatentDataset(
        SyntheticMetaWorldFramePairDataset(dataset_config),
        tmp_path,
        model_config,
        generator_metadata=_generator_metadata(),
    )

    with pytest.raises(ValueError, match="must have rank 4"):
        dataset[0]
