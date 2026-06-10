from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import cache_generated_wan_latents as cache_module
from cache_generated_wan_latents import Args as CacheArgs
from cache_generated_wan_latents import _build_dataset_config, _build_model_config, precompute_generated_wan_latents
from world_model.data import GeneratedWanLatentDataset, create_dataset
from world_model.diffsynth_wan import WAN_LATENT_STAGE


def _cache_args(cache_dir: Path, **overrides) -> CacheArgs:
    values = {
        "dataset_source": "synthetic",
        "repo_id": "brandonyang/generated-wan-test",
        "image_keys": ("corner4.image",),
        "output_dir": str(cache_dir),
        "synthetic_samples": 3,
        "image_size": 16,
        "frame_delta": 1,
        "num_future_frames": 4,
        "action_horizon": 4,
        "batch_size": 2,
        "num_workers": 0,
        "device": "cpu",
        "seed": 5,
        "diffsynth_repo_dir": "/fake/DiffSynth-Studio",
        "checkpoint_dir": "/fake/Wan2.2-TI2V-5B",
        "lora_path": "/fake/epoch-0.safetensors",
        "height": 16,
        "width": 16,
        "num_frames": 5,
        "num_inference_steps": 2,
        "lora_alpha": 0.75,
        "tiled": False,
        "base_seed": 100,
        "prompt_template": "Task: {task}",
        "future_frame_strategy": "first",
        "wan_vae_latent_channels": 48,
        "wan_vae_spatial_stride": 16,
    }
    values.update(overrides)
    return CacheArgs(**values)


class _FakeGenerator:
    def __init__(self, latents: torch.Tensor | None = None, *, num_inference_steps: int = 2) -> None:
        self.latents = latents
        self.num_inference_steps = num_inference_steps
        self.calls: list[dict] = []

    def generate_view_latents(
        self,
        current_image: torch.Tensor,
        *,
        task_text: str,
        seed: int | None = None,
        stop_after_steps: int | None = None,
    ):
        self.calls.append(
            {
                "current_image_shape": tuple(current_image.shape),
                "task_text": task_text,
                "seed": seed,
                "stop_after_steps": stop_after_steps,
            }
        )
        latents = (
            torch.full((1, 48, 2, 1, 1), float(seed), dtype=torch.float32)
            if self.latents is None
            else self.latents.clone()
        )
        denoise_steps_run = self.num_inference_steps if stop_after_steps is None else stop_after_steps
        return SimpleNamespace(
            latents=latents,
            prompt=f"Task: {task_text}",
            seed=seed,
            metadata={
                "source": "diffsynth_wan_lora",
                "latent_stage": WAN_LATENT_STAGE,
                "seed": seed,
                "num_inference_steps": self.num_inference_steps,
                "denoise_steps_run": denoise_steps_run,
                "stop_after_steps": stop_after_steps,
                "denoise_fraction": denoise_steps_run / self.num_inference_steps,
                "denoise_mode": "partial" if denoise_steps_run < self.num_inference_steps else "full",
            },
        )


class _PartialMetadataGenerator(_FakeGenerator):
    def generate_view_latents(
        self,
        current_image: torch.Tensor,
        *,
        task_text: str,
        seed: int | None = None,
        stop_after_steps: int | None = None,
    ):
        self.calls.append(
            {
                "current_image_shape": tuple(current_image.shape),
                "task_text": task_text,
                "seed": seed,
                "stop_after_steps": stop_after_steps,
            }
        )
        latents = torch.full((1, 48, 2, 1, 1), float(seed), dtype=torch.float32)
        return SimpleNamespace(
            latents=latents,
            prompt=f"Task: {task_text}",
            seed=seed,
            metadata={
                "seed": seed,
                "row_tuple": ("normalized", seed),
            },
        )


class _MetadataOverrideGenerator(_FakeGenerator):
    def __init__(self, metadata_updates: dict, *, num_inference_steps: int = 2) -> None:
        super().__init__(num_inference_steps=num_inference_steps)
        self.metadata_updates = metadata_updates

    def generate_view_latents(
        self,
        current_image: torch.Tensor,
        *,
        task_text: str,
        seed: int | None = None,
        stop_after_steps: int | None = None,
    ):
        result = super().generate_view_latents(
            current_image,
            task_text=task_text,
            seed=seed,
            stop_after_steps=stop_after_steps,
        )
        result.metadata.update(self.metadata_updates)
        return result


def _read_manifest(cache_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (cache_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]


def test_generated_wan_latent_cache_happy_path_writes_config_manifest_and_tensors(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    args = _cache_args(cache_dir)
    generator = _FakeGenerator()

    result = precompute_generated_wan_latents(args, generator=generator)

    assert result["written"] == 3
    assert result["num_samples"] == 3
    assert result["elapsed_wall_seconds"] >= 0.0
    assert result["generator_load_wall_seconds"] == 0.0
    assert result["generation_wall_seconds"] >= 0.0
    assert result["generation_wall_seconds_mean"] >= 0.0
    assert result["write_wall_seconds"] >= 0.0
    config = json.loads((cache_dir / "config.json").read_text())
    assert config["cache_schema"] == "generated_wan_latents"
    assert config["num_samples"] == 3
    assert config["image_keys"] == ["corner4.image"]
    assert config["generator"]["source"] == "diffsynth_wan_lora"
    assert config["generator"]["checkpoint_dir"] == "/fake/Wan2.2-TI2V-5B"
    assert config["generator"]["lora_path"] == "/fake/epoch-0.safetensors"
    assert config["generator"]["height"] == 16
    assert config["generator"]["width"] == 16
    assert config["generator"]["num_frames"] == 5
    assert config["generator"]["num_inference_steps"] == 2
    assert config["generator"]["denoise_steps_run"] == 2
    assert config["generator"]["stop_after_steps"] is None
    assert config["generator"]["denoise_fraction"] == 1.0
    assert config["generator"]["denoise_mode"] == "full"
    assert config["generator"]["lora_alpha"] == 0.75
    assert config["generator"]["tiled"] is False
    assert config["generator"]["future_frame_strategy"] == "first"
    assert config["generator"]["latent_stage"] == "post_denoising_post_units_pre_vae_decode"

    rows = _read_manifest(cache_dir)
    assert [row["dataset_index"] for row in rows] == [0, 1, 2]
    assert rows[0]["latent_tensor"] == "latents/sample_000000.pt"
    assert rows[0]["latent_shape"] == [48, 2, 1, 1]
    assert rows[0]["seed"] == 100
    assert rows[0]["prompt"] == "Task: synthetic metaworld task 0"
    assert rows[0]["generator_metadata"]["seed"] == 100
    assert rows[0]["generator_metadata"]["latent_stage"] == "post_denoising_post_units_pre_vae_decode"
    assert rows[0]["generator_metadata"]["base_seed"] == 100
    assert rows[0]["generator_metadata"]["seed_strategy"] == "base_seed_plus_dataset_index"
    assert rows[0]["generator_metadata"]["future_frame_strategy"] == "first"
    assert rows[0]["generator_metadata"]["denoise_steps_run"] == 2
    assert rows[0]["generator_metadata"]["denoise_mode"] == "full"
    assert rows[0]["generation_wall_seconds"] >= 0.0

    latent = torch.load(cache_dir / rows[0]["latent_tensor"], map_location="cpu", weights_only=True)
    assert tuple(latent.shape) == (48, 2, 1, 1)
    assert torch.equal(latent, torch.full((48, 2, 1, 1), 100.0))
    assert [call["seed"] for call in generator.calls] == [100, 101, 102]
    assert {call["current_image_shape"] for call in generator.calls} == {(3, 16, 16)}


def test_generated_wan_latent_cache_cli_records_idm_history_length(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    args = _cache_args(cache_dir, synthetic_samples=1, idm_history_length=2)

    precompute_generated_wan_latents(args, generator=_FakeGenerator())

    config = json.loads((cache_dir / "config.json").read_text())
    assert config["idm_history_length"] == 2
    assert config["dataset_config"]["idm_history_length"] == 2

    dataset = GeneratedWanLatentDataset(
        create_dataset(_build_dataset_config(args)),
        cache_dir,
        _build_model_config(args),
        generator_metadata=config["generator"],
    )
    assert len(dataset) == 1
    assert dataset[0]["history_mask"].shape == (2,)


def test_generated_wan_latent_cache_records_partial_denoise_settings(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    args = _cache_args(cache_dir, synthetic_samples=1, num_inference_steps=3, stop_after_steps=1)
    generator = _FakeGenerator(num_inference_steps=3)

    precompute_generated_wan_latents(args, generator=generator)

    config = json.loads((cache_dir / "config.json").read_text())
    assert config["generator"]["num_inference_steps"] == 3
    assert config["generator"]["denoise_steps_run"] == 1
    assert config["generator"]["stop_after_steps"] == 1
    assert config["generator"]["denoise_fraction"] == pytest.approx(1 / 3)
    assert config["generator"]["denoise_mode"] == "partial"
    assert generator.calls[0]["stop_after_steps"] == 1

    rows = _read_manifest(cache_dir)
    assert rows[0]["generator_metadata"]["num_inference_steps"] == 3
    assert rows[0]["generator_metadata"]["denoise_steps_run"] == 1
    assert rows[0]["generator_metadata"]["stop_after_steps"] == 1
    assert rows[0]["generator_metadata"]["denoise_fraction"] == pytest.approx(1 / 3)
    assert rows[0]["generator_metadata"]["denoise_mode"] == "partial"


def test_generated_wan_latent_cache_records_deterministic_generation_and_write_timing(
    tmp_path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    args = _cache_args(cache_dir, synthetic_samples=2, batch_size=1)
    ticks = iter(float(value) for value in range(100, 110))
    monkeypatch.setattr(cache_module, "_perf_counter", lambda: next(ticks))

    result = precompute_generated_wan_latents(args, generator=_FakeGenerator())

    assert result["written"] == 2
    assert result["elapsed_wall_seconds"] == pytest.approx(9.0)
    assert result["generator_load_wall_seconds"] == 0.0
    assert result["generation_wall_seconds"] == pytest.approx(2.0)
    assert result["generation_wall_seconds_mean"] == pytest.approx(1.0)
    assert result["write_wall_seconds"] == pytest.approx(2.0)
    rows = _read_manifest(cache_dir)
    assert [row["generation_wall_seconds"] for row in rows] == pytest.approx([1.0, 1.0])


def test_generated_wan_latent_cache_preloads_real_generator_pipe_before_row_timing(
    tmp_path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    args = _cache_args(cache_dir, synthetic_samples=1, batch_size=1)

    class _FakeLazyPipeGenerator(_FakeGenerator):
        def __init__(self) -> None:
            super().__init__()
            self.pipe_accesses = 0

        @property
        def pipe(self) -> object:
            self.pipe_accesses += 1
            cache_module._perf_counter()
            return object()

    fake_generator = _FakeLazyPipeGenerator()
    monkeypatch.setattr(cache_module, "_build_generator", lambda build_args: fake_generator)
    ticks = iter(float(value) for value in range(100, 110))
    monkeypatch.setattr(cache_module, "_perf_counter", lambda: next(ticks))

    result = precompute_generated_wan_latents(args)

    assert fake_generator.pipe_accesses == 1
    assert result["written"] == 1
    assert result["generator_load_wall_seconds"] == pytest.approx(2.0)
    assert result["generation_wall_seconds"] == pytest.approx(1.0)
    assert result["generation_wall_seconds_mean"] == pytest.approx(1.0)
    rows = _read_manifest(cache_dir)
    assert rows[0]["generation_wall_seconds"] == pytest.approx(1.0)


def test_generated_wan_latent_cache_merges_partial_result_metadata_into_loadable_rows(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    args = _cache_args(cache_dir, synthetic_samples=1)
    generator = _PartialMetadataGenerator()

    precompute_generated_wan_latents(args, generator=generator)

    rows = _read_manifest(cache_dir)
    row_metadata = rows[0]["generator_metadata"]
    assert row_metadata["base_seed"] == 100
    assert row_metadata["seed_strategy"] == "base_seed_plus_dataset_index"
    assert row_metadata["future_frame_strategy"] == "first"
    assert row_metadata["seed"] == 100
    assert row_metadata["row_tuple"] == ["normalized", 100]

    config = json.loads((cache_dir / "config.json").read_text())
    dataset = GeneratedWanLatentDataset(
        create_dataset(_build_dataset_config(args)),
        cache_dir,
        _build_model_config(args),
        generator_metadata=config["generator"],
    )

    assert len(dataset) == 1
    assert tuple(dataset[0]["wan_vae_latents"].shape) == (48, 2, 1, 1)


def test_generated_wan_latent_cache_resume_skips_existing_manifest_rows(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    args = _cache_args(cache_dir, synthetic_samples=2)
    first_generator = _FakeGenerator()
    second_generator = _FakeGenerator()

    precompute_generated_wan_latents(args, generator=first_generator)
    result = precompute_generated_wan_latents(args, generator=second_generator)

    assert result["written"] == 0
    assert result["num_samples"] == 2
    assert result["generator_load_wall_seconds"] == 0.0
    assert result["generation_wall_seconds"] == 0.0
    assert result["generation_wall_seconds_mean"] == 0.0
    assert result["write_wall_seconds"] == 0.0
    assert len(_read_manifest(cache_dir)) == 2
    assert len(first_generator.calls) == 2
    assert second_generator.calls == []


def test_generated_wan_latent_cache_resume_accepts_legacy_zero_history_metadata(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    args = _cache_args(cache_dir, synthetic_samples=2)
    precompute_generated_wan_latents(args, generator=_FakeGenerator())
    config_path = cache_dir / "config.json"
    metadata = json.loads(config_path.read_text())
    metadata.pop("idm_history_length", None)
    metadata["dataset_config"].pop("idm_history_length", None)
    config_path.write_text(json.dumps(metadata, indent=2) + "\n")
    generator = _FakeGenerator()

    result = precompute_generated_wan_latents(args, generator=generator)

    assert result["written"] == 0
    assert result["num_samples"] == 2
    assert generator.calls == []


def test_generated_wan_latent_cache_rejects_multiple_views(tmp_path) -> None:
    args = _cache_args(tmp_path / "cache", image_keys=("corner.image", "corner4.image"))

    with pytest.raises(ValueError, match="exactly one image key/view"):
        precompute_generated_wan_latents(args, generator=_FakeGenerator())


@pytest.mark.parametrize("bad_stop_after_steps", [0, -1, 3, True, 1.5])
def test_generated_wan_latent_cache_rejects_bad_stop_after_steps(tmp_path, bad_stop_after_steps) -> None:
    args = _cache_args(tmp_path / "cache", num_inference_steps=2, stop_after_steps=bad_stop_after_steps)

    with pytest.raises(ValueError, match="stop_after_steps"):
        precompute_generated_wan_latents(args, generator=_FakeGenerator())


@pytest.mark.parametrize(
    ("latents", "match"),
    [
        (torch.zeros((48, 2, 1, 1)), "rank 5"),
        (torch.zeros((2, 48, 2, 1, 1)), "batch dimension 1"),
        (torch.zeros((1, 48, 3, 1, 1)), "does not match GeneratedWanLatentDataset schema"),
    ],
)
def test_generated_wan_latent_cache_rejects_bad_latent_rank_batch_and_shape(tmp_path, latents, match) -> None:
    args = _cache_args(tmp_path / "cache", synthetic_samples=1)

    with pytest.raises(ValueError, match=match):
        precompute_generated_wan_latents(args, generator=_FakeGenerator(latents=latents))


def test_generated_wan_latent_cache_rejects_existing_config_mismatch_before_generating(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    args = _cache_args(cache_dir)
    precompute_generated_wan_latents(args, generator=_FakeGenerator())
    generator = _FakeGenerator()

    with pytest.raises(ValueError, match="metadata mismatch.*generator"):
        precompute_generated_wan_latents(
            _cache_args(cache_dir, lora_path="/fake/epoch-1.safetensors"),
            generator=generator,
        )

    assert generator.calls == []


def test_generated_wan_latent_cache_rejects_partial_full_config_mismatch_before_generating(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    full_args = _cache_args(cache_dir, num_inference_steps=2, stop_after_steps=None)
    precompute_generated_wan_latents(full_args, generator=_FakeGenerator())
    generator = _FakeGenerator(num_inference_steps=2)

    with pytest.raises(ValueError, match="metadata mismatch.*generator"):
        precompute_generated_wan_latents(
            _cache_args(cache_dir, num_inference_steps=2, stop_after_steps=1),
            generator=generator,
        )

    assert generator.calls == []


@pytest.mark.parametrize(
    ("metadata_updates", "mismatched_key"),
    [
        ({"denoise_mode": "partial"}, "denoise_mode"),
        ({"stop_after_steps": 1}, "stop_after_steps"),
    ],
)
def test_generated_wan_latent_cache_rejects_result_config_core_metadata_mismatch_before_writing_row(
    tmp_path,
    metadata_updates,
    mismatched_key,
) -> None:
    cache_dir = tmp_path / "cache"

    with pytest.raises(ValueError, match=f"result metadata mismatch.*{mismatched_key}"):
        precompute_generated_wan_latents(
            _cache_args(cache_dir, synthetic_samples=1, num_inference_steps=2, stop_after_steps=None),
            generator=_MetadataOverrideGenerator(metadata_updates),
        )

    assert _read_manifest(cache_dir) == []
    assert list((cache_dir / "latents").glob("*.pt")) == []
