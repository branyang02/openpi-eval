from __future__ import annotations

import sys

import pytest
import torch
from PIL import Image

from world_model.diffsynth_wan import (
    DiffSynthWanLoraConfig,
    DiffSynthWanLoraFutureGenerator,
    generate_diffsynth_wan_predecode_latents,
    select_lora_future_frame_indices,
    validate_local_wan_checkpoint,
)


def write_fake_wan_checkpoint(checkpoint_dir) -> None:
    for relative in [
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model-00001-of-00003.safetensors",
        "diffusion_pytorch_model-00002-of-00003.safetensors",
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/tokenizer.json",
        "google/umt5-xxl/tokenizer_config.json",
    ]:
        path = checkpoint_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub\n")


def write_fake_diffsynth_repo(repo_dir) -> None:
    (repo_dir / "diffsynth").mkdir(parents=True)


class FakePipe:
    def __call__(self, **kwargs):
        height = kwargs["height"]
        width = kwargs["width"]
        num_frames = kwargs["num_frames"]
        return [Image.new("RGB", (width, height), color=(index, index, index)) for index in range(num_frames)]


class RecordingPipe:
    """Like FakePipe, but records every kwargs dict it is called with for inspection."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        height = kwargs["height"]
        width = kwargs["width"]
        num_frames = kwargs["num_frames"]
        return [Image.new("RGB", (width, height), color=(index, index, index)) for index in range(num_frames)]


class _FakeWanVaeModel:
    z_dim = 4


class _DecodeForbiddenVae:
    def __init__(self) -> None:
        self.model = _FakeWanVaeModel()
        self.upsampling_factor = 8
        self.decode_calls = 0
        self.decode_framewise_calls = 0

    def decode(self, *args, **kwargs):
        self.decode_calls += 1
        raise AssertionError("latent helper must not call VAE decode")

    def decode_framewise(self, *args, **kwargs):
        self.decode_framewise_calls += 1
        raise AssertionError("latent helper must not call framewise VAE decode")


class _FakeWanScheduler:
    def __init__(self) -> None:
        self.timesteps = []
        self.set_calls: list[dict] = []
        self.step_calls: list[dict] = []

    def set_timesteps(self, num_inference_steps, *, denoising_strength, shift) -> None:
        self.set_calls.append(
            {
                "num_inference_steps": num_inference_steps,
                "denoising_strength": denoising_strength,
                "shift": shift,
            }
        )
        self.timesteps = torch.linspace(1000, 0, num_inference_steps)

    def step(self, noise_pred, timestep, latents):
        self.step_calls.append({"timestep": timestep, "shape": tuple(latents.shape)})
        return latents + noise_pred


class _FakeWanPrepUnit:
    def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        if pipe.latent_override is None:
            num_frames = inputs_shared["num_frames"]
            height = inputs_shared["height"]
            width = inputs_shared["width"]
            latent_shape = (
                1,
                pipe.vae.model.z_dim,
                (num_frames - 1) // 4 + 1,
                height // pipe.vae.upsampling_factor,
                width // pipe.vae.upsampling_factor,
            )
            inputs_shared["latents"] = torch.zeros(latent_shape)
        else:
            inputs_shared["latents"] = pipe.latent_override.clone()
        return inputs_shared, inputs_posi, inputs_nega


class _FakeWanPostUnit:
    def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        pipe.post_unit_calls += 1
        inputs_shared["latents"] = inputs_shared["latents"] + pipe.post_add
        return inputs_shared, inputs_posi, inputs_nega


def _fake_wan_unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega):
    return unit.process(pipe, inputs_shared, inputs_posi, inputs_nega)


class _FakePredecodeWanPipe:
    def __init__(self, *, latent_override: torch.Tensor | None = None, post_add: float = 10.0) -> None:
        self.scheduler = _FakeWanScheduler()
        self.units = [_FakeWanPrepUnit()]
        self.post_units = [_FakeWanPostUnit()]
        self.unit_runner = _fake_wan_unit_runner
        self.in_iteration_models = ("dit",)
        self.in_iteration_models_2 = ("dit2",)
        self.dit = object()
        self.dit2 = None
        self.vace = None
        self.vace2 = None
        self.vae = _DecodeForbiddenVae()
        self.torch_dtype = torch.float32
        self.device = "cpu"
        self.latent_override = latent_override
        self.post_add = post_add
        self.post_unit_calls = 0
        self.model_fn_calls = []
        self.load_model_calls: list[tuple[str, ...]] = []
        self.vae_output_to_video_calls = 0

    def load_models_to_device(self, model_names) -> None:
        self.load_model_calls.append(tuple(model_names))

    def model_fn(self, **kwargs):
        self.model_fn_calls.append(kwargs)
        return torch.ones_like(kwargs["latents"])

    def vae_output_to_video(self, *args, **kwargs):
        self.vae_output_to_video_calls += 1
        raise AssertionError("latent helper must not convert decoded VAE output to video")


def _solid_image(value: int, size: int = 16) -> Image.Image:
    return Image.new("RGB", (size, size), color=(value, value, value))


def _image_from_value(value: int, size: int = 16) -> torch.Tensor:
    return torch.full((1, 3, size, size), value / 255.0)


def _conditioning_video_reader(conditioning_value: int):
    # Wan emits the conditioning image as frame 0; later frames diverge into the future.
    return lambda _path: [_solid_image(conditioning_value)] + [_solid_image((index + 1) * 30) for index in range(4)]


def test_validate_local_wan_checkpoint_groups_dit_shards(tmp_path) -> None:
    checkpoint_dir = tmp_path / "Wan2.2-TI2V-5B"
    write_fake_wan_checkpoint(checkpoint_dir)

    summary = validate_local_wan_checkpoint(checkpoint_dir)

    assert summary["diffusion_shards"] == [
        "diffusion_pytorch_model-00001-of-00003.safetensors",
        "diffusion_pytorch_model-00002-of-00003.safetensors",
    ]
    assert summary["model_paths"] == [
        str(checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
        [
            str(checkpoint_dir / "diffusion_pytorch_model-00001-of-00003.safetensors"),
            str(checkpoint_dir / "diffusion_pytorch_model-00002-of-00003.safetensors"),
        ],
        str(checkpoint_dir / "Wan2.2_VAE.pth"),
    ]


def test_diffsynth_wan_lora_generator_returns_cached_future_shape(tmp_path) -> None:
    generator = DiffSynthWanLoraFutureGenerator(
        DiffSynthWanLoraConfig(
            diffsynth_repo_dir=str(tmp_path / "DiffSynth-Studio"),
            checkpoint_dir=str(tmp_path / "Wan2.2-TI2V-5B"),
            lora_path=str(tmp_path / "epoch-0.safetensors"),
            height=16,
            width=16,
            num_frames=5,
            num_inference_steps=1,
        ),
        pipe_loader=FakePipe,
        video_saver=lambda video, path, fps, quality: open(path, "wb").write(b"fake mp4"),
        video_reader=_conditioning_video_reader(96),
    )

    result = generator.generate_future_stack(
        _image_from_value(96),
        task_text="pick up the nut",
        output_dir=tmp_path / "generated",
        image_size=16,
        num_future_frames=4,
        seed=3,
    )

    assert result.seed == 3
    assert result.future_images.shape == (4, 1, 3, 16, 16)
    assert result.input_image_path.exists()
    assert result.video_path.exists()
    assert result.selected_frame_indices == (1, 2, 3, 4)
    assert result.total_video_frames == 5


def test_diffsynth_wan_lora_generator_rejects_zero_frame_video(tmp_path) -> None:
    generator = DiffSynthWanLoraFutureGenerator(
        DiffSynthWanLoraConfig(
            diffsynth_repo_dir=str(tmp_path / "DiffSynth-Studio"),
            checkpoint_dir=str(tmp_path / "Wan2.2-TI2V-5B"),
            lora_path=str(tmp_path / "epoch-0.safetensors"),
            height=16,
            width=16,
            num_frames=5,
            num_inference_steps=1,
        ),
        pipe_loader=FakePipe,
        video_saver=lambda video, path, fps, quality: open(path, "wb").write(b"fake mp4"),
        video_reader=lambda _path: [],
    )

    with pytest.raises(ValueError, match="did not produce any video frames"):
        generator.generate_future_stack(
            torch.rand(1, 3, 16, 16),
            task_text="pick up the nut",
            output_dir=tmp_path / "generated",
            image_size=16,
            num_future_frames=4,
        )


def test_diffsynth_load_pipe_rejects_missing_repo(tmp_path) -> None:
    generator = DiffSynthWanLoraFutureGenerator(
        DiffSynthWanLoraConfig(
            diffsynth_repo_dir=str(tmp_path / "missing-diffsynth"),
            checkpoint_dir=str(tmp_path / "Wan2.2-TI2V-5B"),
            lora_path=str(tmp_path / "epoch-0.safetensors"),
        )
    )

    with pytest.raises(FileNotFoundError, match="DiffSynth-Studio repo"):
        generator.load_pipe()


def test_diffsynth_load_pipe_rejects_missing_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "path", sys.path.copy())
    repo_dir = tmp_path / "DiffSynth-Studio"
    write_fake_diffsynth_repo(repo_dir)

    generator = DiffSynthWanLoraFutureGenerator(
        DiffSynthWanLoraConfig(
            diffsynth_repo_dir=str(repo_dir),
            checkpoint_dir=str(tmp_path / "missing-wan-checkpoint"),
            lora_path=str(tmp_path / "epoch-0.safetensors"),
        )
    )

    with pytest.raises(FileNotFoundError, match="Wan checkpoint directory"):
        generator.load_pipe()


def test_diffsynth_load_pipe_rejects_missing_lora(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "path", sys.path.copy())
    repo_dir = tmp_path / "DiffSynth-Studio"
    checkpoint_dir = tmp_path / "Wan2.2-TI2V-5B"
    write_fake_diffsynth_repo(repo_dir)
    write_fake_wan_checkpoint(checkpoint_dir)

    generator = DiffSynthWanLoraFutureGenerator(
        DiffSynthWanLoraConfig(
            diffsynth_repo_dir=str(repo_dir),
            checkpoint_dir=str(checkpoint_dir),
            lora_path=str(tmp_path / "missing-lora.safetensors"),
        )
    )

    with pytest.raises(FileNotFoundError, match="LoRA checkpoint"):
        generator.load_pipe()


def test_lora_future_frame_strategy_first_aligns_to_early_frames() -> None:
    assert select_lora_future_frame_indices(17, 4, strategy="first") == [1, 2, 3, 4]
    assert select_lora_future_frame_indices(17, 4, frame_delta=4, strategy="source_offsets") == [4, 8, 12, 16]
    with pytest.raises(ValueError, match="Need at least 17 frames"):
        select_lora_future_frame_indices(16, 4, frame_delta=4, strategy="source_offsets")
    with pytest.raises(ValueError, match="future_frame_strategy must be one of"):
        select_lora_future_frame_indices(17, 4, strategy="linspace")


def _lora_config(tmp_path, **overrides) -> DiffSynthWanLoraConfig:
    base = dict(
        diffsynth_repo_dir=str(tmp_path / "DiffSynth-Studio"),
        checkpoint_dir=str(tmp_path / "Wan2.2-TI2V-5B"),
        lora_path=str(tmp_path / "epoch-0.safetensors"),
        height=16,
        width=16,
        num_frames=5,
        num_inference_steps=1,
    )
    base.update(overrides)
    return DiffSynthWanLoraConfig(**base)


def test_diffsynth_wan_lora_latent_path_returns_predecode_latents_without_video_save(tmp_path) -> None:
    pipe = _FakePredecodeWanPipe()
    save_calls = []
    generator = DiffSynthWanLoraFutureGenerator(
        _lora_config(tmp_path, width=24, height=16, num_frames=5, num_inference_steps=2, base_seed=17),
        pipe_loader=lambda: pipe,
        video_saver=lambda *args: save_calls.append(args),
    )

    result = generator.generate_view_latents(
        _image_from_value(96, size=16)[0],
        task_text="pick up the nut",
        seed=13,
    )

    assert result.latents.shape == (1, 4, 2, 2, 3)
    assert torch.equal(result.latents, torch.full((1, 4, 2, 2, 3), 12.0))
    assert "pick up the nut" in result.prompt
    assert result.seed == 13
    assert result.num_inference_steps == 2
    assert pipe.post_unit_calls == 1
    assert len(pipe.scheduler.step_calls) == 2
    assert pipe.vae.decode_calls == 0
    assert pipe.vae.decode_framewise_calls == 0
    assert pipe.vae_output_to_video_calls == 0
    assert ("vae",) not in pipe.load_model_calls
    assert save_calls == []


def test_diffsynth_wan_lora_latent_path_forwards_partial_stop(tmp_path) -> None:
    pipe = _FakePredecodeWanPipe()
    generator = DiffSynthWanLoraFutureGenerator(
        _lora_config(tmp_path, width=24, height=16, num_frames=5, num_inference_steps=3),
        pipe_loader=lambda: pipe,
    )

    result = generator.generate_view_latents(
        _image_from_value(96, size=16)[0],
        task_text="pick up the nut",
        seed=13,
        stop_after_steps=1,
    )

    assert result.latents.shape == (1, 4, 2, 2, 3)
    assert torch.equal(result.latents, torch.full((1, 4, 2, 2, 3), 11.0))
    assert len(pipe.scheduler.step_calls) == 1
    assert result.metadata["num_inference_steps"] == 3
    assert result.metadata["denoise_steps_run"] == 1
    assert result.metadata["stop_after_steps"] == 1
    assert result.metadata["denoise_fraction"] == pytest.approx(1 / 3)
    assert result.metadata["denoise_mode"] == "partial"


def test_diffsynth_wan_latent_path_partial_stop_runs_fewer_denoise_steps() -> None:
    pipe = _FakePredecodeWanPipe()

    result = generate_diffsynth_wan_predecode_latents(
        pipe,
        prompt="prompt",
        height=16,
        width=16,
        num_frames=5,
        num_inference_steps=3,
        stop_after_steps=1,
    )

    assert torch.equal(result.latents, torch.full((1, 4, 2, 2, 2), 11.0))
    assert len(pipe.scheduler.step_calls) == 1
    assert pipe.post_unit_calls == 1
    assert pipe.vae.decode_calls == 0
    assert pipe.vae.decode_framewise_calls == 0
    assert pipe.vae_output_to_video_calls == 0
    assert result.metadata["num_inference_steps"] == 3
    assert result.metadata["denoise_steps_run"] == 1
    assert result.metadata["completed_denoise_steps"] == 1
    assert result.metadata["stop_after_steps"] == 1
    assert result.metadata["denoise_fraction"] == pytest.approx(1 / 3)
    assert result.metadata["denoise_mode"] == "partial"


@pytest.mark.parametrize("bad_stop_after_steps", [0, -1, 4, True, 1.5])
def test_diffsynth_wan_latent_path_rejects_bad_stop_after_steps(bad_stop_after_steps) -> None:
    pipe = _FakePredecodeWanPipe()

    with pytest.raises(ValueError, match="stop_after_steps"):
        generate_diffsynth_wan_predecode_latents(
            pipe,
            prompt="prompt",
            height=16,
            width=16,
            num_frames=5,
            num_inference_steps=3,
            stop_after_steps=bad_stop_after_steps,
        )


def test_diffsynth_wan_latent_metadata_records_cache_validation_fields(tmp_path) -> None:
    pipe = _FakePredecodeWanPipe()
    generator = DiffSynthWanLoraFutureGenerator(
        _lora_config(tmp_path, width=24, height=16, num_frames=5, num_inference_steps=2, tiled=False),
        pipe_loader=lambda: pipe,
    )

    result = generator.generate_view_latents(
        _image_from_value(96, size=16)[0],
        task_text="open the drawer",
        seed=23,
        stop_after_steps=2,
    )

    assert result.metadata["source"] == "diffsynth_wan_lora"
    assert result.metadata["latent_stage"] == "post_denoising_post_units_pre_vae_decode"
    assert result.metadata["latent_shape"] == (1, 4, 2, 2, 3)
    assert result.metadata["height"] == 16
    assert result.metadata["width"] == 24
    assert result.metadata["num_frames"] == 5
    assert result.metadata["num_inference_steps"] == 2
    assert result.metadata["denoise_steps_run"] == 2
    assert result.metadata["completed_denoise_steps"] == 2
    assert result.metadata["stop_after_steps"] == 2
    assert result.metadata["denoise_fraction"] == pytest.approx(1.0)
    assert result.metadata["denoise_mode"] == "full"
    assert result.metadata["vae_z_dim"] == 4
    assert result.metadata["vae_upsampling_factor"] == 8
    assert result.metadata["tiled"] is False
    assert result.metadata["checkpoint_dir"] == str(tmp_path / "Wan2.2-TI2V-5B")
    assert result.metadata["lora_path"] == str(tmp_path / "epoch-0.safetensors")


def test_diffsynth_wan_latent_shape_validation_rejects_bad_latents() -> None:
    rank_pipe = _FakePredecodeWanPipe(latent_override=torch.zeros(1, 4, 2, 2))
    with pytest.raises(ValueError, match="rank 5"):
        generate_diffsynth_wan_predecode_latents(
            rank_pipe,
            prompt="prompt",
            height=16,
            width=16,
            num_frames=5,
            num_inference_steps=1,
        )

    channel_pipe = _FakePredecodeWanPipe(latent_override=torch.zeros(1, 5, 2, 2, 2))
    with pytest.raises(ValueError, match="channel dimension"):
        generate_diffsynth_wan_predecode_latents(
            channel_pipe,
            prompt="prompt",
            height=16,
            width=16,
            num_frames=5,
            num_inference_steps=1,
        )


def test_diffsynth_wan_lora_generator_rejects_shifted_conditioning_frame(tmp_path) -> None:
    generator = DiffSynthWanLoraFutureGenerator(
        _lora_config(tmp_path),
        pipe_loader=FakePipe,
        video_saver=lambda video, path, fps, quality: open(path, "wb").write(b"fake mp4"),
        # Frame 0 is gray-220, but the conditioning image is gray-96: a shifted/missing frame 0.
        video_reader=lambda _path: [_solid_image(220)] + [_solid_image((index + 1) * 30) for index in range(4)],
    )

    with pytest.raises(ValueError, match="conditioning image"):
        generator.generate_future_stack(
            _image_from_value(96),
            task_text="pick up the nut",
            output_dir=tmp_path / "generated",
            image_size=16,
            num_future_frames=4,
        )


def test_diffsynth_wan_lora_generator_can_disable_conditioning_check(tmp_path) -> None:
    generator = DiffSynthWanLoraFutureGenerator(
        _lora_config(tmp_path, verify_conditioning_frame=False),
        pipe_loader=FakePipe,
        video_saver=lambda video, path, fps, quality: open(path, "wb").write(b"fake mp4"),
        video_reader=lambda _path: [_solid_image(220)] + [_solid_image((index + 1) * 30) for index in range(4)],
    )

    result = generator.generate_future_stack(
        _image_from_value(96),
        task_text="pick up the nut",
        output_dir=tmp_path / "generated",
        image_size=16,
        num_future_frames=4,
    )

    assert result.future_images.shape == (4, 1, 3, 16, 16)
    assert result.selected_frame_indices == (1, 2, 3, 4)


def test_diffsynth_wan_lora_generator_forwards_pipe_kwargs(tmp_path) -> None:
    recording = RecordingPipe()
    generator = DiffSynthWanLoraFutureGenerator(
        # width != height pins the (width, height) argument order; non-default num_inference_steps,
        # tiled, and base_seed catch any hard-coded/dropped value.
        _lora_config(tmp_path, width=24, height=16, num_frames=5, num_inference_steps=3, tiled=False, base_seed=11),
        pipe_loader=lambda: recording,
        video_saver=lambda video, path, fps, quality: open(path, "wb").write(b"fake mp4"),
        video_reader=_conditioning_video_reader(96),
    )

    generator.generate_future_stack(
        _image_from_value(96),
        task_text="pick up the nut",
        output_dir=tmp_path / "generated",
        image_size=16,
        num_future_frames=4,
        seed=5,
    )

    assert len(recording.calls) == 1
    call = recording.calls[0]
    # The per-call seed, the configured sampling steps, and the tiled flag must reach the pipe.
    assert call["seed"] == 5
    assert call["num_inference_steps"] == 3
    assert call["tiled"] is False
    # The conditioning image is forwarded as input_image, resized to (width, height).
    assert isinstance(call["input_image"], Image.Image)
    assert call["input_image"].size == (24, 16)
    assert call["width"] == 24
    assert call["height"] == 16
    assert call["num_frames"] == 5


def test_diffsynth_wan_lora_generator_uses_base_seed_when_seed_omitted(tmp_path) -> None:
    recording = RecordingPipe()
    generator = DiffSynthWanLoraFutureGenerator(
        _lora_config(tmp_path, base_seed=11),
        pipe_loader=lambda: recording,
        video_saver=lambda video, path, fps, quality: open(path, "wb").write(b"fake mp4"),
        video_reader=_conditioning_video_reader(96),
    )

    generator.generate_future_stack(
        _image_from_value(96),
        task_text="pick up the nut",
        output_dir=tmp_path / "generated",
        image_size=16,
        num_future_frames=4,
    )

    # With no per-call seed, the pipe is driven by the config's base_seed.
    assert recording.calls[0]["seed"] == 11
