from __future__ import annotations

import dataclasses
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from world_model.config import FutureFrameStrategy, Wan22Config, validate_future_frame_strategy
from world_model.data import image_to_chw_float
from world_model.wan22 import (
    DEFAULT_CONDITIONING_FRAME_MAX_MAE,
    Wan22Result,
    read_video_frames,
    select_future_frame_indices,
    tensor_image_to_pil,
    verify_conditioning_frame,
)

VideoReader = Callable[[Path], Sequence[Any]]
VideoSaver = Callable[[Sequence[Image.Image], str, int, int], None]
PipeLoader = Callable[[], Any]

WAN_LATENT_STAGE = "post_denoising_post_units_pre_vae_decode"

REQUIRED_WAN_CHECKPOINT_FILES = (
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "diffusion_pytorch_model.safetensors.index.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)


@dataclasses.dataclass(frozen=True)
class DiffSynthWanLoraConfig:
    diffsynth_repo_dir: str
    checkpoint_dir: str
    lora_path: str
    height: int = 64
    width: int = 64
    num_frames: int = 17
    num_inference_steps: int = 2
    lora_alpha: float = 1.0
    device: str = "cuda"
    tiled: bool = True
    fps: int = 15
    base_seed: int = 7
    prompt_template: str = Wan22Config.prompt_template
    frame_delta: int = 1
    future_frame_strategy: FutureFrameStrategy = "first"
    verify_conditioning_frame: bool = True
    conditioning_frame_max_mae: float = DEFAULT_CONDITIONING_FRAME_MAX_MAE

    def __post_init__(self) -> None:
        if self.frame_delta <= 0:
            raise ValueError(f"frame_delta must be positive, got {self.frame_delta}.")
        validate_future_frame_strategy(self.future_frame_strategy)


@dataclasses.dataclass(frozen=True)
class DiffSynthWanLatentResult:
    latents: torch.Tensor
    prompt: str
    seed: int | None
    num_inference_steps: int
    metadata: dict[str, object]


def _identity_progress(iterable: Iterable[Any]) -> Iterable[Any]:
    return iterable


_DIFFSYNTH_WAN_CALL_DEFAULTS: dict[str, Any] = {
    "prompt": "",
    "negative_prompt": "",
    "input_image": None,
    "end_image": None,
    "input_video": None,
    "denoising_strength": 1.0,
    "input_audio": None,
    "audio_embeds": None,
    "audio_sample_rate": 16000,
    "s2v_pose_video": None,
    "s2v_pose_latents": None,
    "motion_video": None,
    "control_video": None,
    "reference_image": None,
    "camera_control_direction": None,
    "camera_control_speed": 1 / 54,
    "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
    "vace_video": None,
    "vace_video_mask": None,
    "vace_reference_image": None,
    "vace_scale": 1.0,
    "animate_pose_video": None,
    "animate_face_video": None,
    "animate_inpaint_video": None,
    "animate_mask_video": None,
    "vap_video": None,
    "vap_prompt": " ",
    "negative_vap_prompt": " ",
    "seed": None,
    "rand_device": "cpu",
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "cfg_scale": 5.0,
    "cfg_merge": False,
    "switch_DiT_boundary": 0.875,
    "num_inference_steps": 50,
    "sigma_shift": 5.0,
    "motion_bucket_id": None,
    "longcat_video": None,
    "tiled": True,
    "tile_size": (30, 52),
    "tile_stride": (15, 26),
    "sliding_window_size": None,
    "sliding_window_stride": None,
    "tea_cache_l1_thresh": None,
    "tea_cache_model_id": "",
    "wantodance_music_path": None,
    "wantodance_reference_image": None,
    "wantodance_fps": 30,
    "wantodance_keyframes": None,
    "wantodance_keyframes_mask": None,
    "framewise_decoding": False,
    "output_type": "quantized",
}


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _resolve_denoise_steps_run(stop_after_steps: int | None, num_inference_steps: int) -> int:
    if stop_after_steps is None:
        return num_inference_steps
    if isinstance(stop_after_steps, bool) or not isinstance(stop_after_steps, int):
        raise ValueError(f"stop_after_steps must be None or an integer, got {stop_after_steps!r}.")
    if stop_after_steps <= 0:
        raise ValueError(f"stop_after_steps must be positive when provided, got {stop_after_steps}.")
    if stop_after_steps > num_inference_steps:
        raise ValueError(
            "stop_after_steps must be less than or equal to num_inference_steps "
            f"({num_inference_steps}), got {stop_after_steps}."
        )
    return stop_after_steps


def _tensor_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def validate_diffsynth_wan_latent_shape(
    latents: Any,
    *,
    pipe: Any,
    height: int,
    width: int,
    num_frames: int,
) -> None:
    if not isinstance(latents, torch.Tensor):
        raise ValueError(f"DiffSynth Wan latent helper expected a torch.Tensor, got {type(latents).__name__}.")
    if latents.ndim != 5:
        raise ValueError(
            "Expected generated Wan latents with rank 5 shaped (B, C, T, H, W), "
            f"got rank {latents.ndim} and shape {_tensor_shape(latents)}."
        )
    if any(dim <= 0 for dim in latents.shape):
        raise ValueError(f"Generated Wan latents must have only positive dimensions, got {_tensor_shape(latents)}.")
    if latents.shape[0] != 1:
        raise ValueError(f"Expected generated Wan latent batch dimension 1, got shape {_tensor_shape(latents)}.")

    expected_frames = (num_frames - 1) // 4 + 1
    if latents.shape[2] != expected_frames:
        raise ValueError(
            "Generated Wan latent temporal dimension does not match num_frames: "
            f"expected {expected_frames} for num_frames={num_frames}, got shape {_tensor_shape(latents)}."
        )

    vae = getattr(pipe, "vae", None)
    z_dim = getattr(getattr(vae, "model", None), "z_dim", None)
    if z_dim is not None and latents.shape[1] != z_dim:
        raise ValueError(
            "Generated Wan latent channel dimension does not match pipe.vae.model.z_dim: "
            f"expected {z_dim}, got shape {_tensor_shape(latents)}."
        )

    upsampling_factor = getattr(vae, "upsampling_factor", None)
    if upsampling_factor is not None:
        expected_height = height // upsampling_factor
        expected_width = width // upsampling_factor
        if latents.shape[3] != expected_height or latents.shape[4] != expected_width:
            raise ValueError(
                "Generated Wan latent spatial dimensions do not match the VAE upsampling factor: "
                f"expected {(expected_height, expected_width)} for image {(height, width)} and "
                f"upsampling_factor={upsampling_factor}, got shape {_tensor_shape(latents)}."
            )


def _diffsynth_wan_latent_metadata(
    *,
    pipe: Any,
    call_args: dict[str, Any],
    inputs_shared: dict[str, Any],
    latents: torch.Tensor,
    stop_after_steps: int | None,
    denoise_steps_run: int,
) -> dict[str, object]:
    vae = getattr(pipe, "vae", None)
    num_inference_steps = int(call_args["num_inference_steps"])
    denoise_fraction = float(denoise_steps_run) / float(num_inference_steps)
    metadata = {
        "format_version": 1,
        "source": "diffsynth_wan_lora",
        "latent_stage": WAN_LATENT_STAGE,
        "pipeline_class": type(pipe).__name__,
        "height": int(inputs_shared["height"]),
        "width": int(inputs_shared["width"]),
        "num_frames": int(inputs_shared["num_frames"]),
        "latent_shape": _tensor_shape(latents),
        "latent_dtype": str(latents.dtype),
        "latent_device": str(latents.device),
        "num_inference_steps": num_inference_steps,
        "denoise_steps_run": int(denoise_steps_run),
        "completed_denoise_steps": int(denoise_steps_run),
        "stop_after_steps": None if stop_after_steps is None else int(stop_after_steps),
        "denoise_fraction": denoise_fraction,
        "denoise_mode": "partial" if denoise_steps_run < num_inference_steps else "full",
        "denoising_strength": float(call_args["denoising_strength"]),
        "sigma_shift": float(call_args["sigma_shift"]),
        "cfg_scale": float(call_args["cfg_scale"]),
        "cfg_merge": bool(call_args["cfg_merge"]),
        "tiled": bool(call_args["tiled"]),
        "tile_size": tuple(call_args["tile_size"]),
        "tile_stride": tuple(call_args["tile_stride"]),
        "vae_z_dim": getattr(getattr(vae, "model", None), "z_dim", None),
        "vae_upsampling_factor": getattr(vae, "upsampling_factor", None),
    }
    diffsynth_repo_dir = getattr(pipe, "_diffsynth_repo_dir", None)
    if diffsynth_repo_dir is not None:
        metadata["diffsynth_repo_dir"] = str(diffsynth_repo_dir)
    return metadata


def generate_diffsynth_wan_predecode_latents(
    pipe: Any,
    *,
    stop_after_steps: int | None = None,
    progress_bar_cmd: Callable[[Iterable[Any]], Iterable[Any]] | None = None,
    **kwargs: Any,
) -> DiffSynthWanLatentResult:
    unknown = sorted(set(kwargs) - set(_DIFFSYNTH_WAN_CALL_DEFAULTS))
    if unknown:
        raise TypeError(f"Unsupported DiffSynth Wan latent generation argument(s): {unknown}")

    call_args = dict(_DIFFSYNTH_WAN_CALL_DEFAULTS)
    call_args.update(kwargs)
    num_inference_steps = _positive_int("num_inference_steps", call_args["num_inference_steps"])
    _positive_int("num_frames", call_args["num_frames"])
    _positive_int("height", call_args["height"])
    _positive_int("width", call_args["width"])
    denoise_steps_run = _resolve_denoise_steps_run(stop_after_steps, num_inference_steps)
    progress = _identity_progress if progress_bar_cmd is None else progress_bar_cmd

    with torch.no_grad():
        pipe.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=call_args["denoising_strength"],
            shift=call_args["sigma_shift"],
        )

        inputs_posi = {
            "prompt": call_args["prompt"],
            "vap_prompt": call_args["vap_prompt"],
            "tea_cache_l1_thresh": call_args["tea_cache_l1_thresh"],
            "tea_cache_model_id": call_args["tea_cache_model_id"],
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": call_args["negative_prompt"],
            "negative_vap_prompt": call_args["negative_vap_prompt"],
            "tea_cache_l1_thresh": call_args["tea_cache_l1_thresh"],
            "tea_cache_model_id": call_args["tea_cache_model_id"],
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": call_args["input_image"],
            "end_image": call_args["end_image"],
            "input_video": call_args["input_video"],
            "denoising_strength": call_args["denoising_strength"],
            "control_video": call_args["control_video"],
            "reference_image": call_args["reference_image"],
            "camera_control_direction": call_args["camera_control_direction"],
            "camera_control_speed": call_args["camera_control_speed"],
            "camera_control_origin": call_args["camera_control_origin"],
            "vace_video": call_args["vace_video"],
            "vace_video_mask": call_args["vace_video_mask"],
            "vace_reference_image": call_args["vace_reference_image"],
            "vace_scale": call_args["vace_scale"],
            "seed": call_args["seed"],
            "rand_device": call_args["rand_device"],
            "height": call_args["height"],
            "width": call_args["width"],
            "num_frames": call_args["num_frames"],
            "cfg_scale": call_args["cfg_scale"],
            "cfg_merge": call_args["cfg_merge"],
            "sigma_shift": call_args["sigma_shift"],
            "motion_bucket_id": call_args["motion_bucket_id"],
            "longcat_video": call_args["longcat_video"],
            "tiled": call_args["tiled"],
            "tile_size": call_args["tile_size"],
            "tile_stride": call_args["tile_stride"],
            "sliding_window_size": call_args["sliding_window_size"],
            "sliding_window_stride": call_args["sliding_window_stride"],
            "input_audio": call_args["input_audio"],
            "audio_sample_rate": call_args["audio_sample_rate"],
            "s2v_pose_video": call_args["s2v_pose_video"],
            "audio_embeds": call_args["audio_embeds"],
            "s2v_pose_latents": call_args["s2v_pose_latents"],
            "motion_video": call_args["motion_video"],
            "animate_pose_video": call_args["animate_pose_video"],
            "animate_face_video": call_args["animate_face_video"],
            "animate_inpaint_video": call_args["animate_inpaint_video"],
            "animate_mask_video": call_args["animate_mask_video"],
            "vap_video": call_args["vap_video"],
            "wantodance_music_path": call_args["wantodance_music_path"],
            "wantodance_reference_image": call_args["wantodance_reference_image"],
            "wantodance_fps": call_args["wantodance_fps"],
            "wantodance_keyframes": call_args["wantodance_keyframes"],
            "wantodance_keyframes_mask": call_args["wantodance_keyframes_mask"],
            "framewise_decoding": call_args["framewise_decoding"],
        }
        for unit in pipe.units:
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                unit,
                pipe,
                inputs_shared,
                inputs_posi,
                inputs_nega,
            )

        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        denoise_timesteps = pipe.scheduler.timesteps[:denoise_steps_run]
        for progress_id, timestep in enumerate(progress(denoise_timesteps)):
            timestep_value = timestep.item() if hasattr(timestep, "item") else float(timestep)
            if (
                timestep_value < call_args["switch_DiT_boundary"] * 1000
                and getattr(pipe, "dit2", None) is not None
                and models.get("dit") is not pipe.dit2
            ):
                pipe.load_models_to_device(pipe.in_iteration_models_2)
                models["dit"] = pipe.dit2
                models["vace"] = pipe.vace2

            if hasattr(timestep, "unsqueeze"):
                timestep_input = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            else:
                timestep_input = torch.tensor([timestep], dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred_posi = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep_input)
            if call_args["cfg_scale"] != 1.0:
                if call_args["cfg_merge"]:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep_input)
                noise_pred = noise_pred_nega + call_args["cfg_scale"] * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            inputs_shared["latents"] = pipe.scheduler.step(
                noise_pred,
                timestep,
                inputs_shared["latents"],
            )
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        vace_reference_image = call_args["vace_reference_image"]
        animate_pose_video = call_args["animate_pose_video"]
        animate_face_video = call_args["animate_face_video"]
        if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
            f = len(vace_reference_image) if vace_reference_image is not None and isinstance(vace_reference_image, list) else 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]

        for unit in pipe.post_units:
            inputs_shared, _, _ = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

        latents = inputs_shared.get("latents")
        validate_diffsynth_wan_latent_shape(
            latents,
            pipe=pipe,
            height=int(inputs_shared["height"]),
            width=int(inputs_shared["width"]),
            num_frames=int(inputs_shared["num_frames"]),
        )
        metadata = _diffsynth_wan_latent_metadata(
            pipe=pipe,
            call_args=call_args,
            inputs_shared=inputs_shared,
            latents=latents,
            stop_after_steps=stop_after_steps,
            denoise_steps_run=denoise_steps_run,
        )
        pipe.load_models_to_device([])

    return DiffSynthWanLatentResult(
        latents=latents,
        prompt=call_args["prompt"],
        seed=call_args["seed"],
        num_inference_steps=num_inference_steps,
        metadata=metadata,
    )


def validate_local_wan_checkpoint(checkpoint_dir: str | Path) -> dict[str, object]:
    path = Path(checkpoint_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Wan checkpoint directory not found: {path}")
    missing = [relative for relative in REQUIRED_WAN_CHECKPOINT_FILES if not (path / relative).exists()]
    shard_paths = sorted(path.glob("diffusion_pytorch_model-*.safetensors"))
    shards = [str(shard.relative_to(path)) for shard in shard_paths]
    if not shards:
        missing.append("diffusion_pytorch_model-*.safetensors")
    if missing:
        raise FileNotFoundError(f"Wan checkpoint directory is missing required file(s): {missing}")
    return {
        "checkpoint_dir": str(path),
        "required_files": list(REQUIRED_WAN_CHECKPOINT_FILES),
        "diffusion_shards": shards,
        "model_paths": [
            str(path / "models_t5_umt5-xxl-enc-bf16.pth"),
            [str(shard) for shard in shard_paths],
            str(path / "Wan2.2_VAE.pth"),
        ],
        "tokenizer_path": str(path / "google" / "umt5-xxl"),
    }


def validate_diffsynth_repo(repo_dir: str | Path) -> Path:
    path = Path(repo_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"DiffSynth-Studio repo not found: {path}")
    if not (path / "diffsynth").exists():
        raise FileNotFoundError(f"DiffSynth package directory not found: {path / 'diffsynth'}")
    return path


def add_diffsynth_to_path(repo_dir: str | Path) -> Path:
    path = validate_diffsynth_repo(repo_dir)
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = path_text if existing_pythonpath is None else f"{path_text}{os.pathsep}{existing_pythonpath}"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return path


def validate_lora_path(lora_path: str | Path) -> Path:
    path = Path(lora_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {path}")
    return path


class DiffSynthWanLoraFutureGenerator:
    """Runs DiffSynth Wan2.2-TI2V with a LoRA and converts the video into IDM future frames."""

    def __init__(
        self,
        config: DiffSynthWanLoraConfig,
        *,
        pipe_loader: PipeLoader | None = None,
        video_reader: VideoReader = read_video_frames,
        video_saver: VideoSaver | None = None,
    ):
        self.config = config
        self.pipe_loader = pipe_loader
        self.video_reader = video_reader
        self.video_saver = video_saver
        self._pipe: Any | None = None

    def generate_future_stack(
        self,
        current_images: torch.Tensor,
        *,
        task_text: str,
        output_dir: str | Path,
        image_size: int,
        num_future_frames: int,
        view_index: int = 0,
        seed: int | None = None,
    ) -> Wan22Result:
        if current_images.ndim != 4:
            raise ValueError(f"Expected current_images with shape (V, 3, H, W), got {tuple(current_images.shape)}.")
        if not 0 <= view_index < current_images.shape[0]:
            raise ValueError(f"view_index {view_index} is out of range for {current_images.shape[0]} view(s).")
        result = self.generate_view(
            current_images[view_index],
            task_text=task_text,
            output_dir=output_dir,
            image_size=image_size,
            num_future_frames=num_future_frames,
            seed=seed,
            stem=f"wan_lora_view{view_index}",
        )
        return dataclasses.replace(result, future_images=result.future_images.unsqueeze(1))

    def generate_view(
        self,
        current_image: torch.Tensor,
        *,
        task_text: str,
        output_dir: str | Path,
        image_size: int,
        num_future_frames: int,
        seed: int | None = None,
        stem: str = "wan_lora_future",
    ) -> Wan22Result:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / f"{stem}_input.png"
        video_path = output_dir / f"{stem}.mp4"
        tensor_image_to_pil(current_image).save(input_path)

        resolved_seed = self.config.base_seed if seed is None else seed
        prompt = self.config.prompt_template.format(task=task_text)
        image = tensor_image_to_pil(current_image).resize((self.config.width, self.config.height))
        video = self.pipe(
            prompt=prompt,
            input_image=image,
            height=self.config.height,
            width=self.config.width,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            seed=resolved_seed,
            tiled=self.config.tiled,
        )
        self.save_video(video, video_path)
        frames = self.video_reader(video_path)
        indices = select_lora_future_frame_indices(
            len(frames),
            num_future_frames,
            frame_delta=self.config.frame_delta,
            strategy=self.config.future_frame_strategy,
        )
        if self.config.verify_conditioning_frame:
            verify_conditioning_frame(
                frames[0],
                current_image,
                image_size=image_size,
                max_mean_abs_error=self.config.conditioning_frame_max_mae,
            )
        future_images = torch.stack([image_to_chw_float(frames[index], image_size) for index in indices], dim=0)
        return Wan22Result(
            prompt=prompt,
            seed=resolved_seed,
            input_image_path=input_path,
            video_path=video_path,
            future_images=future_images,
            selected_frame_indices=tuple(indices),
            total_video_frames=len(frames),
        )

    def generate_view_latents(
        self,
        current_image: torch.Tensor,
        *,
        task_text: str,
        seed: int | None = None,
        stop_after_steps: int | None = None,
    ) -> DiffSynthWanLatentResult:
        if current_image.ndim != 3:
            raise ValueError(f"Expected current_image with shape (3, H, W), got {tuple(current_image.shape)}.")

        resolved_seed = self.config.base_seed if seed is None else seed
        prompt = self.config.prompt_template.format(task=task_text)
        image = tensor_image_to_pil(current_image).resize((self.config.width, self.config.height))
        result = generate_diffsynth_wan_predecode_latents(
            self.pipe,
            prompt=prompt,
            input_image=image,
            height=self.config.height,
            width=self.config.width,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            seed=resolved_seed,
            tiled=self.config.tiled,
            stop_after_steps=stop_after_steps,
        )
        metadata = {
            **result.metadata,
            "diffsynth_repo_dir": self.config.diffsynth_repo_dir,
            "checkpoint_dir": self.config.checkpoint_dir,
            "lora_path": self.config.lora_path,
            "lora_alpha": self.config.lora_alpha,
            "prompt_template": self.config.prompt_template,
        }
        return dataclasses.replace(result, metadata=metadata)

    @property
    def pipe(self) -> Any:
        if self._pipe is None:
            self._pipe = self.load_pipe()
        return self._pipe

    def load_pipe(self) -> Any:
        if self.pipe_loader is not None:
            return self.pipe_loader()
        repo_dir = add_diffsynth_to_path(self.config.diffsynth_repo_dir)
        checkpoint = validate_local_wan_checkpoint(self.config.checkpoint_dir)
        lora_path = validate_lora_path(self.config.lora_path)

        import torch
        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

        model_configs = [ModelConfig(path=path) for path in checkpoint["model_paths"]]
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=self.config.device,
            model_configs=model_configs,
            tokenizer_config=ModelConfig(path=checkpoint["tokenizer_path"]),
        )
        pipe.load_lora(pipe.dit, str(lora_path), alpha=self.config.lora_alpha)
        # Keep the repo path used in validation visible for debuggability.
        pipe._diffsynth_repo_dir = str(repo_dir)
        return pipe

    def save_video(self, video: Sequence[Image.Image], video_path: Path) -> None:
        if self.video_saver is not None:
            self.video_saver(video, str(video_path), self.config.fps, 5)
            return
        from diffsynth.utils.data import save_video

        save_video(video, str(video_path), fps=self.config.fps, quality=5)


def select_lora_future_frame_indices(
    total_frames: int,
    num_future_frames: int,
    *,
    frame_delta: int = 1,
    strategy: FutureFrameStrategy,
) -> list[int]:
    validate_future_frame_strategy(strategy)
    return select_future_frame_indices(
        total_frames,
        num_future_frames,
        frame_delta=frame_delta,
        strategy=strategy,
    )
