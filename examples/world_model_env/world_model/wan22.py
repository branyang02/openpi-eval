from __future__ import annotations

import dataclasses
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from world_model.config import FutureFrameStrategy, Wan22Config, validate_future_frame_strategy
from world_model.data import expected_wan_selected_frame_indices, image_to_chw_float

# Wan I2V/TI2V emits the conditioning (current) image as decoded frame 0, so callers skip it
# and treat frames[1:] as the future. This tolerance is the maximum mean-absolute-error (in
# normalized [0, 1] space) allowed between frame 0 and the conditioning image before the
# contract is considered violated. It is intentionally lenient: real frame 0 is a VAE round-trip
# of the conditioning image (and may be resized), so it is close to but not identical to the
# input, while a dropped/shifted frame 0 differs far more than this.
DEFAULT_CONDITIONING_FRAME_MAX_MAE = 0.2

Runner = Callable[..., subprocess.CompletedProcess[str]]
VideoReader = Callable[[Path], Sequence[np.ndarray]]


@dataclasses.dataclass(frozen=True)
class Wan22Result:
    prompt: str
    seed: int
    input_image_path: Path
    video_path: Path
    future_images: torch.Tensor
    selected_frame_indices: tuple[int, ...]
    total_video_frames: int


def read_video_frames(video_path: Path) -> list[np.ndarray]:
    return [np.asarray(frame)[..., :3] for frame in iio.imiter(video_path)]


def select_future_frame_indices(
    total_frames: int,
    num_future_frames: int,
    *,
    frame_delta: int = 1,
    strategy: FutureFrameStrategy = "first",
) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Wan2.2 did not produce any video frames.")
    if num_future_frames <= 0:
        raise ValueError("num_future_frames must be positive.")
    if frame_delta <= 0:
        raise ValueError(f"frame_delta must be positive, got {frame_delta}.")
    validated_strategy = validate_future_frame_strategy(strategy)
    indices = expected_wan_selected_frame_indices(
        frame_delta,
        num_future_frames,
        strategy=validated_strategy,
    )
    minimum_frames = max(indices) + 1
    if total_frames < minimum_frames:
        raise ValueError(
            f"Need at least {minimum_frames} frames to skip the conditioning frame and select "
            f"future_frame_strategy={validated_strategy!r} with frame_delta={frame_delta}, "
            f"num_future_frames={num_future_frames}; got {total_frames}."
        )
    return indices


def conditioning_frame_mae(first_frame: Any, current_image: Any, *, image_size: int) -> float:
    """Measure how closely generated video frame 0 matches the conditioning image."""

    expected = image_to_chw_float(current_image, image_size)
    actual = image_to_chw_float(first_frame, image_size)
    return float((actual - expected).abs().mean().item())


def verify_conditioning_frame(
    first_frame: Any,
    current_image: torch.Tensor,
    *,
    image_size: int,
    max_mean_abs_error: float,
) -> float:
    """Fail loudly when generated video frame 0 is not the conditioning image.

    Future-frame extraction assumes Wan/DiffSynth emits the conditioning (current) image as
    decoded frame 0 and that the future starts at frame 1. This re-checks that assumption against
    the actual generated output instead of trusting it silently: it compares frame 0 to the
    conditioning image (both normalized to ``image_size``) and raises ``ValueError`` when their
    mean-absolute-error exceeds ``max_mean_abs_error``. Returns the measured error so callers can
    log it.
    """
    mean_abs_error = conditioning_frame_mae(first_frame, current_image, image_size=image_size)
    if mean_abs_error > max_mean_abs_error:
        raise ValueError(
            "Generated video frame 0 does not match the conditioning image "
            f"(mean abs error {mean_abs_error:.4f} > tolerance {max_mean_abs_error:.4f}). "
            "Future-frame extraction assumes frame 0 is the conditioning image and that the future "
            "starts at frame 1; this output violates that contract. Inspect the saved video, or set "
            "verify_conditioning_frame=False / raise conditioning_frame_max_mae if this is expected."
        )
    return mean_abs_error


def tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    if image.shape[0] != 3 or image.ndim != 3:
        raise ValueError(f"Expected image with shape (3, H, W), got {tuple(image.shape)}.")
    array = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


class Wan22FutureGenerator:
    """Runs Wan2.2 TI2V/I2V generation and converts the video into IDM future frames."""

    def __init__(
        self,
        config: Wan22Config,
        *,
        runner: Runner = subprocess.run,
        video_reader: VideoReader = read_video_frames,
        verify_conditioning_frame: bool = True,
        conditioning_frame_max_mae: float = DEFAULT_CONDITIONING_FRAME_MAX_MAE,
    ):
        self.config = config
        self.runner = runner
        self.video_reader = video_reader
        self.verify_conditioning_frame = verify_conditioning_frame
        self.conditioning_frame_max_mae = conditioning_frame_max_mae

    def generate_view(
        self,
        current_image: torch.Tensor,
        *,
        task_text: str,
        output_dir: str | Path,
        image_size: int,
        num_future_frames: int,
        seed: int | None = None,
        stem: str = "wan22_future",
    ) -> Wan22Result:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = (output_dir / f"{stem}_input.png").resolve()
        video_path = (output_dir / f"{stem}.mp4").resolve()
        tensor_image_to_pil(current_image).save(input_path)

        resolved_seed = self.config.base_seed if seed is None else seed
        prompt = self.config.prompt_template.format(task=task_text)
        command = self.build_command(input_path=input_path, video_path=video_path, prompt=prompt, seed=resolved_seed)
        try:
            self.runner(command, cwd=str(self.repo_dir), check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Wan2.2 generation failed.\n"
                f"command: {' '.join(command)}\n"
                f"stdout:\n{exc.stdout or ''}\n"
                f"stderr:\n{exc.stderr or ''}"
            ) from exc

        frames = self.video_reader(video_path)
        indices = select_future_frame_indices(
            len(frames),
            num_future_frames,
            frame_delta=self.config.frame_delta,
            strategy=self.config.future_frame_strategy,
        )
        if self.verify_conditioning_frame:
            verify_conditioning_frame(
                frames[0],
                current_image,
                image_size=image_size,
                max_mean_abs_error=self.conditioning_frame_max_mae,
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
            stem=f"wan22_view{view_index}",
        )
        future_images = result.future_images.unsqueeze(1)
        return dataclasses.replace(result, future_images=future_images)

    @property
    def repo_dir(self) -> Path:
        repo_dir = Path(self.config.repo_dir).expanduser().resolve()
        generate_py = repo_dir / "generate.py"
        if not generate_py.exists():
            raise FileNotFoundError(f"Wan2.2 generate.py not found: {generate_py}")
        return repo_dir

    @property
    def checkpoint_dir(self) -> Path:
        checkpoint_dir = Path(self.config.checkpoint_dir).expanduser().resolve()
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Wan2.2 checkpoint_dir not found: {checkpoint_dir}")
        return checkpoint_dir

    def build_command(self, *, input_path: Path, video_path: Path, prompt: str, seed: int) -> list[str]:
        command = [
            self.config.python_executable,
            str(self.repo_dir / "generate.py"),
            "--task",
            self.config.task,
            "--size",
            self.config.size,
            "--ckpt_dir",
            str(self.checkpoint_dir),
            "--image",
            str(input_path),
            "--prompt",
            prompt,
            "--frame_num",
            str(self.config.frame_num),
            "--base_seed",
            str(seed),
            "--save_file",
            str(video_path),
        ]
        if self.config.sample_steps is not None:
            command.extend(["--sample_steps", str(self.config.sample_steps)])
        if self.config.sample_shift is not None:
            command.extend(["--sample_shift", str(self.config.sample_shift)])
        if self.config.sample_guide_scale is not None:
            command.extend(["--sample_guide_scale", str(self.config.sample_guide_scale)])
        if self.config.offload_model:
            command.extend(["--offload_model", "True"])
        if self.config.convert_model_dtype:
            command.append("--convert_model_dtype")
        if self.config.t5_cpu:
            command.append("--t5_cpu")
        return command
