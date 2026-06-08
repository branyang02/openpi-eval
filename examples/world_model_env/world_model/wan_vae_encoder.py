from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Protocol

import torch

from world_model.config import ModelConfig


class WanVaeEncoder(Protocol):
    def encode_videos(self, videos: torch.Tensor) -> torch.Tensor:
        """Encode videos shaped ``(B, C, T, H, W)`` in ``[-1, 1]`` to Wan latents."""


def add_repo_to_path(repo_dir: str | Path) -> Path:
    path = Path(repo_dir).expanduser().resolve()
    if not (path / "diffsynth").exists():
        raise FileNotFoundError(f"DiffSynth-Studio repo not found or missing diffsynth package: {path}")
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath is None:
        os.environ["PYTHONPATH"] = path_text
    elif path_text not in existing_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{path_text}{os.pathsep}{existing_pythonpath}"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return path


def resolve_wan_vae_checkpoint(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "Wan2.2_VAE.pth"
    if not resolved.exists():
        raise FileNotFoundError(f"Wan2.2 VAE checkpoint not found: {resolved}")
    return resolved


def resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported Wan VAE dtype {name!r}.")


class FrozenDiffSynthWanVaeEncoder:
    """Lazy frozen Wan2.2 VAE encoder backed by DiffSynth-Studio.

    The loaded VAE is intentionally not a child module of the IDM. It remains
    frozen, is omitted from IDM checkpoints, and is reloaded from
    ``wan_vae_checkpoint_path`` when the checkpoint is used again.
    """

    def __init__(
        self,
        *,
        repo_dir: str,
        checkpoint_path: str,
        dtype: str = "bfloat16",
        tiled: bool = False,
    ):
        self.repo_dir = repo_dir
        self.checkpoint_path = checkpoint_path
        self.dtype = dtype
        self.tiled = tiled
        self._vae_by_device: dict[str, object] = {}

    def _load_vae(self, device: torch.device) -> object:
        device_key = str(device)
        if device_key in self._vae_by_device:
            return self._vae_by_device[device_key]

        add_repo_to_path(self.repo_dir)
        checkpoint = resolve_wan_vae_checkpoint(self.checkpoint_path)
        torch_dtype = resolve_torch_dtype(self.dtype)

        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device_key,
            model_configs=[ModelConfig(path=str(checkpoint))],
            tokenizer_config=None,
        )
        if pipe.vae is None:
            raise RuntimeError(f"DiffSynth did not load a Wan VAE from {checkpoint}.")
        pipe.vae.eval().requires_grad_(False)
        self._vae_by_device[device_key] = pipe.vae
        return pipe.vae

    @torch.no_grad()
    def encode_videos(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5:
            raise ValueError(f"videos must have shape (B, C, T, H, W), got {tuple(videos.shape)}.")
        vae = self._load_vae(videos.device)
        encoded = vae.encode(
            [video.to(dtype=resolve_torch_dtype(self.dtype)) for video in videos],
            device=str(videos.device),
            tiled=self.tiled,
        )
        return encoded.to(device=videos.device, dtype=torch.float32)


class FakeWanVaeEncoder:
    """Small deterministic encoder for unit tests; never used by training CLIs."""

    def __init__(self, *, latent_channels: int = 48, spatial_stride: int = 16):
        self.latent_channels = latent_channels
        self.spatial_stride = spatial_stride

    @torch.no_grad()
    def encode_videos(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5:
            raise ValueError(f"videos must have shape (B, C, T, H, W), got {tuple(videos.shape)}.")
        batch_size, _, num_frames, height, width = videos.shape
        latent_frames = (num_frames + 3) // 4
        latent_height = height // self.spatial_stride
        latent_width = width // self.spatial_stride
        pooled = torch.nn.functional.adaptive_avg_pool3d(
            videos.mean(dim=1, keepdim=True),
            (latent_frames, latent_height, latent_width),
        )
        channel_scale = torch.linspace(
            0.5,
            1.5,
            self.latent_channels,
            device=videos.device,
            dtype=videos.dtype,
        ).view(1, self.latent_channels, 1, 1, 1)
        return pooled.expand(batch_size, self.latent_channels, latent_frames, latent_height, latent_width) * channel_scale


def build_frozen_wan_vae_encoder(config: ModelConfig) -> FrozenDiffSynthWanVaeEncoder:
    if config.wan_vae_repo_dir is None:
        raise ValueError("wan_vae_repo_dir is required when idm_visual_encoder='wan_vae'.")
    if config.wan_vae_checkpoint_path is None:
        raise ValueError("wan_vae_checkpoint_path is required when idm_visual_encoder='wan_vae'.")
    return FrozenDiffSynthWanVaeEncoder(
        repo_dir=config.wan_vae_repo_dir,
        checkpoint_path=config.wan_vae_checkpoint_path,
        dtype=config.wan_vae_dtype,
        tiled=config.wan_vae_tiled,
    )
