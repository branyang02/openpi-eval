from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from world_model.wan_prefix_encoder import DEFAULT_WAN_CHECKPOINT_DIR, DEFAULT_WAN_REPO_DIR
from world_model.wan_vae_encoder import add_repo_to_path, resolve_torch_dtype

DIFFSYNTH_WAN22_TI2V_DIT_SOURCE = "diffsynth_wan2.2_ti2v_5b_current_dit_hidden"
WAN_DIT_HIDDEN_POOL_DESCRIPTION = "model_fn_wan_video_block_hooks_mean_pool_selected_layers"
WAN_DIT_HIDDEN_POOL_MEAN = "mean"
WAN_DIT_HIDDEN_POOL_TOKEN_POOL = "token_pool"
DEFAULT_WAN_DIT_LAYERS = (0, 14, 29)
WAN_DIT_FUTURE_LATENT_NOISE_SEED_STRATEGY = "blake2b_63bit_base_seed_and_dataset_index"

ModelFn = Callable[..., torch.Tensor]


def _shared_diffsynth_timestep(value: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((1,), float(value), device=device, dtype=dtype)


def _validate_current_inputs(current_images: torch.Tensor, prompts: Sequence[str]) -> None:
    if current_images.ndim != 4:
        raise ValueError(f"current_images must have shape (B, 3, H, W), got {tuple(current_images.shape)}.")
    if current_images.shape[1] != 3:
        raise ValueError(f"current_images must have 3 RGB channels, got {current_images.shape[1]}.")
    if len(prompts) != current_images.shape[0]:
        raise ValueError(f"prompts length {len(prompts)} must match batch size {current_images.shape[0]}.")


def wan_dit_future_latent_noise_seed(base_seed: int, sample_index: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError(f"base_seed must be an integer, got {base_seed!r}.")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise ValueError(f"sample_index must be an integer, got {sample_index!r}.")
    if base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}.")
    if sample_index < 0:
        raise ValueError(f"sample_index must be non-negative, got {sample_index}.")
    payload = f"{base_seed}:{sample_index}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _normalize_sample_indices(
    sample_indices: Sequence[int] | torch.Tensor | None, *, batch_size: int
) -> list[int] | None:
    if sample_indices is None:
        return None
    if isinstance(sample_indices, torch.Tensor):
        values = [int(value.item()) for value in sample_indices.detach().cpu().reshape(-1)]
    else:
        values = [int(value) for value in sample_indices]
    if len(values) != batch_size:
        raise ValueError(f"sample_indices length {len(values)} must match batch size {batch_size}.")
    for value in values:
        if value < 0:
            raise ValueError(f"sample_indices must be non-negative, got {value}.")
    return values


def _fit_last_dim(values: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}.")
    if values.shape[-1] == width:
        return values
    if values.shape[-1] > width:
        original_shape = values.shape[:-1]
        pooled = F.adaptive_avg_pool1d(values.reshape(-1, 1, values.shape[-1]), width)
        return pooled.reshape(*original_shape, width)
    return F.pad(values, (0, width - values.shape[-1]))


def _resolve_existing_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _resolve_existing_dir(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _resolve_dit_checkpoint_paths(
    checkpoint_dir: str | Path,
    dit_checkpoint_paths: str | Path | Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    if dit_checkpoint_paths is None:
        checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
        paths = tuple(sorted(checkpoint_root.glob("diffusion_pytorch_model-*.safetensors")))
    elif isinstance(dit_checkpoint_paths, str | Path):
        candidate = Path(dit_checkpoint_paths).expanduser().resolve()
        if candidate.is_dir():
            paths = tuple(sorted(candidate.glob("diffusion_pytorch_model-*.safetensors")))
        else:
            paths = (candidate,)
    else:
        paths = tuple(Path(path).expanduser().resolve() for path in dit_checkpoint_paths)

    if not paths:
        raise FileNotFoundError(
            "Wan2.2 DiT checkpoint shard(s) not found. Expected diffusion_pytorch_model-*.safetensors under "
            f"{Path(checkpoint_dir).expanduser().resolve()}."
        )
    missing = [path for path in paths if not path.exists() or not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Wan2.2 DiT checkpoint shard(s) not found: {joined}")
    if any(path.suffix != ".safetensors" for path in paths):
        raise ValueError(f"Wan2.2 DiT checkpoint paths must be .safetensors shard files, got {paths}.")
    return paths


def _normalize_selected_layers(selected_layers: Sequence[int], *, block_count: int | None = None) -> tuple[int, ...]:
    if not selected_layers:
        raise ValueError("selected_layers must contain at least one layer index.")
    normalized: list[int] = []
    for layer in selected_layers:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError(f"selected_layers must contain integer layer indices, got {layer!r}.")
        if layer < 0:
            raise ValueError(f"selected_layers must be non-negative, got {layer}.")
        if block_count is not None and layer >= block_count:
            raise ValueError(f"selected layer {layer} is out of range for {block_count} Wan DiT block(s).")
        normalized.append(layer)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"selected_layers must not contain duplicates, got {tuple(normalized)}.")
    return tuple(normalized)


def _normalize_hidden_pool(hidden_pool: str) -> str:
    if hidden_pool == WAN_DIT_HIDDEN_POOL_DESCRIPTION:
        return WAN_DIT_HIDDEN_POOL_MEAN
    if hidden_pool == WAN_DIT_HIDDEN_POOL_MEAN:
        return WAN_DIT_HIDDEN_POOL_MEAN
    if hidden_pool in {WAN_DIT_HIDDEN_POOL_TOKEN_POOL, "adaptive_token_pool"}:
        return WAN_DIT_HIDDEN_POOL_TOKEN_POOL
    raise ValueError(f"hidden_pool must be 'mean', 'token_pool', or 'adaptive_token_pool', got {hidden_pool!r}.")


def _validate_tokens_per_layer(tokens_per_layer: int) -> int:
    if isinstance(tokens_per_layer, bool) or not isinstance(tokens_per_layer, int):
        raise ValueError(f"tokens_per_layer must be an integer, got {tokens_per_layer!r}.")
    if tokens_per_layer <= 0:
        raise ValueError(f"tokens_per_layer must be positive, got {tokens_per_layer}.")
    return tokens_per_layer


def _hidden_pool_metadata_value(hidden_pool: str) -> str:
    if hidden_pool == WAN_DIT_HIDDEN_POOL_MEAN:
        return WAN_DIT_HIDDEN_POOL_DESCRIPTION
    return hidden_pool


def _default_model_fn() -> ModelFn:
    from diffsynth.pipelines.wan_video import model_fn_wan_video

    return model_fn_wan_video


def _pool_block_output(output: Any, *, layer: int, hidden_pool: str, tokens_per_layer: int) -> torch.Tensor:
    if isinstance(output, tuple | list):
        if not output:
            raise RuntimeError(f"Wan DiT block {layer} returned an empty output sequence.")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise RuntimeError(f"Wan DiT block {layer} returned {type(output).__name__}, expected torch.Tensor.")
    if output.ndim == 3:
        values = output.detach().to(dtype=torch.float32)
        if hidden_pool == WAN_DIT_HIDDEN_POOL_MEAN:
            return values.mean(dim=1, keepdim=True)
        return F.adaptive_avg_pool1d(values.transpose(1, 2), tokens_per_layer).transpose(1, 2).contiguous()
    if output.ndim == 2:
        if hidden_pool == WAN_DIT_HIDDEN_POOL_TOKEN_POOL and tokens_per_layer != 1:
            raise RuntimeError(
                f"Wan DiT block {layer} returned tensor with shape {tuple(output.shape)}; "
                "hidden_pool='token_pool' requires block output shape (B, tokens, D) when tokens_per_layer > 1."
            )
        return output.detach().to(dtype=torch.float32).unsqueeze(1)
    raise RuntimeError(
        f"Wan DiT block {layer} returned tensor with shape {tuple(output.shape)}, expected (B, tokens, D) or (B, D)."
    )


def _metadata_timestep_value(timestep: torch.Tensor) -> float | list[float]:
    values = timestep.detach().to(dtype=torch.float32, device="cpu")
    if values.numel() == 1:
        return float(values.item())
    return [float(value) for value in values.tolist()]


@dataclasses.dataclass(frozen=True)
class WanDiTHiddenFeatureResult:
    prefix_tokens: torch.Tensor
    denoised_latents: torch.Tensor
    captured_layers: tuple[int, ...]
    metadata: Mapping[str, Any]


class WanDiTHiddenFeatureExtractor:
    """Extract pooled hidden features from DiffSynth Wan DiT blocks via forward hooks.

    The extractor intentionally calls ``diffsynth.pipelines.wan_video.model_fn_wan_video``
    instead of ``WanModel.forward`` because this local DiffSynth checkout has a
    ``WanModel.forward``/``patchify`` return-contract mismatch, while
    ``model_fn_wan_video`` is the generation path known to work. Result
    metadata records when DiffSynth's separated-timestep path makes the
    configured timestep apply only to future/noisy latent frames.
    """

    def __init__(
        self,
        dit: torch.nn.Module,
        *,
        selected_layers: Sequence[int] = DEFAULT_WAN_DIT_LAYERS,
        prefix_dim: int | None = None,
        hidden_pool: str = WAN_DIT_HIDDEN_POOL_MEAN,
        tokens_per_layer: int = 1,
        model_fn: ModelFn | None = None,
        freeze: bool = True,
    ) -> None:
        blocks = getattr(dit, "blocks", None)
        if blocks is None:
            raise ValueError("Wan DiT model must expose a blocks attribute.")
        self.block_count = len(blocks)
        self.selected_layers = _normalize_selected_layers(selected_layers, block_count=self.block_count)
        if prefix_dim is not None and prefix_dim <= 0:
            raise ValueError(f"prefix_dim must be positive when provided, got {prefix_dim}.")
        self.hidden_pool = _normalize_hidden_pool(hidden_pool)
        validated_tokens_per_layer = _validate_tokens_per_layer(tokens_per_layer)
        self.tokens_per_layer = 1 if self.hidden_pool == WAN_DIT_HIDDEN_POOL_MEAN else validated_tokens_per_layer
        self.prefix_dim = prefix_dim
        self.dit = dit
        self.model_fn = model_fn if model_fn is not None else _default_model_fn()
        if freeze:
            self.dit.eval()
            self.dit.requires_grad_(False)

    def _register_hooks(self, captures: dict[int, torch.Tensor]) -> list[torch.utils.hooks.RemovableHandle]:
        handles: list[torch.utils.hooks.RemovableHandle] = []
        blocks = getattr(self.dit, "blocks")

        def make_hook(layer: int) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], None]:
            def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                captures[layer] = _pool_block_output(
                    output,
                    layer=layer,
                    hidden_pool=self.hidden_pool,
                    tokens_per_layer=self.tokens_per_layer,
                )

            return hook

        for layer in self.selected_layers:
            handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
        return handles

    @torch.no_grad()
    def extract(
        self,
        *,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        fuse_vae_embedding_in_latents: bool = True,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> WanDiTHiddenFeatureResult:
        if latents.ndim != 5:
            raise ValueError(f"latents must have shape (B, C, T, H, W), got {tuple(latents.shape)}.")
        if context.ndim != 3:
            raise ValueError(f"context must have shape (B, tokens, D), got {tuple(context.shape)}.")
        if context.shape[0] != latents.shape[0]:
            raise ValueError(f"context batch size {context.shape[0]} must match latents batch size {latents.shape[0]}.")
        if timestep.ndim == 0:
            timestep = timestep.reshape(1)
        if timestep.ndim != 1:
            raise ValueError(f"timestep must have shape (B,) or (1,), got {tuple(timestep.shape)}.")

        captures: dict[int, torch.Tensor] = {}
        handles = self._register_hooks(captures)
        was_training = self.dit.training
        self.dit.eval()
        try:
            denoised = self.model_fn(
                dit=self.dit,
                latents=latents,
                timestep=timestep,
                context=context,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                **dict(model_kwargs or {}),
            )
        finally:
            for handle in handles:
                handle.remove()
            self.dit.train(was_training)

        missing = [layer for layer in self.selected_layers if layer not in captures]
        if missing:
            raise RuntimeError(f"Wan DiT hook(s) did not capture selected layer(s): {missing}.")
        prefix_tokens = torch.cat([captures[layer] for layer in self.selected_layers], dim=1)
        if self.prefix_dim is not None:
            prefix_tokens = _fit_last_dim(prefix_tokens, self.prefix_dim)
        timestep_applies_to_future_latents_only = bool(
            getattr(self.dit, "seperated_timestep", False) and fuse_vae_embedding_in_latents
        )
        future_latent_frames = max(int(latents.shape[2]) - 1, 0)
        effective_timestep: float | list[float] = _metadata_timestep_value(timestep)
        if timestep_applies_to_future_latents_only and future_latent_frames == 0:
            effective_timestep = 0.0
        return WanDiTHiddenFeatureResult(
            prefix_tokens=prefix_tokens,
            denoised_latents=denoised.detach(),
            captured_layers=self.selected_layers,
            metadata={
                "source": DIFFSYNTH_WAN22_TI2V_DIT_SOURCE,
                "hidden_pool": _hidden_pool_metadata_value(self.hidden_pool),
                "tokens_per_layer": self.tokens_per_layer,
                "selected_layers": list(self.selected_layers),
                "raw_hidden_dim": int(next(iter(captures.values())).shape[-1]),
                "prefix_dim": int(prefix_tokens.shape[-1]),
                "prefix_token_count": int(prefix_tokens.shape[1]),
                "denoised_latent_shape": list(denoised.shape),
                "timestep_shape": list(timestep.shape),
                "timestep_applies_to_future_latents_only": timestep_applies_to_future_latents_only,
                "future_latent_frames": future_latent_frames,
                "effective_timestep": effective_timestep,
            },
        )

    def extract_hidden_tokens(
        self,
        *,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        fuse_vae_embedding_in_latents: bool = True,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        return self.extract(
            latents=latents,
            timestep=timestep,
            context=context,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            model_kwargs=model_kwargs,
        ).prefix_tokens


class FrozenDiffSynthWanDiTCurrentPrefixEncoder:
    """Frozen Wan2.2 TI2V DiT hidden-prefix encoder for current image + task text only.

    Wan2.2 TI2V uses DiffSynth's separated-timestep path with fused VAE latents:
    the encoded current-frame latent receives timestep 0, while the configured
    ``timestep`` applies only to additional future/noisy latent frames. With the
    default ``num_latent_frames=1``, the prefix is effectively current-frame-only
    at timestep 0.
    """

    def __init__(
        self,
        *,
        repo_dir: str | Path = DEFAULT_WAN_REPO_DIR,
        checkpoint_dir: str | Path = DEFAULT_WAN_CHECKPOINT_DIR,
        dit_checkpoint_paths: str | Path | Sequence[str | Path] | None = None,
        vae_checkpoint_path: str | Path | None = None,
        text_encoder_checkpoint_path: str | Path | None = None,
        tokenizer_dir: str | Path | None = None,
        selected_layers: Sequence[int] = DEFAULT_WAN_DIT_LAYERS,
        prefix_dim: int = 3072,
        dtype: str = "bfloat16",
        timestep: float = 500.0,
        num_latent_frames: int = 1,
        future_latent_fill: str = "zeros",
        future_latent_seed: int = 0,
        hidden_pool: str = WAN_DIT_HIDDEN_POOL_MEAN,
        tokens_per_layer: int = 1,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
        vram_limit: float | None = None,
    ) -> None:
        if prefix_dim <= 0:
            raise ValueError(f"prefix_dim must be positive, got {prefix_dim}.")
        if num_latent_frames <= 0:
            raise ValueError(f"num_latent_frames must be positive, got {num_latent_frames}.")
        if future_latent_fill not in {"zeros", "noise"}:
            raise ValueError(f"future_latent_fill must be 'zeros' or 'noise', got {future_latent_fill!r}.")
        self.hidden_pool = _normalize_hidden_pool(hidden_pool)
        validated_tokens_per_layer = _validate_tokens_per_layer(tokens_per_layer)
        self.tokens_per_layer = 1 if self.hidden_pool == WAN_DIT_HIDDEN_POOL_MEAN else validated_tokens_per_layer

        checkpoint_root = Path(checkpoint_dir).expanduser()
        self.repo_dir = _resolve_existing_dir(repo_dir, label="DiffSynth-Studio repo")
        if not (self.repo_dir / "diffsynth").exists():
            raise FileNotFoundError(f"DiffSynth-Studio repo is missing diffsynth package: {self.repo_dir}")
        self.dit_checkpoint_paths = _resolve_dit_checkpoint_paths(checkpoint_root, dit_checkpoint_paths)
        self.vae_checkpoint_path = _resolve_existing_file(
            vae_checkpoint_path or checkpoint_root / "Wan2.2_VAE.pth",
            label="Wan2.2 VAE checkpoint",
        )
        self.text_encoder_checkpoint_path = _resolve_existing_file(
            text_encoder_checkpoint_path or checkpoint_root / "models_t5_umt5-xxl-enc-bf16.pth",
            label="Wan T5 text encoder checkpoint",
        )
        self.tokenizer_dir = _resolve_existing_dir(
            tokenizer_dir or checkpoint_root / "google" / "umt5-xxl",
            label="Wan T5 tokenizer directory",
        )
        self.selected_layers = _normalize_selected_layers(selected_layers)
        self.prefix_dim = prefix_dim
        self.dtype = dtype
        self.timestep = float(timestep)
        self.num_latent_frames = int(num_latent_frames)
        self.future_latent_fill = future_latent_fill
        self.future_latent_seed = int(future_latent_seed)
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.vram_limit = vram_limit
        self._pipe_by_device: dict[str, object] = {}
        self._extractor_by_device: dict[str, WanDiTHiddenFeatureExtractor] = {}

    def _load_pipeline(self, device: torch.device) -> tuple[object, WanDiTHiddenFeatureExtractor]:
        device_key = str(device)
        if device_key in self._pipe_by_device:
            return self._pipe_by_device[device_key], self._extractor_by_device[device_key]

        add_repo_to_path(self.repo_dir)
        torch_dtype = resolve_torch_dtype(self.dtype)
        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline, model_fn_wan_video

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device_key,
            model_configs=[
                ModelConfig(path=[str(path) for path in self.dit_checkpoint_paths]),
                ModelConfig(path=str(self.vae_checkpoint_path)),
                ModelConfig(path=str(self.text_encoder_checkpoint_path)),
            ],
            tokenizer_config=ModelConfig(path=str(self.tokenizer_dir)),
            redirect_common_files=False,
            vram_limit=self.vram_limit,
        )
        if pipe.dit is None:
            raise RuntimeError(f"DiffSynth did not load a Wan DiT from {self.dit_checkpoint_paths}.")
        if pipe.vae is None:
            raise RuntimeError(f"DiffSynth did not load a Wan VAE from {self.vae_checkpoint_path}.")
        if pipe.text_encoder is None:
            raise RuntimeError(
                f"DiffSynth did not load a Wan T5 text encoder from {self.text_encoder_checkpoint_path}."
            )
        if pipe.tokenizer is None:
            raise RuntimeError(f"DiffSynth did not load a Wan tokenizer from {self.tokenizer_dir}.")
        for module in (pipe.dit, pipe.vae, pipe.text_encoder):
            module.eval().requires_grad_(False)
        extractor = WanDiTHiddenFeatureExtractor(
            pipe.dit,
            selected_layers=self.selected_layers,
            prefix_dim=self.prefix_dim,
            hidden_pool=self.hidden_pool,
            tokens_per_layer=self.tokens_per_layer,
            model_fn=model_fn_wan_video,
        )
        self._pipe_by_device[device_key] = pipe
        self._extractor_by_device[device_key] = extractor
        return pipe, extractor

    def _encode_text_context(self, pipe: object, prompts: Sequence[str]) -> torch.Tensor:
        ids, mask = pipe.tokenizer(list(prompts), return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        context = pipe.text_encoder(ids, mask)
        valid = mask.gt(0)
        context = context.masked_fill(~valid.unsqueeze(-1), 0.0)
        return context

    def _encode_first_frame_latents(self, pipe: object, current_images: torch.Tensor) -> torch.Tensor:
        if current_images.shape[-2] % 16 != 0 or current_images.shape[-1] % 16 != 0:
            raise ValueError(
                "Wan DiT prefix encoding expects image height and width divisible by 16; "
                f"got {tuple(current_images.shape[-2:])}."
            )
        torch_dtype = resolve_torch_dtype(self.dtype)
        videos = current_images.clamp(0.0, 1.0).mul(2.0).sub(1.0).unsqueeze(2).contiguous()
        latents = pipe.vae.encode(
            [video.to(dtype=torch_dtype) for video in videos],
            device=str(current_images.device),
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )
        latents = latents.to(device=current_images.device, dtype=torch_dtype)
        if latents.ndim != 5:
            raise RuntimeError(f"Wan VAE returned latents with shape {tuple(latents.shape)}, expected (B, C, T, H, W).")
        if latents.shape[2] < 1:
            raise RuntimeError(f"Wan VAE returned no latent frames: {tuple(latents.shape)}.")
        return latents[:, :, :1]

    def _build_current_only_latents(
        self,
        first_frame_latents: torch.Tensor,
        *,
        sample_indices: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, channels, _, height, width = first_frame_latents.shape
        latents = first_frame_latents.new_zeros((batch_size, channels, self.num_latent_frames, height, width))
        latents[:, :, :1] = first_frame_latents
        if self.num_latent_frames > 1 and self.future_latent_fill == "noise":
            normalized_indices = _normalize_sample_indices(sample_indices, batch_size=batch_size)
            if normalized_indices is None:
                generator = torch.Generator(device=first_frame_latents.device).manual_seed(self.future_latent_seed)
                latents[:, :, 1:] = torch.randn(
                    (batch_size, channels, self.num_latent_frames - 1, height, width),
                    device=first_frame_latents.device,
                    dtype=first_frame_latents.dtype,
                    generator=generator,
                )
            else:
                future_shape = (1, channels, self.num_latent_frames - 1, height, width)
                for batch_position, sample_index in enumerate(normalized_indices):
                    seed = wan_dit_future_latent_noise_seed(self.future_latent_seed, sample_index)
                    generator = torch.Generator(device=first_frame_latents.device).manual_seed(seed)
                    latents[batch_position : batch_position + 1, :, 1:] = torch.randn(
                        future_shape,
                        device=first_frame_latents.device,
                        dtype=first_frame_latents.dtype,
                        generator=generator,
                    )
        return latents

    @torch.no_grad()
    def encode_prefix(
        self,
        current_images: torch.Tensor,
        prompts: Sequence[str],
        *,
        sample_indices: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_current_inputs(current_images, prompts)
        pipe, extractor = self._load_pipeline(current_images.device)
        first_frame_latents = self._encode_first_frame_latents(pipe, current_images)
        latents = self._build_current_only_latents(first_frame_latents, sample_indices=sample_indices)
        context = self._encode_text_context(pipe, prompts).to(device=current_images.device, dtype=latents.dtype)
        timestep = _shared_diffsynth_timestep(self.timestep, device=current_images.device, dtype=latents.dtype)
        return extractor.extract_hidden_tokens(
            latents=latents,
            timestep=timestep,
            context=context,
            fuse_vae_embedding_in_latents=True,
        ).to(dtype=torch.float32)


@dataclasses.dataclass
class WanDiTRandomSmokeArgs:
    repo_dir: str = DEFAULT_WAN_REPO_DIR
    checkpoint_dir: str = DEFAULT_WAN_CHECKPOINT_DIR
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    selected_layers: tuple[int, ...] = DEFAULT_WAN_DIT_LAYERS
    prefix_dim: int = 3072
    batch_size: int = 1
    latent_channels: int = 48
    latent_frames: int = 1
    latent_height: int = 2
    latent_width: int = 2
    context_tokens: int = 4
    text_dim: int = 4096
    timestep: float = 500.0
    hidden_pool: str = WAN_DIT_HIDDEN_POOL_MEAN
    tokens_per_layer: int = 1
    seed: int = 123
    require_gpu: bool = True


def run_wan_dit_random_feature_smoke(args: WanDiTRandomSmokeArgs) -> dict[str, Any]:
    """Guarded real-load smoke for the DiT hook path using random latents/text context."""

    device = torch.device(args.device)
    if args.require_gpu and device.type != "cuda":
        return {"skipped": True, "reason": f"require_gpu=True but requested device is {device}."}
    if args.require_gpu and not torch.cuda.is_available():
        return {"skipped": True, "reason": "CUDA is not available."}
    try:
        repo_dir = _resolve_existing_dir(args.repo_dir, label="DiffSynth-Studio repo")
        checkpoint_paths = _resolve_dit_checkpoint_paths(args.checkpoint_dir, None)
    except (FileNotFoundError, ValueError) as error:
        return {"skipped": True, "reason": str(error)}

    add_repo_to_path(repo_dir)
    torch_dtype = resolve_torch_dtype(args.dtype)
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline, model_fn_wan_video

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=str(device),
        model_configs=[ModelConfig(path=[str(path) for path in checkpoint_paths])],
        tokenizer_config=None,
        redirect_common_files=False,
    )
    if pipe.dit is None:
        raise RuntimeError(f"DiffSynth did not load a Wan DiT from {checkpoint_paths}.")

    extractor = WanDiTHiddenFeatureExtractor(
        pipe.dit,
        selected_layers=args.selected_layers,
        prefix_dim=args.prefix_dim,
        hidden_pool=args.hidden_pool,
        tokens_per_layer=args.tokens_per_layer,
        model_fn=model_fn_wan_video,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn(
        (args.batch_size, args.latent_channels, args.latent_frames, args.latent_height, args.latent_width),
        device=device,
        dtype=torch_dtype,
        generator=generator,
    )
    context = torch.randn(
        (args.batch_size, args.context_tokens, args.text_dim),
        device=device,
        dtype=torch_dtype,
        generator=generator,
    )
    timestep = _shared_diffsynth_timestep(args.timestep, device=device, dtype=torch_dtype)
    result = extractor.extract(
        latents=latents,
        timestep=timestep,
        context=context,
        fuse_vae_embedding_in_latents=True,
    )
    return {
        "skipped": False,
        "source": DIFFSYNTH_WAN22_TI2V_DIT_SOURCE,
        "selected_layers": list(result.captured_layers),
        "prefix_shape": list(result.prefix_tokens.shape),
        "denoised_latent_shape": list(result.denoised_latents.shape),
        "prefix_dtype": str(result.prefix_tokens.dtype),
        "hidden_pool": result.metadata["hidden_pool"],
        "tokens_per_layer": result.metadata["tokens_per_layer"],
        "timestep_shape": result.metadata["timestep_shape"],
        "effective_timestep": result.metadata["effective_timestep"],
        "device": str(device),
    }


def smoke_main(args: WanDiTRandomSmokeArgs) -> None:
    print(json.dumps(run_wan_dit_random_feature_smoke(args), sort_keys=True))


__all__ = [
    "DEFAULT_WAN_DIT_LAYERS",
    "DIFFSYNTH_WAN22_TI2V_DIT_SOURCE",
    "WAN_DIT_HIDDEN_POOL_DESCRIPTION",
    "WAN_DIT_FUTURE_LATENT_NOISE_SEED_STRATEGY",
    "WAN_DIT_HIDDEN_POOL_MEAN",
    "WAN_DIT_HIDDEN_POOL_TOKEN_POOL",
    "FrozenDiffSynthWanDiTCurrentPrefixEncoder",
    "WanDiTHiddenFeatureExtractor",
    "WanDiTHiddenFeatureResult",
    "WanDiTRandomSmokeArgs",
    "run_wan_dit_random_feature_smoke",
    "wan_dit_future_latent_noise_seed",
]
