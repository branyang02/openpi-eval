from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
import tyro
from torch.utils.data import DataLoader

from world_model.config import DatasetConfig, DatasetSource
from world_model.train_lib import (
    create_dataset_with_optional_cache,
    create_flow_sample_noise,
    enforce_idm_frame_delta_contract,
    get_action_normalizer,
    idm_history_kwargs,
    idm_uses_flow_matching,
    load_idm_checkpoint,
    resolve_device,
    seed_everything,
    to_device,
)


@dataclasses.dataclass
class Args:
    checkpoint: str
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_keys: tuple[str, ...] = ("corner4.image",)
    output_dir: str = "output/idm_horizon_diagnostics"
    cached_future_dir: str | None = None
    image_size: int = 64
    frame_delta: int = 4
    num_future_frames: int = 1
    action_horizon: int = 32
    max_samples: int | None = None
    samples_per_episode: int | None = None
    episodes: tuple[int, ...] | None = None
    synthetic_samples: int = 128
    batch_size: int = 16
    device: str = "auto"
    flow_eval_seed: int | None = 0
    seed: int = 7


def _validate_action_inputs(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    if predicted.ndim != 3:
        raise ValueError(f"predicted must have shape (B, A, D), got {tuple(predicted.shape)}.")
    if tuple(target.shape) != tuple(predicted.shape):
        raise ValueError(f"target shape {tuple(target.shape)} must match predicted shape {tuple(predicted.shape)}.")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (B, A), got {tuple(mask.shape)}.")
    if tuple(mask.shape) != tuple(predicted.shape[:2]):
        raise ValueError(f"mask shape {tuple(mask.shape)} must match action shape {tuple(predicted.shape[:2])}.")


def _maybe_int(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def _masked_mean(total: torch.Tensor, count: torch.Tensor) -> float | None:
    count_value = float(count.detach().cpu().item())
    if count_value == 0.0:
        return None
    return float((total / count).detach().cpu().item())


def _first_valid(values: list[float | None]) -> float | None:
    return values[0] if values else None


def _last_valid(values: list[float | None]) -> float | None:
    return values[-1] if values else None


def horizon_action_diagnostics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    """Compute mask-aware IDM errors by action horizon step.

    ``per_action_step_*`` metrics average over valid samples and action dims at
    that horizon step. ``per_action_step_dim_*`` metrics average over valid
    samples for each action dim independently. Fully masked steps report
    ``None`` so callers can distinguish "no data" from a true zero error.
    """

    _validate_action_inputs(predicted, target, mask)
    predicted = predicted.detach().to(torch.float32)
    target = target.detach().to(device=predicted.device, dtype=predicted.dtype)
    weights = mask.detach().to(device=predicted.device, dtype=predicted.dtype).clamp_min(0.0)

    error = predicted - target
    squared = error.square()
    absolute = error.abs()
    step_count = weights.sum(dim=0)
    action_dim = int(predicted.shape[-1])
    expanded_weights = weights.unsqueeze(-1).expand_as(predicted)
    element_count = expanded_weights.sum()

    idm_mse = _masked_mean((squared * expanded_weights).sum(), element_count)
    idm_mae = _masked_mean((absolute * expanded_weights).sum(), element_count)

    per_action_step_mse: list[float | None] = []
    per_action_step_mae: list[float | None] = []
    per_action_step_dim_mse: list[list[float | None]] = []
    per_action_step_dim_mae: list[list[float | None]] = []
    for step in range(int(predicted.shape[1])):
        count = step_count[step]
        step_weights = weights[:, step].unsqueeze(-1)
        dim_count = count.clamp_min(0.0)
        if float(dim_count.detach().cpu().item()) == 0.0:
            per_action_step_mse.append(None)
            per_action_step_mae.append(None)
            per_action_step_dim_mse.append([None] * action_dim)
            per_action_step_dim_mae.append([None] * action_dim)
            continue

        step_element_count = dim_count * action_dim
        per_action_step_mse.append(
            float(((squared[:, step] * step_weights).sum() / step_element_count).detach().cpu().item())
        )
        per_action_step_mae.append(
            float(((absolute[:, step] * step_weights).sum() / step_element_count).detach().cpu().item())
        )
        per_action_step_dim_mse.append(
            [float(value) for value in ((squared[:, step] * step_weights).sum(dim=0) / dim_count).detach().cpu()]
        )
        per_action_step_dim_mae.append(
            [float(value) for value in ((absolute[:, step] * step_weights).sum(dim=0) / dim_count).detach().cpu()]
        )

    later_weights = weights[:, 1:]
    later_element_count = later_weights.unsqueeze(-1).expand_as(predicted[:, 1:]).sum()
    later_action_mse = _masked_mean((squared[:, 1:] * later_weights.unsqueeze(-1)).sum(), later_element_count)
    later_action_mae = _masked_mean((absolute[:, 1:] * later_weights.unsqueeze(-1)).sum(), later_element_count)
    first_action_mse = _first_valid(per_action_step_mse)
    first_action_mae = _first_valid(per_action_step_mae)
    last_action_mse = _last_valid(per_action_step_mse)
    last_action_mae = _last_valid(per_action_step_mae)
    first_vs_later_mse_ratio = None
    if first_action_mse is not None and later_action_mse not in (None, 0.0):
        first_vs_later_mse_ratio = first_action_mse / later_action_mse

    return {
        "num_valid_actions": _maybe_int(float(step_count.sum().detach().cpu().item())),
        "idm_mse": idm_mse,
        "idm_mae": idm_mae,
        "per_action_step_mse": per_action_step_mse,
        "per_action_step_mae": per_action_step_mae,
        "per_action_step_valid_count": [_maybe_int(float(value)) for value in step_count.detach().cpu().tolist()],
        "first_action_mse": first_action_mse,
        "first_action_mae": first_action_mae,
        "last_action_mse": last_action_mse,
        "last_action_mae": last_action_mae,
        "later_action_mse": later_action_mse,
        "later_action_mae": later_action_mae,
        "first_vs_later_mse_ratio": first_vs_later_mse_ratio,
        "per_action_step_dim_mse": per_action_step_dim_mse,
        "per_action_step_dim_mae": per_action_step_dim_mae,
    }


@torch.no_grad()
def main(args: Args) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    idm, model_config = load_idm_checkpoint(args.checkpoint, device)
    enforce_idm_frame_delta_contract(args.checkpoint, args.frame_delta)
    action_normalizer = get_action_normalizer(idm, device)
    idm.eval()

    dataset_config = DatasetConfig(
        source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=args.image_keys,
        frame_delta=args.frame_delta,
        num_future_frames=args.num_future_frames,
        action_horizon=args.action_horizon,
        image_size=args.image_size,
        max_samples=args.max_samples,
        samples_per_episode=args.samples_per_episode,
        episodes=args.episodes,
        synthetic_samples=args.synthetic_samples,
        idm_history_length=model_config.idm_history_length,
        seed=args.seed,
    )
    dataset = create_dataset_with_optional_cache(dataset_config, args.cached_future_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    flow_generator = None
    if idm_uses_flow_matching(idm) and args.flow_eval_seed is not None:
        flow_generator = torch.Generator(device=device).manual_seed(args.flow_eval_seed)

    target_chunks = []
    predicted_chunks = []
    mask_chunks = []
    for batch in loader:
        device_batch = to_device(batch, device)
        sample_noise = None
        if idm_uses_flow_matching(idm):
            sample_noise = create_flow_sample_noise(
                idm,
                batch_size=int(device_batch["current_images"].shape[0]),
                device=device,
                dtype=device_batch["current_images"].dtype,
                generator=flow_generator,
            )
        model_action = idm(
            device_batch["current_images"],
            device_batch["future_images"],
            device_batch["state"],
            device_batch["task_id"],
            sample_noise=sample_noise,
            **idm_history_kwargs(device_batch, idm=idm, action_normalizer=action_normalizer),
        )
        predicted = model_action if action_normalizer is None else action_normalizer.denormalize(model_action)
        target_chunks.append(device_batch["action_chunk"].detach().cpu())
        predicted_chunks.append(predicted.detach().cpu())
        mask_chunks.append(device_batch["action_mask"].detach().cpu())

    if not target_chunks:
        raise ValueError("Cannot diagnose an empty dataset.")

    horizon_metrics = horizon_action_diagnostics(
        torch.cat(predicted_chunks, dim=0),
        torch.cat(target_chunks, dim=0),
        torch.cat(mask_chunks, dim=0),
    )
    metrics = {
        "dataset_config": dataclasses.asdict(dataset_config),
        "checkpoint": args.checkpoint,
        "cached_future_dir": args.cached_future_dir,
        "num_samples": len(dataset),
        **horizon_metrics,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "idm_horizon_diagnostics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
