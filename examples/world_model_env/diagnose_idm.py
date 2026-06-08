from __future__ import annotations

import contextlib
import dataclasses
import json
from pathlib import Path
from typing import Any

import matplotlib
import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader

from world_model.config import DatasetConfig, DatasetSource, IdmFutureRankingScoreMode
from world_model.data import CachedWanVaeLatentDataset
from world_model.train_lib import (
    _flow_sampled_action_prediction,
    _flow_transition_context_and_visual_tokens,
    create_dataset_with_optional_cache,
    create_flow_sample_noise,
    enforce_idm_frame_delta_contract,
    get_action_normalizer,
    get_state_normalizer,
    idm_history_kwargs,
    idm_uses_flow_matching,
    load_idm_checkpoint,
    masked_mse,
    masked_mse_sum_and_count,
    masked_smooth_l1,
    masked_smooth_l1_sum_and_count,
    normalize_state_for_idm,
    resolve_device,
    resolve_flow_num_samples,
    resolve_flow_sample_noise_scale,
    seed_everything,
    temporary_flow_sampling_config,
    to_device,
    unwrap_model,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_FUTURE_RANKING_CANDIDATES = ("real_gt", "current_repeated", "shuffled", "zero", "noise")
_FUTURE_RANKING_TIME_VALUE = 0.5


@dataclasses.dataclass
class Args:
    checkpoint: str
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_keys: tuple[str, ...] = ("corner4.image",)
    output_dir: str = "output/idm_diagnostics"
    cached_future_dir: str | None = None
    wan_vae_latent_cache_dir: str | None = None
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
    flow_num_samples: int | None = None
    flow_noise_scale: float | None = None
    seed: int = 7
    future_blind_output_delta_mse_min: float = 1e-4
    future_blind_degradation_min: float = 1e-4
    fail_on_future_blind: bool = False
    future_usage_score_mode: IdmFutureRankingScoreMode = "teacher_forced_endpoint"


def _flow_sampling_context(
    idm: torch.nn.Module,
    flow_num_samples: int | None,
    flow_noise_scale: float | None,
):
    if flow_num_samples is None and flow_noise_scale is None:
        return contextlib.nullcontext()
    return temporary_flow_sampling_config(idm, num_samples=flow_num_samples, noise_scale=flow_noise_scale)


def _masked_action_values(actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.detach().cpu().to(dtype=torch.bool)
    if valid.ndim != 2:
        raise ValueError(f"action mask must have shape (B, A), got {tuple(valid.shape)}.")
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape (B, A, D), got {tuple(actions.shape)}.")
    if tuple(actions.shape[:2]) != tuple(valid.shape):
        raise ValueError(f"action/mask shape mismatch: {tuple(actions.shape[:2])} != {tuple(valid.shape)}")
    return actions.detach().cpu()[valid]


def _stats(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        raise ValueError("Cannot compute statistics from an empty tensor.")
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "abs_gt_1_fraction": float((values.abs() > 1.0).to(torch.float32).mean().item()),
    }


def stepwise_action_diagnostics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    """Mask-aware MSE/MAE broken down by action-horizon step.

    ``predicted`` and ``target`` are ``(N, A, D)`` (batch, action horizon, action
    dim); ``mask`` is ``(N, A)``. For each horizon step the squared / absolute
    errors are aggregated over the batch and action dims, counting only valid
    actions. This isolates whether the IDM error is concentrated on the first
    action or grows across the horizon. The metrics are derived from
    already-predicted tensors, so they are deterministic and independent of how
    ``predicted`` was produced (e.g. flow-matching sampling).

    ``per_action_step_valid_count`` is the number of valid action *vectors* per
    step, so it sums to ``num_valid_actions`` and the count-weighted mean of
    ``per_action_step_mse`` reproduces the aggregate ``idm_mse``. Steps with no
    valid actions report ``0.0`` (the count signals the metric is empty).
    ``first_*`` / ``last_*`` are the metrics at horizon index ``0`` and the final
    horizon index. ``per_action_step_dim_mse`` is an ``action_horizon`` x
    ``action_dim`` breakdown (one row per step).
    """
    predicted = predicted.detach().cpu()
    target = target.detach().cpu()
    if predicted.ndim != 3:
        raise ValueError(f"predicted/target must have shape (N, A, D), got {tuple(predicted.shape)}.")
    if predicted.shape != target.shape:
        raise ValueError(f"predicted/target shape mismatch: {tuple(predicted.shape)} != {tuple(target.shape)}")
    valid = mask.detach().cpu().to(dtype=torch.bool)
    if valid.ndim != 2 or tuple(valid.shape) != tuple(predicted.shape[:2]):
        raise ValueError(f"action mask must have shape (N, A)={tuple(predicted.shape[:2])}, got {tuple(valid.shape)}.")

    action_dim = int(predicted.shape[-1])
    mask_float = valid.to(predicted.dtype)  # (N, A)
    error = (predicted - target) * mask_float.unsqueeze(-1)  # (N, A, D), masked entries -> 0
    squared = error.square()
    absolute = error.abs()

    step_count = mask_float.sum(dim=0)  # (A,) valid action vectors per step
    step_elem_count = (step_count * action_dim).clamp_min(1.0)  # valid scalar elements per step
    step_mse = squared.sum(dim=(0, 2)) / step_elem_count  # (A,)
    step_mae = absolute.sum(dim=(0, 2)) / step_elem_count  # (A,)
    step_dim_mse = squared.sum(dim=0) / step_count.clamp_min(1.0).unsqueeze(-1)  # (A, D)

    return {
        "per_action_step_mse": [float(value) for value in step_mse],
        "per_action_step_mae": [float(value) for value in step_mae],
        "per_action_step_valid_count": [int(value) for value in step_count],
        "first_action_mse": float(step_mse[0]),
        "first_action_mae": float(step_mae[0]),
        "last_action_mse": float(step_mse[-1]),
        "last_action_mae": float(step_mae[-1]),
        "per_action_step_dim_mse": [[float(value) for value in row] for row in step_dim_mse],
    }


def _write_trace_plot(target: torch.Tensor, predicted: torch.Tensor, output_path: Path) -> None:
    target = target.detach().cpu()
    predicted = predicted.detach().cpu()
    action_dim = int(target.shape[-1])
    time = range(int(target.shape[0]))
    fig, axes = plt.subplots(action_dim, 1, figsize=(8, 2.1 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]
    for dim, axis in enumerate(axes):
        axis.plot(time, target[:, dim], label="target", linewidth=2)
        axis.plot(time, predicted[:, dim], label="predicted", linewidth=2, linestyle="--")
        axis.set_ylabel(f"a{dim}")
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("action step")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_histogram_plot(target: torch.Tensor, predicted: torch.Tensor, output_path: Path) -> None:
    target = target.detach().cpu()
    predicted = predicted.detach().cpu()
    action_dim = int(target.shape[-1])
    fig, axes = plt.subplots(action_dim, 1, figsize=(8, 2.1 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]
    for dim, axis in enumerate(axes):
        axis.hist(target[:, dim].numpy(), bins=40, alpha=0.55, label="target")
        axis.hist(predicted[:, dim].numpy(), bins=40, alpha=0.55, label="predicted")
        axis.axvline(-1.0, color="black", linewidth=1, alpha=0.35)
        axis.axvline(1.0, color="black", linewidth=1, alpha=0.35)
        axis.set_ylabel(f"a{dim}")
        axis.grid(True, alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("action value")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _wan_vae_kwargs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "wan_vae_latents" not in batch:
        return {}
    return {"wan_vae_latents": batch["wan_vae_latents"]}


def _future_variants(batch: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    future_images = batch["future_images"]
    image_variants = {
        "zero": torch.zeros_like(future_images),
        "current_repeated": batch["current_images"].unsqueeze(1).expand_as(future_images),
        "noise": torch.rand_like(future_images),
    }
    if future_images.shape[0] > 1:
        image_variants["shuffled"] = torch.roll(future_images, shifts=1, dims=0)
    else:
        image_variants["shuffled"] = torch.flip(future_images, dims=(1,))

    if "wan_vae_latents" not in batch:
        return {name: {"future_images": variant} for name, variant in image_variants.items()}

    latents = batch["wan_vae_latents"]
    if latents.ndim != 5:
        raise ValueError(f"wan_vae_latents must have shape (B, C, T, H, W), got {tuple(latents.shape)}.")
    latent_variants = {
        "zero": torch.zeros_like(latents),
        "current_repeated": latents[:, :, :1].expand_as(latents),
        "noise": torch.rand_like(latents),
    }
    if latents.shape[0] > 1:
        latent_variants["shuffled"] = torch.roll(latents, shifts=1, dims=0)
    else:
        latent_variants["shuffled"] = torch.flip(latents, dims=(2,))
    return {
        name: {"future_images": image_variants[name], "wan_vae_latents": latent_variants[name]}
        for name in image_variants
    }


def _empty_sensitivity_totals() -> dict[str, float]:
    return {
        "target_mse_sum": 0.0,
        "target_mse_count": 0.0,
        "target_smooth_l1_sum": 0.0,
        "target_smooth_l1_count": 0.0,
        "output_delta_mse_sum": 0.0,
        "output_delta_mse_count": 0.0,
    }


def _finalize_sensitivity_metrics(totals: dict[str, float]) -> dict[str, float]:
    return {
        "target_mse": totals["target_mse_sum"] / max(totals["target_mse_count"], 1.0),
        "target_smooth_l1": totals["target_smooth_l1_sum"] / max(totals["target_smooth_l1_count"], 1.0),
        "output_delta_mse": totals["output_delta_mse_sum"] / max(totals["output_delta_mse_count"], 1.0),
    }


def _add_masked_mse_total(
    totals: dict[str, float],
    metric_name: str,
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    total, count = masked_mse_sum_and_count(predicted, target, mask)
    totals[f"{metric_name}_sum"] = totals.get(f"{metric_name}_sum", 0.0) + float(total.detach().cpu())
    totals[f"{metric_name}_count"] = totals.get(f"{metric_name}_count", 0.0) + float(count.detach().cpu())


def _masked_mse_per_sample(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_mask = mask.to(device=predicted.device, dtype=predicted.dtype)
    while sample_mask.ndim < predicted.ndim:
        sample_mask = sample_mask.unsqueeze(-1)
    squared = (predicted - target).square() * sample_mask
    per_sample_sum = squared.flatten(1).sum(dim=1)
    per_sample_count = sample_mask.expand_as(predicted).flatten(1).sum(dim=1)
    return per_sample_sum / per_sample_count.clamp_min(1.0), per_sample_count > 0


def _finalize_masked_mse_totals(totals: dict[str, float]) -> dict[str, float]:
    metrics = {}
    for key, total in totals.items():
        if not key.endswith("_sum"):
            continue
        metric_name = key.removesuffix("_sum")
        metrics[metric_name] = total / max(totals.get(f"{metric_name}_count", 0.0), 1.0)
    return metrics


def _empty_future_ranking_totals() -> dict[str, Any]:
    return {
        "candidate_score_mse": {candidate: {"sum": 0.0, "count": 0.0} for candidate in _FUTURE_RANKING_CANDIDATES},
        "real_rank_sum": 0.0,
        "rank_correct_count": 0.0,
        "rank_sample_count": 0.0,
        "gap_sum": 0.0,
    }


def _accumulate_future_ranking_batch(
    totals: dict[str, Any],
    candidate_predictions: dict[str, torch.Tensor],
    target_action: torch.Tensor,
    action_mask: torch.Tensor,
) -> None:
    per_sample_errors = []
    valid_samples = None
    for candidate in _FUTURE_RANKING_CANDIDATES:
        prediction = candidate_predictions[candidate]
        mse_sum, mse_count = masked_mse_sum_and_count(prediction, target_action, action_mask)
        candidate_totals = totals["candidate_score_mse"][candidate]
        candidate_totals["sum"] += float(mse_sum.detach().cpu())
        candidate_totals["count"] += float(mse_count.detach().cpu())

        sample_mse, sample_valid = _masked_mse_per_sample(prediction, target_action, action_mask)
        per_sample_errors.append(sample_mse)
        valid_samples = sample_valid if valid_samples is None else valid_samples & sample_valid

    assert valid_samples is not None
    if not bool(valid_samples.any().detach().cpu()):
        return

    errors = torch.stack(per_sample_errors, dim=1)
    real_error = errors[:, 0]
    negative_errors = errors[:, 1:]
    real_rank = (negative_errors < real_error.unsqueeze(1)).sum(dim=1).to(dtype=torch.float32) + 1.0
    best_negative_error = negative_errors.min(dim=1).values
    valid_real_rank = real_rank[valid_samples]

    totals["real_rank_sum"] += float(valid_real_rank.sum().detach().cpu())
    totals["rank_correct_count"] += float((valid_real_rank == 1.0).to(dtype=torch.float32).sum().detach().cpu())
    totals["rank_sample_count"] += float(valid_samples.to(dtype=torch.float32).sum().detach().cpu())
    totals["gap_sum"] += float((best_negative_error[valid_samples] - real_error[valid_samples]).sum().detach().cpu())


def _finalize_future_ranking_metrics(
    totals: dict[str, Any],
    *,
    score_mode: IdmFutureRankingScoreMode = "teacher_forced_endpoint",
) -> dict[str, Any]:
    candidate_mse = {
        candidate: candidate_totals["sum"] / max(candidate_totals["count"], 1.0)
        for candidate, candidate_totals in totals["candidate_score_mse"].items()
    }
    real_mse = candidate_mse["real_gt"]
    real_candidate_rank = 1 + sum(
        candidate_mse[candidate] < real_mse for candidate in _FUTURE_RANKING_CANDIDATES if candidate != "real_gt"
    )
    ranked_sample_count = max(totals["rank_sample_count"], 1.0)
    # The teacher-forced endpoint scorer reads the velocity field at a fixed time; the sampled-action
    # scorer runs the full sampler, so it has no single time_value. Keep the legacy candidate-MSE key
    # for the default mode and tag every result with the score mode.
    candidate_mse_key = (
        "candidate_sampled_action_mse" if score_mode == "sampled_action" else "candidate_teacher_forced_endpoint_mse"
    )
    return {
        "score_mode": score_mode,
        "time_value": None if score_mode == "sampled_action" else _FUTURE_RANKING_TIME_VALUE,
        "candidate_order": list(_FUTURE_RANKING_CANDIDATES),
        candidate_mse_key: candidate_mse,
        "real_candidate_rank": int(real_candidate_rank),
        "mean_real_candidate_rank": totals["real_rank_sum"] / ranked_sample_count,
        "rank_accuracy": totals["rank_correct_count"] / ranked_sample_count,
        "real_vs_best_negative_gap": totals["gap_sum"] / ranked_sample_count,
        "num_ranked_samples": int(totals["rank_sample_count"]),
    }


def _flow_teacher_forced_probe(
    idm: torch.nn.Module,
    current_images: torch.Tensor,
    future_images: torch.Tensor,
    state: torch.Tensor,
    target_action: torch.Tensor,
    noise: torch.Tensor,
    *,
    time_value: float,
    wan_vae_latents: torch.Tensor | None = None,
    prev_state_history: torch.Tensor | None = None,
    prev_action_history: torch.Tensor | None = None,
    history_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate the flow velocity field at a fixed teacher-forced time."""
    module = unwrap_model(idm)
    context, visual_context_tokens = _flow_transition_context_and_visual_tokens(
        module,
        current_images,
        future_images,
        state,
        wan_vae_latents=wan_vae_latents,
    )
    history_tokens = None
    if prev_state_history is not None or prev_action_history is not None or history_mask is not None:
        history_tokens = module._history_tokens(prev_state_history, prev_action_history, history_mask)
    time = torch.full(
        (target_action.shape[0],),
        time_value,
        device=target_action.device,
        dtype=target_action.dtype,
    )
    time_view = time.view(-1, 1, 1)
    noisy_action = (1.0 - time_view) * noise + time_view * target_action
    target_velocity = target_action - noise
    flow_head_kwargs = {}
    if visual_context_tokens is not None:
        flow_head_kwargs["visual_context_tokens"] = visual_context_tokens
    if history_tokens is not None:
        flow_head_kwargs["history_tokens"] = history_tokens
    predicted_velocity = module.flow_head(context, noisy_action, time, **flow_head_kwargs)
    endpoint_prediction = noisy_action + (1.0 - time_view) * predicted_velocity
    return {
        "predicted_velocity": predicted_velocity,
        "target_velocity": target_velocity,
        "endpoint_prediction": endpoint_prediction,
    }


def future_sensitivity_gate(
    idm_mse: float,
    future_sensitivity: dict[str, dict[str, float]],
    *,
    output_delta_mse_min: float,
    degradation_min: float,
) -> dict[str, Any]:
    """Flag likely future-blind / collapsed IDMs from the ``current_repeated`` probe.

    The ``current_repeated`` variant replaces the real future with the current
    frame repeated, i.e. a "nothing happened" future. An IDM that genuinely
    reads dynamics should react to that swap. Two independent signals trip the
    gate (either one is enough):

    * ``output_delta_mse`` -- how far the action output moves under the swap.
      ~0 means the model emits the same actions regardless of the future, so it
      ignores the future input entirely.
    * degradation -- how much worse the actions fit the *real* targets under the
      swap (``current_repeated`` ``target_mse`` minus the real ``idm_mse``). ~0
      or negative means a static future is as good as the real one, so the
      future is not being used.

    Returns a report dict (always safe to embed in metrics); callers decide
    whether ``future_blind`` should fail the run.
    """
    current_repeated = future_sensitivity["current_repeated"]
    output_delta_mse = float(current_repeated["output_delta_mse"])
    degradation = float(current_repeated["target_mse"]) - float(idm_mse)
    output_delta_collapsed = output_delta_mse < output_delta_mse_min
    degradation_collapsed = degradation < degradation_min

    reasons: list[str] = []
    if output_delta_collapsed:
        reasons.append(
            f"current_repeated output_delta_mse={output_delta_mse:.3e} < {output_delta_mse_min:.3e}: "
            "action output barely changes when the real future is replaced by the repeated current frame."
        )
    if degradation_collapsed:
        reasons.append(
            f"real-vs-current_repeated degradation={degradation:.3e} < {degradation_min:.3e}: "
            "a static (current-frame) future fits the targets ~as well as the real future, "
            "so the future is barely used."
        )

    return {
        "current_repeated_output_delta_mse": output_delta_mse,
        "real_vs_current_repeated_degradation": degradation,
        "output_delta_mse_min": float(output_delta_mse_min),
        "degradation_min": float(degradation_min),
        "output_delta_mse_collapsed": output_delta_collapsed,
        "degradation_collapsed": degradation_collapsed,
        "future_blind": output_delta_collapsed or degradation_collapsed,
        "reasons": reasons,
    }


@torch.no_grad()
def main(args: Args) -> None:
    if args.cached_future_dir is not None and args.wan_vae_latent_cache_dir is not None:
        raise ValueError("--wan-vae-latent-cache-dir uses real dataset futures and cannot be combined with cached futures.")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    idm, model_config = load_idm_checkpoint(
        args.checkpoint,
        device,
        use_cached_wan_vae_latents=args.wan_vae_latent_cache_dir is not None,
    )
    enforce_idm_frame_delta_contract(args.checkpoint, args.frame_delta)
    action_normalizer = get_action_normalizer(idm, device)
    state_normalizer = get_state_normalizer(idm, device)
    idm.eval()
    effective_flow_num_samples = resolve_flow_num_samples(idm, args.flow_num_samples)
    effective_flow_noise_scale = resolve_flow_sample_noise_scale(idm, args.flow_noise_scale)

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
    if args.wan_vae_latent_cache_dir is not None:
        if model_config.idm_visual_encoder != "wan_vae":
            raise ValueError("--wan-vae-latent-cache-dir requires an IDM checkpoint with idm_visual_encoder='wan_vae'.")
        dataset = CachedWanVaeLatentDataset(dataset, args.wan_vae_latent_cache_dir, model_config=model_config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    flow_generator = None
    if idm_uses_flow_matching(idm) and args.flow_eval_seed is not None:
        flow_generator = torch.Generator(device=device).manual_seed(args.flow_eval_seed)
    flow_probe_generator = None
    if idm_uses_flow_matching(idm):
        flow_probe_seed = args.flow_eval_seed if args.flow_eval_seed is not None else 0
        flow_probe_generator = torch.Generator(device=device).manual_seed(flow_probe_seed)

    target_chunks = []
    predicted_chunks = []
    masks = []
    totals = {
        "idm_mse_sum": 0.0,
        "idm_mse_count": 0.0,
        "idm_smooth_l1_sum": 0.0,
        "idm_smooth_l1_count": 0.0,
    }
    sensitivity_totals = {
        "zero": _empty_sensitivity_totals(),
        "current_repeated": _empty_sensitivity_totals(),
        "shuffled": _empty_sensitivity_totals(),
        "noise": _empty_sensitivity_totals(),
    }
    state_sensitivity_totals = {
        "zero": _empty_sensitivity_totals(),
    }
    flow_teacher_totals: dict[str, float] = {}
    flow_teacher_sensitivity_totals = {
        "zero": {},
        "current_repeated": {},
        "shuffled": {},
        "noise": {},
    }
    flow_teacher_state_sensitivity_totals = {
        "zero": {},
    }
    flow_future_ranking_totals = _empty_future_ranking_totals()
    first_target = None
    first_predicted = None
    for batch in loader:
        device_batch = to_device(batch, device)
        idm_state = normalize_state_for_idm(idm, device_batch["state"], state_normalizer)
        flow_probe_state = (
            device_batch["state"] if state_normalizer is None else state_normalizer.normalize(device_batch["state"])
        )
        zero_raw_state = torch.zeros_like(device_batch["state"])
        state_variants = {
            "zero": normalize_state_for_idm(idm, zero_raw_state, state_normalizer),
        }
        flow_probe_state_variants = {
            "zero": zero_raw_state if state_normalizer is None else state_normalizer.normalize(zero_raw_state),
        }
        sample_noise = None
        if idm_uses_flow_matching(idm):
            with _flow_sampling_context(idm, args.flow_num_samples, args.flow_noise_scale):
                sample_noise = create_flow_sample_noise(
                    idm,
                    batch_size=int(device_batch["current_images"].shape[0]),
                    device=device,
                    dtype=device_batch["current_images"].dtype,
                    generator=flow_generator,
                )
        wan_vae_kwargs = _wan_vae_kwargs(device_batch)
        history_kwargs = idm_history_kwargs(
            device_batch,
            idm=idm,
            action_normalizer=action_normalizer,
            state_normalizer=state_normalizer,
        )
        with _flow_sampling_context(idm, args.flow_num_samples, args.flow_noise_scale):
            model_action = idm(
                device_batch["current_images"],
                device_batch["future_images"],
                idm_state,
                device_batch["task_id"],
                sample_noise=sample_noise,
                **wan_vae_kwargs,
                **history_kwargs,
            )
        predicted = model_action if action_normalizer is None else action_normalizer.denormalize(model_action)
        flow_probes = None
        flow_future_ranking_candidates = None
        if idm_uses_flow_matching(idm):
            assert flow_probe_generator is not None
            target_model_action = (
                device_batch["action_chunk"]
                if action_normalizer is None
                else action_normalizer.normalize(device_batch["action_chunk"])
            )
            probe_noise = torch.randn(
                target_model_action.shape,
                device=target_model_action.device,
                dtype=target_model_action.dtype,
                generator=flow_probe_generator,
            )

            def _ranking_prediction(future_images, candidate_wan_vae_latents, teacher_forced_endpoint):
                # The future-ranking score mode chooses how each candidate future is scored: reuse the
                # already-computed teacher-forced endpoint, or run the deterministic flow sampler.
                if args.future_usage_score_mode == "sampled_action":
                    return _flow_sampled_action_prediction(
                        idm,
                        device_batch["current_images"],
                        future_images,
                        flow_probe_state,
                        target_model_action,
                        wan_vae_latents=candidate_wan_vae_latents,
                        **history_kwargs,
                    )
                return teacher_forced_endpoint

            flow_probes = {
                "t0": _flow_teacher_forced_probe(
                    idm,
                    device_batch["current_images"],
                    device_batch["future_images"],
                    flow_probe_state,
                    target_model_action,
                    probe_noise,
                    time_value=0.0,
                    **wan_vae_kwargs,
                    **history_kwargs,
                ),
                "t0_5": _flow_teacher_forced_probe(
                    idm,
                    device_batch["current_images"],
                    device_batch["future_images"],
                    flow_probe_state,
                    target_model_action,
                    probe_noise,
                    time_value=_FUTURE_RANKING_TIME_VALUE,
                    **wan_vae_kwargs,
                    **history_kwargs,
                ),
            }
            flow_future_ranking_candidates = {
                "real_gt": _ranking_prediction(
                    device_batch["future_images"],
                    wan_vae_kwargs.get("wan_vae_latents"),
                    flow_probes["t0_5"]["endpoint_prediction"],
                )
            }
            for probe_name, probe in flow_probes.items():
                endpoint = probe["endpoint_prediction"]
                predicted_velocity = probe["predicted_velocity"]
                target_velocity = probe["target_velocity"]
                denormalized_endpoint = (
                    endpoint if action_normalizer is None else action_normalizer.denormalize(endpoint)
                )
                _add_masked_mse_total(
                    flow_teacher_totals,
                    f"normalized_{probe_name}_endpoint_mse",
                    endpoint,
                    target_model_action,
                    device_batch["action_mask"],
                )
                _add_masked_mse_total(
                    flow_teacher_totals,
                    f"normalized_{probe_name}_velocity_mse",
                    predicted_velocity,
                    target_velocity,
                    device_batch["action_mask"],
                )
                _add_masked_mse_total(
                    flow_teacher_totals,
                    f"denormalized_{probe_name}_endpoint_mse",
                    denormalized_endpoint,
                    device_batch["action_chunk"],
                    device_batch["action_mask"],
                )
                _add_masked_mse_total(
                    flow_teacher_totals,
                    f"normalized_sampled_vs_{probe_name}_endpoint_mse",
                    model_action,
                    endpoint,
                    device_batch["action_mask"],
                )
                _add_masked_mse_total(
                    flow_teacher_totals,
                    f"denormalized_sampled_vs_{probe_name}_endpoint_mse",
                    predicted,
                    denormalized_endpoint,
                    device_batch["action_mask"],
                )
        mse_sum, mse_count = masked_mse_sum_and_count(
            predicted,
            device_batch["action_chunk"],
            device_batch["action_mask"],
        )
        smooth_l1_sum, smooth_l1_count = masked_smooth_l1_sum_and_count(
            predicted,
            device_batch["action_chunk"],
            device_batch["action_mask"],
        )
        totals["idm_mse_sum"] += float(mse_sum.detach().cpu())
        totals["idm_mse_count"] += float(mse_count.detach().cpu())
        totals["idm_smooth_l1_sum"] += float(smooth_l1_sum.detach().cpu())
        totals["idm_smooth_l1_count"] += float(smooth_l1_count.detach().cpu())
        for variant_name, variant_state in state_variants.items():
            with _flow_sampling_context(idm, args.flow_num_samples, args.flow_noise_scale):
                variant_model_action = idm(
                    device_batch["current_images"],
                    device_batch["future_images"],
                    variant_state,
                    device_batch["task_id"],
                    sample_noise=sample_noise,
                    **wan_vae_kwargs,
                    **history_kwargs,
                )
            variant_predicted = (
                variant_model_action
                if action_normalizer is None
                else action_normalizer.denormalize(variant_model_action)
            )
            variant_mse_sum, variant_mse_count = masked_mse_sum_and_count(
                variant_predicted,
                device_batch["action_chunk"],
                device_batch["action_mask"],
            )
            variant_smooth_l1_sum, variant_smooth_l1_count = masked_smooth_l1_sum_and_count(
                variant_predicted,
                device_batch["action_chunk"],
                device_batch["action_mask"],
            )
            delta_mse_sum, delta_mse_count = masked_mse_sum_and_count(
                variant_predicted,
                predicted,
                device_batch["action_mask"],
            )
            state_sensitivity_totals[variant_name]["target_mse_sum"] += float(variant_mse_sum.detach().cpu())
            state_sensitivity_totals[variant_name]["target_mse_count"] += float(variant_mse_count.detach().cpu())
            state_sensitivity_totals[variant_name]["target_smooth_l1_sum"] += float(
                variant_smooth_l1_sum.detach().cpu()
            )
            state_sensitivity_totals[variant_name]["target_smooth_l1_count"] += float(
                variant_smooth_l1_count.detach().cpu()
            )
            state_sensitivity_totals[variant_name]["output_delta_mse_sum"] += float(delta_mse_sum.detach().cpu())
            state_sensitivity_totals[variant_name]["output_delta_mse_count"] += float(delta_mse_count.detach().cpu())
            if flow_probes is not None:
                variant_flow_probe_state = flow_probe_state_variants[variant_name]
                variant_t0_probe = _flow_teacher_forced_probe(
                    idm,
                    device_batch["current_images"],
                    device_batch["future_images"],
                    variant_flow_probe_state,
                    target_model_action,
                    probe_noise,
                    time_value=0.0,
                    **wan_vae_kwargs,
                    **history_kwargs,
                )
                variant_t0_5_probe = _flow_teacher_forced_probe(
                    idm,
                    device_batch["current_images"],
                    device_batch["future_images"],
                    variant_flow_probe_state,
                    target_model_action,
                    probe_noise,
                    time_value=_FUTURE_RANKING_TIME_VALUE,
                    **wan_vae_kwargs,
                    **history_kwargs,
                )
                for probe_name, variant_probe in (("t0", variant_t0_probe), ("t0_5", variant_t0_5_probe)):
                    denormalized_endpoint = (
                        variant_probe["endpoint_prediction"]
                        if action_normalizer is None
                        else action_normalizer.denormalize(variant_probe["endpoint_prediction"])
                    )
                    _add_masked_mse_total(
                        flow_teacher_state_sensitivity_totals[variant_name],
                        f"normalized_{probe_name}_endpoint_target_mse",
                        variant_probe["endpoint_prediction"],
                        target_model_action,
                        device_batch["action_mask"],
                    )
                    _add_masked_mse_total(
                        flow_teacher_state_sensitivity_totals[variant_name],
                        f"denormalized_{probe_name}_endpoint_target_mse",
                        denormalized_endpoint,
                        device_batch["action_chunk"],
                        device_batch["action_mask"],
                    )
                    _add_masked_mse_total(
                        flow_teacher_state_sensitivity_totals[variant_name],
                        f"normalized_{probe_name}_endpoint_output_delta_mse",
                        variant_probe["endpoint_prediction"],
                        flow_probes[probe_name]["endpoint_prediction"],
                        device_batch["action_mask"],
                    )
                    _add_masked_mse_total(
                        flow_teacher_state_sensitivity_totals[variant_name],
                        f"normalized_{probe_name}_velocity_output_delta_mse",
                        variant_probe["predicted_velocity"],
                        flow_probes[probe_name]["predicted_velocity"],
                        device_batch["action_mask"],
                    )
        for variant_name, variant_batch in _future_variants(device_batch).items():
            variant_wan_vae_kwargs = _wan_vae_kwargs(variant_batch)
            with _flow_sampling_context(idm, args.flow_num_samples, args.flow_noise_scale):
                variant_model_action = idm(
                    device_batch["current_images"],
                    variant_batch["future_images"],
                    idm_state,
                    device_batch["task_id"],
                    sample_noise=sample_noise,
                    **variant_wan_vae_kwargs,
                    **history_kwargs,
                )
            variant_predicted = (
                variant_model_action
                if action_normalizer is None
                else action_normalizer.denormalize(variant_model_action)
            )
            variant_mse_sum, variant_mse_count = masked_mse_sum_and_count(
                variant_predicted,
                device_batch["action_chunk"],
                device_batch["action_mask"],
            )
            variant_smooth_l1_sum, variant_smooth_l1_count = masked_smooth_l1_sum_and_count(
                variant_predicted,
                device_batch["action_chunk"],
                device_batch["action_mask"],
            )
            delta_mse_sum, delta_mse_count = masked_mse_sum_and_count(
                variant_predicted,
                predicted,
                device_batch["action_mask"],
            )
            sensitivity_totals[variant_name]["target_mse_sum"] += float(variant_mse_sum.detach().cpu())
            sensitivity_totals[variant_name]["target_mse_count"] += float(variant_mse_count.detach().cpu())
            sensitivity_totals[variant_name]["target_smooth_l1_sum"] += float(variant_smooth_l1_sum.detach().cpu())
            sensitivity_totals[variant_name]["target_smooth_l1_count"] += float(
                variant_smooth_l1_count.detach().cpu()
            )
            sensitivity_totals[variant_name]["output_delta_mse_sum"] += float(delta_mse_sum.detach().cpu())
            sensitivity_totals[variant_name]["output_delta_mse_count"] += float(delta_mse_count.detach().cpu())
            if flow_probes is not None:
                variant_t0_probe = _flow_teacher_forced_probe(
                    idm,
                    device_batch["current_images"],
                    variant_batch["future_images"],
                    flow_probe_state,
                    target_model_action,
                    probe_noise,
                    time_value=0.0,
                    **variant_wan_vae_kwargs,
                    **history_kwargs,
                )
                variant_t0_5_probe = _flow_teacher_forced_probe(
                    idm,
                    device_batch["current_images"],
                    variant_batch["future_images"],
                    flow_probe_state,
                    target_model_action,
                    probe_noise,
                    time_value=_FUTURE_RANKING_TIME_VALUE,
                    **variant_wan_vae_kwargs,
                    **history_kwargs,
                )
                for probe_name, variant_probe in (("t0", variant_t0_probe), ("t0_5", variant_t0_5_probe)):
                    _add_masked_mse_total(
                        flow_teacher_sensitivity_totals[variant_name],
                        f"normalized_{probe_name}_endpoint_output_delta_mse",
                        variant_probe["endpoint_prediction"],
                        flow_probes[probe_name]["endpoint_prediction"],
                        device_batch["action_mask"],
                    )
                    _add_masked_mse_total(
                        flow_teacher_sensitivity_totals[variant_name],
                        f"normalized_{probe_name}_velocity_output_delta_mse",
                        variant_probe["predicted_velocity"],
                        flow_probes[probe_name]["predicted_velocity"],
                        device_batch["action_mask"],
                    )
                assert flow_future_ranking_candidates is not None
                flow_future_ranking_candidates[variant_name] = _ranking_prediction(
                    variant_batch["future_images"],
                    variant_wan_vae_kwargs.get("wan_vae_latents"),
                    variant_t0_5_probe["endpoint_prediction"],
                )
        if flow_future_ranking_candidates is not None:
            _accumulate_future_ranking_batch(
                flow_future_ranking_totals,
                flow_future_ranking_candidates,
                target_model_action,
                device_batch["action_mask"],
            )
        target_chunks.append(batch["action_chunk"].detach().cpu())
        predicted_chunks.append(predicted.detach().cpu())
        masks.append(batch["action_mask"].detach().cpu())
        if first_target is None:
            first_target = batch["action_chunk"][0].detach().cpu()
            first_predicted = predicted[0].detach().cpu()

    target = torch.cat(target_chunks, dim=0)
    predicted = torch.cat(predicted_chunks, dim=0)
    mask = torch.cat(masks, dim=0)
    stepwise = stepwise_action_diagnostics(predicted, target, mask)
    target_values = _masked_action_values(target, mask)
    predicted_values = _masked_action_values(predicted, mask)
    error = predicted_values - target_values
    mean_action = action_normalizer.mean.detach().cpu() if action_normalizer is not None else target_values.mean(dim=0)
    mean_prediction = mean_action.view(1, 1, -1).expand_as(target)
    mean_baseline_mse = masked_mse(mean_prediction, target, mask)
    mean_baseline_smooth_l1 = masked_smooth_l1(mean_prediction, target, mask)

    per_dim_mse = F.mse_loss(predicted_values, target_values, reduction="none").mean(dim=0)
    per_dim_mae = error.abs().mean(dim=0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "action_trace.png"
    histogram_path = output_dir / "action_histograms.png"
    assert first_target is not None
    assert first_predicted is not None
    _write_trace_plot(first_target, first_predicted, trace_path)
    _write_histogram_plot(target_values, predicted_values, histogram_path)

    idm_mse = totals["idm_mse_sum"] / max(totals["idm_mse_count"], 1.0)
    future_sensitivity = {
        variant_name: _finalize_sensitivity_metrics(variant_totals)
        for variant_name, variant_totals in sensitivity_totals.items()
    }
    state_sensitivity = {
        variant_name: _finalize_sensitivity_metrics(variant_totals)
        for variant_name, variant_totals in state_sensitivity_totals.items()
    }
    flow_teacher_forced = None
    if idm_uses_flow_matching(idm):
        flow_teacher_forced = _finalize_masked_mse_totals(flow_teacher_totals)
        if "normalized_t0_5_velocity_mse" in flow_teacher_forced:
            flow_teacher_forced["normalized_velocity_mse"] = flow_teacher_forced["normalized_t0_5_velocity_mse"]
        if "denormalized_t0_5_endpoint_mse" in flow_teacher_forced:
            flow_teacher_forced["denormalized_endpoint_mse"] = flow_teacher_forced[
                "denormalized_t0_5_endpoint_mse"
            ]
        if "denormalized_sampled_vs_t0_5_endpoint_mse" in flow_teacher_forced:
            flow_teacher_forced["sampled_vs_endpoint_mse"] = flow_teacher_forced[
                "denormalized_sampled_vs_t0_5_endpoint_mse"
            ]
        flow_teacher_forced["future_sensitivity"] = {
            variant_name: _finalize_masked_mse_totals(variant_totals)
            for variant_name, variant_totals in flow_teacher_sensitivity_totals.items()
        }
        flow_teacher_forced["state_sensitivity"] = {
            variant_name: _finalize_masked_mse_totals(variant_totals)
            for variant_name, variant_totals in flow_teacher_state_sensitivity_totals.items()
        }
        flow_teacher_forced["future_ranking"] = _finalize_future_ranking_metrics(
            flow_future_ranking_totals,
            score_mode=args.future_usage_score_mode,
        )
    gate = future_sensitivity_gate(
        idm_mse,
        future_sensitivity,
        output_delta_mse_min=args.future_blind_output_delta_mse_min,
        degradation_min=args.future_blind_degradation_min,
    )

    metrics = {
        "dataset_config": dataclasses.asdict(dataset_config),
        "model_config": dataclasses.asdict(model_config),
        "checkpoint": args.checkpoint,
        "cached_future_dir": args.cached_future_dir,
        "wan_vae_latent_cache_dir": args.wan_vae_latent_cache_dir,
        "flow_eval_seed": args.flow_eval_seed if idm_uses_flow_matching(idm) else None,
        "flow_num_samples": effective_flow_num_samples,
        "flow_noise_scale": effective_flow_noise_scale,
        "action_normalizer": action_normalizer.to_dict() if action_normalizer is not None else None,
        "state_normalizer": state_normalizer.to_dict() if state_normalizer is not None else None,
        "num_samples": len(dataset),
        "num_valid_actions": int(target_values.shape[0]),
        "idm_mse": idm_mse,
        "idm_smooth_l1": totals["idm_smooth_l1_sum"] / max(totals["idm_smooth_l1_count"], 1.0),
        "mean_action_baseline": {
            "idm_mse": float(mean_baseline_mse.item()),
            "idm_smooth_l1": float(mean_baseline_smooth_l1.item()),
            "mean_action": [float(value.item()) for value in mean_action],
        },
        "future_sensitivity": future_sensitivity,
        "state_sensitivity": state_sensitivity,
        "future_sensitivity_gate": gate,
        **({"flow_teacher_forced": flow_teacher_forced} if flow_teacher_forced is not None else {}),
        "target_stats": _stats(target_values),
        "predicted_stats": _stats(predicted_values),
        "error_stats": {
            "mean": float(error.mean().item()),
            "std": float(error.std(unbiased=False).item()),
            "mae": float(error.abs().mean().item()),
            "max_abs": float(error.abs().max().item()),
        },
        "per_action_dim_mse": [float(value.item()) for value in per_dim_mse],
        "per_action_dim_mae": [float(value.item()) for value in per_dim_mae],
        **stepwise,
        "action_trace": str(trace_path),
        "action_histograms": str(histogram_path),
    }
    (output_dir / "idm_diagnostics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, sort_keys=True))

    if args.fail_on_future_blind and gate["future_blind"]:
        raise SystemExit(
            "Future-sensitivity gate failed: this IDM looks future-blind / collapsed.\n  "
            + "\n  ".join(gate["reasons"])
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
