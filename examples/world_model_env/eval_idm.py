from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import tyro
from torch.utils.data import DataLoader

from world_model.config import DatasetConfig, DatasetSource, IdmFutureRankingScoreMode
from world_model.data import CachedWanVaeLatentDataset, GeneratedWanLatentDataset
from world_model.train_lib import (
    ActionNormalizer,
    IdmPredictionMode,
    create_dataset_with_optional_cache,
    enforce_idm_frame_delta_contract,
    evaluate_idm,
    evaluate_idm_future_usage,
    get_action_normalizer,
    idm_uses_flow_matching,
    load_idm_checkpoint,
    masked_mse_sum_and_count,
    masked_smooth_l1_sum_and_count,
    resolve_device,
    resolve_flow_num_samples,
    resolve_flow_sample_noise_scale,
)


@dataclasses.dataclass
class Args:
    checkpoint: str
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_keys: tuple[str, ...] = ("corner4.image",)
    output_dir: str = "output/idm_eval"
    cached_future_dir: str | None = None
    wan_vae_latent_cache_dir: str | None = None
    generated_wan_latent_cache_dir: str | None = None
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
    prediction_mode: IdmPredictionMode = "sample"
    future_usage_eval: bool = False
    future_usage_score_mode: IdmFutureRankingScoreMode = "teacher_forced_endpoint"
    seed: int = 7


def _empirical_mean_action(loader: Iterable[dict[str, torch.Tensor]], device: torch.device) -> torch.Tensor:
    """Per-dimension mean over the loader's valid (unmasked) actions."""
    total: torch.Tensor | None = None
    count = 0
    for batch in loader:
        action = batch["action_chunk"].to(device=device, dtype=torch.float32)
        mask = batch["action_mask"].to(device=device, dtype=torch.bool)
        values = action[mask]
        if values.numel() == 0:
            continue
        batch_total = values.sum(dim=0)
        total = batch_total if total is None else total + batch_total
        count += int(values.shape[0])
    if total is None or count == 0:
        raise ValueError("Cannot compute a mean-action baseline from an empty action set.")
    return total / count


def compute_mean_action_baseline(
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    action_normalizer: ActionNormalizer | None,
) -> dict[str, float | list[float]]:
    """Score the trivial "always predict the mean action" baseline.

    Reports the action MSE / smooth-L1 a model would get if it ignored every
    input and emitted a constant mean action at every step. The constant is the
    normalizer's (training-set) mean when one is attached, else the dataset's
    own mean. A trained IDM that fails to clearly beat this baseline is not
    decoding dynamics -- reporting both side by side makes that hard to miss.
    """
    if action_normalizer is not None:
        mean_action = action_normalizer.mean.detach().to(device=device, dtype=torch.float32)
    else:
        mean_action = _empirical_mean_action(loader, device)

    mse_sum = mse_count = smooth_l1_sum = smooth_l1_count = 0.0
    for batch in loader:
        target = batch["action_chunk"].to(device=device, dtype=torch.float32)
        mask = batch["action_mask"].to(device=device)
        prediction = mean_action.view(1, 1, -1).expand_as(target)
        batch_mse_sum, batch_mse_count = masked_mse_sum_and_count(prediction, target, mask)
        batch_l1_sum, batch_l1_count = masked_smooth_l1_sum_and_count(prediction, target, mask)
        mse_sum += float(batch_mse_sum.detach().cpu())
        mse_count += float(batch_mse_count.detach().cpu())
        smooth_l1_sum += float(batch_l1_sum.detach().cpu())
        smooth_l1_count += float(batch_l1_count.detach().cpu())

    idm_mse = mse_sum / max(mse_count, 1.0)
    idm_smooth_l1 = smooth_l1_sum / max(smooth_l1_count, 1.0)
    return {
        "idm_mse": idm_mse,
        "idm_smooth_l1": idm_smooth_l1,
        "dataset_action_mse": idm_mse,
        "dataset_action_smooth_l1": idm_smooth_l1,
        "mean_action": [float(value) for value in mean_action.detach().cpu()],
    }


def _add_dataset_action_metric_aliases(metrics: dict[str, object]) -> None:
    metrics["dataset_action_mse"] = metrics["idm_mse"]
    metrics["dataset_action_smooth_l1"] = metrics["idm_smooth_l1"]


_FINGERPRINT_DATASET_CONFIG_KEYS = (
    "source",
    "repo_id",
    "image_keys",
    "state_key",
    "action_key",
    "task_key",
    "frame_delta",
    "action_horizon",
    "image_size",
    "max_samples",
    "samples_per_episode",
    "episodes",
    "seed",
)


def _json_normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _dataset_fingerprint(dataset_config: DatasetConfig | Mapping[str, Any]) -> dict[str, Any]:
    if dataclasses.is_dataclass(dataset_config):
        config = dataclasses.asdict(dataset_config)
    elif isinstance(dataset_config, Mapping):
        config = dict(dataset_config)
    else:
        raise TypeError(f"dataset_config must be a DatasetConfig or mapping, got {type(dataset_config).__name__}.")
    return {
        "dataset_config": {
            key: _json_normalized(config[key]) for key in _FINGERPRINT_DATASET_CONFIG_KEYS if key in config
        }
    }


def _build_sample_fingerprints(
    dataset_config: DatasetConfig | Mapping[str, Any],
    *,
    num_samples: int | None = None,
) -> dict[str, Any]:
    dataset_fingerprint = _dataset_fingerprint(dataset_config)
    sample_fingerprint: dict[str, Any] = {"dataset_fingerprint": dataset_fingerprint}
    if num_samples is not None:
        sample_fingerprint["num_samples"] = int(num_samples)
    return {
        "dataset_fingerprint": dataset_fingerprint,
        "sample_fingerprint": sample_fingerprint,
    }


def _validate_future_latent_cache_modes(args: Args) -> None:
    selected = [
        flag
        for flag, value in (
            ("--cached-future-dir", args.cached_future_dir),
            ("--wan-vae-latent-cache-dir", args.wan_vae_latent_cache_dir),
            ("--generated-wan-latent-cache-dir", args.generated_wan_latent_cache_dir),
        )
        if value is not None
    ]
    if len(selected) > 1:
        if (
            args.cached_future_dir is not None
            and args.wan_vae_latent_cache_dir is not None
            and args.generated_wan_latent_cache_dir is None
        ):
            raise ValueError(
                "--wan-vae-latent-cache-dir uses real dataset futures and cannot be combined with cached futures. "
                "Future/latent cache modes are mutually exclusive; set only one."
            )
        raise ValueError(
            "Future/latent cache modes are mutually exclusive; set only one of "
            "--cached-future-dir, --wan-vae-latent-cache-dir, or "
            f"--generated-wan-latent-cache-dir (got {', '.join(selected)})."
        )


def _load_generated_wan_latent_generator_metadata(cache_dir: str | Path) -> dict[str, object]:
    config_path = Path(cache_dir).expanduser() / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Generated Wan latent cache config not found: {config_path}")
    try:
        metadata = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Generated Wan latent cache config is invalid JSON: {config_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"Generated Wan latent cache config must be a JSON object: {config_path}")
    generator_metadata = metadata.get("generator")
    if not isinstance(generator_metadata, Mapping):
        raise ValueError(f"Generated Wan latent cache config must contain a generator JSON object: {config_path}")
    return dict(generator_metadata)


def main(args: Args) -> None:
    _validate_future_latent_cache_modes(args)
    generated_wan_latent_generator = (
        _load_generated_wan_latent_generator_metadata(args.generated_wan_latent_cache_dir)
        if args.generated_wan_latent_cache_dir is not None
        else None
    )
    device = resolve_device(args.device)
    idm, model_config = load_idm_checkpoint(
        args.checkpoint,
        device,
        use_cached_wan_vae_latents=(
            args.wan_vae_latent_cache_dir is not None or args.generated_wan_latent_cache_dir is not None
        ),
    )
    enforce_idm_frame_delta_contract(args.checkpoint, args.frame_delta)
    if args.generated_wan_latent_cache_dir is not None and model_config.idm_visual_encoder != "wan_vae":
        raise ValueError(
            "--generated-wan-latent-cache-dir requires an IDM checkpoint with idm_visual_encoder='wan_vae'."
        )
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
    elif args.generated_wan_latent_cache_dir is not None:
        dataset = GeneratedWanLatentDataset(
            dataset,
            args.generated_wan_latent_cache_dir,
            model_config,
            generator_metadata=generated_wan_latent_generator,
        )
    try:
        num_samples = len(dataset)
    except TypeError:
        num_samples = None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    effective_flow_num_samples = resolve_flow_num_samples(idm, args.flow_num_samples)
    effective_flow_noise_scale = resolve_flow_sample_noise_scale(idm, args.flow_noise_scale)
    metrics: dict[str, object] = dict(
        evaluate_idm(
            idm,
            loader,
            device,
            flow_eval_seed=args.flow_eval_seed,
            flow_num_samples=args.flow_num_samples,
            flow_noise_scale=args.flow_noise_scale,
            prediction_mode=args.prediction_mode,
        )
    )
    _add_dataset_action_metric_aliases(metrics)
    if args.future_usage_eval:
        metrics.update(
            evaluate_idm_future_usage(
                idm,
                loader,
                device,
                flow_eval_seed=args.flow_eval_seed,
                flow_num_samples=args.flow_num_samples,
                flow_noise_scale=args.flow_noise_scale,
                score_mode=args.future_usage_score_mode,
            )
        )
    metrics["mean_action_baseline"] = compute_mean_action_baseline(loader, device, get_action_normalizer(idm, device))
    metrics["checkpoint"] = args.checkpoint
    metrics["dataset_config"] = dataclasses.asdict(dataset_config)
    metrics.update(_build_sample_fingerprints(dataset_config, num_samples=num_samples))
    if num_samples is not None:
        metrics["num_samples"] = num_samples
    metrics["cached_future_dir"] = args.cached_future_dir
    metrics["wan_vae_latent_cache_dir"] = args.wan_vae_latent_cache_dir
    metrics["generated_wan_latent_cache_dir"] = args.generated_wan_latent_cache_dir
    metrics["generated_wan_latent_generator"] = generated_wan_latent_generator
    metrics["prediction_mode"] = args.prediction_mode
    metrics["flow_eval_seed"] = args.flow_eval_seed if idm_uses_flow_matching(idm) else None
    metrics["flow_num_samples"] = effective_flow_num_samples
    metrics["flow_noise_scale"] = effective_flow_noise_scale
    metrics["metric_family"] = "dataset_action_mse"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
