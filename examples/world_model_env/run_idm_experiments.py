from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import tyro
from torch.utils.data import DataLoader

from diagnose_idm import Args as DiagnoseArgs
from diagnose_idm import main as diagnose_main
from world_model.config import (
    FLOW_DIT_350M_FF_DIM,
    FLOW_DIT_350M_HEADS,
    FLOW_DIT_350M_LATENT_DIM,
    FLOW_DIT_350M_LAYERS,
    FLOW_DIT_350M_SAMPLING_STEPS,
    FLOW_DIT_DEFAULT_ENDPOINT_LOSS_WEIGHT,
    DatasetConfig,
    DatasetSource,
    IdmArchitecture,
    IdmFlowContextConditioning,
    IdmFlowVisualTokenConditioningMode,
    IdmFlowVisualTokenRepresentation,
    IdmFlowVisualTokenScope,
    IdmFutureConditioning,
    IdmFutureRankingScoreMode,
    IdmVisualEncoder,
    ModelConfig,
    TrainConfig,
)
from world_model.train_lib import (
    create_dataset_with_optional_cache,
    enforce_idm_frame_delta_contract,
    evaluate_idm,
    load_idm_checkpoint,
    resolve_device,
    run_idm_training,
)

CACHED_FUTURES_TRAINING_ERROR = (
    "Generated/cached futures are for eval/ranking only, not IDM training. "
    "run_idm_experiments.py trains IDMs only on ground-truth dataset futures; "
    "use eval_idm.py or diagnose_idm.py to evaluate a trained IDM on cached futures."
)


@dataclasses.dataclass
class Args:
    dataset_source: DatasetSource = "synthetic"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_key: str = "corner4.image"
    output_dir: str = "output/idm_experiments"
    cached_future_dir: str | None = None
    episodes: tuple[int, ...] | None = None
    max_samples: int | None = 64
    samples_per_episode: int | None = None
    synthetic_samples: int = 64
    image_size: int = 64
    frame_delta: int = 1
    num_future_frames: int = 4
    action_horizons: tuple[int, ...] = (4, 8)
    epochs: int = 5
    batch_size: int = 16
    learning_rates: tuple[float, ...] = (3e-4,)
    seeds: tuple[int, ...] = (7,)
    num_workers: int = 0
    device: str = "auto"
    normalize_actions: bool = True
    diagnostics: bool = False
    idm_arch: IdmArchitecture = "flow_transformer"
    idm_visual_encoder: IdmVisualEncoder = "patch"
    latent_dim: int = FLOW_DIT_350M_LATENT_DIM
    idm_transformer_layers: int = FLOW_DIT_350M_LAYERS
    idm_transformer_heads: int = FLOW_DIT_350M_HEADS
    idm_transformer_patch_size: int = 16
    idm_transformer_dropout: float = 0.1
    idm_transformer_ff_dim: int | None = FLOW_DIT_350M_FF_DIM
    idm_flow_sampling_steps: int = FLOW_DIT_350M_SAMPLING_STEPS
    idm_flow_num_samples: int = 1
    idm_flow_sample_noise_scale: float = 1.0
    idm_flow_time_scale: float = 1000.0
    idm_flow_endpoint_loss_weight: float = FLOW_DIT_DEFAULT_ENDPOINT_LOSS_WEIGHT
    idm_flow_endpoint_consistency_loss_weight: float = 0.0
    idm_flow_zero_start_endpoint_loss_weight: float = 0.0
    idm_flow_sampled_action_loss_weight: float = 0.0
    idm_flow_context_conditioning: IdmFlowContextConditioning = "token"
    idm_future_conditioning: IdmFutureConditioning = "full"
    idm_flow_visual_token_conditioning: bool = False
    idm_flow_visual_token_conditioning_mode: IdmFlowVisualTokenConditioningMode = "prefix"
    idm_flow_visual_token_scope: IdmFlowVisualTokenScope = "all"
    idm_flow_visual_token_representation: IdmFlowVisualTokenRepresentation = "encoded"
    idm_flow_train_time_min: float = 0.0
    idm_flow_train_time_max: float = 1.0
    idm_context_action_loss_weight: float = 0.0
    idm_context_action_warmup_epochs: int | None = None
    idm_future_contrastive_weight: float = 0.0
    idm_future_contrastive_margin: float = 0.1
    idm_future_ranking_weight: float = 0.0
    idm_future_ranking_start_epoch: int | None = None
    idm_future_ranking_ramp_epochs: int = 0
    idm_future_ranking_temperature: float = 0.1
    idm_future_ranking_noise_std: float = 1.0
    idm_future_ranking_repeated_current_negative: bool = False
    idm_future_ranking_shuffled_future_negative: bool = False
    idm_future_ranking_noisy_future_negative: bool = False
    idm_future_ranking_zero_future_negative: bool = False
    idm_future_ranking_same_task_negative: bool = False
    idm_future_ranking_score_mode: IdmFutureRankingScoreMode = "teacher_forced_endpoint"
    idm_future_usage_eval: bool = False
    idm_future_usage_rank_accuracy_min: float = 0.55
    idm_future_usage_gap_min: float = 0.0
    idm_future_usage_degradation_min: float = 1e-4
    idm_future_usage_output_delta_mse_min: float = 1e-4
    idm_future_usage_score_mode: IdmFutureRankingScoreMode = "teacher_forced_endpoint"
    idm_same_task_batching: bool = False
    idm_same_task_future_delta_weight: float = 0.0
    idm_same_task_future_delta_time_value: float = 0.5
    idm_same_task_future_delta_max_state_distance: float | None = None
    idm_same_task_future_delta_min_action_delta_mse: float = 0.0
    idm_current_frame_dropout: float = 0.0
    idm_wan_vae_current_latent_dropout: float = 0.0
    idm_wan_vae_latent_noise_prob: float = 0.0
    idm_wan_vae_latent_noise_s_min: float = 0.5
    idm_wan_vae_latent_noise_s_max: float = 1.0
    flow_eval_seed: int | None = 0
    wan_vae_repo_dir: str | None = None
    wan_vae_checkpoint_path: str | None = None
    wan_vae_dtype: str = "bfloat16"
    wan_vae_tiled: bool = False
    wan_vae_latent_channels: int = 48
    wan_vae_spatial_stride: int = 16


def run_full_dataset_eval(
    *,
    checkpoint: Path,
    dataset_config: DatasetConfig,
    cached_future_dir: str | None,
    batch_size: int,
    device: str,
    flow_eval_seed: int | None,
) -> dict[str, float]:
    resolved_device = resolve_device(device)
    idm, _ = load_idm_checkpoint(checkpoint, resolved_device)
    enforce_idm_frame_delta_contract(checkpoint, dataset_config.frame_delta)
    dataset = create_dataset_with_optional_cache(dataset_config, cached_future_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return evaluate_idm(idm, loader, resolved_device, flow_eval_seed=flow_eval_seed)


def run_optional_diagnostics(
    *,
    enabled: bool,
    checkpoint: Path,
    dataset_config: DatasetConfig,
    cached_future_dir: str | None,
    batch_size: int,
    device: str,
    flow_eval_seed: int | None,
    output_dir: Path,
) -> None:
    if not enabled:
        return
    diagnose_main(
        DiagnoseArgs(
            checkpoint=str(checkpoint),
            dataset_source=dataset_config.source,
            repo_id=dataset_config.repo_id,
            image_keys=dataset_config.image_keys,
            output_dir=str(output_dir),
            cached_future_dir=cached_future_dir,
            image_size=dataset_config.image_size,
            frame_delta=dataset_config.frame_delta,
            num_future_frames=dataset_config.num_future_frames,
            action_horizon=dataset_config.action_horizon,
            max_samples=dataset_config.max_samples,
            episodes=dataset_config.episodes,
            synthetic_samples=dataset_config.synthetic_samples,
            batch_size=batch_size,
            device=device,
            flow_eval_seed=flow_eval_seed,
            seed=dataset_config.seed,
        )
    )


def build_experiment_summary(args: Args, rows: list[dict]) -> dict:
    successful_rows = [row for row in rows if row.get("status") == "success"]
    return {
        "args": dataclasses.asdict(args),
        "num_runs": len(rows),
        "runs": rows,
        "best_by_full_eval": (
            min(successful_rows, key=lambda row: row["best_full_eval"]["idm_mse"]) if successful_rows else None
        ),
    }


def write_experiment_summary(output_dir: Path, args: Args, rows: list[dict]) -> None:
    summary = build_experiment_summary(args, rows)
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main(args: Args) -> None:
    if args.cached_future_dir is not None:
        raise ValueError(CACHED_FUTURES_TRAINING_ERROR)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for seed in args.seeds:
        for action_horizon in args.action_horizons:
            for learning_rate in args.learning_rates:
                run_name = f"idm_h{action_horizon}_lr{learning_rate:g}_seed{seed}"
                run_dir = output_dir / run_name
                dataset_config = DatasetConfig(
                    source=args.dataset_source,
                    repo_id=args.repo_id,
                    image_keys=(args.image_key,),
                    image_size=args.image_size,
                    frame_delta=args.frame_delta,
                    num_future_frames=args.num_future_frames,
                    action_horizon=action_horizon,
                    max_samples=args.max_samples,
                    samples_per_episode=args.samples_per_episode,
                    episodes=args.episodes,
                    synthetic_samples=args.synthetic_samples,
                    seed=seed,
                )
                train_config = TrainConfig(
                    dataset=dataset_config,
                    model=ModelConfig(
                        num_views=1,
                        image_size=args.image_size,
                        action_horizon=action_horizon,
                        num_future_frames=args.num_future_frames,
                        idm_arch=args.idm_arch,
                        idm_visual_encoder=args.idm_visual_encoder,
                        latent_dim=args.latent_dim,
                        idm_transformer_layers=args.idm_transformer_layers,
                        idm_transformer_heads=args.idm_transformer_heads,
                        idm_transformer_patch_size=args.idm_transformer_patch_size,
                        idm_transformer_dropout=args.idm_transformer_dropout,
                        idm_transformer_ff_dim=args.idm_transformer_ff_dim,
                        idm_flow_sampling_steps=args.idm_flow_sampling_steps,
                        idm_flow_num_samples=args.idm_flow_num_samples,
                        idm_flow_sample_noise_scale=args.idm_flow_sample_noise_scale,
                        idm_flow_time_scale=args.idm_flow_time_scale,
                        idm_flow_endpoint_loss_weight=args.idm_flow_endpoint_loss_weight,
                        idm_flow_endpoint_consistency_loss_weight=args.idm_flow_endpoint_consistency_loss_weight,
                        idm_flow_zero_start_endpoint_loss_weight=args.idm_flow_zero_start_endpoint_loss_weight,
                        idm_flow_sampled_action_loss_weight=args.idm_flow_sampled_action_loss_weight,
                        idm_flow_context_conditioning=args.idm_flow_context_conditioning,
                        idm_future_conditioning=args.idm_future_conditioning,
                        idm_flow_visual_token_conditioning=args.idm_flow_visual_token_conditioning,
                        idm_flow_visual_token_conditioning_mode=args.idm_flow_visual_token_conditioning_mode,
                        idm_flow_visual_token_scope=args.idm_flow_visual_token_scope,
                        idm_flow_visual_token_representation=args.idm_flow_visual_token_representation,
                        idm_flow_train_time_min=args.idm_flow_train_time_min,
                        idm_flow_train_time_max=args.idm_flow_train_time_max,
                        wan_vae_repo_dir=args.wan_vae_repo_dir,
                        wan_vae_checkpoint_path=args.wan_vae_checkpoint_path,
                        wan_vae_dtype=args.wan_vae_dtype,
                        wan_vae_tiled=args.wan_vae_tiled,
                        wan_vae_latent_channels=args.wan_vae_latent_channels,
                        wan_vae_spatial_stride=args.wan_vae_spatial_stride,
                    ),
                    output_dir=str(run_dir),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=learning_rate,
                    num_workers=args.num_workers,
                    device=args.device,
                    normalize_actions=args.normalize_actions,
                    idm_context_action_loss_weight=args.idm_context_action_loss_weight,
                    idm_future_contrastive_weight=args.idm_future_contrastive_weight,
                    idm_future_contrastive_margin=args.idm_future_contrastive_margin,
                    idm_future_ranking_weight=args.idm_future_ranking_weight,
                    idm_future_ranking_start_epoch=args.idm_future_ranking_start_epoch,
                    idm_future_ranking_ramp_epochs=args.idm_future_ranking_ramp_epochs,
                    idm_future_ranking_temperature=args.idm_future_ranking_temperature,
                    idm_future_ranking_noise_std=args.idm_future_ranking_noise_std,
                    idm_future_ranking_repeated_current_negative=args.idm_future_ranking_repeated_current_negative,
                    idm_future_ranking_shuffled_future_negative=args.idm_future_ranking_shuffled_future_negative,
                    idm_future_ranking_noisy_future_negative=args.idm_future_ranking_noisy_future_negative,
                    idm_future_ranking_zero_future_negative=args.idm_future_ranking_zero_future_negative,
                    idm_future_ranking_same_task_negative=args.idm_future_ranking_same_task_negative,
                    idm_future_ranking_score_mode=args.idm_future_ranking_score_mode,
                    idm_future_usage_eval=args.idm_future_usage_eval,
                    idm_future_usage_rank_accuracy_min=args.idm_future_usage_rank_accuracy_min,
                    idm_future_usage_gap_min=args.idm_future_usage_gap_min,
                    idm_future_usage_degradation_min=args.idm_future_usage_degradation_min,
                    idm_future_usage_output_delta_mse_min=args.idm_future_usage_output_delta_mse_min,
                    idm_future_usage_score_mode=args.idm_future_usage_score_mode,
                    idm_same_task_batching=args.idm_same_task_batching,
                    idm_same_task_future_delta_weight=args.idm_same_task_future_delta_weight,
                    idm_same_task_future_delta_time_value=args.idm_same_task_future_delta_time_value,
                    idm_same_task_future_delta_max_state_distance=args.idm_same_task_future_delta_max_state_distance,
                    idm_same_task_future_delta_min_action_delta_mse=args.idm_same_task_future_delta_min_action_delta_mse,
                    idm_current_frame_dropout=args.idm_current_frame_dropout,
                    idm_wan_vae_current_latent_dropout=args.idm_wan_vae_current_latent_dropout,
                    idm_wan_vae_latent_noise_prob=args.idm_wan_vae_latent_noise_prob,
                    idm_wan_vae_latent_noise_s_min=args.idm_wan_vae_latent_noise_s_min,
                    idm_wan_vae_latent_noise_s_max=args.idm_wan_vae_latent_noise_s_max,
                    seed=seed,
                )
                try:
                    run_kwargs = {}
                    if args.idm_context_action_warmup_epochs is not None:
                        run_kwargs["idm_context_action_warmup_epochs"] = args.idm_context_action_warmup_epochs
                    train_metrics = run_idm_training(train_config, **run_kwargs)
                    final_checkpoint = run_dir / "idm_checkpoint.pt"
                    best_checkpoint = run_dir / "best_idm_checkpoint.pt"
                    final_full_eval = run_full_dataset_eval(
                        checkpoint=final_checkpoint,
                        dataset_config=dataset_config,
                        cached_future_dir=args.cached_future_dir,
                        batch_size=args.batch_size,
                        device=args.device,
                        flow_eval_seed=args.flow_eval_seed,
                    )
                    best_full_eval = run_full_dataset_eval(
                        checkpoint=best_checkpoint,
                        dataset_config=dataset_config,
                        cached_future_dir=args.cached_future_dir,
                        batch_size=args.batch_size,
                        device=args.device,
                        flow_eval_seed=args.flow_eval_seed,
                    )
                    run_optional_diagnostics(
                        enabled=args.diagnostics,
                        checkpoint=best_checkpoint,
                        dataset_config=dataset_config,
                        cached_future_dir=args.cached_future_dir,
                        batch_size=args.batch_size,
                        device=args.device,
                        flow_eval_seed=args.flow_eval_seed,
                        output_dir=run_dir / "best_diagnostics",
                    )
                    row = {
                        "run_name": run_name,
                        "status": "success",
                        "run_dir": str(run_dir),
                        "seed": seed,
                        "action_horizon": action_horizon,
                        "learning_rate": learning_rate,
                        "final_checkpoint": str(final_checkpoint),
                        "best_checkpoint": str(best_checkpoint),
                        "train_final": train_metrics["final"],
                        "train_best": train_metrics["best"],
                        "final_full_eval": final_full_eval,
                        "best_full_eval": best_full_eval,
                    }
                except Exception as error:
                    row = {
                        "run_name": run_name,
                        "status": "failed",
                        "error": str(error),
                    }
                rows.append(row)
                write_experiment_summary(output_dir, args, rows)
                print(json.dumps(row, sort_keys=True))

    write_experiment_summary(output_dir, args, rows)
    print(json.dumps({"output_dir": str(output_dir), "num_runs": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
