"""Print the next matched Wan action-mode comparison commands.

This planner is deliberately CPU-only: it renders the command sequence for a
larger fair run, but never imports Wan or starts GPU jobs.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

DEFAULT_IDM_CHECKPOINT = (
    "output/idm_flow_patch_decoded_wan_smoke_ep0_15_h4/best_idm_checkpoint.pt"
)
DEFAULT_WAN_LORA_PATH = "output/wan_metaworld_ep0_15_1424_128_lora3_e5/epoch-3.safetensors"
DEFAULT_CURRENT_ACTION_EXPERT_CHECKPOINT = (
    "output/pi05_wan_action_expert_dit_train2_eval1_spe4_h4_prefixonly_norm_seed109_e300_h512_l6/checkpoint.pt"
)
DEFAULT_PARTIAL_ACTION_EXPERT_CHECKPOINT = (
    "output/pi05_wan_partial_action_expert_train2_eval1_spe4_h4_lat2_noise_seed0_prefixonly_norm_seed109_e300_h512_l6/checkpoint.pt"
)
DEFAULT_EPISODES = (16, 17, 18, 19, 20, 21, 22, 23)
DEFAULT_DIT_SELECTED_LAYERS = (0, 14, 29)
DEFAULT_DIT_HIDDEN_POOL_CONFIG = "model_fn_wan_video_block_hooks_mean_pool_selected_layers"


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    output_root: Path
    sample_count: int = 256
    samples_per_episode: int | None = 32
    dataset_source: str = "lerobot"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_key: str = "corner4.image"
    episodes: tuple[int, ...] = DEFAULT_EPISODES
    image_size: int = 64
    frame_delta: int = 1
    num_future_frames: int = 4
    action_horizon: int = 4
    seed: int = 7
    batch_size: int = 16
    uv_cache_dir: str = "/tmp/uv-cache"
    idm_checkpoint: str = DEFAULT_IDM_CHECKPOINT
    current_action_expert_checkpoint: str = DEFAULT_CURRENT_ACTION_EXPERT_CHECKPOINT
    partial_action_expert_checkpoint: str = DEFAULT_PARTIAL_ACTION_EXPERT_CHECKPOINT
    diffsynth_repo_dir: str = "/tmp/DiffSynth-Studio"
    wan_checkpoint_dir: str = "/tmp/wan2.2-ti2v-5b"
    wan_lora_path: str = DEFAULT_WAN_LORA_PATH
    wan_lora_height: int = 128
    wan_lora_width: int = 128
    wan_lora_num_frames: int = 17
    wan_lora_num_inference_steps: int = 8
    generation_seed: int = 1007
    prefix_dim: int = 3072
    prefix_backend: str = "dit_hidden"
    dit_hidden_pool: str = "mean"
    dit_selected_layers: tuple[int, ...] = DEFAULT_DIT_SELECTED_LAYERS
    dit_tokens_per_layer: int = 1
    dit_timestep: float = 500.0
    dit_num_latent_frames_current: int = 1
    dit_num_latent_frames_partial: int = 2
    dit_future_latent_fill: str = "noise"
    dit_future_latent_seed: int = 0
    action_sample_steps: int = 16


@dataclass(frozen=True, slots=True)
class PlannedCommand:
    title: str
    command: tuple[str, ...]
    cuda_visible_devices: str | None = None
    note: str | None = None


def _shell_join(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _render_command(command: PlannedCommand, *, uv_cache_dir: str) -> str:
    env_parts = [f"UV_CACHE_DIR={shlex.quote(uv_cache_dir)}"]
    if command.cuda_visible_devices is not None:
        env_parts.insert(0, f"CUDA_VISIBLE_DEVICES={shlex.quote(command.cuda_visible_devices)}")
    return f"{' '.join(env_parts)} {_shell_join(command.command)}"


def _with_continuations(line: str) -> str:
    parts = line.split(" ")
    if len(parts) <= 8:
        return line
    rendered = " ".join(parts[:6])
    rest = parts[6:]
    while rest:
        rendered += " \\\n    " + " ".join(rest[:6])
        rest = rest[6:]
    return rendered


def _dataset_flags(config: PlannerConfig) -> tuple[str, ...]:
    flags = (
        "--dataset-source",
        config.dataset_source,
        "--repo-id",
        config.repo_id,
        "--episodes",
        *(str(episode) for episode in config.episodes),
        "--image-size",
        str(config.image_size),
        "--action-horizon",
        str(config.action_horizon),
        "--seed",
        str(config.seed),
    )
    if config.samples_per_episode is not None:
        return (*flags, "--samples-per-episode", str(config.samples_per_episode))
    return (*flags, "--max-samples", str(config.sample_count))


def _planned_sample_count(config: PlannerConfig) -> int:
    if config.samples_per_episode is not None:
        return len(config.episodes) * config.samples_per_episode
    return config.sample_count


def _decoded_dataset_flags(config: PlannerConfig) -> tuple[str, ...]:
    return (
        *_dataset_flags(config),
        "--image-key",
        config.image_key,
        "--frame-delta",
        str(config.frame_delta),
        "--num-future-frames",
        str(config.num_future_frames),
    )


def _idm_eval_dataset_flags(config: PlannerConfig) -> tuple[str, ...]:
    flags = (
        "--dataset-source",
        config.dataset_source,
        "--repo-id",
        config.repo_id,
        "--image-keys",
        config.image_key,
        "--episodes",
        *(str(episode) for episode in config.episodes),
        "--image-size",
        str(config.image_size),
        "--frame-delta",
        str(config.frame_delta),
        "--num-future-frames",
        str(config.num_future_frames),
        "--action-horizon",
        str(config.action_horizon),
        "--seed",
        str(config.seed),
    )
    if config.samples_per_episode is not None:
        return (*flags, "--samples-per-episode", str(config.samples_per_episode))
    return (*flags, "--max-samples", str(config.sample_count))


def _prefix_cache_command(
    config: PlannerConfig,
    *,
    output_dir: Path,
    dit_num_latent_frames: int,
    future_latent_fill: str,
    future_latent_seed: int,
) -> tuple[str, ...]:
    return (
        "uv",
        "run",
        "python",
        "cache_pi05_wan_prefix_tokens.py",
        *_dataset_flags(config),
        "--image-key",
        config.image_key,
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(config.batch_size),
        "--device",
        "cuda:0",
        "--prefix-dim",
        str(config.prefix_dim),
        "--prefix-backend",
        config.prefix_backend,
        "--wan-repo-dir",
        config.diffsynth_repo_dir,
        "--wan-checkpoint-dir",
        config.wan_checkpoint_dir,
        "--dit-hidden-pool",
        config.dit_hidden_pool,
        "--dit-selected-layers",
        *(str(layer) for layer in config.dit_selected_layers),
        "--dit-tokens-per-layer",
        str(config.dit_tokens_per_layer),
        "--dit-timestep",
        str(config.dit_timestep),
        "--dit-num-latent-frames",
        str(dit_num_latent_frames),
        "--dit-future-latent-fill",
        future_latent_fill,
        "--dit-future-latent-seed",
        str(future_latent_seed),
    )


def _prefix_eval_command(config: PlannerConfig, *, checkpoint: str, cache_dir: Path, output_dir: Path) -> tuple[str, ...]:
    return (
        "uv",
        "run",
        "python",
        "eval_pi05_wan_action_expert.py",
        "--checkpoint",
        checkpoint,
        "--cache-path",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--sample-steps",
        str(config.action_sample_steps),
        "--batch-size",
        str(config.batch_size),
        "--device",
        "cuda:0",
    )


def planned_commands(config: PlannerConfig) -> list[PlannedCommand]:
    planned_sample_count = _planned_sample_count(config)
    if planned_sample_count <= 8:
        raise ValueError(
            "planned sample count should be larger than the current matched n=8 run, "
            f"got {planned_sample_count}."
        )
    if config.frame_delta != 1:
        raise ValueError("Wan hidden-prefix cache rows currently use DatasetConfig(frame_delta=1); keep frame_delta=1.")
    if config.action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {config.action_horizon}.")
    if config.num_future_frames <= 0:
        raise ValueError(f"num_future_frames must be positive, got {config.num_future_frames}.")
    if config.samples_per_episode is not None and config.samples_per_episode <= 0:
        raise ValueError(f"samples_per_episode must be positive, got {config.samples_per_episode}.")
    if not config.dit_selected_layers:
        raise ValueError("dit_selected_layers must contain at least one layer.")
    if config.dit_tokens_per_layer <= 0:
        raise ValueError(f"dit_tokens_per_layer must be positive, got {config.dit_tokens_per_layer}.")
    if config.dit_timestep < 0.0:
        raise ValueError(f"dit_timestep must be non-negative, got {config.dit_timestep}.")

    root = config.output_root
    decoded_cache = root / "decoded_video_idm" / "future_cache"
    decoded_eval = root / "decoded_video_idm" / "eval"
    current_cache = root / "current_wan_prefix_action_expert" / "prefix_cache"
    current_eval = root / "current_wan_prefix_action_expert" / "eval"
    partial_cache = root / "partial_wan_prefix_action_expert" / "prefix_cache"
    partial_eval = root / "partial_wan_prefix_action_expert" / "eval"
    matrix_json = root / "wan_action_mode_matrix.json"

    return [
        PlannedCommand(
            title="Terminal A / GPU 0: cache decoded Wan-LoRA futures",
            cuda_visible_devices="0",
            command=(
                "uv",
                "run",
                "python",
                "cache_future_rollouts.py",
                "--future-source",
                "wan_lora",
                *_decoded_dataset_flags(config),
                "--output-dir",
                str(decoded_cache),
                "--generation-seed",
                str(config.generation_seed),
                "--diffsynth-repo-dir",
                config.diffsynth_repo_dir,
                "--wan-lora-checkpoint-dir",
                config.wan_checkpoint_dir,
                "--wan-lora-path",
                config.wan_lora_path,
                "--wan-lora-height",
                str(config.wan_lora_height),
                "--wan-lora-width",
                str(config.wan_lora_width),
                "--wan-lora-num-frames",
                str(config.wan_lora_num_frames),
                "--wan-lora-num-inference-steps",
                str(config.wan_lora_num_inference_steps),
                "--wan-lora-device",
                "cuda:0",
                "--wan-lora-future-frame-strategy",
                "first",
            ),
            note="This is the expensive decoded-video path; it writes reusable future pixels.",
        ),
        PlannedCommand(
            title="Terminal A / GPU 0: evaluate decoded futures with the IDM",
            cuda_visible_devices="0",
            command=(
                "uv",
                "run",
                "python",
                "eval_idm.py",
                "--checkpoint",
                config.idm_checkpoint,
                "--cached-future-dir",
                str(decoded_cache),
                *_idm_eval_dataset_flags(config),
                "--output-dir",
                str(decoded_eval),
                "--batch-size",
                str(config.batch_size),
                "--prediction-mode",
                "sample",
                "--flow-eval-seed",
                "0",
                "--device",
                "cuda:0",
            ),
            note="The result JSON is decoded_video_idm/eval/eval_metrics.json.",
        ),
        PlannedCommand(
            title="Terminal B / GPU 1: cache current Wan prefix memory",
            cuda_visible_devices="1",
            command=_prefix_cache_command(
                config,
                output_dir=current_cache,
                dit_num_latent_frames=config.dit_num_latent_frames_current,
                future_latent_fill="zeros",
                future_latent_seed=0,
            ),
            note=(
                "This computes the current-frame/text prefix once for action-expert reuse; it is not native Wan "
                "attention KV reuse."
            ),
        ),
        PlannedCommand(
            title="Terminal B / GPU 1: evaluate current Wan prefix action expert",
            cuda_visible_devices="1",
            command=_prefix_eval_command(
                config,
                checkpoint=config.current_action_expert_checkpoint,
                cache_dir=current_cache,
                output_dir=current_eval,
            ),
            note="The result JSON is current_wan_prefix_action_expert/eval/eval_metrics.json.",
        ),
        PlannedCommand(
            title="Terminal B / GPU 1: cache hybrid Wan future-latent memory",
            cuda_visible_devices="1",
            command=_prefix_cache_command(
                config,
                output_dir=partial_cache,
                dit_num_latent_frames=config.dit_num_latent_frames_partial,
                future_latent_fill=config.dit_future_latent_fill,
                future_latent_seed=config.dit_future_latent_seed,
            ),
            note=(
                "Include this hybrid path when a matching checkpoint exists; it runs Wan future-latent generation or "
                "incomplete denoising and tests reusable Wan-derived memory beyond the current-only prefix."
            ),
        ),
        PlannedCommand(
            title="Terminal B / GPU 1: evaluate hybrid Wan memory action expert",
            cuda_visible_devices="1",
            command=_prefix_eval_command(
                config,
                checkpoint=config.partial_action_expert_checkpoint,
                cache_dir=partial_cache,
                output_dir=partial_eval,
            ),
            note="The result JSON is partial_wan_prefix_action_expert/eval/eval_metrics.json.",
        ),
        PlannedCommand(
            title="After both terminals finish: write the comparison matrix JSON",
            command=(
                "uv",
                "run",
                "python",
                "run_wan_action_mode_matrix.py",
                "--decoded-video-idm",
                str(decoded_eval),
                "--current-wan-prefix-action-expert",
                str(current_eval),
                "--partial-wan-prefix-action-expert",
                str(partial_eval),
                "--output-json",
                str(matrix_json),
            ),
            note="The matrix should report sample_sets_match=true before the rows are treated as comparable.",
        ),
    ]


def render_plan(config: PlannerConfig) -> str:
    commands = planned_commands(config)
    sample_selector = (
        f"samples_per_episode={config.samples_per_episode}, planned_samples={_planned_sample_count(config)}"
        if config.samples_per_episode is not None
        else f"max_samples={config.sample_count}, planned_samples={config.sample_count}"
    )
    fingerprint = (
        f"dataset_source={config.dataset_source}, repo_id={config.repo_id}, image_key={config.image_key}, "
        f"episodes={list(config.episodes)}, {sample_selector}, image_size={config.image_size}, "
        f"frame_delta={config.frame_delta}, num_future_frames={config.num_future_frames}, "
        f"action_horizon={config.action_horizon}, seed={config.seed}"
    )
    hidden_prefix_contract = (
        f"prefix_backend={config.prefix_backend}, prefix_dim={config.prefix_dim}, "
        f"selected_layers={list(config.dit_selected_layers)}, hidden_pool={config.dit_hidden_pool}, "
        f"hidden_pool_config={DEFAULT_DIT_HIDDEN_POOL_CONFIG}, "
        f"tokens_per_layer={config.dit_tokens_per_layer}, dit_timestep={config.dit_timestep}, "
        f"current_num_latent_frames={config.dit_num_latent_frames_current}, "
        f"partial_num_latent_frames={config.dit_num_latent_frames_partial}, "
        f"partial_future_latent_fill={config.dit_future_latent_fill}, "
        f"partial_future_latent_seed={config.dit_future_latent_seed}"
    )
    lines = [
        "# Matched Wan action-mode comparison plan",
        f"# Sample fingerprint: {fingerprint}",
        f"# Hidden-prefix contract: {hidden_prefix_contract}",
        "# Two-GPU allocation: Terminal A uses physical GPU 0 for decoded video cache/eval; "
        "Terminal B uses physical GPU 1 for prefix/memory cache/eval.",
        "# True native Wan attention KV cache is not implemented here. Prefix/memory rows may expose learned or "
        "projected action-expert memory, not cached Wan attention KV.",
        "# Run from examples/world_model_env. These commands plan GPU work but this script does not run it.",
        "",
    ]
    for command in commands:
        lines.append(f"# {command.title}")
        if command.note is not None:
            lines.append(f"# {command.note}")
        lines.append(_with_continuations(_render_command(command, uv_cache_dir=config.uv_cache_dir)))
        lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print a matched Wan action-mode experiment command sequence.")
    parser.add_argument("--output-root", type=Path, help="Root output directory for all planned artifacts.")
    parser.add_argument("--sample-count", type=int, default=256, help="Matched eval sample count; must be > 8.")
    parser.add_argument(
        "--samples-per-episode",
        type=int,
        default=32,
        help="Use balanced per-episode sampling instead of --max-samples.",
    )
    parser.add_argument("--dataset-source", default="lerobot")
    parser.add_argument("--repo-id", default="brandonyang/metaworld_ml45")
    parser.add_argument("--image-key", default="corner4.image")
    parser.add_argument("--episodes", type=int, nargs="+", default=list(DEFAULT_EPISODES))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--frame-delta", type=int, default=1)
    parser.add_argument("--num-future-frames", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--uv-cache-dir", default="/tmp/uv-cache")
    parser.add_argument("--idm-checkpoint", default=DEFAULT_IDM_CHECKPOINT)
    parser.add_argument("--current-action-expert-checkpoint", default=DEFAULT_CURRENT_ACTION_EXPERT_CHECKPOINT)
    parser.add_argument("--partial-action-expert-checkpoint", default=DEFAULT_PARTIAL_ACTION_EXPERT_CHECKPOINT)
    parser.add_argument("--diffsynth-repo-dir", default="/tmp/DiffSynth-Studio")
    parser.add_argument("--wan-checkpoint-dir", default="/tmp/wan2.2-ti2v-5b")
    parser.add_argument("--wan-lora-path", default=DEFAULT_WAN_LORA_PATH)
    parser.add_argument("--wan-lora-height", type=int, default=128)
    parser.add_argument("--wan-lora-width", type=int, default=128)
    parser.add_argument("--wan-lora-num-frames", type=int, default=17)
    parser.add_argument("--wan-lora-num-inference-steps", type=int, default=8)
    parser.add_argument("--generation-seed", type=int, default=1007)
    parser.add_argument("--prefix-dim", type=int, default=3072)
    parser.add_argument("--prefix-backend", default="dit_hidden")
    parser.add_argument("--dit-hidden-pool", default="mean")
    parser.add_argument("--dit-selected-layers", type=int, nargs="+", default=list(DEFAULT_DIT_SELECTED_LAYERS))
    parser.add_argument("--dit-tokens-per-layer", type=int, default=1)
    parser.add_argument("--dit-timestep", type=float, default=500.0)
    parser.add_argument("--dit-num-latent-frames-current", type=int, default=1)
    parser.add_argument("--dit-num-latent-frames-partial", type=int, default=2)
    parser.add_argument("--dit-future-latent-fill", choices=("zeros", "noise"), default="noise")
    parser.add_argument("--dit-future-latent-seed", type=int, default=0)
    parser.add_argument("--action-sample-steps", type=int, default=16)
    return parser


def _config_from_args(args: argparse.Namespace) -> PlannerConfig:
    output_root = args.output_root
    if output_root is None:
        episode_span = f"ep{args.episodes[0]}_{args.episodes[-1]}" if args.episodes else "episodes"
        if args.samples_per_episode is not None:
            output_root = Path(
                f"output/wan_action_modes_matched_{episode_span}_spe{args.samples_per_episode}_h{args.action_horizon}"
            )
        else:
            output_root = Path(f"output/wan_action_modes_matched_{episode_span}_first{args.sample_count}")
    return PlannerConfig(
        output_root=output_root,
        sample_count=args.sample_count,
        samples_per_episode=args.samples_per_episode,
        dataset_source=args.dataset_source,
        repo_id=args.repo_id,
        image_key=args.image_key,
        episodes=tuple(args.episodes),
        image_size=args.image_size,
        frame_delta=args.frame_delta,
        num_future_frames=args.num_future_frames,
        action_horizon=args.action_horizon,
        seed=args.seed,
        batch_size=args.batch_size,
        uv_cache_dir=args.uv_cache_dir,
        idm_checkpoint=args.idm_checkpoint,
        current_action_expert_checkpoint=args.current_action_expert_checkpoint,
        partial_action_expert_checkpoint=args.partial_action_expert_checkpoint,
        diffsynth_repo_dir=args.diffsynth_repo_dir,
        wan_checkpoint_dir=args.wan_checkpoint_dir,
        wan_lora_path=args.wan_lora_path,
        wan_lora_height=args.wan_lora_height,
        wan_lora_width=args.wan_lora_width,
        wan_lora_num_frames=args.wan_lora_num_frames,
        wan_lora_num_inference_steps=args.wan_lora_num_inference_steps,
        generation_seed=args.generation_seed,
        prefix_dim=args.prefix_dim,
        prefix_backend=args.prefix_backend,
        dit_hidden_pool=args.dit_hidden_pool,
        dit_selected_layers=tuple(args.dit_selected_layers),
        dit_tokens_per_layer=args.dit_tokens_per_layer,
        dit_timestep=args.dit_timestep,
        dit_num_latent_frames_current=args.dit_num_latent_frames_current,
        dit_num_latent_frames_partial=args.dit_num_latent_frames_partial,
        dit_future_latent_fill=args.dit_future_latent_fill,
        dit_future_latent_seed=args.dit_future_latent_seed,
        action_sample_steps=args.action_sample_steps,
    )


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        rendered = render_plan(_config_from_args(args))
    except ValueError as exc:
        parser.error(str(exc))
    print(rendered, file=out or sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
