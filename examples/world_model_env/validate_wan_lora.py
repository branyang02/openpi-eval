from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import tyro
from PIL import Image

from world_model import train_lib as _train_lib
from world_model.config import DatasetConfig, DatasetSource, FutureFrameStrategy
from world_model.data import (
    expected_wan_selected_frame_indices,
    validate_cached_future_temporal_contract,
)
from world_model.diffsynth_wan import (
    DiffSynthWanLoraConfig,
    DiffSynthWanLoraFutureGenerator,
    validate_diffsynth_repo,
    validate_local_wan_checkpoint,
    validate_lora_path,
)

enforce_idm_frame_delta_contract = _train_lib.enforce_idm_frame_delta_contract
load_idm_training_frame_delta = _train_lib.load_idm_training_frame_delta


@dataclasses.dataclass
class Args:
    """Generate one short validation clip from a trained Wan2.2 LoRA."""

    diffsynth_repo_dir: str
    checkpoint_dir: str
    lora_path: str
    input_image: str
    prompt: str
    output_video: str = "output/wan_lora_validation.mp4"
    height: int = 64
    width: int = 64
    num_frames: int = 17
    num_inference_steps: int = 2
    seed: int = 0
    fps: int = 15
    lora_alpha: float = 1.0
    device: str = "cuda"
    tiled: bool = True


def validate_inputs(args: Args) -> dict[str, object]:
    diffsynth_repo_dir = validate_diffsynth_repo(args.diffsynth_repo_dir)
    lora_path = validate_lora_path(args.lora_path)
    input_image = Path(args.input_image).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    checkpoint = validate_local_wan_checkpoint(args.checkpoint_dir)
    if not input_image.exists():
        raise FileNotFoundError(f"Input image not found: {input_image}")
    if args.num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive.")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive.")
    return {
        "diffsynth_repo_dir": str(diffsynth_repo_dir),
        "checkpoint": checkpoint,
        "lora_path": str(lora_path),
        "input_image": str(input_image),
        "output_video": str(output_video),
    }


def generate_main(args: Args) -> None:
    summary = validate_inputs(args)
    generator = DiffSynthWanLoraFutureGenerator(
        DiffSynthWanLoraConfig(
            diffsynth_repo_dir=str(summary["diffsynth_repo_dir"]),
            checkpoint_dir=args.checkpoint_dir,
            lora_path=str(summary["lora_path"]),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            lora_alpha=args.lora_alpha,
            device=args.device,
            tiled=args.tiled,
            fps=args.fps,
            base_seed=args.seed,
            prompt_template="{task}",
        )
    )
    from world_model.data import image_to_chw_float

    image_tensor = image_to_chw_float(Image.open(summary["input_image"]).convert("RGB"), args.height)
    result = generator.generate_view(
        image_tensor,
        task_text=args.prompt,
        output_dir=Path(summary["output_video"]).parent,
        image_size=args.height,
        num_future_frames=max(args.num_frames - 1, 1),
        seed=args.seed,
        stem=Path(summary["output_video"]).stem,
    )
    output_video = Path(summary["output_video"])
    if result.video_path != output_video:
        result.video_path.replace(output_video)
    summary["generated_video"] = str(output_video)
    print(json.dumps(summary, sort_keys=True))


RankBy = Literal["idm_mse", "idm_decodability_gap", "future_mse"]


@dataclasses.dataclass
class RankArgs:
    """Rank Wan2.2 LoRA checkpoints by pixel future quality and IDM action decodability.

    Each ``--cache-dirs`` entry is a future cache produced by
    ``cache_future_rollouts.py --future-source wan_lora`` for one LoRA checkpoint.
    Dataset shape (image size, future frames, action horizon) is taken from the
    IDM checkpoint so the cache, the dataset, and the IDM cannot silently disagree.
    """

    idm_checkpoint: str
    cache_dirs: tuple[str, ...]
    labels: tuple[str, ...] | None = None
    output_dir: str = "output/wan_lora_ranking"
    dataset_source: DatasetSource = "lerobot"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_key: str = "corner4.image"
    episodes: tuple[int, ...] | None = None
    max_samples: int | None = None
    samples_per_episode: int | None = None
    synthetic_samples: int = 8
    frame_delta: int = 1
    seed: int = 7
    batch_size: int = 16
    device: str = "auto"
    flow_eval_seed: int | None = 0
    # Which key orders the printed ``ranked`` list. ``idm_decodability_gap`` ranks by how close
    # each checkpoint's action decodability is to the ground-truth futures (|gap|, smallest first);
    # ``idm_mse``/``future_mse`` rank by lowest absolute error. All rankings are always written.
    rank_by: RankBy = "idm_mse"
    visual_samples: int = 4
    visual_tile_size: int = 96


# Every ranking key sorts ascending: the smallest sort value ranks first.
#
# ``idm_mse`` and ``future_mse`` are absolute error metrics, so lower is literally better.
#
# ``idm_decodability_gap`` is reported per run as the SIGNED difference
# ``idm_mse(generated) - idm_mse(ground_truth)`` so the sign stays visible (positive: the
# generated futures are harder for the IDM to decode actions from than the real futures;
# negative: artificially easier). For *ranking* we use its MAGNITUDE: the most faithful
# checkpoint is the one whose generated futures are as action-decodable as the real futures
# (gap closest to zero). A large negative gap is as much a fidelity failure as a large positive
# one and must not rank first. Ranking by the signed value would also be redundant with
# ``idm_mse`` -- the ground-truth idm_mse is constant across runs, so signed-gap order is
# identical to idm_mse order.
RANK_KEYS: dict[str, Callable[[dict], float]] = {
    "idm_mse": lambda run: run["idm"]["idm_mse"],
    "idm_decodability_gap": lambda run: abs(run["idm_decodability_gap"]),
    "future_mse": lambda run: run["pixel"]["future_mse"],
}


def resolve_cache_labels(cache_dirs: tuple[str, ...], labels: tuple[str, ...] | None) -> list[str]:
    if labels is not None:
        if len(labels) != len(cache_dirs):
            raise ValueError(
                f"--labels has {len(labels)} entries but --cache-dirs has {len(cache_dirs)}; "
                "provide one label per cache directory."
            )
        resolved = list(labels)
    else:
        resolved = [Path(cache_dir).expanduser().resolve().name for cache_dir in cache_dirs]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Cache labels must be unique, got {resolved}. Pass --labels to disambiguate.")
    return resolved


def validate_ranking_inputs(args: RankArgs) -> dict[str, object]:
    """Fail loudly on any missing IDM checkpoint, cache directory, manifest, or config."""

    idm_checkpoint = Path(args.idm_checkpoint).expanduser().resolve()
    if not idm_checkpoint.exists():
        raise FileNotFoundError(f"IDM checkpoint not found: {idm_checkpoint}")
    if not args.cache_dirs:
        raise ValueError("At least one --cache-dirs entry is required to rank Wan LoRA checkpoints.")

    labels = resolve_cache_labels(args.cache_dirs, args.labels)
    entries: list[dict[str, str]] = []
    for label, cache_dir in zip(labels, args.cache_dirs):
        resolved = Path(cache_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Wan LoRA cache directory not found: {resolved}")
        manifest_path = resolved / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cached future manifest not found: {manifest_path}")
        config_path = resolved / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Cached future config not found: {config_path}")
        entries.append({"label": label, "cache_dir": str(resolved)})
    return {"idm_checkpoint": str(idm_checkpoint), "entries": entries}


def build_ranking_dataset_config(model_config, args: RankArgs) -> DatasetConfig:
    return DatasetConfig(
        source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=(args.image_key,),
        frame_delta=args.frame_delta,
        num_future_frames=model_config.num_future_frames,
        action_horizon=model_config.action_horizon,
        image_size=model_config.image_size,
        max_samples=args.max_samples,
        samples_per_episode=args.samples_per_episode,
        episodes=args.episodes,
        synthetic_samples=args.synthetic_samples,
        task_vocab_size=model_config.task_vocab_size,
        seed=args.seed,
    )


def expected_selected_frame_indices(
    frame_delta: int,
    num_future_frames: int,
    *,
    strategy: FutureFrameStrategy = "first",
) -> list[int]:
    """Generated-video indices selected after skipping Wan's conditioning frame 0."""

    return expected_wan_selected_frame_indices(frame_delta, num_future_frames, strategy=strategy)


def _load_cache_config(cache_dir: str | Path) -> dict[str, Any]:
    config_path = Path(cache_dir).expanduser().resolve() / "config.json"
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Cached future config must be a JSON object: {config_path}")
    return config


def validate_cache_temporal_contract(
    label: str,
    cache_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    requested_frame_delta: int,
    requested_num_future_frames: int,
) -> None:
    """Reject Wan caches whose selected video frames do not satisfy the IDM future contract."""

    config = _load_cache_config(cache_dir)
    validate_cached_future_temporal_contract(
        cache_dir=Path(cache_dir).expanduser().resolve(),
        cache_config=config,
        rows=rows,
        frame_delta=requested_frame_delta,
        num_future_frames=requested_num_future_frames,
    )


def validate_cache_sample_identity(label: str, cached_dataset, expected_num_samples: int) -> None:
    """Reject caches whose manifest does not address exactly the selected base samples.

    A future cache pairs each generated future with a ``dataset_index`` into the base
    dataset, and ``cache_future_rollouts.py`` writes one row per base sample, in order,
    so a valid cache's ``dataset_index`` column is exactly ``range(len(base_dataset))``.
    A same-length manifest that duplicates, shuffles, or points outside that range would
    quietly make the ranking score different samples than the ground-truth reference
    (or weight some samples twice), so we reject all three up front and keep valid
    generated caches working unchanged.
    """

    rows = getattr(cached_dataset, "rows", None)
    if rows is None:
        raise TypeError(f"Cache '{label}' did not expose manifest rows; expected a CachedFutureDataset.")
    try:
        indices = [int(row["dataset_index"]) for row in rows]
    except KeyError as error:
        raise ValueError(
            f"Cache '{label}' manifest has a row missing 'dataset_index'; "
            "regenerate the cache with cache_future_rollouts.py."
        ) from error

    out_of_range = sorted({index for index in indices if index < 0 or index >= expected_num_samples})
    if out_of_range:
        raise ValueError(
            f"Cache '{label}' manifest references dataset_index values {out_of_range} outside the base "
            f"dataset range [0, {expected_num_samples}). The cache and the base dataset describe different "
            "samples; align --max-samples/--episodes with the cache or regenerate it for this selection."
        )

    duplicates = sorted(index for index, count in Counter(indices).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Cache '{label}' manifest repeats dataset_index values {duplicates}; each base sample must "
            "appear exactly once. A duplicated/shuffled manifest silently re-weights or misaligns samples "
            "relative to the ground-truth reference. Regenerate the cache with cache_future_rollouts.py."
        )

    expected_sequence = list(range(expected_num_samples))
    if indices != expected_sequence:
        raise ValueError(
            f"Cache '{label}' manifest dataset_index sequence does not match the selected base dataset. "
            f"Expected {expected_sequence} but got {indices}. The cache must address every base sample "
            "exactly once, in order; regenerate it for this base dataset/episodes selection."
        )


def evaluate_idm_over_dataset(idm, dataset, *, device, batch_size: int, flow_eval_seed: int | None) -> dict[str, float]:
    from torch.utils.data import DataLoader

    from world_model.train_lib import evaluate_idm

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metrics = evaluate_idm(idm, loader, device, flow_eval_seed=flow_eval_seed)
    return {
        "idm_mse": float(metrics["idm_mse"]),
        "idm_smooth_l1": float(metrics["idm_smooth_l1"]),
        "num_samples": int(len(dataset)),
    }


def evaluate_pixel_future_metrics(
    args: RankArgs,
    dataset_config: DatasetConfig,
    cache_dir: str,
    output_dir: Path,
) -> dict[str, object]:
    import evaluate_future_cache

    evaluate_future_cache.main(
        evaluate_future_cache.Args(
            cache_dir=cache_dir,
            dataset_source=args.dataset_source,
            repo_id=args.repo_id,
            image_key=args.image_key,
            output_dir=str(output_dir),
            episodes=args.episodes,
            max_samples=args.max_samples,
            samples_per_episode=args.samples_per_episode,
            synthetic_samples=args.synthetic_samples,
            image_size=dataset_config.image_size,
            frame_delta=dataset_config.frame_delta,
            num_future_frames=dataset_config.num_future_frames,
            action_horizon=dataset_config.action_horizon,
            seed=args.seed,
            visual_samples=args.visual_samples,
            visual_tile_size=args.visual_tile_size,
        )
    )
    metrics_path = output_dir / "future_cache_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    return {
        "future_mse": float(metrics["future_mse"]),
        "future_mae": float(metrics["future_mae"]),
        "future_psnr": float(metrics["future_psnr"]),
        "max_abs_error": float(metrics["max_abs_error"]),
        "num_samples": int(metrics["num_samples"]),
        "metrics_path": str(metrics_path),
        "contact_sheet": metrics["contact_sheet"],
    }


def rank_wan_lora_checkpoints(args: RankArgs) -> dict[str, object]:
    validated = validate_ranking_inputs(args)
    from world_model.train_lib import create_dataset_with_optional_cache, load_idm_checkpoint, resolve_device

    device = resolve_device(args.device)
    # Temporal contract: shape comes from model_config, but frame_delta is only recorded in the
    # training metadata. Reject the run before any expensive work if it disagrees with the IDM.
    enforce_idm_frame_delta_contract(validated["idm_checkpoint"], args.frame_delta)
    idm, model_config = load_idm_checkpoint(validated["idm_checkpoint"], device)
    if model_config.num_views != 1:
        raise ValueError(
            "Wan LoRA futures generate one selected camera view. "
            f"Rank with a single-view IDM checkpoint; got num_views={model_config.num_views}."
        )

    dataset_config = build_ranking_dataset_config(model_config, args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = create_dataset_with_optional_cache(dataset_config, None)
    ground_truth = evaluate_idm_over_dataset(
        idm, base_dataset, device=device, batch_size=args.batch_size, flow_eval_seed=args.flow_eval_seed
    )

    runs: list[dict[str, object]] = []
    for entry in validated["entries"]:
        label = entry["label"]
        cache_dir = entry["cache_dir"]
        cached_dataset = create_dataset_with_optional_cache(dataset_config, cache_dir)
        if len(cached_dataset) != ground_truth["num_samples"]:
            raise ValueError(
                f"Cache '{label}' covers {len(cached_dataset)} samples but the ground-truth dataset has "
                f"{ground_truth['num_samples']} samples. Align --max-samples/--samples-per-episode/--episodes "
                "so the cache and the dataset evaluate the same samples instead of silently comparing different ones."
            )
        # Same length is not enough: the manifest must address exactly the selected base samples
        # so the cache and the ground-truth reference score the same samples, not a shuffled/duplicated set.
        validate_cache_sample_identity(label, cached_dataset, ground_truth["num_samples"])
        validate_cache_temporal_contract(
            label,
            cache_dir,
            cached_dataset.rows,
            requested_frame_delta=dataset_config.frame_delta,
            requested_num_future_frames=dataset_config.num_future_frames,
        )
        idm_metrics = evaluate_idm_over_dataset(
            idm, cached_dataset, device=device, batch_size=args.batch_size, flow_eval_seed=args.flow_eval_seed
        )
        pixel_metrics = evaluate_pixel_future_metrics(args, dataset_config, cache_dir, output_dir / label / "pixel")
        runs.append(
            {
                "label": label,
                "cache_dir": cache_dir,
                "num_samples": idm_metrics["num_samples"],
                "pixel": pixel_metrics,
                "idm": idm_metrics,
                "idm_decodability_gap": idm_metrics["idm_mse"] - ground_truth["idm_mse"],
                "idm_smooth_l1_gap": idm_metrics["idm_smooth_l1"] - ground_truth["idm_smooth_l1"],
            }
        )

    rankings = {f"by_{key}": [run["label"] for run in sorted(runs, key=key_fn)] for key, key_fn in RANK_KEYS.items()}
    summary = {
        "idm_checkpoint": validated["idm_checkpoint"],
        "device": str(device),
        "rank_by": args.rank_by,
        "dataset_config": dataclasses.asdict(dataset_config),
        "model_config": {
            "num_views": model_config.num_views,
            "image_size": model_config.image_size,
            "num_future_frames": model_config.num_future_frames,
            "action_horizon": model_config.action_horizon,
            "action_dim": model_config.action_dim,
            "idm_arch": model_config.idm_arch,
        },
        "ground_truth_reference": ground_truth,
        "runs": runs,
        "rankings": rankings,
        "best": {key: (labels[0] if labels else None) for key, labels in rankings.items()},
        "ranked": sorted(runs, key=RANK_KEYS[args.rank_by]),
        "output_dir": str(output_dir),
    }
    (output_dir / "ranking_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def rank_main(args: RankArgs) -> None:
    summary = rank_wan_lora_checkpoints(args)
    print(
        json.dumps(
            {
                "output_dir": summary["output_dir"],
                "num_caches": len(summary["runs"]),
                "rank_by": summary["rank_by"],
                "ranked": summary["rankings"][f"by_{args.rank_by}"],
                "best": summary["best"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    command = tyro.extras.subcommand_cli_from_dict(
        {
            "rank": RankArgs,
            "generate": Args,
        }
    )
    if isinstance(command, RankArgs):
        rank_main(command)
    else:
        generate_main(command)
