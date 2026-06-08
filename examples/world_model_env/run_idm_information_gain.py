"""Plan, run, and summarize IDM future-conditioning information-gain sweeps.

This helper replaces one-off ``current_only`` / ``future_only`` / ``full``
manual sweeps with a small reusable planner. By default it only writes a plan
and summarizes any metrics that already exist under ``output_dir``; pass
``--run`` to launch the planned ``train_idm.py`` and ``eval_idm.py`` commands.

Examples::

    python run_idm_information_gain.py --output-dir output/idm_info_gain
    python run_idm_information_gain.py --train-cache-config output/train_cache --eval-cache-config output/eval_cache
    python run_idm_information_gain.py --frame-deltas 1 2 4 --run
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_FUTURE_CONDITIONINGS = ("current_only", "future_only", "full")
VALID_FUTURE_CONDITIONINGS = frozenset(DEFAULT_FUTURE_CONDITIONINGS)
DEFAULT_IMAGE_KEYS = ("corner4.image",)
PLAN_FILENAME = "idm_information_gain_plan.json"
REPORT_JSON_FILENAME = "idm_information_gain_report.json"
REPORT_MARKDOWN_FILENAME = "idm_information_gain_report.md"


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return data


def resolve_config_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_dir():
        resolved = resolved / "config.json"
    return resolved


def cache_dataset_config(cache_config_path: str | Path) -> dict[str, Any]:
    path = resolve_config_path(cache_config_path)
    config = _read_json_object(path, label="cache config")
    dataset_config = config.get("dataset_config")
    if isinstance(dataset_config, dict):
        return dict(dataset_config)
    # A few lightweight test fixtures and hand-authored configs store the
    # DatasetConfig fields at top level.
    return config


def _coerce_int_tuple(value: Any, *, field: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)):
        episodes = tuple(int(item) for item in value)
        if len(set(episodes)) != len(episodes):
            raise ValueError(f"{field} must be unique, got {episodes}.")
        return episodes
    raise ValueError(f"{field} must be an integer list/tuple or null, got {value!r}.")


def resolve_split_episodes(
    *,
    cli_episodes: Sequence[int] | None,
    cache_config_path: str | Path | None,
    split_name: str,
) -> dict[str, Any]:
    if cli_episodes is not None:
        episodes = tuple(int(item) for item in cli_episodes)
        if len(set(episodes)) != len(episodes):
            raise ValueError(f"{split_name} episodes must be unique, got {episodes}.")
        return {"episodes": episodes, "source": "cli", "cache_config": None}
    if cache_config_path is None:
        return {"episodes": None, "source": "unspecified", "cache_config": None}

    config_path = resolve_config_path(cache_config_path)
    dataset_config = cache_dataset_config(config_path)
    return {
        "episodes": _coerce_int_tuple(dataset_config.get("episodes"), field=f"{split_name} cache episodes"),
        "source": "cache_config",
        "cache_config": str(config_path),
    }


def _normalize_episode_lengths(value: Any, *, episodes: Sequence[int] | None = None) -> dict[int, int]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {int(episode): int(length) for episode, length in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if all(isinstance(item, Mapping) for item in value):
            lengths: dict[int, int] = {}
            for item in value:
                episode = item.get("episode_index", item.get("episode", item.get("id")))
                length = item.get("length", item.get("num_frames", item.get("frame_count")))
                if episode is None or length is None:
                    raise ValueError(f"Episode length rows need episode and length fields, got {item!r}.")
                lengths[int(episode)] = int(length)
            return lengths
        if episodes is not None:
            if len(value) != len(episodes):
                raise ValueError(
                    "Episode length list length does not match episodes list length: "
                    f"{len(value)} != {len(episodes)}."
                )
            return {int(episode): int(length) for episode, length in zip(episodes, value, strict=True)}
    raise ValueError(f"Unsupported episode length metadata shape: {value!r}.")


def episode_lengths_from_config(cache_config_path: str | Path) -> dict[int, int]:
    path = resolve_config_path(cache_config_path)
    config = _read_json_object(path, label="cache config")
    dataset_config = config.get("dataset_config") if isinstance(config.get("dataset_config"), dict) else {}
    episodes = _coerce_int_tuple(dataset_config.get("episodes") or config.get("episodes"), field="cache episodes")
    for container in (config, dataset_config):
        if not isinstance(container, Mapping):
            continue
        for key in (
            "episode_lengths",
            "episode_frame_counts",
            "episode_lengths_by_id",
            "episode_frame_counts_by_id",
        ):
            if key in container:
                return _normalize_episode_lengths(container[key], episodes=episodes)
    return {}


def load_episode_lengths_json(path: str | Path) -> dict[int, int]:
    data = _read_json_object(Path(path).expanduser(), label="episode lengths JSON")
    return _normalize_episode_lengths(data)


def parse_episode_length_assignments(assignments: Sequence[str] | None) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for assignment in assignments or ():
        if "=" not in assignment:
            raise ValueError(f"Episode length must be EPISODE=LENGTH, got {assignment!r}.")
        episode_text, length_text = assignment.split("=", 1)
        episode = int(episode_text)
        if episode in lengths:
            raise ValueError(f"Duplicate episode length for episode {episode}.")
        lengths[episode] = int(length_text)
    return lengths


def probe_lerobot_episode_lengths(repo_id: str, episodes: Sequence[int] | None) -> dict[int, int]:
    """Count frames per episode from a LeRobot dataset without constructing models."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError("LeRobot is not importable; provide --episode-lengths-json or --episode-length.") from error

    kwargs: dict[str, Any] = {}
    if episodes is not None:
        kwargs["episodes"] = list(episodes)
    try:
        dataset = LeRobotDataset(repo_id, **kwargs)
    except TypeError:
        dataset = LeRobotDataset(repo_id, episodes=kwargs.get("episodes"), delta_timestamps={})

    if not hasattr(dataset, "hf_dataset"):
        raise RuntimeError("LeRobotDataset does not expose hf_dataset; cannot probe episode lengths.")
    raw_episode_values = dataset.hf_dataset["episode_index"]
    return dict(Counter(_scalar_int(value) for value in raw_episode_values))


def _scalar_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError(f"Expected scalar-like episode value, got {value!r}.")
        return _scalar_int(value[0])
    return int(value)


def merge_episode_lengths(
    *,
    explicit_lengths: Mapping[int, int] | None = None,
    episode_lengths_json: str | Path | None = None,
    cache_config_paths: Iterable[str | Path | None] = (),
    cli_assignments: Sequence[str] | None = None,
) -> dict[int, int]:
    merged: dict[int, int] = {}
    for cache_config_path in cache_config_paths:
        if cache_config_path is not None:
            merged.update(episode_lengths_from_config(cache_config_path))
    if episode_lengths_json is not None:
        merged.update(load_episode_lengths_json(episode_lengths_json))
    if explicit_lengths is not None:
        merged.update({int(episode): int(length) for episode, length in explicit_lengths.items()})
    merged.update(parse_episode_length_assignments(cli_assignments))
    return merged


def required_offsets(*, frame_delta: int, num_future_frames: int, action_horizon: int) -> dict[str, int]:
    if frame_delta <= 0:
        raise ValueError(f"frame_delta must be positive, got {frame_delta}.")
    if num_future_frames <= 0:
        raise ValueError(f"num_future_frames must be positive, got {num_future_frames}.")
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}.")
    future_offset = int(frame_delta) * int(num_future_frames)
    action_offset = max(int(action_horizon) - 1, 0)
    return {
        "required_future_offset": future_offset,
        "required_action_offset": action_offset,
        "required_offset": max(future_offset, action_offset),
    }


def filter_episodes_for_request(
    *,
    episodes: Sequence[int] | None,
    episode_lengths: Mapping[int, int],
    frame_delta: int,
    num_future_frames: int,
    action_horizon: int,
    samples_per_episode: int | None,
) -> dict[str, Any]:
    offsets = required_offsets(
        frame_delta=frame_delta,
        num_future_frames=num_future_frames,
        action_horizon=action_horizon,
    )
    if samples_per_episode is not None and samples_per_episode <= 0:
        raise ValueError(f"samples_per_episode must be positive, got {samples_per_episode}.")

    min_valid_windows = int(samples_per_episode) if samples_per_episode is not None else 1
    requested = None if episodes is None else [int(episode) for episode in episodes]
    result: dict[str, Any] = {
        **offsets,
        "min_valid_windows": min_valid_windows,
        "requested": requested,
        "kept": requested,
        "skipped": [],
        "unknown_length": [],
    }
    if episodes is None:
        return result

    kept: list[int] = []
    skipped: list[dict[str, Any]] = []
    unknown: list[int] = []
    for episode in episodes:
        episode = int(episode)
        length = episode_lengths.get(episode)
        if length is None:
            kept.append(episode)
            unknown.append(episode)
            continue
        valid_windows = max(int(length) - int(offsets["required_offset"]), 0)
        if valid_windows < min_valid_windows:
            skipped.append(
                {
                    "episode": episode,
                    "length": int(length),
                    "valid_windows": valid_windows,
                    "reason": (
                        f"valid_windows={valid_windows} < required min_valid_windows={min_valid_windows} "
                        f"for required_offset={offsets['required_offset']}"
                    ),
                }
            )
        else:
            kept.append(episode)

    result["kept"] = kept
    result["skipped"] = skipped
    result["unknown_length"] = unknown
    return result


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return
        command.append(flag)
        command.extend(str(item) for item in value)
        return
    command.extend((flag, str(value)))


def split_extra_args(chunks: Sequence[str] | None) -> list[str]:
    extra: list[str] = []
    for chunk in chunks or ():
        extra.extend(shlex.split(chunk))
    return extra


def command_display(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def build_train_command(
    *,
    python_executable: str,
    script_dir: str | Path,
    dataset_source: str,
    repo_id: str,
    image_keys: Sequence[str],
    output_dir: str | Path,
    episodes: Sequence[int] | None,
    max_samples: int | None,
    samples_per_episode: int | None,
    synthetic_samples: int,
    image_size: int,
    frame_delta: int,
    num_future_frames: int,
    action_horizon: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    device: str,
    seed: int,
    idm_arch: str,
    idm_visual_encoder: str,
    idm_future_conditioning: str,
    extra_args: Sequence[str] = (),
) -> list[str]:
    command = [python_executable, str(Path(script_dir) / "train_idm.py")]
    _append_option(command, "--dataset-source", dataset_source)
    _append_option(command, "--repo-id", repo_id)
    _append_option(command, "--image-keys", tuple(image_keys))
    _append_option(command, "--output-dir", output_dir)
    _append_option(command, "--episodes", tuple(episodes) if episodes is not None else None)
    _append_option(command, "--max-samples", max_samples)
    _append_option(command, "--samples-per-episode", samples_per_episode)
    _append_option(command, "--synthetic-samples", synthetic_samples)
    _append_option(command, "--image-size", image_size)
    _append_option(command, "--frame-delta", frame_delta)
    _append_option(command, "--num-future-frames", num_future_frames)
    _append_option(command, "--action-horizon", action_horizon)
    _append_option(command, "--epochs", epochs)
    _append_option(command, "--batch-size", batch_size)
    _append_option(command, "--learning-rate", learning_rate)
    _append_option(command, "--num-workers", num_workers)
    _append_option(command, "--device", device)
    _append_option(command, "--seed", seed)
    _append_option(command, "--idm-arch", idm_arch)
    _append_option(command, "--idm-visual-encoder", idm_visual_encoder)
    _append_option(command, "--idm-future-conditioning", idm_future_conditioning)
    command.extend(extra_args)
    return command


def build_eval_command(
    *,
    python_executable: str,
    script_dir: str | Path,
    checkpoint: str | Path,
    dataset_source: str,
    repo_id: str,
    image_keys: Sequence[str],
    output_dir: str | Path,
    episodes: Sequence[int] | None,
    max_samples: int | None,
    samples_per_episode: int | None,
    synthetic_samples: int,
    image_size: int,
    frame_delta: int,
    num_future_frames: int,
    action_horizon: int,
    batch_size: int,
    device: str,
    seed: int,
    flow_eval_seed: int | None,
    cached_future_dir: str | None,
    wan_vae_latent_cache_dir: str | None,
    generated_wan_latent_cache_dir: str | None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    command = [python_executable, str(Path(script_dir) / "eval_idm.py")]
    _append_option(command, "--checkpoint", checkpoint)
    _append_option(command, "--dataset-source", dataset_source)
    _append_option(command, "--repo-id", repo_id)
    _append_option(command, "--image-keys", tuple(image_keys))
    _append_option(command, "--output-dir", output_dir)
    _append_option(command, "--episodes", tuple(episodes) if episodes is not None else None)
    _append_option(command, "--max-samples", max_samples)
    _append_option(command, "--samples-per-episode", samples_per_episode)
    _append_option(command, "--synthetic-samples", synthetic_samples)
    _append_option(command, "--image-size", image_size)
    _append_option(command, "--frame-delta", frame_delta)
    _append_option(command, "--num-future-frames", num_future_frames)
    _append_option(command, "--action-horizon", action_horizon)
    _append_option(command, "--batch-size", batch_size)
    _append_option(command, "--device", device)
    _append_option(command, "--seed", seed)
    _append_option(command, "--flow-eval-seed", flow_eval_seed)
    _append_option(command, "--cached-future-dir", cached_future_dir)
    _append_option(command, "--wan-vae-latent-cache-dir", wan_vae_latent_cache_dir)
    _append_option(command, "--generated-wan-latent-cache-dir", generated_wan_latent_cache_dir)
    command.extend(extra_args)
    return command


def build_command_plan(
    *,
    output_dir: str | Path,
    frame_deltas: Sequence[int],
    future_conditionings: Sequence[str],
    train_episodes: Sequence[int] | None,
    eval_episodes: Sequence[int] | None,
    episode_lengths: Mapping[int, int] | None = None,
    train_episode_source: Mapping[str, Any] | None = None,
    eval_episode_source: Mapping[str, Any] | None = None,
    python_executable: str = "python",
    script_dir: str | Path | None = None,
    dataset_source: str = "synthetic",
    repo_id: str = "brandonyang/metaworld_ml45",
    image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
    train_max_samples: int | None = None,
    eval_max_samples: int | None = None,
    samples_per_episode: int | None = None,
    synthetic_samples: int = 128,
    image_size: int = 64,
    num_future_frames: int = 1,
    action_horizon: int = 32,
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    num_workers: int = 0,
    device: str = "auto",
    seed: int = 7,
    idm_arch: str = "flow_transformer",
    idm_visual_encoder: str = "patch",
    flow_eval_seed: int | None = 0,
    eval_cached_future_dir: str | None = None,
    eval_wan_vae_latent_cache_dir: str | None = None,
    eval_generated_wan_latent_cache_dir: str | None = None,
    checkpoint_name: str = "best_idm_checkpoint.pt",
    train_extra_args: Sequence[str] = (),
    eval_extra_args: Sequence[str] = (),
) -> dict[str, Any]:
    if not frame_deltas:
        raise ValueError("At least one frame_delta is required.")
    if not future_conditionings:
        raise ValueError("At least one idm_future_conditioning is required.")
    invalid_conditionings = sorted(set(future_conditionings) - VALID_FUTURE_CONDITIONINGS)
    if invalid_conditionings:
        raise ValueError(
            "idm_future_conditioning values must be one of "
            f"{sorted(VALID_FUTURE_CONDITIONINGS)}, got {invalid_conditionings}."
        )
    if samples_per_episode is not None and dataset_source != "lerobot":
        raise ValueError("samples_per_episode requires dataset_source='lerobot'.")
    if samples_per_episode is not None and (train_max_samples is not None or eval_max_samples is not None):
        raise ValueError("samples_per_episode cannot be combined with train_max_samples or eval_max_samples.")

    output_dir = Path(output_dir)
    script_dir = Path(__file__).parent if script_dir is None else Path(script_dir)
    lengths = {int(episode): int(length) for episode, length in (episode_lengths or {}).items()}
    filters: dict[str, dict[str, Any]] = {"train": {}, "eval": {}}
    runs: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, Any]] = []
    train_extra = list(train_extra_args)
    eval_extra = list(eval_extra_args)

    for frame_delta in frame_deltas:
        train_filter = filter_episodes_for_request(
            episodes=train_episodes,
            episode_lengths=lengths,
            frame_delta=frame_delta,
            num_future_frames=num_future_frames,
            action_horizon=action_horizon,
            samples_per_episode=samples_per_episode,
        )
        eval_filter = filter_episodes_for_request(
            episodes=eval_episodes,
            episode_lengths=lengths,
            frame_delta=frame_delta,
            num_future_frames=num_future_frames,
            action_horizon=action_horizon,
            samples_per_episode=samples_per_episode,
        )
        filters["train"][str(frame_delta)] = train_filter
        filters["eval"][str(frame_delta)] = eval_filter

        train_kept = None if train_filter["kept"] is None else tuple(train_filter["kept"])
        eval_kept = None if eval_filter["kept"] is None else tuple(eval_filter["kept"])
        skip_reason = None
        if train_episodes is not None and not train_kept:
            skip_reason = "no_train_episodes_after_filter"
        elif eval_episodes is not None and not eval_kept:
            skip_reason = "no_eval_episodes_after_filter"

        for conditioning in future_conditionings:
            run_name = f"idm_fd{frame_delta}_{conditioning}_seed{seed}"
            train_output_dir = output_dir / run_name
            eval_output_dir = train_output_dir / "eval"
            checkpoint = train_output_dir / checkpoint_name
            if skip_reason is not None:
                skipped_runs.append(
                    {
                        "run_name": run_name,
                        "frame_delta": int(frame_delta),
                        "idm_future_conditioning": conditioning,
                        "reason": skip_reason,
                    }
                )
                continue

            train_command = build_train_command(
                python_executable=python_executable,
                script_dir=script_dir,
                dataset_source=dataset_source,
                repo_id=repo_id,
                image_keys=image_keys,
                output_dir=train_output_dir,
                episodes=train_kept,
                max_samples=train_max_samples,
                samples_per_episode=samples_per_episode,
                synthetic_samples=synthetic_samples,
                image_size=image_size,
                frame_delta=frame_delta,
                num_future_frames=num_future_frames,
                action_horizon=action_horizon,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_workers=num_workers,
                device=device,
                seed=seed,
                idm_arch=idm_arch,
                idm_visual_encoder=idm_visual_encoder,
                idm_future_conditioning=conditioning,
                extra_args=train_extra,
            )
            eval_command = build_eval_command(
                python_executable=python_executable,
                script_dir=script_dir,
                checkpoint=checkpoint,
                dataset_source=dataset_source,
                repo_id=repo_id,
                image_keys=image_keys,
                output_dir=eval_output_dir,
                episodes=eval_kept,
                max_samples=eval_max_samples,
                samples_per_episode=samples_per_episode,
                synthetic_samples=synthetic_samples,
                image_size=image_size,
                frame_delta=frame_delta,
                num_future_frames=num_future_frames,
                action_horizon=action_horizon,
                batch_size=batch_size,
                device=device,
                seed=seed,
                flow_eval_seed=flow_eval_seed,
                cached_future_dir=eval_cached_future_dir,
                wan_vae_latent_cache_dir=eval_wan_vae_latent_cache_dir,
                generated_wan_latent_cache_dir=eval_generated_wan_latent_cache_dir,
                extra_args=eval_extra,
            )
            runs.append(
                {
                    "run_name": run_name,
                    "frame_delta": int(frame_delta),
                    "idm_future_conditioning": conditioning,
                    "train_output_dir": str(train_output_dir),
                    "eval_output_dir": str(eval_output_dir),
                    "checkpoint": str(checkpoint),
                    "train_episodes": None if train_kept is None else list(train_kept),
                    "eval_episodes": None if eval_kept is None else list(eval_kept),
                    "train_command": train_command,
                    "eval_command": eval_command,
                    "train_command_display": command_display(train_command),
                    "eval_command_display": command_display(eval_command),
                }
            )

    return {
        "output_dir": str(output_dir),
        "settings": {
            "dataset_source": dataset_source,
            "repo_id": repo_id,
            "image_keys": list(image_keys),
            "frame_deltas": [int(frame_delta) for frame_delta in frame_deltas],
            "future_conditionings": list(future_conditionings),
            "num_future_frames": int(num_future_frames),
            "action_horizon": int(action_horizon),
            "samples_per_episode": samples_per_episode,
            "train_max_samples": train_max_samples,
            "eval_max_samples": eval_max_samples,
            "synthetic_samples": int(synthetic_samples),
            "image_size": int(image_size),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "device": device,
            "seed": int(seed),
            "idm_arch": idm_arch,
            "idm_visual_encoder": idm_visual_encoder,
            "flow_eval_seed": flow_eval_seed,
        },
        "episode_sources": {
            "train": dict(train_episode_source or {"source": "unspecified", "episodes": train_episodes}),
            "eval": dict(eval_episode_source or {"source": "unspecified", "episodes": eval_episodes}),
        },
        "episode_lengths": {str(episode): length for episode, length in sorted(lengths.items())},
        "episode_filters": filters,
        "runs": runs,
        "skipped_runs": skipped_runs,
    }


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {"_error": str(error)}
    if not isinstance(data, dict):
        return {"_error": f"Expected JSON object in {path}"}
    return data


def _metric(data: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(data, Mapping):
        return None
    return data.get(key)


def summarize_train_metrics(data: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None
    if "_error" in data:
        return {"error": data["_error"]}
    model_config = data.get("model_config") if isinstance(data.get("model_config"), Mapping) else {}
    final = data.get("final") if isinstance(data.get("final"), Mapping) else {}
    best = data.get("best") if isinstance(data.get("best"), Mapping) else {}
    history = data.get("history") if isinstance(data.get("history"), list) else []
    return {
        "epochs": len(history) or final.get("epoch"),
        "idm_future_conditioning": model_config.get("idm_future_conditioning"),
        "final": _compact_metrics(final),
        "best": _compact_metrics(best),
        "stopped_early": data.get("stopped_early"),
    }


def summarize_eval_metrics(data: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None
    if "_error" in data:
        return {"error": data["_error"]}
    summary = _compact_metrics(data)
    mean_baseline = data.get("mean_action_baseline")
    if isinstance(mean_baseline, Mapping):
        summary["mean_action_baseline"] = _compact_metrics(mean_baseline)
    for key in ("checkpoint", "cached_future_dir", "wan_vae_latent_cache_dir", "generated_wan_latent_cache_dir"):
        if key in data:
            summary[key] = data.get(key)
    return summary


def _compact_metrics(data: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "epoch",
        "train_loss",
        "idm_mse",
        "idm_smooth_l1",
        "idm_generated_mse",
        "future_usage_rank_accuracy",
        "future_usage_gap",
        "future_usage_degradation",
        "future_usage_output_delta_mse",
        "wm_mse",
        "wm_psnr",
    )
    return {key: data[key] for key in keys if key in data}


def _primary_idm_mse(run: Mapping[str, Any]) -> float | None:
    eval_metrics = run.get("eval_metrics")
    if isinstance(eval_metrics, Mapping) and isinstance(eval_metrics.get("idm_mse"), int | float):
        return float(eval_metrics["idm_mse"])
    train_metrics = run.get("train_metrics")
    if isinstance(train_metrics, Mapping):
        best = train_metrics.get("best")
        if isinstance(best, Mapping) and isinstance(best.get("idm_mse"), int | float):
            return float(best["idm_mse"])
        final = train_metrics.get("final")
        if isinstance(final, Mapping) and isinstance(final.get("idm_mse"), int | float):
            return float(final["idm_mse"])
    return None


def build_report(plan: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(plan["output_dir"]))
    rows: list[dict[str, Any]] = []
    for item in plan.get("runs", []):
        if not isinstance(item, Mapping):
            continue
        train_metrics_path = Path(str(item["train_output_dir"])) / "metrics.json"
        eval_metrics_path = Path(str(item["eval_output_dir"])) / "eval_metrics.json"
        train_data = _load_optional_json(train_metrics_path)
        eval_data = _load_optional_json(eval_metrics_path)
        train_metrics = summarize_train_metrics(train_data)
        eval_metrics = summarize_eval_metrics(eval_data)
        metrics_status = "complete" if train_metrics is not None and eval_metrics is not None else "missing"
        if train_metrics is not None and "error" in train_metrics:
            metrics_status = "train_metrics_error"
        if eval_metrics is not None and "error" in eval_metrics:
            metrics_status = "eval_metrics_error"
        row = {
            "run_name": item["run_name"],
            "frame_delta": item["frame_delta"],
            "idm_future_conditioning": item["idm_future_conditioning"],
            "train_metrics_path": str(train_metrics_path),
            "eval_metrics_path": str(eval_metrics_path),
            "metrics_status": metrics_status,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
        }
        row["primary_idm_mse"] = _primary_idm_mse(row)
        rows.append(row)

    current_by_delta: dict[int, float] = {}
    for row in rows:
        if row["idm_future_conditioning"] != "current_only":
            continue
        primary = row.get("primary_idm_mse")
        if isinstance(primary, int | float):
            current_by_delta[int(row["frame_delta"])] = float(primary)

    for row in rows:
        primary = row.get("primary_idm_mse")
        baseline = current_by_delta.get(int(row["frame_delta"]))
        if isinstance(primary, int | float) and baseline is not None:
            row["idm_mse_delta_vs_current"] = baseline - float(primary)
            row["idm_mse_ratio_vs_current"] = float(primary) / baseline if baseline != 0 else None
        else:
            row["idm_mse_delta_vs_current"] = None
            row["idm_mse_ratio_vs_current"] = None

    best_by_delta: dict[str, Any] = {}
    for frame_delta in sorted({int(row["frame_delta"]) for row in rows}):
        candidates = [
            row
            for row in rows
            if int(row["frame_delta"]) == frame_delta and isinstance(row.get("primary_idm_mse"), int | float)
        ]
        best_by_delta[str(frame_delta)] = min(candidates, key=lambda row: row["primary_idm_mse"]) if candidates else None

    return {
        "output_dir": str(output_dir),
        "settings": plan.get("settings", {}),
        "episode_sources": plan.get("episode_sources", {}),
        "episode_filters": plan.get("episode_filters", {}),
        "num_planned_runs": len(plan.get("runs", [])),
        "num_skipped_runs": len(plan.get("skipped_runs", [])),
        "skipped_runs": plan.get("skipped_runs", []),
        "runs": rows,
        "best_by_frame_delta": best_by_delta,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_markdown_report(report: Mapping[str, Any]) -> str:
    settings = report.get("settings") if isinstance(report.get("settings"), Mapping) else {}
    lines = ["# IDM Information Gain Report", ""]
    lines.append(f"Output: `{report.get('output_dir')}`")
    lines.append(
        "Sweep: "
        f"frame_deltas={_fmt(settings.get('frame_deltas'))}, "
        f"future_conditionings={_fmt(settings.get('future_conditionings'))}, "
        f"num_future_frames={_fmt(settings.get('num_future_frames'))}"
    )
    lines.append("")

    lines.append("## Episode Filters")
    filters = report.get("episode_filters") if isinstance(report.get("episode_filters"), Mapping) else {}
    for split in ("train", "eval"):
        split_filters = filters.get(split) if isinstance(filters.get(split), Mapping) else {}
        for frame_delta, info in split_filters.items():
            skipped = info.get("skipped", []) if isinstance(info, Mapping) else []
            unknown = info.get("unknown_length", []) if isinstance(info, Mapping) else []
            kept = info.get("kept") if isinstance(info, Mapping) else None
            lines.append(
                f"- {split} fd={frame_delta}: kept={_fmt(kept)}, "
                f"skipped={len(skipped)}, unknown_length={_fmt(unknown)}"
            )
    lines.append("")

    lines.append("## Runs")
    lines.append("| frame_delta | conditioning | status | eval idm_mse | train best idm_mse | delta vs current |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in report.get("runs", []):
        if not isinstance(row, Mapping):
            continue
        eval_metrics = row.get("eval_metrics") if isinstance(row.get("eval_metrics"), Mapping) else {}
        train_metrics = row.get("train_metrics") if isinstance(row.get("train_metrics"), Mapping) else {}
        best = train_metrics.get("best") if isinstance(train_metrics.get("best"), Mapping) else {}
        lines.append(
            f"| {_fmt(row.get('frame_delta'))} | {_fmt(row.get('idm_future_conditioning'))} "
            f"| {_fmt(row.get('metrics_status'))} | {_fmt(eval_metrics.get('idm_mse'))} "
            f"| {_fmt(best.get('idm_mse'))} | {_fmt(row.get('idm_mse_delta_vs_current'))} |"
        )
    lines.append("")

    skipped_runs = report.get("skipped_runs") or []
    if skipped_runs:
        lines.append("## Skipped Runs")
        for run in skipped_runs:
            lines.append(
                f"- {_fmt(run.get('run_name'))}: {_fmt(run.get('reason'))} "
                f"(fd={_fmt(run.get('frame_delta'))}, conditioning={_fmt(run.get('idm_future_conditioning'))})"
            )
        lines.append("")

    return "\n".join(lines)


def launch_plan(
    plan: Mapping[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> None:
    for item in plan.get("runs", []):
        if not isinstance(item, Mapping):
            continue
        runner(item["train_command"], check=True)
        runner(item["eval_command"], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan/run/report IDM future-conditioning information-gain sweeps.")
    parser.add_argument("--output-dir", default="output/idm_information_gain")
    parser.add_argument("--dataset-source", default="synthetic", choices=("synthetic", "lerobot"))
    parser.add_argument("--repo-id", default="brandonyang/metaworld_ml45")
    parser.add_argument("--image-keys", nargs="+", default=list(DEFAULT_IMAGE_KEYS))
    parser.add_argument("--train-cache-config", default=None)
    parser.add_argument("--eval-cache-config", default=None)
    parser.add_argument("--train-episodes", nargs="*", type=int, default=None)
    parser.add_argument("--eval-episodes", nargs="*", type=int, default=None)
    parser.add_argument("--episode-lengths-json", default=None)
    parser.add_argument(
        "--episode-length",
        action="append",
        default=[],
        help="Episode length as EPISODE=LENGTH. May be repeated.",
    )
    parser.add_argument(
        "--probe-episode-lengths",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Probe LeRobot frame counts for missing episode lengths. Defaults to auto for lerobot episode splits; "
            "use --no-probe-episode-lengths for offline planning."
        ),
    )
    parser.add_argument("--frame-deltas", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--future-conditionings", nargs="+", default=list(DEFAULT_FUTURE_CONDITIONINGS))
    parser.add_argument("--num-future-frames", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--samples-per-episode", type=int, default=None)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--synthetic-samples", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--idm-arch", default="flow_transformer")
    parser.add_argument("--idm-visual-encoder", default="patch")
    parser.add_argument("--flow-eval-seed", type=int, default=0)
    parser.add_argument("--eval-cached-future-dir", default=None)
    parser.add_argument("--eval-wan-vae-latent-cache-dir", default=None)
    parser.add_argument("--eval-generated-wan-latent-cache-dir", default=None)
    parser.add_argument("--checkpoint-name", default="best_idm_checkpoint.pt")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--script-dir", default=str(Path(__file__).parent))
    parser.add_argument(
        "--train-extra",
        action="append",
        default=[],
        help="Additional train_idm.py argument chunk, shell-split. May be repeated.",
    )
    parser.add_argument(
        "--eval-extra",
        action="append",
        default=[],
        help="Additional eval_idm.py argument chunk, shell-split. May be repeated.",
    )
    parser.add_argument("--run", action="store_true", help="Launch train/eval commands after writing the plan.")
    parser.add_argument("--markdown", action="store_true", help="Print Markdown report instead of JSON.")
    parser.add_argument("--no-write-report", action="store_true", help="Do not write report files.")
    return parser


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    train_source = resolve_split_episodes(
        cli_episodes=args.train_episodes,
        cache_config_path=args.train_cache_config,
        split_name="train",
    )
    eval_source = resolve_split_episodes(
        cli_episodes=args.eval_episodes,
        cache_config_path=args.eval_cache_config,
        split_name="eval",
    )
    episode_lengths = merge_episode_lengths(
        episode_lengths_json=args.episode_lengths_json,
        cache_config_paths=(args.train_cache_config, args.eval_cache_config),
        cli_assignments=args.episode_length,
    )
    probe_episodes = sorted(set(train_source["episodes"] or ()) | set(eval_source["episodes"] or ()))
    missing_probe_episodes = [episode for episode in probe_episodes if episode not in episode_lengths]
    should_probe_episode_lengths = (
        args.dataset_source == "lerobot" and bool(missing_probe_episodes)
        if args.probe_episode_lengths is None
        else args.probe_episode_lengths
    )
    if should_probe_episode_lengths:
        probe_episodes = sorted(set(probe_episodes) | set(episode_lengths))
        try:
            probed = probe_lerobot_episode_lengths(args.repo_id, probe_episodes or None)
        except RuntimeError as error:
            parser.error(str(error))
        episode_lengths.update(probed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_command_plan(
        output_dir=output_dir,
        frame_deltas=args.frame_deltas,
        future_conditionings=args.future_conditionings,
        train_episodes=train_source["episodes"],
        eval_episodes=eval_source["episodes"],
        episode_lengths=episode_lengths,
        train_episode_source=train_source,
        eval_episode_source=eval_source,
        python_executable=args.python_executable,
        script_dir=args.script_dir,
        dataset_source=args.dataset_source,
        repo_id=args.repo_id,
        image_keys=tuple(args.image_keys),
        train_max_samples=args.train_max_samples,
        eval_max_samples=args.eval_max_samples,
        samples_per_episode=args.samples_per_episode,
        synthetic_samples=args.synthetic_samples,
        image_size=args.image_size,
        num_future_frames=args.num_future_frames,
        action_horizon=args.action_horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        idm_arch=args.idm_arch,
        idm_visual_encoder=args.idm_visual_encoder,
        flow_eval_seed=args.flow_eval_seed,
        eval_cached_future_dir=args.eval_cached_future_dir,
        eval_wan_vae_latent_cache_dir=args.eval_wan_vae_latent_cache_dir,
        eval_generated_wan_latent_cache_dir=args.eval_generated_wan_latent_cache_dir,
        checkpoint_name=args.checkpoint_name,
        train_extra_args=split_extra_args(args.train_extra),
        eval_extra_args=split_extra_args(args.eval_extra),
    )
    (output_dir / PLAN_FILENAME).write_text(json.dumps(_json_ready(plan), indent=2) + "\n")

    if args.run:
        launch_plan(plan)

    report = build_report(plan)
    markdown = render_markdown_report(report)
    if not args.no_write_report:
        (output_dir / REPORT_JSON_FILENAME).write_text(json.dumps(_json_ready(report), indent=2) + "\n")
        (output_dir / REPORT_MARKDOWN_FILENAME).write_text(markdown + "\n")

    print(markdown if args.markdown else json.dumps(_json_ready(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
