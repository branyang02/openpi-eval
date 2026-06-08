"""Build a compact JSON comparison for the three Wan action modes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from summarize_wan_action_modes import (
    MODE_KEYS,
    MSE_KEYS,
    PER_DIM_MSE_KEYS,
    extract_decoder_arch,
    inference_path_for_mode,
)
from world_model.action_modes import WanActionMode, get_action_mode_spec

DIRECTORY_RESULT_FILENAMES = (
    "metrics.json",
    "eval_metrics.json",
    "wan_idm_action.json",
    "pi05_wan_action.json",
)
DATASET_ACTION_MSE_KEY = "dataset_action_mse"
DATASET_ACTION_PER_DIM_MSE_KEY = "dataset_action_mse_per_action_dim"
MSE_RESULT_KEYS = (DATASET_ACTION_MSE_KEY, *MSE_KEYS, "idm_mse")
PER_DIM_MSE_RESULT_KEYS = (DATASET_ACTION_PER_DIM_MSE_KEY, *PER_DIM_MSE_KEYS)
COMPARISON_WARNING = (
    "Rows are not directly comparable unless they use the same action metric. This matrix prefers the shared "
    "dataset_action_mse metric when present; older decoded-video IDM and pi0.5 action-expert aliases are retained "
    "for historical result files."
)
WAN_CONTRACT_WARNING = (
    "No row reports true native Wan attention KV-cache reuse. Prefix/memory rows may expose learned or projected "
    "action-expert memory, not cached Wan attention KV."
)
EXPLICIT_SAMPLE_FINGERPRINT_KEYS = ("sample_fingerprint", "sample_set_fingerprint")
EXPLICIT_DATASET_FINGERPRINT_KEYS = ("dataset_fingerprint", "dataset_config_fingerprint")
FINGERPRINT_METADATA_CONTAINERS = ("metadata", "result_metadata")
SAMPLE_PATH_FINGERPRINT_KEYS = ("cache_path", "cached_future_dir")
SAMPLE_COUNT_FINGERPRINT_KEYS = ("num_samples", "num_valid_action_steps")


@dataclass(frozen=True, slots=True)
class MseMetric:
    value: float
    metric_name: str
    source_field: str


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, Any]:
    for key in keys:
        if key in data:
            return key, data[key]
    return None, None


def _json_normalized(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_normalized(item) for item in value]
    if isinstance(value, list):
        return [_json_normalized(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_normalized(item) for key, item in value.items()}
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _require_number(value: Any, *, key: str, path: Path) -> float:
    if not _is_number(value):
        raise ValueError(f"MSE field {key!r} in {path} must be numeric, got {type(value).__name__}")
    return float(value)


def _require_object(value: Any, *, key: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"fingerprint field {key!r} in {path} must be an object, got {type(value).__name__}")
    return _json_normalized(value)


def _require_number_list(value: Any, *, key: str, path: Path) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"per-dim MSE field {key!r} in {path} must be a list, got {type(value).__name__}")
    result = []
    for index, item in enumerate(value):
        if not _is_number(item):
            raise ValueError(
                f"per-dim MSE field {key!r} in {path} must contain only numbers; "
                f"item {index} is {type(item).__name__}"
            )
        result.append(float(item))
    return result


def _mse_keys_for_mode(mode: WanActionMode) -> tuple[str, ...]:
    if mode == WanActionMode.DECODED_VIDEO_IDM:
        return (DATASET_ACTION_MSE_KEY, "idm_mse", *MSE_KEYS)
    return MSE_RESULT_KEYS


def _prefixed_mse_keys(keys: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}{key}" for key in keys)


def _metric_family(mode: WanActionMode, metric_name: str | None) -> str:
    if metric_name == DATASET_ACTION_MSE_KEY:
        return "dataset_action_mse"
    if metric_name == "idm_mse" or mode == WanActionMode.DECODED_VIDEO_IDM:
        return "idm_action_mse"
    return "pi05_action_expert_mse"


def _resolve_result_json_path(path: str | Path) -> Path:
    source = Path(path).expanduser()
    if source.is_file():
        return source
    if source.is_dir():
        matches = [source / filename for filename in DIRECTORY_RESULT_FILENAMES if (source / filename).is_file()]
        if len(matches) == 1:
            return matches[0]
        expected = ", ".join(DIRECTORY_RESULT_FILENAMES)
        if not matches:
            raise ValueError(f"output dir {source} does not contain exactly one known result JSON: {expected}")
        found = ", ".join(match.name for match in matches)
        raise ValueError(f"output dir {source} contains multiple candidate result JSON files: {found}")
    raise ValueError(f"result path does not exist: {source}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"failed to read result JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"failed to parse result JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"result JSON must contain an object: {path}")
    return data


def _validate_declared_mode(data: dict[str, Any], *, expected: WanActionMode, path: Path) -> None:
    key, mode_value = _first_present(data, MODE_KEYS)
    if key is None:
        return
    if not isinstance(mode_value, str):
        raise ValueError(f"Wan action mode field {key!r} in {path} must be a string")
    try:
        declared = WanActionMode(mode_value)
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in WanActionMode)
        raise ValueError(f"unknown Wan action mode {mode_value!r} in {path}; expected one of: {valid}") from exc
    if declared != expected:
        raise ValueError(
            f"{path} declares Wan action mode {declared.value!r} via {key!r}, "
            f"but it was supplied as {expected.value!r}"
        )


def _mse_from_mapping(data: dict[str, Any], *, path: Path, keys: tuple[str, ...]) -> MseMetric | None:
    key, value = _first_present(data, keys)
    if key is None:
        return None
    return MseMetric(
        value=_require_number(value, key=key, path=path),
        metric_name=key,
        source_field=key,
    )


def _prefixed_mse_from_mapping(
    data: dict[str, Any], *, path: Path, keys: tuple[str, ...], prefix: str
) -> MseMetric | None:
    key, value = _first_present(data, keys)
    if key is None:
        return None
    return MseMetric(
        value=_require_number(value, key=key, path=path),
        metric_name=key.removeprefix(prefix),
        source_field=key,
    )


def _nested_mse(data: dict[str, Any], *, field: str, path: Path, keys: tuple[str, ...]) -> MseMetric | None:
    if field not in data or data[field] is None:
        return None
    nested = data[field]
    if not isinstance(nested, dict):
        raise ValueError(f"{field!r} in {path} must be an object when present")
    metric = _mse_from_mapping(nested, path=path, keys=keys)
    if metric is None:
        return None
    return MseMetric(
        value=metric.value,
        metric_name=metric.metric_name,
        source_field=f"{field}.{metric.source_field}",
    )


def _history_rows(data: dict[str, Any], *, path: Path) -> list[dict[str, Any]] | None:
    if "history" not in data or data["history"] is None:
        return None
    history = data["history"]
    if not isinstance(history, list):
        raise ValueError(f"'history' in {path} must be a list when present")
    rows = []
    for index, row in enumerate(history):
        if not isinstance(row, dict):
            raise ValueError(f"'history' row {index} in {path} must be an object")
        rows.append(row)
    return rows


def _best_mse(data: dict[str, Any], *, path: Path, mode: WanActionMode) -> MseMetric | None:
    keys = _mse_keys_for_mode(mode)
    explicit = _prefixed_mse_from_mapping(data, path=path, keys=_prefixed_mse_keys(keys, "best_"), prefix="best_")
    if explicit is not None:
        return explicit

    nested = _nested_mse(data, field="best", path=path, keys=keys)
    if nested is not None:
        return nested

    rows = _history_rows(data, path=path)
    if rows:
        candidates = []
        for index, row in enumerate(rows):
            metric = _mse_from_mapping(row, path=path, keys=keys)
            if metric is not None:
                candidates.append(
                    MseMetric(
                        value=metric.value,
                        metric_name=metric.metric_name,
                        source_field=f"history[{index}].{metric.source_field}",
                    )
                )
        if candidates:
            return min(candidates, key=lambda metric: metric.value)

    return _mse_from_mapping(data, path=path, keys=keys)


def _last_mse(data: dict[str, Any], *, path: Path, mode: WanActionMode) -> MseMetric | None:
    keys = _mse_keys_for_mode(mode)
    explicit = _prefixed_mse_from_mapping(data, path=path, keys=_prefixed_mse_keys(keys, "last_"), prefix="last_")
    if explicit is not None:
        return explicit

    for field in ("last", "final"):
        nested = _nested_mse(data, field=field, path=path, keys=keys)
        if nested is not None:
            return nested

    rows = _history_rows(data, path=path)
    if rows:
        metric = _mse_from_mapping(rows[-1], path=path, keys=keys)
        if metric is not None:
            return MseMetric(
                value=metric.value,
                metric_name=metric.metric_name,
                source_field=f"history[{len(rows) - 1}].{metric.source_field}",
            )

    return _mse_from_mapping(data, path=path, keys=keys)


def _per_dim_mse_from_mapping(data: dict[str, Any], *, path: Path) -> list[float] | None:
    key, value = _first_present(data, PER_DIM_MSE_RESULT_KEYS)
    if key is None:
        return None
    return _require_number_list(value, key=key, path=path)


def _per_dim_mse(data: dict[str, Any], *, path: Path) -> list[float] | None:
    direct = _per_dim_mse_from_mapping(data, path=path)
    if direct is not None:
        return direct

    for field in ("best", "final", "last"):
        nested = data.get(field)
        if nested is None:
            continue
        if not isinstance(nested, dict):
            raise ValueError(f"{field!r} in {path} must be an object when present")
        value = _per_dim_mse_from_mapping(nested, path=path)
        if value is not None:
            return value

    rows = _history_rows(data, path=path)
    if rows:
        return _per_dim_mse_from_mapping(rows[-1], path=path)
    return None


def _is_timing_key(key: str) -> bool:
    lower = key.lower()
    if "timestep" in lower:
        return False
    return (
        lower == "timing"
        or lower.startswith("timing_")
        or lower.endswith("_timing")
        or "duration" in lower
        or "elapsed" in lower
        or "seconds" in lower
        or "latency" in lower
        or "throughput" in lower
        or lower.endswith("_time_s")
        or lower.endswith("_time_sec")
        or lower.endswith("_sec")
        or lower.endswith("_secs")
        or lower.endswith("_ms")
        or lower.endswith("_fps")
    )


def _timing_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if _is_timing_key(key)}


def _metadata_field(data: dict[str, Any], key: str) -> tuple[str | None, Any]:
    if key in data:
        return key, data[key]
    for container_key in FINGERPRINT_METADATA_CONTAINERS:
        container = data.get(container_key)
        if isinstance(container, dict) and key in container:
            return f"{container_key}.{key}", container[key]
    return None, None


def _explicit_fingerprint(
    data: dict[str, Any],
    *,
    keys: tuple[str, ...],
    path: Path,
) -> dict[str, Any] | None:
    for key in keys:
        source_field, value = _metadata_field(data, key)
        if source_field is not None:
            return _require_object(value, key=source_field, path=path)
    return None


def _dataset_config_fingerprint(data: dict[str, Any], *, path: Path) -> dict[str, Any] | None:
    source_field, value = _metadata_field(data, "dataset_config")
    if source_field is None or value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"dataset_config field {source_field!r} in {path} must be an object")
    return {"dataset_config": _json_normalized(value)}


def _dataset_fingerprint(data: dict[str, Any], *, path: Path) -> dict[str, Any] | None:
    explicit = _explicit_fingerprint(data, keys=EXPLICIT_DATASET_FINGERPRINT_KEYS, path=path)
    if explicit is not None:
        return explicit
    return _dataset_config_fingerprint(data, path=path)


def _sample_path_fingerprint_value(value: Any, *, key: str, path: Path) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"sample fingerprint field {key!r} in {path} must be a string or null")
    return value


def _sample_count_fingerprint_value(value: Any, *, key: str, path: Path) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"sample fingerprint field {key!r} in {path} must be an integer or null")
    return value


def _derived_sample_fingerprint(
    data: dict[str, Any],
    *,
    path: Path,
    dataset_fingerprint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    fingerprint: dict[str, Any] = {}
    if dataset_fingerprint is not None:
        fingerprint["dataset_fingerprint"] = dataset_fingerprint

    for key in SAMPLE_PATH_FINGERPRINT_KEYS:
        source_field, value = _metadata_field(data, key)
        if source_field is not None:
            fingerprint[key] = _sample_path_fingerprint_value(value, key=source_field, path=path)

    for key in SAMPLE_COUNT_FINGERPRINT_KEYS:
        source_field, value = _metadata_field(data, key)
        if source_field is not None:
            fingerprint[key] = _sample_count_fingerprint_value(value, key=source_field, path=path)

    return fingerprint or None


def _sample_fingerprint(
    data: dict[str, Any],
    *,
    path: Path,
    dataset_fingerprint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    explicit = _explicit_fingerprint(data, keys=EXPLICIT_SAMPLE_FINGERPRINT_KEYS, path=path)
    if explicit is not None:
        return explicit
    return _derived_sample_fingerprint(data, path=path, dataset_fingerprint=dataset_fingerprint)


def _stable_fingerprint_key(fingerprint: dict[str, Any]) -> str:
    return json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))


def _sample_set_compatibility(rows: list[dict[str, Any]]) -> tuple[bool | None, str | None]:
    fingerprints = [row.get("sample_fingerprint") for row in rows]
    if any(fingerprint is None for fingerprint in fingerprints):
        return (
            None,
            "Sample set compatibility is unknown because one or more rows do not expose a sample fingerprint.",
        )

    unique_fingerprints = {_stable_fingerprint_key(fingerprint) for fingerprint in fingerprints}
    if len(unique_fingerprints) == 1:
        return True, None

    metric_families = {row.get("metric_family") for row in rows}
    if len(metric_families) == 1:
        return (
            False,
            "metrics share a family but sample sets differ; compare rows only after aligning sample fingerprints.",
        )
    return False, "Sample sets differ across rows, and the rows also use different metric families."


def _mode_notes(mode: WanActionMode) -> list[str]:
    spec = get_action_mode_spec(mode)
    return [
        f"inference_path={inference_path_for_mode(mode)}",
        f"runs_wan_generation={'yes' if spec.runs_wan_generation else 'no'}",
        f"decoded_action_video={'yes' if spec.generates_video else 'no'}",
        f"reusable_action_memory={'yes' if spec.exposes_reusable_action_memory else 'no'}",
        f"native_wan_kv_cache={'yes' if spec.native_wan_attention_kv_cache else 'no'}",
    ]


def _load_mode_row(mode: WanActionMode, source: str | Path) -> dict[str, Any]:
    path = _resolve_result_json_path(source)
    data = _read_json(path)
    _validate_declared_mode(data, expected=mode, path=path)
    best_mse = _best_mse(data, path=path, mode=mode)
    last_mse = _last_mse(data, path=path, mode=mode)
    primary_metric_name = best_mse.metric_name if best_mse is not None else None
    if primary_metric_name is None and last_mse is not None:
        primary_metric_name = last_mse.metric_name
    dataset_fingerprint = _dataset_fingerprint(data, path=path)
    sample_fingerprint = _sample_fingerprint(data, path=path, dataset_fingerprint=dataset_fingerprint)
    return {
        "mode": mode.value,
        "decoder_arch": extract_decoder_arch(data, path, follow_checkpoint=mode != WanActionMode.DECODED_VIDEO_IDM),
        "inference_path": inference_path_for_mode(mode),
        "true_kv_cache": False,
        "source_path": str(path),
        "metric_name": primary_metric_name,
        "metric_family": _metric_family(mode, primary_metric_name),
        "lower_is_better": True,
        "best_mse": best_mse.value if best_mse is not None else None,
        "last_mse": last_mse.value if last_mse is not None else None,
        "best_mse_metric_name": best_mse.metric_name if best_mse is not None else None,
        "last_mse_metric_name": last_mse.metric_name if last_mse is not None else None,
        "best_mse_source_field": best_mse.source_field if best_mse is not None else None,
        "last_mse_source_field": last_mse.source_field if last_mse is not None else None,
        "per_dim_mse": _per_dim_mse(data, path=path),
        "dataset_fingerprint": dataset_fingerprint,
        "sample_fingerprint": sample_fingerprint,
        "timing": _timing_fields(data),
        "notes": _mode_notes(mode),
    }


def compare_action_modes(
    *,
    decoded_video_idm: str | Path,
    current_wan_prefix_action_expert: str | Path,
    partial_wan_prefix_action_expert: str | Path,
) -> dict[str, Any]:
    """Load the three Wan action-mode result sources and return a normalized JSON-ready comparison."""
    rows = [
        _load_mode_row(WanActionMode.DECODED_VIDEO_IDM, decoded_video_idm),
        _load_mode_row(WanActionMode.CURRENT_WAN_PREFIX_ACTION_EXPERT, current_wan_prefix_action_expert),
        _load_mode_row(WanActionMode.PARTIAL_WAN_PREFIX_ACTION_EXPERT, partial_wan_prefix_action_expert),
    ]
    sample_sets_match, sample_set_warning = _sample_set_compatibility(rows)
    return {
        "comparison_warning": COMPARISON_WARNING,
        "wan_contract_warning": WAN_CONTRACT_WARNING,
        "sample_sets_match": sample_sets_match,
        "sample_set_warning": sample_set_warning,
        "modes": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Wan action-mode result JSONs as a compact matrix.")
    parser.add_argument("--decoded-video-idm", required=True, type=Path, help="Result JSON or output dir for decoded_video_idm.")
    parser.add_argument(
        "--current-wan-prefix-action-expert",
        required=True,
        type=Path,
        help="Result JSON or output dir for current_wan_prefix_action_expert.",
    )
    parser.add_argument(
        "--partial-wan-prefix-action-expert",
        required=True,
        type=Path,
        help="Result JSON or output dir for partial_wan_prefix_action_expert.",
    )
    parser.add_argument("--output-json", type=Path, help="Optional path to write the printed comparison JSON.")
    return parser


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        comparison = compare_action_modes(
            decoded_video_idm=args.decoded_video_idm,
            current_wan_prefix_action_expert=args.current_wan_prefix_action_expert,
            partial_wan_prefix_action_expert=args.partial_wan_prefix_action_expert,
        )
        rendered = json.dumps(comparison, indent=2) + "\n"
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered, encoding="utf-8")
    except ValueError as exc:
        parser.error(str(exc))
    except OSError as exc:
        parser.error(f"failed to write output JSON: {exc}")

    print(rendered, file=out or sys.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
