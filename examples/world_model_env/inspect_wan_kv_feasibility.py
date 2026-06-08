"""Inspect DiffSynth Wan2.2 attention KV-cache feasibility without loading weights."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TextIO

DEFAULT_DIFFSYNTH_REPO_DIR = Path("/tmp/DiffSynth-Studio")
WAN_DIT_RELATIVE_SOURCE = Path("diffsynth/models/wan_video_dit.py")
CURRENT_PREFIX_RELATIVE_SOURCE = Path("world_model/wan_dit_prefix_encoder.py")
DEFAULT_ATTENTION_CLASS_NAMES = ("AttentionModule", "SelfAttention", "CrossAttention")
WAN_ATTENTION_CLASS_NAMES = ("SelfAttention", "CrossAttention")

QUERY_NAMES = {"q", "query", "queries"}
KEY_NAMES = {"k", "key", "keys"}
VALUE_NAMES = {"v", "value", "values"}
KV_CACHE_NAMES = {
    "cache",
    "kv",
    "kv_cache",
    "key_cache",
    "value_cache",
    "key_value_cache",
    "key_values",
    "past_key_values",
}
RETURN_KV_FLAG_NAMES = {
    "return_kv",
    "return_kv_cache",
    "return_key_value",
    "return_key_values",
    "output_kv",
    "output_key_values",
}

WAN_DYNAMIC_HIDDEN_DEPENDENCIES = ("timestep", "noise_latents", "future_latents")
WAN_TEXT_DEPENDENCIES = ("text",)
WAN_CURRENT_IMAGE_DEPENDENCIES = ("current_image",)


def _path_string(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def candidate_wan_dit_source_paths(
    *,
    diffsynth_repo_dir: str | Path | None = DEFAULT_DIFFSYNTH_REPO_DIR,
    source_path: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return likely Wan DiT source paths, ordered by reproducibility."""

    candidates: list[Path] = []
    if source_path is not None:
        candidates.append(Path(source_path).expanduser())
    if diffsynth_repo_dir is not None:
        candidates.append(Path(diffsynth_repo_dir).expanduser() / WAN_DIT_RELATIVE_SOURCE)
    for entry in sys.path:
        if entry:
            candidates.append(Path(entry).expanduser() / WAN_DIT_RELATIVE_SOURCE)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return tuple(unique)


def resolve_wan_dit_source(
    *,
    diffsynth_repo_dir: str | Path | None = DEFAULT_DIFFSYNTH_REPO_DIR,
    source_path: str | Path | None = None,
) -> Path | None:
    """Resolve a local DiffSynth Wan DiT source file if one is available."""

    for candidate in candidate_wan_dit_source_paths(
        diffsynth_repo_dir=diffsynth_repo_dir,
        source_path=source_path,
    ):
        if candidate.is_file():
            return candidate
    return None


def _function_parameters(function: ast.FunctionDef | ast.AsyncFunctionDef | None) -> list[str]:
    if function is None:
        return []
    parameters: list[str] = []
    all_args = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    parameters.extend(argument.arg for argument in all_args)
    if function.args.vararg is not None:
        parameters.append("*" + function.args.vararg.arg)
    if function.args.kwarg is not None:
        parameters.append("**" + function.args.kwarg.arg)
    return parameters


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        return [target.attr]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, ast.Tuple | ast.List):
        names: list[str] = []
        for element in target.elts:
            names.extend(_target_names(element))
        return names
    return []


def _assigned_names(function: ast.FunctionDef | ast.AsyncFunctionDef | None) -> list[str]:
    if function is None:
        return []
    names: list[str] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.extend(_target_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.extend(_target_names(node.target))
        elif isinstance(node, ast.AugAssign):
            names.extend(_target_names(node.target))
    return names


def _is_query_name(name: str) -> bool:
    return name in QUERY_NAMES or name.endswith("_q") or name.endswith("_query")


def _is_key_name(name: str) -> bool:
    return name in KEY_NAMES or name.endswith("_k") or name.endswith("_key")


def _is_value_name(name: str) -> bool:
    return name in VALUE_NAMES or name.endswith("_v") or name.endswith("_value")


def _name_loads(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _dict_key_strings(node: ast.AST | None) -> set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    keys: set[str] = set()
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
    return keys


def _return_contains_key_value(return_value: ast.AST | None) -> bool:
    names = _name_loads(return_value)
    lower_names = {name.lower() for name in names}
    dict_keys = {key.lower() for key in _dict_key_strings(return_value)}
    combined = lower_names | dict_keys
    if combined & KV_CACHE_NAMES:
        return True
    has_key = bool(combined & KEY_NAMES)
    has_value = bool(combined & VALUE_NAMES)
    return has_key and has_value


def _return_expressions(function: ast.FunctionDef | ast.AsyncFunctionDef | None) -> list[str]:
    if function is None:
        return []
    expressions: list[str] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Return):
            expressions.append("None" if node.value is None else ast.unparse(node.value))
    return expressions


def _call_attribute_names(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            names.append(child.func.attr)
    return names


def _projection_role(name: str) -> str | None:
    if _is_query_name(name):
        return "query"
    if _is_key_name(name):
        return "key"
    if _is_value_name(name):
        return "value"
    return None


def _projection_dependencies(*, role: str, loaded_names: set[str], projection_names: set[str]) -> tuple[str, ...]:
    if "img" in loaded_names or any(name.endswith("_img") for name in projection_names):
        return WAN_CURRENT_IMAGE_DEPENDENCIES
    if "ctx" in loaded_names or "y" in loaded_names:
        return WAN_TEXT_DEPENDENCIES
    if role == "query" or "x" in loaded_names or "freqs" in loaded_names:
        return WAN_DYNAMIC_HIDDEN_DEPENDENCIES
    return WAN_DYNAMIC_HIDDEN_DEPENDENCIES


def _projection_lifetime(dependencies: Sequence[str]) -> str:
    if set(dependencies) <= {"current_image", "text"}:
        return "static_current_image_text"
    return "dynamic_timestep_noise_or_action"


def _projection_cache_note(*, role: str, dependencies: Sequence[str]) -> str:
    if role == "query":
        return "Query projection is consumed by the current forward pass and is not a reusable K/V cache entry."
    if _projection_lifetime(dependencies) == "static_current_image_text":
        return "Projected K/V is cacheable for the same text/current-image context if a wrapper exposes it."
    return "Projected K/V is cacheable only for the exact dynamic hidden states, timestep/noise state, and positions."


def _projection_assignment_summaries(
    function: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> list[dict[str, Any]]:
    if function is None:
        return []
    projections: list[dict[str, Any]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        call_attrs = _call_attribute_names(value)
        projection_attrs = {
            attr
            for attr in call_attrs
            if attr in {"q", "k", "v", "k_img", "v_img", "norm_q", "norm_k", "norm_k_img", "rope_apply"}
        }
        target_names = [name for target in targets for name in _target_names(target)]
        loaded_names = _name_loads(value)
        for target_name in target_names:
            role = _projection_role(target_name)
            if role is None:
                continue
            if not projection_attrs and "rope_apply" not in call_attrs:
                continue
            dependencies = _projection_dependencies(
                role=role,
                loaded_names=loaded_names,
                projection_names=projection_attrs | {target_name},
            )
            projections.append(
                {
                    "target": target_name,
                    "role": role,
                    "line": int(node.lineno),
                    "expression": ast.unparse(value),
                    "projection_functions": tuple(sorted(projection_attrs)),
                    "source_names": tuple(sorted(loaded_names - {"self"})),
                    "cacheable_as_kv": role in {"key", "value"},
                    "dependencies": dependencies,
                    "cache_lifetime": _projection_lifetime(dependencies),
                    "note": _projection_cache_note(role=role, dependencies=dependencies),
                }
            )
    return projections


def _returns_key_value_tensors(function: ast.FunctionDef | ast.AsyncFunctionDef | None) -> bool:
    if function is None:
        return False
    return any(_return_contains_key_value(node.value) for node in ast.walk(function) if isinstance(node, ast.Return))


def _find_method(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for child in class_node.body:
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and child.name == method_name:
            return child
    return None


def _kv_related_methods(class_node: ast.ClassDef) -> list[dict[str, Any]]:
    methods: list[dict[str, Any]] = []
    for child in class_node.body:
        if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        parameters = [parameter.lower().lstrip("*") for parameter in _function_parameters(child)]
        has_kv_name = "kv" in child.name.lower()
        has_kv_parameter = bool(set(parameters) & (KV_CACHE_NAMES | RETURN_KV_FLAG_NAMES))
        if has_kv_name or has_kv_parameter:
            methods.append(
                {
                    "name": child.name,
                    "line": int(child.lineno),
                    "parameters": _function_parameters(child),
                    "returns_key_value_tensors": _returns_key_value_tensors(child),
                }
            )
    return methods


def _qkv_assignment_summary(assigned_names: Iterable[str]) -> dict[str, Any]:
    names = sorted(set(assigned_names))
    qkv_names = sorted(name for name in names if _is_query_name(name) or _is_key_name(name) or _is_value_name(name))
    return {
        "computes_query": any(_is_query_name(name) for name in names),
        "computes_key": any(_is_key_name(name) for name in names),
        "computes_value": any(_is_value_name(name) for name in names),
        "computed_in_forward": all(
            (
                any(_is_query_name(name) for name in names),
                any(_is_key_name(name) for name in names),
                any(_is_value_name(name) for name in names),
            )
        ),
        "assigned_qkv_like_names": qkv_names,
    }


def _inspect_attention_class(class_node: ast.ClassDef, *, source_path: str | Path) -> dict[str, Any]:
    forward = _find_method(class_node, "forward")
    parameters = _function_parameters(forward)
    public_parameters = [parameter.lstrip("*") for parameter in parameters if parameter != "self"]
    lower_parameters = {parameter.lower() for parameter in public_parameters}
    assigned_names = _assigned_names(forward)
    accepts_qkv_inputs = bool({"q", "k", "v"} <= lower_parameters)
    accepts_kv_cache = bool(lower_parameters & KV_CACHE_NAMES)
    has_return_kv_flag = bool(lower_parameters & RETURN_KV_FLAG_NAMES)
    projection_summaries = _projection_assignment_summaries(forward)
    return {
        "class_name": class_node.name,
        "source_path": _path_string(source_path),
        "line": int(class_node.lineno),
        "forward": {
            "line": None if forward is None else int(forward.lineno),
            "parameters": parameters,
            "accepts_query_key_value_tensors": accepts_qkv_inputs,
            "accepts_key_value_cache": accepts_kv_cache,
            "has_return_key_value_flag": has_return_kv_flag,
            "returns_key_value_tensors": _returns_key_value_tensors(forward),
            "return_expressions": _return_expressions(forward),
        },
        "qkv": _qkv_assignment_summary(assigned_names),
        "projection_cacheability": {
            "cacheable_projection_targets": tuple(
                summary["target"] for summary in projection_summaries if summary["cacheable_as_kv"]
            ),
            "static_cacheable_projection_targets": tuple(
                summary["target"]
                for summary in projection_summaries
                if summary["cacheable_as_kv"] and summary["cache_lifetime"] == "static_current_image_text"
            ),
            "dynamic_cacheable_projection_targets": tuple(
                summary["target"]
                for summary in projection_summaries
                if summary["cacheable_as_kv"] and summary["cache_lifetime"] != "static_current_image_text"
            ),
            "projections": projection_summaries,
        },
        "kv_related_methods": _kv_related_methods(class_node),
    }


def inspect_attention_source(
    source: str,
    *,
    source_path: str | Path,
    class_names: Sequence[str] | None = DEFAULT_ATTENTION_CLASS_NAMES,
) -> list[dict[str, Any]]:
    """Inspect attention classes in a Python source string with AST only."""

    tree = ast.parse(source)
    requested = None if class_names is None else set(class_names)
    modules: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if requested is not None and node.name not in requested:
            continue
        if requested is None and "Attention" not in node.name:
            continue
        modules.append(_inspect_attention_class(node, source_path=source_path))
    return modules


def inspect_current_prefix_capture(
    source: str,
    *,
    source_path: str | Path,
) -> dict[str, Any]:
    """Summarize the repo's current Wan hidden-prefix capture path."""

    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    extractor = classes.get("WanDiTHiddenFeatureExtractor")
    frozen_encoder = classes.get("FrozenDiffSynthWanDiTCurrentPrefixEncoder")
    source_contains_hooks = "register_forward_hook" in source
    source_contains_prefix_tokens = "prefix_tokens" in source
    source_mentions_kv = any(token in source for token in ("kv_cache", "return_kv", "past_key_values"))
    return {
        "available": extractor is not None,
        "source_path": _path_string(source_path),
        "capture_mode": "hidden_state_prefix_tokens",
        "class_name": "WanDiTHiddenFeatureExtractor" if extractor is not None else None,
        "class_line": None if extractor is None else int(extractor.lineno),
        "current_prefix_encoder_class": "FrozenDiffSynthWanDiTCurrentPrefixEncoder"
        if frozen_encoder is not None
        else None,
        "current_prefix_encoder_line": None if frozen_encoder is None else int(frozen_encoder.lineno),
        "uses_forward_hooks": source_contains_hooks,
        "returns_prefix_tokens": source_contains_prefix_tokens,
        "captures_key_value_tensors": False,
        "kv_cache_terms_present": source_mentions_kv,
        "notes": [
            "WanDiTHiddenFeatureExtractor registers forward hooks on selected DiT blocks and pools block outputs.",
            "The current prefix path returns hidden-state prefix tokens for the pi0.5-style action expert.",
            "No current repo helper returns Wan attention key/value tensors.",
        ],
    }


def _inspect_current_prefix_source(script_dir: Path) -> dict[str, Any]:
    source_path = script_dir / CURRENT_PREFIX_RELATIVE_SOURCE
    if not source_path.is_file():
        return {
            "available": False,
            "source_path": _path_string(source_path),
            "capture_mode": "hidden_state_prefix_tokens",
            "captures_key_value_tensors": False,
            "notes": ["Current prefix source file was not found."],
        }
    return inspect_current_prefix_capture(_read_text(source_path), source_path=source_path)


def _public_wan_attention_modules(attention_modules: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [module for module in attention_modules if module.get("class_name") in WAN_ATTENTION_CLASS_NAMES]


def _forward_kv_available(attention_modules: Sequence[dict[str, Any]]) -> bool:
    public_modules = _public_wan_attention_modules(attention_modules)
    return any(
        bool(module["forward"]["returns_key_value_tensors"])
        or bool(module["forward"]["accepts_key_value_cache"])
        or bool(module["forward"]["has_return_key_value_flag"])
        for module in public_modules
    )


def build_tensor_feasibility(attention_modules: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build the static-vs-dynamic tensor summary for the Wan current-prefix use case."""

    forward_kv_available = _forward_kv_available(attention_modules)
    return {
        "static_tensors_for_current_image_text": [
            {
                "name": "current_rgb_image",
                "source": "input observation image",
                "static_for": "same current frame",
                "cache_note": "Usable as current-only prefix input; not an attention KV tensor.",
            },
            {
                "name": "first_frame_vae_latents",
                "source": "Wan VAE encode(current_rgb_image)",
                "static_for": "same current frame and VAE settings",
                "cache_note": "Current repo can reuse this through hidden-prefix tokens, not through Wan KV.",
            },
            {
                "name": "prompt_token_ids_mask_and_text_context",
                "source": "Wan tokenizer plus T5 text encoder",
                "static_for": "same prompt text",
                "cache_note": "Cross-attention K/V over text would be static in principle, but DiffSynth CrossAttention computes it internally.",
            },
            {
                "name": "rotary_frequencies_for_latent_grid",
                "source": "Wan DiT precomputed freqs sliced to (frames, height, width)",
                "static_for": "same latent grid shape",
                "cache_note": "Position tensor is reusable; it is not sufficient for KV reuse.",
            },
        ],
        "timestep_noise_dependent_tensors": [
            {
                "name": "timestep_embedding_and_t_mod",
                "depends_on": ["timestep", "separated timestep path", "latent frame count"],
                "cache_note": "Changes across denoising steps and modulates every DiT block.",
            },
            {
                "name": "block_hidden_states",
                "depends_on": ["current/noisy latents", "timestep modulation", "previous DiT blocks"],
                "cache_note": "Current WanDiTHiddenFeatureExtractor captures pooled versions of these as prefix tokens.",
            },
            {
                "name": "self_attention_qkv",
                "depends_on": ["block hidden states", "rotary frequencies"],
                "cache_note": "SelfAttention computes q/k/v internally and returns only projected hidden states.",
            },
            {
                "name": "cross_attention_query",
                "depends_on": ["block hidden states", "timestep-conditioned residual path"],
                "cache_note": "The query side changes with the denoising state.",
            },
            {
                "name": "cross_attention_text_key_value",
                "depends_on": ["prompt text context"],
                "cache_note": "Static for the same prompt in principle, but unavailable without a cache-aware CrossAttention wrapper.",
            },
        ],
        "attention_projection_cacheability": {
            module["class_name"]: module.get("projection_cacheability", {}) for module in attention_modules
        },
        "forward_exposes_or_returns_wan_key_value_tensors": forward_kv_available,
    }


def build_conclusion(
    *,
    attention_modules: Sequence[dict[str, Any]],
    current_prefix_capture: dict[str, Any],
) -> dict[str, Any]:
    public_modules = _public_wan_attention_modules(attention_modules)
    computes_internal_qkv = bool(public_modules) and all(
        bool(module["qkv"]["computed_in_forward"]) for module in public_modules
    )
    forward_kv_available = _forward_kv_available(attention_modules)
    return {
        "recommended_current_mode": "hidden_prefix",
        "true_kv_feasibility": "not_available_in_current_path"
        if not forward_kv_available
        else "source_exposes_some_kv_hooks",
        "summary": (
            "DiffSynth Wan SelfAttention/CrossAttention compute q/k/v internally but their public forward paths "
            "do not return key/value tensors; this repo currently captures pooled DiT hidden states via "
            "WanDiTHiddenFeatureExtractor, not KV."
        ),
        "internal_qkv_detected_for_self_and_cross_attention": computes_internal_qkv,
        "current_repo_captures_key_value_tensors": bool(current_prefix_capture.get("captures_key_value_tensors")),
        "recommendation": (
            "Keep hidden-prefix as the reproducible current mode. Treat true Wan KV caching as a separate "
            "experimental branch requiring cache-aware SelfAttention/CrossAttention wrappers plus an action-token "
            "branch/context contract."
        ),
        "serving_training_behavior_changed": False,
    }


def build_probe_report(
    *,
    diffsynth_repo_dir: str | Path | None = DEFAULT_DIFFSYNTH_REPO_DIR,
    wan_dit_source_path: str | Path | None = None,
    cheap: bool = True,
    script_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a reproducible Wan KV feasibility report without loading model weights."""

    if not cheap:
        raise ValueError(
            "Only cheap source probing is implemented; model/pipeline weight loading is intentionally absent."
        )

    resolved_script_dir = (
        Path(script_dir).expanduser().resolve() if script_dir is not None else Path(__file__).resolve().parent
    )
    source_path = resolve_wan_dit_source(
        diffsynth_repo_dir=diffsynth_repo_dir,
        source_path=wan_dit_source_path,
    )
    if source_path is None:
        attention_modules: list[dict[str, Any]] = []
        source_status = {
            "available": False,
            "path": None,
            "searched_paths": [
                str(path)
                for path in candidate_wan_dit_source_paths(
                    diffsynth_repo_dir=diffsynth_repo_dir,
                    source_path=wan_dit_source_path,
                )
            ],
        }
    else:
        attention_modules = inspect_attention_source(_read_text(source_path), source_path=source_path)
        source_status = {
            "available": True,
            "path": _path_string(source_path),
            "searched_paths": [
                str(path)
                for path in candidate_wan_dit_source_paths(
                    diffsynth_repo_dir=diffsynth_repo_dir,
                    source_path=wan_dit_source_path,
                )
            ],
        }

    current_prefix_capture = _inspect_current_prefix_source(resolved_script_dir)
    inspected_wan_attention_classes = tuple(
        module["class_name"] for module in attention_modules if module.get("class_name") in WAN_ATTENTION_CLASS_NAMES
    )
    missing_wan_attention_classes = tuple(
        class_name for class_name in WAN_ATTENTION_CLASS_NAMES if class_name not in inspected_wan_attention_classes
    )
    return {
        "schema_version": 1,
        "probe": {
            "name": "wan2.2_attention_kv_feasibility",
            "mode": "cheap_source",
            "loads_model_weights": False,
            "loads_diffsynth_pipeline": False,
            "diffsynth_repo_dir": _path_string(diffsynth_repo_dir),
            "wan_dit_source": source_status,
            "current_repo_source": {
                "path": current_prefix_capture.get("source_path"),
                "available": current_prefix_capture.get("available", False),
            },
            "wan_attention_classes": {
                "available": not missing_wan_attention_classes,
                "inspected": inspected_wan_attention_classes,
                "missing": missing_wan_attention_classes,
            },
        },
        "attention_modules": attention_modules,
        "current_repo_capture": current_prefix_capture,
        "tensor_feasibility": build_tensor_feasibility(attention_modules),
        "conclusion": build_conclusion(
            attention_modules=attention_modules,
            current_prefix_capture=current_prefix_capture,
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Wan2.2 attention KV-cache feasibility without loading DiffSynth model weights."
    )
    parser.add_argument(
        "--diffsynth-repo-dir",
        default=str(DEFAULT_DIFFSYNTH_REPO_DIR),
        help="Local DiffSynth-Studio checkout to inspect.",
    )
    parser.add_argument(
        "--wan-dit-source-path",
        default=None,
        help="Direct path to diffsynth/models/wan_video_dit.py. Overrides source discovery when present.",
    )
    parser.add_argument(
        "--cheap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use source/AST inspection only. This is the default and never loads model weights.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to also write the JSON report.",
    )
    return parser


def _dump_json(report: dict[str, Any], out: TextIO) -> None:
    print(json.dumps(report, indent=2, sort_keys=True), file=out)


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None, err: TextIO | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output = out if out is not None else sys.stdout
    error_output = err if err is not None else sys.stderr
    try:
        report = build_probe_report(
            diffsynth_repo_dir=args.diffsynth_repo_dir,
            wan_dit_source_path=args.wan_dit_source_path,
            cheap=args.cheap,
        )
        if args.output_json:
            output_path = Path(args.output_json).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _dump_json(report, output)
    except Exception as exc:  # pragma: no cover - argparse-style CLI boundary
        print(f"error: {exc}", file=error_output)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
