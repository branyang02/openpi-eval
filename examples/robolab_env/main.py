from __future__ import annotations

import contextlib
import dataclasses
import importlib.abc
import importlib.machinery
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator, Literal, Optional

import torch
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RECORDER_MODULE = "robolab.core.logging.recorder_manager"
_DEFAULT_TASK_DIRS = ["benchmark"]
_DEFAULT_TASKS = ["BananaInBowlTask"]

PolicyVariant = Literal["pi0", "pi0_fast", "pi05", "paligemma", "paligemma_fast"]
VideoMode = Literal["all", "viewport", "sensor", "none"]


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # Pi0-family policy variant. Must match the checkpoint served by
    # scripts/serve_policy.py.
    policy: PolicyVariant = "pi05"
    # RoboLab task class names. main.py can run one or more tasks; eval_all.py
    # launches one main.py subprocess per task for a full task set.
    task: list[str] = dataclasses.field(default_factory=lambda: list(_DEFAULT_TASKS))
    # RoboLab task subdirectories to register. The release tasks live in
    # robolab/tasks/benchmark.
    task_dirs: list[str] = dataclasses.field(
        default_factory=lambda: list(_DEFAULT_TASK_DIRS)
    )
    # Optional RoboLab task tags. If set and --task is empty, RoboLab resolves
    # registered tasks by tag.
    tag: list[str] = dataclasses.field(default_factory=list)

    # Number of vectorized Isaac environments per task.
    num_envs: int = 1
    # Number of sequential vectorized batches per task. Total episodes per task
    # is num_envs * num_runs unless adaptive sampling is enabled.
    num_runs: int = 1
    # Adaptive episode cap. When set, RoboLab ignores num_runs and keeps adding
    # num_envs-sized batches until the confidence interval target is reached or
    # this cap is hit.
    num_episodes_adaptive: Optional[int] = None
    # Target width for RoboLab's 95% Beta credible interval when adaptive
    # sampling is enabled.
    ci_pp_width: float = 0.14

    # Number of predicted actions to execute before requesting a new chunk. If
    # omitted, RoboLab uses the per-policy default.
    open_loop_horizon: Optional[int] = None
    # Full WebSocket URI for remote serving. Overrides host/port when set.
    remote_uri: Optional[str] = None

    # Absolute or relative output directory for this run. RoboLab's runner calls
    # the underlying flag "output-folder-name"; absolute values are honored and
    # relative values are resolved from the user's shell cwd.
    output_dir: Optional[str] = None

    instruction_type: str = "default"
    video_mode: VideoMode = "all"
    headless: bool = True
    device: str = "cuda:0"
    enable_subtask: bool = False
    enable_verbose: bool = False
    enable_debug: bool = False
    record_image_data: bool = False
    randomize_background: bool = False
    background_seed: Optional[int] = None


def _robolab_root(repo_root: Path = _REPO_ROOT) -> Path:
    return repo_root / "third_party" / "robolab"


def _robolab_runner(repo_root: Path = _REPO_ROOT) -> Path:
    return _robolab_root(repo_root) / "policies" / "pi0_family" / "run.py"


def _slice_uint16_tensor(value: torch.Tensor, env_ids):
    if isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.detach().cpu().numpy()
    else:
        env_ids = list(env_ids)
    return torch.from_numpy(value.detach().cpu().numpy()[env_ids]).to(
        device=value.device
    )


def _patch_recorder_module(module: ModuleType) -> None:
    if getattr(module, "_openpi_eval_uint16_patch", False):
        return

    def _slice_to_envs(value, env_ids):
        if isinstance(value, dict):
            return {key: _slice_to_envs(item, env_ids) for key, item in value.items()}
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.uint16:
                return _slice_uint16_tensor(value, env_ids)
            return value[env_ids]
        return value

    module._slice_to_envs = _slice_to_envs
    module._openpi_eval_uint16_patch = True


class _RecorderPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec):  # noqa: ANN001
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)
        return None

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _patch_recorder_module(module)


class _RecorderPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # noqa: ANN001
        if fullname != _RECORDER_MODULE:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if isinstance(spec.loader, _RecorderPatchLoader):
            return spec

        spec.loader = _RecorderPatchLoader(spec.loader)
        return spec


def _install_recorder_uint16_patch() -> None:
    module = sys.modules.get(_RECORDER_MODULE)
    if module is not None:
        _patch_recorder_module(module)
        return

    if not any(isinstance(finder, _RecorderPatchFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _RecorderPatchFinder())


def _extend_flag(argv: list[str], flag: str, values: list[str]) -> None:
    if values:
        argv.append(flag)
        argv.extend(values)


def _resolve_output_folder_name(output_dir: str | None) -> str | None:
    if output_dir is None:
        return None
    return os.path.abspath(output_dir)


def _validate_args(args: Args) -> None:
    if args.num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {args.num_envs}")
    if args.num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {args.num_runs}")
    if args.num_episodes_adaptive is not None and args.num_episodes_adaptive < 1:
        raise ValueError(
            "num_episodes_adaptive must be >= 1 when set, "
            f"got {args.num_episodes_adaptive}"
        )
    if not (0.0 < args.ci_pp_width <= 1.0):
        raise ValueError(f"ci_pp_width must be in (0, 1], got {args.ci_pp_width}")
    if not args.task and not args.tag:
        raise ValueError("Provide at least one --task or --tag.")


def _build_runner_argv(args: Args, runner: Path | None = None) -> list[str]:
    """Translate this example's typed Args into RoboLab's argparse runner CLI."""
    _validate_args(args)
    runner = runner or _robolab_runner()

    argv = [
        str(runner),
        "--policy",
        args.policy,
        "--remote-host",
        args.host,
        "--remote-port",
        str(args.port),
        "--num-envs",
        str(args.num_envs),
        "--num-runs",
        str(args.num_runs),
        "--ci-pp-width",
        str(args.ci_pp_width),
        "--instruction-type",
        args.instruction_type,
        "--video-mode",
        args.video_mode,
        "--device",
        args.device,
    ]

    _extend_flag(argv, "--task", list(args.task))
    _extend_flag(argv, "--tag", list(args.tag))
    _extend_flag(argv, "--task-dirs", list(args.task_dirs))

    if args.remote_uri is not None:
        argv.extend(["--remote-uri", args.remote_uri])
    if args.open_loop_horizon is not None:
        argv.extend(["--open-loop-horizon", str(args.open_loop_horizon)])
    if args.num_episodes_adaptive is not None:
        argv.extend(["--num-episodes-adaptive", str(args.num_episodes_adaptive)])
    output_folder_name = _resolve_output_folder_name(args.output_dir)
    if output_folder_name is not None:
        argv.extend(["--output-folder-name", output_folder_name])
    if args.headless:
        argv.append("--headless")
    if args.enable_subtask:
        argv.append("--enable-subtask")
    if args.enable_verbose:
        argv.append("--enable-verbose")
    if args.enable_debug:
        argv.append("--enable-debug")
    if args.record_image_data:
        argv.append("--record-image-data")
    if args.randomize_background:
        argv.append("--randomize-background")
    if args.background_seed is not None:
        argv.extend(["--background-seed", str(args.background_seed)])

    return argv


@contextlib.contextmanager
def _temporary_argv(argv: list[str]) -> Iterator[None]:
    original = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = original


@contextlib.contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    path_str = str(path)
    inserted = path_str not in sys.path
    if inserted:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(path_str)


def run_robolab(args: Args, repo_root: Path = _REPO_ROOT) -> None:
    robolab_root = _robolab_root(repo_root)
    runner = _robolab_runner(repo_root)
    if not runner.exists():
        raise SystemExit(
            "RoboLab submodule is missing. Run: "
            "git submodule update --init --recursive third_party/robolab"
        )

    argv = _build_runner_argv(args, runner)
    _install_recorder_uint16_patch()
    with _temporary_sys_path(robolab_root), _temporary_argv(argv):
        runpy.run_path(str(runner), run_name="__main__")


def main(args: Args) -> None:
    run_robolab(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
