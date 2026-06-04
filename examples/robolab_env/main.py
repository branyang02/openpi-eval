from __future__ import annotations

import contextlib
import dataclasses
import importlib.abc
import importlib.machinery
import json
import os
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

    # Absolute or relative top-level output directory for this run. If None,
    # defaults to examples/robolab_env/output/<policy>/, matching the other
    # simulator clients' example-local output roots. RoboLab's runner calls the
    # underlying flag "output-folder-name"; absolute values are honored and
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


def _default_output_dir(args: Args) -> str:
    return str(Path(__file__).resolve().parent / "output" / args.policy)


def _resolve_output_folder_name(args: Args) -> str:
    return os.path.abspath(
        args.output_dir if args.output_dir is not None else _default_output_dir(args)
    )


def _load_existing_episode_results(output_dir: str) -> list[dict]:
    jsonl_path = os.path.join(output_dir, "episode_results.jsonl")
    json_path = os.path.join(output_dir, "episode_results.json")

    if os.path.exists(jsonl_path):
        episodes: list[dict] = []
        with open(jsonl_path) as file_handle:
            for line in file_handle:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        return episodes

    if os.path.exists(json_path):
        with open(json_path) as file_handle:
            data = json.load(file_handle)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]

    return []


def _ensure_output_dir_policy_compatible(output_dir: str, policy: str) -> None:
    policies = {
        episode.get("policy")
        for episode in _load_existing_episode_results(output_dir)
        if episode.get("policy")
    }
    incompatible = sorted(existing for existing in policies if existing != policy)
    if incompatible:
        raise ValueError(
            f"Output directory {output_dir!r} contains RoboLab results from "
            f"policy {', '.join(incompatible)!r}. Choose a fresh --output-dir "
            f"or use the matching --policy {next(iter(incompatible))!r}."
        )


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


def _launch_simulation_app(args: Args):
    import cv2  # noqa: F401 -- must import this before isaaclab.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        {
            "headless": args.headless,
            "enable_cameras": True,
            "device": args.device,
        }
    )
    return launcher.app


def _configure_robolab_runtime(args: Args) -> None:
    import robolab.constants

    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args.enable_subtask
    robolab.constants.RECORD_IMAGE_DATA = args.record_image_data
    robolab.constants.VERBOSE = args.enable_verbose
    robolab.constants.DEBUG = args.enable_debug


def _register_task_envs(args: Args) -> None:
    from robolab.registrations.droid.auto_env_registrations_jointpos import (
        auto_register_droid_envs,
    )

    auto_register_droid_envs(
        task_dirs=args.task_dirs,
        task=args.task,
        randomize_background=args.randomize_background,
        background_seed=args.background_seed,
    )


def _client_kwargs(args: Args) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "remote_host": args.host,
        "remote_port": args.port,
        "remote_uri": args.remote_uri,
        "open_loop_horizon": args.open_loop_horizon,
        "policy_variant": args.policy,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def make_client(args: Args):
    from policies.pi0_family.client import Pi0DroidJointposClient

    return Pi0DroidJointposClient(**_client_kwargs(args))


def _resolve_task_envs(args: Args) -> tuple[list[str], str]:
    from robolab.core.environments.factory import get_envs

    if args.task:
        task_envs = get_envs(task=args.task)
        filter_str = f"tasks: {', '.join(args.task)}"
    elif args.tag:
        task_envs = get_envs(tag=args.tag)
        filter_str = f"tags: {', '.join(args.tag)}"
    else:
        task_envs = get_envs()
        filter_str = "all"

    return task_envs, filter_str


def _total_episodes(args: Args) -> int:
    return (
        args.num_episodes_adaptive
        if args.num_episodes_adaptive is not None
        else args.num_runs * args.num_envs
    )


def _run_task_set(args: Args) -> None:
    import robolab.constants
    from robolab.constants import set_output_dir
    from robolab.core.environments.runtime import create_env
    from robolab.core.logging.results import (
        check_all_episodes_complete,
        check_run_complete,
        init_experiment,
        summarize_experiment_results,
    )
    from robolab.core.utils.adaptive_sampling import (
        count_task_episodes,
        should_continue_sampling,
    )
    from robolab.core.utils.print_utils import print_experiment_summary
    from robolab.eval.episode import run_episode
    from robolab.eval.summarize import summarize_run

    output_dir = _resolve_output_folder_name(args)
    os.makedirs(output_dir, exist_ok=True)

    task_envs, filter_str = _resolve_task_envs(args)
    total_episodes = _total_episodes(args)
    adaptive_max = args.num_episodes_adaptive
    is_adaptive = adaptive_max is not None

    print_experiment_summary(
        task_envs=task_envs,
        filter_str=filter_str,
        num_envs=args.num_envs,
        num_episodes=total_episodes,
        policy=args.policy,
        instruction_type=args.instruction_type,
        output_dir=output_dir,
    )

    episode_results_file, episode_results = init_experiment(output_dir)
    save_videos = args.video_mode != "none"

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        if check_all_episodes_complete(
            episode_results=episode_results,
            env_name=task_env,
            num_episodes=total_episodes,
        ):
            print(f"\033[96m[RoboLab] Task `{task_env}` already done. Skipping.\033[0m")
            continue

        env, env_cfg = create_env(
            task_env,
            device=args.device,
            num_envs=args.num_envs,
            instruction_type=args.instruction_type,
            policy=args.policy,
        )
        client = make_client(args)

        try:
            run_idx = 0
            while True:
                if is_adaptive:
                    k_so_far, n_so_far = count_task_episodes(episode_results, task_env)
                    if not should_continue_sampling(
                        k=k_so_far,
                        n=n_so_far,
                        target_width=args.ci_pp_width,
                        n_max=adaptive_max,
                    ):
                        print(
                            f"\033[96m[RoboLab] Task `{task_env}` adaptive stop at {n_so_far} "
                            f"episodes ({k_so_far}/{n_so_far} success).\033[0m"
                        )
                        break
                elif run_idx >= args.num_runs:
                    break

                run_episode_ids = [
                    run_idx * args.num_envs + eid for eid in range(args.num_envs)
                ]
                if all(
                    check_run_complete(
                        episode_results=episode_results,
                        env_name=task_env,
                        episode=ep_id,
                    )
                    for ep_id in run_episode_ids
                ):
                    print(
                        f"\033[96m[RoboLab] Task `{task_env}` run `{run_idx}` already done. Skipping.\033[0m"
                    )
                    run_idx += 1
                    continue

                if args.instruction_type != "default":
                    run_name = f"{task_env}_{args.instruction_type}_{run_idx}"
                else:
                    run_name = f"{task_env}_{run_idx}"
                print(
                    f"\033[96m[RoboLab] Running {run_name}: '{env_cfg.instruction}' "
                    f"(run {run_idx}, {args.num_envs} envs)\033[0m"
                )

                env_results, msgs, timing = run_episode(
                    env=env,
                    env_cfg=env_cfg,
                    episode=run_idx,
                    client=client,
                    save_videos=save_videos,
                    video_mode=args.video_mode,
                    headless=args.headless,
                )

                episode_results = summarize_run(
                    env_results=env_results,
                    msgs=msgs,
                    env=env,
                    env_cfg=env_cfg,
                    num_envs=args.num_envs,
                    run_idx=run_idx,
                    run_name=run_name,
                    task_env=task_env,
                    scene_output_dir=scene_output_dir,
                    policy=args.policy,
                    episode_results=episode_results,
                    episode_results_file=episode_results_file,
                    enable_subtask_progress=robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING,
                    timing=timing,
                    instruction_type=args.instruction_type,
                )

                env.reset_eval_state()
                run_idx += 1
        finally:
            env.close()

    summarize_experiment_results(episode_results, show_timing=True)


def run_robolab(args: Args, repo_root: Path = _REPO_ROOT) -> None:
    robolab_root = _robolab_root(repo_root)
    if not robolab_root.exists():
        raise SystemExit(
            "RoboLab submodule is missing. Run: "
            "git submodule update --init --recursive third_party/robolab"
        )

    _validate_args(args)
    _ensure_output_dir_policy_compatible(_resolve_output_folder_name(args), args.policy)
    _install_recorder_uint16_patch()
    with _temporary_sys_path(robolab_root):
        simulation_app = _launch_simulation_app(args)
        try:
            _configure_robolab_runtime(args)
            _register_task_envs(args)
            _run_task_set(args)
        finally:
            simulation_app.close()


def main(args: Args) -> None:
    run_robolab(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
