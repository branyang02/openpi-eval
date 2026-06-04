"""
Evaluate a single LIBERO-Pro task using a policy server.

Single task:
    MUJOCO_GL=egl uv run python main.py --task_suite_name libero_goal_task --task_id 0

All tasks in a suite (parallel subprocesses, one per task_id):
    MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_goal_task

This example mirrors examples/libero_env but installs the LIBERO-Pro fork in a
separate venv because both projects expose a Python package named ``libero``.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import math
import os
import pathlib
import re
import sys
from typing import Deque, Dict, List, Optional

os.environ.setdefault("LIBERO_CONFIG_PATH", str(pathlib.Path.home() / ".liberopro"))

_LIBEROPRO_REPO_ROOT = (
    pathlib.Path(__file__).resolve().parents[2] / "third_party" / "liberopro"
)
if str(_LIBEROPRO_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIBEROPRO_REPO_ROOT))

import imageio.v2 as iio
import numpy as np
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from tqdm import tqdm

logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
CAMERA_KEYS = {
    "agentview": "agentview_image",
    "eye_in_hand": "robot0_eye_in_hand_image",
}
BASE_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
LIBEROPRO_DATASET_REPO = "zhouxueyang/LIBERO-Pro"


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # LIBERO-Pro suite name, e.g. libero_goal_task, libero_spatial_lan,
    # libero_10_object, or libero_object_swap.
    task_suite_name: str = "libero_goal_task"
    # Task index within the suite.
    task_id: int = 0
    # Number of episodes / initial states to evaluate.
    num_episodes: int = 1
    # Override the suite default max steps. If None, uses BASE_SUITE_MAX_STEPS.
    max_steps: Optional[int] = None
    # Number of settling steps before policy actions.
    num_steps_wait: int = 10
    # Number of steps to execute from the model's action plan before re-planning.
    replan_steps: int = 5

    # Image resize size for the policy input.
    resize_size: int = 224

    # Cameras to tile into the video output.
    render_cameras: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "eye_in_hand"]
    )

    fps: int = 10

    # RNG seed. Threads through to the env physics seed and np.random, AND
    # acts as an offset into LIBERO's canonical initial-state list. Episode k
    # picks initial_states[(seed + k) % N].
    seed: int = 7

    # Override the per-task output directory (for videos / artifacts). If None,
    # defaults to ``output/{task_suite_name}-task{task_id:02d}``.
    output_dir: Optional[str] = None


def tile_frames(frames: List[np.ndarray]) -> np.ndarray:
    """Arrange N frames into a grid image."""
    n = len(frames)
    height, width, channels = frames[0].shape
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(float(n) / float(cols)))

    grid = np.zeros((rows * height, cols * width, channels), dtype=frames[0].dtype)
    for idx, frame in enumerate(frames):
        row, col = divmod(idx, cols)
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = frame
    return grid


def suite_task_names(task_suite_name: str) -> List[str]:
    task_suite = get_task_suite(task_suite_name)
    return [
        getattr(task_suite.get_task(task_id), "name", "task_{:02d}".format(task_id))
        for task_id in range(task_suite.n_tasks)
    ]


def get_task_suite(task_suite_name: str):
    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(
            "Unknown task_suite_name {!r}. Available: {}".format(
                task_suite_name,
                sorted(benchmark_dict.keys()),
            )
        )
    return benchmark_dict[task_suite_name]()


def get_max_steps(task_suite_name: str, max_steps: Optional[int]) -> int:
    if max_steps is not None:
        return max_steps
    base_suite = base_suite_name(task_suite_name)
    if base_suite not in BASE_SUITE_MAX_STEPS:
        raise ValueError(
            "No default max_steps registered for task suite {!r}. Pass --max_steps to evaluate it.".format(
                task_suite_name,
            ),
        )
    return BASE_SUITE_MAX_STEPS[base_suite]


def base_suite_name(task_suite_name: str) -> str:
    for base_suite in sorted(BASE_SUITE_MAX_STEPS, key=len, reverse=True):
        if task_suite_name == base_suite or task_suite_name.startswith(
            base_suite + "_"
        ):
            return base_suite
    return task_suite_name


def sanitize_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "task"


def make_env(task, resolution: int, seed: int) -> OffScreenRenderEnv:
    task_bddl_file = resolve_bddl_file(task)
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def _missing_liberopro_data_message(path: pathlib.Path, kind: str) -> str:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    local_dir = "/tmp/liberopro-data"
    return "\n".join(
        [
            "Missing LIBERO-Pro {} file: {}".format(kind, path),
            "The LIBERO-Pro Git submodule does not include every generated evaluation file.",
            "Install the generated dataset from Hugging Face, then rerun setup:",
            "  hf download {} --repo-type dataset --include 'bddl_files/*' --include 'init_files/*' --local-dir {}".format(
                LIBEROPRO_DATASET_REPO,
                local_dir,
            ),
            "  cp -a {}/bddl_files/. {}/third_party/liberopro/libero/libero/bddl_files/".format(
                local_dir,
                repo_root,
            ),
            "  cp -a {}/init_files/. {}/third_party/liberopro/libero/libero/init_files/".format(
                local_dir,
                repo_root,
            ),
            "  cd {}/examples/liberopro_env && uv run python setup_liberopro_config.py".format(
                repo_root,
            ),
        ]
    )


def _resolve_required_file(
    root_key: str, folder: str, filename: str, kind: str
) -> pathlib.Path:
    path = pathlib.Path(get_libero_path(root_key)) / folder / filename
    if path.exists():
        return path
    raise FileNotFoundError(_missing_liberopro_data_message(path, kind))


def resolve_bddl_file(task) -> pathlib.Path:
    return _resolve_required_file(
        "bddl_files", task.problem_folder, task.bddl_file, "BDDL"
    )


def resolve_init_states_file(task) -> pathlib.Path:
    return _resolve_required_file(
        "init_states", task.problem_folder, task.init_states_file, "init-state"
    )


def load_init_states(task):
    import torch

    return torch.load(str(resolve_init_states_file(task)))


def rotate_image(image: np.ndarray) -> np.ndarray:
    # Rotate 180 degrees to match LIBERO training preprocessing.
    return np.ascontiguousarray(image[::-1, ::-1])


def build_state(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            obs["robot0_eef_pos"],
            quat_to_axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ],
        axis=0,
    ).astype(np.float32)


def prepare_policy_inputs(
    obs: Dict[str, np.ndarray], resize_size: int
) -> Dict[str, np.ndarray]:
    base_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(
            rotate_image(obs[CAMERA_KEYS["agentview"]]), resize_size, resize_size
        )
    )
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(
            rotate_image(obs[CAMERA_KEYS["eye_in_hand"]]), resize_size, resize_size
        )
    )
    return {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": build_state(obs),
    }


def render_frame(obs: Dict[str, np.ndarray], render_cameras: List[str]) -> np.ndarray:
    frames = []
    for camera in render_cameras:
        if camera not in CAMERA_KEYS:
            raise ValueError(
                "Unknown render camera {!r}. Available: {}".format(
                    camera, sorted(CAMERA_KEYS.keys())
                )
            )
        frames.append(
            image_tools.convert_to_uint8(rotate_image(obs[CAMERA_KEYS[camera]]))
        )
    return tile_frames(frames)


def eval_task(
    task_suite_name: str,
    task_id: int,
    policy: _websocket_client_policy.WebsocketClientPolicy,
    args: Args,
    output_dir: str,
) -> Dict[str, float]:
    """Evaluate a single LIBERO task over args.num_episodes episodes."""
    task_suite = get_task_suite(task_suite_name)
    if task_id < 0 or task_id >= task_suite.n_tasks:
        raise ValueError(
            "task_id must be in [0, {}), got {}".format(task_suite.n_tasks, task_id)
        )

    task = task_suite.get_task(task_id)
    task_name = getattr(task, "name", "task_{:02d}".format(task_id))
    task_description = str(task.language)
    initial_states = load_init_states(task)
    if args.num_episodes > len(initial_states):
        raise ValueError(
            "Requested {} episodes for task {!r}, but only {} initial states are available".format(
                args.num_episodes,
                task_name,
                len(initial_states),
            )
        )

    max_steps = get_max_steps(task_suite_name, args.max_steps)
    task_output_dir = os.path.join(
        output_dir, "{:02d}-{}".format(task_id, sanitize_name(task_name))
    )
    os.makedirs(task_output_dir, exist_ok=True)

    env = make_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    successes = []

    # --seed acts as an offset into LIBERO's canonical initial-state list so
    # different seeds evaluate on disjoint start conditions. Pick seeds at least
    # num_episodes apart when you want non-overlapping eval splits.
    num_init_states = len(initial_states)
    try:
        for episode in range(args.num_episodes):
            state_idx = (args.seed + episode) % num_init_states
            env.reset()
            obs = env.set_init_state(initial_states[state_idx])
            action_plan = collections.deque()  # type: Deque[np.ndarray]
            success = False
            video_path = os.path.join(
                task_output_dir, "episode_{:03d}.mp4".format(episode)
            )

            with iio.get_writer(video_path, fps=args.fps) as video:
                pbar = tqdm(
                    range(args.num_steps_wait + max_steps),
                    desc="[{}] Episode {}/{}".format(
                        task_name, episode + 1, args.num_episodes
                    ),
                    leave=False,
                )
                for step in pbar:
                    video.append_data(render_frame(obs, args.render_cameras))

                    if step < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        continue

                    if not action_plan:
                        element = prepare_policy_inputs(obs, args.resize_size)
                        element["prompt"] = task_description
                        action_chunk = np.asarray(
                            policy.infer(element)["actions"], dtype=np.float32
                        )
                        if action_chunk.ndim != 2:
                            raise ValueError(
                                "Model output must have shape (action_horizon, action_dim), got {}".format(
                                    action_chunk.shape,
                                )
                            )
                        if action_chunk.shape[0] < args.replan_steps:
                            raise ValueError(
                                "Model must output at least {} actions, got {}".format(
                                    args.replan_steps,
                                    action_chunk.shape[0],
                                )
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    success = bool(done)
                    pbar.set_postfix(success=str(success))
                    if success:
                        break

            successes.append(success)
            logger.info(
                "[%s] Episode %d/%d: success=%s, video=%s",
                task_name,
                episode + 1,
                args.num_episodes,
                success,
                video_path,
            )
    finally:
        env.close()

    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "num_episodes": float(len(successes)),
        "task_id": float(task_id),
        "task_name": task_name,
        "task_description": task_description,
    }


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Copied from robosuite's transform utils."""
    quat = np.array(quat, dtype=np.float32, copy=True)
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))

    denominator = float(np.sqrt(1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)

    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator).astype(np.float32)


def main(args: Args) -> None:
    np.random.seed(args.seed)

    policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logger.info("Server metadata: %s", policy.get_server_metadata())

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            os.path.dirname(__file__),
            "output",
            "{}-task{:02d}".format(args.task_suite_name, args.task_id),
        )
    os.makedirs(output_dir, exist_ok=True)

    result = eval_task(
        args.task_suite_name,
        args.task_id,
        policy,
        args,
        output_dir,
    )
    logger.info(
        "[%s/%s/task_%02d] success_rate=%.2f (%d/%d)",
        args.task_suite_name,
        result["task_name"],
        args.task_id,
        result["success_rate"],
        int(result["success_rate"] * result["num_episodes"]),
        int(result["num_episodes"]),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
