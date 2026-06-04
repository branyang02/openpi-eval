"""
Evaluate a single LIBERO task using a policy server.

Single task:
    MUJOCO_GL=egl uv run python main.py --task_suite_name libero_spatial --task_id 0

All tasks in a suite (parallel subprocesses, one per task_id):
    MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_spatial

Like RoboCasa, this example evaluates one env at a time and tiles multiple camera
views from that single env into the saved video.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import re
import sys
import time
from typing import Deque, Dict, List, Literal, Optional

_LIBERO_REPO_ROOT = (
    pathlib.Path(__file__).resolve().parents[2] / "third_party" / "libero"
)
if str(_LIBERO_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIBERO_REPO_ROOT))

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
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # LIBERO suite name.
    task_suite_name: Literal[
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    ] = "libero_10"
    # Task index within the suite.
    task_id: int = 0
    # Number of episodes / initial states to evaluate.
    num_episodes: int = 1
    # Override the suite default max steps. If None, uses SUITE_MAX_STEPS.
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
    # Whether to write per-episode rollout videos. ``record`` keeps the normal
    # evaluator behavior; ``none`` keeps MuJoCo rendering for policy inputs but
    # skips video encoding for throughput benchmarks.
    video_mode: Literal["record", "none"] = "record"
    # Progress bars are useful for interactive single-task runs, but noisy when
    # many subprocesses write logs concurrently.
    progress_mode: Literal["auto", "off"] = "auto"
    # Optional wall-clock barrier used by benchmark_parallel_clients.py after
    # each env has reset to its initial state.
    start_after_unix_s: Optional[float] = None
    # Forwarded to robosuite/MuJoCo. -1 keeps robosuite's default device choice.
    render_gpu_device_id: int = -1

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
    if task_suite_name not in SUITE_MAX_STEPS:
        raise ValueError(
            "No default max_steps registered for task suite {!r}".format(
                task_suite_name
            )
        )
    return SUITE_MAX_STEPS[task_suite_name]


def sanitize_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "task"


def make_env(
    task, resolution: int, seed: int, render_gpu_device_id: int = -1
) -> OffScreenRenderEnv:
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
        render_gpu_device_id=render_gpu_device_id,
    )
    env.seed(seed)
    return env


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
    initial_states = task_suite.get_task_init_states(task_id)
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

    env = make_env(
        task,
        LIBERO_ENV_RESOLUTION,
        args.seed,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    successes = []
    policy_calls = 0
    env_steps = 0
    server_batch_size_counts = collections.Counter()
    server_padded_batch_size_counts = collections.Counter()

    # --seed acts as an offset into LIBERO's canonical initial-state list so
    # different seeds evaluate on disjoint start conditions. Pick seeds at least
    # num_episodes apart when you want non-overlapping eval splits.
    num_init_states = len(initial_states)
    try:
        for episode in range(args.num_episodes):
            state_idx = (args.seed + episode) % num_init_states
            env.reset()
            obs = env.set_init_state(initial_states[state_idx])
            if args.start_after_unix_s is not None:
                sleep_s = args.start_after_unix_s - time.time()
                if sleep_s > 0:
                    time.sleep(sleep_s)
            action_plan = collections.deque()  # type: Deque[np.ndarray]
            success = False
            video_path = os.path.join(
                task_output_dir, "episode_{:03d}.mp4".format(episode)
            )
            video_writer = (
                iio.get_writer(video_path, fps=args.fps)
                if args.video_mode == "record"
                else None
            )

            try:
                pbar = tqdm(
                    range(args.num_steps_wait + max_steps),
                    desc="[{}] Episode {}/{}".format(
                        task_name, episode + 1, args.num_episodes
                    ),
                    leave=False,
                    disable=args.progress_mode == "off",
                )
                for step in pbar:
                    if video_writer is not None:
                        video_writer.append_data(render_frame(obs, args.render_cameras))

                    if step < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        env_steps += 1
                        continue

                    if not action_plan:
                        element = prepare_policy_inputs(obs, args.resize_size)
                        element["prompt"] = task_description
                        response = policy.infer(element)
                        policy_calls += 1
                        server_timing = response.get("server_timing", {})
                        if server_timing:
                            batch_size = int(server_timing.get("batch_size", 1))
                            padded_batch_size = int(
                                server_timing.get("padded_batch_size", batch_size)
                            )
                            server_batch_size_counts[batch_size] += 1
                            server_padded_batch_size_counts[padded_batch_size] += 1
                        action_chunk = np.asarray(response["actions"], dtype=np.float32)
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
                    env_steps += 1
                    success = bool(done)
                    pbar.set_postfix(success=str(success))
                    if success:
                        break
            finally:
                if video_writer is not None:
                    video_writer.close()

            successes.append(success)
            logger.info(
                "[%s] Episode %d/%d: success=%s, video=%s",
                task_name,
                episode + 1,
                args.num_episodes,
                success,
                video_path if args.video_mode == "record" else "disabled",
            )
    finally:
        env.close()

    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "num_episodes": float(len(successes)),
        "task_id": float(task_id),
        "task_name": task_name,
        "task_description": task_description,
        "policy_calls": float(policy_calls),
        "env_steps": float(env_steps),
        "server_batch_size_counts": dict(server_batch_size_counts),
        "server_padded_batch_size_counts": dict(server_padded_batch_size_counts),
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
        "[%s/%s/task_%02d] success_rate=%.2f (%d/%d), policy_calls=%d, env_steps=%d",
        args.task_suite_name,
        result["task_name"],
        args.task_id,
        result["success_rate"],
        int(result["success_rate"] * result["num_episodes"]),
        int(result["num_episodes"]),
        int(result["policy_calls"]),
        int(result["env_steps"]),
    )
    logger.info(
        "metrics_json=%s",
        json.dumps(
            {
                "success_rate": result["success_rate"],
                "num_episodes": result["num_episodes"],
                "policy_calls": result["policy_calls"],
                "env_steps": result["env_steps"],
                "server_batch_size_counts": result["server_batch_size_counts"],
                "server_padded_batch_size_counts": result[
                    "server_padded_batch_size_counts"
                ],
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
