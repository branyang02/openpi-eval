"""
Evaluate a single RoboCasa task using a policy server.

For evaluating one environment:
    MUJOCO_GL=egl uv run examples/robocasa_env/main.py --env_name CloseBlenderLid

To evaluate every environment in a task set (atomic_seen, composite_seen,
composite_unseen, pretrain50, ...) instead, use eval_all.py:
    MUJOCO_GL=egl uv run examples/robocasa_env/eval_all.py --task_set atomic_seen

Note: RoboCasa does NOT support parallel envs (gym.vector.AsyncVectorEnv) because
each MuJoCo env requires its own EGL/OpenGL context, which is not multiprocess-safe.
The grid in the video is built from the multiple cameras of a single env, not from
multiple parallel envs as in the MetaWorld example.
"""

import collections
import dataclasses
import logging
import math
import os
from typing import Optional

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import robocasa  # noqa: F401
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from tqdm import tqdm

logger = logging.getLogger(__name__)

# RoboCasa's gym wrapper hardcodes these three camera observations on every step.
# Keys are short names used in CLI args; values are the obs dict keys.
CAMERA_KEYS = {
    "agentview_left": "video.robot0_agentview_left",
    "agentview_right": "video.robot0_agentview_right",
    "eye_in_hand": "video.robot0_eye_in_hand",
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # RoboCasa task name (e.g., "CloseBlenderLid", "OpenCabinet", "TurnOnMicrowave").
    env_name: str = "CloseBlenderLid"
    # Dataset split: "pretrain" (in-distribution object instances) or "target" (held-out).
    split: str = "pretrain"
    # Number of episodes to run.
    num_episodes: int = 1
    # Override the maximum steps per episode. If None, uses 1.5 * task horizon.
    max_steps: int | None = None
    # Number of steps to execute from the model's action plan before re-planning.
    replan_steps: int = 5

    # Image resize size for the policy input.
    resize_size: int = 224

    # Cameras to tile into the video output.
    render_cameras: list[str] = dataclasses.field(
        default_factory=lambda: ["agentview_left", "agentview_right", "eye_in_hand"]
    )

    fps: int = 24
    seed: int = 7

    # Override the top-level output directory (for videos / artifacts). If
    # None, defaults to ``output/`` — ``eval_task`` nests a per-env subdir, so
    # per-episode videos land at ``output/{env_name}/episode_NNN.mp4``.
    output_dir: Optional[str] = None


def tile_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Arrange N frames into a grid image.

    Grid layout: cols = ceil(sqrt(N)), rows = ceil(N / cols).
    Empty slots are filled with black.
    """
    n = len(frames)
    h, w, c = frames[0].shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    grid = np.zeros((rows * h, cols * w, c), dtype=frames[0].dtype)
    for idx, frame in enumerate(frames):
        r, col = divmod(idx, cols)
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = frame

    return grid


def make_env(env_name: str, split: str, seed: int) -> gym.Env:
    return gym.make(f"robocasa/{env_name}", split=split, seed=seed)


def build_state(obs: dict) -> np.ndarray:
    """Concatenate the proprioceptive state into the 16-dim vector expected by the policy."""
    return np.concatenate(
        [
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ],
        axis=0,
    )


def eval_task(
    env_name: str,
    policy: _websocket_client_policy.WebsocketClientPolicy,
    args: Args,
    output_dir: str,
) -> dict[str, float]:
    """Evaluate a single env over args.num_episodes episodes and return per-task stats.

    The args object only needs to provide: split, num_episodes, max_steps,
    replan_steps, resize_size, render_cameras, fps. Both main.py's Args and
    eval_all.py's Args satisfy this.

    """
    task_output_dir = os.path.join(output_dir, env_name)
    os.makedirs(task_output_dir, exist_ok=True)

    env = make_env(env_name=env_name, split=args.split, seed=args.seed)
    task_horizon = get_task_horizon(env_name)
    max_steps = (
        args.max_steps if args.max_steps is not None else int(task_horizon * 1.5)
    )

    successes: list[bool] = []
    for episode in range(args.num_episodes):
        obs, info = env.reset()
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()
        success = False

        video_path = os.path.join(task_output_dir, f"episode_{episode:03d}.mp4")
        with iio.imopen(video_path, "w", plugin="pyav") as video:
            video.init_video_stream("h264", fps=args.fps)

            pbar = tqdm(
                range(max_steps),
                desc=f"[{env_name}] Episode {episode + 1}/{args.num_episodes}",
                leave=False,
            )
            for step in pbar:
                # Tile multiple cameras of the single env into a grid frame.
                frames = [
                    image_tools.convert_to_uint8(obs[CAMERA_KEYS[cam]])
                    for cam in args.render_cameras
                ]
                video.write_frame(tile_frames(frames))

                if not action_plan:
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            obs[CAMERA_KEYS["agentview_left"]],
                            args.resize_size,
                            args.resize_size,
                        )
                    )
                    # Preserve the second side-view camera in the payload. The
                    # current OpenPI transform ignores it unless a RoboCasa
                    # config consumes this key.
                    img2 = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            obs[CAMERA_KEYS["agentview_right"]],
                            args.resize_size,
                            args.resize_size,
                        )
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            obs[CAMERA_KEYS["eye_in_hand"]],
                            args.resize_size,
                            args.resize_size,
                        )
                    )
                    element = {
                        "observation/image": img,
                        "observation/image2": img2,
                        "observation/wrist_image": wrist_img,
                        "observation/state": build_state(obs),
                        "prompt": task_lang,
                    }
                    result = policy.infer(element)
                    action_chunk = result["actions"]  # (action_horizon, action_dim=12)
                    assert (
                        action_chunk.ndim == 2
                    ), f"Model output must have shape (action_horizon, action_dim), but got {action_chunk.shape}"
                    assert (
                        action_chunk.shape[0] >= args.replan_steps
                    ), f"Model must output at least {args.replan_steps} actions, got {action_chunk.shape[0]}"
                    for t in range(args.replan_steps):
                        action_plan.append(action_chunk[t])

                action = convert_action(action_plan.popleft())
                obs, reward, terminated, truncated, info = env.step(action)
                success = bool(info.get("success", False))

                pbar.set_postfix(success=str(success))

                if success:
                    break

        successes.append(success)
        logger.info(
            f"[{env_name}] Episode {episode + 1}/{args.num_episodes}: success={success}, video={video_path}"
        )

    env.close()

    success_rate = float(np.mean(successes))
    return {"success_rate": success_rate, "num_episodes": float(len(successes))}


def main(args: Args) -> None:
    np.random.seed(args.seed)

    policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        # Bare parent — ``eval_task`` below joins ``env_name`` so the final
        # per-task tree is ``output/{env_name}/episode_NNN.mp4`` (matches the
        # README's documented "Default output" path).
        output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    result = eval_task(args.env_name, policy, args, output_dir)
    logger.info(
        f"[{args.env_name}/{args.split}] success_rate={result['success_rate']:.2f} "
        f"({int(result['success_rate'] * result['num_episodes'])}/{int(result['num_episodes'])})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    main(args)
