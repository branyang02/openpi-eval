"""
Evaluate a single MetaWorld task (using parallel envs) against a policy server.

Single task:
    MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3

All tasks in a split (sequential, single server):
    MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train

"""

import collections
import dataclasses
import logging
import math
import os

import gymnasium as gym
import imageio.v3 as iio
import metaworld  # noqa: F401
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
from tqdm import tqdm
import tyro

logger = logging.getLogger(__name__)

# https://metaworld.farama.org/rendering/rendering/#render-from-a-specific-camera
CAMERA_IDS = {
    "topview": 0,
    "corner": 1,
    "corner2": 2,
    "corner3": 3,
    "corner4": 4,
    "behindGripper": 5,
    "gripperPOV": 6,
}

TASK_TO_PROMPT = {
    "assembly-v3": "pick up the nut and place it onto the peg",
    "disassemble-v3": "pick up the nut and remove it from the peg",
    "basketball-v3": "dunk the basketball into the hoop",
    "soccer-v3": "kick the soccer ball into the goal",
    "bin-picking-v3": "pick up the object and place it into the bin",
    "box-close-v3": "grasp the cover and close the box",
    "button-press-v3": "press the button",
    "button-press-topdown-v3": "press the button from the top",
    "button-press-topdown-wall-v3": "press the button on the wall from the top",
    "button-press-wall-v3": "press the button on the wall",
    "coffee-button-v3": "push the button on the coffee machine",
    "coffee-pull-v3": "pull the mug away from the coffee machine",
    "coffee-push-v3": "push the mug under the coffee machine",
    "dial-turn-v3": "rotate the dial",
    "lever-pull-v3": "pull the lever down",
    "door-close-v3": "close the door",
    "door-lock-v3": "lock the door by rotating the lock",
    "door-open-v3": "open the door",
    "door-unlock-v3": "unlock the door by rotating the lock",
    "drawer-close-v3": "push the drawer closed",
    "drawer-open-v3": "pull the drawer open",
    "faucet-close-v3": "rotate the faucet handle to close it",
    "faucet-open-v3": "rotate the faucet handle to open it",
    "hammer-v3": "hammer the nail into the board",
    "hand-insert-v3": "insert the gripper into the hole",
    "handle-press-v3": "press the handle down",
    "handle-press-side-v3": "press the handle down sideways",
    "handle-pull-v3": "pull the handle up",
    "handle-pull-side-v3": "pull the handle sideways",
    "peg-insert-side-v3": "insert the peg into the hole sideways",
    "peg-unplug-side-v3": "unplug the peg from the hole sideways",
    "pick-out-of-hole-v3": "pick the object out of the hole",
    "pick-place-v3": "pick up the object and place it at the goal",
    "pick-place-wall-v3": "pick up the object and place it at the goal behind the wall",
    "plate-slide-v3": "slide the plate to the goal",
    "plate-slide-back-v3": "slide the plate backwards to the goal",
    "plate-slide-back-side-v3": "slide the plate backwards and sideways to the goal",
    "plate-slide-side-v3": "slide the plate sideways to the goal",
    "push-v3": "push the object to the goal",
    "push-back-v3": "push the object backwards to the goal",
    "push-wall-v3": "push the object around the wall to the goal",
    "reach-v3": "reach the goal position",
    "reach-wall-v3": "reach the goal position behind the wall",
    "shelf-place-v3": "pick up the object and place it on the shelf",
    "stick-pull-v3": "use the stick to pull the object",
    "stick-push-v3": "use the stick to push the object",
    "sweep-v3": "sweep the object off the table",
    "sweep-into-v3": "sweep the object into the hole",
    "window-close-v3": "push the window closed",
    "window-open-v3": "push the window open",
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # Environment name (e.g., "pick-place-v3").
    env_name: str = "pick-place-v3"
    # Number of parallel environments to run.
    num_envs: int = 10
    # Number of episodes to run.
    num_episodes: int = 1
    # Maximum steps per episode.
    max_steps: int = 300
    # Number of steps to execute from the model's action plan before re-planning.
    replan_steps: int = 10

    width: int = 224
    height: int = 224

    # Cameras to use for policy input.
    policy_cameras: list[str] = dataclasses.field(default_factory=lambda: ["corner", "corner4", "gripperPOV"])
    # The camera used for rendering the video output (must be one of the policy cameras).
    render_camera: str = "corner"

    fps: int = 24
    seed: int = 69_420

    # Override the eval-artifact directory (videos). If None, defaults to
    # ``examples/metaworld/output/{env_name}/``. Relative paths are resolved
    # against the user's shell cwd, matching the libero and robocasa examples.
    output_dir: str | None = None


class MultiCameraWrapper(gym.Wrapper):
    """Wrapper that renders multiple cameras and includes images in info dict."""

    def __init__(self, env: gym.Env, camera_names: list[str]):
        super().__init__(env)
        self.camera_names = camera_names

    def _render_cameras(self) -> dict[str, np.ndarray]:
        renderer = self.unwrapped.mujoco_renderer
        images = {}
        for cam_name in self.camera_names:
            # HACK (branyang02): Very Very Very Hacky
            # Take a look at gymnasium.envs.muojoco.mujoco_rendering.MujocoRenderer.render()
            # Implemented solutions from:
            # https://github.com/Farama-Foundation/Metaworld/issues/448
            # https://github.com/Farama-Foundation/Gymnasium/issues/736
            viewer = renderer._get_viewer(render_mode="rgb_array")  # noqa: SLF001
            if len(renderer._viewers.keys()) >= 1:  # noqa: SLF001
                viewer.make_context_current()
            img = viewer.render(render_mode="rgb_array", camera_id=CAMERA_IDS[cam_name])
            images[cam_name] = img[::-1].copy()  # flip vertically
        return images

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["cameras"] = self._render_cameras()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["cameras"] = self._render_cameras()
        return obs, reward, terminated, truncated, info


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


def make_env(env_name: str, num_envs: int, width: int, height: int, seed: int, camera_names: list[str]) -> gym.Env:
    env_fns = [
        lambda i=i: MultiCameraWrapper(
            gym.make("Meta-World/MT1", env_name=env_name, seed=seed + i, width=width, height=height),
            camera_names,
        )
        for i in range(num_envs)
    ]
    return gym.vector.AsyncVectorEnv(env_fns)


def run_episode(
    env: gym.Env,
    policy,
    args: Args,
    episode: int,
    output_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single episode; return ``(total_reward, success)`` per env."""
    prompt = TASK_TO_PROMPT[args.env_name]
    obs, info = env.reset(seed=args.seed + episode)
    camera_views = info["cameras"]
    num_envs = env.num_envs
    success = np.zeros(num_envs, dtype=bool)
    total_reward = np.zeros(num_envs)
    action_plan: collections.deque = collections.deque()

    video_path = os.path.join(output_dir, f"episode_{episode:03d}.mp4")
    with iio.imopen(video_path, "w", plugin="pyav") as video:
        video.init_video_stream("h264", fps=args.fps)

        pbar = tqdm(range(args.max_steps), desc=f"Episode {episode + 1}/{args.num_episodes}")
        for _step in pbar:
            grid_frame = tile_frames(list(camera_views[args.render_camera]))
            video.write_frame(grid_frame)

            if not action_plan:
                obs_dict = {
                    "observation/image": camera_views["corner4"],
                    "observation/wrist_image": camera_views["gripperPOV"],
                    "observation/state": obs.astype(np.float32)[
                        ..., :4
                    ],  # first 4 dims are the true observable state in Metaworld.
                    "prompt": [prompt] * num_envs,
                }
                result = policy.infer(obs_dict)
                action_chunk = np.clip(result["actions"], -1.0, 1.0).astype(np.float32)

                assert (
                    action_chunk.ndim == 3
                ), f"Model output must have shape (batch_size, action_horizon, action_dim), but got {action_chunk.shape}"
                assert action_chunk.shape[1] >= args.replan_steps, "Model must output at least replan_steps actions"
                for t in range(args.replan_steps):
                    action_plan.append(action_chunk[:, t, :])

            action = action_plan.popleft()
            obs, reward, terminated, truncated, info = env.step(action)
            camera_views = info["cameras"]
            total_reward += reward
            step_success = np.asarray(info.get("success", np.zeros(num_envs)), dtype=bool)
            success |= step_success
            if success.all():
                break

            pbar.set_postfix(reward=f"{total_reward.mean():.1f}", success=f"{success.mean():.0%}")

    logger.info(
        f"Episode {episode + 1}/{args.num_episodes}: "
        f"mean_reward={total_reward.mean():.2f}, success_rate={success.mean():.2f}, "
        f"video={video_path}"
    )
    return total_reward, success


def main(args: Args) -> None:
    np.random.seed(args.seed)

    policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(__file__), "output", args.env_name)
    os.makedirs(output_dir, exist_ok=True)

    env = make_env(
        env_name=args.env_name,
        num_envs=args.num_envs,
        width=args.width,
        height=args.height,
        seed=args.seed,
        camera_names=args.policy_cameras,
    )

    all_successes: list[bool] = []
    try:
        for episode in range(args.num_episodes):
            _, success = run_episode(env, policy, args, episode, output_dir)
            all_successes.extend(bool(s) for s in success)
    finally:
        env.close()

    if all_successes:
        sr = float(np.mean(all_successes))
        num_success = sum(all_successes)
        # Matches the parse pattern used by experiments/*/find_best_configs.py.
        logger.info(
            "[%s] success_rate=%.2f (%d/%d)",
            args.env_name,
            sr,
            num_success,
            len(all_successes),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
