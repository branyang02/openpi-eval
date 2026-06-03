"""
MUJOCO_GL=egl uv run examples/metaworld/generate_dataset.py
"""

import dataclasses
import logging

import gymnasium as gym
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import metaworld
from metaworld.policies import ENV_POLICY_MAP
import numpy as np
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
    repo_id: str = "brandonyang/metaworld_ml45"

    # Number of environments running in parallel (vectorized)
    num_envs: int = 50
    # Number of times to roll out all parallel envs; total episodes per task = num_envs * num_episodes
    num_episodes: int = 2
    max_steps: int = 500

    width: int = 224
    height: int = 224

    # Cameras to use for policy input
    policy_cameras: list[str] = dataclasses.field(default_factory=lambda: ["corner", "corner4", "gripperPOV"])

    seed: int = 42


class MultiCameraWrapper(gym.Wrapper):
    """Wrapper that renders multiple cameras and includes images in info dict."""

    def __init__(self, env: gym.Env, camera_names: list[str]):
        super().__init__(env)
        self.camera_names = camera_names

    def _render_cameras(self) -> dict[str, np.ndarray]:
        renderer = self.unwrapped.mujoco_renderer
        images = {}
        for cam_name in self.camera_names:
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


def make_env(env_name: str, num_envs: int, width: int, height: int, seed: int, camera_names: list[str]) -> gym.Env:
    env_fns = [
        lambda i=i: MultiCameraWrapper(
            gym.make("Meta-World/MT1", env_name=env_name, seed=seed + i, width=width, height=height),
            camera_names,
        )
        for i in range(num_envs)
    ]
    return gym.vector.AsyncVectorEnv(env_fns)


def run_env(env_name: str, policy, dataset: LeRobotDataset, args: Args) -> None:
    """Run all batches for a single environment and save trajectory data to the dataset."""
    env = make_env(
        env_name,
        num_envs=args.num_envs,
        width=args.width,
        height=args.height,
        seed=args.seed,
        camera_names=args.policy_cameras,
    )
    num_envs = env.num_envs

    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + episode)
        camera_views = info["cameras"]
        success = np.zeros(num_envs, dtype=bool)
        total_reward = np.zeros(num_envs)

        # Trajectory buffers: one list per parallel env
        traj_obs = [[] for _ in range(num_envs)]
        traj_actions = [[] for _ in range(num_envs)]
        traj_images = [{cam: [] for cam in args.policy_cameras} for _ in range(num_envs)]

        pbar = tqdm(range(args.max_steps), desc=f"[{env_name}] Episode {episode + 1}/{args.num_episodes}")
        for _step in pbar:
            actions = np.stack([policy.get_action(obs[i]) for i in range(num_envs)], axis=0)

            # Record this step for every env that hasn't succeeded yet
            for i in range(num_envs):
                if not success[i]:
                    traj_obs[i].append(obs[i].copy())
                    traj_actions[i].append(actions[i].copy())
                    for cam in args.policy_cameras:
                        traj_images[i][cam].append(camera_views[cam][i].copy())

            obs, reward, terminated, truncated, info = env.step(actions)
            camera_views = info["cameras"]
            total_reward += reward
            success |= np.asarray(info.get("success", np.zeros(num_envs)), dtype=bool)
            if success.all():
                break
            pbar.set_postfix(reward=f"{total_reward.mean():.1f}", success=f"{success.mean():.0%}")

        # Save each successful parallel env's trajectory into the dataset
        for i in range(num_envs):
            if not success[i]:
                continue
            for t in range(len(traj_obs[i])):
                dataset.add_frame(
                    {
                        "observation.state": traj_obs[i][t][:4].astype(np.float32),
                        "observation.environment_state": traj_obs[i][t].astype(np.float32),
                        **{f"{cam}.image": traj_images[i][cam][t] for cam in args.policy_cameras},
                        "actions": traj_actions[i][t].astype(np.float32),
                        "task": TASK_TO_PROMPT[env_name],
                    }
                )
            dataset.save_episode()

        logger.info(
            f"[{env_name}] Episode {episode + 1}/{args.num_episodes}: "
            f"mean_reward={total_reward.mean():.2f}, success_rate={success.mean():.2f}, "
            f"saved={success.sum()} episodes"
        )

    env.close()


def main(args: Args) -> None:
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="metaworld",
        fps=80,  # copied from https://huggingface.co/datasets/lerobot/metaworld_mt50/blob/main/meta/info.json
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["observation_state"],
            },
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (39,),
                "names": ["environment_state"],
            },
            **{
                f"{cam}.image": {
                    "dtype": "image",
                    "shape": (args.height, args.width, 3),
                    "names": ["height", "width", "channel"],
                }
                for cam in args.policy_cameras
            },
            "actions": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    ml45 = metaworld.ML45()
    all_env_names = list(ml45.train_classes.keys())

    for env_name in tqdm(all_env_names, desc="Environments"):
        if env_name not in ENV_POLICY_MAP:
            logger.warning(f"No scripted policy found for '{env_name}'. Skipping.")
            continue

        logger.info(f"Running environment: {env_name}")
        run_env(env_name, ENV_POLICY_MAP[env_name](), dataset, args)

    dataset.push_to_hub()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    main(args)
