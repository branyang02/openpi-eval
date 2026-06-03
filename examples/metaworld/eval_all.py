"""
Evaluate all tasks in an ML45 split (train, test, or curated subset) against a
policy server. One in-process task loop hits one server. Vectorized envs
(``--num_envs``) provide intra-task parallelism, so multi-task subprocess
fan-out (as in libero/robocasa) is not needed here.

Eval:
    MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train
"""

import dataclasses
import json
import logging
import os
from typing import Literal

# Reuse helpers from main.py so single-task and multi-task eval cannot diverge.
# main.py is a sibling module; running via "uv run examples/metaworld/eval_all.py"
# puts its directory on sys.path[0], so "from main import ..." resolves correctly.
from main import CAMERA_IDS  # noqa: F401
from main import TASK_TO_PROMPT  # noqa: F401
from main import Args as _MainArgs
from main import MultiCameraWrapper  # noqa: F401
from main import make_env
from main import run_episode
from main import tile_frames  # noqa: F401
import metaworld
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
from tqdm import tqdm
import tyro

logger = logging.getLogger(__name__)


# Curated subset of 26 ML45-train tasks for faster smoke evaluation.
SUBSET = [
    "assembly-v3",
    "basketball-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "disassemble-v3",
    "door-open-v3",
    "faucet-close-v3",
    "hammer-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "peg-insert-side-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "pick-place-wall-v3",
    "plate-slide-back-side-v3",
    "plate-slide-back-v3",
    "push-back-v3",
    "push-v3",
    "reach-v3",
    "shelf-place-v3",
    "soccer-v3",
    "stick-pull-v3",
    "stick-push-v3",
    "sweep-into-v3",
    "sweep-v3",
]


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # Which ML45 split or curated subset to evaluate (ignored if --tasks is non-empty).
    split: Literal["train", "test", "subset"] = "subset"
    # Subset of task names to evaluate. If empty, uses --split.
    tasks: list[str] = dataclasses.field(default_factory=list)

    # Number of parallel environments per task.
    num_envs: int = 15
    # Number of episodes per task.
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

    # Override the eval-artifact directory (videos, results.json). If None, defaults to
    # ``examples/metaworld/output/ML45-{split}/``. Relative paths are resolved against
    # the user's shell cwd, matching the libero and robocasa examples.
    output_dir: str | None = None


def _resolve_tasks(args: Args) -> list[str]:
    if args.tasks:
        return list(args.tasks)
    if args.split == "subset":
        return list(SUBSET)
    ml45 = metaworld.ML45()
    return list(ml45.train_classes.keys()) if args.split == "train" else list(ml45.test_classes.keys())


def _per_task_args(env_name: str, args: Args, task_output_dir: str) -> _MainArgs:
    """Build a ``main.Args`` for ``run_episode`` to consume for a given task."""
    return _MainArgs(
        host=args.host,
        port=args.port,
        env_name=env_name,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
        width=args.width,
        height=args.height,
        policy_cameras=list(args.policy_cameras),
        render_camera=args.render_camera,
        fps=args.fps,
        seed=args.seed,
        output_dir=task_output_dir,
    )


def eval_task(
    env_name: str,
    policy,
    args: Args,
    output_dir: str,
) -> dict[str, float]:
    """Evaluate a single task over ``num_episodes`` and return mean success rate."""
    env = make_env(
        env_name=env_name,
        num_envs=args.num_envs,
        width=args.width,
        height=args.height,
        seed=args.seed,
        camera_names=args.policy_cameras,
    )

    task_output_dir = os.path.join(output_dir, env_name)
    os.makedirs(task_output_dir, exist_ok=True)

    task_args = _per_task_args(env_name, args, task_output_dir)

    episode_success_rates: list[float] = []
    try:
        for episode in range(args.num_episodes):
            _, success = run_episode(env, policy, task_args, episode, task_output_dir)
            episode_success_rates.append(float(success.mean()))
    finally:
        env.close()

    return {"success_rate": float(np.mean(episode_success_rates))}


def main(args: Args) -> None:
    np.random.seed(args.seed)

    policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(__file__), "output", f"ML45-{args.split}")
    os.makedirs(output_dir, exist_ok=True)

    env_names = _resolve_tasks(args)
    logger.info(f"Evaluating {len(env_names)} tasks")

    results_path = os.path.join(output_dir, "results.json")
    results: dict[str, float] = {}
    for env_name in tqdm(env_names, desc=f"ML45-{args.split}"):
        task_result = eval_task(
            env_name,
            policy,
            args,
            output_dir,
        )
        results[env_name] = task_result["success_rate"]
        logger.info(f"[{env_name}] success_rate={results[env_name]:.2f}")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    mean_success = float(np.mean(list(results.values()))) if results else 0.0
    summary = {
        "mean_success_rate": mean_success,
        "per_task": dict(sorted(results.items(), key=lambda x: x[1], reverse=True)),
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info("=" * 60)
    logger.info(f"Overall mean success rate: {mean_success:.2f} ({mean_success:.0%})")
    logger.info("Per-task results:")
    for env_name, rate in sorted(results.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {env_name:<40s} {rate:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
