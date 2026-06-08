"""
Evaluate a single MetaWorld task (using parallel envs) against a policy server.

Single task:
    MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3

All tasks in a split (sequential, single server):
    MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train

"""

import collections
import dataclasses
import json
import logging
import math
import os
import time

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

    # Override the eval-artifact directory (per-episode ``episode_XXX.mp4`` video and
    # ``episode_XXX.json`` result/latency record). If None, defaults to
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


def _latency_summary(samples_ms: list[float]) -> dict | None:
    """Summary stats (milliseconds) for per-request latencies, or None if empty."""
    values = [float(v) for v in samples_ms]
    if not values:
        return None
    return {
        "count": len(values),
        "mean_ms": float(np.mean(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "p50_ms": float(np.median(values)),
        "total_ms": float(np.sum(values)),
    }


def _extract_server_infer_ms(result: dict) -> float | None:
    """Return the server-reported inference latency (ms) from an ``infer`` response.

    The world-model server attaches ``{"server_timing": {"infer_ms": ...}}`` to each
    response; other policy servers do not, so missing or malformed timing yields None.
    """
    return _extract_server_timing_ms(result).get("infer_ms")


def _extract_server_timing_ms(result: dict) -> dict[str, float]:
    """Return numeric server timing fields (milliseconds) from an ``infer`` response."""
    server_timing = result.get("server_timing")
    if not isinstance(server_timing, dict):
        return {}
    timings: dict[str, float] = {}
    for key, value in server_timing.items():
        try:
            timings[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return timings


def _policy_state(obs: np.ndarray, *, state_dim: int) -> np.ndarray:
    """Return the observable MetaWorld state slice sent to the policy."""
    state = np.asarray(obs, dtype=np.float32)[..., :state_dim]
    if state.shape[-1] != state_dim:
        raise ValueError(f"MetaWorld observation has state dim {state.shape[-1]}, expected at least {state_dim}.")
    return state


def _metadata_int(metadata: dict | None, key: str, default: int) -> int:
    if not metadata or metadata.get(key) is None:
        return default
    try:
        return int(metadata[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Server metadata field {key!r} must be an integer, got {metadata[key]!r}.") from exc


class IdmHistoryBuffer:
    """Raw previous one-step state/action history, oldest first, for IDM serving."""

    def __init__(self, *, num_envs: int, history_length: int, state_dim: int, action_dim: int) -> None:
        if history_length <= 0:
            raise ValueError(f"history_length must be positive, got {history_length}.")
        self.num_envs = int(num_envs)
        self.history_length = int(history_length)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.prev_state_history = np.zeros((self.num_envs, self.history_length, self.state_dim), dtype=np.float32)
        self.prev_action_history = np.zeros((self.num_envs, self.history_length, self.action_dim), dtype=np.float32)
        self.history_mask = np.zeros((self.num_envs, self.history_length), dtype=np.float32)

    @classmethod
    def from_metadata(cls, metadata: dict | None, *, num_envs: int, default_state_dim: int = 4) -> "IdmHistoryBuffer | None":
        history_length = _metadata_int(metadata, "idm_history_length", 0)
        if history_length <= 0:
            return None
        return cls(
            num_envs=num_envs,
            history_length=history_length,
            state_dim=_metadata_int(metadata, "state_dim", default_state_dim),
            action_dim=_metadata_int(metadata, "action_dim", 4),
        )

    def reset(self) -> None:
        self.prev_state_history.fill(0.0)
        self.prev_action_history.fill(0.0)
        self.history_mask.fill(0.0)

    def reset_rows(self, rows: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=bool)
        expected = (self.num_envs,)
        if rows.shape != expected:
            raise ValueError(f"IDM history reset rows mask must have shape {expected}, got {rows.shape}.")
        if not rows.any():
            return
        self.prev_state_history[rows] = 0.0
        self.prev_action_history[rows] = 0.0
        self.history_mask[rows] = 0.0

    def as_obs_dict(self) -> dict[str, np.ndarray]:
        return {
            "prev_state_history": self.prev_state_history.copy(),
            "prev_action_history": self.prev_action_history.copy(),
            "history_mask": self.history_mask.copy(),
        }

    def append(self, state: np.ndarray, action: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        expected_state = (self.num_envs, self.state_dim)
        expected_action = (self.num_envs, self.action_dim)
        if state.shape != expected_state:
            raise ValueError(f"IDM history state must have shape {expected_state}, got {state.shape}.")
        if action.shape != expected_action:
            raise ValueError(f"IDM history action must have shape {expected_action}, got {action.shape}.")

        self.prev_state_history[:, :-1] = self.prev_state_history[:, 1:]
        self.prev_state_history[:, -1] = state
        self.prev_action_history[:, :-1] = self.prev_action_history[:, 1:]
        self.prev_action_history[:, -1] = action
        self.history_mask[:, :-1] = self.history_mask[:, 1:]
        self.history_mask[:, -1] = 1.0


def _episode_record(
    *,
    env_name: str,
    episode: int,
    num_envs: int,
    max_steps: int,
    replan_steps: int,
    total_reward: np.ndarray,
    success: np.ndarray,
    server_infer_ms: list[float],
    client_request_ms: list[float],
    video: str,
    server_stage_timings_ms: dict[str, list[float]] | None = None,
) -> dict:
    """Build the JSON-serializable per-episode result + latency record.

    ``server_infer_ms`` comes from the server's ``server_timing.infer_ms`` field
    (only the world-model server in ``examples/world_model_env`` reports it);
    ``client_request_ms`` is the client-side round-trip per ``policy.infer`` call.
    Each latency section is omitted when its sample list is empty, so the record
    stays correct against policy servers that do not report timing.
    """
    success_flat = np.asarray(success).reshape(-1)
    reward_flat = np.asarray(total_reward).reshape(-1)
    record: dict = {
        "env_name": env_name,
        "episode": int(episode),
        "num_envs": int(num_envs),
        "max_steps": int(max_steps),
        "replan_steps": int(replan_steps),
        "num_inference_requests": len(client_request_ms),
        "success_rate": float(success_flat.mean()) if success_flat.size else 0.0,
        "mean_reward": float(reward_flat.mean()) if reward_flat.size else 0.0,
        "success": [bool(s) for s in success_flat],
        "total_reward": [float(r) for r in reward_flat],
        "video": video,
    }
    server_summary = _latency_summary(server_infer_ms)
    if server_summary is not None:
        server_timing_record = {**server_summary, "per_request": [float(v) for v in server_infer_ms]}
        stage_records = {}
        for key, samples_ms in sorted((server_stage_timings_ms or {}).items()):
            stage_summary = _latency_summary(samples_ms)
            if stage_summary is not None:
                stage_records[key] = {**stage_summary, "per_request": [float(v) for v in samples_ms]}
        if stage_records:
            server_timing_record["stages"] = stage_records
        record["server_timing_ms"] = server_timing_record
    client_summary = _latency_summary(client_request_ms)
    if client_summary is not None:
        record["client_timing_ms"] = {**client_summary, "per_request": [float(v) for v in client_request_ms]}
    return record


def _write_episode_record(output_dir: str, episode: int, record: dict) -> str:
    """Write ``record`` to ``output_dir/episode_{episode:03d}.json``; return the path."""
    path = os.path.join(output_dir, f"episode_{episode:03d}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def run_episode(
    env: gym.Env,
    policy,
    args: Args,
    episode: int,
    output_dir: str,
    policy_metadata: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single episode; return ``(total_reward, success)`` per env."""
    prompt = TASK_TO_PROMPT[args.env_name]
    obs, info = env.reset(seed=args.seed + episode)
    camera_views = info["cameras"]
    num_envs = env.num_envs
    idm_history = IdmHistoryBuffer.from_metadata(policy_metadata, num_envs=num_envs)
    policy_state_dim = idm_history.state_dim if idm_history is not None else _metadata_int(policy_metadata, "state_dim", 4)
    success = np.zeros(num_envs, dtype=bool)
    total_reward = np.zeros(num_envs)
    action_plan: collections.deque = collections.deque()
    server_infer_ms: list[float] = []
    server_stage_timings_ms: dict[str, list[float]] = {}
    client_request_ms: list[float] = []

    video_path = os.path.join(output_dir, f"episode_{episode:03d}.mp4")
    with iio.imopen(video_path, "w", plugin="pyav") as video:
        video.init_video_stream("h264", fps=args.fps)

        pbar = tqdm(range(args.max_steps), desc=f"Episode {episode + 1}/{args.num_episodes}")
        for _step in pbar:
            grid_frame = tile_frames(list(camera_views[args.render_camera]))
            video.write_frame(grid_frame)

            if not action_plan:
                current_state = _policy_state(obs, state_dim=policy_state_dim)
                obs_dict = {
                    "observation/image": camera_views["corner4"],
                    "observation/wrist_image": camera_views["gripperPOV"],
                    "observation/state": current_state,
                    "prompt": [prompt] * num_envs,
                }
                if idm_history is not None:
                    obs_dict.update(idm_history.as_obs_dict())
                request_start = time.perf_counter()
                result = policy.infer(obs_dict)
                client_request_ms.append((time.perf_counter() - request_start) * 1000.0)
                server_timing_ms = _extract_server_timing_ms(result)
                infer_ms = server_timing_ms.get("infer_ms")
                if infer_ms is not None:
                    server_infer_ms.append(infer_ms)
                for key, value in server_timing_ms.items():
                    if key != "infer_ms":
                        server_stage_timings_ms.setdefault(key, []).append(value)
                action_chunk = np.clip(result["actions"], -1.0, 1.0).astype(np.float32)

                assert (
                    action_chunk.ndim == 3
                ), f"Model output must have shape (batch_size, action_horizon, action_dim), but got {action_chunk.shape}"
                assert action_chunk.shape[1] >= args.replan_steps, "Model must output at least replan_steps actions"
                for t in range(args.replan_steps):
                    action_plan.append(action_chunk[:, t, :])

            action = action_plan.popleft()
            state_before_step = _policy_state(obs, state_dim=policy_state_dim) if idm_history is not None else None
            obs, reward, terminated, truncated, info = env.step(action)
            if idm_history is not None:
                # Store the clipped action actually passed to env.step, matching what the env executes.
                idm_history.append(state_before_step, action)
            done_rows = np.asarray(terminated, dtype=bool).reshape(-1) | np.asarray(truncated, dtype=bool).reshape(-1)
            if done_rows.shape != (num_envs,):
                raise ValueError(f"Vector env done mask must have shape ({num_envs},), got {done_rows.shape}.")
            if done_rows.any():
                if idm_history is not None:
                    idm_history.reset_rows(done_rows)
                # Action chunks are queued for the whole vector batch. When any row resets,
                # clear the batch plan so the next step replans against fresh per-row history.
                action_plan.clear()
            camera_views = info["cameras"]
            total_reward += reward
            step_success = np.asarray(info.get("success", np.zeros(num_envs)), dtype=bool)
            success |= step_success
            if success.all():
                break

            pbar.set_postfix(reward=f"{total_reward.mean():.1f}", success=f"{success.mean():.0%}")

    record = _episode_record(
        env_name=args.env_name,
        episode=episode,
        num_envs=num_envs,
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
        total_reward=total_reward,
        success=success,
        server_infer_ms=server_infer_ms,
        client_request_ms=client_request_ms,
        video=os.path.basename(video_path),
        server_stage_timings_ms=server_stage_timings_ms,
    )
    record_path = _write_episode_record(output_dir, episode, record)

    server_summary = record.get("server_timing_ms")
    latency_note = f", server_infer_ms_mean={server_summary['mean_ms']:.1f}" if server_summary else ""
    logger.info(
        f"Episode {episode + 1}/{args.num_episodes}: "
        f"mean_reward={total_reward.mean():.2f}, success_rate={success.mean():.2f}{latency_note}, "
        f"video={video_path}, record={record_path}"
    )
    return total_reward, success


def main(args: Args) -> None:
    np.random.seed(args.seed)

    policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    policy_metadata = policy.get_server_metadata()
    logger.info(f"Server metadata: {policy_metadata}")

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
            _, success = run_episode(env, policy, args, episode, output_dir, policy_metadata=policy_metadata)
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
