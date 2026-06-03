"""Tests for the robocasa_env example.

These tests run inside `examples/robocasa_env/.venv` (the dedicated robocasa
venv), so they can `import main` and `import eval_all` directly without any
mock-the-modules-out shenanigans. `examples/robocasa_env/tests/conftest.py`
adds the example dir to sys.path.

Run them from this directory:

    uv sync                           # one-time, installs the robocasa venv
    MUJOCO_GL=osmesa uv run pytest    # CPU-only software renderer
    # or:
    MUJOCO_GL=egl uv run pytest       # faster, requires a GPU

The pure-Python tests in this file (Args / CAMERA_KEYS / build_state /
tile_frames / eval_task signature / eval_all wiring) need neither GPU nor
the kitchen assets — they exercise the helpers from main.py and eval_all.py
on synthetic numpy arrays.

The TestRobocasaEnv class is marked manual because constructing a real
robocasa env requires `robocasa.scripts.download_kitchen_assets` to have
been run on the runner (~10 GB download), which is impractical for CI.
Run them locally after assets are downloaded:

    uv run pytest tests/test_robocasa_env.py::TestRobocasaEnv -m manual -v
"""

from __future__ import annotations

import dataclasses
import inspect
import math

import numpy as np
import pytest

import eval_all
import main
from main import CAMERA_KEYS, Args, build_state, eval_task, tile_frames

# ── Args ──────────────────────────────────────────────────────────────────────


class TestArgs:
    def test_defaults(self) -> None:
        args = Args()
        assert args.host == "0.0.0.0"
        assert args.port == 8000
        assert args.env_name == "CloseBlenderLid"
        assert args.split == "pretrain"
        assert args.num_episodes == 1
        assert args.max_steps is None
        assert args.replan_steps == 5
        assert args.resize_size == 224
        assert args.fps == 24
        assert args.seed == 7

    def test_default_render_cameras_are_independent_instances(self) -> None:
        """Each Args() instantiation must get its own list to avoid shared mutable state."""
        a, b = Args(), Args()
        assert a.render_cameras == b.render_cameras
        assert a.render_cameras is not b.render_cameras
        a.render_cameras.append("extra")
        assert "extra" not in b.render_cameras

    def test_default_render_cameras_are_valid_camera_keys(self) -> None:
        for cam in Args().render_cameras:
            assert (
                cam in CAMERA_KEYS
            ), f"Default render camera '{cam}' not in CAMERA_KEYS"

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(Args)


# ── CAMERA_KEYS ───────────────────────────────────────────────────────────────


class TestCameraKeys:
    def test_contains_expected_cameras(self) -> None:
        expected = {"agentview_left", "agentview_right", "eye_in_hand"}
        assert set(CAMERA_KEYS.keys()) == expected

    def test_values_match_obs_format(self) -> None:
        """Values must match the obs dict keys produced by robocasa's gym wrapper."""
        assert CAMERA_KEYS["agentview_left"] == "video.robot0_agentview_left"
        assert CAMERA_KEYS["agentview_right"] == "video.robot0_agentview_right"
        assert CAMERA_KEYS["eye_in_hand"] == "video.robot0_eye_in_hand"

    def test_values_use_video_prefix(self) -> None:
        for value in CAMERA_KEYS.values():
            assert value.startswith("video.robot0_"), f"Bad camera key value: {value}"


# ── build_state ───────────────────────────────────────────────────────────────


class TestBuildState:
    def test_concatenates_in_order(self) -> None:
        obs = {
            "state.end_effector_position_relative": np.array([1.0, 2.0, 3.0]),
            "state.end_effector_rotation_relative": np.array([4.0, 5.0, 6.0, 7.0]),
            "state.base_position": np.array([8.0, 9.0, 10.0]),
            "state.base_rotation": np.array([11.0, 12.0, 13.0, 14.0]),
            "state.gripper_qpos": np.array([15.0, 16.0]),
        }
        state = build_state(obs)
        assert state.shape == (16,)
        expected = np.arange(1.0, 17.0)
        np.testing.assert_array_equal(state, expected)

    def test_returns_16_dims(self) -> None:
        """The robocasa policy expects exactly 16 proprioceptive dims."""
        obs = {
            "state.end_effector_position_relative": np.zeros(3),
            "state.end_effector_rotation_relative": np.zeros(4),
            "state.base_position": np.zeros(3),
            "state.base_rotation": np.zeros(4),
            "state.gripper_qpos": np.zeros(2),
        }
        state = build_state(obs)
        assert state.shape == (16,)

    def test_preserves_dtype(self) -> None:
        obs = {
            "state.end_effector_position_relative": np.zeros(3, dtype=np.float32),
            "state.end_effector_rotation_relative": np.zeros(4, dtype=np.float32),
            "state.base_position": np.zeros(3, dtype=np.float32),
            "state.base_rotation": np.zeros(4, dtype=np.float32),
            "state.gripper_qpos": np.zeros(2, dtype=np.float32),
        }
        assert build_state(obs).dtype == np.float32

    def test_field_layout(self) -> None:
        """Verify each slice of the output corresponds to the right field."""
        obs = {
            "state.end_effector_position_relative": np.array([0.1, 0.2, 0.3]),
            "state.end_effector_rotation_relative": np.array([0.4, 0.5, 0.6, 0.7]),
            "state.base_position": np.array([0.8, 0.9, 1.0]),
            "state.base_rotation": np.array([1.1, 1.2, 1.3, 1.4]),
            "state.gripper_qpos": np.array([1.5, 1.6]),
        }
        state = build_state(obs)
        np.testing.assert_array_equal(
            state[0:3], obs["state.end_effector_position_relative"]
        )
        np.testing.assert_array_equal(
            state[3:7], obs["state.end_effector_rotation_relative"]
        )
        np.testing.assert_array_equal(state[7:10], obs["state.base_position"])
        np.testing.assert_array_equal(state[10:14], obs["state.base_rotation"])
        np.testing.assert_array_equal(state[14:16], obs["state.gripper_qpos"])


# ── tile_frames ───────────────────────────────────────────────────────────────


class TestTileFrames:
    def test_single(self) -> None:
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        result = tile_frames([frame])
        assert result.shape == (224, 224, 3)

    def test_three_frames_2x2_grid(self) -> None:
        """3 frames → 2x2 grid (one black slot)."""
        frames = [np.ones((64, 64, 3), dtype=np.uint8) * 255 for _ in range(3)]
        result = tile_frames(frames)
        assert result.shape == (128, 128, 3)
        # Bottom-right slot must be black-padded.
        assert result[64:, 64:].max() == 0

    def test_four_frames_2x2_grid(self) -> None:
        frames = [np.ones((224, 224, 3), dtype=np.uint8) * i for i in range(4)]
        result = tile_frames(frames)
        assert result.shape == (448, 448, 3)

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 9, 10])
    def test_grid_shape(self, n: int) -> None:
        """Output shape matches metaworld's grid layout for any N."""
        h, w = 32, 32
        frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]
        result = tile_frames(frames)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        assert result.shape == (rows * h, cols * w, 3)

    def test_preserves_dtype(self) -> None:
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        assert tile_frames(frames).dtype == np.uint8

    def test_places_frames_in_row_major_order(self) -> None:
        """Frame i should land at grid position (i // cols, i % cols)."""
        n = 3
        h, w = 16, 16
        cols = math.ceil(math.sqrt(n))  # = 2
        # Mark each frame with a unique value so we can find it.
        frames = [np.full((h, w, 3), i + 1, dtype=np.uint8) for i in range(n)]
        grid = tile_frames(frames)
        for i in range(n):
            r, c = divmod(i, cols)
            cell = grid[r * h : (r + 1) * h, c * w : (c + 1) * w]
            assert (
                cell[0, 0, 0] == i + 1
            ), f"Frame {i} not at position (row={r}, col={c})"


# ── eval_task signature ───────────────────────────────────────────────────────


class TestEvalTaskSignature:
    def test_is_callable_and_documented(self) -> None:
        """eval_task must be a top-level function (so eval_all.py can import it)."""
        assert callable(eval_task)
        assert eval_task.__doc__ is not None
        assert eval_task.__doc__.strip() != ""

    def test_signature(self) -> None:
        """eval_task takes (env_name, policy, args, output_dir)."""
        sig = inspect.signature(eval_task)
        params = list(sig.parameters)
        assert params == [
            "env_name",
            "policy",
            "args",
            "output_dir",
        ], f"Expected (env_name, policy, args, output_dir), got {params}"

    def test_args_field_compatibility(self) -> None:
        """eval_task's args parameter is duck-typed; main.Args must provide all the
        fields it accesses (split, num_episodes, max_steps, replan_steps, resize_size,
        render_cameras, fps, seed)."""
        required = {
            "split",
            "num_episodes",
            "max_steps",
            "replan_steps",
            "resize_size",
            "render_cameras",
            "fps",
            "seed",
        }
        field_names = {f.name for f in dataclasses.fields(Args)}
        missing = required - field_names
        assert not missing, f"main.Args missing fields needed by eval_task: {missing}"


class TestDefaultOutputDirDoesNotDoubleEnvName:
    """Regression guard for a bug where main.py's standalone-mode default
    ``output_dir`` included ``args.env_name``. Since ``eval_task``
    unconditionally appends ``env_name`` again, that doubled the path to
    ``output/{env_name}/{env_name}/episode_NNN.mp4`` — contradicting the
    documented layout ``output/{env_name}/...``.

    The contract locked in here:
    1. ``main()``'s ``else`` branch (no ``--output_dir``) must NOT reference
       ``env_name`` — the bare ``output/`` parent is correct because
       ``eval_task`` provides the per-env nesting.
    2. ``eval_task`` MUST keep its inner ``os.path.join(output_dir, env_name)``
       so ``eval_all.py`` (which forwards an already-per-task-free parent) gets
       the per-env subdir.
    """

    def test_main_default_output_dir_is_bare_output(self) -> None:
        src = inspect.getsource(main.main)
        # Pull the line that computes the default ``output_dir`` in the
        # ``else`` branch: ``output_dir = os.path.join(...)`` that references
        # both ``__file__`` and the literal ``"output"``.
        candidate = next(
            (
                line.strip()
                for line in src.splitlines()
                if "os.path.join" in line and '"output"' in line and "__file__" in line
            ),
            None,
        )
        assert candidate is not None, (
            "main.main must assign a default ``output_dir`` via "
            "``os.path.join(os.path.dirname(__file__), 'output', ...)``"
        )
        # The default must not reference ``env_name`` — eval_task adds it.
        assert "env_name" not in candidate, (
            "main.py standalone default output_dir includes env_name, which "
            "eval_task's inner join will double to "
            "``output/{env_name}/{env_name}/...``. "
            f"Offending line: {candidate!r}"
        )

    def test_eval_task_still_nests_env_name(self) -> None:
        src = inspect.getsource(eval_task)
        assert "os.path.join(output_dir, env_name)" in src, (
            "eval_task must keep ``os.path.join(output_dir, env_name)`` so the "
            "final per-task tree is ``{output_dir}/{env_name}/...``. Removing "
            "this nesting would break eval_all.py's batch layout."
        )


# ── eval_all wiring ───────────────────────────────────────────────────────────


class TestEvalAll:
    def test_args_defaults(self) -> None:
        args = eval_all.Args()
        assert args.host == "0.0.0.0"
        assert args.port == 8000
        assert args.task_set == "subset"
        assert args.tasks == []
        assert args.split == "pretrain"
        assert args.num_episodes == 15
        assert args.max_steps is None
        assert args.replan_steps == 5
        assert args.resize_size == 224
        assert args.fps == 24
        assert args.seed == 7

    def test_args_default_render_cameras_match_main(self) -> None:
        """eval_all.Args.render_cameras default should match main.Args.render_cameras default."""
        assert eval_all.Args().render_cameras == Args().render_cameras

    def test_args_default_render_cameras_are_independent_instances(self) -> None:
        a, b = eval_all.Args(), eval_all.Args()
        assert a.render_cameras is not b.render_cameras

    def test_args_provides_eval_task_required_fields(self) -> None:
        """eval_all.Args must satisfy the same duck-typed contract as main.Args
        so it can be passed to eval_task."""
        required = {
            "split",
            "num_episodes",
            "max_steps",
            "replan_steps",
            "resize_size",
            "render_cameras",
            "fps",
            "seed",
        }
        field_names = {f.name for f in dataclasses.fields(eval_all.Args)}
        missing = required - field_names
        assert (
            not missing
        ), f"eval_all.Args missing fields needed by eval_task: {missing}"

    def test_does_not_import_eval_task_in_parent_process(self) -> None:
        """After the parallel-subprocess migration, eval_all.py dispatches each
        env to its own main.py subprocess and never calls eval_task in the
        parent process. So eval_all must NOT have an ``eval_task`` attribute
        bound at module level — that would imply the old in-process
        architecture. main.eval_task itself still exists (subprocesses import
        it), but eval_all's own module namespace should not."""
        assert not hasattr(eval_all, "eval_task")
        # Sanity: main.eval_task still exists and is the one subprocesses use.
        assert callable(eval_task)

    def test_main_rejects_unknown_task_set(self) -> None:
        """Passing an unknown task_set must raise a clear ValueError before any
        server connection happens. Uses the real robocasa registry from the venv."""
        args = eval_all.Args(task_set="this_task_set_does_not_exist")
        with pytest.raises(ValueError, match="Unknown task_set"):
            eval_all.main(args)

    def test_supported_task_sets_present_in_real_registry(self) -> None:
        """The user-facing task sets actually exist in robocasa's real registry
        (not a mock). Catches drift if robocasa renames or removes a task set."""
        from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

        for name in ["atomic_seen", "composite_seen", "composite_unseen"]:
            assert (
                name in TASK_SET_REGISTRY
            ), f"'{name}' missing from real TASK_SET_REGISTRY"


# ── env smoke tests (manual: need ~10GB kitchen assets) ───────────────────────


@pytest.mark.manual
class TestRobocasaEnv:
    """End-to-end smoke tests against a real robocasa env. Need
    `robocasa.scripts.download_kitchen_assets` to have been run on the
    runner (~10GB download), which is impractical for CI. Run locally:

        cd examples/robocasa_env
        uv run python -m robocasa.scripts.download_kitchen_assets
        MUJOCO_GL=osmesa uv run pytest tests/test_robocasa_env.py::TestRobocasaEnv -m manual -v
    """

    def test_seed_controls_initial_state(self) -> None:
        """Different ``--seed`` values yield different initial env observations.

        Regression test for the claim that RoboCasa's seed (passed into
        ``gym.make(..., seed=args.seed)`` at env construction) actually
        randomizes the scene per seed. The env's internal RNG is seeded
        once at construction; each ``env.reset()`` then draws a fresh
        configuration from that seeded RNG stream.
        """
        from main import make_env

        env_a = make_env(env_name="CloseBlenderLid", split="pretrain", seed=100)
        env_b = make_env(env_name="CloseBlenderLid", split="pretrain", seed=200)
        env_a_repeat = make_env(env_name="CloseBlenderLid", split="pretrain", seed=100)
        try:
            obs_a, _ = env_a.reset()
            obs_b, _ = env_b.reset()
            obs_a_repeat, _ = env_a_repeat.reset()

            # Different seeds → different scene → different first-camera image.
            cam_key = next(iter(CAMERA_KEYS.values()))
            diff = float(
                np.abs(
                    obs_a[cam_key].astype(np.int32) - obs_b[cam_key].astype(np.int32)
                ).mean()
            )
            assert diff > 1.0, (
                f"expected seed=100 vs seed=200 to render different initial scenes "
                f"(mean |Δpixel| > 1.0), got {diff:.3f}"
            )

            # Same seed → same initial scene (deterministic construction).
            np.testing.assert_array_equal(obs_a[cam_key], obs_a_repeat[cam_key])
        finally:
            env_a.close()
            env_b.close()
            env_a_repeat.close()

    def test_make_env_reset_and_step(self) -> None:
        """The env can be created, reset, and stepped once with a no-op action."""
        from main import make_env

        env = make_env(env_name="CloseBlenderLid", split="pretrain", seed=7)
        try:
            obs, info = env.reset()
            # All three cameras should be present in the obs dict.
            for key in CAMERA_KEYS.values():
                assert key in obs, f"camera key '{key}' missing from obs"

            # Proprioceptive state should produce a 16-dim vector.
            state = build_state(obs)
            assert state.shape == (16,)

            # Step once with a zero action (12 dims). RoboCasa wraps this via convert_action.
            from robocasa.utils.env_utils import convert_action

            action = convert_action(np.zeros(12, dtype=np.float32))
            obs2, reward, terminated, truncated, info2 = env.step(action)
            assert isinstance(reward, int | float)
            assert isinstance(terminated, bool) or isinstance(truncated, bool)
            for key in CAMERA_KEYS.values():
                assert key in obs2
        finally:
            env.close()

    def test_eval_task_runs_with_stub_policy_and_writes_video(self, tmp_path) -> None:
        """The full eval_task loop runs to completion with a stub policy and
        writes a video. Uses max_steps=2 to keep the rollout short."""

        class StubPolicy:
            """Returns 12-dim zero actions matching robocasa's action space."""

            def infer(self, element: dict) -> dict:
                # eval_task hands us a single-example obs dict (no batch dim).
                assert element["observation/image"].shape == (224, 224, 3)
                assert element["observation/wrist_image"].shape == (224, 224, 3)
                assert element["observation/state"].shape == (16,)
                assert isinstance(element["prompt"], str)
                return {"actions": np.zeros((10, 12), dtype=np.float32)}

        args = Args(
            num_episodes=1,
            max_steps=2,
            replan_steps=1,
            fps=2,
            seed=7,
        )
        result = eval_task(
            env_name="CloseBlenderLid",
            policy=StubPolicy(),
            args=args,
            output_dir=str(tmp_path),
        )

        assert set(result.keys()) >= {"success_rate", "num_episodes"}
        assert result["success_rate"] in {0.0, 1.0}
        assert result["num_episodes"] == 1.0

        # eval_task writes one .mp4 per episode under <output_dir>/<env_name>/episode_NNN.mp4
        videos = sorted(tmp_path.rglob("*.mp4"))
        assert len(videos) == 1, f"expected 1 video, got {len(videos)}: {videos}"
        assert videos[0].stat().st_size > 0
