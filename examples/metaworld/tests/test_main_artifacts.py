"""Tests for the per-episode timing/result artifacts written by main.py.

These cover the pure helpers that build and persist the per-episode JSON
sidecar (``episode_XXX.json``) written next to each rollout video. They do not
spin up a policy server or a real MetaWorld env, so they run on CPU in well
under a second inside the root repo venv:

    cd examples/metaworld
    uv run pytest tests/test_main_artifacts.py -v
"""

from __future__ import annotations

import contextlib
import json

import main
from main import IdmHistoryBuffer
from main import _episode_record
from main import _extract_server_infer_ms
from main import _extract_server_timing_ms
from main import _latency_summary
from main import _write_episode_record
import numpy as np


class TestLatencySummary:
    def test_returns_none_for_no_samples(self):
        assert _latency_summary([]) is None

    def test_single_sample(self):
        assert _latency_summary([12.5]) == {
            "count": 1,
            "mean_ms": 12.5,
            "min_ms": 12.5,
            "max_ms": 12.5,
            "p50_ms": 12.5,
            "total_ms": 12.5,
        }

    def test_multiple_samples(self):
        summary = _latency_summary([10.0, 20.0, 30.0])
        assert summary["count"] == 3
        assert summary["mean_ms"] == 20.0
        assert summary["min_ms"] == 10.0
        assert summary["max_ms"] == 30.0
        assert summary["p50_ms"] == 20.0
        assert summary["total_ms"] == 60.0


class TestEpisodeRecord:
    def _record(self, **overrides):
        kwargs = {
            "env_name": "reach-v3",
            "episode": 0,
            "num_envs": 2,
            "max_steps": 300,
            "replan_steps": 10,
            "total_reward": np.array([1.5, 2.5]),
            "success": np.array([True, False]),
            "server_infer_ms": [40.0, 60.0],
            "client_request_ms": [50.0, 70.0],
            "video": "episode_000.mp4",
        }
        kwargs.update(overrides)
        return _episode_record(**kwargs)

    def test_core_result_fields(self):
        record = self._record()
        assert record["env_name"] == "reach-v3"
        assert record["episode"] == 0
        assert record["num_envs"] == 2
        assert record["max_steps"] == 300
        assert record["replan_steps"] == 10
        assert record["success"] == [True, False]
        assert record["total_reward"] == [1.5, 2.5]
        assert record["success_rate"] == 0.5
        assert record["mean_reward"] == 2.0
        assert record["num_inference_requests"] == 2
        assert record["video"] == "episode_000.mp4"

    def test_includes_server_and_client_timing(self):
        record = self._record()
        assert record["server_timing_ms"]["count"] == 2
        assert record["server_timing_ms"]["mean_ms"] == 50.0
        assert record["server_timing_ms"]["per_request"] == [40.0, 60.0]
        assert record["client_timing_ms"]["per_request"] == [50.0, 70.0]

    def test_includes_server_stage_timing_when_present(self):
        record = self._record(
            server_stage_timings_ms={
                "future_provider_ms": [30.0, 45.0],
                "idm_ms": [10.0, 15.0],
            }
        )

        stages = record["server_timing_ms"]["stages"]
        assert stages["future_provider_ms"]["mean_ms"] == 37.5
        assert stages["future_provider_ms"]["per_request"] == [30.0, 45.0]
        assert stages["idm_ms"]["mean_ms"] == 12.5

    def test_omits_server_timing_when_absent(self):
        # Backward compat: a non-world-model server returns no server_timing,
        # so the server section is dropped rather than filled with zeros.
        record = self._record(server_infer_ms=[])
        assert "server_timing_ms" not in record
        # Client-side request latency + request count are still recorded.
        assert record["num_inference_requests"] == 2
        assert record["client_timing_ms"]["count"] == 2

    def test_record_is_json_serializable(self):
        # numpy scalars must be converted to native types so json.dump works
        # mid-rollout. json.dumps raises TypeError on raw numpy types.
        record = self._record()
        roundtripped = json.loads(json.dumps(record))
        assert roundtripped["success"] == [True, False]
        assert roundtripped["server_timing_ms"]["per_request"] == [40.0, 60.0]


class TestExtractServerInferMs:
    def test_returns_infer_ms_when_present(self):
        result = {"actions": [], "server_timing": {"infer_ms": 42.0}}
        assert _extract_server_infer_ms(result) == 42.0

    def test_returns_none_when_no_server_timing(self):
        # Standard (non-world-model) policy servers return only actions.
        assert _extract_server_infer_ms({"actions": []}) is None

    def test_returns_none_when_server_timing_lacks_infer_ms(self):
        assert _extract_server_infer_ms({"server_timing": {"other": 1.0}}) is None

    def test_returns_none_when_server_timing_not_a_dict(self):
        assert _extract_server_infer_ms({"server_timing": "oops"}) is None

    def test_returns_none_when_infer_ms_is_not_numeric(self):
        assert _extract_server_infer_ms({"server_timing": {"infer_ms": "oops"}}) is None


class TestExtractServerTimingMs:
    def test_returns_numeric_timing_fields(self):
        result = {
            "actions": [],
            "server_timing": {
                "infer_ms": 42.0,
                "future_provider_ms": "30.5",
                "idm_ms": np.float32(11.5),
                "bad": "oops",
            },
        }

        assert _extract_server_timing_ms(result) == {
            "infer_ms": 42.0,
            "future_provider_ms": 30.5,
            "idm_ms": 11.5,
        }

    def test_returns_empty_dict_when_no_server_timing(self):
        assert _extract_server_timing_ms({"actions": []}) == {}


class TestWriteEpisodeRecord:
    def test_writes_json_sidecar_and_returns_path(self, tmp_path):
        record = {"env_name": "reach-v3", "episode": 3, "success_rate": 1.0}
        path = _write_episode_record(str(tmp_path), 3, record)
        assert path == str(tmp_path / "episode_003.json")
        with open(path) as f:
            assert json.load(f) == record


class TestIdmHistoryBuffer:
    def test_rolls_masks_and_resets(self):
        buffer = IdmHistoryBuffer(num_envs=1, history_length=2, state_dim=4, action_dim=4)

        assert buffer.history_mask.tolist() == [[0.0, 0.0]]
        buffer.append(np.ones((1, 4), dtype=np.float32), np.full((1, 4), 2.0, dtype=np.float32))
        assert buffer.history_mask.tolist() == [[0.0, 1.0]]
        buffer.append(np.full((1, 4), 3.0, dtype=np.float32), np.full((1, 4), 4.0, dtype=np.float32))
        assert buffer.history_mask.tolist() == [[1.0, 1.0]]
        assert buffer.prev_state_history.tolist() == [[[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]]]

        buffer.reset()

        assert buffer.history_mask.tolist() == [[0.0, 0.0]]
        assert np.count_nonzero(buffer.prev_state_history) == 0
        assert np.count_nonzero(buffer.prev_action_history) == 0

    def test_resets_selected_rows_only(self):
        buffer = IdmHistoryBuffer(num_envs=2, history_length=2, state_dim=4, action_dim=4)
        state = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
        action = np.array([[3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]], dtype=np.float32)
        buffer.append(state, action)

        buffer.reset_rows(np.array([True, False]))

        assert buffer.history_mask.tolist() == [[0.0, 0.0], [0.0, 1.0]]
        assert np.count_nonzero(buffer.prev_state_history[0]) == 0
        assert np.count_nonzero(buffer.prev_action_history[0]) == 0
        assert buffer.prev_state_history[1, -1].tolist() == [2.0, 2.0, 2.0, 2.0]
        assert buffer.prev_action_history[1, -1].tolist() == [4.0, 4.0, 4.0, 4.0]


# --------------------------------------------------------------------------------------
# run_episode integration: drive the loop with a fake vectorized env + fake policy and a
# stubbed video writer, so no MuJoCo / pyav / policy server is required.
# --------------------------------------------------------------------------------------
class _FakeEnv:
    def __init__(self, num_envs: int = 2, img: int = 8):
        self.num_envs = num_envs
        self._img = img
        self._cams = ("corner", "corner4", "gripperPOV")

    def _info(self):
        cams = {c: np.zeros((self.num_envs, self._img, self._img, 3), dtype=np.uint8) for c in self._cams}
        return {"cameras": cams, "success": np.zeros(self.num_envs, dtype=bool)}

    def reset(self, seed=None):
        return np.zeros((self.num_envs, 4), dtype=np.float32), self._info()

    def step(self, action):
        zeros = np.zeros(self.num_envs, dtype=bool)
        return np.zeros((self.num_envs, 4), dtype=np.float32), np.ones(self.num_envs), zeros, zeros, self._info()

    def close(self):
        pass


class _TerminatingRowEnv(_FakeEnv):
    def __init__(self, num_envs: int = 2, img: int = 8):
        super().__init__(num_envs=num_envs, img=img)
        self.steps = 0
        self.actions = []

    def reset(self, seed=None):
        self.steps = 0
        self.actions = []
        return super().reset(seed=seed)

    def step(self, action):
        self.steps += 1
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        terminated = np.array([self.steps == 1, False], dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        obs = np.full((self.num_envs, 4), float(self.steps), dtype=np.float32)
        return obs, np.ones(self.num_envs), terminated, truncated, self._info()


class _FakePolicy:
    def __init__(
        self,
        infer_ms: float | None = 12.5,
        future_provider_ms: float | None = 7.5,
        idm_ms: float | None = 4.5,
        horizon: int = 10,
        dim: int = 4,
    ):
        self.infer_ms, self.horizon, self.dim, self.calls = infer_ms, horizon, dim, 0
        self.future_provider_ms = future_provider_ms
        self.idm_ms = idm_ms
        self.observations = []

    def infer(self, obs):
        self.calls += 1
        self.observations.append(
            {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in obs.items()}
        )
        batch = obs["observation/state"].shape[0]
        out = {"actions": np.zeros((batch, self.horizon, self.dim), dtype=np.float32)}
        if self.infer_ms is not None:
            out["server_timing"] = {"infer_ms": self.infer_ms}
            if self.future_provider_ms is not None:
                out["server_timing"]["future_provider_ms"] = self.future_provider_ms
            if self.idm_ms is not None:
                out["server_timing"]["idm_ms"] = self.idm_ms
        return out


class _ReplanningPolicy(_FakePolicy):
    def __init__(self):
        super().__init__(infer_ms=None, horizon=2, dim=4)

    def infer(self, obs):
        self.calls += 1
        self.observations.append(
            {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in obs.items()}
        )
        batch = obs["observation/state"].shape[0]
        actions = np.zeros((batch, self.horizon, self.dim), dtype=np.float32)
        base = 0.2 + 0.3 * (self.calls - 1)
        actions[:, 0, :] = base
        actions[:, 1, :] = base + 0.05
        return {"actions": actions}


class _NoopVideo:
    def init_video_stream(self, *args, **kwargs):
        pass

    def write_frame(self, frame):
        pass


@contextlib.contextmanager
def _noop_imopen(*args, **kwargs):
    yield _NoopVideo()


class TestRunEpisodeArtifacts:
    def test_writes_sidecar_with_server_timing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main.iio, "imopen", _noop_imopen)
        policy = _FakePolicy(infer_ms=12.5)
        args = main.Args(env_name="reach-v3", num_envs=2, max_steps=30, replan_steps=10)

        total_reward, success = main.run_episode(_FakeEnv(2), policy, args, episode=0, output_dir=str(tmp_path))

        # Backward-compatible (total_reward, success) contract relied on by eval_all.py.
        assert total_reward.shape == (2,)
        assert success.shape == (2,)

        record = json.loads((tmp_path / "episode_000.json").read_text())
        # 30 steps / 10 replan_steps => 3 inference requests.
        assert record["num_inference_requests"] == policy.calls == 3
        assert record["server_timing_ms"]["count"] == 3
        assert record["server_timing_ms"]["mean_ms"] == 12.5
        assert record["server_timing_ms"]["stages"]["future_provider_ms"]["mean_ms"] == 7.5
        assert record["server_timing_ms"]["stages"]["idm_ms"]["per_request"] == [4.5, 4.5, 4.5]
        assert record["client_timing_ms"]["count"] == 3
        assert record["video"] == "episode_000.mp4"

    def test_sends_client_supplied_idm_history_when_metadata_requests_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main.iio, "imopen", _noop_imopen)
        policy = _FakePolicy(infer_ms=None, horizon=1, dim=4)
        args = main.Args(env_name="reach-v3", num_envs=1, max_steps=3, replan_steps=1)

        main.run_episode(
            _FakeEnv(1),
            policy,
            args,
            episode=0,
            output_dir=str(tmp_path),
            policy_metadata={"idm_history_length": 2, "state_dim": 4, "action_dim": 4},
        )

        assert policy.calls == 3
        assert policy.observations[0]["history_mask"].tolist() == [[0.0, 0.0]]
        assert policy.observations[1]["history_mask"].tolist() == [[0.0, 1.0]]
        assert policy.observations[2]["history_mask"].tolist() == [[1.0, 1.0]]
        assert policy.observations[0]["prev_state_history"].shape == (1, 2, 4)
        assert policy.observations[0]["prev_action_history"].shape == (1, 2, 4)

    def test_done_row_resets_history_and_clears_queued_batch_plan(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main.iio, "imopen", _noop_imopen)
        env = _TerminatingRowEnv(2)
        policy = _ReplanningPolicy()
        args = main.Args(env_name="reach-v3", num_envs=2, max_steps=3, replan_steps=2)

        main.run_episode(
            env,
            policy,
            args,
            episode=0,
            output_dir=str(tmp_path),
            policy_metadata={"idm_history_length": 2, "state_dim": 4, "action_dim": 4},
        )

        assert policy.calls == 2
        assert np.allclose(env.actions[0], np.full((2, 4), 0.2, dtype=np.float32))
        assert np.allclose(env.actions[1], np.full((2, 4), 0.5, dtype=np.float32))
        assert policy.observations[1]["history_mask"].tolist() == [[0.0, 0.0], [0.0, 1.0]]
        assert np.count_nonzero(policy.observations[1]["prev_state_history"][0]) == 0
        assert np.count_nonzero(policy.observations[1]["prev_action_history"][0]) == 0

    def test_sidecar_omits_server_timing_for_plain_server(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main.iio, "imopen", _noop_imopen)
        policy = _FakePolicy(infer_ms=None)  # standard server: no server_timing in response
        args = main.Args(env_name="reach-v3", num_envs=2, max_steps=30, replan_steps=10)

        main.run_episode(_FakeEnv(2), policy, args, episode=0, output_dir=str(tmp_path))

        record = json.loads((tmp_path / "episode_000.json").read_text())
        assert "server_timing_ms" not in record
        # Request count + client-side latency are still captured.
        assert record["num_inference_requests"] == 3
        assert record["client_timing_ms"]["count"] == 3
