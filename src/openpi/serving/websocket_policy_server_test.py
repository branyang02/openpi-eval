import asyncio
import contextlib

import numpy as np
from openpi_client import base_policy

from openpi.serving import websocket_policy_server


class RecordingPolicy(base_policy.BasePolicy):
    def __init__(self) -> None:
        self.single_calls: list[int] = []
        self.batches: list[list[int]] = []

    def infer(self, obs: dict) -> dict:
        request_id = int(obs["id"])
        self.single_calls.append(request_id)
        return {"actions": np.asarray([request_id])}

    def infer_many(self, obs: list[dict]) -> list[dict]:
        request_ids = [int(item["id"]) for item in obs]
        self.batches.append(request_ids)
        return [{"actions": np.asarray([request_id])} for request_id in request_ids]


async def _stop_worker(worker: asyncio.Task) -> None:
    worker.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker


def test_infer_action_uses_single_request_path_by_default() -> None:
    async def run() -> None:
        policy = RecordingPolicy()
        server = websocket_policy_server.WebsocketPolicyServer(policy)

        output, timing = await server._infer_action({"id": 7})  # noqa: SLF001

        assert int(output["actions"][0]) == 7
        assert timing["infer_ms"] >= 0
        assert policy.single_calls == [7]
        assert policy.batches == []

    asyncio.run(run())


def test_batch_worker_groups_concurrent_requests_and_preserves_order() -> None:
    async def run() -> None:
        policy = RecordingPolicy()
        server = websocket_policy_server.WebsocketPolicyServer(
            policy,
            max_batch_size=4,
            min_batch_size=4,
            max_batch_wait_ms=100,
        )
        server._request_queue = asyncio.Queue()  # noqa: SLF001
        worker = asyncio.create_task(server._batch_worker())  # noqa: SLF001

        try:
            results = await asyncio.gather(*(server._infer_action({"id": i}) for i in range(4)))  # noqa: SLF001
        finally:
            await _stop_worker(worker)

        assert policy.single_calls == []
        assert policy.batches == [[0, 1, 2, 3]]
        assert [int(output["actions"][0]) for output, _ in results] == [0, 1, 2, 3]
        assert [timing["batch_size"] for _, timing in results] == [4, 4, 4, 4]
        assert [timing["padded_batch_size"] for _, timing in results] == [4, 4, 4, 4]
        assert all(timing["infer_ms"] >= 0 for _, timing in results)
        assert all(timing["queue_wait_ms"] >= 0 for _, timing in results)

    asyncio.run(run())


def test_batch_worker_flushes_partial_batch_after_wait() -> None:
    async def run() -> None:
        policy = RecordingPolicy()
        server = websocket_policy_server.WebsocketPolicyServer(
            policy,
            max_batch_size=4,
            min_batch_size=4,
            max_batch_wait_ms=1,
        )
        server._request_queue = asyncio.Queue()  # noqa: SLF001
        worker = asyncio.create_task(server._batch_worker())  # noqa: SLF001

        try:
            results = await asyncio.gather(*(server._infer_action({"id": i}) for i in range(3)))  # noqa: SLF001
        finally:
            await _stop_worker(worker)

        assert policy.batches == [[0, 1, 2]]
        assert [int(output["actions"][0]) for output, _ in results] == [0, 1, 2]
        assert [timing["batch_size"] for _, timing in results] == [3, 3, 3]

    asyncio.run(run())


def test_batch_worker_can_pad_to_power_of_two_bucket() -> None:
    async def run() -> None:
        policy = RecordingPolicy()
        server = websocket_policy_server.WebsocketPolicyServer(
            policy,
            max_batch_size=8,
            min_batch_size=3,
            max_batch_wait_ms=100,
            pad_to_batch_bucket=True,
        )
        server._request_queue = asyncio.Queue()  # noqa: SLF001
        worker = asyncio.create_task(server._batch_worker())  # noqa: SLF001

        try:
            results = await asyncio.gather(*(server._infer_action({"id": i}) for i in range(3)))  # noqa: SLF001
        finally:
            await _stop_worker(worker)

        assert policy.batches == [[0, 1, 2, 2]]
        assert [int(output["actions"][0]) for output, _ in results] == [0, 1, 2]
        assert [timing["batch_size"] for _, timing in results] == [3, 3, 3]
        assert [timing["padded_batch_size"] for _, timing in results] == [4, 4, 4]

    asyncio.run(run())
