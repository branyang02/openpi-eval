import asyncio
from collections.abc import Sequence
import concurrent.futures
import contextlib
import threading

import numpy as np
from openpi_client import base_policy
from openpi_client import websocket_client_policy
import pytest
import websockets.asyncio.server as _server

from openpi.serving import websocket_policy_server


class RecordingPolicy(base_policy.BasePolicy):
    def __init__(self) -> None:
        self.single_calls: list[int] = []
        self.batches: list[list[int]] = []
        self.warmup_batches: list[int] = []

    def infer(self, obs: dict) -> dict:
        request_id = int(obs["id"])
        self.single_calls.append(request_id)
        return {"actions": np.asarray([request_id])}

    def infer_many(self, obs: list[dict]) -> list[dict]:
        request_ids = [int(item["id"]) for item in obs]
        self.batches.append(request_ids)
        return [{"actions": np.asarray([request_id])} for request_id in request_ids]

    def warmup_many(self, obs: dict, batch_sizes: Sequence[int]) -> None:
        self.warmup_batches.extend(batch_sizes)
        request_id = int(obs["id"])
        for batch_size in batch_sizes:
            if batch_size == 1:
                self.single_calls.append(request_id)
            else:
                self.batches.append([request_id] * batch_size)


class BlockingPolicy(RecordingPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.first_batch_started = threading.Event()
        self.release_first_batch = threading.Event()

    def infer_many(self, obs: list[dict]) -> list[dict]:
        request_ids = [int(item["id"]) for item in obs]
        self.batches.append(request_ids)
        if len(self.batches) == 1:
            self.first_batch_started.set()
            if not self.release_first_batch.wait(timeout=5):
                raise TimeoutError("Timed out waiting to release the first batch.")
        return [{"actions": np.asarray([request_id])} for request_id in request_ids]


async def _stop_worker(worker: asyncio.Task) -> None:
    worker.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker


async def _wait_until(predicate, timeout_s: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not predicate():
        if loop.time() >= deadline:
            raise TimeoutError("Timed out waiting for condition.")
        await asyncio.sleep(0.001)


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


def test_websocket_microbatch_serving_preserves_client_result_mapping() -> None:
    async def run() -> None:
        policy = RecordingPolicy()
        server = websocket_policy_server.WebsocketPolicyServer(
            policy,
            host="127.0.0.1",
            max_batch_size=8,
            min_batch_size=2,
            max_batch_wait_ms=50,
        )
        server._request_queue = asyncio.Queue()  # noqa: SLF001
        worker = asyncio.create_task(server._batch_worker())  # noqa: SLF001

        try:
            async with _server.serve(
                server._handler,  # noqa: SLF001
                "127.0.0.1",
                0,
                compression=None,
                max_size=None,
                ping_timeout=600,
            ) as ws_server:
                port = ws_server.sockets[0].getsockname()[1]
                results, metadata = await asyncio.to_thread(_run_websocket_clients, port)
        finally:
            await _stop_worker(worker)

        assert metadata["microbatch"]["max_batch_size"] == 8
        assert len(results) == 32 * 4
        for request_id, response_id, batch_size in results:
            assert response_id == request_id
            assert batch_size >= 1
        assert policy.single_calls == []
        assert any(len(batch) > 1 for batch in policy.batches)

    def _run_websocket_clients(port: int) -> tuple[list[tuple[int, int, int]], dict]:
        num_clients = 32
        requests_per_client = 4
        barrier = threading.Barrier(num_clients)
        metadata = {}

        def run_client(client_id: int) -> list[tuple[int, int, int]]:
            client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", port)
            try:
                if client_id == 0:
                    metadata.update(client.get_server_metadata())
                barrier.wait(timeout=5)
                client_results = []
                for request_index in range(requests_per_client):
                    request_id = client_id * 1000 + request_index
                    response = client.infer({"id": request_id})
                    server_timing = response["server_timing"]
                    client_results.append(
                        (
                            request_id,
                            int(response["actions"][0]),
                            int(server_timing.get("batch_size", 1)),
                        )
                    )
                return client_results
            finally:
                client._ws.close()  # noqa: SLF001

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_clients) as pool:
            nested_results = list(pool.map(run_client, range(num_clients)))

        return [item for result in nested_results for item in result], metadata

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


def test_batch_worker_warms_bucket_sizes_once_from_first_request() -> None:
    async def run() -> None:
        policy = RecordingPolicy()
        server = websocket_policy_server.WebsocketPolicyServer(
            policy,
            max_batch_size=8,
            min_batch_size=2,
            max_batch_wait_ms=100,
            pad_to_batch_bucket=True,
            warmup_batch_buckets=True,
        )
        server._request_queue = asyncio.Queue()  # noqa: SLF001
        worker = asyncio.create_task(server._batch_worker())  # noqa: SLF001

        try:
            first_results = await asyncio.gather(*(server._infer_action({"id": i}) for i in range(2)))  # noqa: SLF001
            second_results = await asyncio.gather(*(server._infer_action({"id": i}) for i in range(2, 4)))  # noqa: SLF001
        finally:
            await _stop_worker(worker)

        assert policy.warmup_batches == [1, 2, 4, 8]
        assert policy.single_calls == [0]
        assert policy.batches == [
            [0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1],
            [2, 3],
        ]
        assert [int(output["actions"][0]) for output, _ in first_results + second_results] == [0, 1, 2, 3]
        assert all("warmup_ms" in timing for _, timing in first_results)
        assert all("warmup_ms" not in timing for _, timing in second_results)

    asyncio.run(run())


def test_bucket_aware_batching_waits_for_next_bucket() -> None:
    async def run() -> None:
        policy = RecordingPolicy()
        server = websocket_policy_server.WebsocketPolicyServer(
            policy,
            max_batch_size=4,
            min_batch_size=2,
            max_batch_wait_ms=100,
            pad_to_batch_bucket=True,
            bucket_aware_batching=True,
        )
        server._request_queue = asyncio.Queue()  # noqa: SLF001
        worker = asyncio.create_task(server._batch_worker())  # noqa: SLF001

        try:
            waiting_results = [asyncio.create_task(server._infer_action({"id": i})) for i in range(3)]  # noqa: SLF001
            await _wait_until(
                lambda: server._active_collection_batch is not None  # noqa: SLF001
                and len(server._active_collection_batch) == 3,  # noqa: SLF001
            )
            await asyncio.sleep(0.01)
            assert policy.batches == []
            assert all(not result.done() for result in waiting_results)

            waiting_results.append(asyncio.create_task(server._infer_action({"id": 3})))  # noqa: SLF001
            results = await asyncio.gather(*waiting_results)
        finally:
            await _stop_worker(worker)

        assert policy.batches == [[0, 1, 2, 3]]
        assert [int(output["actions"][0]) for output, _ in results] == [0, 1, 2, 3]
        assert [timing["batch_size"] for _, timing in results] == [4, 4, 4, 4]

    asyncio.run(run())


def test_bucket_aware_batching_flushes_partial_batch_after_wait() -> None:
    async def run() -> None:
        policy = RecordingPolicy()
        server = websocket_policy_server.WebsocketPolicyServer(
            policy,
            max_batch_size=4,
            min_batch_size=2,
            max_batch_wait_ms=20,
            pad_to_batch_bucket=True,
            bucket_aware_batching=True,
        )
        server._request_queue = asyncio.Queue()  # noqa: SLF001
        worker = asyncio.create_task(server._batch_worker())  # noqa: SLF001

        try:
            waiting_results = [asyncio.create_task(server._infer_action({"id": i})) for i in range(3)]  # noqa: SLF001
            await _wait_until(
                lambda: server._active_collection_batch is not None  # noqa: SLF001
                and len(server._active_collection_batch) == 3,  # noqa: SLF001
            )
            results = await asyncio.gather(*waiting_results)
        finally:
            await _stop_worker(worker)

        assert policy.batches == [[0, 1, 2, 2]]
        assert [int(output["actions"][0]) for output, _ in results] == [0, 1, 2]
        assert [timing["batch_size"] for _, timing in results] == [3, 3, 3]
        assert [timing["padded_batch_size"] for _, timing in results] == [4, 4, 4]

    asyncio.run(run())


def test_batch_collector_prepares_next_batch_while_model_is_busy() -> None:
    async def run() -> None:
        policy = BlockingPolicy()
        server = websocket_policy_server.WebsocketPolicyServer(
            policy,
            max_batch_size=2,
            min_batch_size=2,
            max_batch_wait_ms=100,
        )
        server._request_queue = asyncio.Queue()  # noqa: SLF001
        worker = asyncio.create_task(server._batch_worker())  # noqa: SLF001

        try:
            first_results = [asyncio.create_task(server._infer_action({"id": i})) for i in range(2)]  # noqa: SLF001
            assert await asyncio.to_thread(policy.first_batch_started.wait, 1.0)

            second_results = [asyncio.create_task(server._infer_action({"id": i})) for i in range(2, 4)]  # noqa: SLF001
            await _wait_until(
                lambda: server._ready_batch_queue is not None and server._ready_batch_queue.qsize() >= 1,  # noqa: SLF001
            )

            assert all(not result.done() for result in first_results)
            assert policy.batches == [[0, 1]]

            policy.release_first_batch.set()
            results = await asyncio.gather(*first_results, *second_results)
        finally:
            policy.release_first_batch.set()
            await _stop_worker(worker)

        assert policy.batches == [[0, 1], [2, 3]]
        assert [int(output["actions"][0]) for output, _ in results] == [0, 1, 2, 3]

    asyncio.run(run())


def test_batch_collector_keeps_partial_batch_open_while_model_is_busy() -> None:
    async def run() -> None:
        policy = BlockingPolicy()
        server = websocket_policy_server.WebsocketPolicyServer(
            policy,
            max_batch_size=4,
            min_batch_size=4,
            max_batch_wait_ms=1,
        )
        server._request_queue = asyncio.Queue()  # noqa: SLF001
        worker = asyncio.create_task(server._batch_worker())  # noqa: SLF001

        try:
            first_results = [asyncio.create_task(server._infer_action({"id": i})) for i in range(4)]  # noqa: SLF001
            assert await asyncio.to_thread(policy.first_batch_started.wait, 1.0)

            second_results = [asyncio.create_task(server._infer_action({"id": i})) for i in range(4, 6)]  # noqa: SLF001
            await asyncio.sleep(0.01)
            assert server._ready_batch_queue is not None  # noqa: SLF001
            assert server._ready_batch_queue.qsize() == 0  # noqa: SLF001

            second_results.extend(
                asyncio.create_task(server._infer_action({"id": i}))  # noqa: SLF001
                for i in range(6, 8)
            )
            await _wait_until(lambda: server._ready_batch_queue is not None and server._ready_batch_queue.qsize() == 1)  # noqa: SLF001

            policy.release_first_batch.set()
            results = await asyncio.gather(*first_results, *second_results)
        finally:
            policy.release_first_batch.set()
            await _stop_worker(worker)

        assert policy.batches == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert [int(output["actions"][0]) for output, _ in results] == list(range(8))

    asyncio.run(run())


def test_batch_collector_waits_for_min_batch_after_model_becomes_ready() -> None:
    async def run() -> None:
        policy = BlockingPolicy()
        server = websocket_policy_server.WebsocketPolicyServer(
            policy,
            max_batch_size=4,
            min_batch_size=4,
            max_batch_wait_ms=50,
        )
        server._request_queue = asyncio.Queue()  # noqa: SLF001
        worker = asyncio.create_task(server._batch_worker())  # noqa: SLF001

        try:
            first_results = [asyncio.create_task(server._infer_action({"id": i})) for i in range(4)]  # noqa: SLF001
            assert await asyncio.to_thread(policy.first_batch_started.wait, 1.0)

            second_results = [asyncio.create_task(server._infer_action({"id": i})) for i in range(4, 6)]  # noqa: SLF001
            await asyncio.sleep(0.01)
            policy.release_first_batch.set()
            await asyncio.gather(*first_results)
            await asyncio.sleep(0.01)

            assert all(not result.done() for result in second_results)
            assert server._ready_batch_queue is not None  # noqa: SLF001
            assert server._ready_batch_queue.qsize() == 0  # noqa: SLF001

            second_results.extend(
                asyncio.create_task(server._infer_action({"id": i}))  # noqa: SLF001
                for i in range(6, 8)
            )
            results = await asyncio.gather(*second_results)
        finally:
            policy.release_first_batch.set()
            await _stop_worker(worker)

        assert policy.batches == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert [int(output["actions"][0]) for output, _ in results] == [4, 5, 6, 7]

    asyncio.run(run())


def test_batch_worker_fails_waiting_requests_on_shutdown() -> None:
    async def run() -> None:
        policy = RecordingPolicy()
        server = websocket_policy_server.WebsocketPolicyServer(
            policy,
            max_batch_size=4,
            min_batch_size=4,
            max_batch_wait_ms=1000,
        )
        server._request_queue = asyncio.Queue()  # noqa: SLF001
        worker = asyncio.create_task(server._batch_worker())  # noqa: SLF001
        request_task = asyncio.create_task(server._infer_action({"id": 99}))  # noqa: SLF001

        await _wait_until(
            lambda: server._active_collection_batch is not None  # noqa: SLF001
            and len(server._active_collection_batch) == 1,  # noqa: SLF001
        )

        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker

        with pytest.raises(RuntimeError, match="microbatch worker stopped"):
            await request_task
        with pytest.raises(RuntimeError, match="microbatch worker stopped"):
            await server._infer_action({"id": 100})  # noqa: SLF001

    asyncio.run(run())
