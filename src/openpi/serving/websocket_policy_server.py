import asyncio
import contextlib
import dataclasses
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _PendingRequest:
    obs: dict
    future: asyncio.Future
    enqueued_time: float


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        *,
        max_batch_size: int = 1,
        max_batch_wait_ms: float = 0.0,
        min_batch_size: int = 1,
        pad_to_batch_bucket: bool = False,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._max_batch_size = max(1, int(max_batch_size))
        self._min_batch_size = min(self._max_batch_size, max(1, int(min_batch_size)))
        self._max_batch_wait_s = max(0.0, float(max_batch_wait_ms) / 1000)
        self._pad_to_batch_bucket = bool(pad_to_batch_bucket)
        self._metadata = dict(metadata or {})
        self._metadata["microbatch"] = {
            "max_batch_size": self._max_batch_size,
            "min_batch_size": self._min_batch_size,
            "max_batch_wait_ms": self._max_batch_wait_s * 1000,
            "pad_to_batch_bucket": self._pad_to_batch_bucket,
        }
        self._request_queue: asyncio.Queue[_PendingRequest] | None = None
        self._ready_batch_queue: asyncio.Queue[list[_PendingRequest]] | None = None
        self._model_ready_event: asyncio.Event | None = None
        self._batching_stopped = False
        self._active_collection_batch: list[_PendingRequest] | None = None
        self._active_inference_batch: list[_PendingRequest] | None = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        batch_worker = None
        if self._max_batch_size > 1:
            self._request_queue = asyncio.Queue()
            batch_worker = asyncio.create_task(self._batch_worker())
            logger.info(
                "Enabled inference microbatching: max_batch_size=%d, min_batch_size=%d, "
                "max_batch_wait_ms=%.3f, pad_to_batch_bucket=%s",
                self._max_batch_size,
                self._min_batch_size,
                self._max_batch_wait_s * 1000,
                self._pad_to_batch_bucket,
            )

        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            ping_timeout=600,
        ) as server:
            try:
                await server.serve_forever()
            finally:
                if batch_worker is not None:
                    batch_worker.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await batch_worker

    async def _infer_action(self, obs: dict) -> tuple[dict, dict]:
        if self._request_queue is None:
            start_time = time.monotonic()
            action = self._policy.infer(obs)
            return action, {"infer_ms": (time.monotonic() - start_time) * 1000}
        if self._batching_stopped:
            raise self._batch_worker_stopped_error()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request = _PendingRequest(obs=obs, future=future, enqueued_time=time.monotonic())
        await self._request_queue.put(request)
        if self._batching_stopped and not future.done():
            self._fail_request(request, self._batch_worker_stopped_error())
        return await future

    async def _batch_worker(self) -> None:
        assert self._request_queue is not None
        self._ready_batch_queue = asyncio.Queue()
        self._model_ready_event = asyncio.Event()
        self._batching_stopped = False

        collector = asyncio.create_task(self._batch_collector_worker())
        inference_worker = asyncio.create_task(self._batch_inference_worker())
        try:
            await asyncio.gather(collector, inference_worker)
        finally:
            self._batching_stopped = True
            for task in (collector, inference_worker):
                task.cancel()
            for task in (collector, inference_worker):
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            self._fail_pending_requests(self._batch_worker_stopped_error())

    @staticmethod
    def _batch_worker_stopped_error() -> RuntimeError:
        return RuntimeError("Inference microbatch worker stopped before completing request.")

    @staticmethod
    def _fail_request(request: _PendingRequest, exc: BaseException) -> None:
        if not request.future.done():
            request.future.set_exception(exc)

    def _fail_requests(self, requests: list[_PendingRequest], exc: BaseException) -> None:
        for request in requests:
            self._fail_request(request, exc)

    def _fail_pending_requests(self, exc: BaseException) -> None:
        if self._active_collection_batch is not None:
            self._fail_requests(self._active_collection_batch, exc)
        if self._active_inference_batch is not None:
            self._fail_requests(self._active_inference_batch, exc)

        if self._request_queue is not None:
            while True:
                try:
                    self._fail_request(self._request_queue.get_nowait(), exc)
                except asyncio.QueueEmpty:
                    break

        if self._ready_batch_queue is not None:
            while True:
                try:
                    self._fail_requests(self._ready_batch_queue.get_nowait(), exc)
                except asyncio.QueueEmpty:
                    break

    async def _batch_collector_worker(self) -> None:
        assert self._ready_batch_queue is not None
        assert self._model_ready_event is not None

        while True:
            batch = []
            try:
                batch = await self._collect_batch()
                active_batch = [request for request in batch if not request.future.cancelled()]
                if active_batch:
                    if self._model_ready_event.is_set():
                        self._model_ready_event.clear()
                    await self._ready_batch_queue.put(active_batch)
            except asyncio.CancelledError:
                self._fail_requests(batch, self._batch_worker_stopped_error())
                raise

    async def _batch_inference_worker(self) -> None:
        assert self._ready_batch_queue is not None
        assert self._model_ready_event is not None

        while True:
            if self._ready_batch_queue.empty():
                self._model_ready_event.set()
            batch = await self._ready_batch_queue.get()
            self._model_ready_event.clear()
            active_batch = [request for request in batch if not request.future.cancelled()]
            if not active_batch:
                continue

            self._active_inference_batch = active_batch
            try:
                start_time = time.monotonic()
                outputs, infer_ms, padded_batch_size = await asyncio.to_thread(
                    self._run_policy_batch,
                    active_batch,
                )
            except asyncio.CancelledError:
                self._fail_requests(active_batch, self._batch_worker_stopped_error())
                raise
            except Exception as exc:
                for request in active_batch:
                    if not request.future.cancelled():
                        request.future.set_exception(exc)
                continue
            finally:
                if self._active_inference_batch is active_batch:
                    self._active_inference_batch = None

            if len(outputs) != len(active_batch):
                exc = RuntimeError(f"Policy returned {len(outputs)} outputs for {len(active_batch)} batched requests.")
                for request in active_batch:
                    if not request.future.cancelled():
                        request.future.set_exception(exc)
                continue

            for request, output in zip(active_batch, outputs, strict=True):
                if request.future.cancelled():
                    continue
                timing = {
                    "infer_ms": infer_ms,
                    "batch_size": len(active_batch),
                    "padded_batch_size": padded_batch_size,
                    "queue_wait_ms": (start_time - request.enqueued_time) * 1000,
                }
                request.future.set_result((output, timing))

    async def _collect_batch(self) -> list[_PendingRequest]:
        assert self._request_queue is not None
        assert self._model_ready_event is not None

        batch: list[_PendingRequest] = []
        self._active_collection_batch = batch
        try:
            first = await self._request_queue.get()
            batch.append(first)
            deadline = time.monotonic() + self._max_batch_wait_s

            while len(batch) < self._max_batch_size:
                while len(batch) < self._max_batch_size:
                    try:
                        batch.append(self._request_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                if len(batch) >= self._max_batch_size:
                    break

                if self._model_ready_event.is_set():
                    if len(batch) >= self._min_batch_size:
                        break

                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self._request_queue.get(), timeout=timeout))
                    except TimeoutError:
                        break
                    continue

                get_request = asyncio.create_task(self._request_queue.get())
                wait_for_model = asyncio.create_task(self._model_ready_event.wait())
                done, pending = await asyncio.wait(
                    {get_request, wait_for_model},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in pending:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                if get_request in done:
                    batch.append(get_request.result())
                if wait_for_model in done:
                    await self._fill_waiting_batch_after_model_ready(batch)
                    break

            return batch
        except asyncio.CancelledError:
            self._fail_requests(batch, self._batch_worker_stopped_error())
            raise
        finally:
            if self._active_collection_batch is batch:
                self._active_collection_batch = None

    async def _fill_waiting_batch_after_model_ready(self, batch: list[_PendingRequest]) -> None:
        assert self._request_queue is not None

        deadline = time.monotonic() + self._max_batch_wait_s
        while len(batch) < self._max_batch_size:
            while len(batch) < self._max_batch_size:
                try:
                    batch.append(self._request_queue.get_nowait())
                    deadline = time.monotonic() + self._max_batch_wait_s
                except asyncio.QueueEmpty:
                    break

            if len(batch) >= self._max_batch_size:
                break

            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._request_queue.get(), timeout=timeout))
                deadline = time.monotonic() + self._max_batch_wait_s
            except TimeoutError:
                break

    def _run_policy_batch(self, batch: list[_PendingRequest]) -> tuple[list[dict], float, int]:
        start_time = time.monotonic()
        padded_batch_size = len(batch)
        if len(batch) == 1:
            outputs = [self._policy.infer(batch[0].obs)]
        else:
            batch_obs = [request.obs for request in batch]
            padded_batch_size = self._padded_batch_size(len(batch_obs))
            if padded_batch_size > len(batch_obs):
                batch_obs.extend([batch_obs[-1]] * (padded_batch_size - len(batch_obs)))
            outputs = self._policy.infer_many(batch_obs)[: len(batch)]
        infer_ms = (time.monotonic() - start_time) * 1000
        return outputs, infer_ms, padded_batch_size

    def _padded_batch_size(self, batch_size: int) -> int:
        if not self._pad_to_batch_bucket:
            return batch_size
        if batch_size <= 1:
            return batch_size
        return min(self._max_batch_size, 1 << (batch_size - 1).bit_length())

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                action, server_timing = await self._infer_action(obs)
                action = dict(action)
                action["server_timing"] = server_timing
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
