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
        max_batch_size: int = 1,
        max_batch_wait_ms: float = 0.0,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_batch_wait_s = max(0.0, float(max_batch_wait_ms) / 1000)
        self._request_queue: asyncio.Queue[_PendingRequest] | None = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        batch_worker = None
        if self._max_batch_size > 1:
            self._request_queue = asyncio.Queue()
            batch_worker = asyncio.create_task(self._batch_worker())
            logger.info(
                "Enabled inference microbatching: max_batch_size=%d, max_batch_wait_ms=%.3f",
                self._max_batch_size,
                self._max_batch_wait_s * 1000,
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

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._request_queue.put(_PendingRequest(obs=obs, future=future, enqueued_time=time.monotonic()))
        return await future

    async def _batch_worker(self) -> None:
        assert self._request_queue is not None

        while True:
            first = await self._request_queue.get()
            batch = [first]
            deadline = time.monotonic() + self._max_batch_wait_s

            while len(batch) < self._max_batch_size:
                try:
                    batch.append(self._request_queue.get_nowait())
                    continue
                except asyncio.QueueEmpty:
                    pass

                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._request_queue.get(), timeout=timeout))
                except TimeoutError:
                    break

            active_batch = [request for request in batch if not request.future.cancelled()]
            if not active_batch:
                continue

            try:
                start_time = time.monotonic()
                if len(active_batch) == 1:
                    outputs = [self._policy.infer(active_batch[0].obs)]
                else:
                    outputs = self._policy.infer_many([request.obs for request in active_batch])
                infer_ms = (time.monotonic() - start_time) * 1000
            except Exception as exc:
                for request in active_batch:
                    if not request.future.cancelled():
                        request.future.set_exception(exc)
                continue

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
                    "queue_wait_ms": (start_time - request.enqueued_time) * 1000,
                }
                request.future.set_result((output, timing))

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
