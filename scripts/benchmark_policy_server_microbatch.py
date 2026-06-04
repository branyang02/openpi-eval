"""Benchmark OpenPI WebSocket serving with many LIBERO-shaped clients.

This measures policy-server request handling without launching the LIBERO
simulator. It is intended to compare serving strategies such as batch-1
inference versus server-side microbatching under the same client load.
"""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import json
import statistics
import threading
import time
from typing import Literal

from openpi_client import websocket_client_policy
import tyro

from openpi.policies import libero_policy
from openpi.policies import robocasa_policy


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    payload: Literal["libero", "robocasa"] = "libero"
    num_clients: int = 64
    requests_per_client: int = 10
    warmup_requests_per_client: int = 1
    output_json: str | None = None


def _make_payload(payload: str) -> dict:
    match payload:
        case "libero":
            return libero_policy.make_libero_example()
        case "robocasa":
            return robocasa_policy.make_robocasa_example()
        case _:
            raise ValueError(f"Unknown payload: {payload}")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = round((len(values) - 1) * percentile / 100.0)
    return sorted(values)[index]


def _summarize_timings(results: list[dict], *, wall_s: float, args: Args) -> dict:
    latencies = [result["latency_ms"] for result in results]
    server_timings = [result["server_timing"] for result in results]
    infer_ms = [float(timing.get("infer_ms", 0.0)) for timing in server_timings]
    queue_wait_ms = [float(timing.get("queue_wait_ms", 0.0)) for timing in server_timings]
    batch_sizes = [int(timing.get("batch_size", 1)) for timing in server_timings]
    padded_batch_sizes = [
        int(timing.get("padded_batch_size", timing.get("batch_size", 1))) for timing in server_timings
    ]

    total_requests = len(results)
    return {
        "payload": args.payload,
        "num_clients": args.num_clients,
        "requests_per_client": args.requests_per_client,
        "total_requests": total_requests,
        "wall_s": wall_s,
        "throughput_rps": total_requests / wall_s if wall_s > 0 else 0.0,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": max(latencies, default=0.0),
        },
        "server_infer_ms": {
            "mean": statistics.fmean(infer_ms) if infer_ms else 0.0,
            "p50": _percentile(infer_ms, 50),
            "p95": _percentile(infer_ms, 95),
            "max": max(infer_ms, default=0.0),
        },
        "server_queue_wait_ms": {
            "mean": statistics.fmean(queue_wait_ms) if queue_wait_ms else 0.0,
            "p50": _percentile(queue_wait_ms, 50),
            "p95": _percentile(queue_wait_ms, 95),
            "max": max(queue_wait_ms, default=0.0),
        },
        "server_batch_size_counts": dict(sorted(collections.Counter(batch_sizes).items())),
        "server_padded_batch_size_counts": dict(sorted(collections.Counter(padded_batch_sizes).items())),
    }


def _run_clients(
    clients: list[websocket_client_policy.WebsocketClientPolicy],
    payloads: list[dict],
    *,
    requests_per_client: int,
    record: bool,
) -> tuple[list[dict], float]:
    start_barrier = threading.Barrier(len(clients) + 1)

    def run_one(client_index: int) -> list[dict]:
        client = clients[client_index]
        payload = payloads[client_index]
        local_results = []
        start_barrier.wait()
        for request_index in range(requests_per_client):
            start = time.perf_counter()
            response = client.infer(payload)
            latency_ms = (time.perf_counter() - start) * 1000
            if record:
                local_results.append(
                    {
                        "client_index": client_index,
                        "request_index": request_index,
                        "latency_ms": latency_ms,
                        "server_timing": response.get("server_timing", {}),
                    }
                )
        return local_results

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients)) as pool:
        futures = [pool.submit(run_one, client_index) for client_index in range(len(clients))]
        start = time.perf_counter()
        start_barrier.wait()
        nested_results = [future.result() for future in futures]
        wall_s = time.perf_counter() - start

    return [item for result in nested_results for item in result], wall_s


def main(args: Args) -> None:
    if args.num_clients <= 0:
        raise ValueError("--num-clients must be positive")
    if args.requests_per_client <= 0:
        raise ValueError("--requests-per-client must be positive")
    if args.warmup_requests_per_client < 0:
        raise ValueError("--warmup-requests-per-client must be non-negative")

    payloads = [_make_payload(args.payload) for _ in range(args.num_clients)]
    clients = [websocket_client_policy.WebsocketClientPolicy(args.host, args.port) for _ in range(args.num_clients)]

    metadata = clients[0].get_server_metadata() if clients else {}
    if args.warmup_requests_per_client:
        _run_clients(
            clients,
            payloads,
            requests_per_client=args.warmup_requests_per_client,
            record=False,
        )

    results, wall_s = _run_clients(
        clients,
        payloads,
        requests_per_client=args.requests_per_client,
        record=True,
    )
    summary = _summarize_timings(results, wall_s=wall_s, args=args)
    summary["server_metadata"] = metadata

    text = json.dumps(summary, indent=2)
    print(text)
    if args.output_json is not None:
        with open(args.output_json, "w") as file_handle:
            file_handle.write(text)
            file_handle.write("\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
