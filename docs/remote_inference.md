# Remote inference

Use `scripts/serve_policy.py` to host a policy on a machine with the checkpoint
and simulator-compatible dependencies installed. Simulator clients connect over
WebSocket and send observations to the server.

## Start a policy server

Serve a checkpoint by passing the training config and checkpoint directory:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=checkpoints/openpi-libero-9000
```

Additional focused-release configs:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/openpi-metaworld-25000

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_robocasa \
    --policy.dir=checkpoints/pi05_pretrain_human300/multitask_learning/75000
```

For `pi05`, PyTorch serving is optional. The first PyTorch run converts the JAX
checkpoint in place:

```bash
uv run scripts/serve_policy.py --pytorch policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=checkpoints/openpi-libero-9000
```

The server listens on port `8000` by default. Override it with `--port`.

## Query a policy server

Install the lightweight client in the environment that runs the simulator or
robot code:

```bash
cd packages/openpi-client
pip install -e .
```

Minimal client usage:

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
action_chunk = client.infer(observation)["actions"]
```

The observation schema depends on the policy config. See the current simulator
clients for concrete examples:

- `examples/metaworld/main.py`
- `examples/libero_env/main.py`
- `examples/robocasa_env/main.py`
