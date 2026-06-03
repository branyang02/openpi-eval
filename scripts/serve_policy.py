import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi05_libero").
    config: str
    # Checkpoint directory (e.g., "checkpoints/openpi-libero-9000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default local pi05 LIBERO checkpoint."""

    config: str = "pi05_libero"
    dir: str = "checkpoints/openpi-libero-9000"


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Use PyTorch backend for inference. Auto-converts the JAX checkpoint if needed.
    pytorch: bool = False

    # Apply torch.compile(sample_actions, mode="max-autotune") at model load.
    torch_compile: bool = False

    # Specifies how to load the policy. If not provided, the local pi05 LIBERO checkpoint will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    policy_args = Checkpoint(args.policy.config, args.policy.dir)
    if args.pytorch:
        from openpi.models_pytorch.convert import ensure_pytorch_checkpoint

        ensure_pytorch_checkpoint(policy_args.dir, policy_args.config)

    match args.policy:
        case Checkpoint() | Default():
            return _policy_config.create_trained_policy(
                _config.get_config(policy_args.config),
                policy_args.dir,
                default_prompt=args.default_prompt,
                torch_compile=args.torch_compile,
                use_pytorch=args.pytorch,
            )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
