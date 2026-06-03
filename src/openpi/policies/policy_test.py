from openpi_client import action_chunk_broker
import pytest

from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(config, "checkpoints/openpi-libero-9000")

    example = libero_policy.make_libero_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 7)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(config, "checkpoints/openpi-libero-9000")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = libero_policy.make_libero_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (7,)
