import jax
import numpy as np
from openpi.policies import aloha_policy, policy as _policy, policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import action_chunk_broker
import pytest


def test_reset_restores_initial_rng():
    policy = object.__new__(_policy.Policy)
    policy._is_pytorch_model = False
    policy._initial_rng = jax.random.key(17)
    policy._rng, _ = jax.random.split(policy._initial_rng)
    assert not np.array_equal(jax.random.key_data(policy._rng), jax.random.key_data(policy._initial_rng))

    policy.reset()

    np.testing.assert_array_equal(jax.random.key_data(policy._rng), jax.random.key_data(policy._initial_rng))


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
