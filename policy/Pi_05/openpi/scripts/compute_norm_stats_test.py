from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _assert_arm_delta_absolute_gripper_contract(config_name: str) -> None:
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    assert data_config.norm_stat_transforms is not None

    state = np.arange(14, dtype=np.float32)
    actions = np.stack([state + 1, state + 2])
    original_actions = actions.copy()
    sample = {"observation.state": state, "action": actions}
    for transform in data_config.norm_stat_transforms.inputs:
        sample = transform(sample)

    expected = original_actions.copy()
    arm_mask = np.asarray([True] * 6 + [False] + [True] * 6 + [False])
    expected[..., arm_mask] -= state[arm_mask]
    np.testing.assert_allclose(sample["state"], state)
    np.testing.assert_allclose(sample["actions"], expected)
    np.testing.assert_allclose(sample["actions"][..., [6, 13]], original_actions[..., [6, 13]])


def test_real_piper_norm_transforms_use_arm_deltas_and_absolute_grippers():
    _assert_arm_delta_absolute_gripper_contract("pi05_base_aloha_full_real_piper_seed_0")


def test_real_arx_x5_config_uses_distinct_dataset_and_arm_delta_absolute_gripper_contract():
    config = _config.get_config("pi05_base_aloha_full_real_arx-x5_seed_0")
    assert config.data.repo_id == "RoboDojo-real_arx_x5_6task-bimanual_arx_x5-joint"
    assert config.data.adapt_to_pi is False
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 50
    assert config.batch_size == 256
    assert config.fsdp_devices == 2
    assert config.num_train_steps == 30_000
    assert config.wandb_enabled is False
    assert config.tensorboard_enabled is True
    _assert_arm_delta_absolute_gripper_contract(config.name)


def test_disable_video_loading_only_changes_in_memory_features():
    features = {
        "observation.state": {"dtype": "float32"},
        "action": {"dtype": "float32"},
        "observation.images.cam_high": {"dtype": "video"},
    }
    dataset = SimpleNamespace(meta=SimpleNamespace(info={"features": features.copy()}))

    _data_loader._disable_video_loading(dataset)

    assert list(dataset.meta.info["features"]) == ["observation.state", "action"]
    assert "observation.images.cam_high" in features
