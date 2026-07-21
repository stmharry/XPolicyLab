from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def test_real_piper_norm_transforms_use_arm_deltas_and_absolute_grippers():
    config = _config.get_config("pi05_base_aloha_full_real_piper_seed_0")
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
