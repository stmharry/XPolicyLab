import numpy as np
import pytest
import json
from pathlib import Path

from XPolicyLab.policy.LeRobot_Pi05_OpenArm.protocol import (
    clamp_relative_target,
    interpolate_action,
    physics_tick_pattern,
)


def test_240hz_to_90hz_phase_pattern_preserves_30hz_interval():
    pattern = physics_tick_pattern()
    assert pattern == (3, 3, 2)
    assert sum(pattern) == 8


def test_interpolation_matches_lerobot_multiplier_three():
    previous = np.zeros(16, dtype=np.float32)
    current = np.full(16, 3.0, dtype=np.float32)
    np.testing.assert_allclose(interpolate_action(previous, current), [np.ones(16), np.full(16, 2), current])
    np.testing.assert_allclose(interpolate_action(None, current), current[None, :])


def test_relative_target_and_absolute_joint_limits():
    current = np.zeros(16, dtype=np.float32)
    target = np.full(16, 100.0, dtype=np.float32)
    safe = clamp_relative_target(target, current)
    np.testing.assert_allclose(safe[:7], 8.0)
    assert safe[7] == 0.0
    np.testing.assert_allclose(safe[8:15], 8.0)
    assert safe[15] == 0.0
    with pytest.raises(ValueError, match="finite"):
        clamp_relative_target(np.full(16, np.nan), current)


def test_xpolicylab_registry_exposes_16d_openarm_profile():
    root = Path(__file__).resolve().parents[2]
    info = json.loads((root / "utils/robot/_robot_info.json").read_text())["openarm_cloth_folding"]
    assert sum(info["arm_dim"]) + sum(info["ee_dim"]) == 16
