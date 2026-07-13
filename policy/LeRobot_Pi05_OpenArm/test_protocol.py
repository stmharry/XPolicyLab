import json
from pathlib import Path

import numpy as np
import pytest

from XPolicyLab.policy.LeRobot_Pi05_OpenArm.deploy import _ActionQueue, _diagnostic_mode
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


@pytest.mark.parametrize("profile", ["openarm_lerobot", "openarm_wowrobo_v1_1", "openarm_anvil_v2"])
def test_xpolicylab_registry_exposes_current_16d_openarm_profiles(profile):
    root = Path(__file__).resolve().parents[2]
    registry = json.loads((root / "utils/robot/_robot_info.json").read_text())
    info = registry[profile]
    assert sum(info["arm_dim"]) + sum(info["ee_dim"]) == 16
    assert "openarm_cloth_folding" not in registry


def test_action_queue_exposes_original_and_processed_leftovers_separately():
    queue = _ActionQueue()
    processed = np.arange(30 * 16, dtype=np.float32).reshape(30, 16)
    original = processed + 10_000
    queue.merge(processed, original, real_delay=2)

    np.testing.assert_array_equal(queue.get(), processed[2])
    np.testing.assert_array_equal(queue.processed_leftover(), processed[3:])
    np.testing.assert_array_equal(queue.original_leftover(), original[3:])
    assert queue.action_index() == 1


def test_official_rtc_is_the_default(monkeypatch):
    monkeypatch.delenv("ROBODOJO_OPENARM_RTC_MODE", raising=False)
    assert _diagnostic_mode() == "official"
