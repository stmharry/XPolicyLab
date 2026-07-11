"""LeRobot/OpenARM execution helpers shared by the policy and simulator runtimes."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

POLICY_FPS = 30
INTERPOLATION_MULTIPLIER = 3
CONTROL_FPS = POLICY_FPS * INTERPOLATION_MULTIPLIER
PHYSICS_FPS = 240
ACTION_QUEUE_SIZE = 30
RTC_EXECUTION_HORIZON = 20
RTC_MAX_GUIDANCE_WEIGHT = 5.0
MAX_RELATIVE_TARGET_DEG = 8.0

RIGHT_LIMITS = np.asarray(
    [[-75, 75], [-9, 90], [-85, 85], [0, 135], [-85, 85], [-40, 40], [-80, 80]],
    dtype=np.float32,
)
LEFT_LIMITS = np.asarray(
    [[-75, 75], [-90, 9], [-85, 85], [0, 135], [-85, 85], [-40, 40], [-80, 80]],
    dtype=np.float32,
)
GRIPPER_LIMITS = np.asarray([-65.0, 0.0], dtype=np.float32)
GRIPPER_OPENING_M = 0.044
GRIPPER_DEGREES = 65.0


def gripper_m_to_degrees(value: float | np.ndarray) -> np.ndarray:
    return np.clip(-GRIPPER_DEGREES * np.asarray(value) / GRIPPER_OPENING_M, -GRIPPER_DEGREES, 0.0)


def gripper_degrees_to_m(value: float | np.ndarray) -> np.ndarray:
    return np.clip(-np.asarray(value) * GRIPPER_OPENING_M / GRIPPER_DEGREES, 0.0, GRIPPER_OPENING_M)


def pack_openarm_state(observation: dict) -> np.ndarray:
    state = observation["state"]
    right = np.rad2deg(np.asarray(state["right_arm_joint_state"], dtype=np.float32))
    left = np.rad2deg(np.asarray(state["left_arm_joint_state"], dtype=np.float32))
    right_gripper = gripper_m_to_degrees(np.asarray(state["right_ee_joint_state"])[0:1])
    left_gripper = gripper_m_to_degrees(np.asarray(state["left_ee_joint_state"])[0:1])
    packed = np.concatenate((right, right_gripper, left, left_gripper)).astype(np.float32)
    if packed.shape != (16,) or not np.isfinite(packed).all():
        raise ValueError(f"expected finite OpenARM state shape (16,), got {packed.shape}")
    return packed


def unpack_openarm_action(action_degrees: np.ndarray) -> dict[str, np.ndarray]:
    action = clip_absolute_limits(action_degrees)
    return {
        "left_arm_joint_state": np.deg2rad(action[8:15]).astype(np.float32),
        "left_ee_joint_state": np.asarray([gripper_degrees_to_m(action[15])], dtype=np.float32),
        "right_arm_joint_state": np.deg2rad(action[:7]).astype(np.float32),
        "right_ee_joint_state": np.asarray([gripper_degrees_to_m(action[7])], dtype=np.float32),
    }


def physics_tick_pattern(
    physics_fps: int = PHYSICS_FPS,
    control_fps: int = CONTROL_FPS,
    count: int = INTERPOLATION_MULTIPLIER,
) -> tuple[int, ...]:
    """Return phase-accumulated physics ticks for successive control targets.

    At 240 Hz physics and 90 Hz control this yields ``(3, 3, 2)``. Three
    targets always consume eight physics ticks, preserving the exact 30 Hz
    observation interval.
    """

    if physics_fps <= 0 or control_fps <= 0 or count <= 0:
        raise ValueError("physics_fps, control_fps, and count must be positive")
    boundaries = [math.ceil(i * physics_fps / control_fps) for i in range(count + 1)]
    pattern = tuple(boundaries[i + 1] - boundaries[i] for i in range(count))
    if any(ticks <= 0 for ticks in pattern):
        raise ValueError(f"control rate {control_fps} exceeds supported physics rate {physics_fps}")
    return pattern


def interpolate_action(previous: np.ndarray | None, current: np.ndarray, multiplier: int = 3) -> np.ndarray:
    """Match LeRobot ``ActionInterpolator`` for one policy action."""

    current = np.asarray(current, dtype=np.float32).reshape(-1)
    if current.shape != (16,) or not np.isfinite(current).all():
        raise ValueError(f"expected finite current action shape (16,), got {current.shape}")
    if multiplier < 1:
        raise ValueError("multiplier must be at least one")
    if previous is None:
        return current[None, :]
    previous = np.asarray(previous, dtype=np.float32).reshape(-1)
    if previous.shape != (16,) or not np.isfinite(previous).all():
        raise ValueError(f"expected finite previous action shape (16,), got {previous.shape}")
    return np.stack(
        [previous + (i / multiplier) * (current - previous) for i in range(1, multiplier + 1)]
    ).astype(np.float32)


def clip_absolute_limits(action_degrees: np.ndarray) -> np.ndarray:
    action = np.asarray(action_degrees, dtype=np.float32).reshape(-1).copy()
    if action.shape != (16,) or not np.isfinite(action).all():
        raise ValueError(f"expected finite OpenARM action shape (16,), got {action.shape}")
    action[:7] = np.clip(action[:7], RIGHT_LIMITS[:, 0], RIGHT_LIMITS[:, 1])
    action[7] = np.clip(action[7], *GRIPPER_LIMITS)
    action[8:15] = np.clip(action[8:15], LEFT_LIMITS[:, 0], LEFT_LIMITS[:, 1])
    action[15] = np.clip(action[15], *GRIPPER_LIMITS)
    return action


def clamp_relative_target(
    target_degrees: np.ndarray,
    current_degrees: np.ndarray,
    max_delta_degrees: float = MAX_RELATIVE_TARGET_DEG,
) -> np.ndarray:
    """Apply the OpenARM driver's per-command relative target safety limit."""

    target = clip_absolute_limits(target_degrees)
    current = np.asarray(current_degrees, dtype=np.float32).reshape(-1)
    if current.shape != (16,) or not np.isfinite(current).all():
        raise ValueError(f"expected finite current state shape (16,), got {current.shape}")
    if not math.isfinite(max_delta_degrees) or max_delta_degrees <= 0:
        raise ValueError("max_delta_degrees must be finite and positive")
    safe = current + np.clip(target - current, -max_delta_degrees, max_delta_degrees)
    return clip_absolute_limits(safe)


def finite_action_chunk(actions: Iterable[Iterable[float]]) -> np.ndarray:
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.shape != (ACTION_QUEUE_SIZE, 16) or not np.isfinite(chunk).all():
        raise ValueError(f"expected finite action chunk (30, 16), got {chunk.shape}")
    return chunk
