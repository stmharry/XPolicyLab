"""Policy-facing state/action contract for two non-X AgileX PiPER arms."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

ENVIRONMENT_NAME = "bimanual_piper"
STATE_DIM = 14
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CAMERA_SHAPE = (3, 224, 224)
GRIPPER_INDICES = (6, 13)
ARM_SLICES = (slice(0, 6), slice(7, 13))
JOINT_LIMITS = np.asarray(
    [
        [-2.618, 2.618],
        [0.0, 3.14],
        [-2.967, 0.0],
        [-1.745, 1.745],
        [-1.22, 1.22],
        [-2.0944, 2.0944],
    ],
    dtype=np.float32,
)
SIMULATOR_FINGER_LIMIT_M = (0.0, 0.035)
CHECKPOINT_JAW_LIMIT_M = (0.0, 0.070)

logger = logging.getLogger(__name__)


def validate_environment(env_cfg_type: str | None) -> None:
    if env_cfg_type != ENVIRONMENT_NAME:
        raise ValueError(f"bimanual PiPER checkpoints require env_cfg_type={ENVIRONMENT_NAME!r}.")


def validate_robot_contract(robot_action_dim_info: dict[str, Any]) -> None:
    arm_dims = list(robot_action_dim_info.get("arm_dim", ()))
    gripper_dims = list(robot_action_dim_info.get("ee_dim", ()))
    if arm_dims != [6, 6] or gripper_dims != [1, 1]:
        raise ValueError(
            f"Bimanual PiPER requires [6,1,6,1] dimensions; got arm_dim={arm_dims}, ee_dim={gripper_dims}."
        )


def _validate_arm_limits(values: np.ndarray, *, field: str) -> None:
    for arm_index, arm_slice in enumerate(ARM_SLICES):
        arm = values[arm_slice]
        outside = np.flatnonzero((arm < JOINT_LIMITS[:, 0]) | (arm > JOINT_LIMITS[:, 1]))
        if outside.size:
            joint_index = int(outside[0])
            value = float(arm[joint_index])
            lower, upper = JOINT_LIMITS[joint_index]
            raise ValueError(
                f"PiPER {field} arm {arm_index} joint{joint_index + 1}={value} is outside "
                f"[{float(lower)}, {float(upper)}]."
            )


def simulator_state_to_checkpoint(state: Any) -> np.ndarray:
    """Map half-jaw simulator state to the checkpoint's total jaw opening."""
    result = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    if result.shape != (STATE_DIM,):
        raise ValueError(f"PiPER state must have shape ({STATE_DIM},), got {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError("PiPER state contains non-finite values.")
    _validate_arm_limits(result, field="state")
    fingers = result[list(GRIPPER_INDICES)]
    if np.any(fingers < SIMULATOR_FINGER_LIMIT_M[0]) or np.any(fingers > SIMULATOR_FINGER_LIMIT_M[1]):
        raise ValueError(f"PiPER simulator fingers must be in [0, 0.035] m; got {fingers.tolist()}.")
    result[list(GRIPPER_INDICES)] = 2.0 * fingers
    return result


def _clip_with_log(value: float, lower: float, upper: float, *, channel: str, action_index: int) -> float:
    clipped = float(np.clip(value, lower, upper))
    if clipped != value:
        logger.warning(
            "Clipped PiPER checkpoint action[%d] %s from %.9f to %.9f within [%.9f, %.9f].",
            action_index,
            channel,
            value,
            clipped,
            lower,
            upper,
        )
    return clipped


def checkpoint_actions_to_simulator(actions: Any, *, executed_horizon: int = 8) -> np.ndarray:
    """Clamp/log absolute targets, convert jaw openings, and take a fixed prefix."""
    result = np.asarray(actions, dtype=np.float32).copy()
    expected = (50, STATE_DIM)
    if result.shape != expected:
        raise ValueError(f"PiPER policy must return actions with shape {expected}, got {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError("PiPER policy returned non-finite actions.")
    if executed_horizon != 8:
        raise ValueError(f"PiPER fixed-prefix execution requires 8 actions; got {executed_horizon}.")

    for action_index in range(result.shape[0]):
        for arm_index, arm_slice in enumerate(ARM_SLICES):
            for joint_index in range(6):
                lower, upper = JOINT_LIMITS[joint_index]
                source_index = arm_slice.start + joint_index
                result[action_index, source_index] = _clip_with_log(
                    float(result[action_index, source_index]),
                    float(lower),
                    float(upper),
                    channel=f"arm{arm_index}/joint{joint_index + 1}",
                    action_index=action_index,
                )
        for gripper_index in GRIPPER_INDICES:
            total_opening = _clip_with_log(
                float(result[action_index, gripper_index]),
                CHECKPOINT_JAW_LIMIT_M[0],
                CHECKPOINT_JAW_LIMIT_M[1],
                channel=f"jaw{gripper_index}",
                action_index=action_index,
            )
            result[action_index, gripper_index] = total_opening / 2.0
    return result[:executed_horizon].copy()


def validate_camera_payload(images: dict[str, Any]) -> None:
    if tuple(images) != CAMERA_KEYS:
        raise ValueError(f"PiPER camera order must be {', '.join(CAMERA_KEYS)}; got {tuple(images)}.")
    for key in CAMERA_KEYS:
        image = np.asarray(images[key])
        if image.shape != CAMERA_SHAPE or image.dtype != np.uint8:
            raise ValueError(
                f"{key} must be uint8 CHW with shape {CAMERA_SHAPE}; got dtype={image.dtype}, shape={image.shape}."
            )
