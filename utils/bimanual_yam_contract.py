"""Canonical policy-facing contract for RoboDojo's bimanual YAM embodiment."""

from __future__ import annotations

from typing import Any

import numpy as np

ENVIRONMENT_NAME = "bimanual_yam"
STATE_DIM = 14
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CAMERA_SHAPE = (3, 360, 640)
CAMERA_SHAPES = (CAMERA_SHAPE, (3, 480, 640))
GRIPPER_INDICES = (6, 13)


def validate_environment(env_cfg_type: str | None) -> None:
    if env_cfg_type != ENVIRONMENT_NAME:
        raise ValueError(f"bimanual YAM checkpoints require env_cfg_type={ENVIRONMENT_NAME!r}.")


def validate_robot_contract(robot_action_dim_info: dict[str, Any]) -> None:
    arm_dims = list(robot_action_dim_info.get("arm_dim", ()))
    gripper_dims = list(robot_action_dim_info.get("ee_dim", ()))
    if arm_dims != [6, 6] or gripper_dims != [1, 1]:
        raise ValueError(
            f"Bimanual YAM requires [6,1,6,1] dimensions; got arm_dim={arm_dims}, ee_dim={gripper_dims}."
        )


def validate_state(state: Any) -> np.ndarray:
    result = np.asarray(state, dtype=np.float32).reshape(-1)
    if result.shape != (STATE_DIM,):
        raise ValueError(f"YAM state must have shape ({STATE_DIM},), got {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError("YAM state contains non-finite values.")
    grippers = result[list(GRIPPER_INDICES)]
    if np.any(grippers < 0.0) or np.any(grippers > 1.0):
        raise ValueError(f"YAM grippers must be normalized to [0,1], got {grippers.tolist()}.")
    return result


def validate_camera_payload(images: dict[str, Any]) -> None:
    if tuple(images) != CAMERA_KEYS:
        raise ValueError(f"YAM camera order must be {', '.join(CAMERA_KEYS)}; got {tuple(images)}.")
    for key in CAMERA_KEYS:
        image = np.asarray(images[key])
        if image.shape not in CAMERA_SHAPES or image.dtype != np.uint8:
            raise ValueError(
                f"{key} must be uint8 CHW with one of {CAMERA_SHAPES}, "
                f"got dtype={image.dtype}, shape={image.shape}."
            )


def validate_action_chunk(
    actions: Any,
    *,
    predicted_horizon: int,
    executed_horizon: int,
) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32)
    expected = (predicted_horizon, STATE_DIM)
    if result.shape != expected:
        raise ValueError(f"YAM policy must return actions with shape {expected}, got {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError("YAM policy returned non-finite actions.")
    if not 1 <= executed_horizon <= predicted_horizon:
        raise ValueError("executed_horizon must be within the predicted action chunk")

    result = result[:executed_horizon].copy()
    grippers = result[:, list(GRIPPER_INDICES)]
    if np.any(grippers < -0.05) or np.any(grippers > 1.05):
        raise ValueError("YAM policy returned grippers outside the normalized contract.")
    result[:, list(GRIPPER_INDICES)] = np.clip(grippers, 0.0, 1.0)
    return result
