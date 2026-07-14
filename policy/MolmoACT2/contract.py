"""Public MolmoAct2 Bimanual YAM checkpoint contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from XPolicyLab.utils.robodojo_paths import model_weight_root

PROFILE_NAME = "molmoact2_bimanual_yam"
SOURCE_REPOSITORY = "https://github.com/allenai/molmoact2.git"
SOURCE_REVISION = "c2282820f9b188b60e66ea1636b3efd81c45cbb4"
HF_REPO_ID = "allenai/MolmoAct2-BimanualYAM"
HF_REVISION = "8dcbed66f2380e4393189c303ea72488eb9e63c2"
NORM_TAG = "yam_dual_molmoact2"
STATE_DIM = 14
PREDICTED_HORIZON = 30
EXECUTED_HORIZON = 25
FLOW_STEPS = 10
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CAMERA_SHAPE = (3, 360, 640)
GRIPPER_INDICES = (6, 13)
YAM_JOINT_5_INDICES = (4, 11)


def checkpoint_path() -> str:
    return str(model_weight_root("MolmoACT2", PROFILE_NAME, HF_REVISION))


def apply_checkpoint_profile(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Expand the public alias while leaving every other checkpoint untouched."""

    cfg = dict(model_cfg)
    if cfg.get("ckpt_name") != PROFILE_NAME:
        return cfg

    cfg.update(
        {
            "checkpoint_backend": "original_hf",
            "pretrained_path": checkpoint_path(),
            "hf_repo_id": HF_REPO_ID,
            "hf_revision": HF_REVISION,
            "norm_tag": NORM_TAG,
            "inference_action_mode": "continuous",
            "dtype": "float32",
            "num_steps": FLOW_STEPS,
            "enable_depth_reasoning": False,
            "enable_inference_cuda_graph": True,
            "warmup_runs": 3,
            "predicted_horizon": PREDICTED_HORIZON,
            "actions_per_chunk": EXECUTED_HORIZON,
        }
    )
    return cfg


def uses_public_yam_joint_sign_bridge(model_cfg: dict[str, Any]) -> bool:
    """Return whether the pinned original-HF YAM boundary needs sign bridging."""

    return (
        model_cfg.get("ckpt_name") == PROFILE_NAME
        and model_cfg.get("checkpoint_backend") == "original_hf"
    )


def _negate_yam_joint_5(values: Any, *, value_name: str) -> np.ndarray:
    """Return a copy with each arm's ``dof_joint5`` sign negated."""

    result = np.array(values, copy=True)
    if result.ndim < 1 or result.shape[-1] != STATE_DIM:
        raise ValueError(f"{value_name} must end in dimension {STATE_DIM}, got shape {result.shape}.")
    result[..., list(YAM_JOINT_5_INDICES)] = -result[..., list(YAM_JOINT_5_INDICES)]
    return result


def simulator_state_to_checkpoint(state: Any) -> np.ndarray:
    """Map RoboDojo YAM state into the public checkpoint's joint convention."""

    return _negate_yam_joint_5(state, value_name="YAM simulator state")


def checkpoint_actions_to_simulator(actions: Any) -> np.ndarray:
    """Map public-checkpoint YAM actions into RoboDojo's joint convention."""

    return _negate_yam_joint_5(actions, value_name="YAM checkpoint actions")


def validate_robot_contract(robot_action_dim_info: dict[str, Any]) -> None:
    arm_dims = list(robot_action_dim_info.get("arm_dim", ()))
    gripper_dims = list(robot_action_dim_info.get("ee_dim", ()))
    if arm_dims != [6, 6] or gripper_dims != [1, 1]:
        raise ValueError(
            f"MolmoAct2 Bimanual YAM requires [6,1,6,1] dimensions; got arm_dim={arm_dims}, ee_dim={gripper_dims}."
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
        raise ValueError(
            f"MolmoAct2 camera order must be cam_high, cam_left_wrist, cam_right_wrist; got {tuple(images)}."
        )
    for key in CAMERA_KEYS:
        image = np.asarray(images[key])
        if image.shape != CAMERA_SHAPE or image.dtype != np.uint8:
            raise ValueError(
                f"{key} must be uint8 CHW with shape {CAMERA_SHAPE}, got dtype={image.dtype}, shape={image.shape}."
            )


def validate_and_select_actions(actions: Any) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32)
    expected = (PREDICTED_HORIZON, STATE_DIM)
    if result.shape != expected:
        raise ValueError(f"MolmoAct2 must return actions with shape {expected}, got {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError("MolmoAct2 returned non-finite actions.")

    result = result[:EXECUTED_HORIZON].copy()
    grippers = result[:, list(GRIPPER_INDICES)]
    if np.any(grippers < -0.05) or np.any(grippers > 1.05):
        raise ValueError("MolmoAct2 returned grippers outside the normalized YAM contract.")
    result[:, list(GRIPPER_INDICES)] = np.clip(grippers, 0.0, 1.0)
    return result
