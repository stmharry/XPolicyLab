"""Public PI0.5 ARX X5 multitask checkpoint contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from XPolicyLab.utils.robodojo_paths import model_weight_root

PROFILE_NAME = "pi05_arx5_multitask_v1"
HF_REPO_ID = "pravsels/pi05-arx5-multitask-v1"
HF_REVISION = "880fa61406540d80b1c3b9824f12c19b903a233f"
CHECKPOINT_STEP = 55000
TRAIN_CONFIG_NAME = "pi05_arx5_multitask_v1"
ACTION_DIM = 14
ACTION_HORIZON = 50
GRIPPER_INDICES = (6, 13)
GRIPPER_OFFSET = -0.01
GRIPPER_SPAN = 0.054


def snapshot_path() -> Path:
    return model_weight_root("Pi_05", PROFILE_NAME, HF_REVISION)


def checkpoint_path() -> Path:
    return snapshot_path() / "checkpoints" / str(CHECKPOINT_STEP)


def apply_checkpoint_profile(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Expand the public alias while retaining local/path checkpoint behavior."""

    cfg = dict(model_cfg)
    if cfg.get("ckpt_name") != PROFILE_NAME:
        return cfg

    cfg.update(
        {
            "checkpoint_profile": PROFILE_NAME,
            "model_path": str(checkpoint_path()),
            "norm_stats_path": str(snapshot_path() / "assets"),
            "hf_repo_id": HF_REPO_ID,
            "hf_revision": HF_REVISION,
            "checkpoint_num": CHECKPOINT_STEP,
            "train_config_name": TRAIN_CONFIG_NAME,
            "repo_id": PROFILE_NAME,
            "gripper_contract": "arx5_physical",
            "actions_per_chunk": ACTION_HORIZON,
        }
    )
    return cfg


def validate_robot_contract(robot_action_dim_info: dict[str, Any]) -> None:
    arm_dims = list(robot_action_dim_info.get("arm_dim", ()))
    gripper_dims = list(robot_action_dim_info.get("ee_dim", ()))
    if arm_dims != [6, 6] or gripper_dims != [1, 1]:
        raise ValueError(f"PI0.5 ARX X5 requires [6,1,6,1] dimensions; got arm_dim={arm_dims}, ee_dim={gripper_dims}.")


def validate_profile_checkpoint(path: Path) -> None:
    expected = checkpoint_path().resolve()
    actual = path.resolve()
    if actual != expected:
        raise ValueError(f"{PROFILE_NAME} requires checkpoint step {CHECKPOINT_STEP}: {expected}; got {actual}.")
    if not (actual / "params").is_dir():
        raise FileNotFoundError(f"Missing Orbax params directory: {actual / 'params'}")


def robodojo_to_checkpoint(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    if result.shape[-1] != ACTION_DIM:
        raise ValueError(f"ARX state/action last dimension must be {ACTION_DIM}, got {result.shape[-1]}.")
    if not np.isfinite(result).all():
        raise ValueError("ARX state/action contains non-finite values.")
    grippers = result[..., list(GRIPPER_INDICES)]
    if np.any(grippers < 0.0) or np.any(grippers > 1.0):
        raise ValueError(f"RoboDojo ARX grippers must be normalized to [0,1], got {grippers}.")
    result[..., list(GRIPPER_INDICES)] = GRIPPER_OFFSET + GRIPPER_SPAN * grippers
    return result


def checkpoint_to_robodojo(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    expected = (ACTION_HORIZON, ACTION_DIM)
    if result.shape != expected:
        raise ValueError(f"PI0.5 must return actions with shape {expected}, got {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError("PI0.5 returned non-finite actions.")
    physical = result[:, list(GRIPPER_INDICES)]
    result[:, list(GRIPPER_INDICES)] = np.clip(
        (physical - GRIPPER_OFFSET) / GRIPPER_SPAN,
        0.0,
        1.0,
    )
    return result
