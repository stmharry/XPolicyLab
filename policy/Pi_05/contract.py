"""Pinned public PI0.5 checkpoint contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.utils import bimanual_yam_contract as _yam, yam_molmoact2_frame as _yam_frame
from XPolicyLab.utils.robodojo_paths import model_weight_root

# Keep the original names as compatibility aliases for callers that imported the
# first public checkpoint contract directly.
PROFILE_NAME = "pi05_arx5_multitask_v1"
ARX_PROFILE_NAME = PROFILE_NAME
HF_REPO_ID = "pravsels/pi05-arx5-multitask-v1"
HF_REVISION = "880fa61406540d80b1c3b9824f12c19b903a233f"
CHECKPOINT_STEP = 55000
TRAIN_CONFIG_NAME = ARX_PROFILE_NAME

YAM_PROFILE_NAME = "pi05_yam_molmoact2"
YAM_HF_REPO_ID = "robocurve/pi05-yam-molmoact2"
YAM_HF_REVISION = "df991e11e8f6540098338c56342b1143fac5b952"
YAM_TRAIN_CONFIG_NAME = "yam_pi05"
YAM_NORM_ASSET_ID = "yam-bimanual-merged"

ACTION_DIM = 14
ARX_ACTION_HORIZON = 50
ACTION_HORIZON = ARX_ACTION_HORIZON
# The released YAM checkpoint predicts 16 actions at 30 Hz. Replan after the
# first half of each chunk so closed-loop observations are incorporated every
# 0.267 seconds instead of playing the full 0.533-second prediction open-loop.
YAM_ACTION_HORIZON = 16
YAM_EXECUTED_HORIZON = 8
YAM_CONTROL_HZ = 30
GRIPPER_INDICES = (6, 13)
GRIPPER_OFFSET = -0.01
GRIPPER_SPAN = 0.054
PUBLIC_PROFILE_NAMES = frozenset((ARX_PROFILE_NAME, YAM_PROFILE_NAME))


def snapshot_path(profile_name: str = ARX_PROFILE_NAME) -> Path:
    if profile_name == ARX_PROFILE_NAME:
        revision = HF_REVISION
    elif profile_name == YAM_PROFILE_NAME:
        revision = YAM_HF_REVISION
    else:
        raise ValueError(f"Unknown public PI0.5 checkpoint profile: {profile_name}")
    return model_weight_root("Pi_05", profile_name, revision)


def checkpoint_path(profile_name: str = ARX_PROFILE_NAME) -> Path:
    snapshot = snapshot_path(profile_name)
    if profile_name == ARX_PROFILE_NAME:
        return snapshot / "checkpoints" / str(CHECKPOINT_STEP)
    return snapshot


def apply_checkpoint_profile(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Expand a public alias while retaining local/path checkpoint behavior."""

    cfg = dict(model_cfg)
    profile_name = cfg.get("ckpt_name")
    if profile_name == ARX_PROFILE_NAME:
        cfg.update(
            {
                "checkpoint_profile": ARX_PROFILE_NAME,
                "model_path": str(checkpoint_path(ARX_PROFILE_NAME)),
                "norm_stats_path": str(snapshot_path(ARX_PROFILE_NAME) / "assets"),
                "hf_repo_id": HF_REPO_ID,
                "hf_revision": HF_REVISION,
                "checkpoint_num": CHECKPOINT_STEP,
                "train_config_name": TRAIN_CONFIG_NAME,
                "repo_id": ARX_PROFILE_NAME,
                "gripper_contract": "arx5_physical",
                "actions_per_chunk": ARX_ACTION_HORIZON,
            }
        )
    elif profile_name == YAM_PROFILE_NAME:
        snapshot = snapshot_path(YAM_PROFILE_NAME)
        # deploy.yml retains a legacy local-checkpoint default. The released
        # YAM artifact is itself an Orbax checkpoint root and has no step dir.
        cfg.pop("checkpoint_num", None)
        cfg.update(
            {
                "checkpoint_profile": YAM_PROFILE_NAME,
                "model_path": str(snapshot),
                "norm_stats_path": str(snapshot / "assets" / YAM_NORM_ASSET_ID),
                "hf_repo_id": YAM_HF_REPO_ID,
                "hf_revision": YAM_HF_REVISION,
                "train_config_name": YAM_TRAIN_CONFIG_NAME,
                "repo_id": YAM_NORM_ASSET_ID,
                "embodiment_contract": _yam.ENVIRONMENT_NAME,
                "dataset_frame": "yam_molmoact2",
                "predicted_horizon": YAM_ACTION_HORIZON,
                "executed_horizon": YAM_EXECUTED_HORIZON,
                "actions_per_chunk": YAM_EXECUTED_HORIZON,
                "control_hz": YAM_CONTROL_HZ,
            }
        )
    return cfg


def is_public_checkpoint_profile(profile_name: str | None) -> bool:
    return profile_name in PUBLIC_PROFILE_NAMES


def validate_profile_runtime(
    model_cfg: dict[str, Any],
    robot_action_dim_info: dict[str, Any] | None,
) -> None:
    profile_name = model_cfg.get("checkpoint_profile")
    if not is_public_checkpoint_profile(profile_name):
        return
    if model_cfg.get("action_type", "joint") != "joint":
        raise ValueError(f"{profile_name} requires action_type='joint'.")
    if robot_action_dim_info is None:
        raise ValueError(f"{profile_name} requires an env_cfg_type.")

    env_cfg_type = model_cfg.get("env_cfg_type")
    if profile_name == ARX_PROFILE_NAME:
        if env_cfg_type != "arx_x5":
            raise ValueError(f"{ARX_PROFILE_NAME} requires env_cfg_type='arx_x5'.")
        validate_robot_contract(robot_action_dim_info)
    elif profile_name == YAM_PROFILE_NAME:
        _yam.validate_environment(env_cfg_type)
        _yam.validate_robot_contract(robot_action_dim_info)
        validate_yam_timing_contract(model_cfg)


def validate_yam_timing_contract(model_cfg: dict[str, Any]) -> None:
    """Require the public YAM profile's prediction and execution cadence."""

    expected = {
        "predicted_horizon": YAM_ACTION_HORIZON,
        "executed_horizon": YAM_EXECUTED_HORIZON,
        "actions_per_chunk": YAM_EXECUTED_HORIZON,
        "control_hz": YAM_CONTROL_HZ,
    }
    for key, expected_value in expected.items():
        actual = model_cfg.get(key, expected_value)
        if isinstance(actual, bool) or actual != expected_value:
            raise ValueError(f"{YAM_PROFILE_NAME} requires {key}={expected_value}; got {actual!r}.")


def validate_robot_contract(robot_action_dim_info: dict[str, Any]) -> None:
    """Validate the ARX profile's bimanual state/action dimensions."""

    arm_dims = list(robot_action_dim_info.get("arm_dim", ()))
    gripper_dims = list(robot_action_dim_info.get("ee_dim", ()))
    if arm_dims != [6, 6] or gripper_dims != [1, 1]:
        raise ValueError(f"PI0.5 ARX X5 requires [6,1,6,1] dimensions; got arm_dim={arm_dims}, ee_dim={gripper_dims}.")


def validate_profile_checkpoint(path: Path, profile_name: str = ARX_PROFILE_NAME) -> None:
    expected = checkpoint_path(profile_name).resolve()
    actual = path.resolve()
    if actual != expected:
        if profile_name == ARX_PROFILE_NAME:
            detail = f"checkpoint step {CHECKPOINT_STEP}"
        else:
            detail = "the pinned snapshot root"
        raise ValueError(f"{profile_name} requires {detail}: {expected}; got {actual}.")
    if not (actual / "params").is_dir():
        raise FileNotFoundError(f"Missing Orbax params directory: {actual / 'params'}")
    if profile_name == YAM_PROFILE_NAME:
        norm_stats = actual / "assets" / YAM_NORM_ASSET_ID / "norm_stats.json"
        if not norm_stats.is_file():
            raise FileNotFoundError(f"Missing PI0.5 YAM normalization stats: {norm_stats}")


def _robodojo_arx_to_checkpoint(values: Any) -> np.ndarray:
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


def _checkpoint_arx_to_robodojo(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    expected = (ARX_ACTION_HORIZON, ACTION_DIM)
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


def robodojo_state_to_checkpoint(model_cfg: dict[str, Any], values: Any) -> np.ndarray:
    profile_name = model_cfg.get("checkpoint_profile")
    if profile_name == ARX_PROFILE_NAME:
        return _robodojo_arx_to_checkpoint(values)
    if profile_name == YAM_PROFILE_NAME:
        return _yam_frame.simulator_to_dataset(_yam.validate_state(values))
    return np.asarray(values, dtype=np.float32)


def checkpoint_actions_to_robodojo(model_cfg: dict[str, Any], values: Any) -> np.ndarray:
    profile_name = model_cfg.get("checkpoint_profile")
    if profile_name == ARX_PROFILE_NAME:
        return _checkpoint_arx_to_robodojo(values)
    if profile_name == YAM_PROFILE_NAME:
        validate_yam_timing_contract(model_cfg)
        actions = _yam.validate_action_chunk(
            values,
            predicted_horizon=YAM_ACTION_HORIZON,
            executed_horizon=YAM_EXECUTED_HORIZON,
        )
        return _yam_frame.dataset_to_simulator(actions)
    return np.asarray(values, dtype=np.float32)


def validate_profile_camera_payload(model_cfg: dict[str, Any], images: dict[str, Any]) -> None:
    if model_cfg.get("checkpoint_profile") == YAM_PROFILE_NAME:
        _yam.validate_camera_payload(images)


def robodojo_to_checkpoint(values: Any) -> np.ndarray:
    """Compatibility wrapper for the original ARX profile transform."""

    return _robodojo_arx_to_checkpoint(values)


def checkpoint_to_robodojo(values: Any) -> np.ndarray:
    """Compatibility wrapper for the original ARX profile transform."""

    return _checkpoint_arx_to_robodojo(values)
