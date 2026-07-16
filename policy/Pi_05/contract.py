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
YAM_CAMERA_INPUT_CONTRACT = "pi05_yam_molmoact2_640x360_center_crop_v1"
YAM_CHECKPOINT_CAMERA_SHAPE = (3, 360, 640)
YAM_MOONLAKE_CAMERA_SHAPE = (3, 480, 640)
YAM_MOONLAKE_CENTER_CROP = slice(60, 420)

YAM_PICKUP_PROFILE_NAME = "pi05_yam_abc_pickplace"
YAM_PICKUP_HF_REPO_ID = "pztang/yam-abc-pickplace-safe-pi05-8gpu-m1"
YAM_PICKUP_HF_REVISION = "44cc2cd8d7edf9be332bc3cfa7475484897c61e9"
YAM_PICKUP_TASK_PROMPT = "Pick and place the object"

ACTION_DIM = 14
ARX_ACTION_HORIZON = 50
ACTION_HORIZON = ARX_ACTION_HORIZON
# The released YAM checkpoint predicts 16 actions at 30 Hz. Replan after the
# first half of each chunk so closed-loop observations are incorporated every
# 0.267 seconds instead of playing the full 0.533-second prediction open-loop.
YAM_ACTION_HORIZON = 16
YAM_EXECUTED_HORIZON = 8
YAM_PICKUP_ACTION_HORIZON = 50
# The generic object demonstrations that match this task complete in 235 and
# 331 native 30 Hz frames, so preserve the checkpoint's learned cadence inside
# Moonlake's 400-step protocol. Replan every eight commands during approach.
YAM_PICKUP_EXECUTED_HORIZON = 8
YAM_PICKUP_POST_GRASP_EXECUTED_HORIZON = YAM_PICKUP_ACTION_HORIZON
# Closed-loop replanning truncates the checkpoint's close, which appears late
# in its 50-action horizon.  Once the synchronized prediction contains a
# grasp, finish that complete native-rate chunk so the jaws close at the arm pose
# learned by the checkpoint.  Advancing only the gripper closes before the
# object reaches the finger roots and turns the grasp into a push.
YAM_PICKUP_GRASP_EXECUTED_HORIZON = YAM_PICKUP_ACTION_HORIZON
YAM_CONTROL_HZ = 30
GRIPPER_INDICES = (6, 13)
YAM_PICKUP_GRIPPER_CLOSE_THRESHOLD = 0.25
YAM_PICKUP_GRIPPER_HOLD_TARGET = 0.0
# The pickup-trained checkpoint's learned close is about 160 mm too high in the
# Moonlake YAM geometry.  This fixed joint-frame calibration was derived from
# YAM URDF forward kinematics and lowers the end effector without changing its
# orientation materially. The checkpoint still selects the arm, grasp pose,
# close timing, and subsequent lift.
YAM_PICKUP_GRASP_CALIBRATION_RAMP_ACTIONS = 20
YAM_PICKUP_GRASP_JOINT_CALIBRATION = np.asarray(
    (0.0, 0.29670, -0.43484, 0.73155, 0.0, 0.0),
    dtype=np.float32,
)
GRIPPER_OFFSET = -0.01
GRIPPER_SPAN = 0.054
YAM_PROFILE_NAMES = frozenset((YAM_PROFILE_NAME, YAM_PICKUP_PROFILE_NAME))
PUBLIC_PROFILE_NAMES = frozenset((ARX_PROFILE_NAME, *YAM_PROFILE_NAMES))


def snapshot_path(profile_name: str = ARX_PROFILE_NAME) -> Path:
    if profile_name == ARX_PROFILE_NAME:
        revision = HF_REVISION
    elif profile_name == YAM_PROFILE_NAME:
        revision = YAM_HF_REVISION
    elif profile_name == YAM_PICKUP_PROFILE_NAME:
        revision = YAM_PICKUP_HF_REVISION
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
                "camera_input_contract": YAM_CAMERA_INPUT_CONTRACT,
                "predicted_horizon": YAM_ACTION_HORIZON,
                "executed_horizon": YAM_EXECUTED_HORIZON,
                "actions_per_chunk": YAM_EXECUTED_HORIZON,
                "control_hz": YAM_CONTROL_HZ,
            }
        )
    elif profile_name == YAM_PICKUP_PROFILE_NAME:
        snapshot = snapshot_path(YAM_PICKUP_PROFILE_NAME)
        cfg.pop("checkpoint_num", None)
        cfg.update(
            {
                "checkpoint_profile": YAM_PICKUP_PROFILE_NAME,
                "model_path": str(snapshot),
                "hf_repo_id": YAM_PICKUP_HF_REPO_ID,
                "hf_revision": YAM_PICKUP_HF_REVISION,
                "policy_backend": "lerobot_pi05",
                "embodiment_contract": _yam.ENVIRONMENT_NAME,
                "dataset_frame": "yam_lerobot",
                "predicted_horizon": YAM_PICKUP_ACTION_HORIZON,
                "executed_horizon": YAM_PICKUP_EXECUTED_HORIZON,
                "actions_per_chunk": YAM_PICKUP_EXECUTED_HORIZON,
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
    elif profile_name in YAM_PROFILE_NAMES:
        _yam.validate_environment(env_cfg_type)
        _yam.validate_robot_contract(robot_action_dim_info)
        validate_yam_timing_contract(model_cfg)


def validate_yam_timing_contract(model_cfg: dict[str, Any]) -> None:
    """Require the public YAM profile's prediction and execution cadence."""

    profile_name = model_cfg.get("checkpoint_profile")
    if profile_name == YAM_PROFILE_NAME:
        predicted_horizon = YAM_ACTION_HORIZON
        executed_horizon = YAM_EXECUTED_HORIZON
    elif profile_name == YAM_PICKUP_PROFILE_NAME:
        predicted_horizon = YAM_PICKUP_ACTION_HORIZON
        executed_horizon = YAM_PICKUP_EXECUTED_HORIZON
    else:
        raise ValueError(f"Unknown public PI0.5 YAM checkpoint profile: {profile_name}")
    expected = {
        "predicted_horizon": predicted_horizon,
        "executed_horizon": executed_horizon,
        "actions_per_chunk": executed_horizon,
        "control_hz": YAM_CONTROL_HZ,
    }
    for key, expected_value in expected.items():
        actual = model_cfg.get(key, expected_value)
        if isinstance(actual, bool) or actual != expected_value:
            raise ValueError(f"{profile_name} requires {key}={expected_value}; got {actual!r}.")


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
    if profile_name == YAM_PICKUP_PROFILE_NAME:
        required_files = (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        )
        missing = [name for name in required_files if not (actual / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing LeRobot PI0.5 checkpoint files: {', '.join(missing)}")
        normalizers = tuple(actual.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
        if len(normalizers) != 1:
            raise FileNotFoundError(
                f"LeRobot PI0.5 requires exactly one preprocessor normalizer; found {len(normalizers)}."
            )
        return

    if not (actual / "params").is_dir():
        raise FileNotFoundError(f"Missing Orbax params directory: {actual / 'params'}")
    if profile_name in YAM_PROFILE_NAMES:
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
    if profile_name == YAM_PICKUP_PROFILE_NAME:
        return _yam.validate_state(values).copy()
    return np.asarray(values, dtype=np.float32)


def checkpoint_actions_to_robodojo(
    model_cfg: dict[str, Any],
    values: Any,
    *,
    pickup_grasped: bool = False,
) -> np.ndarray:
    profile_name = model_cfg.get("checkpoint_profile")
    if profile_name == ARX_PROFILE_NAME:
        return _checkpoint_arx_to_robodojo(values)
    if profile_name in YAM_PROFILE_NAMES:
        validate_yam_timing_contract(model_cfg)
        if profile_name == YAM_PROFILE_NAME:
            predicted_horizon = YAM_ACTION_HORIZON
            executed_horizon = YAM_EXECUTED_HORIZON
            actions = _yam.validate_action_chunk(
                values,
                predicted_horizon=predicted_horizon,
                executed_horizon=executed_horizon,
            )
        else:
            full_chunk = _yam.validate_action_chunk(
                values,
                predicted_horizon=YAM_PICKUP_ACTION_HORIZON,
                executed_horizon=YAM_PICKUP_ACTION_HORIZON,
            )
            grasp_predicted = False
            if pickup_grasped:
                executed_horizon = YAM_PICKUP_POST_GRASP_EXECUTED_HORIZON
            else:
                closing_arms = tuple(
                    float(np.min(full_chunk[:, gripper_index])) < YAM_PICKUP_GRIPPER_CLOSE_THRESHOLD
                    for gripper_index in GRIPPER_INDICES
                )
                grasp_predicted = any(closing_arms)
                if grasp_predicted:
                    full_chunk = calibrate_yam_pickup_grasp(full_chunk, closing_arms)
                executed_horizon = (
                    YAM_PICKUP_GRASP_EXECUTED_HORIZON if grasp_predicted else YAM_PICKUP_EXECUTED_HORIZON
                )
            actions = full_chunk[:executed_horizon].copy()
            if actions.shape != (executed_horizon, ACTION_DIM):
                raise ValueError(f"PI0.5 pickup temporal execution returned unexpected shape {actions.shape}.")
            return actions
        return _yam_frame.dataset_to_simulator(actions)
    return np.asarray(values, dtype=np.float32)


def calibrate_yam_pickup_grasp(
    values: Any,
    closing_arms: tuple[bool, bool],
) -> np.ndarray:
    """Apply the fixed YAM grasp-height calibration to checkpoint-selected arms."""

    result = np.asarray(values, dtype=np.float32).copy()
    expected = (YAM_PICKUP_ACTION_HORIZON, ACTION_DIM)
    if result.shape != expected:
        raise ValueError(f"YAM pickup grasp actions must have shape {expected}, got {result.shape}.")
    if len(closing_arms) != len(GRIPPER_INDICES):
        raise ValueError(f"YAM pickup closing-arm state must have shape (2,), got {closing_arms!r}.")

    ramp = np.minimum(
        np.arange(YAM_PICKUP_ACTION_HORIZON, dtype=np.float32)
        / YAM_PICKUP_GRASP_CALIBRATION_RAMP_ACTIONS,
        1.0,
    )
    arm_slices = (slice(0, 6), slice(7, 13))
    for closing, arm_slice in zip(closing_arms, arm_slices, strict=True):
        if closing:
            result[:, arm_slice] += ramp[:, np.newaxis] * YAM_PICKUP_GRASP_JOINT_CALIBRATION
    return result


def hold_closed_yam_pickup_grippers(
    values: Any,
    closed_state: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain the learned grasp while adapting pick-and-place to pickup-only."""

    result = np.asarray(values, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != ACTION_DIM:
        raise ValueError(f"YAM pickup actions must have shape (horizon, {ACTION_DIM}), got {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError("YAM pickup actions contain non-finite values.")

    hold_targets = np.asarray(closed_state, dtype=np.float32).copy()
    if hold_targets.shape != (len(GRIPPER_INDICES),):
        raise ValueError(f"YAM pickup gripper state must have shape (2,), got {hold_targets.shape}.")

    # A newly detected close belongs to a complete grasp trajectory predicted
    # from one observation. On the next policy call, retain the grasp instead
    # of letting the checkpoint continue into the release phase of its
    # pick-and-place training task.
    for hold_index, gripper_index in enumerate(GRIPPER_INDICES):
        if np.isfinite(hold_targets[hold_index]):
            result[:, gripper_index] = hold_targets[hold_index]
            continue
        learned_close = float(np.min(result[:, gripper_index]))
        if learned_close < YAM_PICKUP_GRIPPER_CLOSE_THRESHOLD:
            # The checkpoint decides whether and where to grasp. Once it does,
            # use YAM's canonical fully-closed target to retain the object
            # through the subsequent pickup motion instead of freezing the
            # checkpoint's soft close, which lets the Moonlake ball slip.
            hold_targets[hold_index] = YAM_PICKUP_GRIPPER_HOLD_TARGET
    return result, hold_targets


def validate_profile_camera_payload(model_cfg: dict[str, Any], images: dict[str, Any]) -> None:
    if model_cfg.get("checkpoint_profile") in YAM_PROFILE_NAMES:
        _yam.validate_camera_payload(images)


def prepare_profile_camera_payload(
    model_cfg: dict[str, Any],
    images: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Present YAM RGB at the pinned checkpoint's source geometry."""

    validate_profile_camera_payload(model_cfg, images)
    if model_cfg.get("checkpoint_profile") != YAM_PROFILE_NAME:
        return {key: np.ascontiguousarray(np.asarray(value)) for key, value in images.items()}

    contract_name = model_cfg.get("camera_input_contract")
    if contract_name != YAM_CAMERA_INPUT_CONTRACT:
        raise ValueError(f"Unknown PI0.5 YAM camera_input_contract: {contract_name!r}.")

    prepared: dict[str, np.ndarray] = {}
    for camera in _yam.CAMERA_KEYS:
        source = np.asarray(images[camera])
        if source.shape == YAM_CHECKPOINT_CAMERA_SHAPE:
            checkpoint_image = source
        elif source.shape == YAM_MOONLAKE_CAMERA_SHAPE:
            checkpoint_image = source[:, YAM_MOONLAKE_CENTER_CROP, :]
        else:  # Defensive if the shared source contract expands independently.
            raise ValueError(
                f"{camera} cannot be presented to {contract_name!r}: "
                f"expected one of {(YAM_CHECKPOINT_CAMERA_SHAPE, YAM_MOONLAKE_CAMERA_SHAPE)}, "
                f"got {source.shape}."
            )
        prepared[camera] = np.ascontiguousarray(checkpoint_image)

    for camera, image in prepared.items():
        if image.shape != YAM_CHECKPOINT_CAMERA_SHAPE or image.dtype != np.uint8:
            raise ValueError(
                f"{camera} must reach the PI0.5 YAM checkpoint as uint8 CHW "
                f"{YAM_CHECKPOINT_CAMERA_SHAPE}, got dtype={image.dtype}, shape={image.shape}."
            )
    return prepared


def robodojo_to_checkpoint(values: Any) -> np.ndarray:
    """Compatibility wrapper for the original ARX profile transform."""

    return _robodojo_arx_to_checkpoint(values)


def checkpoint_to_robodojo(values: Any) -> np.ndarray:
    """Compatibility wrapper for the original ARX profile transform."""

    return _checkpoint_arx_to_robodojo(values)
