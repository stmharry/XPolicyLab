"""Public MolmoAct2 Bimanual YAM checkpoint contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from XPolicyLab.utils import bimanual_yam_contract as _yam, yam_molmoact2_frame as _yam_frame
from XPolicyLab.utils.robodojo_paths import model_weight_root

CAMERA_KEYS = _yam.CAMERA_KEYS
CAMERA_SHAPE = _yam.CAMERA_SHAPE
GRIPPER_INDICES = _yam.GRIPPER_INDICES
STATE_DIM = _yam.STATE_DIM
validate_camera_payload = _yam.validate_camera_payload
validate_environment = _yam.validate_environment
validate_robot_contract = _yam.validate_robot_contract
validate_state = _yam.validate_state
YAM_JOINT_5_INDICES = _yam_frame.JOINT_SIGN_INDICES
checkpoint_actions_to_simulator = _yam_frame.dataset_to_simulator
simulator_state_to_checkpoint = _yam_frame.simulator_to_dataset

PROFILE_NAME = "molmoact2_bimanual_yam"
SOURCE_REPOSITORY = "https://github.com/allenai/molmoact2.git"
SOURCE_REVISION = "c2282820f9b188b60e66ea1636b3efd81c45cbb4"
HF_REPO_ID = "allenai/MolmoAct2-BimanualYAM"
HF_REVISION = "8dcbed66f2380e4393189c303ea72488eb9e63c2"
NORM_TAG = "yam_dual_molmoact2"
PREDICTED_HORIZON = 30
EXECUTED_HORIZON = 30
FLOW_STEPS = 10
CANDIDATE_COUNT = 16
CAMERA_INPUT_CONTRACT = "molmoact2_bimanual_yam_640x360_center_crop_v1"
CHECKPOINT_CAMERA_SHAPE = (3, 360, 640)
MOONLAKE_CAMERA_SHAPE = (3, 480, 640)
MOONLAKE_CENTER_CROP = slice(60, 420)


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
            "candidate_count": CANDIDATE_COUNT,
            "enable_depth_reasoning": False,
            "enable_inference_cuda_graph": True,
            "warmup_runs": 3,
            "predicted_horizon": PREDICTED_HORIZON,
            "actions_per_chunk": EXECUTED_HORIZON,
            "embodiment_contract": "bimanual_yam",
            "dataset_frame": "yam_molmoact2",
            "camera_input_contract": CAMERA_INPUT_CONTRACT,
        }
    )
    return cfg


def resolve_camera_input_contract(model_cfg: dict[str, Any]) -> str | None:
    """Resolve the public checkpoint's image geometry without affecting local runs."""

    contract_name = model_cfg.get("camera_input_contract")
    if contract_name is None:
        return None
    if contract_name != CAMERA_INPUT_CONTRACT:
        raise ValueError(f"Unknown MolmoAct2 camera_input_contract: {contract_name!r}.")
    if not uses_public_yam_joint_sign_bridge(model_cfg):
        raise ValueError(
            f"camera_input_contract={CAMERA_INPUT_CONTRACT!r} is reserved for "
            f"ckpt_name={PROFILE_NAME!r} with checkpoint_backend='original_hf'."
        )
    return CAMERA_INPUT_CONTRACT


def prepare_checkpoint_camera_payload(
    images: dict[str, Any],
    *,
    camera_input_contract: str,
) -> dict[str, np.ndarray]:
    """Present source RGB at the pinned public checkpoint's 640x360 geometry."""

    if camera_input_contract != CAMERA_INPUT_CONTRACT:
        raise ValueError(f"Unknown MolmoAct2 camera_input_contract: {camera_input_contract!r}.")

    validate_camera_payload(images)
    prepared: dict[str, np.ndarray] = {}
    for camera in CAMERA_KEYS:
        source = np.asarray(images[camera])
        if source.shape == CHECKPOINT_CAMERA_SHAPE:
            checkpoint_image = source
        elif source.shape == MOONLAKE_CAMERA_SHAPE:
            checkpoint_image = source[:, MOONLAKE_CENTER_CROP, :]
        else:  # Defensive if the shared source contract expands independently.
            raise ValueError(
                f"{camera} cannot be presented to {camera_input_contract!r}: "
                f"expected one of {(CHECKPOINT_CAMERA_SHAPE, MOONLAKE_CAMERA_SHAPE)}, "
                f"got {source.shape}."
            )
        prepared[camera] = np.ascontiguousarray(checkpoint_image)

    for camera, image in prepared.items():
        if image.shape != CHECKPOINT_CAMERA_SHAPE or image.dtype != np.uint8:
            raise ValueError(
                f"{camera} must reach the MolmoAct2 checkpoint as uint8 CHW "
                f"{CHECKPOINT_CAMERA_SHAPE}, got dtype={image.dtype}, shape={image.shape}."
            )
    return prepared


def uses_public_yam_joint_sign_bridge(model_cfg: dict[str, Any]) -> bool:
    """Return whether the pinned original-HF YAM boundary needs sign bridging."""

    return model_cfg.get("ckpt_name") == PROFILE_NAME and model_cfg.get("checkpoint_backend") == "original_hf"


def validate_and_select_actions(actions: Any) -> np.ndarray:
    return _yam.validate_action_chunk(
        actions,
        predicted_horizon=PREDICTED_HORIZON,
        executed_horizon=EXECUTED_HORIZON,
    )
