from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.utils import prepare_observation_for_inference
import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate

RIGHT_LIMITS = np.asarray(
    [[-75, 75], [-9, 90], [-85, 85], [0, 135], [-85, 85], [-40, 40], [-80, 80]], dtype=np.float32
)
LEFT_LIMITS = np.asarray(
    [[-75, 75], [-90, 9], [-85, 85], [0, 135], [-85, 85], [-40, 40], [-80, 80]], dtype=np.float32
)
GRIPPER_DEGREES = 65.0
GRIPPER_OPENING_M = 0.044


def gripper_m_to_degrees(value):
    return np.clip(-GRIPPER_DEGREES * np.asarray(value) / GRIPPER_OPENING_M, -GRIPPER_DEGREES, 0.0)


def gripper_degrees_to_m(value):
    return np.clip(-np.asarray(value) * GRIPPER_OPENING_M / GRIPPER_DEGREES, 0.0, GRIPPER_OPENING_M)


def pack_openarm_state(observation: dict[str, Any]) -> np.ndarray:
    state = observation["state"]
    right = np.rad2deg(np.asarray(state["right_arm_joint_state"], dtype=np.float32))
    left = np.rad2deg(np.asarray(state["left_arm_joint_state"], dtype=np.float32))
    right_gripper = gripper_m_to_degrees(np.asarray(state["right_ee_joint_state"])[0:1])
    left_gripper = gripper_m_to_degrees(np.asarray(state["left_ee_joint_state"])[0:1])
    packed = np.concatenate((right, right_gripper, left, left_gripper)).astype(np.float32)
    if packed.shape != (16,) or not np.isfinite(packed).all():
        raise ValueError(f"expected finite OpenARM state shape (16,), got {packed.shape}")
    return packed


def unpack_openarm_action(action) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (16,) or not np.isfinite(action).all():
        raise ValueError(f"expected finite OpenARM action shape (16,), got {action.shape}")
    right_deg = np.clip(action[:7], RIGHT_LIMITS[:, 0], RIGHT_LIMITS[:, 1])
    left_deg = np.clip(action[8:15], LEFT_LIMITS[:, 0], LEFT_LIMITS[:, 1])
    return {
        "left_arm_joint_state": np.deg2rad(left_deg).astype(np.float32),
        "left_ee_joint_state": np.asarray([gripper_degrees_to_m(action[15])], dtype=np.float32),
        "right_arm_joint_state": np.deg2rad(right_deg).astype(np.float32),
        "right_ee_joint_state": np.asarray([gripper_degrees_to_m(action[7])], dtype=np.float32),
    }


def _decode_image(value) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("color", value.get("rgb"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = np.frombuffer(value, dtype=np.uint8)
    image = np.asarray(value)
    if image.ndim == 1:
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"expected HWC camera image, got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    return np.ascontiguousarray(image.astype(np.uint8, copy=False))


def _camera(observation, name):
    vision = observation["vision"]
    aliases = {
        "base": ("cam_head", "base", "cam_high"),
        "left_wrist": ("cam_left_wrist", "left_wrist"),
        "right_wrist": ("cam_right_wrist", "right_wrist"),
    }
    for key in aliases[name]:
        if key in vision:
            return _decode_image(vision[key])
    raise KeyError(f"missing {name} camera; tried {aliases[name]}")


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        checkpoint = Path(model_cfg["checkpoint_path"]).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parent / checkpoint
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        self.prompt = model_cfg.get("prompt", "Fold the T-shirt properly.")
        self.chunk_size = int(model_cfg.get("chunk_size", 30))
        if self.chunk_size != 30:
            raise ValueError("folding_final requires chunk_size=30")
        self.device = torch.device("cuda")
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if config.input_features["observation.state"].shape != (16,):
            raise ValueError("folding_final checkpoint must accept 16-dimensional state")
        self.policy = PI05Policy.from_pretrained(checkpoint, config=config).to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        self.model = self.policy
        self.observation = None

    def update_obs(self, obs):
        self.observation = obs

    def update_obs_batch(self, obs):
        self.observation = obs

    def _action_chunk(self, observation):
        raw = {
            "observation.state": pack_openarm_state(observation),
            "observation.images.left_wrist": _camera(observation, "left_wrist"),
            "observation.images.right_wrist": _camera(observation, "right_wrist"),
            "observation.images.base": _camera(observation, "base"),
        }
        prepared = prepare_observation_for_inference(raw, self.device, self.prompt, "bi_openarm_follower")
        prepared = self.preprocessor(prepared)
        actions = []
        with torch.inference_mode(), nullcontext():
            for _ in range(self.chunk_size):
                action = self.policy.select_action(prepared)
                action = self.postprocessor(action)
                actions.append(unpack_openarm_action(action.detach().cpu().numpy()))
        return actions

    def get_action(self):
        if self.observation is None:
            raise RuntimeError("update_obs must be called before get_action")
        return self._action_chunk(self.observation)

    def get_action_batch(self, env_idx_list=None):
        observations = self.observation if isinstance(self.observation, list) else [self.observation]
        return [self._action_chunk(obs) for obs in observations]

    def reset(self):
        self.observation = None
        self.policy.reset()
        for processor in (self.preprocessor, self.postprocessor):
            if hasattr(processor, "reset"):
                processor.reset()
