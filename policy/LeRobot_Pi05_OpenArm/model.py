import os
from pathlib import Path
from typing import Any

import cv2
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import NormalizerProcessorStep, RelativeActionsProcessorStep, TransitionKey, create_transition
from lerobot.processor.relative_action_processor import to_relative_actions
import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.policy.LeRobot_Pi05_OpenArm.protocol import (
    ACTION_QUEUE_SIZE,
    LEFT_LIMITS,
    RIGHT_LIMITS,
    RTC_EXECUTION_HORIZON,
    RTC_MAX_GUIDANCE_WEIGHT,
    finite_action_chunk,
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


def _reanchor_relative_rtc_prefix(
    previous_absolute: torch.Tensor,
    current_state: torch.Tensor,
    relative_step: RelativeActionsProcessorStep,
    normalizer_step: NormalizerProcessorStep | None,
    device: torch.device,
) -> torch.Tensor:
    """Mirror LeRobot's pinned relative-action RTC prefix conversion."""

    state = current_state.detach().cpu()
    if state.ndim == 1:
        state = state.unsqueeze(0)
    absolute = previous_absolute.detach().cpu()
    mask = relative_step._build_mask(absolute.shape[-1])
    relative = to_relative_actions(absolute, state, mask)
    transition = create_transition(action=relative)
    if normalizer_step is not None:
        transition = normalizer_step(transition)
    return transition[TransitionKey.ACTION].to(device)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.zero_action_smoke = os.environ.get("ROBODOJO_OPENARM_ZERO_ACTION") == "1"
        self.prompt = model_cfg.get("prompt", "Fold the T-shirt properly.")
        self.chunk_size = int(model_cfg.get("chunk_size", ACTION_QUEUE_SIZE))
        if self.chunk_size != ACTION_QUEUE_SIZE:
            raise ValueError("folding_final requires chunk_size=30")
        self.observation = None
        if self.zero_action_smoke:
            self.device = torch.device("cpu")
            self.policy = None
            self.model = None
            return

        checkpoint = Path(model_cfg["checkpoint_path"]).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parent / checkpoint
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        self.device = torch.device("cuda")
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if config.input_features["observation.state"].shape != (16,):
            raise ValueError("folding_final checkpoint must accept 16-dimensional state")
        # The pinned real-robot evaluator defaults use_torch_compile=false and
        # overrides the saved training config before constructing PI0/PI0.5.
        config.compile_model = False
        config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=RTC_EXECUTION_HORIZON,
            max_guidance_weight=RTC_MAX_GUIDANCE_WEIGHT,
            prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
        )
        self.policy = PI05Policy.from_pretrained(checkpoint, config=config).to(self.device).eval()
        self.policy.config.rtc_config = config.rtc_config
        self.policy.init_rtc_processor()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        self.relative_step = next(
            (
                step
                for step in self.preprocessor.steps
                if isinstance(step, RelativeActionsProcessorStep) and step.enabled
            ),
            None,
        )
        self.normalizer_step = next(
            (step for step in self.preprocessor.steps if isinstance(step, NormalizerProcessorStep)),
            None,
        )
        if self.relative_step is None:
            raise ValueError("folding_final must retain its saved relative-action processor")
        if self.relative_step.action_names is None:
            action_names = getattr(config, "action_feature_names", None)
            if not action_names:
                raise ValueError("folding_final checkpoint is missing action feature names")
            self.relative_step.action_names = list(action_names)
        self.model = self.policy

    def update_obs(self, obs):
        self.observation = obs

    def update_obs_batch(self, obs):
        self.observation = obs

    def _predict_action_chunk(self, observation):
        packed_state = pack_openarm_state(observation)
        if self.zero_action_smoke:
            held = np.repeat(packed_state[None, :], ACTION_QUEUE_SIZE, axis=0)
            return {"processed_actions": held, "original_actions": held.copy()}
        raw = {
            "observation.state": packed_state,
            "observation.images.left_wrist": _camera(observation, "left_wrist"),
            "observation.images.right_wrist": _camera(observation, "right_wrist"),
            "observation.images.base": _camera(observation, "base"),
        }
        prepared = prepare_observation_for_inference(raw, self.device, self.prompt, "bi_openarm_follower")
        prepared = self.preprocessor(prepared)

        rtc_meta = observation.get("_rtc", {})
        inference_delay = max(0, int(rtc_meta.get("inference_delay", 0)))
        previous_actions = rtc_meta.get("previous_actions")
        prefix_space = rtc_meta.get("prefix_space", "none")
        previous_prefix = None
        if previous_actions is not None:
            previous_actions = torch.as_tensor(previous_actions, dtype=torch.float32)
            if previous_actions.ndim != 2 or previous_actions.shape[-1] != 16:
                raise ValueError("RTC previous_actions must have shape (remaining_steps, 16)")
            if prefix_space == "original":
                previous_prefix = previous_actions.to(self.device)
            elif prefix_space == "absolute":
                previous_prefix = _reanchor_relative_rtc_prefix(
                    previous_absolute=previous_actions,
                    current_state=torch.as_tensor(packed_state, dtype=torch.float32),
                    relative_step=self.relative_step,
                    normalizer_step=self.normalizer_step,
                    device=self.device,
                )
            else:
                raise ValueError("RTC prefix_space must be original or absolute when a prefix is supplied")

        # RTC guidance differentiates its overlap objective with respect to the
        # candidate action chunk. The official evaluator therefore calls this
        # method without an outer no-grad/inference-mode context.
        actions = self.policy.predict_action_chunk(
            prepared,
            inference_delay=inference_delay,
            prev_chunk_left_over=previous_prefix,
        )
        original_actions = actions.squeeze(0).detach().cpu()
        processed_actions = self.postprocessor(actions).squeeze(0).detach().cpu()

        original = finite_action_chunk(original_actions.numpy())
        processed = finite_action_chunk(processed_actions.numpy())
        return {
            "processed_actions": processed,
            "original_actions": original,
        }

    def _action_chunk(self, observation):
        result = self._predict_action_chunk(observation)
        return [unpack_openarm_action(action) for action in result["processed_actions"]]

    def get_action(self):
        if self.observation is None:
            raise RuntimeError("update_obs must be called before get_action")
        result = self._predict_action_chunk(self.observation)
        return {
            "processed_actions": result["processed_actions"].tolist(),
            "original_actions": result["original_actions"].tolist(),
        }

    def get_action_batch(self, env_idx_list=None):
        observations = self.observation if isinstance(self.observation, list) else [self.observation]
        results = []
        for observation in observations:
            result = self._predict_action_chunk(observation)
            results.append(
                {
                    "processed_actions": result["processed_actions"].tolist(),
                    "original_actions": result["original_actions"].tolist(),
                }
            )
        return results

    def reset(self):
        self.observation = None
        if self.policy is not None:
            self.policy.reset()
            for processor in (self.preprocessor, self.postprocessor):
                if hasattr(processor, "reset"):
                    processor.reset()
