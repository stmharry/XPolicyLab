"""Protocol-faithful asynchronous π0.5 rollout for the OpenARM simulator."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
from threading import Lock, Thread
import time

import numpy as np

from XPolicyLab.policy.LeRobot_Pi05_OpenArm.protocol import (
    ACTION_QUEUE_SIZE,
    INTERPOLATION_MULTIPLIER,
    POLICY_FPS,
    RTC_EXECUTION_HORIZON,
    clamp_relative_target,
    finite_action_chunk,
    interpolate_action,
    pack_openarm_state,
    physics_tick_pattern,
    unpack_openarm_action,
)


class _ActionQueue:
    def __init__(self):
        self._processed = np.empty((0, 16), dtype=np.float32)
        self._original = np.empty((0, 16), dtype=np.float32)
        self._index = 0
        self._lock = Lock()

    def qsize(self) -> int:
        with self._lock:
            return max(0, len(self._processed) - self._index)

    def get(self) -> np.ndarray | None:
        with self._lock:
            if self._index >= len(self._processed):
                return None
            action = self._processed[self._index].copy()
            self._index += 1
            return action

    def processed_leftover(self) -> np.ndarray | None:
        with self._lock:
            if self._index >= len(self._processed):
                return None
            return self._processed[self._index :].copy()

    def original_leftover(self) -> np.ndarray | None:
        with self._lock:
            if self._index >= len(self._original):
                return None
            return self._original[self._index :].copy()

    def action_index(self) -> int:
        with self._lock:
            return self._index

    def merge(self, processed, original, real_delay: int) -> None:
        processed = finite_action_chunk(processed)
        original = finite_action_chunk(original)
        delay = max(0, min(int(real_delay), ACTION_QUEUE_SIZE))
        with self._lock:
            self._processed = processed[delay:].copy()
            self._original = original[delay:].copy()
            self._index = 0


class _InferenceRequest:
    def __init__(
        self,
        model_client,
        observation: dict,
        previous_actions,
        inference_delay: int,
        *,
        prefix_space: str,
        action_index: int,
    ):
        request_observation = deepcopy(observation)
        request_observation["_rtc"] = {
            "inference_delay": int(inference_delay),
            "prefix_space": prefix_space,
            "previous_actions": None if previous_actions is None else np.asarray(previous_actions).tolist(),
        }
        self.action_index = int(action_index)
        self.result = None
        self.error = None
        self.latency_s = 0.0

        def run():
            started = time.perf_counter()
            try:
                model_client.call(func_name="update_obs", obs=request_observation)
                self.result = model_client.call(func_name="get_action")
            except BaseException as exc:  # propagated on the simulator thread
                self.error = exc
            finally:
                self.latency_s = time.perf_counter() - started

        self.thread = Thread(target=run, daemon=True, name="OpenArmPi05Inference")
        self.thread.start()

    @property
    def done(self) -> bool:
        return not self.thread.is_alive()

    def wait(self) -> None:
        self.thread.join()
        if self.error is not None:
            raise self.error


def _merge_finished_request(request: _InferenceRequest, queue: _ActionQueue, latencies: list[float]):
    request.wait()
    if not isinstance(request.result, dict):
        raise TypeError("RTC policy response must contain processed_actions and original_actions")
    latency_steps = math.ceil(request.latency_s * POLICY_FPS)
    queue.merge(
        request.result["processed_actions"],
        request.result["original_actions"],
        latency_steps,
    )
    latencies.append(request.latency_s)


class _RolloutTrace:
    def __init__(self, task_env, mode: str):
        configured = os.environ.get("ROBODOJO_OPENARM_TRACE_PATH")
        self.enabled = bool(
            configured or os.environ.get("ROBODOJO_OPENARM_TRACE") == "1" or mode != "current"
        )
        self._file = None
        if not self.enabled:
            self.path = None
            return
        self.path = Path(configured) if configured else Path(task_env.save_dir) / "openarm_trace.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self.write("metadata", mode=mode, policy_fps=POLICY_FPS)

    @staticmethod
    def _value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def write(self, event: str, **values) -> None:
        if not self.enabled:
            return
        payload = {"event": event, "monotonic_s": time.monotonic()}
        payload.update({key: self._value(value) for key, value in values.items()})
        self._file.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def _diagnostic_mode() -> str:
    mode = os.environ.get("ROBODOJO_OPENARM_RTC_MODE", "current").strip().lower()
    if mode not in {"current", "official", "synchronous"}:
        raise ValueError("ROBODOJO_OPENARM_RTC_MODE must be current, official, or synchronous")
    return mode


def _camera_summaries(observation: dict) -> dict:
    summaries = {}
    for name, value in observation.get("vision", {}).items():
        if isinstance(value, dict):
            value = value.get("color", value.get("rgb"))
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            summaries[name] = {"shape": list(image.shape), "encoded": image.ndim == 1}
            continue
        rgb = image[..., :3].astype(np.float32)
        if np.issubdtype(image.dtype, np.floating) and rgb.max(initial=0.0) <= 1.0:
            rgb *= 255.0
        summaries[name] = {
            "shape": list(image.shape),
            "mean_rgb": rgb.mean(axis=(0, 1)).tolist(),
            "black_pixel_fraction": float(np.all(rgb <= 2.0, axis=-1).mean()),
            "saturated_pixel_fraction": float(np.all(rgb >= 253.0, axis=-1).mean()),
        }
    return summaries


def eval_one_episode(TASK_ENV, model_client):
    smoke_steps = os.environ.get("ROBODOJO_OPENARM_SMOKE_STEPS")
    if smoke_steps is not None:
        if os.environ.get("ROBODOJO_OPENARM_ZERO_ACTION") != "1":
            raise ValueError("ROBODOJO_OPENARM_SMOKE_STEPS is restricted to zero-action smoke mode")
        TASK_ENV.step_lim = int(smoke_steps)
        if TASK_ENV.step_lim < max((0, 10, 30)):
            raise ValueError("visual smoke must include reference frame 30")
    model_client.call(func_name="reset")
    mode = _diagnostic_mode()
    trace = _RolloutTrace(TASK_ENV, mode)
    queue = _ActionQueue()
    latencies: list[float] = []
    previous_policy_action = None
    outer_deadline = time.monotonic() + 2000.0
    current_observation = TASK_ENV.get_obs()
    trace.write("observation", step=0, state=pack_openarm_state(current_observation))
    trace.write("camera_summary", step=0, cameras=_camera_summaries(current_observation))
    request = _InferenceRequest(
        model_client,
        current_observation,
        None,
        inference_delay=0,
        prefix_space="none",
        action_index=0,
    )

    # There is no prior chunk to execute during cold start.
    request.wait()
    trace.write(
        "inference",
        latency_s=request.latency_s,
        action_index_before=request.action_index,
        action_index_after=queue.action_index(),
        original_actions=np.asarray(request.result["original_actions"]),
        processed_actions=np.asarray(request.result["processed_actions"]),
    )
    _merge_finished_request(request, queue, latencies)
    request = None

    while not TASK_ENV.is_episode_end() and time.monotonic() < outer_deadline:
        if request is not None and request.done:
            request.wait()
            trace.write(
                "inference",
                latency_s=request.latency_s,
                action_index_before=request.action_index,
                action_index_after=queue.action_index(),
                original_actions=np.asarray(request.result["original_actions"]),
                processed_actions=np.asarray(request.result["processed_actions"]),
            )
            _merge_finished_request(request, queue, latencies)
            request = None

        threshold = ACTION_QUEUE_SIZE if mode == "current" else ACTION_QUEUE_SIZE - RTC_EXECUTION_HORIZON
        if mode == "synchronous":
            threshold = 0
        if request is None and queue.qsize() <= threshold:
            historical_delay = math.ceil((max(latencies) if latencies else 0.0) * POLICY_FPS)
            if mode == "official":
                previous_actions = queue.original_leftover()
                prefix_space = "original"
            elif mode == "current":
                previous_actions = queue.processed_leftover()
                prefix_space = "absolute"
            else:
                previous_actions = None
                prefix_space = "none"
            request = _InferenceRequest(
                model_client,
                current_observation,
                previous_actions,
                inference_delay=0 if mode == "synchronous" else historical_delay,
                prefix_space=prefix_space,
                action_index=queue.action_index(),
            )

        policy_action = queue.get()
        if policy_action is None:
            if request is None:
                raise RuntimeError("RTC action queue is empty without an active inference request")
            time.sleep(0.001)
            continue

        step_started = time.perf_counter()
        interpolated = interpolate_action(
            previous_policy_action,
            policy_action,
            multiplier=INTERPOLATION_MULTIPLIER,
        )
        current_degrees = pack_openarm_state(current_observation)
        safe_actions = []
        for target in interpolated:
            unclamped = np.asarray(target, dtype=np.float32)
            clamped = clamp_relative_target(unclamped, current_degrees)
            trace.write(
                "control_target",
                step=int(TASK_ENV.take_action_cnt[0]) + 1,
                measured_before=current_degrees,
                target=unclamped,
                clamped=clamped,
                clamp_delta=clamped - unclamped,
            )
            current_degrees = clamped
            safe_actions.append(unpack_openarm_action(current_degrees))

        if len(safe_actions) == INTERPOLATION_MULTIPLIER:
            tick_pattern = physics_tick_pattern()
        else:
            # LeRobot intentionally does not interpolate the first action. It
            # still occupies one complete 30 Hz policy interval in simulation.
            tick_pattern = (sum(physics_tick_pattern()),)
        TASK_ENV.take_interpolated_action(safe_actions, tick_pattern)
        previous_policy_action = policy_action

        if not TASK_ENV.is_episode_end():
            current_observation = TASK_ENV.get_obs()
            trace.write(
                "observation",
                step=int(TASK_ENV.take_action_cnt[0]),
                state=pack_openarm_state(current_observation),
                queue_size=queue.qsize(),
            )

        elapsed = time.perf_counter() - step_started
        time.sleep(max(0.0, 1.0 / POLICY_FPS - elapsed))

    if request is not None:
        request.wait()
    trace.close()


def eval_one_episode_batch(TASK_ENV, model_client):
    if getattr(TASK_ENV, "num_envs", 1) != 1:
        raise NotImplementedError("protocol-faithful RTC evaluation currently requires eval_batch=false")
    return eval_one_episode(TASK_ENV, model_client)
