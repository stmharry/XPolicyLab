"""Protocol-faithful asynchronous π0.5 rollout for the OpenARM simulator."""

from __future__ import annotations

from copy import deepcopy
import math
import os
from threading import Lock, Thread
import time

import numpy as np

from XPolicyLab.policy.LeRobot_Pi05_OpenArm.protocol import (
    ACTION_QUEUE_SIZE,
    INTERPOLATION_MULTIPLIER,
    POLICY_FPS,
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

    def merge(self, processed, original, real_delay: int) -> None:
        processed = finite_action_chunk(processed)
        original = finite_action_chunk(original)
        delay = max(0, min(int(real_delay), ACTION_QUEUE_SIZE))
        with self._lock:
            self._processed = processed[delay:].copy()
            self._original = original[delay:].copy()
            self._index = 0


class _InferenceRequest:
    def __init__(self, model_client, observation: dict, previous_absolute, inference_delay: int):
        request_observation = deepcopy(observation)
        request_observation["_rtc"] = {
            "inference_delay": int(inference_delay),
            "previous_absolute_actions": (
                None if previous_absolute is None else np.asarray(previous_absolute).tolist()
            ),
        }
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


def eval_one_episode(TASK_ENV, model_client):
    smoke_steps = os.environ.get("ROBODOJO_OPENARM_SMOKE_STEPS")
    if smoke_steps is not None:
        if os.environ.get("ROBODOJO_OPENARM_ZERO_ACTION") != "1":
            raise ValueError("ROBODOJO_OPENARM_SMOKE_STEPS is restricted to zero-action smoke mode")
        TASK_ENV.step_lim = int(smoke_steps)
        if TASK_ENV.step_lim < max((0, 10, 30)):
            raise ValueError("visual smoke must include reference frame 30")
    model_client.call(func_name="reset")
    queue = _ActionQueue()
    latencies: list[float] = []
    previous_policy_action = None
    outer_deadline = time.monotonic() + 2000.0
    current_observation = TASK_ENV.get_obs()
    request = _InferenceRequest(model_client, current_observation, None, inference_delay=0)

    # There is no prior chunk to execute during cold start.
    _merge_finished_request(request, queue, latencies)
    request = None

    while not TASK_ENV.is_episode_end() and time.monotonic() < outer_deadline:
        if request is not None and request.done:
            _merge_finished_request(request, queue, latencies)
            request = None

        if request is None and queue.qsize() <= ACTION_QUEUE_SIZE:
            historical_delay = math.ceil((max(latencies) if latencies else 0.0) * POLICY_FPS)
            request = _InferenceRequest(
                model_client,
                current_observation,
                queue.processed_leftover(),
                inference_delay=historical_delay,
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
            current_degrees = clamp_relative_target(target, current_degrees)
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

        elapsed = time.perf_counter() - step_started
        time.sleep(max(0.0, 1.0 / POLICY_FPS - elapsed))

    if request is not None:
        request.wait()


def eval_one_episode_batch(TASK_ENV, model_client):
    if getattr(TASK_ENV, "num_envs", 1) != 1:
        raise NotImplementedError("protocol-faithful RTC evaluation currently requires eval_batch=false")
    return eval_one_episode(TASK_ENV, model_client)
