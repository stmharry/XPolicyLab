from __future__ import annotations

import unittest

import numpy as np

from XPolicyLab.policy.Pi_05 import contract, deploy


class _SingleEnv:
    def __init__(self, step_limit: int = 16):
        self.step_limit = step_limit
        self.actions: list[np.ndarray] = []

    def is_episode_end(self):
        return len(self.actions) >= self.step_limit

    def get_obs(self):
        return {"step": len(self.actions)}

    def take_action(self, action):
        self.actions.append(np.asarray(action))


class _BatchEnv(_SingleEnv):
    def get_running_env_idx_list(self):
        return [] if self.is_episode_end() else [0]

    def get_obs_batch(self, env_idx_list):
        return [{"env_idx": env_idx, "step": len(self.actions)} for env_idx in env_idx_list]

    def take_action_batch(self, actions, env_idx_list):
        self.actions.extend(np.asarray(action) for action, _ in zip(actions, env_idx_list, strict=True))


class _Client:
    def __init__(self, env):
        self.env = env
        self.inference_steps: list[int] = []
        self.chunk = self._make_chunk()

    @staticmethod
    def _make_chunk():
        cfg = contract.apply_checkpoint_profile({"ckpt_name": contract.YAM_PROFILE_NAME})
        raw = np.zeros((contract.YAM_ACTION_HORIZON, contract.ACTION_DIM), dtype=np.float32)
        return contract.checkpoint_actions_to_robodojo(cfg, raw)

    def call(self, func_name=None, obs=None, **kwargs):
        if func_name in {"reset", "update_obs", "update_obs_batch"}:
            return None
        if func_name == "get_action":
            self.inference_steps.append(len(self.env.actions))
            return self.chunk.copy()
        if func_name == "get_action_batch":
            self.inference_steps.append(len(self.env.actions))
            return [self.chunk.copy()]
        raise AssertionError(f"unexpected call: {func_name}")


class Pi05DeployCadenceTest(unittest.TestCase):
    def test_single_env_reinfers_after_eight_actions(self):
        env = _SingleEnv()
        client = _Client(env)

        deploy.eval_one_episode(env, client)

        self.assertEqual(client.chunk.shape, (8, 14))
        self.assertEqual(client.inference_steps, [0, 8])
        self.assertEqual(len(env.actions), 16)

    def test_batch_env_reinfers_after_eight_actions(self):
        env = _BatchEnv()
        client = _Client(env)

        deploy.eval_one_episode_batch(env, client)

        self.assertEqual(client.chunk.shape, (8, 14))
        self.assertEqual(client.inference_steps, [0, 8])
        self.assertEqual(len(env.actions), 16)


if __name__ == "__main__":
    unittest.main()
