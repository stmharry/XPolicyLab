"""Coordinate transform for YAM data released through the MolmoAct2 dataset."""

from __future__ import annotations

from typing import Any

import numpy as np

from XPolicyLab.utils.bimanual_yam_contract import STATE_DIM

JOINT_SIGN_INDICES = (4, 11)


def _transform(values: Any, *, value_name: str) -> np.ndarray:
    result = np.array(values, copy=True)
    if result.ndim < 1 or result.shape[-1] != STATE_DIM:
        raise ValueError(f"{value_name} must end in dimension {STATE_DIM}, got shape {result.shape}.")
    result[..., list(JOINT_SIGN_INDICES)] = -result[..., list(JOINT_SIGN_INDICES)]
    return result


def simulator_to_dataset(values: Any) -> np.ndarray:
    return _transform(values, value_name="YAM simulator values")


def dataset_to_simulator(values: Any) -> np.ndarray:
    return _transform(values, value_name="MolmoAct2 YAM dataset values")
