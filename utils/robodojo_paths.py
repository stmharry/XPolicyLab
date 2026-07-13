"""RoboDojo checkout and runtime-storage path discovery for policy adapters."""

from __future__ import annotations

import os
from pathlib import Path

_XPOLICYLAB_ROOT = Path(__file__).resolve().parents[1]


def find_robodojo_root() -> Path:
    configured = os.environ.get("ROBODOJO_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "configs" / "environment").is_dir():
            raise FileNotFoundError(f"ROBODOJO_ROOT does not contain configs/environment: {root}")
        return root

    for candidate in (_XPOLICYLAB_ROOT, *_XPOLICYLAB_ROOT.parents):
        if (candidate / "configs" / "environment").is_dir() and (candidate / "XPolicyLab").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the RoboDojo checkout. Set ROBODOJO_ROOT explicitly.")


def storage_root() -> Path:
    configured = os.environ.get("ROBODOJO_STORAGE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return find_robodojo_root() / ".robodojo"


def model_weight_root(policy_name: str, profile: str, revision: str) -> Path:
    return storage_root() / "model_weights" / policy_name / profile / revision
