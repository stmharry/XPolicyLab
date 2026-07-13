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


def _shared_checkout_root(worktree_root: Path) -> Path:
    """Return the primary checkout that owns a linked worktree's common Git dir."""

    git_file = worktree_root / ".git"
    if not git_file.is_file():
        return worktree_root

    try:
        git_lines = git_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return worktree_root
    if not git_lines:
        return worktree_root
    first_line = git_lines[0].strip()
    prefix = "gitdir:"
    if not first_line.lower().startswith(prefix):
        return worktree_root

    git_dir = Path(first_line[len(prefix) :].strip()).expanduser()
    if not git_dir.is_absolute():
        git_dir = worktree_root / git_dir
    git_dir = git_dir.resolve()

    common_dir_file = git_dir / "commondir"
    if not common_dir_file.is_file():
        return worktree_root
    try:
        common_dir_text = common_dir_file.read_text(encoding="utf-8").strip()
    except OSError:
        return worktree_root
    if not common_dir_text:
        return worktree_root
    common_dir = Path(common_dir_text).expanduser()
    if not common_dir.is_absolute():
        common_dir = git_dir / common_dir
    primary_root = common_dir.resolve().parent
    if (primary_root / "configs" / "environment").is_dir() and (primary_root / "XPolicyLab").is_dir():
        return primary_root
    return worktree_root


def storage_root() -> Path:
    configured = os.environ.get("ROBODOJO_STORAGE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return _shared_checkout_root(find_robodojo_root()) / ".robodojo"


def model_weight_root(policy_name: str, profile: str, revision: str) -> Path:
    return storage_root() / "model_weights" / policy_name / profile / revision
