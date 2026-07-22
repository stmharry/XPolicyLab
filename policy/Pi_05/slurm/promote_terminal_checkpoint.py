from __future__ import annotations

import argparse
from pathlib import Path


def promote(checkpoint_root: Path, *, source_step: int, target_step: int) -> Path:
    checkpoint_root = checkpoint_root.resolve()
    source = checkpoint_root / str(source_step)
    target = checkpoint_root / str(target_step)
    if target.exists():
        for item in ("params", "assets"):
            if not (target / item).exists():
                raise FileNotFoundError(f"Terminal checkpoint is incomplete: {target / item}")
        return target
    if not source.is_dir():
        raise FileNotFoundError(f"Source terminal checkpoint does not exist: {source}")
    for item in ("params", "assets"):
        if not (source / item).exists():
            raise FileNotFoundError(f"Source terminal checkpoint is incomplete: {source / item}")
    source.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--source-step", type=int, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    args = parser.parse_args()
    print(promote(args.checkpoint_root, source_step=args.source_step, target_step=args.target_step))


if __name__ == "__main__":
    main()
