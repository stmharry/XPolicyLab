from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path

from validate_tensorboard import validate as validate_tensorboard


def finalize(args: argparse.Namespace) -> Path:
    manifest = json.loads(args.manifest.read_text())
    pipeline_jobs = json.loads(args.pipeline_jobs.read_text())
    offline_validation = json.loads(args.offline_validation.read_text())
    checkpoint = args.checkpoint_root / str(args.checkpoint_step)
    for item in ("params", "assets"):
        if not (checkpoint / item).exists():
            raise FileNotFoundError(f"Required final checkpoint is incomplete: {checkpoint / item}")
    tags = validate_tensorboard(args.tensorboard_dir)
    manifest.update(
        {
            "phase": "finalized",
            "finalized_at": datetime.now(UTC).isoformat(),
            "pipeline_jobs": pipeline_jobs,
            "finalizer": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
            },
            "required_checkpoint": args.checkpoint_step,
            "tensorboard_tags": tags,
            "offline_validation": offline_validation,
        }
    )
    temporary = args.manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.manifest)
    return args.manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pipeline-jobs", type=Path, required=True)
    parser.add_argument("--offline-validation", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=30_000)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(finalize(parse_args()))
