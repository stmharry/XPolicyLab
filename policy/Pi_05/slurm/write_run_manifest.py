from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def _latest_metrics(tensorboard_dir: Path) -> dict[str, dict[str, float | int]]:
    if not tensorboard_dir.exists():
        return {}
    events = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0})
    events.Reload()
    metrics = {}
    for tag in events.Tags()["scalars"]:
        values = events.Scalars(tag)
        if values:
            metrics[tag] = {"step": values[-1].step, "value": values[-1].value}
    return metrics


def write_manifest(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    existing: dict[str, Any] = json.loads(output.read_text()) if output.exists() else {}
    xpolicy_root = Path(__file__).resolve().parents[3]
    robodojo_root = xpolicy_root.parent
    dataset_manifest = json.loads(args.dataset_manifest.read_text())
    checkpoints = []
    if args.checkpoint_dir.exists():
        checkpoints = sorted(
            int(path.name) for path in args.checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit()
        )
    payload = {
        **existing,
        "phase": args.phase,
        "run_name": args.run_name,
        "slurm": {"job_id": args.job_id, "node": args.node, "cluster": os.environ.get("SLURM_CLUSTER_NAME")},
        "source": {
            "robodojo_commit": _command("git", "rev-parse", "HEAD", cwd=robodojo_root),
            "xpolicylab_commit": _command("git", "rev-parse", "HEAD", cwd=xpolicy_root),
        },
        "dataset": dataset_manifest,
        "training": {
            "config": args.config,
            "base_checkpoint": args.base_checkpoint,
            "batch_size": args.batch_size,
            "fsdp_devices": args.fsdp_devices,
            "num_train_steps": int(os.environ.get("OPENPI_NUM_TRAIN_STEPS", "30000")),
            "tensorboard_dir": str(args.tensorboard_dir),
        },
        "gpus": _command("nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader").splitlines(),
        "checkpoints": checkpoints,
        "latest_metrics": _latest_metrics(args.tensorboard_dir),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("starting", "complete"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--config", default="pi05_base_aloha_full_real_piper_seed_0")
    parser.add_argument("--base-checkpoint", default="gs://openpi-assets/checkpoints/pi05_base")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fsdp-devices", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    print(write_manifest(parse_args()))
