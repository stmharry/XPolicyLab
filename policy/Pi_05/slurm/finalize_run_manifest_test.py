from __future__ import annotations

import argparse
import json
from pathlib import Path

import finalize_run_manifest


def test_finalize_combines_pipeline_checkpoint_tensorboard_and_offline_gate(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    pipeline_path = tmp_path / "pipeline.json"
    offline_path = tmp_path / "offline.json"
    checkpoint_root = tmp_path / "checkpoints"
    (checkpoint_root / "30000" / "params").mkdir(parents=True)
    (checkpoint_root / "30000" / "assets").mkdir()
    manifest_path.write_text(json.dumps({"source": {"xpolicylab_commit": "x", "robodojo_commit": "r"}}))
    pipeline_path.write_text(json.dumps({"training_job_id": "1", "finalizer_job_id": "2"}))
    offline_path.write_text(json.dumps({"model_action_shape": [50, 32], "physical_action_shape": [50, 14]}))
    monkeypatch.setattr(finalize_run_manifest, "validate_tensorboard", lambda _: {"scalars": ["loss"]})
    monkeypatch.setenv("SLURM_JOB_ID", "2")
    monkeypatch.setenv("SLURMD_NODENAME", "gpu-node")
    args = argparse.Namespace(
        manifest=manifest_path,
        pipeline_jobs=pipeline_path,
        offline_validation=offline_path,
        checkpoint_root=checkpoint_root,
        checkpoint_step=30_000,
        tensorboard_dir=tmp_path / "tensorboard",
    )

    finalize_run_manifest.finalize(args)

    result = json.loads(manifest_path.read_text())
    assert result["phase"] == "finalized"
    assert result["pipeline_jobs"]["training_job_id"] == "1"
    assert result["finalizer"] == {"job_id": "2", "node": "gpu-node"}
    assert result["required_checkpoint"] == 30_000
    assert result["offline_validation"]["model_action_shape"] == [50, 32]
    assert result["tensorboard_tags"]["scalars"] == ["loss"]
