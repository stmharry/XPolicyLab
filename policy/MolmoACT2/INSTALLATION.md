# MolmoACT2 Installation

`install.sh` creates separate environments for the original-HF public baseline and LeRobot training/local checkpoints.

## 1. Environment Overview

| Environment | Directory | Purpose |
| --- | --- | --- |
| Original-HF XPolicyLab inference | `molmoact2/.venv` | `molmoact2_bimanual_yam` and upstream server examples. |
| LeRobot training/local evaluation | `molmoact2/lerobot/.venv` | `lerobot_train` and existing local checkpoints. |

The upstream source is pinned to `c2282820f9b188b60e66ea1636b3efd81c45cbb4` and is not tracked in XPolicyLab.

## 2. One-command Install

```bash
cd XPolicyLab/policy/MolmoACT2
bash install.sh          # LeRobot training/eval env + XPolicyLab
bash install.sh all      # LeRobot plus original-HF inference environments
bash install.sh infer    # Original-HF environment plus XPolicyLab
bash prepare_checkpoint.sh  # Pinned Bimanual YAM checkpoint and checksum verification
```

## 3. Manual XPolicyLab Environment

```bash
cd XPolicyLab/policy/MolmoACT2/molmoact2/lerobot
UV_LINK_MODE=copy uv pip install -e ".[molmoact2,training,scipy-dep]" --index-strategy unsafe-best-match
source .venv/bin/activate
cd ../../..
pip install -e .
pip install h5py opencv-python
```

## 4. Optional Upstream FastAPI Environment

Used by the public Bimanual YAM alias and official upstream server workflows:

```bash
cd XPolicyLab/policy/MolmoACT2/molmoact2
uv sync
export HF_HUB_ENABLE_HF_TRANSFER=1
bash ../prepare_checkpoint.sh
```

## 5. Useful Variables

| Variable | Meaning |
| --- | --- |
| `MOLMOACT2_CHECKPOINT_PATH` | Training starting checkpoint; defaults to HF `allenai/MolmoAct2`. |
| `MOLMOACT2_DATASET_ROOT` | LeRobot v3.0 dataset root. |
| `MOLMOACT2_DATASET_REPO_ID` | LeRobot dataset repo id. |
| `MOLMOACT2_OUTPUT_ROOT` | Training output root. |
| `MOLMOACT2_LOCAL_CACHE_ROOT` | Local HF datasets cache for multi-host training. |
| `SKIP_XPOLICYLAB=1` | Skip XPolicyLab install during `install.sh`. |
| `ROBODOJO_STORAGE_ROOT` | Overrides the `.robodojo` runtime storage root. |
| `DEPLOY_PROXY_URL` | Opt-in HTTP/HTTPS proxy for policy server startup. |
| `MOLMOACT2_USE_CU128` | `auto` installs the tested cu128 PyTorch pair on Blackwell; set `0` or `1` to override detection. |
| `MOLMOACT2_TORCH_VERSION` | cu128 override version, default `2.10.0`. |

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `get_policy_class('molmoact2')` fails | Local LeRobot checkpoints require `molmoact2/lerobot/.venv`; the public alias uses `molmoact2/.venv`. |
| `import XPolicyLab` fails | Re-run the matching `install.sh infer`, `train`, or `all` mode. |
| transformers version conflict | Keep original-HF inference in `molmoact2/.venv` and LeRobot work in `lerobot/.venv`. |
| `torchcodec` version conflict | Use `--index-strategy unsafe-best-match` during install. |
| `no kernel image ... sm_120` | Re-run `MOLMOACT2_USE_CU128=1 bash install.sh infer`; do not run a later bare `uv sync`, which restores upstream cu121 PyTorch. |
| Slow multi-host dataloading | Put `MOLMOACT2_LOCAL_CACHE_ROOT` on local disk. |
