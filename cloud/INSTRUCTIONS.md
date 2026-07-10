# SATG Cloud Training Guide

Complete walkthrough for training Structure-Aware Trust Gating (SATG) on a cloud GPU instance. From empty VM to completed experiments.

---

## Table of Contents

1. [GPU Instance Guide](#1-gpu-instance-guide)
2. [Quick Start](#2-quick-start)
3. [Step-by-Step Setup](#3-step-by-step-setup)
4. [Phase 8 — Baseline Experiments](#4-phase-8--baseline-experiments)
5. [Phase 9 — SATG Main Experiments](#5-phase-9--satg-main-experiments)
6. [Phase 9 — Ablations A–G](#6-phase-9--ablations-a-g)
7. [Phase 9 — Extended Ablations (Grid Search)](#7-phase-9--extended-ablations-tau-grid)
8. [WandB Monitoring](#8-wandb-monitoring)
9. [Downloading Results](#9-downloading-results)
10. [Estimated Costs & Timings](#10-estimated-costs--timings)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. GPU Instance Guide

### Recommended Instances

| Instance Type | GPU | VRAM | Cloud Cost (approx) | Est. per 40k-iter run |
|--------------|-----|------|--------------------|----------------------|
| **g5.xlarge** (AWS) | A10G | 24 GiB | ~$1.01/hr | ~$1.50–2.50 |
| **a100-40gb** (Lambda/Static) | A100 40GB | 40 GiB | ~$1.50/hr | ~$2.20–3.70 |
| **a100-80gb** (Lambda/Static) | A100 80GB | 80 GiB | ~$1.90/hr | ~$2.80–4.60 |
| **h100-80gb** (Lambda/Static) | H100 80GB | 80 GiB | ~$3.50/hr | ~$5.20–8.60 |
| **RTX 6000 Ada** (AWS g6.12xlarge) | Ada Lovelace | 48 GiB | ~$2.50/hr | ~$3.70–6.20 |

> **Recommendation**: An A100 40GB (~$1.50/hr) is the sweet spot — enough VRAM, widely available, and cost-effective for 40k-iteration runs.

### Requirements

- **VRAM**: ≥16 GiB (batch_size=4 with 512×512 crops)
- **CUDA**: ≥12.4 (for PyTorch 2.5+)
- **Storage**: ≥200 GB (GTA5 ~60GB + Cityscapes ~13GB + heatmaps ~50GB + checkpoints)
- **CPU**: ≥8 cores (for data loading and heatmap precomputation)

### Cloud Providers

- **Lambda Labs** (`lambdalabs.com`): Best price/performance for A100/H100
- **AWS EC2** (g5, p4d, p5): Good availability, more expensive
- **RunPod** / **Vast.ai**: Cheapest spot instances, more setup
- **Google Cloud** (a2-highgpu): A100 instances, slightly higher cost

---

## 2. Quick Start

```bash
# 1. Upload code to instance
rsync -avz --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    /path/to/SATG user@instance:/home/user/SATG/

# 2. SSH into instance
ssh user@instance

# 3. Setup environment
cd ~/SATG
bash cloud/setup.sh

# 4. Prepare data (download + precompute)
bash cloud/prepare_data.sh

# 5. Run baselines (Phase 8)
bash cloud/run_phase8.sh

# 6. Run SATG experiments (Phase 9)
bash cloud/run_phase9.sh

# 7. Download results back
rsync -avz user@instance:/home/user/SATG/checkpoints/ /path/to/local/SATG/checkpoints/
rsync -avz user@instance:/home/user/SATG/cloud/logs/ /path/to/local/SATG/cloud/logs/
```

---

## 3. Step-by-Step Setup

### 3.1 Launch a GPU Instance

Using Lambda Labs CLI as an example:

```bash
# Launch instance (1× A100 80GB, 200GB disk)
lambda instance create \
    --type gpu_1x_a100_80gb_sxm \
    --region us-west-1 \
    --disk-size 200 \
    --ssh-key ~/.ssh/id_ed25519.pub
```

For AWS EC2:

```bash
# Using the AWS CLI with a Deep Learning AMI
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --instance-type g5.xlarge \
    --block-device-mappings DeviceName=/dev/sda1,Ebs={VolumeSize=200} \
    --key-name your-key
```

### 3.2 Upload Code

```bash
# From your local machine
cd /path/to/local/SATG
rsync -avz \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.coverage' \
    --exclude='data/' \
    --exclude='checkpoints/' \
    --exclude='*.log' \
    ./ user@instance:/home/user/SATG/
```

### 3.3 Run Environment Setup

```bash
ssh user@instance
cd ~/SATG
bash cloud/setup.sh
```

Expected output:

```
==========================================
  SATG Cloud GPU Environment Setup
==========================================

--- [1/4] Checking GPU ---
  nvidia-smi found:
  0, NVIDIA A100-SXM4-80GB, 550.54.15, 81920 MiB

--- [2/4] Installing Python dependencies ---
  ...

--- [3/4] Verifying PyTorch CUDA ---
  PyTorch version:  2.5.1
  CUDA available:   True
  CUDA version:     12.4
  GPU device:       NVIDIA A100-SXM4-80GB
  VRAM:             79.2 GiB

  >>> GPU READY <<<

--- [4/4] Environment Summary ---
  Work dir: /home/user/SATG
  Python:   Python 3.10.12

  Environment ready — proceed to data prep
```

If you see `>>> GPU READY <<<` you're good to proceed.

### 3.4 Prepare Data

```bash
# One-command data prep (skips Cityscapes if not downloaded yet)
bash cloud/prepare_data.sh

# Or with custom paths:
bash cloud/prepare_data.sh --data_root /mnt/data
```

This will:

1. **Download GTA5** (automated, ~57GB images + ~700MB labels)
   - Downloads 10 ZIP parts from TU Darmstadt
   - Uses resume-safe download (can be interrupted and restarted)
   - Total: 24,966 images + 24,966 label maps
   - **Takes 2–6 hours** depending on bandwidth

2. **Preprocess GTA5 labels** (maps GTA5 IDs → Cityscapes trainIds)

3. **Prompt for Cityscapes download** (manual — requires registration)

4. **Precompute heatmaps** (if Cityscapes is present)
   - 2,975 training images × 4 components = 11,900 `.npy` files
   - **Takes ~10–20 minutes** with 8 workers

After downloading Cityscapes manually, resume data prep:

```bash
# If Cityscapes is now in place, just run the heatmap step
python -m precompute.compute_heatmaps \
    --data_root "$(pwd)/data/cityscapes" \
    --num_workers 8 \
    --resume \
    --verify
```

### 3.5 Start a Screen/Tmux Session

Training runs for hours. Use `tmux` or `screen` to keep processes alive:

```bash
tmux new -s satg-training

# Inside tmux:
cd ~/SATG
bash cloud/run_phase8.sh

# Detach with Ctrl+B, D
# Reattach with: tmux attach -t satg-training
```

---

## 4. Phase 8 — Baseline Experiments

### Overview

6 runs establishing baselines:

| # | Experiment | Config | Seeds | Est. Time (A100) |
|---|-----------|--------|-------|-----------------|
| 1–3 | Source Only | `source_only.yaml` | 42, 1337, 2024 | ~30–40 min each |
| 4–6 | Mean Teacher | `baseline_mean_teacher.yaml` | 42, 1337, 2024 | ~60–90 min each |

> **Source Only** (`lambda_target=0.0, skip_heatmap=true`): Trains only on GTA5 source data with no target adaptation. Expected mIoU: ~30%.
>
> **Mean Teacher** (`trust_gate.type=hard, tau_conf=0.90, tau_struct=1.01, skip_heatmap=true`): Standard self-training with confidence thresholding but no structural prior (tau_struct=1.01 gates nothing). Expected mIoU: ~35–40%.

### Run Commands

```bash
# Launch all 6 baselines (auto-logged to cloud/logs/phase8_*.log)
bash cloud/run_phase8.sh

# Or run individually:
# Source Only
python -m training.trainer --config configs/source_only.yaml seed=42
python -m training.trainer --config configs/source_only.yaml seed=1337
python -m training.trainer --config configs/source_only.yaml seed=2024

# Mean Teacher
python -m training.trainer --config configs/baseline_mean_teacher.yaml seed=42
python -m training.trainer --config configs/baseline_mean_teacher.yaml seed=1337
python -m training.trainer --config configs/baseline_mean_teacher.yaml seed=2024
```

---

## 5. Phase 9 — SATG Main Experiments

### Overview

| # | Experiment | Config | Seeds | Est. Time (A100) |
|---|-----------|--------|-------|-----------------|
| 7–9 | SATG Hard | `satg_hard.yaml` | 42, 1337, 2024 | ~60–90 min each |
| 10–12 | SATG Soft-Weight | `satg_soft_weight.yaml` | 42, 1337, 2024 | ~60–90 min each |
| 13–17 | SATG Soft-Label | `satg_soft_label.yaml` | 42, 1337, 2024, 7, 99 | ~60–90 min each |

> **SATG Hard** (`trust_gate.type=hard`): Binary trust mask — tau_conf=0.90, tau_struct=0.60.
>
> **SATG Soft-Weight** (`trust_gate.type=soft_weight`): Continuous trust weights via sigmoid gating.
>
> **SATG Soft-Label** (`trust_gate.type=soft_label`): **Primary contribution** — temperature-scaled soft pseudo-labels. Uses 5 seeds for statistical power (§1.8 of constitution).

### Run Commands

```bash
# Launch all SATG experiments (auto-logged)
bash cloud/run_phase9.sh

# Or run individually:
# SATG Hard (3 seeds)
python -m training.trainer --config configs/satg_hard.yaml seed=42
python -m training.trainer --config configs/satg_hard.yaml seed=1337
python -m training.trainer --config configs/satg_hard.yaml seed=2024

# SATG Soft-Weight (3 seeds)
python -m training.trainer --config configs/satg_soft_weight.yaml seed=42
python -m training.trainer --config configs/satg_soft_weight.yaml seed=1337
python -m training.trainer --config configs/satg_soft_weight.yaml seed=2024

# SATG Soft-Label (5 seeds)
python -m training.trainer --config configs/satg_soft_label.yaml seed=42
python -m training.trainer --config configs/satg_soft_label.yaml seed=1337
python -m training.trainer --config configs/satg_soft_label.yaml seed=2024
python -m training.trainer --config configs/satg_soft_label.yaml seed=7
python -m training.trainer --config configs/satg_soft_label.yaml seed=99
```

---

## 6. Phase 9 — Ablations A–G

### Ablation A — Prior Component Contribution

Isolates each structural prior component. All use seed=42.

```bash
# Edge-only: only edge density contributes
python -m training.trainer --config configs/satg_hard.yaml \
    structural_prior.edge_weight=1.0 \
    structural_prior.variance_weight=0.0 \
    structural_prior.entropy_weight=0.0 \
    structural_prior.cornerness_weight=0.0 \
    seed=42

# Variance-only: only local variance contributes
python -m training.trainer --config configs/satg_hard.yaml \
    structural_prior.edge_weight=0.0 \
    structural_prior.variance_weight=1.0 \
    structural_prior.entropy_weight=0.0 \
    structural_prior.cornerness_weight=0.0 \
    seed=42

# Entropy-only: only local entropy contributes
python -m training.trainer --config configs/satg_hard.yaml \
    structural_prior.edge_weight=0.0 \
    structural_prior.variance_weight=0.0 \
    structural_prior.entropy_weight=1.0 \
    structural_prior.cornerness_weight=0.0 \
    seed=42

# Cornerness-only: only corner response contributes
python -m training.trainer --config configs/satg_hard.yaml \
    structural_prior.edge_weight=0.0 \
    structural_prior.variance_weight=0.0 \
    structural_prior.entropy_weight=0.0 \
    structural_prior.cornerness_weight=1.0 \
    seed=42
```

### Ablation B — Confidence Threshold (τ_conf) Sweep

Varies `tau_conf` with `tau_struct=0.60` fixed. All use seed=42.

```bash
# tau_conf = 0.80
python -m training.trainer --config configs/satg_hard.yaml \
    trust_gate.tau_conf=0.80 seed=42

# tau_conf = 0.90 (same as default satg_hard — already run above)
# python -m training.trainer --config configs/satg_hard.yaml seed=42

# tau_conf = 0.95
python -m training.trainer --config configs/satg_hard.yaml \
    trust_gate.tau_conf=0.95 seed=42
```

### Ablation C — Structure Threshold (τ_struct) Sweep

Varies `tau_struct` with `tau_conf=0.90` fixed. All use seed=42.

```bash
# tau_struct = 0.40
python -m training.trainer --config configs/satg_hard.yaml \
    trust_gate.tau_struct=0.40 seed=42

# tau_struct = 0.60 (same as default satg_hard — already run above)
# python -m training.trainer --config configs/satg_hard.yaml seed=42

# tau_struct = 0.70
python -m training.trainer --config configs/satg_hard.yaml \
    trust_gate.tau_struct=0.70 seed=42
```

### Ablation D — Hard vs Soft Comparison

No new runs — compare best-performing hard gate (Ablation B+C results) against best soft-gate (main results). See results in EXPERIMENTS.md.

### Ablation E — Kernel Size & Sigma

Tests sensitivity to structural prior kernel parameters. All use seed=42.

```bash
# Sigma = 0.5, Kernel = 7
python -m training.trainer --config configs/satg_hard.yaml \
    structural_prior.gaussian_sigma=0.5 \
    structural_prior.cornerness_sigma=0.5 \
    structural_prior.edge_kernel_size=7 \
    structural_prior.variance_kernel_size=7 \
    structural_prior.cornerness_kernel_size=7 \
    seed=42

# Sigma = 1.0, Kernel = 7
python -m training.trainer --config configs/satg_hard.yaml \
    structural_prior.gaussian_sigma=1.0 \
    structural_prior.cornerness_sigma=1.0 \
    structural_prior.edge_kernel_size=7 \
    structural_prior.variance_kernel_size=7 \
    structural_prior.cornerness_kernel_size=7 \
    seed=42

# Sigma = 2.0, Kernel = 7
python -m training.trainer --config configs/satg_hard.yaml \
    structural_prior.gaussian_sigma=2.0 \
    structural_prior.cornerness_sigma=2.0 \
    structural_prior.edge_kernel_size=7 \
    structural_prior.variance_kernel_size=7 \
    structural_prior.cornerness_kernel_size=7 \
    seed=42

# Sigma = 0.5, Kernel = 15
python -m training.trainer --config configs/satg_hard.yaml \
    structural_prior.gaussian_sigma=0.5 \
    structural_prior.cornerness_sigma=0.5 \
    seed=42

# Sigma = 1.0, Kernel = 15
python -m training.trainer --config configs/satg_hard.yaml \
    structural_prior.gaussian_sigma=1.0 \
    structural_prior.cornerness_sigma=1.0 \
    seed=42

# Sigma = 2.0, Kernel = 15 (default — already run as SATG Hard)
# python -m training.trainer --config configs/satg_hard.yaml seed=42

# Sigma = 0.5, Kernel = 31
python -m training.trainer --config configs/satg_hard.yaml \
    structural_prior.gaussian_sigma=0.5 \
    structural_prior.cornerness_sigma=0.5 \
    structural_prior.edge_kernel_size=31 \
    structural_prior.variance_kernel_size=31 \
    structural_prior.cornerness_kernel_size=31 \
    seed=42

# Sigma = 1.0, Kernel = 31
python -m training.trainer --config configs/satg_hard.yaml \
    structural_prior.gaussian_sigma=1.0 \
    structural_prior.cornerness_sigma=1.0 \
    structural_prior.edge_kernel_size=31 \
    structural_prior.variance_kernel_size=31 \
    structural_prior.cornerness_kernel_size=31 \
    seed=42

# Sigma = 2.0, Kernel = 31
python -m training.trainer --config configs/satg_hard.yaml \
    structural_prior.gaussian_sigma=2.0 \
    structural_prior.cornerness_sigma=2.0 \
    structural_prior.edge_kernel_size=31 \
    structural_prior.variance_kernel_size=31 \
    structural_prior.cornerness_kernel_size=31 \
    seed=42
```

### Ablation F — Soft Mechanism Comparison

No new runs — compare SATG Hard, SATG Soft-Weight, and SATG Soft-Label side-by-side. See EXPERIMENTS.md.

### Ablation G — Soft-Label k Sensitivity

Varies the temperature scaling constant `k` (default: 4.0). All use seed=42.

```bash
# k = 2.0 (flatter temperature curve)
python -m training.trainer --config configs/satg_soft_label.yaml \
    trust_gate.soft_label_k=2.0 seed=42

# k = 6.0 (steeper temperature curve)
python -m training.trainer --config configs/satg_soft_label.yaml \
    trust_gate.soft_label_k=6.0 seed=42

# k = 4.0 is the default — already run as SATG Soft-Label (seed=42)
```

---

## 7. Phase 9 — Extended Ablations (Tau Grid)

Combined τ_conf × τ_struct grid search to detect interaction effects (9 runs).

```bash
# tau_conf=0.85 × tau_struct=0.50
python -m training.trainer --config configs/satg_hard.yaml \
    trust_gate.tau_conf=0.85 trust_gate.tau_struct=0.50 seed=42

# tau_conf=0.85 × tau_struct=0.60
python -m training.trainer --config configs/satg_hard.yaml \
    trust_gate.tau_conf=0.85 trust_gate.tau_struct=0.60 seed=42

# tau_conf=0.85 × tau_struct=0.70
python -m training.trainer --config configs/satg_hard.yaml \
    trust_gate.tau_conf=0.85 trust_gate.tau_struct=0.70 seed=42

# tau_conf=0.90 × tau_struct=0.50
python -m training.trainer --config configs/satg_hard.yaml \
    trust_gate.tau_conf=0.90 trust_gate.tau_struct=0.50 seed=42

# tau_conf=0.90 × tau_struct=0.60 (same as default satg_hard)
# python -m training.trainer --config configs/satg_hard.yaml seed=42

# tau_conf=0.90 × tau_struct=0.70
python -m training.trainer --config configs/satg_hard.yaml \
    trust_gate.tau_conf=0.90 trust_gate.tau_struct=0.70 seed=42

# tau_conf=0.95 × tau_struct=0.50
python -m training.trainer --config configs/satg_hard.yaml \
    trust_gate.tau_conf=0.95 trust_gate.tau_struct=0.50 seed=42

# tau_conf=0.95 × tau_struct=0.60
python -m training.trainer --config configs/satg_hard.yaml \
    trust_gate.tau_conf=0.95 trust_gate.tau_struct=0.60 seed=42

# tau_conf=0.95 × tau_struct=0.70
python -m training.trainer --config configs/satg_hard.yaml \
    trust_gate.tau_conf=0.95 trust_gate.tau_struct=0.70 seed=42
```

---

## 8. WandB Monitoring

All training runs log metrics to **Weights & Biases** (wandb) project `satg-uda`.

### Setup

```bash
# Login (one-time)
wandb login

# Your API key is at: https://wandb.ai/authorize
```

### Dashboard

Each run appears with the name `{config_stem}_seed{seed}` (e.g. `satg_hard_seed42`).

**Key metrics tracked per iteration:**
- `total_loss` — combined source + target loss
- `source_loss` — supervised loss on GTA5 source
- `target_loss` — unsupervised loss on Cityscapes target
- `trust_coverage_ratio` — fraction of target pixels trusted (hard gate) or mean weight (soft gate)
- `mean_temperature` — average temperature in trusted regions (soft-label only)
- `val/miou` — Cityscapes validation mIoU (logged every `eval_interval` iterations)

### Comparing Runs

In the WandB UI:
1. Open the `satg-uda` project
2. Check runs to compare (e.g. all `source_only_*` vs `mean_teacher_*`)
3. Use the "Parallel Coordinates" plot for ablation sweeps

---

## 9. Downloading Results

### Checkpoints

Each run saves checkpoints to `checkpoints/{config_stem}_seed{seed}/`:

```bash
# Download from instance to local
rsync -avz user@instance:/home/user/SATG/checkpoints/ /path/to/local/SATG/checkpoints/
```

Checkpoint structure:
```
checkpoints/
  source_only_seed42/
    best.pth       # Best validation mIoU
    last.pth       # Last iteration
    config.yaml    # Full config snapshot
    metrics.csv    # Per-iteration metrics
  mean_teacher_seed42/
    ...
  satg_hard_seed42/
    ...
```

### Logs

Experiment launchers save stdout/stderr to `cloud/logs/`:

```bash
rsync -avz user@instance:/home/user/SATG/cloud/logs/ /path/to/local/SATG/cloud/logs/
```

---

## 10. Estimated Costs & Timings

### Per-Run Time (A100 40GB)

| Configuration | Iterations | Est. Time | Est. Cost ($1.50/hr) |
|--------------|-----------|-----------|---------------------|
| Source Only | 40,000 | ~30–40 min | ~$0.75–1.00 |
| Mean Teacher | 40,000 | ~60–90 min | ~$1.50–2.25 |
| SATG Hard | 40,000 | ~60–90 min | ~$1.50–2.25 |
| SATG Soft-Weight | 40,000 | ~60–90 min | ~$1.50–2.25 |
| SATG Soft-Label | 40,000 | ~60–90 min | ~$1.50–2.25 |

### Total Estimated Costs

| Phase | Runs | Est. Time | Est. Cost (A100) |
|------|------|-----------|-----------------|
| Phase 8 (Baselines) | 6 | ~5–9 hr | ~$8–14 |
| Phase 9 Main | 11 | ~11–17 hr | ~$17–26 |
| Phase 9 Ablations A–G | ~17 | ~17–26 hr | ~$26–40 |
| Phase 9 Grid Search | ~8 | ~8–12 hr | ~$12–18 |
| **Total** | **~42** | **~41–64 hr** | **~$63–96** |

> Actual costs vary by GPU type, cloud provider, and whether you use spot/preemptible instances. H100 instances cost ~2× but complete runs ~30% faster.

---

## 11. Troubleshooting

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate ... MiB
```

**Fixes (in order of likelihood):**
1. Reduce `batch_size` override: `training.batch_size=2`
2. Disable `cudnn_benchmark`: `training.cudnn_benchmark=false`
3. Reduce crop size: `training.crop_size="[384,384]"`
4. Reduce `num_workers`: `training.num_workers=2`

```bash
# Example with reduced memory footprint
python -m training.trainer --config configs/satg_hard.yaml \
    training.batch_size=2 training.crop_size="[384,384]" seed=42
```

### Dataset Not Found

```
FileNotFoundError: data/cityscapes/leftImg8bit/train not found
```

- Check paths: `ls data/cityscapes/leftImg8bit/train/`
- Script expects Cityscapes structure: `data/cityscapes/leftImg8bit/train/{city}/*.png`
- If data lives elsewhere, use the `--data_root` argument with `prepare_data.sh`, or symlink:
  ```bash
  ln -s /actual/path/to/cityscapes data/cityscapes
  ```

### GTA5 Downloads Failing

```
curl: (56) Recv failure: Connection reset by peer
```

- The download server can be flaky — the script auto-retries
- If it keeps failing, download manually:
  ```bash
  wget https://download.visinf.tu-darmstadt.de/data/from_games/data/01_images.zip
  ```
  Then extract into `data/GTA5/images/`

### Heatmap Count Mismatch

```
WARNING: Heatmap count mismatch. Expected N, found M.
```

- Run the precompute with `--resume` to fill in missing files:
  ```bash
  python -m precompute.compute_heatmaps \
      --data_root data/cityscapes --num_workers 8 --resume
  ```
- Cityscapes should have exactly 2,975 training images (not including val/test)
- Each image produces 4 `.npy` files (edge, var, ent, corn)

### WandB Not Logging

- Make sure `wandb login` was run
- Check `logging.backend=wandb` and `logging.project=satg-uda` in config
- Offline mode: `wandb offline` and sync later: `wandb sync`

### Process Dies After SSH Disconnect

Use `tmux` or `screen`:

```bash
tmux new -s satg
./cloud/run_phase8.sh
# Ctrl+B, D to detach
# tmux attach -t satg to reattach
```

Or use `nohup`:

```bash
nohup ./cloud/run_phase8.sh > cloud/logs/phase8_full.log 2>&1 &
```

### Config Override Syntax

Overrides use OmegaConf dot-list syntax:

```bash
# Override nested keys with dots:
python -m training.trainer --config configs/satg_hard.yaml \
    training.iterations=10000 \
    training.eval_interval=500 \
    trust_gate.tau_conf=0.95 \
    seed=42
```

List values need quotes:

```bash
python -m training.trainer --config configs/default.yaml \
    training.crop_size="[384,384]"
```

---

## Appendix: Quick Reference

```bash
# === Setup ===
bash cloud/setup.sh                                  # Check GPU + install deps

# === Data ===
bash cloud/prepare_data.sh                           # Full data prep
python -m precompute.compute_heatmaps --data_root data/cityscapes --num_workers 8 --resume

# === Phase 8 (6 runs) ===
bash cloud/run_phase8.sh

# === Phase 9 (~28 runs, excludes grid) ===
bash cloud/run_phase9.sh

# === Single run ===
python -m training.trainer --config configs/satg_hard.yaml seed=42

# === Check results ===
tail -f cloud/logs/phase8_source_only_seed42.log     # Watch live output
grep "val mIoU" cloud/logs/*.log                     # Scan all results
```
