# Quickstart: Structure-Aware Trust Gating (SATG)

**Date**: 2026-06-25 | **Feature**: 001-structure-aware-trust-gating

## Prerequisites

- Python 3.10+
- CUDA 12.6+ (for GPU training)
- GTA5 dataset downloaded and preprocessed
- Cityscapes dataset with `gtFine_labelTrainIds.png` generated

## Installation

```bash
pip install -r requirements.txt
```

## Step 1: Precompute Structural Heatmaps

```bash
# Compute heatmaps for all Cityscapes training images
python -m precompute.compute_heatmaps \
    --data_root /data/cityscapes/leftImg8bit/train \
    --num_workers 8
```

**Expected output**: One `*_satg_heatmap.npy` file per image, plus console statistics (min/max/mean).

**Validation**: Check that heatmaps exist alongside images:
```bash
ls /data/cityscapes/leftImg8bit/train/*/*_satg_heatmap.npy | wc -l
# Should return 2975
```

## Step 2: Run Dry Run (10 images, 10 iterations)

```bash
# Verify pipeline works end-to-end
python -m training.trainer \
    training.iterations=10 \
    training.batch_size=1 \
    training.eval_interval=5
```

**Expected**: No errors; loss values logged; checkpoint saved.

## Step 3: Train Source Only Baseline

```bash
python -m training.trainer --config configs/source_only.yaml
```

**Expected**: mIoU ~30% on Cityscapes val (lower bound).

## Step 4: Train Standard Mean Teacher

```bash
python -m training.trainer --config configs/baseline_mean_teacher.yaml
```

**Expected**: mIoU ~35-40% on Cityscapes val.

## Step 5: Train SATG

```bash
# Hard rejection variant
python -m training.trainer --config configs/satg_hard.yaml

# Soft weighting variant
python -m training.trainer --config configs/satg_soft.yaml
```

**Expected**: mIoU improvement over Mean Teacher baseline.

## Step 6: Evaluate

```bash
python -m evaluation.evaluator \
    --checkpoint checkpoints/best.pth \
    --config configs/satg_hard.yaml
```

**Expected**: Per-class IoU table + overall mIoU printed.

## Step 7: Generate Visualizations

```bash
python -m visualization.visualize \
    --checkpoint checkpoints/best.pth \
    --config configs/satg_hard.yaml \
    --num_images 10
```

**Expected**: 1×5 panel PNG files in `visualizations/satg_hard/`.

## Step 8: Run Tests

```bash
pytest tests/ --cov=satg --cov=models --cov=data --cov=training --cov=evaluation \
    --cov-report=term-missing --cov-fail-under=80
```

**Expected**: All tests pass, coverage ≥80%.

## Validation Scenarios

| Scenario | Command | Expected Outcome |
|----------|---------|-----------------|
| Heatmap precompute | Step 1 | 2975 .npy files, all float32 |
| Dry run | Step 2 | 10 iterations, no errors |
| Source only | Step 3 | mIoU ~30% |
| Mean teacher | Step 4 | mIoU ~35-40% |
| SATG hard | Step 5a | mIoU > Mean Teacher |
| SATG soft | Step 5b | mIoU > Mean Teacher |
| Unit tests | Step 8 | ≥80% coverage, all pass |

## Key Config Overrides

```bash
# Change trust thresholds
python -m training.trainer --config configs/satg_hard.yaml \
    trust_gate.tau_conf=0.85 trust_gate.tau_struct=0.55

# Change learning rate
python -m training.trainer --config configs/satg_hard.yaml \
    training.lr=5e-4

# Change crop size
python -m training.trainer --config configs/satg_hard.yaml \
    training.crop_size=[384,384]
```
