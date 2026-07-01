# Implementation Plan: Structure-Aware Trust Gating (SATG)

**Branch**: `001-structure-aware-trust-gating` | **Date**: 2026-06-25 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-structure-aware-trust-gating/spec.md`

## Summary

SATG is a UDA semantic segmentation method that augments standard teacher-student pseudo-labeling with an image-space structural prior. The structural prior (edge density + local variance) identifies structurally complex regions where the teacher is likely overconfident about wrong predictions. A trust gate modulates pseudo-label weights based on both teacher confidence and structural complexity, reducing confirmation bias in complex regions. The system uses DeepLabV3+ ResNet50 with EMA teacher updates, trained on GTA5→Cityscapes with 19-class Cityscapes label space.

## Technical Context

**Language/Version**: Python 3.10+, PyTorch 2.7.0 + CUDA 12.6/12.8

**Primary Dependencies**:
- PyTorch 2.7.0 + torchvision 0.22 (DeepLabV3+ ResNet50, pretrained backbone)
- OpenCV 4.x (`cv2.Canny`, `cv2.blur`, `cv2.filter2D` for structural prior)
- NumPy 1.24+ (heatmap .npy storage, computation)
- OmegaConf 2.3+ (YAML config management)
- WandB (experiment tracking, free tier: 5GB storage)
- pytest + pytest-cov (≥80% coverage threshold)
- tqdm (progress bars)
- black + flake8 (code formatting, max line length 100)

**Storage**: Filesystem only (precomputed .npy heatmaps, PyTorch .pth checkpoints)

**Testing**: pytest + pytest-cov, coverage ≥80% on satg/, models/, data/, training/, evaluation/

**Target Platform**: Linux server, single GPU with ≥16GB VRAM

**Project Type**: Research library (modular Python packages)

**Performance Goals**: Precompute 2,975 heatmaps in <20min (8 cores); training overhead <10% vs baseline

**Constraints**: ≤16GB VRAM; batch_size=1 (1 source + 1 target per GPU); 512×512 crops; no custom CUDA kernels

**Scale/Scope**: 24,966 source images (GTA5), 2,975 target training + 500 val (Cityscapes); 40k training iterations

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Evaluation Model**: Final mIoU is evaluated using the **student model** (not the EMA teacher). The teacher is used only for pseudo-label generation during training. All reported results use student model inference on the Cityscapes validation split.

| Principle | Status | Notes |
|-----------|--------|-------|
| §1.1 Seed Management | ✅ PASS | Fixed seeds {42, 1337, 2024} configurable via YAML |
| §1.2 Multi-Seed Evaluation | ✅ PASS | 3 seeds required, mean ± std |
| §1.3 Config Documentation | ✅ PASS | All hyperparams in YAML configs |
| §1.4 No Cherry-Picking | ✅ PASS | All results in EXPERIMENTS.md |
| §1.5 Structural Prior Validity | ✅ PASS | Classical CV ops only, no learned params |
| §1.6 Target Label Isolation | ✅ PASS | No target GT access during training |
| §1.7 Compute Documentation | ✅ PASS | GPU type/hours/memory in EXPERIMENTS.md |
| §1.8 Statistical Significance | ✅ PASS | <0.5 mIoU discussed as marginal |
| §1.9 Per-Class IoU Reporting | ✅ PASS | All 19 classes reported |
| §1.10 Trust Mask Coverage | ✅ PASS | Coverage ratio logged per batch |
| §2.1 Library-First | ✅ PASS | Each module independently importable |
| §2.2 Test-First | ✅ PASS | TDD red-green-refactor enforced |
| §2.3 Test Coverage | ✅ PASS | ≥80% line coverage enforced |
| §2.4 Type Annotations | ✅ PASS | Google-style docstrings, type hints |
| §2.5 Config-Driven | ✅ PASS | Zero hardcoded numeric constants |
| §2.6 Tensor Shape Comments | ✅ PASS | Shape annotations on non-obvious ops |
| §2.7 Linting/Formatting | ✅ PASS | black + flake8 enforced |
| §2.8 Swappable Backbones | ✅ PASS | Backbone configurable via YAML |
| §2.9 Heatmap Naming | ✅ PASS | `{stem}_satg_heatmap.npy` convention |
| §3.1 Mandatory Ablations | ✅ PASS | 9 ablation variants enumerated (A–G + grid + k-sensitivity) |
| §3.2 Source Only Baseline | ✅ PASS | Lower bound always included |
| §3.3 Dry Run Validation | ✅ PASS | 10 images, 10 iters before full runs |
| §4.1 README Completeness | ✅ PASS | Install + usage + CLI syntax |
| §4.2 Experiment Comparison Table | ✅ PASS | All methods in EXPERIMENTS.md |
| §4.3 Visualization Outputs | ✅ PASS | ≥10 images per config |
| §5.1 Research Delegation | ✅ PASS | Parallel subagents for research |
| §5.2 Implementation Delegation | ✅ PASS | Independent modules parallelizable |
| §5.3 Output Verification | ✅ PASS | All subagent outputs verified |

**Gate Result**: ✅ ALL PASS — No violations. Proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/001-structure-aware-trust-gating/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── module-api.md    # Module API contracts
└── tasks.md             # Phase 2 output (/speckit-tasks)
```

### Source Code (repository root)

```text
satg-project/
├── configs/
│   ├── default.yaml               # Master config with all defaults
│   ├── satg_hard.yaml             # SATG hard rejection (extends default)
│   ├── satg_soft_weight.yaml       # SATG soft-weight variant (extends default)
│   ├── satg_soft_label.yaml       # SATG soft-label variant (extends default)
│   ├── baseline_mean_teacher.yaml # Conf-threshold only (extends default)
│   └── source_only.yaml           # No target pseudo-labeling
├── data/
│   ├── __init__.py
│   ├── gta5_loader.py             # GTA5 Dataset class
│   ├── cityscapes_loader.py       # Cityscapes Dataset class (train + val)
│   └── label_mapping.py           # GTA5→Cityscapes 19-class mapping dict
├── models/
│   ├── __init__.py
│   ├── segmentation.py            # DeepLabV3+ ResNet50 wrapper (19 classes)
│   └── ema.py                     # EMA teacher model class
├── satg/
│   ├── __init__.py
│   ├── structural_prior.py        # StructuralPrior class
│   ├── trust_gate.py              # HardTrustGate + SoftTrustGate classes
│   ├── soft_label.py              # TemperatureSoftLabel class
│   └── losses.py                  # SATGLoss + SoftLabelKLLoss classes
├── precompute/
│   └── compute_heatmaps.py        # CLI script: offline heatmap precomputation
├── training/
│   ├── __init__.py
│   └── trainer.py                 # Main UDA training loop
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py               # mIoU + per-class IoU computation
├── visualization/
│   └── visualize.py               # 5-panel trust mask visualization
├── tests/
│   ├── conftest.py                # Shared fixtures
│   ├── test_structural_prior.py
│   ├── test_trust_gate.py
│   ├── test_soft_label.py
│   ├── test_losses.py
│   ├── test_ema.py
│   ├── test_segmentation.py
│   ├── test_data_loaders.py
│   ├── test_precompute.py
│   ├── test_trainer.py
│   ├── test_evaluator.py
│   ├── test_visualization.py
│   ├── test_data_leakage.py
│   └── test_configs.py
├── EXPERIMENTS.md                 # All results + configs
├── README.md                      # Installation + usage
├── requirements.txt               # Pinned dependencies
├── pyproject.toml                 # Project config (black, flake8, coverage)
└── .coveragerc                    # pytest-cov config (backup)
```

**Structure Decision**: Single project layout. All Python packages at repo root for flat imports (`from satg.structural_prior import StructuralPrior`). Separate `tests/` directory mirroring source layout.

## Complexity Tracking

> No constitution violations — no complexity tracking needed.

## Module Specifications

### satg/soft_label.py: TemperatureSoftLabel class

- `__init__(cfg: OmegaConf)`: reads `k` (default 4.0), `T_max` (default 5.0), `tau_conf` (shared with hard gate, for the pre-filter)
- `compute_temperature(struct: Tensor[B,H,W]) -> Tensor[B,H,W]`: T = 1.0 + k * struct, then clamp to [1.0, T_max]
- `compute_soft_targets(teacher_logits: Tensor[B,C,H,W], struct: Tensor[B,H,W]) -> Tensor[B,C,H,W]`:
  1. Compute per-pixel temperature T [B,H,W]
  2. Expand T to [B,1,H,W] and divide teacher_logits by T (broadcast across class dim)
  3. Apply softmax over class dim → soft_targets [B,C,H,W]
  4. Return soft_targets (each pixel's C-vector sums to 1.0)
- `compute_confidence_mask(teacher_logits: Tensor[B,C,H,W], tau_conf: float) -> Tensor[B,H,W]`: reuse the SAME confidence computation as the existing hard trust gate (max softmax prob > tau_conf), used as a binary pre-filter before the distributional loss is applied

### satg/losses.py: SoftLabelKLLoss class (nn.Module), alongside existing SATGLoss

- `__init__(ignore_index=255)`
- `forward(student_logits: Tensor[B,C,H,W], soft_targets: Tensor[B,C,H,W], confidence_mask: Tensor[B,H,W]) -> Tensor (scalar)`
- Computation:
  1. `student_log_probs = log_softmax(student_logits, dim=1)` → [B,C,H,W]
  2. `per_pixel_kl = sum(soft_targets * (log(soft_targets + eps) - student_log_probs), dim=1)` → [B,H,W]
  3. `masked_kl = per_pixel_kl * confidence_mask` → [B,H,W]
  4. If `confidence_mask.sum() == 0`: return `torch.tensor(0.0)`
  5. Else: return `masked_kl.sum() / confidence_mask.sum()`
- Safety: eps=1e-8 added inside log to avoid log(0); no NaN/Inf under any input including confidence_mask=all-zero

## Config Schema

```yaml
trust_gate:
  type: hard               # "hard" | "soft_weight" | "soft_label"
  tau_conf: 0.90
  tau_struct: 0.60          # used by "hard" and as structural input to others
  soft_weight_temp_conf: 0.1     # used by soft_weight
  soft_weight_temp_struct: 0.1   # used by soft_weight
  soft_label_k: 4.0               # temperature scaling constant
  soft_label_t_max: 5.0           # temperature cap
```

## Training Flow

```python
if cfg.trust_gate.type == "hard":
    trust_weights = hard_gate.compute_mask(confidence, tgt_heatmaps)
    target_loss = satg_loss(tgt_student_logits, pseudo_labels, trust_weights)

elif cfg.trust_gate.type == "soft_weight":
    trust_weights = soft_weight_gate.compute_weights(confidence, tgt_heatmaps)
    target_loss = satg_loss(tgt_student_logits, pseudo_labels, trust_weights)

elif cfg.trust_gate.type == "soft_label":
    soft_targets = soft_label_module.compute_soft_targets(
                       tgt_teacher_logits, tgt_heatmaps)
    confidence_mask = soft_label_module.compute_confidence_mask(
                       tgt_teacher_logits, cfg.trust_gate.tau_conf)
    target_loss = soft_label_kl_loss(tgt_student_logits, soft_targets,
                                      confidence_mask)
```
