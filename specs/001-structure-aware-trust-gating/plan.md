# Implementation Plan: Structure-Aware Trust Gating (SATG)

**Branch**: `001-structure-aware-trust-gating` | **Date**: 2026-06-25 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-structure-aware-trust-gating/spec.md`

## Summary

SATG is a UDA semantic segmentation method that augments standard teacher-student pseudo-labeling with an image-space structural prior. **The primary contribution is soft pseudo-label modulation** — temperature-scaled soft targets (Soft-Label) and continuous trust weighting (Soft-Weight) — rather than binary hard rejection. The structural prior (edge density + local variance) identifies regions where the teacher is likely overconfident, and soft modulation gracefully reduces supervision signal strength in these regions rather than discarding pseudo-labels entirely. The system uses DeepLabV3+ ResNet50 with EMA teacher updates, trained on GTA5→Cityscapes with 19-class Cityscapes label space.

**Why soft modulation is primary**: Hard rejection discards useful information at threshold boundaries and creates unstable training dynamics. Soft modulation preserves partial supervision from structurally complex regions, enabling the student to learn from imperfect-but-informative pseudo-labels. This is a stronger scientific contribution because it tests whether *graceful degradation* of supervision is more effective than *binary filtering*.

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

**Storage Considerations**: Precomputed heatmaps for 2,975 Cityscapes images at 2048×1024 float32 = ~24GB. Mitigation: store as float16 (halves to ~12GB) or use memory-mapped loading. Heatmaps are computed once and reused across all experiments.

## Critical Risk Assessment

### Risk 1: Structural Complexity / Class Frequency Confounding

The spec claims structural confirmation bias is "orthogonal" to standard confirmation bias (addressed by DAFormer's Rare Class Sampling). However, the structurally complex classes (pole, fence, traffic sign, rider) are also the rare classes that DAFormer targets. This creates a confound: improvements attributed to structural complexity might actually come from implicitly handling class frequency.

**Mitigation**: 
- Ablation F (Soft mechanism comparison) must include per-class IoU breakdowns
- Report per-class trust coverage to detect class bias in trusted pixels
- Explicitly discuss whether structural complexity and class frequency are truly independent signals

### Risk 2: Hypothesis Validation Required

The core assumption — that structural complexity correlates with pseudo-label error — is plausible but not guaranteed. The domain gap (GTA5→Cityscapes) is primarily about appearance (synthetic vs. real textures), not structural complexity. A structurally simple region (sky) might have high pseudo-label error due to appearance mismatch, while a structurally complex region (building facade) might be correctly classified.

**Mitigation**: Before full implementation, the training pipeline must log structural complexity vs. pseudo-label error correlation metrics. This data validates or invalidates the core hypothesis without requiring a separate diagnostic test.

### Risk 3: Statistical Power with 3 Seeds

With 3 seeds and typical UDA variance (~2-3 mIoU std), detecting a 1 mIoU improvement requires ~50 seeds for 80% power at p<0.05. With only 3 seeds, the experiment is severely underpowered for modest improvements.

**Mitigation**:
- Acknowledge that improvements <0.5 mIoU are marginal per constitution §1.8
- For key experiments (SATG Hard vs. Soft-Label vs. Mean Teacher), increase to 5 seeds {42, 1337, 2024, 7, 99}
- Report confidence intervals alongside mean±std

### Risk 4: Missing Combination Experiments

The spec claims SATG is complementary to DAFormer/MIC/HRDA, but the experiments only compare against Standard Mean Teacher. Without testing SATG+DAFormer > DAFormer, the complementarity claim is unsubstantiated.

**Mitigation**: Add combination experiments in Phase 9: SATG Soft-Label applied on top of DAFormer-trained model. This requires either:
- (a) Loading a DAFormer checkpoint and continuing training with SATG, or
- (b) Implementing DAFormer's Rare Class Sampling within the SATG training loop

Option (b) is preferred because it enables true joint training. This adds ~4 experiments (DAFormer baseline + SATG+DAFormer × 2 seeds).

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

### satg/soft_label.py: TemperatureSoftLabel class (PRIMARY CONTRIBUTION)

This is the primary module of SATG. It implements temperature-scaled soft pseudo-labels that gracefully degrade supervision signal strength in structurally complex regions.

- `__init__(cfg: OmegaConf)`: reads `k` (default 4.0), `T_max` (default 5.0), `tau_conf` (shared with hard gate, for the pre-filter)
- `compute_temperature(struct: Tensor[B,H,W]) -> Tensor[B,H,W]`: T = 1.0 + k * struct, then clamp to [1.0, T_max]
- `compute_soft_targets(teacher_logits: Tensor[B,C,H,W], struct: Tensor[B,H,W]) -> Tensor[B,C,H,W]`:
  1. Compute per-pixel temperature T [B,H,W]
  2. Expand T to [B,1,H,W] and divide teacher_logits by T (broadcast across class dim)
  3. Apply softmax over class dim → soft_targets [B,C,H,W]
  4. Return soft_targets (each pixel's C-vector sums to 1.0)
- `compute_confidence_mask(teacher_logits: Tensor[B,C,H,W], tau_conf: float) -> Tensor[B,H,W]`: reuse the SAME confidence computation as the existing hard trust gate (max softmax prob > tau_conf), used as a binary pre-filter before the distributional loss is applied

**Why this is primary**: Soft-label modulation preserves partial supervision from structurally complex regions. The student learns "this is probably class X, but I'm uncertain" rather than receiving either a hard label or no label at all. This enables graceful degradation rather than binary filtering.

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

### satg/trust_gate.py: HardTrustGate, SoftTrustGate (SECONDARY)

Hard and soft trust gates are secondary mechanisms. Hard rejection is the simplest baseline; soft weighting provides continuous trust scores.

### satg/structural_prior.py: StructuralPrior (FOUNDATION)

The structural prior is the foundation that all trust mechanisms depend on. It must be validated before any trust mechanism can be tested.

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

## Hypothesis Validation Framework

Before full training runs, the system must validate the core hypothesis through in-training metrics:

### Correlation Tracking

During training, for each batch:
1. Compute per-pixel pseudo-label error: `error = (pseudo_label != ground_truth_for_source_equivalent)` — since we don't have target ground truth, use the proxy: compute teacher confidence vs. student prediction disagreement as a proxy for unreliability
2. Compute per-pixel structural heatmap value
3. Log Pearson correlation coefficient between structural heatmap and prediction disagreement
4. If correlation is weak (<0.1) or negative, the hypothesis is invalidated

### Per-Class Trust Coverage Analysis

For each of the 19 Cityscapes classes:
1. Compute fraction of pixels belonging to that class that are trusted
2. Flag any class with <10% or >90% mean coverage as potential bias
3. Report in EXPERIMENTS.md alongside per-class IoU

### Structural Complexity Distribution Analysis

Log histograms of:
1. Structural heatmap values for trusted pixels
2. Structural heatmap values for rejected pixels
3. Teacher confidence for trusted vs. rejected pixels

These distributions reveal whether the trust gate is making meaningful distinctions.

## Combination Experiments (Phase 9 Extension)

To validate the complementarity claim (SATG + DAFormer > DAFormer), add:

### DAFormer Rare Class Sampling Integration

Implement Rare Class Sampling as a config option:
```yaml
data:
  rare_class_sampling: true  # Enable inverse frequency weighting for source batches
  class_frequencies: [0.36, 0.05, 0.19, 0.01, 0.01, 0.01, 0.01, 0.01, 0.11, 0.01, 0.05, 0.01, 0.01, 0.10, 0.01, 0.01, 0.00, 0.01, 0.01]
```

### Combination Experiments

| Experiment | Config | Purpose |
|-----------|--------|---------|
| DAFormer baseline | `configs/baseline_daformer.yaml` | Establish DAFormer performance |
| SATG Soft-Label + DAFormer | `configs/satg_soft_label_daformer.yaml` | Test complementarity |
| SATG Soft-Weight + DAFormer | `configs/satg_soft_weight_daformer.yaml` | Test complementarity |

These experiments require either:
- (a) Loading a DAFormer checkpoint and continuing with SATG training, or
- (b) Implementing Rare Class Sampling within the SATG training loop (preferred)

Option (b) enables true joint training and is the recommended approach.

## Training Flow

The training flow prioritizes soft-label modulation as the primary mechanism. Hard rejection is the simplest baseline; soft weighting is an intermediate; soft-label is the full contribution.

```python
# PRIMARY: Soft-label modulation (temperature-scaled pseudo-labels)
if cfg.trust_gate.type == "soft_label":
    soft_targets = soft_label_module.compute_soft_targets(
                       tgt_teacher_logits, tgt_heatmaps)
    confidence_mask = soft_label_module.compute_confidence_mask(
                       tgt_teacher_logits, cfg.trust_gate.tau_conf)
    target_loss = soft_label_kl_loss(tgt_student_logits, soft_targets,
                                      confidence_mask)

# SECONDARY: Soft-weight modulation (continuous trust weights)
elif cfg.trust_gate.type == "soft_weight":
    trust_weights = soft_weight_gate.compute_weights(confidence, tgt_heatmaps)
    target_loss = satg_loss(tgt_student_logits, pseudo_labels, trust_weights)

# BASELINE: Hard rejection (binary mask)
elif cfg.trust_gate.type == "hard":
    trust_weights = hard_gate.compute_mask(confidence, tgt_heatmaps)
    target_loss = satg_loss(tgt_student_logits, pseudo_labels, trust_weights)
```

### Hypothesis Validation During Training

To validate the core hypothesis (structural complexity correlates with pseudo-label error), the training loop must log:
1. Per-batch correlation between structural heatmap values and pseudo-label error rates
2. Per-class trust coverage ratios (detect class bias)
3. Distribution of structural complexity values across trusted vs. rejected pixels

These metrics are logged to WandB and analyzed post-training to validate or invalidate the core assumption.
