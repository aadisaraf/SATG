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
| §3.1 Mandatory Ablations | ✅ PASS | 7 ablation variants enumerated |
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
│   ├── satg_soft.yaml             # SATG soft weighting (extends default)
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
│   └── losses.py                  # SATGLoss class
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
│   ├── test_losses.py
│   ├── test_ema.py
│   └── test_data_loaders.py
├── EXPERIMENTS.md                 # All results + configs
├── README.md                      # Installation + usage
├── requirements.txt               # Pinned dependencies
├── pyproject.toml                 # Project config (black, flake8, coverage)
└── .coveragerc                    # pytest-cov config (backup)
```

**Structure Decision**: Single project layout. All Python packages at repo root for flat imports (`from satg.structural_prior import StructuralPrior`). Separate `tests/` directory mirroring source layout.

## Complexity Tracking

> No constitution violations — no complexity tracking needed.
