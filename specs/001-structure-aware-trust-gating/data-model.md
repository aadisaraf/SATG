# Data Model: Structure-Aware Trust Gating (SATG)

**Date**: 2026-06-25 | **Feature**: 001-structure-aware-trust-gating

## Entities

### 1. Structural Heatmap

Per-pixel float32 map encoding local structural complexity.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| shape | (H, W) | matches input image | Spatial dimensions |
| dtype | float32 | — | NumPy array |
| values | float | [0.0, 1.0] | 0=simple, 1=complex |

**Computation**: `H = w₁·edge_density + w₂·local_variance`, both normalized to [0,1]
**Storage**: `.npy` files, naming convention `{image_stem}_satg_heatmap.npy`
**Validation**: All values in [0.0, 1.0]; shape matches source image; deterministic (same input → same output)

---

### 2. Teacher Confidence Map

Per-pixel maximum softmax probability from teacher model.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| shape | [B, H, W] | batch × spatial | One score per pixel |
| dtype | float32 | — | PyTorch tensor |
| values | float | (0, 1) | Max softmax prob per pixel |

**Computation**: `confidence = softmax(teacher_logits).max(dim=1)`
**Computation context**: `torch.no_grad()` — no gradient tracking
**Validation**: All values in (0, 1); shape matches input spatial dims

---

### 3. Pseudo-Label Map

Per-pixel argmax class from teacher model.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| shape | [B, H, W] | batch × spatial | One class index per pixel |
| dtype | int64 | — | PyTorch tensor |
| values | int | {0, ..., 18} | Cityscapes 19-class trainID |

**Computation**: `pseudo_labels = teacher_logits.argmax(dim=1)`
**Computation context**: `torch.no_grad()`
**Validation**: All values in [0, 18]; shape matches input spatial dims

---

### 4. Trust Mask / Weights

Per-pixel trust weight applied to per-pixel cross-entropy loss.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| shape | [B, H, W] | batch × spatial | One weight per pixel |
| dtype | float32 | — | PyTorch tensor |
| values (hard) | float | {0.0, 1.0} | Binary accept/reject |
| values (soft) | float | [0.0, 1.0] | Continuous trust weight |

**Hard computation**: `mask = (confidence > τ_conf) AND (structure < τ_struct).float()`
**Soft computation**: `weights = sigmoid(β₀ + β₁·confidence - β₂·structure)`
**Edge case**: If all weights are 0.0, target loss = 0.0 (no NaN/Inf)

---

### 5. SATG Loss

Scalar target-domain loss weighting student predictions by trust.

| Field | Type | Description |
|-------|------|-------------|
| output | float32 scalar tensor | Weighted cross-entropy loss |

**Computation**: 
```
per_pixel_ce = F.cross_entropy(student_logits, pseudo_labels, reduction='none')
loss = (per_pixel_ce * trust_weights).sum() / (trust_weights.sum() + 1e-8)
```
**Edge case**: If `trust_weights.sum() == 0`, return `torch.tensor(0.0, device=...)`

---

### 6. GTA5 Dataset

| Property | Value |
|----------|-------|
| Total images | 24,966 |
| Resolution | 1914×1052 |
| Label format | RGB palette → 19-class index maps (preprocessed) |
| Split | All images for training |
| Source domain | Synthetic (game engine) |

---

### 7. Cityscapes Dataset

| Property | Value |
|----------|-------|
| Training images | 2,975 |
| Validation images | 500 |
| Resolution | 2048×1024 |
| Label format | gtFine_labelTrainIds.png (19-class index) |
| Source domain | Real (dashcam) |

---

## Relationships

```
GTA5 Dataset ──(source batches)──→ Trainer
                                       │
Cityscapes Dataset ──(target batches)──→ Trainer
                                            │
Precomputed Heatmaps ──(target batches)──→ Trainer
                                            │
                                    ┌───────┴───────┐
                                    │               │
                              Source Loss    Target Loss
                                    │               │
                                    │        ┌──────┴──────┐
                                    │        │             │
                                    │   Trust Gate   Student Logits
                                    │        │             │
                                    │   Trust Weights      │
                                    │        │             │
                                    │   SATG Loss ◄───────┘
                                    │        │
                                    └───► Total Loss
                                            │
                                    EMA Update → Teacher
```

## State Transitions

### Training Step State Machine

```
[IDLE] → load source batch → [SOURCE_LOADED]
[SOURCE_LOADED] → load target batch → [TARGET_LOADED]
[TARGET_LOADED] → teacher forward (no_grad) → [PSEUDO_LABELS_READY]
[PSEUDO_LABELS_READY] → student forward → [LOGITS_READY]
[LOGITS_READY] → compute trust weights → [WEIGHTS_READY]
[WEIGHTS_READY] → compute losses → [LOSSES_COMPUTED]
[LOSSES_COMPUTED] → backward + optimizer step → [STEP_COMPLETE]
[STEP_COMPLETE] → EMA update → [IDLE]
```

### Checkpoint State

```
[NO_CHECKPOINT] → first eval → [HAS_BEST (miou=X)]
[HAS_BEST] → eval improves → [HAS_BEST (miou=Y>X)]
[HAS_BEST] → eval doesn't improve → [HAS_BEST (unchanged)]
```
