# SATG — Method

## Overview

SATG adds a structural prior branch to any UDA semantic segmentation pipeline.
For each input image, it computes a per-pixel structural complexity heatmap.
This heatmap modulates the pseudo-label loss during self-training: pixels in
structurally complex regions contribute less to the gradient.

The method is **model-agnostic** (works with any teacher-student UDA setup)
and **lightweight** (adds ~5–10 ms per image for a classical CV prior).

## Structural Prior

### Formulation

The structural complexity $S(p) \in [0, 1]$ at pixel $p$ is a weighted
combination of two cues computed from the grayscale input image $I$:

$$S(p) = \alpha \cdot \bar{E}(p) + \beta \cdot \bar{V}(p)$$

where $\alpha + \beta = 1$ (default: $\alpha = \beta = 0.5$).

### Edge Component ($\bar{E}$)

Computed via **Canny edge detection** with thresholds (low=50, high=150):

1. Apply Canny to the grayscale image → binary edge map $E_0(p) \in \{0, 255\}$
2. Min-max normalise: $\bar{E}(p) = (E_0(p) - \min(E_0)) / (\max(E_0) - \min(E_0))$

Since Canny produces a binary output, $\bar{E}(p)$ is either **0** (non-edge)
or **1** (edge). This creates a hard categorical split in the complexity score:

- Non-edge pixels: $S(p) = \beta \cdot \bar{V}(p)$ (range $[0, \beta]$)
- Edge pixels: $S(p) = \alpha + \beta \cdot \bar{V}(p)$ (range $[\alpha, 1]$)

**Consequence**: Since $\alpha = 0.5$, edge pixels always have
$S(p) \geq 0.5$. See 04-RESULTS.md for the implications of this.

### Local Variance Component ($\bar{V}$)

Computed via **Gaussian blur** (kernel 5×5, $\sigma = 1.0$):

1. Compute blurred image: $B = G_\sigma * I$
2. Compute blurred squared image: $B_{sq} = G_\sigma * I^2$
3. Variance: $V_0(p) = \max(0, B_{sq}(p) - B(p)^2)$
4. Min-max normalise: $\bar{V}(p) = (V_0(p) - \min(V_0)) / (\max(V_0) - \min(V_0))$

### Per-Image Normalisation

**Important caveat**: Both $\bar{E}$ and $\bar{V}$ are min-max normalised
**per image**, not globally. This means:
- A pixel with edge strength = 100 in a low-contrast image may be assigned
  $\bar{E} = 1.0$, while the same pixel in a high-contrast image gets
  $\bar{E} = 0.5$.
- This makes **cross-image complexity bin comparisons imprecise** (see
  04-RESULTS.md for analysis).

Alternative (global normalisation) was considered but rejected because:
- Edge/variance statistics vary dramatically across images
- A fixed threshold would gate/unreasonably different fractions per image
- Per-image norm ensures consistent within-image coverage

### Resolution

The prior runs at the input image resolution (1024×2048 for Cityscapes val).
It is computed before any model inference and cached.

## Soft-Weighting Strategy (Primary)

The complexity heatmap $S(p)$ is converted to a per-pixel loss weight
$w(p) \in [0, 1]$:

$$w(p) = f(S(p))$$

where $f$ is a monotonically decreasing function.

**Proposed function** (design space — to be tuned):
- Piecewise linear: $w(p) = \max(0, 1 - S(p) / \gamma)$
- Sigmoid: $w(p) = 1 - \sigma(k \cdot (S(p) - \tau))$
- Power: $w(p) = (1 - S(p))^\gamma$

The weight multiplies the cross-entropy loss at each pixel:
$$\mathcal{L} = -\sum_p w(p) \cdot \hat{y}(p) \log y(p)$$

### Design Requirements (from diagnostic test findings)

1. The weight function should drop most sharply in the complexity range
   $[0, 0.3]$ where the error rate increase is most pronounced.
2. Above $S \approx 0.3$, the weight should plateau (error rate stabilises).
3. Edge pixels ($S \geq 0.5$) receive reduced but non-zero weight
   (soft, not hard gating).

## Hard Gating Strategy (Baseline)

An optional hard threshold $\tau$ gates pixels:
$$w(p) = \begin{cases} 1 & \text{if } S(p) < \tau \\ 0 & \text{otherwise} \end{cases}$$

**Not recommended as primary approach** because:
- Most pixels in Cityscapes are low-complexity ($S < 0.1$), so the gate
  affects few pixels
- The error rate difference is modest, so hard gating loses signal
- Requires careful threshold tuning per dataset

## Integration into UDA Pipeline

```
Input Image (x)
                ┌──────────────────────┐
                │  SATG Structural Prior │
                │  (Canny + Variance)    │
                └──────┬───────────────┘
                       │ complexity map S(p)
                       ▼
Teacher Model ──→ Pseudo-labels ŷ(p) ──→ Weight w(p) = f(S(p))
                                                │
                                                ▼
Student Model ──→ Prediction y(p) ──→ CrossEntropy(y(p), ŷ(p))
                                               × w(p)
                                                │
                                                ▼
                                            Loss
```

The student receives **weighted** pseudo-label supervision — complex regions
contribute less, forcing the student to rely more on structurally simple,
likely-correct regions and to generalise better to complex ones.

## Comparison to Standard DACS

In standard DACS/DAFormer:
- All pseudo-labels above the confidence threshold contribute equally
- Loss: $\mathcal{L} = -\sum_p \mathbf{1}[\text{conf}(p) > \theta] \cdot \hat{y}(p) \log y(p)$

In SATG:
- Confidence thresholding retained (to filter low-quality predictions)
- Structural weight adds a second dimension:
  $\mathcal{L} = -\sum_p \mathbf{1}[\text{conf}(p) > \theta] \cdot w(S(p)) \cdot \hat{y}(p) \log y(p)$

## Computational Cost

- **Edge detection**: ~2–5 ms per 1024×2048 image (OpenCV optimised)
- **Local variance**: ~3–5 ms per image (two Gaussian blurs)
- **Total overhead**: ~5–10 ms per image, negligible vs. model inference (~5–10 s)
- **Memory**: one float32 heatmap at image resolution (~8 MB for Cityscapes)
