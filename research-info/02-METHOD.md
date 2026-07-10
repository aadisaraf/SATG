# SATG — Method

## Overview

SATG adds a structural prior branch to any UDA semantic segmentation pipeline.
For each input image, it computes a per-pixel structural complexity heatmap.
This heatmap modulates the pseudo-label loss during self-training: pixels in
structurally complex regions contribute less to the gradient.

The method is **model-agnostic** (works with any teacher-student UDA setup)
and **lightweight** (adds ~5–10 ms per image for a classical CV prior).

## Structural Prior

### Overview

The structural heatmap $H(p) \in [0, 1]$ at pixel $p$ is a weighted
combination of **four cues** computed from the grayscale input image $I$:

$$H(p) = w_1 \cdot \bar{E}(p) + w_2 \cdot \bar{V}(p) + w_3 \cdot \bar{H}_{ent}(p) + w_4 \cdot \lambda_2(p)$$

where $w_1 + w_2 + w_3 + w_4 = 1$ (default: all 0.25).

Each cue captures a distinct aspect of structural complexity that the
others miss. Together they provide a richer, more robust prior than any
two-cue combination.

**Why four cues?** See 05-ANALYSIS.md §2 for the three identified failure
modes of the original two-cue prior (occlusion boundaries, domain-shift
vegetation, complex background + clean foreground).

### Cue 1: Edge Density ($\bar{E}$)

Computed via **Canny edge detection** with thresholds (low=50, high=150):

1. Apply Canny to the grayscale image → binary edge map $E_0(p) \in \{0, 255\}$
2. Convolve with uniform kernel (15×15) for sliding-window density
3. Normalise: $\bar{E}(p) = E_{density}(p) / 255.0$ → range [0, 1]

**What it captures**: Number of edge pixels in a local window.
**Limitation**: Canny is binary. A crisp road-edge and a fuzzy occlusion
boundary both register as "edge present". No directional information.

### Cue 2: Local Variance ($\bar{V}$)

Computed via **box blur** (kernel 15×15):

1. $mean = blur(I)$; $mean_{sq} = blur(I^2)$
2. $V_0(p) = \max(0, mean_{sq}(p) - mean(p)^2)$
3. Min-max normalise: $\bar{V}(p) = (V_0 - \min(V_0)) / (\max(V_0) - \min(V_0))$

**What it captures**: Intensity spread (second statistical moment) in a
local window.
**Limitation**: Variance captures the SPREAD of values but not their
DISTRIBUTION. Two regions with identical variance can have very different
structural complexity (e.g., 50/50 black-white vs. 10 evenly-spaced greys).

### Cue 3: Local Shannon Entropy ($\bar{H}_{ent}$)

Computed via **local histogram entropy** (disk radius 7, 32 bins):

$$H_{ent}[p] = -\sum_{b=1}^{B} p_b \log(p_b)$$

where $p_b$ is the fraction of pixels in a disk window centred at $p$ with
intensity in bin $b$ (B = 32 quantization bins).

1. Quantize grayscale image to $B$ bins
2. For each pixel, compute histogram over local disk window
3. Compute Shannon entropy from histogram
4. Min-max normalise: $\bar{H}_{ent} = (H_{ent} - \min(H_{ent})) / (\max(H_{ent}) - \min(H_{ent}))$

Implementation: `skimage.filters.rank.entropy(gray_uint8, disk(radius))`.

**What it captures that variance cannot**: Variance = spread from mean
(second moment). Entropy = information content (number of distinct states).

- A 50/50 black-and-white striped patch: **HIGH variance, LOW entropy**
  (only 2 states, highly predictable)
- A gradient-rich multi-textured patch: **HIGH variance, HIGH entropy**
  (many distinct intensity values)
- GTA5 vegetation (low entropy) vs. Cityscapes vegetation (high entropy):
  entropy captures the distributional richness difference that variance
  underestimates

**Where it matters most**:
- Vegetation (complex real-world texture vs. uniform synthetic)
- Mixed-material backgrounds (building+vegetation+sign together)
- Horizon zones where multiple semantic classes blend

### Cue 4: Structure Tensor Minimum Eigenvalue ($\lambda_2$, Cornerness)

Computed via **structure tensor** analysis:

1. Compute image gradients: $G_x = \text{Sobel}_x(I)$, $G_y = \text{Sobel}_y(I)$
2. Elementwise products: $G_{xx} = G_x^2$, $G_{xy} = G_x \cdot G_y$, $G_{yy} = G_y^2$
3. Blur each component: $\bar{G}_{xx} = G_\sigma * G_{xx}$, etc.
4. Per-pixel structure tensor: $J(p) = \begin{bmatrix} \bar{G}_{xx}(p) & \bar{G}_{xy}(p) \\ \bar{G}_{xy}(p) & \bar{G}_{yy}(p) \end{bmatrix}$
5. Minimum eigenvalue: $\lambda_2(p) = \frac{1}{2}\left(J_{xx}+J_{yy} - \sqrt{(J_{xx}-J_{yy})^2 + 4J_{xy}^2}\right)$
6. Min-max normalise to [0, 1]

**What it captures that edge density cannot**:
- $\lambda_2 \approx 0$: either a flat region (no gradient) or a clean edge
  (gradient all in one direction). Both are structurally simple.
- $\lambda_2 \gg 0$: gradients exist in MULTIPLE DIRECTIONS in the local
  window. This is the mathematical signature of: a corner, a junction, an
  occlusion boundary, or chaotic multi-object texture.

**Why uniquely valuable for UDA**:
- Under domain shift, the teacher's segmentation boundaries are most likely
  wrong at JUNCTIONS — where multiple objects meet. These are precisely
  where $\lambda_2$ is highest.
- A clean road edge ($\lambda_2 \approx 0$): teacher probably correct.
- A pedestrian in front of a building ($\lambda_2 \gg 0$): teacher is
  navigating a multi-directional gradient field and is most likely to error.
- Harris (1988) used $\lambda_2$ as a corner detector for exactly this
  reason — it is the classical measure of multi-directional gradient
  complexity.

### Per-Image Normalisation

**Important caveat**: All four cues are min-max normalised **per image**,
not globally. This means:
- A pixel with edge strength = 100 in a low-contrast image may be assigned
  $\bar{E} = 1.0$, while the same pixel in a high-contrast image gets
  $\bar{E} = 0.5$.
- This makes **cross-image complexity bin comparisons imprecise** (see
  04-RESULTS.md for analysis).

Alternative (global normalisation) was considered but rejected because:
- Edge/variance/entropy/cornerness statistics vary dramatically across images
- A fixed threshold would gate unreasonably different fractions per image
- Per-image norm ensures consistent within-image coverage

### Separate Component Storage

To support efficient ablation without re-precomputation, the four cues are
stored as **separate .npy files** per image and combined at dataloader time:

Per-image files:
```
{stem}_satg_edge.npy    — edge density [H,W] float32
{stem}_satg_var.npy     — local variance [H,W] float32
{stem}_satg_ent.npy     — local entropy [H,W] float32
{stem}_satg_corn.npy    — cornerness λ₂ [H,W] float32
```

At training time:
```python
H = w1 * edge + w2 * var + w3 * ent + w4 * corn
H = np.clip(H, 0, 1).astype(np.float32)
```

This enables any ablation variant — "edge-only", "entropy-only",
"all-four equal" — with zero re-computation cost.

### Resolution

The prior runs at the input image resolution (1024×2048 for Cityscapes val).
It is computed before any model inference and cached.

## Soft-Weighting Strategy (Primary)

The combined complexity heatmap $H(p)$ (from 4 cues) is converted to a
per-pixel loss weight $w(p) \in [0, 1]$:

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
                ┌──────────────────────────────────┐
                │  SATG Structural Prior           │
                │  (Canny + Variance + Entropy     │
                │   + Structure Tensor Cornerness) │
                └──────┬───────────────────────────┘
                       │ combined heatmap H(p)
                       ▼
Teacher Model ──→ Pseudo-labels ŷ(p) ──→ Weight w(p) = f(H(p))
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
- Structural weight (from 4-cue combination) adds a second dimension:
  $\mathcal{L} = -\sum_p \mathbf{1}[\text{conf}(p) > \theta] \cdot w(H(p)) \cdot \hat{y}(p) \log y(p)$
  where $H(p) = w_1\bar{E} + w_2\bar{V} + w_3\bar{H}_{ent} + w_4\lambda_2$

## Computational Cost

- **Edge detection**: ~2–5 ms per 1024×2048 image (OpenCV optimised)
- **Local variance**: ~3–5 ms per image (two box blurs)
- **Local entropy**: ~300–800 ms per image (skimage vectorised)
- **Structure tensor cornerness**: ~100 ms per image (OpenCV Sobel + blur)
- **Total overhead**: ~0.5–1.0 s per image, still negligible vs. model inference (~5–10 s)
- **Storage per image**: 4 × float32 heatmap at image resolution ≈ 4 × 8 MB = 32 MB
- **Precomputation**: ~3,000 images × 1s ≈ 50 minutes with 8 cores, still well under 2-hour budget
