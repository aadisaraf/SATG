# SATG — Critical Analysis

## 1. Risk Assessment

### Risk 1: Structural Complexity / Class Frequency Confounding

**Severity**: MEDIUM
**Status**: NOT YET FULLY ADDRESSED

The diagnostic test shows error rates correlate with structural complexity.
But structural complexity may correlate with object class. Rare classes
(e.g., train, motorcycle, rider) may appear in structurally complex regions
more often than common classes (road, sky, building).

If class frequency is the true driver of error rate (rare classes have
less training data → higher errors), and complexity happens to correlate
with class frequency, then the apparent complexity-error relationship is
**confounded** by class frequency.

**What we need**: Per-class decomposition of error rates and complexity
distributions in the diagnostic results. If rare classes show high error
rates regardless of complexity, and common classes show low error rates
regardless of complexity, the confounding risk is high. If error rate
varies with complexity within each class class, the hypothesis is
independently supported.

**Mitigation in plan**: Soft-weighting instead of hard gating reduces
the risk (no pixel is fully discarded). The plan now includes a
hypothesis validation framework (T000c, T033d) to explicitly check
this before full training runs.

### Risk 2: Statistical Power (Seed Count)

**Severity**: MEDIUM → LOW (mitigated)
**Status**: ADDRESSED

Original plan used 3 seeds per experiment. Updated plan (T048b) uses
5 seeds. This reduces the risk that results are seed-dependent.
With 5 seeds, a difference of ~1–2 mIoU points should be detectable
at standard significance levels.

### Risk 3: No DAFormer Combination Experiments

**Severity**: HIGH → LOW (mitigated)
**Status**: ADDRESSED

Original plan only tested SATG with DeepLabV2 (weaker baseline).
Updated plan adds T048e–T048h: DAFormer + SATG in 4 configurations.
This ensures SATG is tested on the strongest baseline, making the
results relevant to the current state-of-the-art.

### Risk 4: Bin-5 Edge Artefact

**Severity**: LOW
**Status**: ANALYZED

The bin-5 anomaly (see 04-RESULTS.md) is fully explained by the
Canny binary edge map + low-variance combination. It is not a bug
but a feature of the prior design. Implications:

- SATG will weight edge pixels (especially thin edges) the most heavily.
- This is desirable — edges are where errors concentrate.
- Care needed: some edge "errors" may be annotation noise at boundaries
  rather than genuine prediction failures. Pixel-wise accuracy is an
  imperfect metric at object boundaries.
- **Recommended mitigation**: Include boundary-aware evaluation (e.g.,
  boundary IoU) in the full training experiments to distinguish genuine
  boundary errors from annotator uncertainty.

### Risk 5: Per-Image Normalisation

**Severity**: LOW
**Status**: UNDERSTOOD

Per-image normalisation of the structural prior means cross-image
complexity bin comparisons are not strictly valid. However, for the
soft-weighting application, per-image normalisation is actually
appropriate — the weight function operates per-image and the relative
ranking within each image is what matters.

**For the diagnostic test specifically**: The aggregate bin statistics
are still meaningful for detecting the presence of a trend but the
exact numerical values should be interpreted with caution. Future
diagnostic tests should consider recording global statistics alongside
per-image-min-max statistics.

## 2. Critique of the Structural Prior

### Strengths
- Computationally free (~5–10 ms per image)
- Interpretable (edges + texture = structural complexity)
- Model-agnostic (same prior works for any architecture)
- Per-image normalisation adapts automatically to different image types

### Weaknesses
- Canny binary edge map introduces a hard split at $S = 0.5$ — complexity
  is not a smooth measure
- Per-image normalisation limits cross-image comparisons
- Two fixed parameters (Canny thresholds, Gaussian kernel size) — may need
  tuning for different datasets
- Only captures edges and local texture — misses global structure
  (layout, perspective, object relationships)
- 5×5 Gaussian window is small; may miss mid-range texture patterns

### Potential Improvements
1. **Multi-scale variance**: Use multiple Gaussian kernel sizes (3×3, 7×7,
   15×15) and aggregate
2. **Gradient magnitude instead of Canny**: Replace binary edge with
   continuous Sobel magnitude
3. **Superpixel-based complexity**: Compute complexity over superpixel
   regions rather than per-pixel
4. **Learned prior**: Train a small network to predict complexity from
   the input image
5. **Global normalisation**: Normalise the prior over the entire dataset
   or a held-out reference set

## 3. Effect Size Analysis

The maximum observed effect: error rate increases from 2.89% (bin 0) to
4.70% (bin 5). In absolute terms: **+1.81 percentage points**.

In practical terms for UDA semantic segmentation:
- If 4.7% of bin-5 pixels (4.1% of all pixels) are errors, and 2.89% of
  bin-0 pixels (95.6% of pixels) are errors, the weighted average is 2.97%.
- If SATG reduces bin-5 errors by 20% (relative) — from 4.70% to 3.76% —
  the overall error rate drops from 2.97% to 2.93%, a 0.04 percentage point
  improvement.
- This translates to roughly **0.3–0.6 mIoU points** improvement estimate.
- **This is meaningful but not dramatic.** SATG is best viewed as a
  complementary improvement, not a breakthrough.

### When SATG Would Help More

- **Larger domain gaps** (e.g., synthia→cityscapes): Pseudo-label errors
  are more numerous, so there's more room for improvement.
- **Weaker baselines**: The DAFormer is already quite good. SATG may help
  more with DeepLabV2 or other weaker backbones.
- **Different structural priors**: A better prior could reveal a stronger
  complexity-error correlation.

## 4. Comparison to Alternative Approaches

| Approach | Cost | Expected Gain | Complementary to SATG? |
|----------|------|---------------|----------------------|
| **Confidence weighting** | Free | Moderate | Yes — SATG adds structure dimension |
| **Uncertainty estimation** | 2× inference | Higher | Partially overlapping |
| **Curriculum learning** | Free | Moderate | Yes — orthogonal |
| **Instance-level weighting** | Free | Moderate | Yes — orthogonal |
| **Boundary losses** | Free | Low–moderate | Partially (SATG already covers boundaries) |
| **Ensemble** | 3–5× inference | Higher | Yes — orthogonal |
| **Data augmentation** | Free | High | Yes — orthogonal |

SATG's unique position: **cheapest complexity signal** (classical CV) that is
**orthogonal to most other improvements**.

## 5. Recommended Next Steps

### Immediate (Pre-Implementation)
1. **Add per-class decomposition to the diagnostic test** — rule out or
   confirm the confounding risk
2. **Run the hypothesis validation dry run (T000c)** — quick training
   test to check if SATG weighting affects the loss landscape
3. **Add logging hooks (T033d)** — enable real-time monitoring of
   structural complexity distribution during training

### Short-Term (Implementation Phase)
4. **Implement soft-weighting module** with tunable weight function
5. **Integrate into DACS pipeline**
6. **Run DeepLabV2 + SATG experiments** (5 seeds each, T048b)

### Medium-Term (Validation)
7. **Run DAFormer + SATG combination experiments** (T048e–T048h)
8. **Evaluate with boundary IoU** alongside standard mIoU
9. **Check for class-confounding in results**

### Long-Term (Extensions)
10. **Explore improved structural priors** (multi-scale, continuous gradient)
11. **Test on additional domain gaps** (Synthia, ACDC, DarkZurich)
12. **Ablation on weight function design** (piecewise, sigmoid, power)
