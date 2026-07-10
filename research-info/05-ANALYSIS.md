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
- Computationally free (~0.5–1.0 s per image for all 4 cues)
- Interpretable (edges + texture + distributional richness + multi-directional gradients)
- Model-agnostic (same prior works for any architecture)
- Per-image normalisation adapts automatically to different image types
- Four cues capture complementary aspects of structural complexity —
  the combination is more robust than any two-cue subset
- Separate component storage enables zero-cost ablations

### Why the Original Two-Cue Prior Was Insufficient

The original prior (Canny edge density + local variance) captured important
but incomplete structural information. Three specific failure modes motivated
the extension to four cues:

**Failure 1 — Occlusion Boundaries**:
A pedestrian half-behind a pole. The boundary region has gradients from TWO
directions simultaneously (pedestrian edge + pole edge). Edge density sees
"there are edges here" — correct. Local variance sees "there is intensity
variation" — correct. But neither captures the FACT that gradients are
coming from multiple directions at once, which is the precise geometric
signature of an occlusion. Cornerness ($\lambda_2$) specifically identifies
multi-directional gradient fields.

**Failure 2 — Domain-Shift Vegetation**:
GTA5 trees are geometrically simple (repetitive polygon meshes). Cityscapes
trees have complex, irregular leaf structure with many subtle intensity
gradations. Edge density is moderately high in both. Local variance captures
some difference. But what truly separates them is the DISTRIBUTIONAL RICHNESS
of intensity values — Cityscapes vegetation has many distinct micro-level
intensity values (high entropy) whereas GTA5 vegetation has fewer distinct
values arranged more predictably (lower entropy). Entropy is the right tool.

**Failure 3 — Complex Backgrounds with Clean Foreground**:
A sign post (structurally simple foreground) in front of a busy intersection
(high-complexity background). The local window centered on the post edge
captures BOTH the clean post boundary AND the chaotic background. Edge
density and variance average out the complexity. But multi-directional
gradient complexity (cornerness) specifically identifies the junction zones
where boundaries from multiple objects overlap, even at pixel-level
precision.

**Variance vs. Entropy — The Key Distinction**:
- Variance = second statistical moment = spread from mean.
- Entropy = information content = number of distinct states.
- A 50/50 black-and-white striped patch has HIGH variance but LOW entropy
  (only two states). A gradient-rich multi-textured patch has HIGH variance
  AND HIGH entropy.
- For Cityscapes: the transition between vegetation and sky, or between
  building textures and road surface, produces the exact multi-state
  intensity distribution that entropy captures and variance underestimates.

**Edge Density vs. Cornerness — The Key Distinction**:
- Edge density counts how many edges exist in a window (binary count).
- Cornerness ($\lambda_2$) specifically identifies where edges from
  DIFFERENT directions intersect — the most dangerous locations for
  pseudo-label errors.
- Harris (1988) used $\lambda_2$ as a corner detector: it is the classical
  measure of multi-directional gradient complexity.

### Weaknesses
- Canny binary edge map introduces a hard split at $S = 0.5$ — complexity
  is not a smooth measure across the edge/non-edge boundary
- Entropy entropy computation is ~300–800 ms per image (skimage) — the
  bottleneck of the 4-cue pipeline
- Per-image normalisation limits cross-image comparisons
- Four tunable parameters (Canny thresholds, Gaussian kernel sizes,
  entropy radius, cornerness sigma) — configuration space is larger
- All four cues operate at the pixel level — misses global structure
  (layout, perspective, object relationships)
- Skimage entropy depends on `disk` structural element which assumes
  isotropic neighbourhoods; anisotropic scenes (e.g., long horizon lines)
  may be better served by rectangular windows
- 24 GB storage for 4 separate components (vs. 6 GB for single heatmap)

### Potential Improvements (Deferred)
1. **Multi-scale cues**: Compute each cue at multiple scales and aggregate
2. **Gradient magnitude instead of Canny**: Replace binary edge with
   continuous Sobel magnitude — avoids the hard split at S=0.5
3. **Superpixel-based complexity**: Compute complexity over superpixel
   regions rather than per-pixel
4. **Learned prior**: Train a small network to predict complexity from
   the input image — only after evaluating the 4-cue classical prior
5. **Global normalisation**: Normalise the prior over the entire dataset
   or a held-out reference set
6. **Anisotropic entropy**: Replace `disk` with `rectangle` structural
   elements for direction-aware entropy
7. **Orientation-aware cornerness**: Decompose cornerness into
   principal orientations for finer-grained junction analysis

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
