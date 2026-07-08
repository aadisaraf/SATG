# SATG — Introduction

## Problem Statement

Unsupervised Domain Adaptation (UDA) for semantic segmentation aims to
transfer a model trained on a labelled source domain (e.g. GTA, a synthetic
city) to an unlabelled target domain (e.g. Cityscapes, real urban scenes).
The dominant paradigm is **self-training with pseudo-labels**: the teacher
model generates pseudo-labels on the target domain, and the student model
trains on them.

**Core weakness**: pseudo-labels contain errors — especially high-confidence
errors — and the standard loss function (pixel-wise cross-entropy with a
confidence threshold) treats all pseudo-labels above the threshold equally.
It does not distinguish between structurally simple regions where the
teacher is likely correct and structurally complex regions where the teacher
is more likely to be confidently wrong.

## Hypothesis

> **SATG Hypothesis**: In semantic segmentation UDA, the teacher model's
> high-confidence pseudo-label errors are concentrated in structurally
> complex regions of the image. By detecting structural complexity from the
> input image alone (without running the model), we can modulate the
> contribution of each pseudo-labelled pixel during training — down-weighting
> pixels in complex regions where errors are more likely.

Formally:
- Let $S(x) \in [0, 1]$ be a structural complexity measure at pixel $x$
  computed from the input image.
- Let $E(x) = \mathbf{1}[\hat{y}(x) \neq y(x)]$ be an indicator of whether
  the teacher's pseudo-label is incorrect at $x$.
- The hypothesis claims: $\mathbb{P}(E(x) = 1 \mid S(x) > \tau) > \mathbb{P}(E(x) = 1 \mid S(x) \leq \tau)$
  for some threshold $\tau$. Equivalently, error rate is positively correlated
  with structural complexity.

## Key Insight: Masking vs Soft-Weighting

There are two ways to use a structural complexity signal:

1. **Hard rejection (masking/gating)**: Discard pseudo-labels in high-complexity
   regions entirely. Simple but risks losing signal in regions where the teacher
   may still be correct.

2. **Soft-weighting (preference)**: Apply a continuous weight $w(S(x)) \in [0, 1]$
   to each pixel's loss, where weight decreases with increasing complexity.
   Preserves all signal but reduces the influence of likely-erroneous pixels.

**Decision**: Soft-weighting is the primary contribution. Hard rejection is
retained as a baseline and fallback mechanism. Rationale: the diagnostic test
shows the effect is modest in absolute terms (error rate increases from ~2.9%
to ~4.4% for complex pixels), so hard-gating would lose too much signal.

## Related Work

The SATG approach is related to several lines of work:

- **Confidence-based weighting** (standard DAFormer/DACS): weights pixels
  by softmax confidence. Does not account for structural context.

- **Uncertainty estimation**: Bayesian methods (Monte Carlo dropout, ensemble)
  estimate predictive uncertainty but add inference cost. SATG is cheaper
  (classical CV on the input image).

- **Attention-based methods**: Some works learn to attend to reliable regions.
  SATG differs by using a fixed, interpretable prior rather than a learned
  one, and by operating at the pixel-weight level rather than the attention level.

- **Curriculum learning**: Easy-first training strategies. SATG is complementary
  — it weights all pixels throughout training rather than ordering them.

- **Instance-level weighting**: Some methods weight by class frequency or
  detection confidence. SATG adds a complementary structural dimension.

The key novelty is the **specific prior** (Canny + local variance) combined
with **per-pixel soft-weighting** applied to pseudo-labels in UDA, and the
**demonstration that structural complexity correlates with pseudo-label error**
via the diagnostic test.

## Related Work Links (Directly Comparable)

The DAFormer paper (Hoyer et al., 2022) is the baseline we build on:
- Architecture: MiT-B5 encoder + DAFormer decoder
- UDA method: DACS with thing-class feature distance
- Training: GTA→Cityscapes, 40K iterations
- Baseline mIoU: ~68% on Cityscapes val (GTA→CS)

The structural prior is connected to:
- Classical edge detection (Canny, Sobel) — used in low-level vision
- Image complexity metrics — used in image quality assessment
- Attention mechanisms in transformers — SATG provides a structural
  attention signal complementary to learned self-attention

## Evolution of the Hypothesis

1. **Initial formulation**: Structural complexity can predict pseudo-label errors.
   Use it to gate (hard-reject) high-complexity pixels.

2. **After critical analysis (see 05-ANALYSIS.md)**: Identified risk of
   confounding (complexity correlates with object class frequency; rare
   classes may cluster in complex regions). Made soft-weighting primary
   to mitigate this.

3. **After diagnostic test**: Hypothesis partially supported. Error rate
   clearly increases in structurally complex regions (2.89% → 4.44% for
   bins 0→2, a +54% relative increase). But magnitude is modest. Bin-5
   anomaly discovered (see 04-RESULTS.md). Soft-weighting confirmed as
   the correct approach.
