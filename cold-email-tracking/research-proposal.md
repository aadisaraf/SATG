# SATG: Structure-Aware Trust Gating for Unsupervised Domain Adaptation in Semantic Segmentation

**Author:** Aadi Saraf
**Contact:** aadi.saraf@outlook.com

## Problem

Self-training via teacher-student distillation is the dominant paradigm for UDA in semantic segmentation. The teacher generates pseudo-labels on target images, and the student learns from them. But pseudo-labels suffer from **confirmation bias** — the teacher is confidently wrong on out-of-distribution regions, and the student inherits those errors.

Standard filtering uses a confidence threshold (discard predictions below 0.9). This fails for two reasons: (1) neural networks can be highly confident even in structurally ambiguous regions (occlusions, clutter, unfamiliar textures), and (2) useful supervision below the threshold is thrown away entirely.

## Key Insight

The central novelty of SATG is **decoupling predictive confidence from structural reliability**. A teacher's 99% confidence on a heavily occluded region should not be trusted if an independent structural prior says the region is untrustworthy.

## Proposed Approach

SATG augments standard pseudo-labeling with an **image-space structural prior** computed per target image:

1. **Structural Prior Extraction** — Precompute edge density (Sobel gradient magnitude with box blur smoothing) and local variance heatmaps. These are domain-agnostic — edges remain edges whether in GTA5 or Cityscapes.
2. **Dual Gating** — Two independent masks are computed per pixel:
   - **Confidence mask**: `teacher_conf > 0.7` (coarse noise filter)
   - **Trust mask**: `structural_heatmap < tau` (low clutter = high trust)
   - **Final gate**: pixel must be both confident AND structurally trustworthy
3. **Soft-Label Variant** — Instead of binary rejection, apply temperature-scaled soft targets in structurally volatile regions: higher temperature = softer distribution, so the student learns cautiously rather than not at all.

### Gating Logic (PyTorch)

```python
teacher_probs = softmax(teacher_logits, dim=1)
teacher_conf, pseudo_labels = max(teacher_probs, dim=1)

conf_mask = teacher_conf > conf_thresh   # conventional: 0.9; see note below
trust_mask = structural_heatmap < struct_tau  # low clutter = high trust
final_mask = conf_mask & trust_mask

loss = cross_entropy(student_logits, pseudo_labels, reduction='none')
return (loss * final_mask).mean()
```

> **Note on `conf_thresh`:** The field standard is 0.9–0.95 — a conservative heuristic to bound confirmation bias, but it's arbitrary and penalizes structurally complex classes. SATG's contribution is the *second* trust gate: once structural reliability is measured independently, the confidence threshold can potentially be relaxed (e.g., lowered to 0.8) because the trust mask catches false positives confidence alone misses. Optimal `conf_thresh` *given* a trust mask is an experimental question we'll ablate.

The confidence threshold is intentionally lowered from the standard 0.9 to 0.7. This lets through structurally simple regions where the teacher may be only moderately confident (60–80%) but not pathologically wrong. The structural trust gate handles the fine-grained filtering, so the two gates operate in complementary regimes: confidence removes obviously unreliable predictions, structure controls the ambiguous ones.

```python
# Without SATG (standard Mean Teacher, conf > 0.9): ~45% pixels masked
# With SATG (conf > 0.7 + struct < tau):     ~40% pixels masked
# But composition is different — structurally clean mid-conf regions recovered,
# structurally noisy high-conf regions rejected.
```

## Experimental Setup

| Component | Specification |
|-----------|---------------|
| Benchmark | GTA5 (synthetic) → Cityscapes (real) |
| Architecture | DeepLabV3+ ResNet50 (PyTorch 2.7.0) |
| Task | 19-class semantic segmentation |
| Training | 40k iterations, batch size 4, EMA teacher update |
| Baselines | Source-Only, Standard Mean Teacher |
| Ablations | Sweeps over conf_thresh, struct_tau, prior type, and gating mode (see below) |

## Ablation Plan

| Dimension | Values | What It Tests |
|-----------|--------|---------------|
| `conf_thresh` | 0.6, 0.7, 0.8, 0.9 | How much can the confidence gate be relaxed before the trust gate saturates? |
| `struct_tau` | percentile: p50, p70, p85, p95 | How aggressive should the structural filter be? |
| Prior type | edge density (Sobel) only, local variance only, both | Which structural signal carries the most orthogonal information? |
| Gating mode | hard (binary mask), soft-weight (continuous), soft-label (temp-scaling) | Is graceful degradation strictly better than hard rejection? |
| Temp. scale `k` | 1.0, 2.0, 3.0, 5.0 | How soft should soft labels be in structurally complex regions? |
| Backbone | ResNet50 vs SegFormer-B1 | Does SATG generalize beyond one architecture? |

Each ablation is run for the full 40k training schedule. Primary metric: per-class and mean IoU on Cityscapes val. Secondary metric: gate retention rate (% pixels masked) to measure whether structural and confidence gates are selecting complementary subsets.

## Risk Mitigation

**Biased rejection** — A strict structural filter may systematically reject inherently complex classes (poles, bicycles, traffic signs). *Mitigation:* Class-balanced trust filtering that normalizes the structural score per class using the teacher's initial prediction, preventing entire classes from being gated out.

**Computation overhead** — Structural priors are pre-computed offline and stored as tensor arrays, so runtime overhead during training is limited to tensor indexing.

## Timeline (8 Weeks)

| Phase | Weeks | Key Deliverables |
|-------|-------|------------------|
| Baselines | 1–2 | Source-Only + Mean Teacher mIoU |
| Prior Engineering | 3–4 | Structural heatmap pipeline + offline pre-computation |
| SATG Integration | 5–6 | Dual gating in training loop, ablations running |
| Evaluation | 7–8 | Quantitative results, qualitative failure analysis, paper draft |

## Open Questions

1. Is structural complexity orthogonal to class frequency as a source of pseudo-label noise?
2. Does soft-label modulation particularly benefit structurally complex classes (pole, fence, traffic light)?
3. How does SATG interact with feature-level alignment (e.g., OT-based alignment, adversarial DA)?
4. Can structural priors be learned end-to-end rather than computed via classical CV ops?

## Seeking Mentorship

I'm seeking guidance on experimental design, statistical analysis, and positioning the contribution within the broader UDA landscape. I'm particularly interested in exploring how SATG combines with recent feature-level alignment methods (DAFormer, MIC, HRDA).

---

*This document accompanies cold email outreach to professors whose research aligns with UDA, domain adaptation, semantic segmentation, self-supervised learning, or related areas.*
