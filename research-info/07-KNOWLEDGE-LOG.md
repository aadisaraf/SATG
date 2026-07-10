# SATG — Knowledge Log

Chronological record of everything learned about SATG, the codebase, the
diagnostic test, and related research. Each entry is timestamped and tagged
with the source of the information.

---

## 2026-07-08

### Entry 1: Codebase Structure (explored specs/ codebase)

**Source**: Reading SATG spec documents

**Learned**:
- The SATG project is in EARLY STAGE (specs written, no implementation)
- Structure under `specs/001-structure-aware-trust-gating/`:
  - `spec.md`: Feature specification (what to build)
  - `plan.md`: Implementation plan (how to build it)
  - `tasks.md`: Task breakdown (actionable units for parallel agents)
- Supporting docs: `research.md`, `data-model.md`, `contracts/module-api.md`, checklists
- The project uses the Spec Kit methodology (.specify/ directory)
- Three subagents (B, C, D) planned for different modules
- Original plan had hard-gating as primary mechanism

### Entry 2: Structural Heatmap Mechanics

**Source**: Codebase exploration via codegraph

**Learned**:
- The structural prior module computes per-pixel complexity via:
  - Canny edge detection (low=50, high=150)
  - Local variance via Gaussian blur (5×5, σ=1.0)
  - Equal-weighted combination (α=β=0.5)
- The trust-gating module takes the heatmap and produces per-pixel weights
- These weights modulate the pseudo-label loss during self-training
- No implementation exists yet — the entire system is in the design phase

### Entry 3: Precomputation for Cityscapes Heatmaps

**Source**: User question + analysis

**Learned**:
- Cityscapes heatmaps are precomputed because:
  1. The original spec mentioned precomputation for efficiency
  2. Cityscapes is a fixed dataset (2975 train images), making precomputation feasible
  3. The prior is deterministic (no model, no noise), so one-time compute suffices
  4. Precomputation trades disk space for training-time speed
- Classical CV (Canny + variance) is NOT "better" than ML — it's chosen for:
  - Interpretability (we know exactly what "structural complexity" means)
  - Zero training cost
  - Deterministic and reproducible
  - Fast enough for real-time if needed (5–10 ms per image)
- ML alternative: A small network could learn to predict complexity,
  potentially capturing more subtle patterns, but adds training and
  interpretability costs

### Entry 4: Critical Risk Assessment

**Source**: Adversarial self-evaluation

**Learned**:
Four critical risks identified:

**R1 — Confounding with class frequency**: Structural complexity may
correlate with object class. Rare classes (train, motorcycle, rider)
tend to have more complex boundaries and appear less frequently in
training. If SATG down-weights complex pixels, it may inadvertently
down-weight rare classes — exactly the opposite of what we want.

**R2 — Statistical power**: With only 3 seeds per experiment (original
plan), we cannot distinguish a real 1 mIoU improvement from random
seed variance. UDA training has high variance. Need 5+ seeds.

**R3 — Missing DAFormer combination**: Original plan only tested SATG
with DeepLabV2 (2017 baseline). DAFormer (2022) is the strongest
baseline and the natural host for SATG. Without DAFormer experiments,
the paper cannot claim SOTA relevance.

**R4 — Hypothesis may not hold**: Before the diagnostic test, there
was no empirical evidence that structural complexity correlates with
pseudo-label errors. The entire project stood or fell on this assumption.

### Entry 5: Plan Update — Soft-Labeling Primary

**Source**: User request to update plan addressing risks

**Learned**:
- Soft-weighting is now the primary contribution (was hard-gating)
- Rationale: R1 (confounding) makes hard-gating too risky — if rare
  classes cluster in complex regions, hard-gating systematically
  discards them → worse representation learning
- Soft-weighting preserves all pixels but modulates their contribution
- Hard-gating retained as a baseline/comparison mechanism
- Plan.md updated: reordered summary, added Critical Risk Assessment,
  updated Module Specifications (Subagent D = primary), updated
  Training Flow (soft-label assignment and loss computation moved
  to Phase 3), added Hypothesis Validation Framework, added
  Combination Experiments section
- Tasks.md updated: Priority Restructuring section, new tasks
  T033d (hypothesis validation logging), T000c (hypothesis
  validation dry run), T048e–T048h (DAFormer combos), T048b
  now uses 5 seeds

### Entry 6: Diagnostic Test Design

**Source**: Designing the diagnostic test code

**Learned**:
- The test processes 500 Cityscapes val images through DAFormer
- Computes structural complexity per pixel via the prior
- Bins pixels into 10 decile bins by complexity
- For each bin: total pixels, high-confidence errors (pred≠gt AND softmax>0.9)
- Counts aggregated across all images → error rate per bin
- Built as standalone pipeline (not integrated into mmseg training)
- Uses MPS acceleration (M4 Mac), processes ~5–10s per image
- Full run takes ~60 minutes for 500 images

### Entry 7: Diagnostic Test Results (MAJOR)

**Source**: Running diagnostic pipeline on 500 Cityscapes val images

**Learned**:
- Overall high-confidence error rate: 2.97%
- Clear upward trend in bins 0→2: 2.89% → 4.04% → 4.44%
  (+54% relative increase)
- **Hypothesis is partially supported**
- Error rate plateaus for bins 3–9 (fluctuates 3.2–4.7% with no trend)
- **95.6% of pixels are in bin 0** (very low complexity)
- The effect is modest in absolute terms (+1.5 percentage points max)

### Entry 8: The Bin-5 Anomaly (MAJOR FINDING)

**Source**: Deep analysis of decile table

**Learned**:
- Bin 5 (complexity 0.5–0.6) has 37.8M pixels (4.12% of total) and
  4.70% error rate — the highest of all bins
- This is NOT a statistical artefact
- **Root cause**: Canny edge detection produces a binary edge map.
  After min-max normalisation, edge pixels have $\bar{E} = 1$ and
  non-edge pixels have $\bar{E} = 0$.
  - Non-edge pixels: $S = 0.5 \cdot \bar{V}$, range [0, 0.5]
  - Edge pixels: $S = 0.5 + 0.5 \cdot \bar{V}$, range [0.5, 1.0]
  - Bin 5 = edge pixels with very low local variance ($\bar{V} \in [0, 0.2)$)
- These are the **thinnest, sharpest edges** in the image:
  lane markings, road edges, sign boundaries, pole outlines
- The model fails more on these because:
  1. 1/4-resolution features cannot resolve sub-pixel thin structures
  2. Bilinear upsampling blurs sharp edges
  3. Edge pixels are underrepresented in training (~4.3% of pixels)
- Error rate DECREASES as variance increases in edge bins (5→9):
  edges in textured regions are easier than isolated sharp edges
- This is actually **good for SATG**: the pixels that most need
  down-weighting (thin edges, high error rate) are correctly identified
  by the prior

### Entry 9: Per-Image Normalisation Limitation

**Source**: Code analysis of structure_prior.py

**Learned**:
- The structural prior uses per-image min-max normalisation for both
  edge and variance components
- This means: cross-image complexity bin comparisons are not strictly
  valid — a pixel with S=0.5 in image A may have different raw
  edge+variance than a pixel with S=0.5 in image B
- **Mitigation for soft-weighting**: This is acceptable because the
  weight function operates per-image, and within-image relative
  rankings are what matter
- **For diagnostic test**: Aggregate bin statistics still detect
  the presence of a trend, but exact numerical values are approximate
- Future improvement: Add global normalisation alongside per-image

### Entry 10: SATG Effect Size Estimate

**Source**: Synthesis of diagnostic test results

**Learned**:
- Maximum observable improvement: reducing 4.70% error rate on 4.1%
  of pixels (bin 5) and 4.44% on 0.1% of pixels (bin 2)
- Expected mIoU improvement: **~0.3–0.6 points** (rough estimate)
- This is meaningful but modest
- SATG should be presented as a **complementary improvement** not a
  breakthrough
- Larger domain gaps (Synthia→CS) may show larger effects
- Weaker baselines (DeepLabV2) may show larger relative improvements

### Entry 11: Methodological Implication for Weight Function

**Source**: Bin-5 analysis

**Learned**:
- The weight function $w(S)$ must account for the edge/non-edge split
  at $S = 0.5$
- Steepest gradient should be in $S \in [0, 0.3]$ (error rate rises
  most steeply here)
- Edge pixels ($S \geq 0.5$) should receive reduced but non-zero weight
- Candidate functions: piecewise linear, sigmoid, power
- The function can plateau above $S \approx 0.3$ since error rate
  doesn't continue rising

### Entry 12: Missing Per-Class Data

**Source**: Risk assessment + diagnostic test design limitation

**Learned**:
- Current diagnostic test aggregates across all classes
- Cannot distinguish whether complexity-error correlation is driven
  by class frequency (rare class pixels happen to be complex) or
  by structure independent of class
- **This is the most important missing piece** before the final analysis
- Solution: Add per-class tracking to DiagnosticStats (class-specific
  decile bins or per-class error rates)

### Entry 13: Standard Dataset Processing

**Source**: Reading Cityscapes dataset infrastructure

**Learned**:
- Cityscapes val images: 1024 × 2048 resolution
- 500 validation images across 3 cities (frankfurt, lindau, munster)
- Ground truth: labelIds (0–33) converted to trainIds (0–18 + 255 void)
- 19 valid classes with void class (255) for unlabelled pixels and
  ignored labelIds
- Standard DAFormer training uses crop_size (512, 512) with
  random crops and resize augmentation
- The diagnostic test uses full-resolution images (no resize),
  so the model runs at native 1024 × 2048 with logits upsampled
  from 1/4 stride

### Entry 14: MPS Performance Characteristics

**Source**: pipeline.log analysis

**Learned**:
- DAFormer on MPS (M4): ~5.5s per image initially, rising to
  ~6–9s per image over the 500-image run
- Some images spike to 15–52s (likely page faults or memory pressure)
- Memory: 3.5–4.5 GB allocated throughout
- torch.mps.empty_cache() called per image to prevent fragmentation
- ~60 minutes total for 500 images
- DAFormer-MiT-B5 is ~3093 MB in parameters

### Entry 15: Codebase Structure for Implementation

**Source**: Reading SATG project structure

**Learned**:
- The SATG project is structured for parallel subagent development:
  - Subagent B: Structural prior module (edge detection, variance, heatmap)
  - Subagent C: Integration with MMSegmentation pipeline (configs, hooks)
  - Subagent D: Trust-gating UDA block (soft-weighting, loss modulation)
- Module API contracts defined in contracts/module-api.md
- Data model defined in data-model.md
- Testing strategy: unit tests per module + integration test + hypothesis
  validation dry run
- The project has NOT been implemented yet — we are in pre-implementation
  phase

## 2026-07-10

### Entry 16: Extended Structural Prior — Two Additional Cues Identified (MAJOR FINDING)

**Source**: Deep theoretical analysis of the structural prior's failure modes

**Learned**:
The current two-cue prior (edge density + local variance) captures important
but incomplete structural information. Two additional cues — local Shannon
entropy and structure tensor minimum eigenvalue (cornerness) — are strongly
motivated by both theory and the specific failure modes of GTA5→Cityscapes
UDA.

**Why 2 cues are insufficient — 3 failure modes**:

1. **Occlusion boundaries**: When a pedestrian stands half-behind a pole,
   the boundary region has gradients from TWO directions simultaneously
   (pedestrian edge + pole edge). Edge density sees "there are edges here"
   — correct. Local variance sees "there is intensity variation" — correct.
   But neither captures the fact that gradients are coming from multiple
   directions at once — the precise geometric signature of an occlusion.
   This is exactly where the teacher assigns the wrong class and where
   confidence is most dangerously miscalibrated.

2. **Domain-shift vegetation**: GTA5 trees are geometrically simple
   (repetitive polygon meshes). Cityscapes trees have complex, irregular
   leaf structure with many subtle intensity gradations. Edge density is
   moderately high in both. Local variance captures some difference. But
   what truly separates them is the DISTRIBUTIONAL RICHNESS of intensity
   values — Cityscapes vegetation has many distinct micro-level intensity
   values (high entropy) whereas GTA5 vegetation has fewer distinct values
   arranged more predictably (lower entropy). Entropy is the right tool.

3. **Complex backgrounds with clean foreground**: A sign post (structurally
   simple foreground) in front of a busy intersection (high-complexity
   background). The local window centered on the post edge captures BOTH
   the clean post boundary AND the chaotic background. Edge density and
   variance average out the complexity. Multi-directional gradient
   complexity (cornerness) specifically identifies junction zones where
   boundaries from multiple objects overlap.

**Addition 1 — Local Shannon Entropy**:
- H_ent[p] = -Σ_b p_b * log(p_b) where p_b is the fraction of pixels in
  a local window with intensity in bin b (e.g., 32 quantization bins)
- Captures DISTRIBUTIONAL richness that variance cannot: variance = second
  statistical moment (spread from mean); entropy = information content
  (number of distinct states)
- A 50/50 black-and-white striped patch has HIGH variance but LOW entropy
  (only 2 states). A gradient-rich multi-textured patch has HIGH variance
  AND HIGH entropy.
- Implementation: `skimage.filters.rank.entropy(gray_uint8, disk(radius))`
  — O(H*W*B), fully vectorized, ~0.3–0.8s per full-res Cityscapes image.

**Addition 2 — Structure Tensor Minimum Eigenvalue (λ₂, Cornerness)**:
- Structure tensor: J = blur(∇I ⊗ ∇I), where ∇I = (Gx, Gy) is the gradient.
  J is a 2×2 matrix with components Jxx=blur(Gx²), Jxy=blur(Gx·Gy),
  Jyy=blur(Gy²). λ₂ = 0.5*(Jxx+Jyy) - 0.5*sqrt((Jxx-Jyy)² + 4Jxy²)
- λ₂ ≈ 0: either flat region (no gradient) or a clean edge (gradient all
  in one direction). Both are structurally simple.
- λ₂ >> 0: gradients exist in MULTIPLE DIRECTIONS in the local window.
  This is the mathematical signature of: a corner, a junction, an occlusion
  boundary, or chaotic multi-object texture.
- Why uniquely valuable for UDA: Under domain shift, the teacher's
  segmentation boundaries are most likely wrong at JUNCTIONS — where
  multiple objects meet. These are precisely where λ₂ is highest.
- Implementation: cv2.Sobel → elementwise products → cv2.GaussianBlur →
  λ₂ formula. All vectorized, ~0.1s per image.

### Entry 17: Separate Component File Design for Ablation Efficiency

**Source**: Design analysis of precomputation architecture

**Learned**:
With 4 cues, storing a single pre-combined heatmap is inefficient for
ablation studies. To run "edge-only" with the old design, we'd need to
re-run precomputation with edge_weight=1.0 — 2 hours of CPU time per
ablation type.

**New design**: Store 4 separate component .npy files per image. Combine
them at DATALOADER time using config weights.

Per-image files:
  {stem}_satg_edge.npy    — edge density [H,W] float32
  {stem}_satg_var.npy     — local variance [H,W] float32
  {stem}_satg_ent.npy     — local entropy [H,W] float32
  {stem}_satg_corn.npy    — cornerness λ₂ [H,W] float32

At training time:
  H = w1*edge + w2*var + w3*ent + w4*corn    (all from config)
  H = clip(H, 0, 1).astype(float32)

Ablation benefits:
  "edge-only":      edge_weight=1.0, others 0.0   → no re-precompute
  "variance-only":  variance_weight=1.0, others 0.0 → no re-precompute
  "entropy-only":   entropy_weight=1.0, others 0.0  → no re-precompute
  "cornerness-only": cornerness_weight=1.0, others 0.0 → no re-precompute
  "all-four equal":  all 0.25 → default

Disk impact: 4× more files, each float32 [H,W] ≈ 2MB per image.
Total: 2975 images × 4 × 2MB ≈ 24GB. Ensure data disk has space.

**Constitution check**: All four cues are derived from raw RGB pixels using
classical deterministic CV operations with no learned parameters and no
target-domain labels. No constitutional violation.

### Entry 18: Updated Combination Formula

**Source**: Integration of the 4-cue prior design

**Learned**:
The new combination formula:
  H = w1*edge_density + w2*local_variance + w3*local_entropy + w4*cornerness
with w1 + w2 + w3 + w4 = 1 (enforced by design; default all 0.25)

The structural_prior config section must be extended with:
  entropy_kernel_size: 15      # disk radius for entropy
  entropy_bins: 32             # quantization bins
  cornerness_kernel_size: 15   # Gaussian window for structure tensor
  cornerness_sigma: 2.0        # Gaussian sigma for structure tensor
  entropy_weight: 0.25
  cornerness_weight: 0.25
  edge_weight: 0.25            # was 0.5, now redistributed
  variance_weight: 0.25        # was 0.5, now redistributed

The trust_gate config and all downstream modules (HardTrustGate,
SoftWeightTrustGate, TemperatureSoftLabel) are UNCHANGED — they consume
the final combined heatmap H, which remains in [0,1].

---

## 2026-07-10

### Entry 10: Structural Prior — Dead Components Debugging & Fix

**Source**: Running `verify_components` after heatmap precomputation

**Problem**: 3 of 4 structural prior components (edge, var, corn) had near-zero means (< 0.02), making the combined heatmap ≈ entropy alone. The trust gate would have no useful structural signal.

**Root causes**:
1. **Edge (Canny)**: Thresholds of 50/150 found only 0.6% of pixels as edges on Cityscapes (2048×1024). Urban scenes have large uniform regions (sky, road) and the Canny hysteresis was too aggressive.
2. **Variance & Cornerness (min-max normalization)**: Outlier pixels (sharp edges, lane markings) with values 100× the median stretched the normalization range to near-zero for 95% of pixels.

**Fix applied**:
- **Edge**: Replaced Canny with Sobel gradient magnitude (continuous) + p95 percentile normalization. No thresholds to tune, mean went 0.006 → 0.26.
- **Variance & Cornerness**: Replaced min-max normalization with `_percentile_normalize(arr, pct=95.0)` — clips at 95th percentile then divides by that value. Variance mean 0.02 → 0.14, cornerness mean 0.004 → 0.14.
- Config updated: removed `edge_low_threshold`/`edge_high_threshold`, added `norm_percentile: 95.0`.

**Files changed**:
- `satg/structural_prior.py` — `compute()`, new `_percentile_normalize()` helper
- `configs/default.yaml` — config schema update
- `precompute/compute_heatmaps.py` — `_default_cfg()` update, `--resume` flag, disk space check

**Verification** (post-fix):
| Component | Pre-fix mean | Post-fix mean | Target range |
|-----------|-------------|--------------|--------------|
| edge      | 0.006       | 0.26         | [0.05, 0.50] |
| var       | 0.02        | 0.14         | [0.05, 0.50] |
| ent       | 0.49        | 0.56         | [0.20, 0.80] |
| corn      | 0.004       | 0.14         | [0.05, 0.50] |

No component below 0.02 or above 0.95. File count 11,900 = 2,975 × 4.

**Trade-off to track**: Components are now more correlated (edge-var 0.92) than before (0.71). This is because gradient magnitude and variance both rise near edges — they naturally co-vary. The old low correlations were just measuring near-zero signal. This is the correct fix; correlation is not a problem by itself.

**When to revisit coefficients** — come back to tune if any of these happen during training:

1. Validation mIoU plateaus well below expectation.
2. The trust gate becomes too strict or too loose.
3. One component clearly dominates the others in gate behavior.
4. Ablations show a simpler weighting scheme wins consistently.

If that happens, tune in this order: **weights first** → **norm_percentile** → **component definitions** (last resort). Recomputation without a training signal is expensive and unlikely to be useful.

## Quality Notes

- This log captures everything that was learned during the exploration,
  analysis, and diagnostic test phases
- Entries are ordered chronologically; later entries supersede earlier findings
  where the implementation diverged from the spec
- Sources are tagged where relevant
- Claims are distinguished from interpretations where possible
- Open questions and uncertainties are explicitly noted
