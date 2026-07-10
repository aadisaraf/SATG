# SATG — Plan Evolution

This document tracks how the implementation plan changed in response to
findings from codebase analysis, critical evaluation, and the diagnostic test.

## v1: Initial Plan (Pre-Critical Analysis)

- Hard-gating as primary mechanism
- DeepLabV2 as sole evaluation baseline (3 seeds)
- No hypothesis validation step
- No per-class decomposition
- Simple structural prior, fixed parameters

## v2: After Critical Risk Assessment

**Changes**:
1. **Soft-weighting promoted to primary approach**; hard-gating retained
   as baseline. Rationale: confounding risk (R1) means hard-gating could
   systematically discard rare classes. Soft-weighting mitigates this.

2. **Added Hypothesis Validation Framework**:
   - T000c: Hypothesis validation dry run (quick 5K-iter training with
     SATG weighting vs. baseline, check loss divergence)
   - T033d: Add logging hooks for structural complexity distribution
     during training

3. **Increased seed count**: T048b updated from 3 to 5 seeds for
   statistical significance

4. **Added DAFormer combination experiments**: T048e (DAFormer + SATG,
   no hard gating), T048f (DAFormer + SATG + hard gating), T048g
   (DAFormer + SATG, soft-weighting only), T048h (DAFormer + SATG,
   soft + hard combined)

5. **Restructured priorities**:
   - Subagent D (soft-weighting UDA block) = P1 (was P2)
   - US-04b (hard-gating ablation) = P3 (was P1)
   - US-04 (comprehensive comparison) = P4 (was P3)
   - US-03 (confidence only baseline) = P5 (was P3)

6. **Updated training flow**: Soft-label assignment and loss computation
   moved to Phase 3 (was Phase 4). Hard-gating moved to Phase 4.

## v3: After Diagnostic Test Results

**Key finding driving the update**: The hypothesis is partially supported
with a modest effect size (error rate 2.89% → 4.70%, peak at bin 5).

**Changes**:
1. **Confirmed soft-weighting as correct primary approach** — the modest
   effect size means hard-gating would lose too much signal.

2. **Bin-5 analysis incorporated into method design**:
   - The weight function should handle the edge/non-edge split at S=0.5
   - Edge pixels (S ≥ 0.5) should get reduced weight, not zero weight
   - The steepest weight gradient should be in S ∈ [0, 0.3]

3. **Per-class decomposition added as prerequisite** to diagnostic test
   before final analysis. Critical for addressing confounding risk.

4. **Boundary IoU added as evaluation metric** alongside standard mIoU,
   since SATG specifically targets boundary errors.

## v4: After Extended Structural Prior Research (Precomputation Phase)

**Key finding driving the update**: The original two-cue prior (edge density +
local variance) misses critical structural information that entropy and
cornerness capture — specifically occlusion boundaries, vegetation texture
complexity, and multi-directional gradient junctions.

**Changes**:
1. **Structural prior expanded from 2 cues to 4 cues**:
   - Added Local Shannon Entropy ($\bar{H}_{ent}$): captures distributional
     richness that variance misses
   - Added Structure Tensor Minimum Eigenvalue ($\lambda_2$, cornerness):
     captures multi-directional gradient complexity that binary edge density misses
   - All four cues computed from raw RGB using classical deterministic CV
     operations — no constitutional violation

2. **Changed from single combined heatmap to 4 separate component files**:
   - Per-image files: `{stem}_satg_edge.npy`, `{stem}_satg_var.npy`,
     `{stem}_satg_ent.npy`, `{stem}_satg_corn.npy`
   - Combined at dataloader time using config weights
   - Enables zero-cost ablation: any weight combination without re-precomputation
   - Disk impact: ~24 GB (up from ~6 GB for single heatmap)

3. **Updated combination formula**:
   - Old: $H = 0.5 \cdot \bar{E} + 0.5 \cdot \bar{V}$
   - New: $H = w_1\bar{E} + w_2\bar{V} + w_3\bar{H}_{ent} + w_4\lambda_2$
   - Default: $w_1 = w_2 = w_3 = w_4 = 0.25$

4. **Precomputation budget re-evaluated**:
   - Entropy is the bottleneck (~300–800 ms per image via skimage)
   - Total per-image time: ~0.5–1.0s (up from ~10 ms)
   - With 8 cores: ~50 minutes for 2,975 images — still well under 2-hour budget
   - Storage: ~24 GB for 4 components (ensure data disk has space)

5. **Config schema extended** — new keys under `structural_prior`:
   ```yaml
   structural_prior:
     # existing keys unchanged ...
     entropy_kernel_size: 15      # disk radius for entropy
     entropy_bins: 32             # quantization bins
     cornerness_kernel_size: 15   # Gaussian window for structure tensor
     cornerness_sigma: 2.0        # Gaussian sigma for structure tensor
     entropy_weight: 0.25
     cornerness_weight: 0.25
     edge_weight: 0.25            # was 0.5
     variance_weight: 0.25        # was 0.5
   ```

6. **Downstream modules UNCHANGED**: HardTrustGate, SoftWeightTrustGate,
   and TemperatureSoftLabel all consume the final combined heatmap $H$ which
   remains in [0,1]. No architectural changes needed beyond the prior.

### Decision 5: Add entropy + cornerness to the structural prior

- **Date**: 2026-07-10 (pre-implementation research phase)
- **Rationale**:
  1. Edge density + variance miss occlusion boundary signatures
     (multi-directional gradients) → cornerness fills this gap
  2. Edge density + variance underestimate vegetation complexity
     difference (GTA5 vs. Cityscapes) → entropy fills this gap
  3. Three identified failure modes (occlusion, vegetation, complex
     background) each map cleanly to specific cues
  4. Adding cues NOW, before precomputation, costs zero extra compute
     time since heatmaps haven't been generated yet
  5. Separate component storage makes the cost of adding cues purely
     additive (no re-precompute for arbitrary weight combinations)
- **Alternatives considered**:
  - Single combined heatmap (original design): rejected because ablation
    would require full re-precomputation per weight variant
  - Multi-scale variance: rejected because it still operates in variance's
    second-moment paradigm and misses distributional information
  - Learned prior (small network): deferred until classical 4-cue prior
    is evaluated
  - Sobel gradient magnitude: could replace Canny in future but cornerness
    provides strictly more information (direction counts)
- **Constitutional check**: All 4 cues derived from raw RGB pixels using
  classical deterministic CV operations — no learned parameters, no target
  labels. ✅ PASS
- **Impact on existing design**: Downstream modules (trust gates, soft-label)
  are completely unaffected. Only the structural prior module and
  precomputation script change. The dataloader must now load 4 files per
  image and combine them.

## Decision Log

### Decision 1: Soft-weighting over hard-gating

- **Date**: During critical analysis (before diagnostic test)
- **Rationale**: Confounding risk (R1), modest effect size expected
- **Alternatives considered**: Hard-gating only, no weighting
- **Evidence**: Diagnostic test later confirmed modest effect size,
  supporting this decision

### Decision 2: Canny + Variance prior

- **Date**: Initial design
- **Rationale**: Classical CV, computationally free, interpretable,
  no training required
- **Alternatives considered**: Learned prior, Sobel, multi-scale
- **Evidence**: Diagnostic test showed the prior captures meaningful
  variation (error rate ranges from 2.89% to 4.70% across bins).
  However, binary Canny creates a split at S=0.5 that complicates
  the distribution.

### Decision 3: Per-image normalisation

- **Date**: Initial design
- **Rationale**: Auto-adapts to different image types, ensures
  consistent within-image coverage, avoids need for dataset statistics
- **Trade-off**: Cross-image bin comparisons are imprecise
- **Reaffirmed after**: Diagnostic test confirmed this is acceptable
  for the soft-weighting application

### Decision 4: Include DAFormer experiments

- **Date**: After critical analysis
- **Rationale**: Without DAFormer experiments, the paper cannot claim
  relevance to the current SOTA. DAFormer is the strongest UDA baseline
  and the natural host for SATG integration.
- **Cost**: 4 additional experiments × 5 seeds = 20 training runs
- **Justification**: Necessary for publication-quality results

## Open Questions

1. **Should we test on Synthia→Cityscapes as well?**
   - Pro: Demonstrates generalisation beyond GTA
   - Con: 2× the experiments
   - Decision: Deferred — do GTA→CS first, add Synthia if time permits

2. **Should we learn the structural prior end-to-end?**
   - Pro: Could achieve stronger correlation
   - Con: Loses interpretability, adds complexity
   - Decision: Deferred — do fixed prior first, learn if results are weak

3. **What is the optimal weight function?**
   - Need to run an ablation study
   - Candidates: piecewise linear, sigmoid, power, step
   - Decision: Test in ablation experiments (US-04b scope)
