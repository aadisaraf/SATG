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
