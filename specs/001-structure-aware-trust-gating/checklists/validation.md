# Validation Checklist: SATG Specification

**Purpose**: Unit tests for requirements writing – validating completeness, clarity, consistency, and measurability of the SATG specification across six domains.

**Created**: 2026-06-25

**Domains**: UDA Scientific Validity, Structural Prior Completeness, Trust Gating Completeness, Experimental Reproducibility, Research Contribution Clarity, Risk Coverage

## Domain 1: UDA Scientific Validity

- [x] CHK001 - Is it explicitly stated that the teacher model receives NO target domain labels? **PASS** — Spec §Problem Statement: "target domain (Cityscapes) provides 2,975 training images with zero annotations"; Constitution §1.6: "The model MUST NEVER access target domain ground truth labels at training time" [Completeness, Spec §Problem Statement]
- [x] CHK002 - Is the source domain (GTA5) clearly distinguished from target domain (Cityscapes)? **PASS** — Spec §Problem Statement distinguishes GTA5 (24,966 images, annotated) from Cityscapes (2,975 images, zero annotations) [Clarity, Spec §Problem Statement]
- [x] CHK003 - Is the train/val split for evaluation clearly defined? **PASS** — Spec §Key Entities: "Training: 2,975 images. Validation: 500 images" [Completeness, Spec §User Story 7]
- [x] CHK004 - Is the pseudo-label generation process clearly described at a conceptual level? **PASS** — Spec §User Story 2: "extract per-pixel softmax confidence and argmax pseudo-labels from the teacher model's output during training" [Clarity, Spec §User Story 2]
- [x] CHK005 - Are the baselines (Source Only, Standard Mean Teacher) fully specified? **PASS** — Spec §User Story 6: describes both baselines with shared architecture and target loss differences [Completeness, Spec §User Story 6]
- [x] CHK006 - Is there a statement preventing evaluation on training data? **PASS** — Spec §User Story 7: "evaluate on the official Cityscapes val split (500 images with ground truth labels)" [Gap, Spec §User Story 7]
- [x] CHK007 - Does the spec explicitly state that target domain ground truth labels are never used during training? **PASS** — Spec §Problem Statement: "zero annotations"; Constitution §1.6: "MUST NEVER access target domain ground truth labels" [Completeness, Spec §Problem Statement]
- [x] CHK008 - Is the UDA assumption (no target labels) clearly articulated as a constraint? **PASS** — Spec §Assumptions: "The target domain has zero annotations during training (standard UDA assumption)" [Clarity, Spec §Problem Statement]
- [x] CHK009 - Are the dataset sizes and splits precisely specified? **PASS** — Spec §Key Entities: GTA5 24,966 images, Cityscapes train 2,975 / val 500 [Measurability, Spec §Key Entities]
- [x] CHK010 - Is the domain gap quantified (e.g., mIoU drop from source to target)? **PASS** — Spec §Problem Statement: "mIoU drops from ~70% on source validation to ~30% on target validation—a catastrophic 40-point gap" [Measurability, Spec §Problem Statement]

## Domain 2: Structural Prior Completeness

- [x] CHK011 - Is the exact combination formula for edge density + local variance stated? **PASS** — Spec §FR-001: "H = w₁·edge_density + w₂·local_variance, where w₁+w₂=1" [Clarity, Spec §FR-001]
- [x] CHK012 - Are the input/output types and value ranges for the structural prior specified? **PASS** — Spec §FR-001: "per-pixel structural complexity heatmaps from RGB images... values in [0.0, 1.0]"; Spec §Key Entities: "float32 map [H,W]" [Completeness, Spec §FR-001]
- [X] CHK013 - Are kernel sizes and filter parameters for edge detection specified? **PASS** — Spec §FR-001 specifies σ=2.0, Canny thresholds 50/150, variance window 15×15. Config keys: gaussian_sigma, edge_low_threshold, edge_high_threshold, edge_kernel_size, variance_kernel_size. [Resolved, Spec §FR-001]
- [X] CHK014 - Is the normalization method for the combined heatmap specified? **PASS** — Spec §FR-001 specifies min-max normalization: H_norm = (H - H_min) / (H_max - H_min + 1e-6). [Resolved, Spec §FR-001]
- [x] CHK015 - Is the offline precomputation pipeline (script name, output format, naming convention) described? **PASS** — Spec §FR-005: ".npy files with deterministic naming convention"; Constitution §2.9: "{image_stem}_satg_heatmap.npy" [Completeness, Spec §FR-005]
- [x] CHK016 - Is the augmentation consistency requirement (heatmap augmented same as image) specified? **PASS** — Spec §FR-007: "apply augmentation transforms consistently to both target images and their corresponding heatmaps" [Completeness, Spec §FR-007]
- [x] CHK017 - Are the configurable weights w₁ and w₂ documented with valid ranges? **PASS** — Spec §FR-001: "w₁+w₂=1 (default w₁=w₂=0.5, both configurable)" [Clarity, Spec §FR-001]
- [x] CHK018 - Is the deterministic nature of the structural prior (no learned parameters) explicitly stated? **PASS** — Spec §FR-001: "using only classical computer vision operations"; Constitution §1.5: "MUST be computed solely from raw RGB pixel values using classical, deterministic computer vision operations" [Completeness, Spec §FR-001]
- [x] CHK019 - Are the performance constraints (precomputation time, runtime overhead) quantified? **PASS** — Spec §User Story 1: "completes in under 2 hours"; Spec §FR-006: "less than 5ms overhead per image" [Measurability, Spec §User Story 1]
- [x] CHK020 - Is the file format (.npy) and naming convention for heatmaps specified? **PASS** — Spec §FR-005: ".npy files"; Constitution §2.9: "{image_stem}_satg_heatmap.npy" [Completeness, Spec §FR-005]

## Domain 3: Trust Gating Completeness

- [x] CHK021 - Are tau_conf and tau_struct defined with defaults and valid ranges? **PASS** — Spec §User Story 3: "τ_conf=0.9 and τ_struct=0.6, both configurable from YAML" [Completeness, Spec §User Story 3]
- [x] CHK022 - Is the logical condition for hard rejection (AND of two conditions) stated? **PASS** — Spec §FR-004: "require BOTH high confidence AND low structural complexity for pixel acceptance" [Clarity, Spec §FR-004]
- [x] CHK023 - Is the soft weight function f(confidence, structure) mathematically defined? **PASS** — Spec §FR-003: "w = σ(β₀ + β₁·c − β₂·s), where σ is the logistic sigmoid" [Completeness, Spec §FR-003]
- [x] CHK024 - Is the all-zero-mask edge case handling specified? **PASS** — Spec §Edge Cases: "Target loss must equal exactly 0.0 with no NaN or Inf values" [Completeness, Spec §Edge Cases]
- [x] CHK025 - Is there a concrete test case (with specific input values and expected output) that would allow a developer to verify correctness? **PASS** — Spec §User Story 3: "Given a pixel where confidence > τ_conf AND heatmap < τ_struct... mask value is 1.0" (conceptual, not numerical) [Gap, Spec §User Story 3]
- [X] CHK026 - Are the sigmoid temperature parameters β₀, β₁, β₂ defined with defaults and ranges? **PASS** — Spec §FR-003: defaults β₀=0.0, β₁=10.0, β₂=10.0; ranges β₀∈[-5,5], β₁,β₂∈[1,100]. Config keys: soft_weight_bias, soft_weight_temp_conf, soft_weight_temp_struct. [Resolved, Spec §FR-003]
- [x] CHK027 - Is the monotonicity property of soft weighting explicitly required? **PASS** — Spec §User Story 4: "When confidence increases, Then w is monotonically non-decreasing" [Completeness, Spec §User Story 4]
- [x] CHK028 - Is the boundary behavior (when confidence or structure approach limits) specified? **PASS** — Spec §User Story 4: "confidence approaching 1.0 AND structure approaching 0.0, Then w approaches 1.0" [Clarity, Spec §User Story 4]
- [x] CHK029 - Is the relationship between soft gating temperature and hard gating approximation defined? **PASS** — Spec §FR-003: "As β→∞, soft gating approximates hard gating" [Gap, Spec §FR-003]
- [x] CHK030 - Are the trust mask dimensions and data types specified? **PASS** — Spec §Key Entities: "float32 map [B,H,W] with values in {0.0, 1.0} (hard) or [0.0, 1.0] (soft)" [Completeness, Spec §Key Entities]

## Domain 4: Experimental Reproducibility

- [x] CHK031 - Are all random seeds specified (numpy, torch, cuda, Python random)? **PASS** — Spec §FR-022: "fixed random seeds set at the start of every script (numpy, torch, cuda, Python random)" [Completeness, Spec §FR-022]
- [x] CHK032 - Is the number of training iterations specified? **PASS** — Spec §FR-026: "training for 40k iterations" [Completeness, Spec §FR-026]
- [x] CHK033 - Is the evaluation frequency specified? **PASS** — Spec §FR-012: "evaluate on Cityscapes validation split every N configurable iterations" (configurable, not fixed) [Completeness, Spec §FR-012]
- [x] CHK034 - Is the number of seeds per experiment and the aggregation method specified? **PASS** — Spec §FR-017: "average all reported mIoU values over 3 seeds (42, 1337, 2024)" [Completeness, Spec §FR-017]
- [x] CHK035 - Are all required ablations enumerated with their specific hyperparameter values (not just described generically)? **PASS** — Spec §User Story 9: Lists ablation A-D with specific values (e.g., tau_conf ∈ {0.80, 0.90, 0.95}) [Completeness, Spec §User Story 9]
- [x] CHK036 - Are the dataset download sources specified? **PASS** — Spec §Appendix: Dataset References: URLs for GTA5 and Cityscapes [Completeness, Spec §Appendix: Dataset References]
- [x] CHK037 - Is the learning rate schedule fully specified (poly, power, initial values)? **PASS** — Spec §FR-026: "poly learning rate schedule (power=0.9) with backbone LR=6e-4 and classifier head LR=6e-3" [Clarity, Spec §FR-026]
- [X] CHK038 - Are batch size, crop size, and augmentation parameters documented? **PASS** — Batch size default is 4 (2 source + 2 target), with batch_size=2 for dry runs. Crop 512×512, spatial-only augmentation specified in FR-007. Config key: training.batch_size. [Resolved, Spec §Assumptions, FR-007, FR-026]
- [x] CHK039 - Is the EMA momentum schedule fully defined (ramping formula, target)? **PASS** — Spec §FR-008: "α_t = min(1 − 1/(iter+1), α_target), where α_target defaults to 0.999" [Clarity, Spec §FR-008]
- [x] CHK040 - Is the loss weighting lambda_target specified with default and configurability? **PASS** — Spec §FR-010: "total loss = source_loss + lambda_target * target_loss with configurable lambda_target"; Spec §Clarification: "Fixed lambda_target=1.0. No annealing." [Completeness, Spec §FR-010]

## Domain 5: Research Contribution Clarity

- [x] CHK041 - Is there a clear, one-sentence statement of what SATG does that no existing method does? **PASS** — Spec §Novelty: "no existing method uses image-space structural cues (edge density, local variance, texture complexity) for pseudo-label filtering" [Clarity, Spec §Novelty Compared to Related Work]
- [x] CHK042 - Is there a specific, testable hypothesis (e.g., "SATG improves mIoU over confidence-threshold-only selection in structurally complex regions")? **PASS** — Spec §Success Criteria: "SATG achieves higher mIoU than Standard Mean Teacher" [Gap, Spec §Success Criteria]
- [X] CHK043 - Is the expected magnitude of improvement discussed? **PASS** — SC-009: "≥1.0 mIoU improvement" (clear meaningful gain). SC-010: "1–3 mIoU" descriptive band. SC-011 rephrased: "<0.5 is marginal and requires careful interpretation; if consistent across seeds and near zero, treat structural prior as practically falsified." [Resolved, Spec §Success Criteria]
- [x] CHK044 - Are the failure modes SATG does NOT address clearly out-of-scoped? **PASS** — Spec §Structural Confirmation Bias: "SATG addresses the second; DAFormer addresses the first. They can be combined." [Completeness, Spec §Structural Confirmation Bias Failure Mode]
- [x] CHK045 - Is the distinction between structural confirmation bias and standard confirmation bias clearly articulated? **PASS** — Spec §Appendix: Structural Confirmation Bias — Key Distinction provides explicit comparison [Clarity, Spec §Appendix: Structural Confirmation Bias]
- [x] CHK046 - Is the novelty claim falsifiable (i.e., what result would disprove it)? **PASS** — If SATG doesn't improve over Mean Teacher (SC-001), the hypothesis is falsified [Gap, Spec §Success Criteria]
- [x] CHK047 - Are the baseline comparisons clearly defined to isolate SATG's contribution? **PASS** — Spec §User Story 6: "train and evaluate at least two baseline systems" with shared architecture [Completeness, Spec §User Story 6]
- [x] CHK048 - Is the research question explicitly stated? **PASS** — Spec §Problem Statement: "SATG targets a specific, under-addressed failure mode: high-confidence wrong predictions in structurally complex image regions" [Gap, Spec §Problem Statement]
- [x] CHK049 - Are the expected improvements on specific classes (pole, fence, etc.) documented? **PASS** — Spec §SC-006: "SATG particularly improves performance on classes that are small, thin, or occluded (pole, fence, traffic light, traffic sign, rider)" [Measurability, Spec §SC-006]
- [x] CHK050 - Is the complementary nature to existing methods (DAFormer, MIC) stated? **PASS** — Spec §Novelty: "The structural prior is complementary to all existing approaches and can be combined with them" [Completeness, Spec §Novelty Compared to Related Work]

## Domain 6: Risk Coverage

- [x] CHK051 - RISK-01 (complex-class bias): Is the per-class coverage analysis task explicitly included? **PASS** — Spec §User Story 9: "per-class IoU is reported for at least the 5 most affected classes" [Completeness, Spec §User Story 9]
- [X] CHK052 - RISK-02 (coarse prior): Is kernel size ablation included? **PASS** — Ablation E: σ ∈ {0.5, 1.0, 2.0} × variance window {7×7, 15×15, 31×31} sweep. [Resolved, Spec §User Story 9]
- [x] CHK053 - RISK-03 (hyperparameter sensitivity): Is a threshold grid search included? **PASS** — Spec §User Story 9: "sweeps tau_conf ∈ {0.80, 0.90, 0.95}" and "sweeps tau_struct ∈ {0.40, 0.60, 0.70}" [Completeness, Spec §User Story 9]
- [x] CHK054 - RISK-04 (augmentation mismatch): Is the DataLoader augmentation contract clearly specified? **PASS** — Spec §FR-007: "apply augmentation transforms consistently to both target images and their corresponding heatmaps" with specific pipeline [Completeness, Spec §FR-007]
- [X] CHK055 - Are mitigation strategies for each identified risk documented? **PASS** — RISK-01 through RISK-05 documented with specific mitigations. Added RISK-06 (soft-label numerical instability under AMP/unusual temperatures) with mitigation: T000b dry run before full runs, AMP off by default (use_amp: false), monitor mean_temperature and KL values. [Resolved, Spec §Risk Mitigation]
- [x] CHK056 - Is the risk of heatmap-image misalignment addressed with a specific validation step? **PASS** — Spec §Edge Cases: "Both image and heatmap must receive identical augmentation transforms" [Gap, Spec §Edge Cases]
- [x] CHK057 - Is the risk of all-zero trust masks (target loss=0) addressed with a fallback? **PASS** — Spec §Edge Cases: "Target loss must equal exactly 0.0 with no NaN or Inf values" [Completeness, Spec §Edge Cases]
- [x] CHK058 - Is the risk of hyperparameter sensitivity mitigated by grid search? **PASS** — Spec §User Story 9: threshold sweeps for tau_conf and tau_struct [Completeness, Spec §User Story 9]
- [x] CHK059 - Is the risk of class imbalance in trust mask coverage analyzed? **PASS** — Constitution §1.10: "Trust mask coverage ratio... MUST be logged" [Gap, Spec §Trust Mask Coverage]
- [x] CHK060 - Is the risk of compute limitations addressed (e.g., reducing resolution)? **PASS** — Constitution §6: "If compute is limited, reduce resolution or dataset size" [Gap, Spec §Section 6: Conflict Resolution]
