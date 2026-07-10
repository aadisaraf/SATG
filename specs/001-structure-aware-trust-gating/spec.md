# Feature Specification: Structure-Aware Trust Gating (SATG)

**Feature Branch**: `001-structure-aware-trust-gating`

**Created**: 2026-06-24

**Status**: Draft

**Input**: User description: "Deploy the following parallel subagents and do not proceed until all four have reported their findings: SUBAGENT A — PROBLEM DOMAIN BACKGROUND: Search academic papers and documentation to answer precisely: 1. What is Unsupervised Domain Adaptation (UDA) for semantic segmentation? 2. What is the teacher-student framework in UDA? How does EMA (Exponential Moving Average) update work? 3. What is confirmation bias in pseudo-label UDA? 4. What are the standard 19 Cityscapes classes? 5. What are the accepted evaluation metrics? SUBAGENT B — RELATED WORK ON PSEUDO-LABEL NOISE: 1. How do DAFormer, MIC, HRDA, and ProDA handle pseudo-label noise? 2. Do ANY existing UDA methods use image-space structural cues? 3. What are the known weaknesses of confidence-threshold-only pseudo-label selection? 4. What is the 'structural confirmation bias' failure mode? SUBAGENT C — DATASETS AND EVALUATION SETUP: 1. GTA5 dataset details 2. Cityscapes dataset details 3. GTA5→Cityscapes 19-class label mapping 4. Standard image crops 5. Standard data augmentation. SUBAGENT D — STRUCTURAL CUES IN COMPUTER VISION: 1. Edge density as a structural feature 2. Local variance computation 3. Other classical image features 4. How existing papers use structural maps 5. What 'structural complexity' looks like in Cityscapes. After ALL four subagents have reported, synthesize the findings and write the complete specification for the SATG system."

## Problem Statement

### The UDA Segmentation Challenge

Unsupervised Domain Adaptation (UDA) for semantic segmentation addresses a fundamental practical problem: training accurate pixel-level classifiers requires expensive manual annotations, but deploying models in new environments introduces domain shift that degrades performance. In the canonical GTA5→Cityscapes benchmark, a model trained on synthetic GTA5 game engine imagery must transfer to real-world dashcam footage from European cities. The source domain (GTA5) provides 24,966 fully annotated images rendered from a game engine, while the target domain (Cityscapes) provides 2,975 training images with zero annotations. The domain gap manifests across appearance (synthetic vs. real textures), lighting (game engine vs. camera sensors), geometry (viewpoint distributions), and class frequency (different object distributions). Without adaptation, mIoU drops from ~70% on source validation to ~30% on target validation—a catastrophic 40-point gap. The teacher model NEVER accesses target domain ground truth labels during training; this is the fundamental UDA assumption.

### The Teacher-Student Framework and Confirmation Bias

The dominant paradigm for UDA segmentation uses a teacher-student framework with Exponential Moving Average (EMA) weight updates. A student network is trained on source labels and target pseudo-labels; a teacher network—an EMA copy of the student—generates stable pseudo-labels for unlabeled target images. The EMA update (θ_t ← α·θ_t + (1-α)·θ_s, typically α=0.999) produces temporal ensembling that smooths noisy predictions. However, this creates a critical failure mode: confirmation bias. When the teacher generates incorrect pseudo-labels, the student learns these errors, and the EMA update incorporates them back into the teacher. Over iterations, this self-reinforcing feedback loop amplifies initially small errors into systematic biases—particularly in regions where the teacher is confidently wrong.

### The Structural Confirmation Bias Failure Mode

SATG targets a specific, under-addressed failure mode: high-confidence wrong predictions in structurally complex image regions that escape confidence thresholding. Confidence thresholding (keeping pseudo-labels where max softmax probability > τ) fails because softmax confidence measures the network's belief, not actual correctness. In structurally complex regions—object boundaries, dense vegetation, cluttered intersections, thin structures (poles, fences, traffic signs)—the teacher can be systematically overconfident about wrong classes. The network's feature extraction is inadequate for fine-grained spatial discrimination in these regions, but global context still produces high-confidence (wrong) answers. These errors are spatially coherent, so the EMA update smooths and amplifies them over iterations. This is distinct from standard confirmation bias (addressed by DAFormer's Rare Class Sampling, which is class-frequency-dependent). Structural confirmation bias is spatial/texture-dependent and is not addressed by any existing method.

### Why a Structural Prior Is the Right Solution

The key insight is that image-space structural complexity—edge density, local variance, texture patterns—provides an independent, complementary signal to model confidence. Regions where the teacher is likely to be wrong can be identified *before* pseudo-labels are generated, simply by measuring how structurally complex the local image neighborhood is. A structural prior computed from raw RGB pixels using classical computer vision operations (Canny edges, sliding-window variance, normalization) requires no learned parameters, no target labels, and no additional model inference. It is deterministic, precomputable, and adds negligible runtime overhead. This makes it both practically convenient and scientifically valid within the UDA assumption that no target labels are available.

### Novelty Compared to Related Work

The structural prior approach is novel. Extensive search across the UDA semantic segmentation literature confirms that no existing method uses image-space structural cues (edge density, local variance, texture complexity) for pseudo-label filtering. The closest prior work is ProDA (Zheng et al., CVPR 2021), which uses prototypical feature-space distances for denoising—but this operates in feature space, not image space. DAFormer uses Rare Class Sampling (class-frequency-based), MIC uses Masked Image Consistency (output-space consistency), and HRDA uses multi-resolution training (architectural). None use pixel-level image structural features as a pseudo-label confidence modulator. The structural prior is complementary to all existing approaches and can be combined with them.

### Research Question

Does incorporating image-space structural complexity as a trust modulator improve pseudo-label quality and downstream mIoU in UDA semantic segmentation?

## User Scenarios & Testing

### User Story 1 - Structural Prior Computation (Priority: P1)

As a researcher, I want to compute a per-pixel structural complexity heatmap for any target-domain RGB image using only classical computer vision operations, so that each pixel has a score in [0,1] indicating how structurally complex its local neighborhood is relative to the image.

**Why this priority**: The structural prior is the foundation of SATG. Without reliable heatmaps, the trust gating mechanism has no input signal. This must work correctly before any downstream components can be validated.

**Independent Test**: Can be fully tested by running the precomputation script on Cityscapes training images and verifying that output heatmaps have correct dimensions, value ranges, and expected patterns (low scores on sky/road, high scores on vegetation/intersections).

**Acceptance Scenarios**:

1. **Given** a Cityscapes RGB image at any resolution, **When** the structural prior module processes it, **Then** the output is a float32 heatmap of identical H×W dimensions with all values in [0.0, 1.0]
2. **Given** a sky region or flat road surface, **When** the mean heatmap score is computed, **Then** the mean score is less than 0.30
3. **Given** dense vegetation, cluttered backgrounds, or mixed foreground objects, **When** the mean heatmap score is computed, **Then** the mean score is greater than 0.60
4. **Given** the computation pipeline, **When** operations are inspected, **Then** ONLY these operations are used: color space conversion, Gaussian blur, Canny edge detection, sliding-window operations, linear weighting, normalization—no neural networks, no learned parameters
5. **Given** all 2,975 Cityscapes training images, **When** precomputation runs on a single CPU core, **Then** it completes in under 2 hours (or under 20 minutes with 8 cores)
6. **Given** a precomputed heatmap file, **When** loaded during training, **Then** it adds less than 5ms per image overhead

---

### User Story 2 - Teacher Confidence Extraction (Priority: P2)

As a researcher, I want to extract per-pixel softmax confidence and argmax pseudo-labels from the teacher model's output during training, without gradient computation, so that they can be used as inputs to the trust gating module.

**Why this priority**: The teacher's confidence and pseudo-labels are the other input to the trust gate. Without clean extraction, the gating mechanism cannot function.

**Independent Test**: Can be tested by passing random logits through the extraction function and verifying output shapes, value ranges, and absence of gradient computation.

**Acceptance Scenarios**:

1. **Given** teacher logits [B, C, H, W] with C=19, **When** extraction runs, **Then** outputs are confidence map [B, H, W] (max softmax probability per pixel) and pseudo-label map [B, H, W] (argmax class per pixel)
2. **Given** the softmax computation, **When** confidence values are inspected, **Then** all values are in (0, 1) and the sum-to-1 constraint holds across the class dimension before taking max
3. **Given** the extraction operation, **When** it executes, **Then** all operations use torch.no_grad() with zero gradient computation
4. **Given** input spatial dimensions [H, W], **When** outputs are produced, **Then** spatial resolution matches input exactly

---

### User Story 3 - Trust Gating: Hard Rejection (Priority: P3)

As a researcher, I want to generate a binary trust mask that accepts only pseudo-labels where the teacher is both highly confident AND the pixel is in a structurally simple region, so that only reliably labeled pixels contribute to the student's target-domain loss.

**Why this priority**: Hard rejection is the simpler, more interpretable gating mechanism. It establishes the core SATG mechanism and serves as the primary experimental variant.

**Independent Test**: Can be tested with synthetic confidence maps and structural heatmaps, verifying that the mask is 1.0 only where both conditions hold, and 0.0 elsewhere.

**Acceptance Scenarios**:

1. **Given** confidence map [B,H,W] and structural heatmap [B,H,W], **When** trust gating runs, **Then** output is binary float mask [B,H,W] where 1.0=trusted and 0.0=rejected
2. **Given** a pixel where confidence > τ_conf AND heatmap < τ_struct, **When** the mask is computed, **Then** the mask value is 1.0
3. **Given** a pixel where confidence ≤ τ_conf OR heatmap ≥ τ_struct, **When** the mask is computed, **Then** the mask value is 0.0
4. **Given** default thresholds, **When** the config is inspected, **Then** τ_conf=0.9 and τ_struct=0.6, both configurable from YAML without code changes. Valid ranges: tau_conf ∈ (0, 1), tau_struct ∈ (0, 1).
5. **Given** an entire batch where the mask is zero, **When** target loss is computed, **Then** the loss equals exactly 0.0 with no NaN or Inf values
6. **Given** the mask and per-pixel cross-entropy loss, **When** loss is aggregated, **Then** the mask is applied element-wise before aggregation (not to logits or probabilities)
7. **Given** confidence=0.95, structure=0.3, **When** hard rejection runs, **Then** mask=1.0; confidence=0.85, structure=0.3 → mask=0.0; confidence=0.95, structure=0.7 → mask=0.0

---

### User Story 4 - Trust Gating: Soft-Weight Variant (Priority: P4)

As a researcher, I want an alternative to hard rejection where pseudo-labels receive continuous trust weights rather than binary accept/reject decisions, so that the student's supervision signal degrades smoothly with structural complexity rather than cutting off abruptly.

**Why this priority**: Soft weighting provides a complementary mechanism that may outperform hard rejection by avoiding information loss at threshold boundaries. It enables ablation comparison between hard and soft gating.

**Independent Test**: Can be tested by verifying monotonicity properties and boundary behaviors with synthetic inputs.

**Acceptance Scenarios**:

1. **Given** confidence map [B,H,W] and structural heatmap [B,H,W], **When** soft weighting runs, **Then** output is continuous weight map w[B,H,W] with all values in [0.0, 1.0]
2. **Given** fixed structural complexity, **When** confidence increases, **Then** w is monotonically non-decreasing
3. **Given** fixed confidence, **When** structural complexity increases, **Then** w is monotonically non-increasing
4. **Given** confidence approaching 1.0 AND structure approaching 0.0, **When** w is computed, **Then** w approaches 1.0
5. **Given** confidence approaching 0.0 OR structure approaching 1.0, **When** w is computed, **Then** w approaches 0.0
6. **Given** very steep temperature parameters, **When** soft weighting is computed, **Then** it approximates hard rejection behavior

---

### User Story 4b - Trust Gating: Soft-Label Variant — Temperature-Scaled Pseudo-Labels (Priority: P4b)

As a researcher, I want an alternative trust mechanism that softens the teacher's predicted class distribution itself — not just the loss weight — so that the student is trained on "this is probably class X, but I'm not certain" rather than a hard one-hot label scaled down in importance. This tests whether softening the supervision target (distributional uncertainty) is more effective than softening the supervision weight (US-04's mechanism).

**Why this priority**: This variant isolates a fundamentally different axis of softening — target distribution vs. loss weight — enabling a direct comparison of which mechanism is more effective for pseudo-label denoising.

**Independent Test**: Can be tested by verifying that soft targets are valid probability distributions summing to 1.0 per pixel, temperature scaling is monotonic with structural complexity, and KL divergence/soft cross-entropy loss computes correctly.

**Acceptance Scenarios**:

1. **Given** teacher logits [B,C,H,W] and structural heatmap [B,H,W], **When** soft-label computation runs, **Then** output is a soft target distribution [B,C,H,W] where each pixel's C-length vector sums to 1.0 (still a valid probability distribution)
2. **Given** the softening temperature T, **When** it is computed per-pixel as a function of structural complexity, **Then** T[b,h,w] = 1.0 + k * structural_heatmap[b,h,w], where k is a configurable scaling constant (default k=4.0)
3. **Given** structural_heatmap=0 (structurally simple regions), **When** temperature is computed, **Then** T=1.0 (no softening); the teacher's original confident distribution is preserved
4. **Given** structural_heatmap>0, **When** temperature is computed, **Then** T>1.0 (softened, flatter distribution); softening increases monotonically with structural complexity
5. **Given** teacher logits and per-pixel T values, **When** soft target is computed, **Then** it is softmax(teacher_logits / T), applied per-pixel with per-pixel T values (NOT a single scalar T for the whole image)
6. **Given** the soft target and student predictions, **When** target loss is computed, **Then** it uses a distributional loss (KL divergence or soft cross-entropy) between the student's predicted distribution and the soft target distribution — NOT a hard-label cross-entropy
7. **Given** confidence gating as a pre-filter, **When** pixels have teacher confidence < tau_conf, **Then** they contribute zero loss regardless of temperature (preserves the "teacher must be at least somewhat confident" requirement from the original trust gating philosophy)
8. **Given** tau_conf gating excludes all pixels in a batch, **When** loss is computed, **Then** loss=0.0 with no NaN/Inf (same safety contract as existing SATGLoss)
9. **Given** documentation, configs, and code, **When** this variant is referenced, **Then** it must be clearly distinguished from US-04 (SATG Soft-Weight), which instead scales the loss magnitude while keeping the target a hard one-hot label

---

### User Story 5 - Complete UDA Training Pipeline (Priority: P5)

As a researcher, I want to run a complete UDA training loop that processes labeled source batches and structural-heatmap-annotated target batches simultaneously, updates the teacher via EMA, and applies SATG trust gating to all target pseudo-labels.

**Why this priority**: The training pipeline integrates all components and enables the actual experiments. Without it, the individual modules cannot be validated in context.

**Independent Test**: Can be tested with a dry run (10 images, 10 iterations) to verify loss computation, EMA updates, and logging without errors.

**Acceptance Scenarios**:

1. **Given** a training step, **When** the pipeline runs, **Then** it processes one source batch (images+labels) from GTA5 and one target batch (images+precomputed heatmaps) from Cityscapes
2. **Given** source images and labels, **When** source loss is computed, **Then** it equals cross-entropy(student(source_img), source_labels)
3. **Given** target images, teacher pseudo-labels, trust weights, and heatmaps, **When** target loss is computed, **Then** it uses SATGLoss with the trust gate and precomputed heatmap
4. **Given** source and target losses, **When** total loss is computed, **Then** total_loss = source_loss + lambda_target * target_loss with configurable lambda_target
5. **Given** a completed training step, **When** teacher update runs, **Then** teacher weights are updated via EMA from student weights
6. **Given** a training step, **When** logging runs, **Then** total_loss, source_loss, target_loss, and trust_coverage_ratio are logged
7. **Given** training progress, **When** evaluation interval is reached, **Then** evaluation runs on Cityscapes validation split
8. **Given** validation mIoU, **When** a new best is found, **Then** the checkpoint is saved automatically

---

### User Story 6 - Baseline Comparisons (Priority: P6)

As a researcher, I want to train and evaluate at least two baseline systems (Source Only and Standard Mean Teacher) using identical infrastructure, so that the effect of SATG can be isolated and fairly compared.

**Why this priority**: Fair comparison requires identical infrastructure. Without baselines, SATG's contribution cannot be quantified.

**Independent Test**: Can be verified by running all three configurations and confirming identical backbone, optimizer, and source preprocessing across runs.

**Acceptance Scenarios**:

1. **Given** Source Only configuration, **When** training runs, **Then** it uses identical architecture, identical source training, and zero target pseudo-labeling—representing the lower bound
2. **Given** Standard Mean Teacher configuration, **When** training runs, **Then** it uses EMA teacher-student with confidence threshold only (tau_conf), no structural prior, no heatmap loading
3. **Given** all three configurations, **When** configs are inspected, **Then** they share the same backbone, same optimizer, same source data preprocessing—only the target loss mechanism differs
4. **Given** final results, **When** evaluation completes, **Then** all results are averaged over 3 seeds (42, 1337, 2024)

---

### User Story 7 - Evaluation and Reporting (Priority: P7)

As a researcher, I want to compute and report mIoU and per-class IoU on the Cityscapes validation set for all configurations, so that results can be compared against published baselines.

**Why this priority**: Quantitative evaluation is the primary output of the research. Without standardized metrics, results cannot be compared or published.

**Evaluation Model**: All mIoU is evaluated using the **student model** (not the EMA teacher). The teacher is used only for pseudo-label generation during training. This matches DAFormer/MIC/HRDA convention.

**Independent Test**: Can be verified by running evaluation on a checkpoint and comparing mIoU against known DAFormer/SOTA values.

**Acceptance Scenarios**:

1. **Given** a trained model, **When** evaluation runs, **Then** it evaluates on the official Cityscapes val split (500 images with ground truth labels). Evaluation MUST be performed only on the Cityscapes validation split. Evaluation on training images is prohibited.
2. **Given** evaluation output, **When** per-class results are computed, **Then** per-class IoU is reported for all 19 standard classes
3. **Given** per-class IoU values, **When** overall mIoU is computed, **Then** it equals the mean over all 19 classes
4. **Given** evaluation images, **When** pixels with label=255 are encountered, **Then** they are excluded from IoU computation
5. **Given** results, **When** the results table is formatted, **Then** it follows the EXPERIMENTS.md format with columns: Method | mIoU (mean±std) | per-class IoU columns

---

### User Story 8 - Visualization of Trust Gating (Priority: P8)

As a researcher, I want visual comparisons showing which pixels SATG trusts vs. rejects, alongside the teacher confidence and structural heatmap, so that I can understand, explain, and present the mechanism intuitively.

**Why this priority**: Visualizations provide qualitative evidence that complements quantitative metrics and are essential for paper figures and presentations.

**Independent Test**: Can be verified by generating visualizations for sample images and confirming all five panels are present and correctly colorized.

**Acceptance Scenarios**:

1. **Given** a selected image, **When** visualization is generated, **Then** a 1×5 panel is produced: (1) Original RGB, (2) Teacher confidence map (colorized 0=dark to 1=bright), (3) Structural heatmap (colorized 0=dark to 1=bright), (4) Trust mask overlaid on image (accepted=colored, rejected=grey), (5) SATG-filtered pseudo-label colorized with Cityscapes palette
2. **Given** the visualization set, **When** images are selected, **Then** at least 10 diverse images are visualized per configuration, including at least 3 examples where SATG rejects high-confidence teacher predictions
3. **Given** visualizations, **When** files are saved, **Then** they are stored as PNG in visualizations/{config_name}/ directory

---

### User Story 9 - Ablation Studies (Priority: P9)

As a researcher, I want systematic ablation experiments varying the structural prior type, trust function design, and threshold values, so that I can isolate the contribution of each design choice.

**Why this priority**: Ablations are mandatory for research integrity (Constitution Section 3.1). They isolate component contributions and validate design decisions.

**Independent Test**: Can be verified by confirming all ablation configurations are defined in YAML and all results appear in EXPERIMENTS.md.

**Acceptance Scenarios**:

1. **Given** ablation A, **When** it runs, **Then** it compares edge-density-only, local-variance-only, and combined structural priors
2. **Given** ablation B, **When** it runs, **Then** it sweeps tau_conf ∈ {0.80, 0.90, 0.95} with tau_struct=0.60 fixed
3. **Given** ablation C, **When** it runs, **Then** it sweeps tau_struct ∈ {0.40, 0.60, 0.70} with tau_conf=0.90 fixed
4. **Given** ablation D, **When** it runs, **Then** it compares hard rejection vs. soft weighting (best of each)
5. **Given** ablation E, **When** it runs, **Then** it varies edge detection kernel σ ∈ {0.5, 1.0, 2.0} and local variance window ∈ {7×7, 15×15, 31×31}
6. **Given** ablation results, **When** EXPERIMENTS.md is updated, **Then** all ablation results are included—no runs hidden or omitted
7. **Given** ablation results, **When** per-class IoU is reported, **Then** it is shown for at least the 5 most affected classes
8. **Given** ablation F — Soft mechanism comparison, **When** it runs, **Then** it compares SATG Soft-Weight vs. SATG Soft-Label, both compared against SATG Hard and Mean Teacher, single seed=42 each, to determine whether weight-scaling or distribution-softening is the more effective soft mechanism
9. **Given** ablation G — Temperature scaling constant, **When** it runs, **Then** it sweeps k ∈ {2.0, 4.0, 6.0} for SATG Soft-Label, single seed=42 each

---

### Edge Cases

- What happens when the entire target batch has zero trusted pixels (all rejected by the trust gate)?
  → Target loss must equal exactly 0.0 with no NaN or Inf values. The student trains only on source loss for that step.
- What happens when the structural heatmap is spatially misaligned with the target image due to augmentation?
  → Both image and heatmap must receive identical augmentation transforms. The DataLoader contract specifies this.
- What happens when tau_struct is set so high that all pixels are trusted?
  → SATG degrades to standard Mean Teacher behavior (confidence-only gating). This is a valid ablation configuration.
- What happens when tau_struct is set so low that no pixels are trusted?
  → Target loss is always 0.0. The model trains as Source Only. This is a degenerate but valid configuration.
- What happens when teacher confidence is exactly at the threshold boundary?
  → Pixels with confidence == tau_conf are rejected (strict inequality required for trust).

### Risk Mitigation

- RISK-01 (complex-class bias): Mitigate via per-class IoU analysis and trust mask coverage per class.
- RISK-02 (coarse prior): Mitigate via kernel size ablation (Ablation E).
- RISK-03 (hyperparameter sensitivity): Mitigate via grid search over tau_conf and tau_struct (Ablations B and C).
- RISK-04 (augmentation mismatch): Mitigate via DataLoader contract requiring identical augmentation for image and heatmap.
- RISK-05 (temperature over-softening): Temperature softening may be too aggressive at high structural complexity, causing the soft target to approach a near-uniform distribution and providing near-zero useful gradient signal. Mitigation: Cap maximum temperature (e.g., T_max=5.0) via config; ablate k sensitivity similarly to tau_conf/tau_struct.
- Validation step: After augmentation, verify heatmap-image alignment by checking that heatmap dimensions match image dimensions and that spatial transforms are consistent.
- Analyze trust mask coverage per class to detect class imbalance in trusted pixels. Log per-class coverage ratios during training.
- If GPU memory is limited, reduce crop size (e.g., 384×384) or use gradient accumulation, but never skip required ablations or the Source Only baseline (per Constitution §6).

## Requirements

### Functional Requirements

- **FR-001**: System MUST compute per-pixel structural complexity heatmaps from RGB images using only classical computer vision operations. The heatmap H is computed as a weighted sum of 4 components: H = w_e·edge_density + w_v·local_variance + w_n·entropy + w_c·cornerness. All 4 weights are configurable from YAML with default 0.25 each and need not sum to 1 (each component is independently normalized to [0,1]). edge_density is the normalized Canny edge map; local_variance is the normalized sliding-window variance; entropy is the normalized local image entropy via rank filtering; cornerness is the normalized minimum eigenvalue of the structure tensor. All weights are configurable from YAML. Edge detection uses Canny with Gaussian blur kernel size σ=2.0 (configurable), low threshold=50, high threshold=150 (configurable). Local variance uses a sliding window of 15×15 pixels (configurable). Entropy uses a disk radius of 7 (configurable) with 32 histogram bins. Cornerness uses structure tensor kernel size 15 and integration sigma 2.0 (both configurable). Normalization uses min-max scaling per component: X_norm = (X - X_min) / (X_max - X_min + ε), where ε=1e-6.
- **FR-002**: System MUST extract per-pixel softmax confidence and argmax pseudo-labels from teacher logits without gradient computation
- **FR-003**: System MUST support three trust gating modes: hard rejection (binary mask), soft weighting (continuous weights), and soft-label (temperature-scaled distributional targets). The soft weight is computed as w = σ(β₀ + β₁·c − β₂·s), where σ is the logistic sigmoid, c is teacher confidence, s is structural heatmap value, and β₀, β₁, β₂ are configurable temperature/bias parameters. As β→∞, soft gating approximates hard gating. Default parameters: β₀=0.0, β₁=10.0, β₂=10.0. Ranges: β₀∈[-5,5], β₁∈[1,100], β₂∈[1,100]. The soft-label mode uses per-pixel temperature scaling T=1+k·s with configurable k (default 4.0) and T_max (default 5.0), producing soft target distributions via softmax(teacher_logits/T) trained with KL divergence loss.
- **FR-004**: Trust gating MUST require BOTH high confidence AND low structural complexity for pixel acceptance (neither alone is sufficient)
- **FR-005**: System MUST precompute and store per-pixel structural heatmap components as .npy files with deterministic naming convention before training. Precomputation script: `precompute/compute_heatmaps.py`. Output format: 4 NumPy .npy files per image, each saved as float32 (H, W) array with values in [0,1]. Naming convention: `{image_stem}_satg_edge.npy`, `{image_stem}_satg_var.npy`, `{image_stem}_satg_ent.npy`, `{image_stem}_satg_corn.npy`, stored alongside source images. The 4 components are combined at load time using configurable weights from `structural_prior` section of the YAML config.
- **FR-006**: System MUST load precomputed heatmaps during training with less than 5ms overhead per image
- **FR-007**: System MUST apply augmentation transforms consistently to both target images and their corresponding heatmaps. The target augmentation pipeline consists of spatial-only transforms: random resize (0.5–2.0×), random crop to 512×512, random horizontal flip. No color jitter, Gaussian blur, or grayscale is applied to target images. All spatial transforms are applied identically to both image and heatmap.
- **FR-007a**: Trust gating MUST operate at full image resolution. Teacher logits are bilinearly upsampled to match the precomputed heatmap resolution before the trust gate is applied. This ensures pixel-to-pixel alignment with the final cross-entropy loss computation.
- **FR-008**: System MUST update teacher weights via EMA after every training step using a scheduled momentum: α_t = min(1 − 1/(iter+1), α_target), where α_target defaults to 0.999 and is configurable. This matches the DAFormer/MIC/HRDA convention, ramping from near 0 to α_target over ~1000 iterations.
- **FR-009**: System MUST process one source batch and one target batch per training step
- **FR-009a**: Pseudo-labeling and trust gating MUST start from iteration 1 with no warmup delay. The scheduled EMA ramp (FR-008) handles early-training instability. There is no source-only warmup phase.
- **FR-010**: System MUST compute total loss as source_loss + lambda_target * target_loss with configurable lambda_target. Default lambda_target=1.0, configurable from YAML.
- **FR-011**: System MUST log total_loss, source_loss, target_loss, and trust_coverage_ratio per training step
- **FR-012**: System MUST evaluate on Cityscapes validation split every N configurable iterations. Default evaluation frequency: every 2000 iterations. Configurable via `eval_interval` in YAML.
- **FR-013**: System MUST save best checkpoint automatically based on validation mIoU
- **FR-014**: System MUST compute and report mIoU and per-class IoU for all 19 standard Cityscapes classes
- **FR-015**: System MUST exclude pixels with label=255 from IoU computation
- **FR-016**: System MUST support training with four configurations: Source Only, Standard Mean Teacher, SATG Hard/Soft-Weight, and SATG Soft-Label
- **FR-017**: System MUST average all reported mIoU values over 3 seeds (42, 1337, 2024)
- **FR-018**: System MUST generate 1×5 panel visualizations for at least 10 diverse images per configuration
- **FR-019**: System MUST perform ablation studies on prior type, trust function, tau_conf, and tau_struct
- **FR-020**: System MUST document all GPU compute usage (type, hours, memory peak) for each experiment
- **FR-021**: System MUST log all hyperparameters to YAML config files for every experimental result
- **FR-022**: System MUST use fixed random seeds set at the start of every script (numpy, torch, cuda, Python random)
- **FR-023**: System MUST achieve at least 80% test coverage for all modules
- **FR-024**: System MUST use only operations available in pip-installable packages (no custom CUDA kernels)
- **FR-025**: System MUST run on a single GPU with at least 16GB VRAM
- **FR-026**: System MUST use a poly learning rate schedule (power=0.9) with backbone LR=6e-4 and classifier head LR=6e-3 (10× scaling), training for 40k iterations. All LR parameters are configurable from YAML.
- **FR-027**: System MUST use standard trainable batch normalization with running statistics updated from both source and target batches. No BN freezing, no separate BN for source/target.
- **FR-028**: System MUST use standard cross-entropy with trust weights for target loss (SATGLoss). Class-frequency handling (Rare Class Sampling) is applied independently to source batches and is not part of SATG's loss computation.

### Key Entities

- **Structural Heatmap**: Per-pixel float32 map [H,W] with values in [0.0, 1.0] encoding local structural complexity. Computed from RGB using classical CV operations. Precomputed and stored as .npy files before training.
- **Teacher Confidence Map**: Per-pixel float32 map [B,H,W] with values in (0,1) representing maximum softmax probability from teacher logits. Computed during training with torch.no_grad().
- **Pseudo-Label Map**: Per-pixel int64 map [B,H,W] with values in {0,...,18} representing argmax class from teacher logits. Computed during training with torch.no_grad().
- **Trust Mask**: Per-pixel float32 map [B,H,W] with values in {0.0, 1.0} (hard) or [0.0, 1.0] (soft). Applied element-wise to per-pixel cross-entropy loss.
- **SATGLoss**: Target-domain loss function that computes cross-entropy between student predictions and teacher pseudo-labels, weighted by the trust mask.
- **GTA5 Dataset**: 24,966 synthetic images (1914×1052) with pixel-level annotations from 33 classes remapped to 19 Cityscapes-compatible classes.
- **Cityscapes Dataset**: Real-world dashcam images. Training: 2,975 images. Validation: 500 images. Resolution: 2048×1024. Annotation format: gtFine.

## Success Criteria

### Measurable Outcomes

- **SC-001**: SATG achieves higher mIoU than Standard Mean Teacher on GTA5→Cityscapes, demonstrating that the structural prior improves pseudo-label quality
- **SC-002**: SATG reduces high-confidence wrong predictions in structurally complex regions (intersections, vegetation boundaries, thin structures) as measured by per-class IoU improvement on complex classes (pole, fence, traffic sign, rider)
- **SC-003**: Structural heatmap precomputation for all 2,975 Cityscapes training images completes in under 2 hours on a single CPU core
- **SC-004**: SATG training overhead is less than 10% reduction in iterations/second compared to Standard Mean Teacher baseline
- **SC-005**: All experiments are reproducible given the same config file and random seed
- **SC-006**: Per-class IoU analysis reveals that SATG particularly improves performance on classes that are small, thin, or occluded (pole, fence, traffic light, traffic sign, rider)
- **SC-007**: Ablation studies isolate the contribution of each component (prior type, trust function, thresholds) with all results documented
- **SC-008**: Visualizations demonstrate that SATG correctly rejects high-confidence wrong predictions in structurally complex regions while accepting correct predictions in simple regions
- **SC-009**: Hypothesis: SATG improves mIoU by at least 1.0 point over Standard Mean Teacher on GTA5→Cityscapes, with the largest gains (>2.0 mIoU) on structurally complex classes (pole, fence, traffic sign, rider)
- **SC-010**: Expected improvement: 1–3 mIoU overall, with >2 mIoU on complex classes. Improvements <0.5 mIoU are considered marginal (per Constitution §1.8)
- **SC-011**: Falsification criterion: If SATG does not improve mIoU over Standard Mean Teacher across 3 seeds, or if improvements are <0.5 mIoU, the hypothesis is falsified

## Evaluation Model

Final mIoU is evaluated using the **student model** (not the EMA teacher). The teacher is used only for pseudo-label generation during training. All reported results use student model inference on the Cityscapes validation split. This matches the convention in DAFormer, MIC, and HRDA.

## Assumptions

- The GTA5 and Cityscapes datasets are available and correctly formatted following the standard UDA preprocessing pipeline (AdaptSegNet/DAFormer conventions)
- The backbone architecture is a standard segmentation network (e.g., ResNet-101 + FPN or HRNet) that can be swapped via config
- The structural prior computed from classical CV operations provides a meaningful signal that correlates with pseudo-label unreliability—this is the core hypothesis being tested
- The 512×512 crop size is used as the default, following the established DAFormer/ProDA convention
- The standard ImageNet normalization (mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) is applied to all inputs
- A single GPU with 16GB VRAM is sufficient for training with batch size 1 (source + target). Default batch size is 1 (one source image + one target image per GPU). Configurable via `batch_size` in YAML.
- The target domain has zero annotations during training (standard UDA assumption)
- The structural heatmap must be augmented consistently with the target image during training (spatial transforms applied to both). No color augmentation (jitter, blur, grayscale) is applied to target images—only spatial transforms (resize, crop, flip).
- Training uses poly LR schedule (power=0.9) with backbone LR=6e-4, head LR=6e-3, for 40k iterations.
- Standard trainable batch normalization is used; running statistics are updated from both source and target batches.
- Class-frequency handling (Rare Class Sampling) is kept separate from SATG and applied independently to source batches.
- If GPU memory is limited, reduce crop size (e.g., 384×384) or use gradient accumulation, but never skip required ablations or the Source Only baseline (per Constitution §6).

## Clarifications

### Session 2026-06-24

- Q: What is the exact formula for combining edge density and local variance into the single structural heatmap? → A: Weighted sum: H = w₁·edge_density + w₂·local_variance, w₁+w₂=1, default w₁=w₂=0.5, configurable from YAML.
- Q: What is the trust weight function for the soft-weight variant? → A: Sigmoid: w = σ(β₀ + β₁·c − β₂·s), where σ is the logistic function, c is teacher confidence, s is structural heatmap value, β₀/β₁/β₂ configurable. As β→∞, approximates hard gating.
- Q: EMA momentum — constant vs. scheduled? → A: Scheduled ramp: α_t = min(1 − 1/(iter+1), α_target), default α_target=0.999. Matches DAFormer/MIC/HRDA convention.
- Q: How to handle spatial resolution mismatch between heatmap and teacher output? → A: Full resolution. Upsample teacher logits to match heatmap resolution before gating. Gating operates at the same resolution as the final cross-entropy loss.
- Q: Warmup period before SATG kicks in? → A: No delay. Pseudo-labeling starts from iteration 1. The scheduled EMA ramp handles early instability. Matches DAFormer/MIC/HRDA convention.

### Session 2026-06-25

- Q: What data augmentation pipeline is used for target images, and must heatmaps receive identical augmentation? → A: Spatial-only augmentation (random resize 0.5–2.0×, random crop to 512×512, random horizontal flip). Applied identically to both image and heatmap. No color jitter, Gaussian blur, or grayscale on target. This avoids heatmap-image misalignment risk.
- Q: What learning rate schedule, initial LR, and backbone vs. head LR scaling? → A: Poly schedule (power=0.9), backbone LR=6e-4, classifier head LR=6e-3 (10× higher), 40k total iterations. Matches DAFormer/HRDA conventions.
- Q: What is the default lambda_target, and should it be annealed? → A: Fixed lambda_target=1.0. No annealing. The scheduled EMA ramp already handles early-training instability. Matches DAFormer/MIC/HRDA convention.
- Q: What batch normalization strategy is used during UDA training? → A: Standard trainable BN. Running statistics updated from both source and target batches. No freezing, no separate BN for source/target. Matches DAFormer/MIC/HRDA convention.
- Q: Should class-frequency weighting be applied on top of SATG, or kept separate? → A: Separate. SATG uses standard cross-entropy with trust weights. Rare Class Sampling is applied independently to source batches. Keeps SATG composable with existing methods.

## Appendix: Standard 19 Cityscapes Classes

| TrainID | Class Name | Category | RGB Color | Complexity |
|---------|-----------|----------|-----------|------------|
| 0 | road | flat | (128, 64, 128) | Easy |
| 1 | sidewalk | flat | (244, 35, 232) | Medium |
| 2 | building | construction | (70, 70, 70) | Easy |
| 3 | wall | construction | (102, 102, 156) | Medium |
| 4 | fence | construction | (190, 153, 153) | Complex |
| 5 | pole | object | (153, 153, 153) | Complex |
| 6 | traffic light | object | (250, 170, 30) | Complex |
| 7 | traffic sign | object | (220, 220, 0) | Complex |
| 8 | vegetation | nature | (107, 142, 35) | Easy |
| 9 | terrain | nature | (152, 251, 152) | Medium |
| 10 | sky | sky | (70, 130, 180) | Easy |
| 11 | person | human | (220, 20, 60) | Medium |
| 12 | rider | human | (255, 0, 0) | Complex |
| 13 | car | vehicle | (0, 0, 142) | Easy |
| 14 | truck | vehicle | (0, 0, 70) | Medium |
| 15 | bus | vehicle | (0, 60, 100) | Medium |
| 16 | train | vehicle | (0, 80, 100) | Complex |
| 17 | motorcycle | vehicle | (0, 0, 230) | Medium |
| 18 | bicycle | vehicle | (119, 11, 32) | Medium |

## Appendix: Dataset References

| Dataset | URL | Key Properties |
|---------|-----|----------------|
| GTA5 | `https://download.visinf.tu-darmstadt.de/data/from_games/` | 24,966 images, 1914×1052, all used for training |
| Cityscapes | `https://www.cityscapes-dataset.com/` | Train: 2,975, Val: 500, 2048×1024, gtFine annotations |

## Appendix: Current SOTA on GTA5→Cityscapes

| Method | Year | mIoU | Key Technique |
|--------|------|------|---------------|
| AdaptSegNet | 2018 | ~35.0 | Adversarial output-space alignment |
| CLAN | 2019 | ~39.3 | Category-level adversarial network |
| DAFormer | 2022 | ~56.2 | Transformer + masked image modeling |
| MIC | 2023 | ~58.0 | Masked Image Consistency |
| Recent methods | 2024 | ~60-62 | Foundation model backbones + advanced pseudo-labeling |

## Appendix: Structural Confirmation Bias — Key Distinction

Standard confirmation bias (addressed by DAFormer's Rare Class Sampling): class-frequency-dependent. The teacher is wrong about rare classes because the student saw too few source examples.

Structural confirmation bias (addressed by SATG): spatial/texture-dependent. The teacher is wrong in structurally complex regions (boundaries, cluttered areas, thin structures) because the network's feature extraction is inadequate for fine-grained spatial discrimination, but global context still produces high-confidence wrong answers.

These are orthogonal failure modes. SATG addresses the second; DAFormer addresses the first. They can be combined.
