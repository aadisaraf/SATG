===SATG AUDIT REPORT START===

---

SECTION 1: DIRECT CONTRADICTIONS

  [CONTRADICTION-01]
  Documents: plan.md (Config Schema) vs spec.md (FR-003)
  Topic: Soft-weight sigmoid default parameter values
  File A says: plan.md Config Schema lists `soft_weight_temp_conf: 0.1` and `soft_weight_temp_struct: 0.1`
  File B says: spec.md FR-003 states defaults β₀=0.0, β₁=10.0, β₂=10.0 with ranges β₁∈[1,100], β₂∈[1,100]
  My question: The plan.md values (0.1) are 100× smaller than the spec values (10.0). Which is correct? With β₁=0.1, the sigmoid is nearly flat (w ≈ σ(0.1c) ≈ 0.54 for most confidence values), making soft weighting nearly useless. With β₁=10.0, the sigmoid is steep and approximates hard gating as intended. The spec value seems correct.

  [CONTRADICTION-02]
  Documents: quickstart.md (Step 5) vs plan.md (Project Structure) vs tasks.md (T005)
  Topic: Config file name for soft-weight variant
  File A says: quickstart.md Step 5 references `configs/satg_soft.yaml`
  File B says: plan.md Project Structure and tasks.md T005 reference `configs/satg_soft_weight.yaml`
  My question: Which filename is canonical? Any CLI command using the wrong name will fail with FileNotFoundError.

  [CONTRADICTION-03]
  Documents: contracts/module-api.md (SoftTrustGate) vs plan.md (Config Schema)
  Topic: Config key names for soft-weight sigmoid parameters
  File A says: module-api.md uses `trust_gate.soft_temp_conf`, `trust_gate.soft_temp_struct`, `trust_gate.soft_bias`
  File B says: plan.md Config Schema uses `trust_gate.soft_weight_temp_conf`, `trust_gate.soft_weight_temp_struct` (no explicit bias key)
  My question: Which naming convention is canonical for the YAML config keys? The module-api keys are shorter but omit the "weight" prefix; the plan keys are longer but more explicit. Both default to different values (module-api: 10.0/10.0/0.0; plan: 0.1/0.1).

  [CONTRADICTION-04]
  Documents: checklists/validation.md (CHK013, CHK014, CHK026, CHK038, CHK043, CHK052, CHK055) vs spec.md
  Topic: Validation checklist claims spec is missing information that spec actually contains
  File A says: validation.md marks CHK013 (kernel sizes), CHK014 (normalization), CHK026 (sigmoid defaults), CHK038 (batch size), CHK043 (expected improvement), CHK052 (kernel ablation), CHK055 (risk mitigation) as FAIL
  File B says: spec.md FR-001 specifies σ=2.0, thresholds 50/150, window 15×15, min-max normalization with ε=1e-6; FR-003 specifies β₀=0.0, β₁=10.0, β₂=10.0; Assumptions section specifies batch_size=1; SC-009/SC-010 specify expected 1–3 mIoU; User Story 9 item 5 specifies kernel ablation; Risk Mitigation section exists with RISK-01 through RISK-05
  My question: Is validation.md a stale artifact that should be updated or archived? It gives the false impression that the spec is incomplete when the spec has been updated to address all flagged items. This could cause confusion if someone re-validates against the stale checklist.

---

SECTION 2: UNDEFINED OR UNDERSPECIFIED THINGS

  [UNDEFINED-01]
  Location: plan.md (Training Flow) and contracts/module-api.md (SATGTrainer)
  What is undefined: How the DeepLabV3+ auxiliary loss is handled during training. The constructor uses `aux_loss=True`, producing both `dict["out"]` and `dict["aux"]`, but the training flow only shows `source_loss = CE(student(source_img), source_labels)` — no mention of auxiliary loss weighting or addition to total loss.
  What I would have to assume: Either (a) ignore the auxiliary output entirely and use only the main classifier output, or (b) add auxiliary loss with a standard weight of 0.1 (common in segmentation literature). Research.md mentions `aux_loss=True` but never says how to use it.
  Why this matters: Ignoring auxiliary loss wastes a training signal that stabilizes early convergence. Using it with the wrong weight changes training dynamics. The DAFormer/MIC convention uses aux loss with weight 0.1 — if that's the intent, it must be stated.

  [UNDEFINED-02]
  Location: plan.md (SATGTrainer.__init__) and tasks.md (T029)
  What is undefined: Optimizer SGD hyperparameters beyond learning rate. The plan says "SGD with param groups" but never specifies momentum, weight decay, or dampening. The spec FR-026 only specifies LR values.
  What I would have to assume: Standard UDA defaults: momentum=0.9, weight_decay=1e-4, dampening=0. These are used by DAFormer/MIC but are never explicitly stated.
  Why this matters: Wrong momentum or weight decay can significantly affect convergence. These are not "safe to guess" values — momentum=0.9 vs 0.99 makes a real difference.

  [UNDEFINED-03]
  Location: plan.md (SATGTrainer) and contracts/module-api.md (SATGTrainer)
  What is undefined: DataLoader `num_workers` parameter. No config key or default is specified anywhere.
  What I would have to assume: num_workers=4 (common default). But this should be configurable.
  Why this matters: Too few workers cause GPU starvation; too many cause memory issues. On cloud instances with limited CPU cores, this matters.

  [UNDEFINED-04]
  Location: plan.md (PolynomialLR) and research.md (PolynomialLR)
  What is undefined: The `total_iters` parameter for PyTorch's PolynomialLR scheduler. The spec says 40k iterations, but it's unclear if total_iters=40000 is passed to the constructor or if the scheduler simply steps for 40k iterations.
  What I would have to assume: `PolynomialLR(optimizer, total_iters=40000, power=0.9)`. This is the standard approach.
  Why this matters: If total_iters doesn't match the actual training iterations, the LR schedule will be wrong (either decaying too fast or not reaching zero).

  [UNDEFINED-05]
  Location: research.md (Source augmentation) and tasks.md (T009)
  What is undefined: The exact source data augmentation pipeline is only in research.md (resize 0.5–2.0×, crop 512×512, flip, color jitter brightness=0.4/contrast=0.4/saturation=0.4/hue=0.1), but this is never formalized in spec.md or plan.md. The spec only formalizes the target augmentation (FR-007).
  What I would have to assume: Research.md is authoritative for source augmentation. But it's a research document, not a specification — it could be treated as "findings" rather than "requirements."
  Why this matters: Source augmentation affects domain randomization and generalization. Inconsistent augmentation between source and target could introduce unintended domain shift.

  [UNDEFINED-06]
  Location: plan.md (Training Flow) and spec.md (FR-007a)
  What is undefined: The exact upsampling mechanism for teacher logits to match heatmap resolution. FR-007a says "bilinearly upsampled" but the plan's code sketch doesn't show this step. At what point in the pipeline does upsampling happen?
  What I would have to assume: Upsample teacher logits via `F.interpolate(logits, size=heatmap.shape, mode='bilinear', align_corners=False)` immediately after teacher forward pass, before confidence/pseudo_label extraction.
  Why this matters: If logits are at 1/8 resolution (DeepLabV3+ default) and heatmaps are at full resolution, the trust gate would be operating at wrong resolution without upsampling, causing spatial misalignment.

  [UNDEFINED-07]
  Location: research.md (AMP decision) and plan.md (no mention)
  What is undefined: Whether Automatic Mixed Precision (AMP) is used in the training loop. Research.md explicitly decides to use `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")`, but the training flow in plan.md and tasks.md never mention AMP.
  What I would have to assume: AMP is used per research.md decision. But this is a significant implementation detail that affects numerical stability, training speed, and loss scaling.
  Why this matters: AMP changes the dtype of intermediate tensors. If the trust gate or loss function has float16 underflow issues (e.g., log(softmax) near zero), AMP could cause NaN. The research.md notes that cross_entropy stays in float32, but this needs to be verified for the KL divergence loss in SoftLabelKLLoss.

  [UNDEFINED-08]
  Location: plan.md (EMAModel) and contracts/module-api.md (EMAModel)
  What is undefined: The `state_dict()` and `load_state_dict()` methods for checkpointing are mentioned as invariants but not specified in the API contract. What exactly is saved? Just shadow params? Or also the iteration count needed for scheduled momentum?
  What I would have to assume: state_dict returns shadow params dict + current iteration number. load_state_dict restores both.
  Why this matters: Without saving the iteration count, resuming training would reset the scheduled momentum to near-0, causing a training instability spike.

  [UNDEFINED-09]
  Location: tasks.md (T009b) and spec.md (FR-028)
  What is undefined: The exact Rare Class Sampling strategy. FR-028 says "Class-frequency handling (Rare Class Sampling) is applied independently to source batches" and T009b says "inverse frequency weighting" but no formula, no class frequency table, and no configuration details are provided.
  What I would have to assume: Standard inverse frequency weighting: weight[class] = total_pixels / (count[class] * num_classes). Apply as class weights in cross-entropy for source loss.
  Why this matters: Different inverse frequency implementations (sqrt, log, raw inverse) produce very different class balances. The exact formula affects which classes get boosted.

  [UNDEFINED-10]
  Location: plan.md (Project Structure) — visualization/plot_metrics.py
  What is undefined: Tasks T031c and T050b reference `visualization/plot_metrics.py` for coverage ratio plotting, but this file is not in the Project Structure tree in plan.md. It appears to be an ad-hoc addition.
  What I would have to assume: The file is added to the visualization/ directory as part of T031c/T050b.
  Why this matters: The project structure should be the single source of truth for file locations. Undocumented files cause confusion.

---

SECTION 3: NAMING INCONSISTENCIES

  [NAMING-01]
  Concept: Config file for soft-weight variant
  Called "satg_soft.yaml" in: quickstart.md Step 5
  Called "satg_soft_weight.yaml" in: plan.md Project Structure, tasks.md T005, T005b
  Which name should be canonical: `satg_soft_weight.yaml` — it's consistent with the naming pattern (satg_hard.yaml, satg_soft_weight.yaml, satg_soft_label.yaml) and is used in the authoritative plan and tasks documents.

  [NAMING-02]
  Concept: Soft-weight sigmoid temperature/bias config keys
  Called "soft_weight_temp_conf/soft_weight_temp_struct" in: plan.md Config Schema
  Called "soft_temp_conf/soft_temp_struct/soft_bias" in: contracts/module-api.md
  Called "β₀/β₁/β₂" in: spec.md FR-003
  Which name should be canonical: `soft_weight_temp_conf` and `soft_weight_temp_struct` for the temperatures (plan.md pattern), `soft_weight_bias` for β₀ (to match the `soft_weight_` prefix). The spec's Greek letters are mathematical notation, not config keys.

  [NAMING-03]
  Concept: Evaluation interval config key
  Called "training.eval_interval" in: quickstart.md (CLI override syntax)
  Called "eval_interval" in: spec.md FR-012 (no namespace prefix)
  Which name should be canonical: `training.eval_interval` — the plan.md config schema uses flat keys but the training params logically belong under a `training` namespace. The quickstart.md CLI syntax suggests OmegaConf dot notation. Must be consistent.

  [NAMING-04]
  Concept: Soft-label variant config file
  Called "satg_soft_label.yaml" in: plan.md, tasks.md T005c, T048b
  Not present in: quickstart.md (which only references satg_soft.yaml / satg_soft_weight.yaml)
  Which name should be canonical: `satg_soft_label.yaml` — used in the authoritative plan and tasks. quickstart.md should be updated to include this variant.

---

SECTION 4: MISSING COMPONENTS

  [MISSING-01]
  Referenced in: research.md (DeepLabV3+ API), plan.md (SATGTrainer.__init__)
  What is referenced: Auxiliary loss handling — DeepLabV3+ returns both main and auxiliary outputs when aux_loss=True, but no task or spec defines how the auxiliary loss is weighted and added to the total source loss.
  Defined/specified in: NOWHERE
  Impact: Without this, the implementer must guess whether to use aux_loss (and with what weight) or discard it. Affects training convergence and is not reproducible.

  [MISSING-02]
  Referenced in: tasks.md T009 ("reads preprocessed index-map labels")
  What is referenced: GTA5 label preprocessing script/task. T009 assumes GTA5 labels are preprocessed into 19-class index maps, but there is no task for creating this preprocessing script. The cloud setup script T039B also doesn't include GTA5 label preprocessing.
  Defined/specified in: NOWHERE
  Impact: GTA5 labels are RGB palette-encoded with collisions (Building and Car both = (0,0,142) per research.md). Without preprocessing, the data loader will produce wrong labels. This is a critical pipeline gap.

  [MISSING-03]
  Referenced in: plan.md (SATGTrainer.__init__), research.md (AMP decision)
  What is referenced: AMP (Automatic Mixed Precision) integration in the training loop. Research.md explicitly decides to use torch.amp, but no task defines when and where autocast/GradScaler are used.
  Defined/specified in: NOWHERE in tasks.md or plan.md Training Flow
  Impact: AMP affects numerical precision, training speed, and loss scaling. If omitted, training is slower; if added incorrectly, can cause NaN in KL divergence loss.

  [MISSING-04]
  Referenced in: tasks.md T031c, T050b
  What is referenced: `visualization/plot_metrics.py` — a file for plotting trust coverage ratios over training
  Defined/specified in: NOWHERE in plan.md Project Structure tree
  Impact: Minor — the file is implied but not in the canonical project structure. Could cause confusion about where visualization utilities live.

  [MISSING-05]
  Referenced in: research.md (itertools.cycle decision)
  What is referenced: The mechanism for handling different-length source and target dataloaders. Research.md decides on `itertools.cycle()` on the target loader, but this is never formalized in the spec, plan training flow, or tasks.
  Defined/specified in: NOWHERE in spec.md or plan.md
  Impact: Without this, the implementer might use different cycling strategies (e.g., min-length truncation, random sampling) that affect training dynamics and reproducibility.

  [MISSING-06]
  Referenced in: contracts/module-api.md (EMAModel.state_dict)
  What is referenced: Checkpoint format specification — what keys are saved in the .pth file. Module-api mentions state_dict/load_state_dict but never defines the checkpoint dictionary structure.
  Defined/specified in: NOWHERE
  Impact: Different checkpoint formats break resume-from-checkpoint functionality. Must be standardized before implementation.

  [MISSING-07]
  Referenced in: spec.md FR-009a, plan.md Training Flow
  What is referenced: Whether source and target batches are iterated independently (two separate dataloader iterators) or as paired tuples. The training flow shows them as separate steps but doesn't specify the iteration mechanism.
  Defined/specified in: NOWHERE explicitly
  Impact: Affects whether source and target images are correlated per step. Independent iteration (as implied by itertools.cycle on target) is standard for UDA but should be explicit.

  [MISSING-08]
  Referenced in: plan.md (PolynomialLR) and research.md
  What is referenced: Whether PolynomialLR is applied per-iteration or per-epoch. Research.md says "lr_scheduler.step() must be called per iteration, not per epoch" but this is a note, not a task requirement. The total_iters parameter value is not specified.
  Defined/specified in: Implicitly 40000 iterations, but constructor call is undefined
  Impact: Wrong step frequency or total_iters breaks the learning rate schedule entirely.

---

SECTION 5: IMPLEMENTATION AMBIGUITIES (design choices left open)

  [AMBIGUITY-01]
  Location: plan.md (Training Flow) and research.md (DeepLabV3+ API)
  What is ambiguous: Whether to use the auxiliary classifier output from DeepLabV3+ and, if so, with what loss weight.
  Option A: Ignore auxiliary output entirely. Simpler code, but loses a training signal that stabilizes early convergence.
  Option B: Add auxiliary CE loss with weight 0.1 (DAFormer/MIC convention). Standard approach but adds complexity to the training loop and loss computation.
  Option C: Make it configurable via YAML (`aux_loss_weight: 0.1` or `aux_loss_weight: 0.0`). Most flexible but adds a config key not currently in the schema.
  My recommendation: Option C — add `training.aux_loss_weight: 0.1` to the config schema. This is a one-line addition that preserves flexibility while defaulting to the standard convention.

  [AMBIGUITY-02]
  Location: research.md (AMP decision) vs plan.md (no mention)
  What is ambiguous: Whether to use Automatic Mixed Precision in the training loop.
  Option A: Use AMP (torch.amp.autocast + GradScaler). Faster training (~1.5-2×), lower VRAM, but requires care with loss scaling and float16-sensitive operations (KL divergence, log_softmax).
  Option B: Train in full float32. Simpler, no AMP-related bugs, but slower and uses more VRAM.
  My recommendation: Option B for initial implementation, add AMP as a configurable option later. Getting AMP right with KL divergence loss requires careful testing, and the initial implementation should be correct before being fast.

  [AMBIGUITY-03]
  Location: spec.md (FR-007a) and plan.md (Training Flow)
  What is ambiguous: Where exactly in the pipeline teacher logits are upsampled to heatmap resolution — before or after extracting confidence/pseudo-labels.
  Option A: Upsample logits first, then extract confidence and pseudo-labels at full resolution. Trust gate operates at full resolution. Matches FR-007a literally.
  Option B: Extract confidence and pseudo-labels at model output resolution (e.g., 1/8), then upsample the confidence map and pseudo-labels to full resolution. Trust gate operates at lower resolution, which is cheaper but may lose fine-grained spatial detail.
  My recommendation: Option A — upsample logits to full resolution before extraction, per FR-007a. The whole point of SATG is fine-grained spatial discrimination; operating at lower resolution defeats the purpose.

  [AMBIGUITY-04]
  Location: plan.md (SATGTrainer) and research.md
  What is ambiguous: How the source loss is computed — standard CE, or CE with class weights from Rare Class Sampling.
  Option A: Standard CE for source loss, no class weighting. Simplest, matches the spec's statement that "Rare Class Sampling is applied independently to source batches."
  Option B: CE with inverse-frequency class weights for source loss. Standard UDA practice but adds a weight tensor to the loss.
  My recommendation: Option B — use inverse-frequency class weights for source CE. This is the standard in DAFormer/MIC and is what "Rare Class Sampling" means in practice. But this needs a concrete formula and config key.

  [AMBIGUITY-05]
  Location: plan.md (Checkpoint State) and contracts/module-api.md
  What is ambiguous: Whether to save only the best checkpoint or both best and last.
  Option A: Save best.pth (by val mIoU) and last.pth every eval interval. Standard practice.
  Option B: Save only best.pth. Simpler but loses the ability to resume from the latest iteration.
  My recommendation: Option A — save both. The plan already says "Save last.pth every eval interval; save best.pth when val mIoU improves" so this is actually already decided. But the checkpoint contents (what keys) are not specified.

---

SECTION 6: CONSTITUTION VIOLATIONS

  NONE

  Note: The constitution check in plan.md shows ALL PASS. I verified this against all constitution sections. The plan and tasks comply with all MUST principles. However, several items in the validation checklist (validation.md) are stale and should be updated to reflect the current spec state — this is a documentation maintenance issue, not a constitution violation.

---

SECTION 7: THINGS I AM CONFIDENT ABOUT

  CONFIDENT: StructuralPrior class — compute(), batch_compute(), all parameters (Canny thresholds, Gaussian σ, kernel sizes, weights w₁/w₂), normalization formula, config keys, output format. Fully specified in spec FR-001, module-api.md §1, and research.md §3.
  CONFIDENT: HardTrustGate class — compute_mask() formula (confidence > tau_conf AND struct < tau_struct), config keys, binary output. Fully specified in spec FR-003/FR-004, module-api.md §2.
  CONFIDENT: SoftTrustGate class — compute_weights() sigmoid formula, monotonicity requirements. Fully specified (once config key naming is resolved). Note: plan.md default values are wrong (0.1 vs 10.0).
  CONFIDENT: TemperatureSoftLabel class — compute_temperature(), compute_soft_targets(), compute_confidence_mask(). Fully specified in spec US-04b and plan.md §soft_label.py.
  CONFIDENT: SoftLabelKLLoss class — KL divergence formula, masking, zero-mask handling, eps safety. Fully specified in plan.md §losses.py and module-api.md.
  CONFIDENT: SATGLoss class — per-pixel CE with trust weights, ignore_index=255, zero-weight guard. Fully specified in spec FR-003, module-api.md §3.
  CONFIDENT: EMAModel class — shadow copy, scheduled momentum formula α_t = min(1−1/(iter+1), α_target), eval mode, no gradients. Fully specified in spec FR-008, module-api.md §4.
  CONFIDENT: Cityscapes 19-class mapping — TrainID table with RGB colors fully specified in spec Appendix.
  CONFIDENT: Heatmap precomputation pipeline — CLI args, naming convention, output format, multiprocessing. Fully specified in spec FR-005, module-api.md §5.
  CONFIDENT: Config file structure — YAML hierarchy, variant configs (satg_hard, satg_soft_weight, satg_soft_label, mean_teacher, source_only). Fully specified in plan.md Config Schema and Project Structure.
  CONFIDENT: Evaluator class — mIoU computation, per-class IoU, ignore_index=255, student-only evaluation. Fully specified in spec US-07, module-api.md §7.
  CONFIDENT: Training seeds — {42, 1337, 2024}, all four RNG sources, configurable. Fully specified in constitution §1.1/§1.2 and spec FR-022.
  CONFIDENT: Dry run validation — 10 images, 10 iterations, specific acceptance criteria. Fully specified in constitution §3.3 and tasks.md T000.

---

SECTION 8: QUESTIONS ABOUT THE SOFT-LABEL AMENDMENT

  [SOFTLABEL-01]
  What I am confused about: SoftLabelKLLoss declares `ignore_index=255` in __init__ but the forward() signature `(student_logits, soft_targets, confidence_mask)` has no pseudo_labels argument. The ignore_index parameter is never used in the forward computation.
  Where I looked for the answer: plan.md §losses.py, contracts/module-api.md §SoftLabelKLLoss, spec.md US-04b
  What I found: The forward method computes KL divergence between student_log_probs and soft_targets, masked by confidence_mask. There is no label tensor to apply ignore_index to. The parameter appears to be dead code.
  What I need clarified: Should ignore_index=255 be removed from SoftLabelKLLoss (it's meaningless here), or is there a use case where soft_targets should be zeroed out at certain pixels based on ground-truth labels? For the target domain, there are no ground truth labels, so this seems unused.

  [SOFTLABEL-02]
  What I am confused about: The soft-label variant uses temperature-scaled teacher logits to produce soft_targets. But teacher logits are at model output resolution (typically 1/8 of input), while heatmaps are at full resolution. The plan.md training flow passes `tgt_teacher_logits` and `tgt_heatmaps` directly to `compute_soft_targets()` without showing the upsampling step.
  Where I looked for the answer: plan.md Training Flow, spec.md FR-007a, contracts/module-api.md §SoftLabel
  What I found: FR-007a says "Teacher logits are bilinearly upsampled to match the precomputed heatmap resolution before the trust gate is applied." But the code sketch in plan.md doesn't show this. The module-api compute_soft_targets() takes teacher_logits [B,C,H,W] and struct [B,H,W] — implying they must already be the same spatial size.
  What I need clarified: Is the upsampling step implied (must be done before calling compute_soft_targets), or should compute_soft_targets handle upsampling internally? This matters for the API contract — if the caller must upsample, the module-api should document the expected input resolution.

  [SOFTLABEL-03]
  What I am confused about: How "trust_coverage_ratio" is defined for the soft-label variant. T033c says "define it as confidence_mask.mean() for this variant" — but this is only the fraction of pixels passing the confidence threshold, not a measure of how much the temperature softening affects the loss. Is this the right metric?
  Where I looked for the answer: tasks.md T033c, spec.md FR-011, constitution §1.10
  What I found: T033c explicitly says to use confidence_mask.mean() for the soft-label variant. Constitution §1.10 requires "Trust mask coverage ratio (percentage of target pixels trusted per batch) must be logged."
  What I need clarified: confidence_mask.mean() measures what fraction of pixels have high enough teacher confidence — it doesn't capture the temperature softening effect. Should we also log the mean temperature T across trusted pixels? This would give insight into how aggressively the softening is happening during training.

  [SOFTLABEL-04]
  What I am confused about: Whether the soft-label variant's KL divergence loss is normalized per-pixel or per-batch. The module-api shows `masked_kl.sum() / confidence_mask.sum()` — this normalizes by the number of trusted pixels, making the loss independent of how many pixels are trusted. But the hard/soft-weight variants use `(per_pixel_ce * trust_weights).sum() / (trust_weights.sum() + 1e-8)` — same normalization pattern. Are these directly comparable in magnitude?
  Where I looked for the answer: plan.md §losses.py, contracts/module-api.md §SATGLoss and §SoftLabelKLLoss
  What I found: Both use the same normalization pattern (sum / count). However, KL divergence and cross-entropy have different scales — KL is always ≥ 0 and can be much larger than CE for mismatched distributions. The magnitude comparison between SATGLoss and SoftLabelKLLoss is not straightforward.
  What I need clarified: Should lambda_target be different for soft-label vs hard/soft-weight training to account for different loss scales? Or is the implicit assumption that lambda_target=1.0 works for all variants?

  [SOFTLABEL-05]
  What I am confused about: The config key `soft_label_t_max` appears in plan.md Config Schema but is not present in the spec FR-003 or the Clarifications section. The spec says T_max=5.0 but doesn't name the config key.
  Where I looked for the answer: plan.md Config Schema, spec.md FR-003, contracts/module-api.md §SoftLabel
  What I found: plan.md has `soft_label_t_max: 5.0` in the config schema. module-api.md has `T_max` in __init__. The spec just says "T_max (default 5.0)."
  What I need clarified: Is `soft_label_t_max` the canonical config key? This should be stated in the spec's FR-003 for completeness.

---

SECTION 9: OVERALL READINESS ASSESSMENT

The documentation is substantially complete and well-structured — the spec, plan, tasks, module-api contracts, and data model together provide a strong foundation for implementation. The core SATG modules (structural prior, trust gates, losses, EMA) are fully specified with concrete formulas, config keys, and API contracts. The main gaps fall into two categories: (1) integration-level details that are missing from the training loop (auxiliary loss handling, AMP, optimizer params, GTA5 preprocessing), and (2) naming inconsistencies between documents (soft-weight config keys, file names) that will cause import errors and debugging time if not resolved before coding. The validation checklist (validation.md) is stale and should be updated or archived. Implementation can begin immediately for the independent modules (Phase 2: structural_prior, trust_gate, soft_label, losses) which are fully specified. The training loop (Phase 5) has the most gaps and requires the most clarification before implementation. If I started implementing RIGHT NOW with zero clarifications, I estimate approximately 65–70% of the codebase could be written correctly — the core SATG modules, data loaders, evaluator, and visualization are solid, but the training loop integration (auxiliary loss, AMP, optimizer config, dataloader cycling) would require informed guesses that could need rework.

===SATG AUDIT REPORT END===
