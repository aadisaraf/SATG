# Tasks: Structure-Aware Trust Gating (SATG)

**Input**: Design documents from `/specs/001-structure-aware-trust-gating/`
**Spec**: spec.md | **Plan**: plan.md | **Contracts**: contracts/module-api.md

## Priority Restructuring

**Soft-labeling/soft-weighting is the PRIMARY contribution.** Hard rejection is a baseline mechanism. The user stories are reordered to reflect this:

- US-04b (Soft-Label): **P3** — Primary contribution
- US-04 (Soft-Weight): **P4** — Secondary soft mechanism
- US-03 (Hard Rejection): **P5** — Baseline mechanism

## Multi-Subagent Coordination

After Phase 0 is complete, tasks are distributed as follows:

| Subagent | Scope | Tasks |
|----------|-------|-------|
| **Primary** | PHASE 1 (data loaders), PHASE 5 (training loop), PHASE 6 (eval), PHASE 7 (viz), PHASE 7.5 (cloud scripts), PHASE 8–10 | T006–T011, T009b, T010b, T028–T033, T031b, T031c, T033b, T033c, T034–T036, T037–T039, T039A–T039E, T040–T047, T047b, T048–T050, T050b, T048b–T048d, T_FINAL |
| **Subagent A** | PHASE 2 (structural_prior.py + tests) | T012–T014 |
| **Subagent B** | PHASE 2 (trust_gate.py + tests) + PHASE 2 (losses.py) | T015–T019 |
| **Subagent C** | PHASE 3 (segmentation.py + ema.py + tests) | T020–T023 |
| **Subagent D** | PHASE 2 (soft_label.py + SoftLabelKLLoss + tests) **PRIMARY** | T019a–T019d |
| **Primary** | PHASE 4 (precompute script) | T024–T027 |

Subagents A, B, C, D run in parallel after Phase 0 completes. **Subagent D is the primary contribution** (soft-label modulation). Primary handles PHASE 1, 4, 5–10 sequentially.

---

## PHASE 0: Environment & Configuration

- [ ] T001 — — Create project directory structure per plan.md: `configs/`, `data/`, `models/`, `satg/`, `precompute/`, `training/`, `evaluation/`, `visualization/`, `tests/` with `__init__.py` files in all Python packages. Files: `configs/`, `data/__init__.py`, `models/__init__.py`, `satg/__init__.py`, `precompute/__init__.py`, `training/__init__.py`, `evaluation/__init__.py`, `visualization/__init__.py`, `tests/__init__.py`. Estimated time: 10–15 min. Acceptance Criterion: All directories and `__init__.py` files exist; `python -c "import satg; import models; import data; import training; import evaluation"` succeeds. Subagent hint: Primary.

- [ ] T002 — — Create `pyproject.toml` with black (line-length=100), flake8, and pytest-cov config. Create `.coveragerc` with `source = ["satg", "models", "data", "training", "evaluation"]` and `fail_under = 80`. Files: `pyproject.toml`, `.coveragerc`. Estimated time: 10–15 min. Acceptance Criterion: `black --check .` and `flake8` pass on empty files; `pytest --co` discovers test directory. Subagent hint: Primary.

- [ ] T002b — — Write coverage enforcement test: run `pytest --cov=satg --cov=models --cov=data --cov=training --cov=evaluation --cov-fail-under=80` and verify it passes. Files: (validation only, no new files). Estimated time: 10–15 min. Acceptance Criterion: Coverage report shows ≥80% line coverage for all target packages. Subagent hint: Primary.

- [ ] T003 — — Create `requirements.txt` with pinned dependencies: torch>=2.7.0, torchvision>=0.22, opencv-python>=4.x, numpy>=1.24, omegaconf>=2.3, wandb, pytest, pytest-cov, tqdm, black, flake8. Files: `requirements.txt`. Estimated time: 5–10 min. Acceptance Criterion: `pip install -r requirements.txt` succeeds on a fresh environment. Subagent hint: Primary.

- [ ] T004 [P] — — Create master config `configs/default.yaml` with all hyperparameters: seed=42, seeds=[42,1337,2024], structural_prior params (edge thresholds, kernel sizes, weights), trust_gate params (tau_conf=0.9, tau_struct=0.6, soft params), training params (lr=6e-4, head_lr=6e-3, iterations=40000, batch_size=1, crop_size=[512,512], eval_interval=2000, lambda_target=1.0, ema_momentum=0.999), model params (backbone=resnet50, num_classes=19). Files: `configs/default.yaml`. Estimated time: 15–20 min. Acceptance Criterion: `OmegaConf.load("configs/default.yaml")` resolves without error; all constitution §1.3 hyperparameters present. Subagent hint: Primary.

- [ ] T005 — — Create variant configs: `configs/satg_hard.yaml`, `configs/satg_soft_weight.yaml`, `configs/baseline_mean_teacher.yaml`, `configs/source_only.yaml` — each extending default.yaml with overrides. Files: `configs/satg_hard.yaml`, `configs/satg_soft_weight.yaml`, `configs/baseline_mean_teacher.yaml`, `configs/source_only.yaml`. Estimated time: 15–20 min. Acceptance Criterion: Each config loads via OmegaConf and contains expected override keys; source_only has lambda_target=0.0. Subagent hint: Primary.

- [ ] T005b — — Write config isolation verification tests in `tests/test_configs.py`: test source_only has lambda_target=0.0, test mean_teacher has no heatmap loading keys, test all 4 variant configs share identical backbone/optimizer/source data keys and differ only in target loss mechanism. Files: `tests/test_configs.py`. Estimated time: 15–20 min. Acceptance Criterion: All assertions pass; any config deviation from isolation contract raises AssertionError. Additionally verify that mean_teacher config, when parsed by trainer, results in zero heatmap file I/O (runtime isolation check). Subagent hint: Primary.

- [ ] T005c — — Create `configs/satg_soft_label.yaml` extending `default.yaml`: set `trust_gate.type=soft_label`, `trust_gate.soft_label_k=4.0`, `trust_gate.soft_label_t_max=5.0`. Files: `configs/satg_soft_label.yaml`. Estimated time: 5–10 min. Acceptance Criterion: `OmegaConf.load("configs/satg_soft_label.yaml")` resolves; `trust_gate.type == "soft_label"`. Subagent hint: Primary.

**Checkpoint**: Phase 0 complete — project structure, configs, and tooling ready. Subagents A, B, C can now begin in parallel.

---

## PHASE 1: Data Infrastructure (GTA5 + Cityscapes loaders, label mapping)

- [ ] T006 — [US1] Write tests for label mapping in `tests/test_data_loaders.py`: test GTA5→Cityscapes 19-class mapping covers all 19 classes, test mapping dict completeness (33 GTA5 classes → 19 trainIDs), test that unmapped classes raise ValueError. Files: `tests/test_data_loaders.py`. Estimated time: 20–30 min. Acceptance Criterion: Tests exist and FAIL (red phase — mapping module not yet implemented). Subagent hint: Primary.

- [ ] T007 — [US1] Implement `data/label_mapping.py` with GTA5→Cityscapes 19-class mapping dict and `map_gta5_label(label_rgb: np.ndarray) -> np.ndarray` function. Files: `data/label_mapping.py`. Estimated time: 20–30 min. Acceptance Criterion: All tests in T006 pass (green phase). Subagent hint: Primary.

- [ ] T008 — [US1] Write tests for GTA5 loader in `tests/test_data_loaders.py`: test Dataset.__len__ returns correct count, test __getitem__ returns (image, label) tuple with correct shapes and dtypes, test label values are in [0,18] or 255. Files: `tests/test_data_loaders.py`. Estimated time: 20–30 min. Acceptance Criterion: Tests exist and FAIL (red phase). Subagent hint: Primary.

- [ ] T009 — [US1] Implement `data/gta5_loader.py` GTA5Dataset class: reads preprocessed index-map labels, applies source augmentation (spatial + color jitter per research.md), returns (image tensor [3,512,512], label tensor [512,512]). Files: `data/gta5_loader.py`. Estimated time: 30–40 min. Acceptance Criterion: All GTA5 loader tests pass. Subagent hint: Primary.

- [ ] T009b — [US1] Write tests for and implement Rare Class Sampling for source batches in `data/gta5_loader.py` or a separate `data/rare_class_sampler.py`: test that source batch class frequencies are balanced according to Rare Class Sampling strategy (inverse frequency weighting), test that the sampler is configurable via YAML. Files: `tests/test_data_loaders.py`, `data/gta5_loader.py` or `data/rare_class_sampler.py`. Estimated time: 30–40 min. Acceptance Criterion: Source batch class distribution is more uniform than naive sampling; sampler is config-driven. Subagent hint: Primary.

- [ ] T010 — [US1] Write tests for Cityscapes loader in `tests/test_data_loaders.py`: test train split returns (image, heatmap) tuples, test val split returns (image, label) tuples, test heatmap shape matches image spatial dims, test heatmap values in [0,1]. Files: `tests/test_data_loaders.py`. Estimated time: 20–30 min. Acceptance Criterion: Tests exist and FAIL (red phase). Subagent hint: Primary.

- [ ] T010b — [US1/RISK-04] Write augmentation consistency test in `tests/test_data_loaders.py`: (a) apply known spatial transforms (resize, crop, flip) and verify image and heatmap output dimensions match, (b) compare transform parameter sets for image vs heatmap — both must receive identical spatial distortion, (c) verify no color augmentation (jitter, blur, grayscale) is applied to target images, (d) verify heatmap values remain in [0,1] after augmentation. Files: `tests/test_data_loaders.py`. Estimated time: 15–20 min. Acceptance Criterion: Test explicitly verifies spatial transform identity between image and heatmap; any mismatch raises AssertionError. This directly validates RISK-04 (augmentation mismatch). Subagent hint: Primary.

- [ ] T011 — [US1] Implement `data/cityscapes_loader.py` CityscapesDataset class: loads images + precomputed heatmaps for train split, images + gtFine_labelTrainIds for val split, spatial-only augmentation for target (resize, crop, flip) applied identically to image and heatmap. Files: `data/cityscapes_loader.py`. Estimated time: 30–40 min. Acceptance Criterion: All Cityscapes loader tests pass; heatmap augmentation matches image augmentation. Subagent hint: Primary.

**Checkpoint**: Phase 1 complete — data loaders ready. Both source and target data can be loaded with correct formats and augmentation.

---

## PHASE 2: Core SATG Modules (structural prior, trust gate, loss) [PARALLEL OK]

> Subagents A, B, D work in parallel on PHASE 2. **Subagent D (soft_label) is the primary contribution.**

- [x] T019a [P] — [US4b/PRIMARY] Write tests for TemperatureSoftLabel in `tests/test_soft_label.py`. Tests to include:
  - test_temperature_at_zero_struct_is_one: struct=0 -> T=1.0 exactly
  - test_temperature_increases_with_struct: struct=0.2 -> T=1.8, struct=0.5 -> T=3.0 (using k=4.0 default)
  - test_temperature_capped_at_t_max: struct=1.0, k=10 -> T clamped to T_max
  - test_soft_target_sums_to_one: output[b,:,h,w].sum() ≈ 1.0 for all pixels
  - test_soft_target_flattens_with_temperature: as struct increases for a fixed teacher_logits input, the resulting distribution's entropy strictly increases (confirms softening actually happens)
  - test_high_temp_approaches_uniform: at T=T_max with large struct, the output distribution is close to uniform (1/C for each class)
  - test_confidence_mask_matches_hard_gate_logic: confidence_mask uses the same tau_conf comparison as the existing hard gate
  Parallel: [P] — independent of other Phase 2 tasks
  Dependencies: NONE (uses synthetic fixtures from conftest.py)
  Files: tests/test_soft_label.py. Estimated time: 30–40 min. Acceptance Criterion: Tests exist and FAIL (red phase — TemperatureSoftLabel not implemented). Subagent hint: Subagent D (primary).

- [x] T019b — [US4b/PRIMARY] Implement `satg/soft_label.py` TemperatureSoftLabel class: `__init__(cfg)` reads k, T_max, tau_conf; `compute_temperature(struct)` computes T=1.0+k*struct clamped to [1.0,T_max]; `compute_soft_targets(teacher_logits, struct)` divides logits by per-pixel T then softmax; `compute_confidence_mask(teacher_logits, tau_conf)` reuses hard gate logic. Files: `satg/soft_label.py`. Estimated time: 30–40 min. Acceptance Criterion: All tests in T019a pass (green phase). Subagent hint: Subagent D (after T019a).

- [x] T019c [P] — [US4b/PRIMARY] Write tests for SoftLabelKLLoss in `tests/test_soft_label.py` (extend existing file). Tests to include:
  - test_zero_confidence_mask_gives_zero_loss
  - test_no_nan_with_zero_mask
  - test_identical_distributions_give_near_zero_kl: if student_logits produce the exact same distribution as soft_targets, KL ≈ 0
  - test_loss_increases_with_distribution_mismatch
  - test_output_is_scalar
  Dependencies: NONE. Files: tests/test_soft_label.py. Estimated time: 20–30 min. Acceptance Criterion: Tests exist and FAIL (red phase — SoftLabelKLLoss not implemented). Subagent hint: Subagent D (new).

- [x] T019d — [US4b/PRIMARY] Implement `satg/losses.py` SoftLabelKLLoss class (nn.Module): `__init__()`; `forward(student_logits, soft_targets, confidence_mask)` computes KL divergence masked by confidence, returns 0.0 when mask is all-zero, eps=1e-8 for log safety. Files: `satg/losses.py` (extend existing). Estimated time: 20–30 min. Acceptance Criterion: All new SoftLabelKLLoss tests pass; existing SATGLoss tests still pass. Subagent hint: Subagent D (after T019c).

- [x] T012 [P] — [US1] Write tests for StructuralPrior in `tests/test_structural_prior.py`: test output shape matches input, test output values in [0,1], test sky region mean < 0.30, test vegetation mean > 0.60, test deterministic (same input → same output), test only classical CV ops used. Files: `tests/test_structural_prior.py`. Estimated time: 30–40 min. Acceptance Criterion: Tests exist and FAIL (red phase — StructuralPrior not implemented). Subagent hint: Subagent A.

- [x] T013 [P] — [US1] Implement `satg/structural_prior.py` StructuralPrior class: RGB→grayscale, Gaussian blur, Canny edge detection, edge density via cv2.filter2D, local variance via cv2.blur, weighted combination, min-max normalization. Config-driven parameters from OmegaConf. Files: `satg/structural_prior.py`. Estimated time: 40–60 min. Acceptance Criterion: All tests in T012 pass; per-image compute < 50ms on CPU. Subagent hint: Subagent A.

- [x] T014 [P] — [US1] Write and pass batch_compute test: test `batch_compute([img1, img2])` returns list of heatmaps with correct shapes. Update `tests/test_structural_prior.py`. Files: `tests/test_structural_prior.py`, `satg/structural_prior.py`. Estimated time: 15–20 min. Acceptance Criterion: batch_compute works for lists of varying-size images. Subagent hint: Subagent A.

- [x] T015 [P] — [US3] Write tests for HardTrustGate and SoftTrustGate in `tests/test_trust_gate.py`: test hard mask is binary {0.0, 1.0}, test hard mask = 1.0 iff confidence > tau_conf AND struct < tau_struct, test soft weights in [0,1], test soft monotonicity (increasing confidence → non-decreasing weight, increasing structure → non-increasing weight), test edge case all-rejected → zero weights. Files: `tests/test_trust_gate.py`. Estimated time: 30–40 min. Acceptance Criterion: Tests exist and FAIL (red phase). Subagent hint: Subagent B.

- [x] T016 [P] — [US3] Implement `satg/trust_gate.py` HardTrustGate class: binary mask = (confidence > tau_conf) AND (structure < tau_struct). Files: `satg/trust_gate.py`. Estimated time: 15–20 min. Acceptance Criterion: HardTrustGate tests pass. Subagent hint: Subagent B.

- [x] T017 [P] — [US4] Implement `satg/trust_gate.py` SoftTrustGate class: sigmoid((conf-0.5)*temp_conf + bias) * sigmoid((tau_struct-struct)*temp_struct). Extend `tests/test_trust_gate.py` with soft variant tests. Files: `satg/trust_gate.py`, `tests/test_trust_gate.py`. Estimated time: 20–30 min. Acceptance Criterion: SoftTrustGate tests pass; monotonicity verified. Subagent hint: Subagent B.

- [x] T018 — [US3] Write tests for SATGLoss in `tests/test_losses.py`: test output is scalar ≥ 0, test zero trust weights → loss = 0.0, test no NaN/Inf, test ignore_index=255 excluded. Files: `tests/test_losses.py`. Estimated time: 20–30 min. Acceptance Criterion: Tests exist and FAIL (red phase). Subagent hint: Subagent B (after T017; depends on trust_gate.py).

- [x] T019 — [US3] Implement `satg/losses.py` SATGLoss class: per-pixel CE with reduction='none', weighted sum with trust_weights, zero-weight guard. Files: `satg/losses.py`. Estimated time: 20–30 min. Acceptance Criterion: All SATGLoss tests pass; zero-weight edge case returns 0.0. Subagent hint: Subagent B (after T018; depends on trust_gate.py and test interface).

- [x] T019a [P] — [US4b] Write tests for TemperatureSoftLabel in `tests/test_soft_label.py`. Tests to include:
  - test_temperature_at_zero_struct_is_one: struct=0 -> T=1.0 exactly
  - test_temperature_increases_with_struct: struct=0.2 -> T=1.8, struct=0.5 -> T=3.0 (using k=4.0 default)
  - test_temperature_capped_at_t_max: struct=1.0, k=10 -> T clamped to T_max
  - test_soft_target_sums_to_one: output[b,:,h,w].sum() ≈ 1.0 for all pixels
  - test_soft_target_flattens_with_temperature: as struct increases for a fixed teacher_logits input, the resulting distribution's entropy strictly increases (confirms softening actually happens)
  - test_high_temp_approaches_uniform: at T=T_max with large struct, the output distribution is close to uniform (1/C for each class)
  - test_confidence_mask_matches_hard_gate_logic: confidence_mask uses the same tau_conf comparison as the existing hard gate
  Parallel: [P] — independent of other Phase 2 tasks
  Dependencies: NONE (uses synthetic fixtures from conftest.py)
  Files: tests/test_soft_label.py. Estimated time: 30–40 min. Acceptance Criterion: Tests exist and FAIL (red phase — TemperatureSoftLabel not implemented). Subagent hint: Subagent D (new).

- [x] T019b — [US4b] Implement `satg/soft_label.py` TemperatureSoftLabel class: `__init__(cfg)` reads k, T_max, tau_conf; `compute_temperature(struct)` computes T=1.0+k*struct clamped to [1.0,T_max]; `compute_soft_targets(teacher_logits, struct)` divides logits by per-pixel T then softmax; `compute_confidence_mask(teacher_logits, tau_conf)` reuses hard gate logic. Files: `satg/soft_label.py`. Estimated time: 30–40 min. Acceptance Criterion: All tests in T019a pass (green phase). Subagent hint: Subagent D (after T019a).

- [x] T019c [P] — [US4b] Write tests for SoftLabelKLLoss in `tests/test_soft_label.py` (extend existing file). Tests to include:
  - test_zero_confidence_mask_gives_zero_loss
  - test_no_nan_with_zero_mask
  - test_identical_distributions_give_near_zero_kl: if student_logits produce the exact same distribution as soft_targets, KL ≈ 0
  - test_loss_increases_with_distribution_mismatch
  - test_output_is_scalar
  Dependencies: NONE. Files: tests/test_soft_label.py. Estimated time: 20–30 min. Acceptance Criterion: Tests exist and FAIL (red phase — SoftLabelKLLoss not implemented). Subagent hint: Subagent D (new).

- [x] T019d — [US4b] Implement `satg/losses.py` SoftLabelKLLoss class (nn.Module): `__init__()`; `forward(student_logits, soft_targets, confidence_mask)` computes KL divergence masked by confidence, returns 0.0 when mask is all-zero, eps=1e-8 for log safety. Files: `satg/losses.py` (extend existing). Estimated time: 20–30 min. Acceptance Criterion: All new SoftLabelKLLoss tests pass; existing SATGLoss tests still pass. Subagent hint: Subagent D (after T019c).

**Checkpoint**: Phase 2 complete — structural prior, trust gates, and loss function all tested and working independently.

---

## PHASE 3: Model and EMA (DeepLabV3+ wrapper, EMA class) [PARALLEL OK with P2]

> Subagent C works on PHASE 3 in parallel with PHASE 2.

- [ ] T020 [P] — [US2] Write tests for DeepLabV3+ wrapper in `tests/test_segmentation.py`: test output shape [B,19,H,W], test forward pass with random input produces valid logits, test auxiliary output present. Files: `tests/test_segmentation.py`. Estimated time: 20–30 min. Acceptance Criterion: Tests exist and FAIL (red phase). Subagent hint: Subagent C.

- [ ] T021 [P] — [US2] Implement `models/segmentation.py` SegmentationModel wrapper: wraps `torchvision.models.segmentation.deeplabv3_resnet50(num_classes=19)`, replaces final classifier layers for 19-class output, provides `extract_teacher_info()` method returning (confidence, pseudo_labels) under `torch.no_grad()`. Files: `models/segmentation.py`. Estimated time: 30–40 min. Acceptance Criterion: All segmentation tests pass; extract_teacher_info returns correct shapes and no gradients. Subagent hint: Subagent C.

- [ ] T022 [P] — [US2] Write tests for EMAModel in `tests/test_ema.py`: test shadow params match student initially, test update moves shadow toward student, test scheduled momentum formula, test eval mode, test no gradients. Files: `tests/test_ema.py`. Estimated time: 20–30 min. Acceptance Criterion: Tests exist and FAIL (red phase). Subagent hint: Subagent C.

- [ ] T023 [P] — [US2] Implement `models/ema.py` EMAModel class: shadow dict copy, update with scheduled momentum α_t = min(1−1/(iter+1), α_target), forward pass, state_dict/load_state_dict for checkpointing. Files: `models/ema.py`. Estimated time: 30–40 min. Acceptance Criterion: All EMA tests pass; scheduled momentum formula verified. Subagent hint: Subagent C.

**Checkpoint**: Phase 3 complete — model wrapper and EMA teacher ready.

---

## PHASE 4: Precomputation Pipeline (offline heatmap generation)

- [ ] T024 — — Write tests for heatmap precomputation in `tests/test_precompute.py`: test CLI discovers images recursively, test output naming convention `{stem}_satg_heatmap.npy`, test output dtype float32, test output shape matches input. Files: `tests/test_precompute.py`. Estimated time: 20–30 min. Acceptance Criterion: Tests exist and FAIL (red phase). Subagent hint: Primary.

- [ ] T025 — — Implement `precompute/compute_heatmaps.py` CLI script: argparse for `--data_root` and `--num_workers`, recursive PNG discovery, StructuralPrior.compute per image, np.save with naming convention, multiprocessing with tqdm progress, print statistics. Files: `precompute/compute_heatmaps.py`. Estimated time: 30–40 min. Acceptance Criterion: All precompute tests pass; `python -m precompute.compute_heatmaps --data_root <test_dir> --num_workers 1` produces correct .npy files. Subagent hint: Primary.

- [ ] T026 — — Add `precompute/__init__.py` and verify `python -m precompute.compute_heatmaps --help` prints usage. Files: `precompute/__init__.py`. Estimated time: 5–10 min. Acceptance Criterion: Module is importable and CLI help works. Subagent hint: Primary.

- [ ] T027 — — Run precompute validation on 10 sample Cityscapes images: verify all heatmaps have shape (H,W), dtype float32, values in [0,1], and naming convention correct. Files: (validation only, no new files). Estimated time: 10–15 min. Acceptance Criterion: 10/10 heatmaps pass validation checks. Subagent hint: Primary.

**Checkpoint**: Phase 4 complete — heatmap precomputation pipeline working.

---

## PHASE 5: Training Loop Integration

- [ ] T028 — [US5] Write tests for trainer in `tests/test_trainer.py`: test trainer initializes all components, test one training step executes without error, test EMA update occurs after step, test loss logging occurs. Files: `tests/test_trainer.py`. Estimated time: 30–40 min. Acceptance Criterion: Tests exist and FAIL (red phase). Subagent hint: Primary.

- [ ] T029 — [US5] Implement `training/trainer.py` SATGTrainer class: init from OmegaConf config (student, EMA teacher, loaders, optimizer SGD with param groups, PolynomialLR scheduler, trust gate, SATGLoss, evaluator, wandb). Files: `training/trainer.py`. Estimated time: 60–90 min. Acceptance Criterion: Trainer initializes without error; all components created from config. Subagent hint: Primary.

- [ ] T030 — [US5] Implement training loop: source forward → source CE loss; teacher forward (no_grad) → pseudo_labels + confidence; student target forward → logits; trust gate → weights; SATGLoss → target loss; total = source + λ·target; backward + optimizer step; EMA update; lr_scheduler.step() per iteration. Files: `training/trainer.py`. Estimated time: 60–90 min. Acceptance Criterion: One full training step completes; losses are finite scalars; EMA weights differ from student after update. Subagent hint: Primary.

- [ ] T031 — [US5] Add logging: WandB init with config, per-step logging of total_loss, source_loss, target_loss, trust_coverage_ratio; evaluation at eval_interval; checkpoint saving (last.pth every eval, best.pth on mIoU improvement). Files: `training/trainer.py`. Estimated time: 30–40 min. Acceptance Criterion: WandB receives logged metrics; checkpoint files created. Subagent hint: Primary.

- [ ] T031b — [US5] Add per-class trust coverage logging to `training/trainer.py`: for each of the 19 Cityscapes classes, compute the fraction of pixels belonging to that class that are trusted by the trust gate. Log all 19 per-class coverage ratios to WandB as separate metrics (e.g., `coverage/class_road`, `coverage/class_sidewalk`, etc.). This detects RISK-01 (class bias in trusted pixels). Files: `training/trainer.py`. Estimated time: 20–30 min. Acceptance Criterion: WandB receives 19 per-class coverage metrics per logging step; coverage ratios sum to overall trust_coverage_ratio. Subagent hint: Primary.

- [ ] T031c — [US9/RISK-01] Compute per-class trust coverage summary: after training, aggregate per-class coverage ratios from WandB logs (or training CSV exports). Compute mean, std, min, max trust coverage per class across training iterations. Include summary table in EXPERIMENTS.md. Flag any class with <10% or >90% mean coverage as potential bias. Files: EXPERIMENTS.md, `visualization/plot_metrics.py` (extend). Estimated time: 15–20 min. Acceptance Criterion: Per-class coverage summary table in EXPERIMENTS.md; bias classes flagged; overall coverage stats included. Subagent hint: Primary.

- [ ] T032 — [US5] Add seed management: set np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all, random.seed at start of train(); add torch.backends.cudnn.deterministic/benchmark settings. Files: `training/trainer.py`. Estimated time: 10–15 min. Acceptance Criterion: Same seed → same training trajectory (losses match within tolerance). Subagent hint: Primary.

- [ ] T033 — — Run trainer tests to verify all pass. Files: (no new files). Estimated time: 10–15 min. Acceptance Criterion: `pytest tests/test_trainer.py -v` all pass. Subagent hint: Primary.

- [ ] T033b — — Write target label leakage tests in `tests/test_data_leakage.py`: (1) test CityscapesDataset train split __getitem__ returns (image, heatmap) with NO label tensor, (2) test teacher forward pass operates under torch.no_grad() (zero gradient computation), (3) test trainer never passes target ground truth labels to any loss function. Files: `tests/test_data_leakage.py`. Estimated time: 20–30 min. Acceptance Criterion: All 3 tests pass; any attempt to access target labels in train mode raises an error. Subagent hint: Primary.

- [ ] T033c — [US4b] Add soft_label branch to `training/trainer.py`: implement the three-way branch (hard / soft_weight / soft_label) per plan.md TRAINING FLOW. When `cfg.trust_gate.type == "soft_label"`, compute soft_targets via TemperatureSoftLabel, compute confidence_mask, and pass both to SoftLabelKLLoss instead of SATGLoss. Ensure trust_coverage_ratio logging still works for soft_label (define it as confidence_mask.mean() for this variant). Files: `training/trainer.py` (extend existing). Estimated time: 30–40 min. Acceptance Criterion: Trainer runs correctly with `trust_gate.type=soft_label`; trust_coverage_ratio is logged as confidence_mask.mean(); no regression on hard/soft_weight modes. Subagent hint: Primary.

- [ ] T033d — [HYPOTHESIS] Add hypothesis validation logging to `training/trainer.py`: during training, log per-batch metrics for hypothesis validation: (1) correlation between structural heatmap values and pseudo-label prediction disagreement (proxy for error since target GT is unavailable), (2) per-class trust coverage ratios for all 19 classes, (3) distribution statistics of structural complexity for trusted vs. rejected pixels. Log all metrics to WandB. Files: `training/trainer.py` (extend existing). Estimated time: 30–40 min. Acceptance Criterion: All hypothesis validation metrics are logged to WandB; correlation metric is finite; per-class coverage ratios sum to overall trust_coverage_ratio. Subagent hint: Primary.

**Checkpoint**: Phase 5 complete — training loop integrated and tested.

---

## T000: Dry Run Verification

> Must appear after PHASE 5 is complete.

- [ ] T000 — — Dry Run Verification: Run training for 10 iterations on 10 images with `training.iterations=10 training.batch_size=1 training.eval_interval=5`. Verify: (1) no NaN/Inf in any logged loss, (2) EMA weights update (teacher ≠ student after step 1), (3) checkpoint file saved, (4) WandB logs received, (5) all 10 iterations complete without error. Files: (no new files, validation only). Estimated time: 20–30 min. Acceptance Criterion: All 5 criteria met; `python -m training.trainer training.iterations=10 training.eval_interval=5` exits cleanly. Subagent hint: Primary.

- [ ] T000b — — Mini Dry Run — SATG Soft-Label: Before full training, run 10 iterations with `configs/satg_soft_label.yaml` on the same 10-image subset used in T000. Verify: no NaN/Inf, soft_targets sum to 1.0 per pixel, confidence_mask is in {0,1}, KL loss is non-negative, checkpoint saves correctly. Files: (no new files, validation only). Estimated time: 15–20 min. Acceptance Criterion: All checks pass; `python -m training.trainer --config configs/satg_soft_label.yaml training.iterations=10 training.eval_interval=5` exits cleanly. Must PASS before the 3-seed SATG Soft-Label training task begins. Subagent hint: Primary.

- [ ] T000c — — **Hypothesis Validation Dry Run**: Run training for 100 iterations with `configs/satg_soft_label.yaml`. Log and verify: (1) structural heatmap values are in [0,1], (2) teacher confidence values are in (0,1), (3) pseudo-label error rate is computable, (4) correlation between structural heatmap and prediction disagreement is logged, (5) per-class trust coverage ratios are logged. Files: (no new files, validation only). Estimated time: 20–30 min. Acceptance Criterion: All 5 metrics are logged to WandB; correlation metric is finite and not NaN. This validates that hypothesis tracking infrastructure works before full training. Subagent hint: Primary.

---

## PHASE 6: Evaluation Pipeline

- [ ] T034 — [US7] Write tests for evaluator in `tests/test_evaluator.py`: test mIoU computation matches expected formula, test per-class IoU for 19 classes, test ignore_index=255 excluded, test empty prediction handling. Files: `tests/test_evaluator.py`. Estimated time: 20–30 min. Acceptance Criterion: Tests exist and FAIL (red phase). Subagent hint: Primary.

- [ ] T035 — [US7] Implement `evaluation/evaluator.py` Evaluator class: load checkpoint, run inference on Cityscapes val split (500 images), compute per-pixel confusion matrix, compute per-class IoU and mIoU, print formatted table. Files: `evaluation/evaluator.py`. Estimated time: 40–60 min. Acceptance Criterion: All evaluator tests pass; `python -m evaluation.evaluator --checkpoint <path> --config <path>` prints 19-class IoU table. Subagent hint: Primary.

- [ ] T036 — — Add `evaluation/__init__.py` and verify module importability. Files: `evaluation/__init__.py`. Estimated time: 5 min. Acceptance Criterion: `from evaluation.evaluator import Evaluator` succeeds. Subagent hint: Primary.

**Checkpoint**: Phase 6 complete — evaluation pipeline produces mIoU and per-class IoU.

---

## PHASE 7: Visualization

- [ ] T037 — [US8] Write tests for visualization in `tests/test_visualization.py`: test 5-panel output has correct shape, test saved as PNG, test trust mask overlay colors are correct. Files: `tests/test_visualization.py`. Estimated time: 15–20 min. Acceptance Criterion: Tests exist and FAIL (red phase). Subagent hint: Primary.

- [ ] T038 — [US8] Implement `visualization/visualize.py`: 1×5 panel generation (RGB, confidence map, structural heatmap, trust mask overlay, SATG-filtered pseudo-label), colorize with Cityscapes palette, save as PNG to `visualizations/{config_name}/`. Files: `visualization/visualize.py`. Estimated time: 30–40 min. Acceptance Criterion: All visualization tests pass; `python -m visualization.visualize --checkpoint <path> --config <path> --num_images 10` produces 10 PNG files. Subagent hint: Primary.

- [ ] T039 — — Run visualization on 10 sample images and verify ≥3 show SATG rejecting high-confidence predictions in complex regions. Files: (no new files, validation only). Estimated time: 15–20 min. Acceptance Criterion: 10 PNG files exist in `visualizations/`; at least 3 demonstrate structural rejection. Subagent hint: Primary.

**Checkpoint**: Phase 7 complete — visualizations generated and qualitatively validated.

---

## PHASE 7.5: Cloud Setup and Training Instructions

> **This phase generates everything you need to train on the cloud. The agent produces the script and instructions; you run them on the cloud instance.**

- [ ] T039A — — Write `cloud/setup.sh`: cloud environment setup script that (1) checks for CUDA/GPU, (2) installs pip dependencies from requirements.txt, (3) sets random seeds, (4) verifies PyTorch + CUDA availability with `torch.cuda.is_available()`, (5) prints GPU info (`torch.cuda.get_device_name(0)`, VRAM). Files: `cloud/setup.sh`. Estimated time: 15–20 min. Acceptance Criterion: Script runs cleanly on a cloud GPU instance; prints "GPU ready" on success. Subagent hint: Primary.

- [ ] T039B — — Write `cloud/prepare_data.sh`: data download and preprocessing script that (1) downloads GTA5 dataset to `--data_root`, (2) downloads Cityscapes dataset, (3) runs `cityscapesscripts` to generate `gtFine_labelTrainIds.png`, (4) runs `python -m precompute.compute_heatmaps` on Cityscapes training images, (5) validates heatmap count = 2975. Accepts `--data_root` and `--cityscapes_root` arguments. Files: `cloud/prepare_data.sh`. Estimated time: 20–30 min. Acceptance Criterion: Script contains all data prep steps with correct CLI syntax; user can follow it step-by-step. Subagent hint: Primary.

- [ ] T039C — — Write `cloud/INSTRUCTIONS.md`: complete cloud training guide with (1) recommended GPU instances and estimated costs (A100 ~$1.50/hr, H100 ~$3.50/hr), (2) step-by-step setup: SSH into instance → upload code → run setup.sh → run prepare_data.sh, (3) exact training commands for PHASE 8 (6 runs: source_only × 3 seeds + mean_teacher × 3 seeds), (4) exact training commands for PHASE 9 (SATG hard/soft-weight/soft-label × 3 seeds + all ablation configs including soft-label k-sensitivity), (5) how to monitor via WandB, (6) how to download checkpoints + results back to local, (7) estimated total cost and time per configuration, (8) troubleshooting section (CUDA out of memory, dataset not found, heatmap mismatch). Files: `cloud/INSTRUCTIONS.md`. Estimated time: 30–40 min. Acceptance Criterion: A user with zero context can follow INSTRUCTIONS.md from cloud login to completed training. Subagent hint: Primary.

- [ ] T039D — — Write `cloud/run_phase8.sh`: PHASE 8 launch script that runs all 6 baseline training commands (source_only + mean_teacher × 3 seeds) sequentially, logs output to `cloud/logs/phase8_*.log`, and prints a summary table on completion. Files: `cloud/run_phase8.sh`. Estimated time: 10–15 min. Acceptance Criterion: Script contains all 6 commands with correct config paths and seed overrides. Subagent hint: Primary.

- [ ] T039E — — Write `cloud/run_phase9.sh`: PHASE 9 launch script that runs all SATG experiments and ablations sequentially (SATG hard/soft-weight/soft-label main runs + Ablations A–G), logs output to `cloud/logs/phase9_*.log`, and prints a summary table on completion. Files: `cloud/run_phase9.sh`. Estimated time: 15–20 min. Acceptance Criterion: Script contains all ~34 training commands with correct config paths and seed overrides. Subagent hint: Primary.

**Checkpoint**: Phase 7.5 complete — cloud setup script, data prep script, instructions, and training launch scripts ready. User can now upload to cloud and begin PHASE 8.

---

## PHASE 8: Baseline Experiments (Source Only + Standard Mean Teacher)

- [ ] T040 — [US6] Train Source Only baseline: `python -m training.trainer --config configs/source_only.yaml` for 40k iterations with seed=42. Record mIoU and per-class IoU. Files: EXPERIMENTS.md (append). Estimated time: 60–120 min (compute-bound). Acceptance Criterion: mIoU ~30% on Cityscapes val; results recorded in EXPERIMENTS.md with config snapshot. Subagent hint: Primary.

- [ ] T041 — [US6] Train Standard Mean Teacher baseline: `python -m training.trainer --config configs/baseline_mean_teacher.yaml` for 40k iterations with seed=42. Record mIoU and per-class IoU. Files: EXPERIMENTS.md (append). Estimated time: 60–120 min (compute-bound). Acceptance Criterion: mIoU ~35-40% on Cityscapes val; results recorded in EXPERIMENTS.md with config snapshot; training logs confirm zero heatmap file I/O (mean_teacher uses confidence threshold only, no structural prior). Subagent hint: Primary.

- [ ] T042 — — Run Source Only and Mean Teacher with seeds {1337, 2024} for 3-seed averaging per constitution §1.2. Files: EXPERIMENTS.md (update with mean±std). Estimated time: 120–240 min (compute-bound, can run 2 seeds in parallel). Acceptance Criterion: 3-seed mean±std reported for both baselines. Subagent hint: Primary.

**Checkpoint**: Phase 8 complete — baselines established with 3-seed averaging.

---

## PHASE 9: Main SATG Experiments + Ablations

- [ ] T043 — — Train SATG Hard Rejection (seed=42): `python -m training.trainer --config configs/satg_hard.yaml`. Record mIoU and per-class IoU. Files: EXPERIMENTS.md (append). Estimated time: 60–120 min. Acceptance Criterion: mIoU > Mean Teacher baseline; results in EXPERIMENTS.md. Subagent hint: Primary.

- [ ] T044 — — Train SATG Soft-Weight (seed=42): `python -m training.trainer --config configs/satg_soft_weight.yaml`. Record mIoU. Files: EXPERIMENTS.md (append). Estimated time: 60–120 min. Acceptance Criterion: Results in EXPERIMENTS.md. Subagent hint: Primary.

- [ ] T045 — — Run SATG Hard + Soft-Weight with seeds {1337, 2024}. Files: EXPERIMENTS.md (update with mean±std). Estimated time: 120–240 min. Acceptance Criterion: 3-seed mean±std for both SATG variants. Subagent hint: Primary.

- [ ] T046 — — Ablation A: Edge-density-only vs local-variance-only vs combined prior (3 configs, seed=42 each). Create ablation config files. Files: `configs/ablation_edge_only.yaml`, `configs/ablation_variance_only.yaml`, `configs/ablation_combined.yaml`, EXPERIMENTS.md. Estimated time: 120–180 min. Acceptance Criterion: All 3 ablation results in EXPERIMENTS.md; combined prior ≥ individual priors. Subagent hint: Primary.

- [ ] T047 — — Ablation B: tau_conf sweep {0.80, 0.90, 0.95} with tau_struct=0.60 fixed (3 configs, seed=42 each). Ablation C: tau_struct sweep {0.40, 0.60, 0.70} with tau_conf=0.90 fixed (3 configs, seed=42 each). Ablation D: hard vs soft (best of each, seed=42). Files: `configs/ablation_tau_conf_*.yaml`, `configs/ablation_tau_struct_*.yaml`, EXPERIMENTS.md. Estimated time: 180–240 min. Acceptance Criterion: All ablation results in EXPERIMENTS.md; per-class IoU for top-5 affected classes. Subagent hint: Primary.

- [ ] T047b — [US9/RISK-03] Run combined tau_conf × tau_struct grid search: sweep tau_conf ∈ {0.85, 0.90, 0.95} × tau_struct ∈ {0.50, 0.60, 0.70} (9 configs, seed=42 each). Create combined ablation config files. Record mIoU and per-class IoU for all 9 configurations. This tests interaction effects between the two thresholds that independent sweeps (T047) cannot detect. Files: `configs/ablation_tau_grid_*.yaml`, EXPERIMENTS.md. Estimated time: 240–360 min. Acceptance Criterion: 9 ablation results in EXPERIMENTS.md; interaction effects visible in results table; best (tau_conf, tau_struct) pair identified. Subagent hint: Primary.

- [ ] T048 — — Ablation E: Kernel σ ∈ {0.5, 1.0, 2.0} and variance window ∈ {7×7, 15×15, 31×31} (6 configs, seed=42 each). Files: `configs/ablation_kernel_*.yaml`, EXPERIMENTS.md. Estimated time: 180–240 min. Acceptance Criterion: All 6 kernel ablation results in EXPERIMENTS.md; results show kernel size affects mIoU, confirming RISK-02 (coarse prior) is a real factor; optimal kernel configuration identified and reported. Subagent hint: Primary.

- [ ] T048b — — Train SATG Soft-Label (**5 seeds**: {42, 1337, 2024, 7, 99}): run `python -m training.trainer --config configs/satg_soft_label.yaml`. Record mean±std mIoU and per-class IoU in EXPERIMENTS.md. Dependencies: T033c (soft_label trainer branch) complete, T000b (soft-label dry run) passed. Files: EXPERIMENTS.md (append). Estimated time: 300–600 min (5 × 40k iterations). Acceptance Criterion: 5-seed mean±std mIoU reported for SATG Soft-Label; per-class IoU for all 19 classes in EXPERIMENTS.md. Subagent hint: Primary.

- [ ] T048c — — Ablation F — Soft mechanism comparison: compile a side-by-side comparison table of SATG Hard, SATG Soft-Weight, SATG Soft-Label, all using their already-completed main run results. Include mIoU (mean±std) and per-class IoU for the top-5 most affected classes. Files: EXPERIMENTS.md (append comparison table). Estimated time: 15–20 min. Acceptance Criterion: Comparison table in EXPERIMENTS.md with all three variants; determines whether weight-scaling or distribution-softening is the more effective soft mechanism. Dependencies: SATG Hard (T043+T045), SATG Soft-Weight (T044+T045), and SATG Soft-Label (T048b) main runs all complete. Subagent hint: Primary.

- [ ] T048d — — Ablation G — Temperature constant k sensitivity: run `configs/satg_soft_label.yaml` with k ∈ {2.0, 6.0} (k=4.0 already run as the main soft-label run in T048b), single seed=42 each. Create ablation config files with overrides. Record in EXPERIMENTS.md under "Ablation G: soft_label_k". Files: `configs/ablation_soft_label_k2.yaml`, `configs/ablation_soft_label_k6.yaml`, EXPERIMENTS.md. Estimated time: 120–180 min (2 × 40k iterations). Acceptance Criterion: Both k-sensitivity results in EXPERIMENTS.md; sensitivity to k characterized. Dependencies: SATG Soft-Label main run (T048b) complete for the k=4.0 baseline. Subagent hint: Primary.

### Combination Experiments (Phase 9 Extension)

- [ ] T048e — — Implement Rare Class Sampling as config option in `data/gta5_loader.py` or `data/rare_class_sampler.py`: inverse frequency weighting for source batches, configurable via YAML `data.rare_class_sampling=true`. Files: `data/rare_class_sampler.py` or `data/gta5_loader.py`. Estimated time: 30–40 min. Acceptance Criterion: Sampler works correctly; source batch class distribution is more uniform; sampler is config-driven. Subagent hint: Primary.

- [ ] T048f — — Train DAFormer baseline: `python -m training.trainer --config configs/baseline_daformer.yaml` with Rare Class Sampling enabled, seed=42. Record mIoU and per-class IoU. Files: `configs/baseline_daformer.yaml`, EXPERIMENTS.md (append). Estimated time: 60–120 min. Acceptance Criterion: mIoU ~56% on Cityscapes val; results in EXPERIMENTS.md. Subagent hint: Primary.

- [ ] T048g — — Train SATG Soft-Label + DAFormer: `python -m training.trainer --config configs/satg_soft_label_daformer.yaml` with both Rare Class Sampling and soft-label modulation, seeds {42, 1337}. Record mIoU and per-class IoU. Files: `configs/satg_soft_label_daformer.yaml`, EXPERIMENTS.md (append). Estimated time: 120–240 min. Acceptance Criterion: mIoU > DAFormer alone; results in EXPERIMENTS.md; complementarity demonstrated. Subagent hint: Primary.

- [ ] T048h — — Compile combination experiment comparison table: DAFormer alone vs. SATG Soft-Label + DAFormer. Include mIoU (mean±std) and per-class IoU for all 19 classes. Files: EXPERIMENTS.md (append). Estimated time: 15–20 min. Acceptance Criterion: Comparison table in EXPERIMENTS.md; complementarity quantified. Dependencies: T048f and T048g complete. Subagent hint: Primary.

**Checkpoint**: Phase 9 complete — all main experiments and ablations finished.

---

## PHASE 10: Documentation and Analysis

- [ ] T049 — — Write `README.md` with: project description, installation (pip install -r requirements.txt), dataset download instructions, precomputation command, training commands (all configs), evaluation command, visualization command, CLI syntax examples. Files: `README.md`. Estimated time: 30–40 min. Acceptance Criterion: README covers all sections per constitution §4.1; all CLI commands are syntactically correct. Subagent hint: Primary.

- [ ] T050 — — Populate `EXPERIMENTS.md` with full comparison table: Method | mIoU (mean±std) | per-class IoU for all 19 classes | GPU type | training hours | GPU memory peak. Include all baselines, SATG variants, and ablation results. Files: `EXPERIMENTS.md`. Estimated time: 30–40 min. Acceptance Criterion: Table has all methods; all results have 3-seed mean±std; per-class IoU for all 19 classes. Subagent hint: Primary.

- [ ] T050b — — Generate trust_coverage_ratio vs. iteration plot: read WandB logs or CSV exports, plot overall trust_coverage_ratio and per-class coverage ratios over training iterations, save as PNG to `visualizations/training_metrics/`. This visualizes that the trust gating mechanism is active and not collapsing. Files: `visualization/plot_metrics.py` or extend `visualization/visualize.py`. Estimated time: 15–20 min. Acceptance Criterion: PNG plots show coverage ratio trends over training; at least 2 plots generated (overall + per-class). Subagent hint: Primary.

- [ ] T_FINAL — — Generate EXPERIMENTS.md comparison table and update README.md. Verify: (1) all ablation results from §3.1 present, (2) source-only baseline included, (3) compute documented per §1.7, (4) all configs logged per §1.3. Files: `EXPERIMENTS.md`, `README.md`. Estimated time: 20–30 min. Acceptance Criterion: EXPERIMENTS.md is complete with all methods; README.md has full usage guide. Subagent hint: Primary.

---

## Dependencies & Execution Order

### Phase Dependencies

```
PHASE 0 ──┬──→ PHASE 1 (Primary)
           ├──→ PHASE 2A (Subagent A: structural_prior)
           ├──→ PHASE 2B (Subagent B: trust_gate)
           ├──→ PHASE 2C (Subagent C: model + ema)
           └──→ PHASE 4 (Primary: precompute)

PHASE 1 ──→ PHASE 5 (Training loop — needs data loaders)
PHASE 2A + 2B + 2C ──→ PHASE 5 (needs all core modules)
PHASE 4 ──→ PHASE 5 (needs precomputed heatmaps for training)

PHASE 5 ──→ T000 (Dry Run)
T000 ──→ PHASE 6 (Evaluation)
PHASE 6 ──→ PHASE 8 (Baselines need eval)
PHASE 7.5 (Cloud Setup) ──→ PHASE 8 (user needs scripts before training)
PHASE 8 ──→ PHASE 9 (Main experiments need baselines for comparison)
PHASE 7 (Visualization) ──→ PHASE 10 (needs trained models)
PHASE 9 ──→ PHASE 10 (needs all results)
```

### Parallel Opportunities

After PHASE 0 completes, the following run in parallel:
- **Subagent A**: T012 → T013 → T014 (structural prior)
- **Subagent B**: T015 → T016 → T017 → T018 → T019 (trust gate + loss)
- **Subagent C**: T020 → T021, T022 → T023 (model + EMA)
- **Subagent D**: T019a → T019b, T019c → T019d (soft label + KL loss)
- **Primary**: T006 → T007 → T008 → T009 → T010 → T010b → T011 (data loaders), then T024 → T025 → T026 → T027 (precompute)

Within PHASE 2: T012 || T015 (different files, no deps), T013 || T016, T014 || T017
Within PHASE 3: T020 || T022 (different files, no deps)

### Implementation Strategy

1. **MVP First**: PHASE 0 → PHASE 1 → PHASE 2 → PHASE 3 → PHASE 5 → T000 → STOP AND VALIDATE
2. **Evaluation**: PHASE 6 → run on dry-run checkpoint
3. **Cloud Prep**: PHASE 7.5 → generate setup scripts + instructions
4. **Upload to Cloud**: user uploads code, runs setup.sh + prepare_data.sh
5. **Baselines**: PHASE 8 (Source Only + Mean Teacher) — run on cloud
6. **Main Experiments**: PHASE 9 (SATG Hard + Soft + ablations) — run on cloud
7. **Documentation**: PHASE 10 (agent or user, after results are in)

---

## Notes

- [P] tasks run in parallel (different files, no dependencies on incomplete tasks)
- All test tasks follow test-first (red → green) per constitution §2.2
- Constitution §1.2 requires 3 seeds {42, 1337, 2024} for all reported mIoU
- **Exception**: SATG Soft-Label (T048b) uses 5 seeds {42, 1337, 2024, 7, 99} for better statistical power on the primary contribution
- Constitution §3.1 requires all 7 ablation variants
- Constitution §3.2 requires Source Only baseline
- Constitution §3.3 requires dry run (T000) before any full training
- PHASE 7.5 generates all cloud scripts — after this, PHASE 8–9 are user-executed on cloud
- Compute-bound tasks (PHASE 8–9) have wide time estimates; actual time depends on GPU
- Cloud cost estimate: ~$400-600 for full PHASE 8+9 on A100 instances (increased due to combination experiments and 5-seed soft-label run)
- **Evaluation model**: Final mIoU uses student model (not EMA teacher), per spec.md and plan.md
- **New tasks added**: T002b (coverage enforcement), T005b (config isolation), T005c (soft_label config), T009b (Rare Class Sampling), T010b (augmentation consistency), T019a–T019d (soft_label module + KL loss), T031b (per-class coverage logging), T031c (per-class coverage summary), T033b (target label leakage), T033c (soft_label trainer branch), T033d (hypothesis validation logging), T000b (soft-label dry run), T000c (hypothesis validation dry run), T047b (combined tau grid), T048b (soft-label 5-seed training), T048c (soft mechanism comparison), T048d (k sensitivity ablation), T048e (Rare Class Sampling implementation), T048f (DAFormer baseline), T048g (SATG Soft-Label + DAFormer), T048h (combination comparison table), T050b (coverage visualization)
- **Subagent reassignment**: T018-T019 (SATGLoss) moved from Primary/Subagent A to Subagent B (who owns trust_gate.py and completes T017 first)
- **Config key alignment**: All configs use `lambda_target` (not `target_loss_weight`)
- **Priority restructuring**: Soft-labeling/soft-weighting is the PRIMARY contribution; hard rejection is a baseline mechanism
- **Combination experiments**: T048e–T048h test SATG+DAFormer complementarity (untested claim in original spec)
- **Hypothesis validation**: T033d and T000c add infrastructure to validate the core assumption (structural complexity correlates with pseudo-label error)
