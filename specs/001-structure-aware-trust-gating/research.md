# Research: Structure-Aware Trust Gating (SATG)

**Date**: 2026-06-25 | **Feature**: 001-structure-aware-trust-gating

Research consolidated from 4 parallel subagents covering: PyTorch model stack, dataset handling, structural prior CV, and training infrastructure.

---

## 1. PyTorch and Model Stack

### Decision: PyTorch 2.7.0 + CUDA 12.6/12.8

- **Rationale**: Latest stable PyTorch 2.x with full segmented model support. CUDA 12.6/12.8 are official options for PyTorch 2.7.0.
- **Alternatives considered**: PyTorch 2.5 (older, less optimized) rejected; CUDA 11.8 kept as fallback for older GPUs.

### DeepLabV3+ ResNet50 API

- Constructor: `torchvision.models.segmentation.deeplabv3_resnet50(num_classes=19, weights_backbone=ResNet50_Weights.IMAGENET1K_V1, aux_loss=True)`
- Forward returns `dict["out"]` (main) and `dict["aux"]` (auxiliary), both interpolated to input size
- `model.classifier` is `nn.Sequential`; final layer is `model.classifier[-1]` = `Conv2d(256, 19, 1)`
- `model.aux_classifier[-1]` = `Conv2d(256, 19, 1)`
- Weights: only `COCO_WITH_VOC_LABELS_V1` (21 classes); use `num_classes=19` at construction for random head init

### Decision: Manual EMA class (not external library)

- **Rationale**: Simple dict-based shadow copy; update per optimizer step; swap for eval
- **Pattern**: `ema_param = momentum * ema_param + (1 - momentum) * student_param`
- **Scheduled momentum**: `α_t = min(1 - 1/(iter+1), α_target)` matching DAFormer convention
- **Gotchas**: EMA shadow must be a Python dict (not nn.Parameter) to avoid optimizer pickup

### Decision: Batch size 1 per GPU (source + target)

- **Rationale**: Spec requires ≤16GB VRAM. DeepLabV3+ ResNet50 at 512×512 uses ~4GB for batch_size=1. With AMP, batch_size=4 fits comfortably.
- **Alternative**: batch_size=4 with AMP for faster training if VRAM allows

### Decision: AMP with torch.amp (modern API)

- **Rationale**: `torch.cuda.amp` is deprecated in PyTorch 2.x. Use `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")`.
- **DeepLabV3+ AMP safety**: cross_entropy and BatchNorm stay in float32 automatically; bilinear interpolation stays in float16 — all safe.

### Decision: PolynomialLR (built-in) for poly schedule

- **Rationale**: PyTorch 2.x has `torch.optim.lr_scheduler.PolynomialLR` built-in. No custom implementation needed.
- **Critical**: `lr_scheduler.step()` must be called per iteration, not per epoch.
- **LR scaling**: backbone_lr=6e-4, head_lr=6e-3 (10× higher), aux_lr=6e-3 (10× higher)

---

## 2. Dataset Handling

### Decision: Preprocess GTA5 labels to index maps offline

- **Rationale**: GTA5 uses RGB palette-encoded labels with collisions (Building and Car both = (0,0,142)). Preprocessing resolves collisions and produces 19-class index maps.
- **Pattern**: Read RGB label → map to Cityscapes trainID using lookup dict → save as uint8 index map

### Decision: Use gtFine_labelTrainIds.png for Cityscapes

- **Rationale**: Standard 19-class index format used by all UDA papers. Generated via `cityscapesscripts` preparation script.
- **Note**: Not included in default Cityscapes download; must be generated separately.

### Decision: itertools.cycle() on Cityscapes loader

- **Rationale**: GTA5 (24,966 images) >> Cityscapes (2,975 images). `itertools.cycle()` on shorter loader ensures infinite iteration without restarts.
- **Pattern**: `target_iter = itertools.cycle(target_loader); src_imgs, src_labels = next(source_loader); tgt_imgs, tgt_heatmaps = next(target_iter)`

### Decision: Spatial-only augmentation for target images

- **Rationale**: Spec FR-007 requires identical augmentation for image and heatmap. Color jitter/blur/grayscale would invalidate precomputed heatmaps.
- **Target pipeline**: random resize (0.5–2.0×), random crop to 512×512, random horizontal flip (p=0.5)
- **Source pipeline**: same spatial transforms + color jitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

---

## 3. Structural Prior Implementation

### Decision: cv2.Canny with thresholds (50, 150)

- **Rationale**: Balanced for urban scenes; captures sky (0.01–0.05 density) vs intersections (0.15–0.30 density)
- **API**: `cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)`
- **Output**: `np.uint8` with values {0, 255} — must divide by 255 for [0,1] range

### Decision: cv2.blur for local variance (not scipy)

- **Rationale**: cv2.blur is 2–3× faster than scipy.ndimage.uniform_filter for this use case
- **Formula**: `var = blur(I²) - blur(I)²` where I is grayscale float32, blur uses uniform kernel
- **Kernel**: 15×15 default (configurable via YAML)

### Decision: Edge density via cv2.filter2D

- **Rationale**: Convolve binary edge map with uniform kernel for sliding-window density
- **Formula**: `density = filter2D(edges_float, -1, kernel) / kernel.sum()` where kernel is uniform 15×15
- **Alternative**: `torch.nn.functional.avg_pool2d` rejected — CPU-only pipeline, no GPU needed

### Performance

- Per-image pipeline: ~10ms on CPU (no GPU needed)
- 2,975 images with 8-core multiprocessing: ~4 seconds total
- Far under the 2-hour single-core budget (spec User Story 1)

### Heatmap normalization

- **Decision**: Min-max per image: `H_norm = (H - H_min) / (H_max - H_min + 1e-6)`
- **Weights**: `H = 0.5 * edge_density + 0.5 * local_variance` (both normalized to [0,1])
- **Output**: float32 NumPy array, shape (H, W), saved as .npy

---

## 4. Training Infrastructure

### Decision: OmegaConf for all configs

- **Rationale**: Hierarchical YAML with variable interpolation; CLI override support
- **Pattern**: `cfg = OmegaConf.load("configs/default.yaml"); cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())`
- **WandB logging**: `wandb.init(config=OmegaConf.to_container(cfg, resolve=True))`

### Decision: WandB primary, TensorBoard fallback

- **Rationale**: WandB has native segmentation mask overlay API (`wandb.Image(masks={...})`); free tier sufficient for individual research
- **Free tier**: 5GB storage, 5 seats — adequate for this project
- **Per-class IoU logging**: `wandb.Table(columns=["class", "IoU"])` with all 19 classes

### Decision: Checkpoint saving with best mIoU tracking

- **Pattern**: Save `last.pth` every eval interval; save `best.pth` when val mIoU improves
- **Checkpoint contents**: model_state_dict, ema_model_state_dict, optimizer_state_dict, scheduler_state_dict, best_miou, config
- **Load**: `torch.load("best.pth", map_location=device)` → load model or EMA model

### Decision: pyproject.toml for coverage config

- **Rationale**: Preferred over .coveragerc for modern Python projects
- **Config**: `source = ["satg", "models", "data", "training", "evaluation"]`, `fail_under = 80`
- **Command**: `pytest --cov=satg --cov=models --cov=data --cov=training --cov=evaluation --cov-fail-under=80`

### Decision: tqdm.contrib.concurrent.process_map for heatmap precomputation

- **Rationale**: Simplest API for multiprocessing with progress bar
- **Alternative**: `pool.imap` + `tqdm` wrapper also works; process_map is cleaner

---

## Summary of Key Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| PyTorch | 2.7.0 + CUDA 12.6 | Latest stable, full support |
| Model | DeepLabV3+ ResNet50, num_classes=19 | Standard UDA backbone |
| EMA | Manual class, scheduled momentum | Simple, matches DAFormer |
| Batch size | 1 per GPU (source + target) | ≤16GB VRAM constraint |
| AMP | torch.amp (modern API) | torch.cuda.amp deprecated |
| LR schedule | PolynomialLR(power=0.9) | Built-in, no custom code |
| Heatmap | cv2.Canny(50,150) + cv2.blur variance | Fast, CPU-only, deterministic |
| Config | OmegaConf YAML | Hierarchical, CLI overrides |
| Tracking | WandB | Native segmentation support |
| Coverage | pyproject.toml ≥80% | Modern Python standard |
