# Module API Contracts: Structure-Aware Trust Gating (SATG)

**Date**: 2026-06-25 | **Feature**: 001-structure-aware-trust-gating

---

## 1. `satg/structural_prior.py` — StructuralPrior

### Class: `StructuralPrior`

```python
class StructuralPrior:
    def __init__(self, cfg: OmegaConf) -> None:
        """Load edge/variance params from config.
        
        Config keys used:
            structural_prior.edge_low_threshold (int, default=50)
            structural_prior.edge_high_threshold (int, default=150)
            structural_prior.gaussian_sigma (float, default=2.0)
            structural_prior.edge_kernel_size (int, default=15)
            structural_prior.variance_kernel_size (int, default=15)
            structural_prior.edge_weight (float, default=0.5)
            structural_prior.variance_weight (float, default=0.5)
        """
    
    def compute(self, image_rgb: np.ndarray) -> np.ndarray:
        """Compute structural complexity heatmap for a single image.
        
        Args:
            image_rgb: RGB image, shape (H, W, 3), dtype uint8, values [0, 255]
        
        Returns:
            Heatmap, shape (H, W), dtype float32, values [0.0, 1.0]
        
        Steps:
            1. RGB → grayscale (cv2.cvtColor)
            2. Gaussian blur (cv2.GaussianBlur, sigma=gaussian_sigma)
            3. Canny edge detection (cv2.Canny, thresholds from config)
            4. Edge density: uniform kernel convolution → [0, 1]
            5. Local variance: var = blur(I²) - blur(I)² → normalize to [0, 1]
            6. Weighted combination: H = w₁·edge_density + w₂·local_variance
            7. Clip to [0, 1], return float32
        """
    
    def batch_compute(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Compute heatmaps for a list of images.
        
        Args:
            images: List of RGB images, each (H_i, W_i, 3), dtype uint8
        
        Returns:
            List of heatmaps, each (H_i, W_i), dtype float32
        """
```

### Invariants
- Output shape matches input spatial dimensions exactly
- All output values in [0.0, 1.0]
- Deterministic: same input → same output (no randomness)
- No neural network operations; classical CV only

---

## 2. `satg/trust_gate.py` — HardTrustGate, SoftTrustGate

### Class: `HardTrustGate`

```python
class HardTrustGate:
    def __init__(self, cfg: OmegaConf) -> None:
        """Read thresholds from config.
        
        Config keys:
            trust_gate.tau_conf (float, default=0.90)
            trust_gate.tau_struct (float, default=0.60)
        """
    
    def compute_mask(
        self,
        confidence: Tensor,  # [B, H, W], float32, (0, 1)
        struct: Tensor,       # [B, H, W], float32, [0, 1]
    ) -> Tensor:
        """Compute binary trust mask.
        
        Args:
            confidence: Per-pixel teacher confidence
            struct: Per-pixel structural complexity
        
        Returns:
            Binary mask, shape [B, H, W], dtype float32, values {0.0, 1.0}
        
        Formula: mask[i,j] = 1.0 iff confidence[i,j] > tau_conf AND struct[i,j] < tau_struct
        """
```

### Class: `SoftTrustGate`

```python
class SoftTrustGate:
    def __init__(self, cfg: OmegaConf) -> None:
        """Read temperature parameters from config.
        
        Config keys:
            trust_gate.soft_temp_conf (float, default=10.0)
            trust_gate.soft_temp_struct (float, default=10.0)
            trust_gate.soft_bias (float, default=0.0)
        """
    
    def compute_weights(
        self,
        confidence: Tensor,  # [B, H, W], float32, (0, 1)
        struct: Tensor,       # [B, H, W], float32, [0, 1]
    ) -> Tensor:
        """Compute continuous trust weights.
        
        Args:
            confidence: Per-pixel teacher confidence
            struct: Per-pixel structural complexity
        
        Returns:
            Weight map, shape [B, H, W], dtype float32, values [0.0, 1.0]
        
        Formula: w = sigmoid(β₀ + β₁·confidence - β₂·structure)
        Monotonicity: w non-decreasing in confidence, non-increasing in structure
        """
```

### Invariants
- Hard: output values exactly {0.0, 1.0}
- Soft: output values in [0.0, 1.0]
- Both: output shape matches input shape exactly
- Both: no gradient computation (inference only)

---

## 3. `satg/losses.py` — SATGLoss

```python
class SATGLoss(nn.Module):
    def __init__(self, ignore_index: int = 255) -> None:
        """Initialize SATG loss.
        
        Args:
            ignore_index: Class index to ignore in CE computation (default 255)
        """
    
    def forward(
        self,
        student_logits: Tensor,   # [B, C, H, W], C=19
        pseudo_labels: Tensor,     # [B, H, W], int64, values {0..18}
        trust_weights: Tensor,     # [B, H, W], float32, values [0.0, 1.0]
    ) -> Tensor:
        """Compute trust-weighted cross-entropy loss.
        
        Args:
            student_logits: Student model output logits
            pseudo_labels: Teacher pseudo-labels (argmax)
            trust_weights: Trust gate weights (hard or soft)
        
        Returns:
            Scalar loss tensor, dtype float32
        
        Formula:
            per_pixel_ce = F.cross_entropy(logits, labels, reduction='none', ignore_index=255)
            loss = (per_pixel_ce * trust_weights).sum() / (trust_weights.sum() + 1e-8)
        
        Edge case:
            If trust_weights.sum() == 0, return torch.tensor(0.0, device=logits.device)
        """
```

### Invariants
- Output is scalar tensor (no batch dimension)
- Output ≥ 0.0 (cross-entropy is non-negative, weights are non-negative)
- No NaN or Inf in output (guarded by epsilon and zero-sum check)
- ignore_index=255 pixels excluded from both numerator and denominator

---

## 4. `models/ema.py` — EMAModel

```python
class EMAModel:
    def __init__(self, model: nn.Module, momentum: float = 0.999) -> None:
        """Create EMA teacher as shadow copy of student.
        
        Args:
            model: Student model to copy
            momentum: EMA decay factor (default 0.999)
        
        Behavior:
            - Copies all parameters from model
            - Sets eval mode
            - All params require_grad=False
            - Shadow stored as dict (not nn.Parameters)
        """
    
    def update(self, student: nn.Module) -> None:
        """Update EMA weights from student.
        
        Formula: teacher_param = momentum * teacher_param + (1 - momentum) * student_param
        
        Scheduled momentum (per iteration):
            alpha_t = min(1 - 1/(iter+1), alpha_target)
        """
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through EMA model (inference only)."""
```

### Invariants
- EMA model always in eval mode
- No gradient computation through EMA model
- Shadow params stored as Python dict, not nn.Parameter
- `state_dict()` returns shadow dict for checkpointing

---

## 5. `precompute/compute_heatmaps.py` — CLI Script

```
Usage: python -m precompute.compute_heatmaps --data_root <path> [--num_workers 8]

Arguments:
    --data_root    Path to Cityscapes leftImg8bit/train/ directory
    --num_workers  Number of parallel workers (default: 8)

Behavior:
    1. Recursively find all *.png images under data_root
    2. For each image: load with cv2, compute StructuralPrior.compute()
    3. Save as {stem}_satg_heatmap.npy next to original image
    4. Use multiprocessing.Pool with tqdm progress bar
    5. Print statistics: min/max/mean of 20 random heatmaps

Output:
    - One .npy file per input image
    - Console output: progress bar + statistics
```

### Invariants
- Deterministic output (same input → same heatmap)
- Output naming: `{image_stem}_satg_heatmap.npy`
- Output dtype: float32
- Output shape: (H, W) matching input image

---

## 6. `training/trainer.py` — Training Loop

```python
class SATGTrainer:
    def __init__(self, cfg: OmegaConf) -> None:
        """Initialize all components from config.
        
        Components:
            - student model (DeepLabV3+ ResNet50)
            - teacher model (EMAModel)
            - source loader (GTA5)
            - target loader (Cityscapes + heatmaps)
            - optimizer (SGD with param groups)
            - lr_scheduler (PolynomialLR)
            - trust gate (Hard or Soft based on config)
            - satg_loss (SATGLoss)
            - evaluator
            - wandb logger
        """
    
    def train(self) -> None:
        """Main training loop.
        
        For each iteration:
            1. Load source batch (images, labels)
            2. Load target batch (images, heatmaps)
            3. Source forward → source_loss (CE)
            4. Teacher forward (no_grad) → pseudo_labels, confidence
            5. Student target forward → student_logits
            6. Trust gate → trust_weights
            7. SATG loss → target_loss
            8. total_loss = source_loss + λ * target_loss
            9. Backward + optimizer step
            10. EMA update
            11. Log metrics
            12. Eval every N iterations
            13. Save best checkpoint
        """
```

### Invariants
- Teacher never receives target ground truth labels
- EMA update happens after every optimizer step
- lr_scheduler.step() called per iteration
- All metrics logged to WandB
- Best checkpoint saved by val mIoU
