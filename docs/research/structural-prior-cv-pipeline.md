# Structural Prior CV Pipeline — Reference

## 1. `cv2.Canny` — Exact Signature & Urban Scene Parameters

```python
cv2.Canny(
    image,           # Input: uint8 single-channel (grayscale) image
    threshold1,      # int/float: lower hysteresis threshold
    threshold2,      # int/float: upper hysteresis threshold
    apertureSize=3,  # int: Sobel kernel size, must be 3, 5, or 7
    L2gradient=False # bool: False = L1 norm (|Gx|+|Gy|), True = L2 norm (sqrt(Gx²+Gy²))
) -> np.ndarray      # Returns: uint8 binary mask (0 or 255)
```

**Output dtype:** `np.uint8` — values are exactly `0` or `255`, not `0`/`1`.

### Recommended Parameters for Urban Scenes (GTA5 / Cityscapes)

| Scenario | threshold1 | threshold2 | apertureSize | L2gradient | Rationale |
|---|---|---|---|---|---|
| **Conservative (low FP)** | 100 | 200 | 3 | False | Standard OpenCV default; picks up strong structural edges (building boundaries, road edges, vehicles) |
| **Moderate (balanced)** | 50 | 150 | 3 | False | Captures finer edges (lane markings, sidewalk texture); good for density maps |
| **Aggressive (low FN)** | 25 | 75 | 3 | True | Captures near-texture edges; use for very smooth/low-contrast domains |

**Recommendation for SATG:** Start with `(50, 150)` — it captures the edge-density contrast between sky (~0.01-0.05) and intersections (~0.15-0.30) well. Use `L2gradient=False` for speed; L2 gives marginally better edge localization but ~5% slower.

**Input requirement:** Must be `uint8` grayscale. For RGB input:
```python
gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
```

---

## 2. Local Variance — Efficient Computation

### Formula
```
var(I) = E[I²] - E[I]² = blur(I²) - blur(I)²
```

### Implementation with `cv2.blur`

```python
import cv2
import numpy as np

def local_variance(gray: np.ndarray, ksize: int = 15) -> np.ndarray:
    """
    Compute local variance using box blur identity.
    
    Args:
        gray: uint8 or float image, single channel
        ksize: kernel size for box filter (odd int)
    Returns:
        float64 array, same spatial dims as input
    """
    I = gray.astype(np.float64)
    I2 = I * I
    mu = cv2.blur(I2, (ksize, ksize))    # E[I²]
    mu2 = cv2.blur(I, (ksize, ksize))    # E[I]
    var = mu - mu2 * mu2                  # var = E[I²] - E[I]²
    return var
```

### Key Signatures

```python
cv2.blur(
    src,                       # Input array (any depth)
    ksize,                     # Size(width, height) — tuple of two ints
    anchor=(-1, -1),           # Point: center of kernel
    borderType=BORDER_DEFAULT  # int
) -> np.ndarray
```

### Equivalent with `scipy.ndimage`

```python
from scipy.ndimage import uniform_filter

def local_variance_scipy(gray: np.ndarray, ksize: int = 15) -> np.ndarray:
    I = gray.astype(np.float64)
    mu = uniform_filter(I, size=ksize)
    mu2 = uniform_filter(I * I, size=ksize)
    return mu2 - mu * mu
```

**Performance note:** `cv2.blur` is ~2-3x faster than `scipy.ndimage.uniform_filter` for typical image sizes due to optimized SIMD/NEON paths.

### Recommended Kernel Sizes

| ksize | Behavior | Use case |
|---|---|---|
| 11 | Fine-grained variance | Texture detail |
| 15 | Balanced | **Default for SATG** — captures local patch complexity |
| 25 | Coarse | Region-level variance, less noisy |

---

## 3. Edge Density — Computation Methods

Edge density = fraction of edge pixels in a local patch. Two approaches:

### Method A: `cv2.filter2D` (OpenCV, fast)

```python
def edge_density_filter2d(
    edge_map: np.ndarray,      # uint8 binary (0/255) from Canny
    ksize: int = 15
) -> np.ndarray:
    """
    Compute edge density via uniform filtering of binary edge map.
    
    Args:
        edge_map: uint8 (0 or 255) from cv2.Canny
        ksize: patch size for density computation
    Returns:
        float64 array in [0.0, 1.0]
    """
    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
    density = cv2.filter2D(edge_map.astype(np.float32), -1, kernel)
    return density / 255.0  # normalize [0, 255] -> [0.0, 1.0]
```

### Method B: `cv2.blur` (even faster, equivalent for uniform kernel)

```python
def edge_density_blur(
    edge_map: np.ndarray,
    ksize: int = 15
) -> np.ndarray:
    density = cv2.blur(edge_map.astype(np.float32), (ksize, ksize))
    return density / 255.0
```

`cv2.blur` is equivalent to `cv2.filter2D` with a normalized box kernel and is marginally faster (~5-10%).

### Method C: `torch.nn.functional.avg_pool2d` (GPU batch)

```python
import torch
import torch.nn.functional as F

def edge_density_torch(
    edge_map_batch: torch.Tensor,  # (B, 1, H, W), uint8 or float
    ksize: int = 15
) -> torch.Tensor:
    """
    Batch edge density on GPU.
    Returns: (B, 1, H, W) float in [0, 1]
    """
    x = edge_map_batch.float() / 255.0
    pad = ksize // 2
    density = F.avg_pool2d(x, kernel_size=ksize, stride=1, padding=pad)
    return density
```

### Recommended Kernel Sizes for Edge Density

| ksize | Coverage (at 1024×512) | Behavior |
|---|---|---|
| 11 | ~11×11 pixels | Fine-grained, noisy |
| 15 | ~15×15 pixels | **Default for SATG** |
| 21 | ~21×21 pixels | Smoother, region-level |
| 31 | ~31×31 pixels | Very coarse, sky vs. intersection |

---

## 4. Performance Estimates — 2,975 Images at 1024×512

### Per-Image Timing (single CPU core, ~3.5 GHz)

| Operation | Time per image | Notes |
|---|---|---|
| `cv2.cvtColor` (RGB→Gray) | ~1-2 ms | SIMD-optimized |
| `cv2.Canny(gray, 50, 150)` | ~3-5 ms | Sobel + NMS + hysteresis |
| `cv2.blur(I², ksize=15)` | ~1-2 ms | |
| `cv2.blur(I, ksize=15)` | ~1-2 ms | |
| Variance subtraction | ~0.1 ms | element-wise |
| `cv2.blur(edge_map, ksize=15)` | ~1-2 ms | |
| Normalize to [0,1] | ~0.1 ms | |
| **Total per image** | **~8-13 ms** | |

### Multiprocessing Estimates

Using `multiprocessing.Pool` with `os.cpu_count()` workers:

| Cores | Time per image (wall) | Total time (2,975 images) | Throughput |
|---|---|---|---|
| 1 | ~10 ms | ~30 sec | ~100 img/s |
| 4 | ~10 ms | ~8 sec | ~370 img/s |
| 8 | ~10 ms | ~4 sec | ~740 img/s |
| 16 | ~10 ms | ~2 sec | ~1,490 img/s |
| 32 | ~10 ms | ~1 sec | ~2,975 img/s |

**Note:** Per-image time stays ~10ms because the GIL doesn't affect CPU-bound cv2 calls (OpenCV releases GIL). The bottleneck is I/O for writing .npy files (~2ms each).

### Multiprocessing Skeleton

```python
import os
import cv2
import numpy as np
from multiprocessing import Pool
from functools import partial

def compute_structural_prior(
    img_path: str,
    output_dir: str,
    canny_low: int = 50,
    canny_high: int = 150,
    variance_ksize: int = 15,
    density_ksize: int = 15,
) -> dict:
    """Compute edge density + local variance for one image."""
    # Read & convert
    rgb = cv2.imread(img_path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    
    # Edge map
    edges = cv2.Canny(gray, canny_low, canny_high)
    
    # Edge density
    density = cv2.blur(edges.astype(np.float32), (density_ksize, density_ksize)) / 255.0
    
    # Local variance
    I = gray.astype(np.float64)
    I2 = I * I
    mu = cv2.blur(I2, (variance_ksize, variance_ksize))
    mu2 = cv2.blur(I, (variance_ksize, variance_ksize))
    var = mu - mu2 * mu2
    
    # Save
    base = os.path.splitext(os.path.basename(img_path))[0]
    np.save(os.path.join(output_dir, f"{base}_density.npy"), density.astype(np.float32))
    np.save(os.path.join(output_dir, f"{base}_variance.npy"), var.astype(np.float32))
    
    return {"path": img_path, "density_range": (density.min(), density.max()),
            "variance_range": (var.min(), var.max())}


def batch_compute(img_paths: list, output_dir: str, workers: int = None):
    if workers is None:
        workers = os.cpu_count()
    os.makedirs(output_dir, exist_ok=True)
    fn = partial(compute_structural_prior, output_dir=output_dir)
    with Pool(workers) as pool:
        results = pool.map(fn, img_paths)
    return results
```

---

## 5. Dtype Details & Normalization

### Output dtypes

| Operation | Output dtype | Value range |
|---|---|---|
| `cv2.Canny(...)` | `np.uint8` | {0, 255} — binary, NOT {0, 1} |
| `cv2.blur(uint8, ksize)` | `np.uint8` | [0, 255] (default) |
| `cv2.blur(uint32, ksize)` | `np.uint32` | [0, 255 * ksize²] |
| `cv2.filter2D(uint8, -1, float32_kernel)` | `np.uint8` | [0, 255] |
| `cv2.filter2D(uint8, cv2.CV_32F, ...)` | `np.float32` | [0.0, 1.0] if kernel normalized |

### How to Normalize Edge Density to [0, 1]

Since Canny output is `{0, 255}` (uint8):

```python
# After blur:
density = cv2.blur(edges.astype(np.float32), (k, k))  # [0.0, 255.0] float32
density /= 255.0  # now [0.0, 1.0]
```

Alternative (no intermediate float):
```python
density_uint8 = cv2.blur(edges, (k, k))  # uint8 [0, 255]
density = density_uint8.astype(np.float32) / 255.0
```

### Save as `.npy`

```python
# Recommended: float32 for disk space and speed
np.save("density.npy", density.astype(np.float32))   # 1024×512 × 4 bytes = 2 MB per file
np.save("variance.npy", var.astype(np.float32))       # same

# For reference: float64 would be 4 MB per file — not worth it
```

### Memory Budget

Per image: 2 maps × 1024 × 512 × 4 bytes (float32) = **4 MB per image**.
Total: 2,975 × 4 MB = ~12 GB disk for all .npy files.

---

## Summary Table

| Parameter | Value | Notes |
|---|---|---|
| Canny thresholds | (50, 150) | Balanced for urban scenes |
| Canny apertureSize | 3 | Default, sufficient |
| L2gradient | False | ~5% faster, acceptable accuracy |
| Variance ksize | 15 | Good balance of locality/smoothness |
| Density ksize | 15 | Matches variance scale |
| Edge density normalization | `/ 255.0` | Canny outputs {0, 255} |
| Output dtype | `np.float32` | 2 MB per map |
| Total disk | ~12 GB | 2,975 × 2 × 2 MB |
| Total compute (8 cores) | ~4 sec | All CPU-bound, no GPU needed |
| Pipeline per image | ~10 ms | Canny + blur + variance + save |
