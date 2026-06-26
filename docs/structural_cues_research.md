# Structural Cues for Structure-Aware Trust Gating (SATG)

## 1. Edge Density as a Structural Feature

### Definition
Edge density measures the proportion of edge pixels within a local region of an image. It quantifies how "busy" or structurally complex a local patch is by counting the fraction of pixels that fall on detected edges.

### Computation from Canny Edge Map

The Canny edge detector produces a binary edge map where each pixel is classified as edge (1) or non-edge (0). Edge density is computed as:

**Edge Density** = (Number of edge pixels in patch) / (Total pixels in patch)

For a local patch of size H×W:
```
F_edgeness = |{p | Mag(p) > T}| / N
```
where:
- `Mag(p)` is the gradient magnitude at pixel p
- `T` is a threshold
- `N` = H × W is the total number of pixels in the patch

### Canny Edge Detection Process (from Wikipedia)
1. **Gaussian Filter**: Smooth image with Gaussian kernel (typically 5×5, σ=1.4)
2. **Gradient Computation**: Compute intensity gradients using Sobel operators (Gx, Gy)
3. **Gradient Magnitude**: G = √(Gx² + Gy²)
4. **Non-maximum Suppression**: Thin edges to single-pixel width
5. **Double Thresholding**: Apply high and low thresholds
6. **Hysteresis Tracking**: Connect weak edges to strong edges

### Typical Kernel Sizes
- **Gaussian smoothing kernel**: 3×3 to 7×7 (5×5 is most common, σ=1.4)
- **Sobel operators**: 3×3 for horizontal/vertical gradients
- **Edge density computation window**: 7×7 to 31×31 (application-dependent)
- **For driving scenes**: 15×15 to 25×25 patches are typical for structural analysis

### Practical Considerations
- Smaller kernels detect fine edges but are noise-sensitive
- Larger kernels detect coarse structure but may miss small objects
- For Cityscapes (1024×2048), patch sizes of 16×16 to 32×32 at feature map resolution are common

---

## 2. Local Variance as a Structural Feature

### Definition
Local variance measures the spread of pixel intensities within a local neighborhood. High variance indicates texture complexity, clutter, or structural detail. Low variance indicates smooth, homogeneous regions.

### Computation (Sliding-Window Operation)

For each pixel (x, y), compute variance over a local window of size k×k:

```
μ(x,y) = (1/k²) Σ I(x+i, y+j)  for i,j in [-k/2, k/2]

σ²(x,y) = (1/k²) Σ [I(x+i, y+j) - μ(x,y)]²
```

Where:
- I(x,y) is the intensity at pixel (x,y)
- μ(x,y) is the local mean
- σ²(x,y) is the local variance
- k is the window size (typically 5×5, 7×7, or 11×11)

### Fast Computation Using Box Filters
```
Sum = box_filter(I)
Sum_sq = box_filter(I²)
Variance = (Sum_sq / k²) - (Sum / k²)²
```

### How It Captures Texture Complexity and Clutter

**Texture Complexity:**
- High local variance → complex texture (tree foliage, brick patterns, pavement texture)
- Low local variance → smooth surfaces (sky, road surface, building walls)

**Clutter Detection:**
- Regions with many objects (vehicles, pedestrians, signs) have high variance
- Background clutter creates high-frequency intensity changes
- Mixed boundaries between objects create elevated variance

**Scale Dependence:**
- Small windows (5×5): Detect fine texture details
- Medium windows (11×11): Capture object boundaries
- Large windows (21×21): Measure overall scene complexity

### Typical Parameters for Urban Driving
- Window size: 7×7 to 15×15 at input resolution
- Applied to grayscale or single-channel feature maps
- Often computed at multiple scales (pyramid approach)

---

## 3. Other Classical Image Features for Structural Complexity

### 3.1 Gradient Magnitude Maps

**Definition:** The magnitude of the intensity gradient at each pixel, computed as:
```
G = √(Gx² + Gy²)
```
where Gx and Gy are horizontal and vertical gradients (Sobel, Scharr, or Prewitt operators).

**Relevance to Structural Complexity:**
- High gradient magnitude indicates strong edges/boundaries
- Dense gradient regions indicate structural complexity
- Gradient orientation histograms reveal edge distribution patterns

**Application in Driving Scenes:**
- Road boundaries have strong gradients
- Vehicle edges create gradient clusters
- Vegetation creates scattered gradient responses

**Key Reference:** Sobel operator (3×3 kernels), Scharr filter (better rotational symmetry)

### 3.2 Laplacian / Laplacian of Gaussian (LoG)

**Definition:** Second-order derivative operator measuring local curvature:
```
∇²f = ∂²f/∂x² + ∂²f/∂y²
```

**Laplacian of Gaussian (LoG):**
```
LoG = -∇²Gσ * I
```
where Gσ is a Gaussian with scale σ.

**Relevance:**
- Zero crossings indicate edges
- Response magnitude indicates edge strength
- Scale-space LoG responses at multiple scales detect features at different sizes

**Application:**
- Blob detection (pedestrians, traffic signs as blobs)
- Corner detection (building corners, intersection features)
- Texture analysis (response distribution indicates complexity)

**Typical Scales for Driving:** σ = 1.0 to 3.0 pixels at input resolution

### 3.3 Local Entropy

**Definition:** Shannon entropy computed over a local histogram:
```
H = -Σ p(i) log₂(p(i))
```
where p(i) is the probability of intensity i in the local window.

**Relevance:**
- High entropy → diverse intensity distribution → complex texture
- Low entropy → uniform distribution → simple/smooth regions
- Information-theoretic measure of local unpredictability

**Application in Driving:**
- Road surface: low entropy (uniform)
- Vegetation: high entropy (random texture)
- Building facades: medium entropy (regular patterns)

**Typical Parameters:** 8×8 to 16×16 windows, 16-64 gray levels for histogram

### 3.4 Haralick Texture Features (GLCM-based)

**Definition:** Features derived from Gray-Level Co-occurrence Matrix (GLCM):
- **Contrast:** Local variation in GLCM → edge intensity
- **Correlation:** Linear dependency of gray levels
- **Energy (Angular Second Moment):** Uniformity of GLCM
- **Homogeneity (Inverse Difference Moment):** Closeness of GLCM diagonal

**Key Reference:** Haralick et al., "Textural Features for Image Classification," IEEE Trans. SMC, 1973

**Application:**
- Contrast → structural complexity
- Energy → texture regularity
- Homogeneity → smoothness

### 3.5 Local Binary Patterns (LBP)

**Definition:** Encodes local texture by comparing each pixel with its neighbors:
- For each pixel, compare with 8 neighbors (clockwise)
- Create 8-bit binary code (1 if neighbor > center, 0 otherwise)
- Compute histogram of these codes in local region

**Relevance:**
- Rotation-invariant LBP captures texture patterns
- Uniform LBP patterns reduce dimensionality (59 patterns from 256)
- Histogram distribution indicates texture complexity

**Application:**
- Distinguishes pavement textures
- Identifies vegetation patterns
- Separates smooth vs. rough surfaces

**Key Reference:** Ojala et al., "Performance Evaluation of Texture Measures with Classification," ICPR 1994

### 3.6 Laws Texture Energy Measures

**Definition:** Convolve image with 1D vectors to create 2D masks:
- L5 = [1, 4, 6, 4, 1] (Level)
- E5 = [-1, -2, 0, 2, 1] (Edge)
- S5 = [-1, 0, 2, 0, -1] (Spot)
- R5 = [1, -4, 6, -4, 1] (Ripple)

**Relevance:**
- L5E5/E5L5 → edge content (vertical/horizontal)
- S5S5 → spot detection
- R5R5 → ripple patterns

**Application:**
- Edge maps from Laws filters indicate structural boundaries
- Energy maps provide texture complexity descriptors

### 3.7 Fractal Dimension

**Definition:** Measures self-similarity across scales:
```
D = lim [log(N(ε)) / log(1/ε)]
```
where N(ε) is the number of boxes of size ε needed to cover the structure.

**Relevance:**
- High fractal dimension → complex, irregular texture
- Low fractal dimension → smooth, regular structure
- Captures multi-scale complexity

**Application:**
- Vegetation: high fractal dimension (D ≈ 2.5-2.8)
- Road surface: low fractal dimension (D ≈ 2.1-2.3)
- Building facades: medium fractal dimension

---

## 4. Existing Papers on Structural Maps for Image Analysis

### 4.1 Saliency Priors

**Paper:** "Contextual Encoder-Decoder Network for Visual Saliency Prediction" (Kroner et al., 2019, arXiv:1902.06634)
- Uses multi-scale features and contextual information for saliency prediction
- Saliency maps implicitly capture structural complexity
- Demonstrates that complex scenes require multi-scale analysis

**Paper:** "Structure-Consistent Weakly Supervised Salient Object Detection with Local Saliency Coherence" (Yu et al., 2020, arXiv:2012.04404)
- Proposes structure consistency loss for saliency detection
- Uses local coherence to propagate labels based on image features
- Shows that structural information improves saliency detection

**Paper:** "Sharp Eyes: A Salient Object Detector Working The Same Way as Human Visual Characteristics" (Zhu et al., 2023, arXiv:2301.07431)
- Separates object from scene before fine segmentation
- Addresses cluttered backgrounds where targets have similar color/texture to background
- Uses fractal structure to produce saliency features with expanded boundaries

### 4.2 Scene Complexity in Urban Driving

**Paper:** "Pixel-wise Energy-biased Abstention Learning for Anomaly Segmentation on Complex Urban Driving Scenes" (Tian et al., 2021, arXiv:2111.12264)
- Addresses anomaly segmentation in complex urban scenes
- Uses energy-based models to capture pixel-level complexity
- Demonstrates that complex scenes require adaptive handling

**Paper:** "Semantic-Guided Inpainting Network for Complex Urban Scenes Manipulation" (Ardino et al., 2020, arXiv:2010.09334)
- Manipulates complex urban scenes with multiple semantics
- Uses semantic segmentation to model content and structure
- Shows that complex scenes contain cluttered/ambiguous objects

**Paper:** "Elastic Interaction Energy-Informed Real-Time Traffic Scene Perception" (Feng et al., 2023, arXiv:2310.01449)
- Addresses fine and complex geometric objects in traffic scenes
- Uses elastic interaction energy loss for better segmentation
- Demonstrates that fine-scale structures (pedestrians, signs, lanes) are most challenging

### 4.3 Clutter and Complexity Maps

**Paper:** "STC: A Simple to Complex Framework for Weakly-supervised Semantic Segmentation" (Wei et al., 2015, arXiv:1509.03150)
- Distinguishes simple images (single category, clean background) from complex images (multiple categories, cluttered background)
- Uses saliency maps to identify simple vs. complex regions
- Demonstrates that complexity classification improves segmentation

**Paper:** "RF-DETR Object Detection vs YOLOv12 for Complex Orchard Environments" (Sapkota et al., 2025, arXiv:2504.13099)
- Compares architectures for detecting objects in cluttered scenes
- Addresses label ambiguity, occlusions, and background blending
- Shows that transformer-based architectures excel in complex spatial scenarios

**Paper:** "GLUE: Global-Local Unified Encoding for Imitation Learning via Key-Patch Tracking" (Chen et al., 2025, arXiv:2509.23220)
- Addresses Out-of-Distribution settings characterized by clutter and occlusion
- Shows that visual attention can be diluted in complex scenes
- Uses local representations to handle clutter

### 4.4 Structural Entropy and Information Theory

**Paper:** "Breaking Degradation Coupling: A Structural Entropy Guided Decoupled Framework" (Li et al., 2026, arXiv:2604.22886)
- Uses structural entropy to measure information content
- Aggregates features under structural-entropy criterion
- Yields representations with structural fidelity

**Paper:** "VLA-InfoEntropy: A Training-Free Vision-Attention Information Entropy Approach" (Liu et al., 2026, arXiv:2604.05323)
- Uses image entropy to quantify grayscale distribution characteristics
- Visual entropy identifies texture-rich or structurally informative regions
- Attention entropy captures distribution of attention scores

### 4.5 Texture Analysis in Medical and Other Domains

**Paper:** "Texture Feature Analysis for Classification of Early-Stage Prostate Cancer in mpMRI" (Muftah et al., 2024, arXiv:2406.15571)
- Analyzes Haralick texture features and local binary patterns
- Shows that many features are correlated
- Identifies small set of features that determine classification

**Paper:** "Multiscale Analysis for Improving Texture Classification" (Ataky et al., 2022, arXiv:2204.09841)
- Uses Gaussian-Laplacian pyramid for multiscale analysis
- Combines bio-inspired texture descriptors, information-theoretic measures, GLCM features, and Haralick features
- Demonstrates importance of multiscale analysis

---

## 5. Structural Complexity in Cityscapes

### 5.1 What "Structural Complexity" Looks Like Visually

**High Structural Complexity:**
- **Intersections:** Multiple road segments, crosswalks, traffic lights, signs, pedestrians, vehicles, buildings at various angles
- **Dense urban areas:** Many overlapping objects, complex occlusion patterns
- **Vegetation-heavy scenes:** Trees with complex branching, overlapping leaves, mixed with built structures
- **Mixed backgrounds:** Buildings with windows, signs, graffiti, mixed textures

**Low Structural Complexity:**
- **Highway segments:** Smooth road, uniform sky, simple lane markings
- **Open parking lots:** Large homogeneous surfaces, sparse objects
- **Simple road segments:** Clear road-sky boundary, minimal objects

### 5.2 Cityscapes Scene Types by Complexity

**Very High Complexity:**
1. **Urban intersections** (e.g., Cityscapes images with multiple crossing roads):
   - Many object categories visible simultaneously
   - Complex spatial relationships
   - High edge density at object boundaries
   - High local variance due to mixed textures

2. **Dense pedestrian areas:**
   - Many small objects (pedestrians) with complex silhouettes
   - Occlusion between pedestrians
   - Mixed with background clutter (signs, poles, vegetation)

3. **Complex intersections with vegetation:**
   - Overlapping tree canopies with road/buildings
   - High-frequency texture from leaves
   - Mixed structural elements (natural + man-made)

**Medium Complexity:**
1. **Urban streets with parked cars:**
   - Clear road structure
   - Moderate number of objects
   - Some texture variation (building facades, signs)

2. **Residential areas:**
   - Buildings with regular patterns (windows, doors)
   - Moderate vegetation
   - Clear road-sidewalk boundaries

**Low Complexity:**
1. **Highway/rural roads:**
   - Smooth road surface
   - Clear sky region
   - Minimal objects
   - Low edge density

2. **Open areas:**
   - Large homogeneous regions (parking lots, plazas)
   - Simple geometric structure

### 5.3 Quantitative Indicators from Cityscapes

Based on typical Cityscapes images (1024×2048 resolution):

**Edge Density Values (typical ranges):**
- Sky: 0.01-0.05 (very low)
- Road: 0.02-0.08 (low)
- Building walls: 0.05-0.15 (medium)
- Vegetation: 0.10-0.25 (medium-high)
- Complex intersections: 0.15-0.30 (high)
- Mixed vegetation+buildings: 0.20-0.35 (very high)

**Local Variance Values (typical ranges):**
- Sky: 50-150 (low)
- Road surface: 100-300 (low-medium)
- Building facades: 200-500 (medium)
- Vegetation: 400-800 (high)
- Complex scenes: 500-1000+ (very high)

**Local Entropy Values:**
- Uniform regions: 2-3 bits
- Moderate texture: 4-5 bits
- Complex texture: 5-7 bits

### 5.4 Structural Complexity Patterns in Urban Driving

**Pattern 1: Vertical Structure Dominance**
- Buildings create strong vertical edges
- Poles, signs add vertical elements
- Trees create vertical texture
- Creates high edge density in vertical direction

**Pattern 2: Horizontal Layering**
- Road surface at bottom
- Vehicles/pedestrians in middle
- Buildings/sky at top
- Creates horizontal variance gradients

**Pattern 3: Occlusion Complexity**
- Vehicles occluding each other
- Pedestrians behind poles/trees
- Buildings occluding other buildings
- Creates high local variance at occlusion boundaries

**Pattern 4: Texture Gradient**
- Foreground objects: high detail, high variance
- Background objects: lower detail, lower variance
- Creates scale-dependent complexity

---

## 6. Implementation Recommendations for SATG

### 6.1 Multi-Scale Feature Extraction

```
For each scale s in [1, 2, 4, 8]:
    1. Compute edge map using Canny (or differentiable approximation)
    2. Compute local variance with window size k_s
    3. Compute gradient magnitude
    4. Compute local entropy
    5. Aggregate features per scale
```

### 6.2 Feature Fusion Strategy

**Concatenation approach:**
```
structural_features = concat([
    edge_density_map,
    local_variance_map,
    gradient_magnitude_map,
    local_entropy_map
])
```

**Weighted combination:**
```
complexity_score = w1 * edge_density + 
                   w2 * local_variance + 
                   w3 * gradient_magnitude + 
                   w4 * local_entropy
```

### 6.3 Kernel Size Recommendations

| Feature | Small Window | Medium Window | Large Window |
|---------|-------------|---------------|--------------|
| Edge Density | 7×7 | 15×15 | 31×31 |
| Local Variance | 5×5 | 11×11 | 21×21 |
| Gradient Magnitude | 3×3 (Sobel) | 5×5 | 7×7 |
| Local Entropy | 8×8 | 16×16 | 32×32 |

### 6.4 Differentiable Approximations

For end-to-end training:
- **Edge detection:** Use learnable edge detectors or soft thresholding
- **Variance:** Use differentiable box filters
- **Entropy:** Use soft histogram approximation
- **Gradient:** Use Sobel-like convolutional layers

---

## 7. Key References

1. Canny, J. (1986). "A Computational Approach to Edge Detection." IEEE PAMI, 8(6):679-698.
2. Haralick, R.M., et al. (1973). "Textural Features for Image Classification." IEEE SMC, 3(6):610-621.
3. Ojala, T., et al. (1994). "Performance Evaluation of Texture Measures with Classification." ICPR.
4. Laws, K. (1980). "Textured Image Segmentation." Ph.D. Dissertation, USC.
5. Lindeberg, T. (1998). "Edge Detection and Ridge Detection with Automatic Scale Selection." IJCV, 30(2):117-154.
6. Kroner, A., et al. (2019). "Contextual Encoder-Decoder Network for Visual Saliency Prediction." arXiv:1902.06634.
7. Tian, Y., et al. (2021). "Pixel-wise Energy-biased Abstention Learning for Anomaly Segmentation on Complex Urban Driving Scenes." arXiv:2111.12264.
8. Ardino, P., et al. (2020). "Semantic-Guided Inpainting Network for Complex Urban Scenes Manipulation." arXiv:2010.09334.
9. Wei, Y., et al. (2015). "STC: A Simple to Complex Framework for Weakly-supervised Semantic Segmentation." arXiv:1509.03150.

---

*Document generated for SATG research. Last updated: 2026-06-24*
