# SATG — Experiments

## Diagnostic Test (Complete)

### Purpose
Validate the core SATG hypothesis: do high-confidence pseudo-label errors
concentrate in structurally complex regions?

### Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Model** | DAFormer (MiT-B5 encoder, DAFormer decoder with sep-ASPP) |
| **Checkpoint** | GTA→Cityscapes UDA (HRDA-inspired warmup, thing-class feature distance, rare-class sampling, pseudo-label crop) |
| **Training iterations** | 40,000 |
| **Source domain** | GTA (synthetic) |
| **Target domain** | Cityscapes (real, validation set) |
| **Dataset size** | 500 images |
| **Image resolution** | 1024 × 2048 (raw Cityscapes val) |
| **Model resolution** | Native 1024 × 2048 (logits upsampled from 1/4 stride features) |
| **Device** | Apple MPS (M4) |
| **Confidence threshold** | 0.90 (softmax max-prob) |
| **Structural prior** | Canny (50,150) + GaussianBlur (5×5, σ=1.0), α=β=0.5 |
| **Statistics** | 10 decile bins per-pixel, accumulated over all images |
| **Total pixels analyzed** | 917,018,489 |
| **Skipped pixels (NaN)** | 131,557,511 (void/NaN confidence near void GT labels) |
| **Processing time** | ~5,400–27,000 ms per image (mean ~9,500 ms) |
| **Total wall time** | ~60 minutes for 500 images |

### Data Pipeline

1. CityscapesDataset loads raw 1024×2048 RGB images and ground-truth
   label maps (labelIds converted to trainIds via lookup table).
2. Structural complexity computed on the raw RGB image.
3. Model inference runs at native resolution; 1/4-stride logits are
   bilinearly upsampled to match input resolution.
4. Per-pixel statistics accumulated per decile bin:
   - Bin 0: complexity [0.0, 0.1) ... Bin 9: [0.9, 1.0]
   - For each pixel: total count, error count (pred ≠ gt AND conf > 0.9),
     confidence sum, confidence-squared sum
5. Overall error rate computed from aggregated counts.

### Important Implementation Details

- **No resizing**: The diagnostic pipeline bypasses the standard MMSegmentation
  test pipeline and feeds full-resolution images directly. This differs from
  the standard DAFormer test-time protocol (resize to height=512, keep aspect
  ratio). This is acceptable for relative comparisons across bins but may
  affect absolute error rates slightly.
- **MPS shader cache warmup**: A 128×128 dummy forward pass is run before
  the main loop to avoid on-the-fly graph recompilation.
- **MPS cache flushing**: `torch.mps.empty_cache()` called after each image
  to prevent memory fragmentation.
- **NaN handling**: Pixels with NaN confidence (rare, occurs near void labels
  at image boundaries) are excluded entirely from analysis.

### Statistical Power

With 917M pixels and ~27M errors, the test has extremely high statistical
power. Any real difference of 0.1% or more in error rates would be detected
with high confidence. The decile binning provides 10 discrete complexity
levels for trend analysis.

### Limitations of This Experiment

1. **Single model check**: Only DAFormer (GTA→CS) tested. Results may not
   generalise to other architectures (DeepLabV2, SegFormer) or domain gaps
   (synthia→cityscapes, cityscapes→darkzurich).
2. **Single confidence threshold**: 0.9 is standard but may miss patterns
   at lower thresholds.
3. **Per-image normalisation**: The structural prior normalises per image,
   which limits cross-image bin interpretability (see 04-RESULTS.md).
4. **No class decomposition**: Error rates are aggregated across all
   classes. Different classes may have different complexity-error profiles.
5. **Single prior configuration**: Only (50,150) Canny thresholds and
   5×5 Gaussian kernel tested. Other configurations may yield different
   complexity distributions.

## Planned Experiments (Future Work)

| Experiment | Purpose | Status |
|------------|---------|--------|
| Soft-weighting SATG training | Measure mIoU improvement | Not started |
| Hard-gating SATG training | Baseline comparison | Not started |
| Different confidence thresholds (0.8, 0.95) | Robustness check | Not started |
| Per-class error decomposition | Identify confounding with class frequency | Not started |
| Other priors (Sobel, Gabor, learned) | Prior quality comparison | Not started |
| Synthia→Cityscapes domain gap | Generalisation check | Not started |
| DeepLabV2 baseline | Architecture generality | Not started |
| DAFormer + SATG combination | Best-case scenario | Planned (T048e–T048h) |
| 5 seeds per experiment | Statistical significance | Planned (T048b updated) |
| Hypothesis validation dry run | Quick validation before full training | Planned (T000c) |
