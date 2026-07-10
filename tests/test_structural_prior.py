"""Tests for StructuralPrior (RED phase — all must fail until module is implemented)."""

import numpy as np
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def cfg():
    """Minimal config with default structural_prior params (including new keys)."""
    return OmegaConf.create(
        {
            "structural_prior": {
                "edge_low_threshold": 50,
                "edge_high_threshold": 150,
                "gaussian_sigma": 2.0,
                "edge_kernel_size": 15,
                "variance_kernel_size": 15,
                "edge_weight": 0.25,
                "variance_weight": 0.25,
                "entropy_kernel_radius": 7,
                "entropy_n_bins": 32,
                "entropy_weight": 0.25,
                "cornerness_kernel_size": 15,
                "cornerness_sigma": 2.0,
                "cornerness_weight": 0.25,
            }
        }
    )


@pytest.fixture
def gray_256():
    """Solid medium-gray 256×256 image — negligible edges, near-zero variance."""
    return np.full((256, 256, 3), 128, dtype=np.uint8)


@pytest.fixture
def high_texture_256():
    """High-texture 256×256 image — random binary noise, strong edges + high variance."""
    rng = np.random.RandomState(42)
    noise = (rng.rand(256, 256) > 0.5).astype(np.uint8) * 255
    return np.stack([noise] * 3, axis=-1)


# ---------------------------------------------------------------------------
# Existing tests (updated for dict return type)
# ---------------------------------------------------------------------------


def test_solid_gray_low_score(cfg, gray_256):
    """A uniform gray image should have a very low structural complexity score."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    heatmap = prior.compute_combined(gray_256)
    assert heatmap.mean() < 0.15, (
        f"Expected mean < 0.15 for uniform gray, got {heatmap.mean():.4f}"
    )


def test_high_texture_high_score(cfg, high_texture_256):
    """A high-texture image (random binary noise) should have a high structural complexity score."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    heatmap = prior.compute_combined(high_texture_256)
    assert heatmap.mean() > 0.40, (
        f"Expected mean > 0.40 for high-texture image, got {heatmap.mean():.4f}"
    )


def test_output_shape_matches_input(cfg, gray_256):
    """Input [H, W, 3] → output [H, W]."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    heatmap = prior.compute_combined(gray_256)
    assert heatmap.shape == (256, 256), (
        f"Expected (256, 256), got {heatmap.shape}"
    )


def test_output_range(cfg, high_texture_256):
    """All output values must lie in [0.0, 1.0]."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    result = prior.compute(high_texture_256)
    for key in ['edge', 'var', 'ent', 'corn', 'combined']:
        hm = result[key]
        assert hm.min() >= 0.0, f"Key '{key}' min value {hm.min()} < 0.0"
        assert hm.max() <= 1.0, f"Key '{key}' max value {hm.max()} > 1.0"


def test_float32_dtype(cfg, gray_256):
    """Output must be np.float32."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    result = prior.compute(gray_256)
    for key in ['edge', 'var', 'ent', 'corn', 'combined']:
        assert result[key].dtype == np.float32, (
            f"Key '{key}' expected float32, got {result[key].dtype}"
        )


def test_batch_compute_length(cfg, gray_256):
    """batch_compute returns one result per input image."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    images = [gray_256, gray_256, gray_256]
    results = prior.batch_compute(images)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    for r in results:
        assert isinstance(r, dict)
        for key in ['edge', 'var', 'ent', 'corn', 'combined']:
            assert r[key].shape == (256, 256)
            assert r[key].dtype == np.float32


# ---------------------------------------------------------------------------
# New tests for dict return type and component correctness
# ---------------------------------------------------------------------------


def test_compute_returns_dict(cfg, gray_256):
    """compute() must return a dict with exactly the 5 expected keys."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    result = prior.compute(gray_256)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'edge', 'var', 'ent', 'corn', 'combined'}


def test_all_components_in_range(cfg, high_texture_256):
    """Each component in {edge, var, ent, corn, combined} must be in [0, 1]."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    result = prior.compute(high_texture_256)
    for key in ['edge', 'var', 'ent', 'corn', 'combined']:
        arr = result[key]
        assert arr.min() >= 0.0, f"Key '{key}' min {arr.min()} < 0"
        assert arr.max() <= 1.0, f"Key '{key}' max {arr.max()} > 1"


def test_entropy_flat_image_low(cfg):
    """A completely uniform image has zero entropy."""
    from satg.structural_prior import StructuralPrior

    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    prior = StructuralPrior(cfg)
    result = prior.compute(flat)
    assert result['ent'].mean() < 0.1, (
        f"Expected entropy < 0.1 for flat image, got {result['ent'].mean():.4f}"
    )


def test_entropy_noisy_image_high(cfg):
    """A random noise image has maximum entropy."""
    from satg.structural_prior import StructuralPrior

    noisy = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    prior = StructuralPrior(cfg)
    result = prior.compute(noisy)
    assert result['ent'].mean() > 0.5, (
        f"Expected entropy > 0.5 for noisy image, got {result['ent'].mean():.4f}"
    )


def test_cornerness_flat_region_low(cfg):
    """A completely uniform image has no gradients → λ₂ ≈ 0."""
    from satg.structural_prior import StructuralPrior

    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    prior = StructuralPrior(cfg)
    result = prior.compute(flat)
    assert result['corn'].mean() < 0.05, (
        f"Expected cornerness < 0.05 for flat image, got {result['corn'].mean():.4f}"
    )


def test_cornerness_corner_high(cfg):
    """An image with a corner (gradients in two directions) has high λ₂."""
    from satg.structural_prior import StructuralPrior

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:32, :32] = 255  # top-left quadrant white → creates corner at center
    prior = StructuralPrior(cfg)
    result = prior.compute(img)
    # The center region (corner junction) should have high cornerness
    center_val = result['corn'][28:36, 28:36].mean()
    assert center_val > 0.25, (
        f"Expected corner cornerness > 0.25, got {center_val:.4f}"
    )


def test_combined_is_weighted_sum(cfg, high_texture_256):
    """With equal weights (0.25 each), combined = 0.25 * (e + v + ent + c)."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    result = prior.compute(high_texture_256)
    expected = 0.25 * (result['edge'] + result['var'] + result['ent'] + result['corn'])
    expected = np.clip(expected, 0, 1)
    np.testing.assert_allclose(result['combined'], expected, atol=1e-5)


def test_components_not_identical(cfg, high_texture_256):
    """Edge density and cornerness should not be perfectly correlated."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    result = prior.compute(high_texture_256)
    corr = np.corrcoef(result['edge'].flatten(), result['corn'].flatten())[0, 1]
    assert abs(corr) < 0.98, (
        f"Edge and cornerness are too correlated: {corr:.3f}"
    )


def test_compute_combined_backward_compat(cfg, gray_256):
    """compute_combined() returns only the combined heatmap [H, W] float32."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    hm = prior.compute_combined(gray_256)
    assert isinstance(hm, np.ndarray)
    assert hm.shape == (256, 256)
    assert hm.dtype == np.float32
    assert hm.min() >= 0.0
    assert hm.max() <= 1.0


def test_batch_compute_returns_list_of_dicts(cfg, gray_256):
    """batch_compute returns list of dicts with correct keys."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    results = prior.batch_compute([gray_256, gray_256])
    assert len(results) == 2
    for r in results:
        assert isinstance(r, dict)
        assert set(r.keys()) == {'edge', 'var', 'ent', 'corn', 'combined'}
