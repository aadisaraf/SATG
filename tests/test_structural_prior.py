"""Tests for StructuralPrior (RED phase — all must fail until module is implemented)."""

import numpy as np
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def cfg():
    """Minimal config with default structural_prior params."""
    return OmegaConf.create(
        {
            "structural_prior": {
                "edge_low_threshold": 50,
                "edge_high_threshold": 150,
                "gaussian_sigma": 2.0,
                "edge_kernel_size": 15,
                "variance_kernel_size": 15,
                "edge_weight": 0.5,
                "variance_weight": 0.5,
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
# Tests
# ---------------------------------------------------------------------------


def test_solid_gray_low_score(cfg, gray_256):
    """A uniform gray image should have a very low structural complexity score."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    heatmap = prior.compute(gray_256)
    assert heatmap.mean() < 0.15, (
        f"Expected mean < 0.15 for uniform gray, got {heatmap.mean():.4f}"
    )


def test_high_texture_high_score(cfg, high_texture_256):
    """A high-texture image (random binary noise) should have a high structural complexity score."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    heatmap = prior.compute(high_texture_256)
    assert heatmap.mean() > 0.45, (
        f"Expected mean > 0.45 for high-texture image, got {heatmap.mean():.4f}"
    )


def test_output_shape_matches_input(cfg, gray_256):
    """Input [H, W, 3] → output [H, W]."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    heatmap = prior.compute(gray_256)
    assert heatmap.shape == (256, 256), (
        f"Expected (256, 256), got {heatmap.shape}"
    )


def test_output_range(cfg, high_texture_256):
    """All output values must lie in [0.0, 1.0]."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    heatmap = prior.compute(high_texture_256)
    assert heatmap.min() >= 0.0, f"Min value {heatmap.min()} < 0.0"
    assert heatmap.max() <= 1.0, f"Max value {heatmap.max()} > 1.0"


def test_float32_dtype(cfg, gray_256):
    """Output must be np.float32."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    heatmap = prior.compute(gray_256)
    assert heatmap.dtype == np.float32, (
        f"Expected float32, got {heatmap.dtype}"
    )


def test_batch_compute_length(cfg, gray_256):
    """batch_compute returns one result per input image."""
    from satg.structural_prior import StructuralPrior

    prior = StructuralPrior(cfg)
    images = [gray_256, gray_256, gray_256]
    results = prior.batch_compute(images)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    for r in results:
        assert r.shape == (256, 256)
        assert r.dtype == np.float32
