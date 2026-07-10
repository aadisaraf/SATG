"""Shared test fixtures for SATG (Phase 0/1)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "configs"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return _CONFIG_DIR


@pytest.fixture(scope="session")
def default_cfg():
    """Load the master default config."""
    return OmegaConf.load(str(_CONFIG_DIR / "default.yaml"))


# ---------------------------------------------------------------------------
# Small synthetic fixtures (function-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_rgb():
    """Return a tiny RGB image (64, 64, 3), uint8, values in [0, 255]."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def tiny_heatmap():
    """Return a tiny heatmap (64, 64), float32, values in [0, 1]."""
    return np.random.rand(64, 64).astype(np.float32)


@pytest.fixture
def tiny_cfg():
    """Load default config then override crop_size for small tests."""
    cfg = OmegaConf.load(str(_CONFIG_DIR / "default.yaml"))
    return OmegaConf.merge(cfg, OmegaConf.create({"training": {"crop_size": [64, 64]}}))


@pytest.fixture
def fake_logits():
    """Return fake teacher logits (B=2, C=19, H=64, W=64)."""
    return torch.randn(2, 19, 64, 64)


@pytest.fixture
def fake_conf():
    """Return fake confidence map (B=2, H=64, W=64), values in (0,1)."""
    return torch.rand(2, 64, 64)


@pytest.fixture
def fake_struct():
    """Return fake structural heatmap (B=2, H=64, W=64), values in (0,1)."""
    return torch.rand(2, 64, 64)


@pytest.fixture
def fake_heatmap_np():
    """Return a fake heatmap as numpy array (64, 64), float32, [0, 1]."""
    return np.random.rand(64, 64).astype(np.float32)
