"""Tests for trust gate module (RED phase — all tests should fail initially)."""

import pytest
import torch
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> OmegaConf:
    """Build a minimal trust_gate config with sensible defaults."""
    base = {
        "trust_gate": {
            "type": "hard",
            "tau_conf": 0.90,
            "tau_struct": 0.60,
            "soft_weight_temp_conf": 10.0,
            "soft_weight_temp_struct": 10.0,
            "soft_weight_bias": 0.0,
            "soft_label_k": 4.0,
            "soft_label_t_max": 5.0,
        }
    }
    cfg = OmegaConf.create(base)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


# ===========================================================================
# HardTrustGate tests
# ===========================================================================


class TestHardTrustGate:
    """HardTrustGate — binary mask via strict thresholds."""

    def test_trusted(self):
        """conf=0.95 (>0.90), struct=0.30 (<0.60) → mask=1.0"""
        from satg.trust_gate import HardTrustGate

        cfg = _make_cfg()
        gate = HardTrustGate(cfg)
        conf = torch.tensor([[[0.95]]])
        struct = torch.tensor([[[0.30]]])
        mask = gate.compute_mask(conf, struct)
        assert mask.shape == (1, 1, 1)
        assert mask.item() == 1.0

    def test_low_conf(self):
        """conf=0.85 (< tau_conf=0.90) → mask=0.0 even if struct is fine"""
        from satg.trust_gate import HardTrustGate

        cfg = _make_cfg()
        gate = HardTrustGate(cfg)
        conf = torch.tensor([[[0.85]]])
        struct = torch.tensor([[[0.30]]])
        mask = gate.compute_mask(conf, struct)
        assert mask.item() == 0.0

    def test_high_struct(self):
        """struct=0.80 (> tau_struct=0.60) → mask=0.0 even if conf is fine"""
        from satg.trust_gate import HardTrustGate

        cfg = _make_cfg()
        gate = HardTrustGate(cfg)
        conf = torch.tensor([[[0.95]]])
        struct = torch.tensor([[[0.80]]])
        mask = gate.compute_mask(conf, struct)
        assert mask.item() == 0.0

    def test_boundary_excluded(self):
        """conf == tau_conf exactly → 0.0 (strictly greater, not ≥)"""
        from satg.trust_gate import HardTrustGate

        cfg = _make_cfg()
        gate = HardTrustGate(cfg)
        conf = torch.tensor([[[0.90]]])  # exactly tau_conf
        struct = torch.tensor([[[0.30]]])
        mask = gate.compute_mask(conf, struct)
        assert mask.item() == 0.0

    def test_zero_mask_no_nan(self):
        """All-zero output is valid — no NaN anywhere."""
        from satg.trust_gate import HardTrustGate

        cfg = _make_cfg()
        gate = HardTrustGate(cfg)
        conf = torch.zeros(2, 4, 4)
        struct = torch.ones(2, 4, 4)  # struct > tau_struct → all rejected
        mask = gate.compute_mask(conf, struct)
        assert mask.shape == (2, 4, 4)
        assert mask.sum() == 0.0
        assert not torch.isnan(mask).any()

    def test_output_shape(self):
        """Input [B, H, W] for both → output [B, H, W]."""
        from satg.trust_gate import HardTrustGate

        cfg = _make_cfg()
        gate = HardTrustGate(cfg)
        B, H, W = 3, 16, 16
        conf = torch.rand(B, H, W)
        struct = torch.rand(B, H, W)
        mask = gate.compute_mask(conf, struct)
        assert mask.shape == (B, H, W)


# ===========================================================================
# SoftWeightTrustGate tests
# ===========================================================================


class TestSoftWeightTrustGate:
    """SoftWeightTrustGate — continuous weights via sigmoid gating."""

    def test_range(self):
        """All weights in [0.0, 1.0]."""
        from satg.trust_gate import SoftWeightTrustGate

        cfg = _make_cfg()
        gate = SoftWeightTrustGate(cfg)
        conf = torch.rand(2, 8, 8)
        struct = torch.rand(2, 8, 8)
        w = gate.compute_weights(conf, struct)
        assert w.shape == (2, 8, 8)
        assert (w >= 0.0).all()
        assert (w <= 1.0).all()

    def test_monotone_confidence(self):
        """Higher conf → non-decreasing weight (fix struct, vary conf)."""
        from satg.trust_gate import SoftWeightTrustGate

        cfg = _make_cfg()
        gate = SoftWeightTrustGate(cfg)
        struct = torch.full((1, 1, 1), 0.3)
        confs = torch.linspace(0.0, 1.0, 20).view(20, 1, 1)
        structs = struct.expand(20, 1, 1)
        w = gate.compute_weights(confs, structs)
        diffs = w[1:] - w[:-1]
        assert (diffs >= -1e-6).all(), "weights should be non-decreasing with confidence"

    def test_monotone_structure(self):
        """Higher struct → non-increasing weight (fix conf, vary struct)."""
        from satg.trust_gate import SoftWeightTrustGate

        cfg = _make_cfg()
        gate = SoftWeightTrustGate(cfg)
        conf = torch.full((1, 1, 1), 0.95)
        structs = torch.linspace(0.0, 1.0, 20).view(20, 1, 1)
        confs = conf.expand(20, 1, 1)
        w = gate.compute_weights(confs, structs)
        diffs = w[1:] - w[:-1]
        assert (diffs <= 1e-6).all(), "weights should be non-increasing with structure"

    def test_output_shape(self):
        """[B, H, W] → [B, H, W]."""
        from satg.trust_gate import SoftWeightTrustGate

        cfg = _make_cfg()
        gate = SoftWeightTrustGate(cfg)
        B, H, W = 2, 12, 12
        conf = torch.rand(B, H, W)
        struct = torch.rand(B, H, W)
        w = gate.compute_weights(conf, struct)
        assert w.shape == (B, H, W)

    def test_reads_temp_conf_key(self):
        """Uses cfg.trust_gate.soft_weight_temp_conf (NOT soft_temp_conf)."""
        from satg.trust_gate import SoftWeightTrustGate

        cfg = _make_cfg()
        # Verify the correct key exists
        assert "soft_weight_temp_conf" in cfg.trust_gate, \
            "Config must have 'soft_weight_temp_conf' key"
        gate = SoftWeightTrustGate(cfg)
        conf = torch.tensor([[[0.95]]])
        struct = torch.tensor([[[0.30]]])
        w = gate.compute_weights(conf, struct)
        assert w.shape == (1, 1, 1)
        assert 0.0 <= w.item() <= 1.0

    def test_reads_bias_key(self):
        """Uses cfg.trust_gate.soft_weight_bias (NOT soft_bias)."""
        from satg.trust_gate import SoftWeightTrustGate

        cfg = _make_cfg()
        # Verify the correct key exists
        assert "soft_weight_bias" in cfg.trust_gate, \
            "Config must have 'soft_weight_bias' key"
        gate = SoftWeightTrustGate(cfg)
        conf = torch.tensor([[[0.95]]])
        struct = torch.tensor([[[0.30]]])
        w = gate.compute_weights(conf, struct)
        assert w.shape == (1, 1, 1)