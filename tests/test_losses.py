"""Tests for SATGLoss (Phase 0/1)."""

import pytest
import torch
from torch.nn import functional as F


@pytest.fixture(scope="module")
def B():
    return 2


@pytest.fixture(scope="module")
def C():
    return 19


@pytest.fixture(scope="module")
def H():
    return 64


@pytest.fixture(scope="module")
def W():
    return 64


# ---------------------------------------------------------------------------
# SATGLoss tests
# ---------------------------------------------------------------------------


class TestSATGLoss:
    """Test suite for SATGLoss."""

    @pytest.fixture(autouse=True)
    def _loss(self):
        from satg.losses import SATGLoss

        self.loss_fn = SATGLoss()

    def _make_tensors(self, B, C, H, W, label_fill=0, trust_fill=1.0):
        """Helper: create student_logits, pseudo_labels, trust_weights.

        By default pseudo_labels are all 0 and trust_weights are all 1.0.
        """
        student_logits = torch.randn(B, C, H, W)
        pseudo_labels = torch.full((B, H, W), fill_value=label_fill, dtype=torch.long)
        trust_weights = torch.full((B, H, W), fill_value=trust_fill, dtype=torch.float32)
        return student_logits, pseudo_labels, trust_weights

    # ------------------------------------------------------------------
    # Test 1: zero trust_weights → loss = 0.0, no NaN
    # ------------------------------------------------------------------

    def test_zero_mask_gives_zero(self, B, C, H, W):
        student_logits, pseudo_labels, _ = self._make_tensors(B, C, H, W)
        trust_weights = torch.zeros(B, H, W)

        loss = self.loss_fn(student_logits, pseudo_labels, trust_weights)

        assert loss.item() == 0.0, f"Expected 0.0, got {loss.item()}"
        assert not torch.isnan(loss), "Loss should not be NaN"

    # ------------------------------------------------------------------
    # Test 2: all-ones trust_weights → loss ≈ standard CE
    # ------------------------------------------------------------------

    def test_ones_mask_equals_ce(self, B, C, H, W):
        student_logits, pseudo_labels, trust_weights = self._make_tensors(B, C, H, W)

        loss_satg = self.loss_fn(student_logits, pseudo_labels, trust_weights)

        # Standard CE (mean reduction)
        loss_ce = F.cross_entropy(student_logits, pseudo_labels, ignore_index=255)

        assert loss_satg.shape == loss_ce.shape, "Shape mismatch"
        assert torch.isclose(loss_satg, loss_ce, atol=1e-6).item(), (
            f"SATGLoss {loss_satg.item()} != CE {loss_ce.item()}"
        )

    # ------------------------------------------------------------------
    # Test 3: 50% masked (zero trust) → 0 < loss < full CE
    # ------------------------------------------------------------------

    def test_partial_mask(self, B, C, H, W):
        # Use controlled logits so the masked-out region has high loss
        # (confidently wrong predictions) while the kept region has low loss.
        # Left half: confidently predict wrong class → high loss
        # Right half: confidently predict correct class → low loss
        student_logits = torch.randn(B, C, H, W) * 1.0
        pseudo_labels = torch.zeros(B, H, W, dtype=torch.long)
        # Make left-half predictions confidently WRONG (class 18, not 0)
        student_logits[:, :, :, : W // 2] = -10.0  # suppress correct class
        student_logits[:, 18, :, : W // 2] = 10.0  # boost wrong class

        # Full CE baseline (all weights = 1.0)
        trust_ones = torch.ones(B, H, W)
        loss_full = self.loss_fn(student_logits, pseudo_labels, trust_ones)

        # Mask out the high-loss left half — loss should drop
        trust_half = torch.ones(B, H, W)
        trust_half[:, :, : W // 2] = 0.0
        loss_half = self.loss_fn(student_logits, pseudo_labels, trust_half)

        assert loss_half.item() > 0.0, "Partial mask loss should be > 0"
        assert loss_half.item() < loss_full.item(), (
            f"Partial loss {loss_half.item()} should be < full loss {loss_full.item()}"
        )

    # ------------------------------------------------------------------
    # Test 4: ignore_index=255 — label=255 pixels excluded
    # ------------------------------------------------------------------

    def test_ignore_255_excluded(self, B, C, H, W):
        student_logits, _, trust_weights = self._make_tensors(B, C, H, W)

        # All labels = 255 (should all be ignored → loss = 0)
        pseudo_labels = torch.full((B, H, W), fill_value=255, dtype=torch.long)

        loss = self.loss_fn(student_logits, pseudo_labels, trust_weights)

        assert loss.item() == 0.0, (
            f"Expected 0.0 when all labels are 255, got {loss.item()}"
        )

    # ------------------------------------------------------------------
    # Test 5: output is scalar (0-dim tensor)
    # ------------------------------------------------------------------

    def test_scalar_output(self, B, C, H, W):
        student_logits, pseudo_labels, trust_weights = self._make_tensors(B, C, H, W)

        loss = self.loss_fn(student_logits, pseudo_labels, trust_weights)

        assert loss.ndim == 0, f"Expected scalar (0-dim), got shape {loss.shape}"

    # ------------------------------------------------------------------
    # Test 6: safe division — very small sum of trust_weights → no NaN
    # ------------------------------------------------------------------

    def test_safe_division(self, B, C, H, W):
        student_logits, pseudo_labels, _ = self._make_tensors(B, C, H, W)

        # Tiny but non-zero trust weights
        trust_weights = torch.full((B, H, W), fill_value=1e-10, dtype=torch.float32)

        loss = self.loss_fn(student_logits, pseudo_labels, trust_weights)

        assert not torch.isnan(loss), "Loss should not be NaN with tiny trust weights"
        assert not torch.isinf(loss), "Loss should not be Inf with tiny trust weights"