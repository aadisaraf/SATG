"""Tests for TemperatureSoftLabel and SoftLabelKLLoss (RED phase)."""

import pytest
import torch
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_cfg():
    """Minimal config with trust_gate defaults."""
    return OmegaConf.create({
        "trust_gate": {
            "tau_conf": 0.90,
            "soft_label_k": 4.0,
            "soft_label_t_max": 5.0,
        },
    })


@pytest.fixture
def B():
    return 2


@pytest.fixture
def C():
    return 19


@pytest.fixture
def H():
    return 4


@pytest.fixture
def W():
    return 4


@pytest.fixture
def logits(B, C, H, W):
    """Synthetic teacher logits — structured so max class varies per pixel."""
    t = torch.randn(B, C, H, W)
    # ensure some spread
    return t * 2.0


@pytest.fixture
def soft_label(default_cfg):
    from satg.soft_label import TemperatureSoftLabel
    return TemperatureSoftLabel(default_cfg)


@pytest.fixture
def kl_loss():
    from satg.losses import SoftLabelKLLoss
    return SoftLabelKLLoss()


# ===========================================================================
# TemperatureSoftLabel tests
# ===========================================================================


class TestTemperatureSoftLabel:

    def test_temp_zero_struct(self, soft_label, B, H, W):
        """struct=0 -> T=1.0 exactly."""
        struct = torch.zeros(B, H, W)
        T = soft_label.compute_temperature(struct)
        assert torch.allclose(T, torch.ones_like(T)), f"T = {T}"

    def test_temp_proportional(self, soft_label, B, H, W):
        """struct=0.5, k=4.0 -> T=3.0."""
        struct = torch.full((B, H, W), 0.5)
        T = soft_label.compute_temperature(struct)
        assert torch.allclose(T, torch.full_like(T, 3.0)), f"T = {T}"

    def test_temp_capped(self, default_cfg, B, H, W):
        """struct=1.0, k=10.0, T_max=5.0 -> T=5.0 (capped, not 11.0)."""
        cfg = OmegaConf.merge(
            default_cfg,
            OmegaConf.create({"trust_gate": {"soft_label_k": 10.0, "soft_label_t_max": 5.0}}),
        )
        from satg.soft_label import TemperatureSoftLabel
        sl = TemperatureSoftLabel(cfg)
        struct = torch.ones(B, H, W)
        T = sl.compute_temperature(struct)
        assert torch.allclose(T, torch.full_like(T, 5.0)), f"T = {T}"

    def test_soft_target_sums_to_one(self, soft_label, logits, B, C, H, W):
        """output.sum(dim=1) ~= 1.0 everywhere (all pixels sum to 1.0 across classes)."""
        struct = torch.rand(B, H, W)
        soft_targets = soft_label.compute_soft_targets(logits, struct)
        sums = soft_targets.sum(dim=1)  # (B, H, W)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            f"sums range [{sums.min().item():.6f}, {sums.max().item():.6f}]"

    def test_entropy_increases_with_struct(self, soft_label, logits, B, C, H, W):
        """Same logits, higher struct -> flatter distribution (entropy strictly increases)."""
        struct_low = torch.full((B, H, W), 0.1)
        struct_high = torch.full((B, H, W), 0.9)

        t_low = soft_label.compute_soft_targets(logits, struct_low)
        t_high = soft_label.compute_soft_targets(logits, struct_high)

        def entropy(p):
            # Shannon entropy per pixel, averaged
            return -(p * torch.log(p + 1e-8)).sum(dim=1).mean()

        ent_low = entropy(t_low)
        ent_high = entropy(t_high)
        assert ent_high > ent_low, \
            f"entropy low={ent_low:.6f} high={ent_high:.6f}"

    def test_approaches_uniform_at_max_temp(self, default_cfg, logits, B, C, H, W):
        """T=T_max -> distribution near 1/C per class."""
        cfg = OmegaConf.merge(
            default_cfg,
            OmegaConf.create({"trust_gate": {"soft_label_k": 1e6, "soft_label_t_max": 1e6}}),
        )
        from satg.soft_label import TemperatureSoftLabel
        sl = TemperatureSoftLabel(cfg)
        struct = torch.ones(B, H, W)
        soft_targets = sl.compute_soft_targets(logits, struct)
        # At very high temperature softmax approaches uniform
        expected = 1.0 / C
        mean_val = soft_targets.mean(dim=1)  # (B, H, W)
        assert torch.allclose(mean_val, torch.full_like(mean_val, expected), atol=1e-2), \
            f"mean = {mean_val.mean().item():.6f}, expected {expected:.6f}"

    def test_confidence_mask_uses_tau_conf(self, soft_label, default_cfg, logits, B, H, W):
        """mask = max_softmax > tau_conf (strictly greater, not >=)."""
        tau_conf = default_cfg.trust_gate.tau_conf  # 0.90
        mask = soft_label.compute_confidence_mask(logits, tau_conf)

        # Verify shape
        assert mask.shape == (B, H, W), f"mask shape {mask.shape}"
        assert mask.dtype == torch.bool, f"mask dtype {mask.dtype}"

        # Manual check: for each pixel, verify the rule
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
        max_probs = probs.max(dim=1).values  # (B, H, W)
        expected_mask = max_probs > tau_conf
        assert torch.equal(mask, expected_mask), \
            "confidence mask does not match max > tau_conf"


# ===========================================================================
# SoftLabelKLLoss tests
# ===========================================================================


class TestSoftLabelKLLoss:

    def test_zero_mask_gives_zero(self, kl_loss, B, C, H, W):
        """conf_mask=zeros -> loss=0.0, no NaN."""
        student_logits = torch.randn(B, C, H, W)
        soft_targets = torch.softmax(torch.randn(B, C, H, W), dim=1)
        confidence_mask = torch.zeros(B, H, W, dtype=torch.bool)
        loss = kl_loss(student_logits, soft_targets, confidence_mask)
        assert loss.numel() == 1, "loss is not scalar"
        assert loss.item() == 0.0, f"loss = {loss.item()}"
        assert not torch.isnan(loss), "loss is NaN"

    def test_identical_distributions_near_zero_kl(self, kl_loss, B, C, H, W):
        """student~=soft_targets -> KL~=0."""
        logits = torch.randn(B, C, H, W)
        soft_targets = torch.softmax(logits, dim=1)
        student_logits = logits.clone()  # identical
        confidence_mask = torch.ones(B, H, W, dtype=torch.bool)
        loss = kl_loss(student_logits, soft_targets, confidence_mask)
        assert loss.item() < 1e-4, f"KL divergence = {loss.item():.6f} (expected ~0)"

    def test_loss_increases_with_mismatch(self, kl_loss, B, C, H, W):
        """Divergent distributions -> loss > 0."""
        # Student and teacher completely different
        student_logits = torch.randn(B, C, H, W) * 10.0
        soft_targets = torch.softmax(torch.randn(B, C, H, W) * 10.0, dim=1)
        confidence_mask = torch.ones(B, H, W, dtype=torch.bool)
        loss = kl_loss(student_logits, soft_targets, confidence_mask)
        assert loss.item() > 0.01, f"loss = {loss.item():.6f}"

    def test_nonnegative(self, kl_loss, B, C, H, W):
        """KL divergence is always >= 0."""
        for _ in range(5):
            student_logits = torch.randn(B, C, H, W)
            soft_targets = torch.softmax(torch.randn(B, C, H, W), dim=1)
            confidence_mask = torch.ones(B, H, W, dtype=torch.bool)
            loss = kl_loss(student_logits, soft_targets, confidence_mask)
            assert loss.item() >= -1e-6, f"Negative KL = {loss.item():.6f}"

    def test_scalar_output(self, kl_loss, B, C, H, W):
        """Output is a 0-dim tensor."""
        student_logits = torch.randn(B, C, H, W)
        soft_targets = torch.softmax(torch.randn(B, C, H, W), dim=1)
        confidence_mask = torch.ones(B, H, W, dtype=torch.bool)
        loss = kl_loss(student_logits, soft_targets, confidence_mask)
        assert loss.dim() == 0, f"loss.dim() = {loss.dim()}"

    def test_eps_prevents_log_zero(self, kl_loss, B, C, H, W):
        """No NaN when soft_targets has near-zero values (eps=1e-8 in log)."""
        student_logits = torch.randn(B, C, H, W)
        # Make one class nearly zero probability
        soft_targets = torch.full((B, C, H, W), 1e-10)
        soft_targets[:, 0, :, :] = 1.0 - (C - 1) * 1e-10  # renormalise
        confidence_mask = torch.ones(B, H, W, dtype=torch.bool)
        loss = kl_loss(student_logits, soft_targets, confidence_mask)
        assert not torch.isnan(loss), f"loss is NaN: {loss}"
        assert not torch.isinf(loss), f"loss is Inf: {loss}"
