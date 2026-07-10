"""Tests for EMAModel (RED phase — all must fail until module is implemented)."""

import copy

import torch
import torch.nn as nn
import pytest


# ---------------------------------------------------------------------------
# Helper: minimal segmentation model that mimics DeepLabV3+ output dict
# ---------------------------------------------------------------------------


class _TestSegModel(nn.Module):
    """Minimal model that produces {'out': [B,19,H/8,W/8], 'aux': [B,19,H/8,W/8]}."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Conv2d(3, 64, 3, padding=1)
        self.classifier = nn.Conv2d(64, 19, 1)
        self.aux_classifier = nn.Conv2d(64, 19, 1)

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)  # [B, 64, H, W]
        H, W = x.shape[2], x.shape[3]
        pooled = nn.functional.adaptive_avg_pool2d(features, (H // 8, W // 8))
        return {"out": self.classifier(pooled), "aux": self.aux_classifier(pooled)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def student_model():
    return _TestSegModel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_initial_weights_match(student_model):
    """After init, teacher params must equal student params."""
    from models.ema import EMAModel

    ema = EMAModel(model=student_model, momentum=0.999)

    for (n1, p1), (n2, p2) in zip(
        student_model.named_parameters(), ema.model.named_parameters()
    ):
        assert torch.equal(p1.data, p2.data), (
            f"Parameter {n1} differs after EMAModel init"
        )


def test_ema_update_momentum_half(student_model):
    """1 update with momentum=0.5 → teacher = 0.5*initial + 0.5*new_student."""
    from models.ema import EMAModel

    model_copy = copy.deepcopy(student_model)
    ema = EMAModel(model=model_copy, momentum=0.5)

    # Snapshot initial teacher weights
    init_teacher = {n: p.data.clone() for n, p in ema.model.named_parameters()}

    # Perturb student
    for p in student_model.parameters():
        p.data.add_(torch.randn_like(p.data) * 0.1)

    # One EMA update
    ema.update(student_model)

    # Verify: tp_new = 0.5 * tp_old + 0.5 * sp_new
    for (n, tp), (_, sp) in zip(
        ema.model.named_parameters(), student_model.named_parameters()
    ):
        expected = 0.5 * init_teacher[n] + 0.5 * sp.data
        assert torch.allclose(tp.data, expected, atol=1e-6), (
            f"Parameter {n} does not match momentum-weighted average"
        )


def test_teacher_always_eval(student_model):
    """Teacher must remain in eval mode after init and after update."""
    from models.ema import EMAModel

    ema = EMAModel(model=student_model, momentum=0.999)

    assert not ema.model.training, "Teacher should be in eval mode after init"

    ema.update(student_model)
    assert not ema.model.training, "Teacher should remain in eval mode after update"


def test_teacher_no_grad(student_model):
    """All teacher params must have requires_grad=False."""
    from models.ema import EMAModel

    ema = EMAModel(model=student_model, momentum=0.999)

    for p in ema.model.parameters():
        assert not p.requires_grad, (
            "Teacher parameter should have requires_grad=False"
        )


def test_model_attribute_exists(student_model):
    """EMAModel must expose .model attribute directly."""
    from models.ema import EMAModel

    ema = EMAModel(model=student_model, momentum=0.999)

    assert hasattr(ema, "model"), "EMAModel should expose .model attribute"
    assert ema.model is not None


def test_forward_shape(student_model):
    """ema.model(torch.randn(2,3,512,512)) → main:[2,19,64,64], aux:[2,19,64,64]."""
    from models.ema import EMAModel

    ema = EMAModel(model=student_model, momentum=0.999)

    x = torch.randn(2, 3, 512, 512)
    out = ema.model(x)

    assert isinstance(out, dict), "Output should be a dict"
    assert "out" in out and "aux" in out, "Output dict should have 'out' and 'aux'"
    assert out["out"].shape == (2, 19, 64, 64), (
        f"Expected main out shape (2,19,64,64), got {out['out'].shape}"
    )
    assert out["aux"].shape == (2, 19, 64, 64), (
        f"Expected aux out shape (2,19,64,64), got {out['aux'].shape}"
    )


def test_state_dict_keys(student_model):
    """ema.state_dict() must contain 'shadow_params' and 'iteration'."""
    from models.ema import EMAModel

    ema = EMAModel(model=student_model, momentum=0.999)

    sd = ema.state_dict()
    assert "shadow_params" in sd, "state_dict should contain 'shadow_params'"
    assert "iteration" in sd, "state_dict should contain 'iteration'"
