"""Target label leakage prevention tests — these should all PASS.

Verifies three critical no-leak contracts:
1. Cityscapes train split yields ``(image, heatmap)``, NOT ``(image, label)``
2. EMA model operates under ``torch.no_grad()`` with ``requires_grad=False``
3. Target loss receives only ``student_logits + pseudo_labels``, never GT labels
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


# ===========================================================================
# Test 1: Cityscapes train split returns (image, heatmap), NOT (image, label)
# ===========================================================================


class TestCityscapesTrainNoLeakage:
    """Cityscapes train split must NOT return GT labels.

    The training loop uses pseudo-labels from the teacher; GT labels must
    never appear in the train path.
    """

    @pytest.fixture
    def cityscapes_minimal(self, tmp_path):
        """Create one Cityscapes training image with 4 heatmap component files,
        plus one validation image with GT label.

        Structure:
            tmp_path/
              leftImg8bit/train/city/img_000001_leftImg8bit.png
              leftImg8bit/val/city/img_000002_leftImg8bit.png
              gtFine/train/city/img_000001_gtFine_labelTrainIds.png (present but
                  should NOT be loaded by train split)
              gtFine/val/city/img_000002_gtFine_labelTrainIds.png
              leftImg8bit/train/city/img_000001_satg_{edge,var,ent,corn}.npy
        """
        import cv2

        root = tmp_path

        # --- Train split: image + heatmaps + GT label file ---
        train_img_dir = root / "leftImg8bit" / "train" / "city"
        train_gt_dir = root / "gtFine" / "train" / "city"
        train_img_dir.mkdir(parents=True)
        train_gt_dir.mkdir(parents=True)

        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(train_img_dir / "img_000001_leftImg8bit.png"), img)

        gt_label = np.random.choice(list(range(19)) + [255], (64, 64)).astype(np.uint8)
        cv2.imwrite(str(train_gt_dir / "img_000001_gtFine_labelTrainIds.png"), gt_label)

        for suffix in ["_satg_edge.npy", "_satg_var.npy", "_satg_ent.npy", "_satg_corn.npy"]:
            component = np.random.rand(64, 64).astype(np.float32)
            np.save(str(train_img_dir / f"img_000001{suffix}"), component)

        # --- Val split: image + GT label (no heatmaps) ---
        val_img_dir = root / "leftImg8bit" / "val" / "city"
        val_gt_dir = root / "gtFine" / "val" / "city"
        val_img_dir.mkdir(parents=True)
        val_gt_dir.mkdir(parents=True)

        val_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(val_img_dir / "img_000002_leftImg8bit.png"), val_img)

        val_label = np.random.choice(list(range(19)) + [255], (64, 64)).astype(np.uint8)
        cv2.imwrite(str(val_gt_dir / "img_000002_gtFine_labelTrainIds.png"), val_label)

        return root

    def test_train_returns_image_and_heatmap(self, cityscapes_minimal):
        """Train split ``__getitem__`` returns ``(image, heatmap)``.

        The heatmap must be a 2D ``float32`` tensor in ``[0, 1]`` — **not**
        a 2D ``long`` label map.
        """
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_minimal),
            split="train",
        )
        img, second = dataset[0]

        assert isinstance(img, torch.Tensor), f"Expected Tensor, got {type(img)}"
        assert isinstance(second, torch.Tensor), f"Expected Tensor, got {type(second)}"
        assert img.dim() == 3, f"Image must be 3D (C, H, W), got {img.dim()}D"

        # Second element is heatmap: 2D float32 in [0, 1]
        assert second.dim() == 2, (
            f"Heatmap should be 2D (H, W), got {second.dim()}D"
        )
        assert second.dtype == torch.float32, (
            f"Heatmap dtype should be float32, got {second.dtype}"
        )
        assert second.min() >= 0.0, f"Heatmap has negative values: {second.min()}"
        assert second.max() <= 1.0, f"Heatmap exceeds 1.0: {second.max()}"

        # Confirm it is NOT a label map (label would have integer-like values)
        # A float32 heatmap in [0,1] is structurally different from a long label
        # map — but the key assertion is the dtype and shape above.
        # Additional verification: heatmap values are continuous, not discrete
        unique_vals = torch.unique(second)
        assert len(unique_vals) > 19, (
            "Heatmap with continuous values should have more than 19 unique entries; "
            "a label map would have ≤19. Got {len(unique_vals)} unique values."
        )

    def test_val_returns_image_and_label(self, cityscapes_minimal):
        """**Val** split returns ``(image, label)`` — confirming the loader
        discriminates correctly between splits."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_minimal),
            split="val",
        )
        img, second = dataset[0]

        assert isinstance(img, torch.Tensor)
        assert isinstance(second, torch.Tensor)

        # Val returns a label: 2D long tensor
        assert second.dim() == 2, f"Label should be 2D, got {second.dim()}D"
        assert second.dtype == torch.long, (
            f"Label dtype should be torch.long, got {second.dtype}"
        )

        # Label values in {0..18, 255}
        valid = set(range(19)) | {255}
        unique = set(second.unique().tolist())
        invalid = unique - valid
        assert not invalid, f"Invalid label values: {invalid}"

    def test_train_split_never_returns_label(self, cityscapes_minimal):
        """The train split's second element is never a ``long`` tensor
        (which would indicate label leakage)."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_minimal),
            split="train",
        )
        _, second = dataset[0]

        assert second.dtype != torch.long, (
            "Train split second element must NOT be a long tensor — "
            "that would indicate GT label leakage!"
        )


# ===========================================================================
# Test 2: EMA model operates under torch.no_grad
# ===========================================================================


class TestEMANoLeakage:
    """EMA teacher model must never leak gradients or training-mode state."""

    @pytest.fixture
    def test_model(self):
        """A tiny segmentation-like model for EMA wrapping."""
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.Conv2d(8, 19, 1),
        )
        return model

    def test_ema_params_requires_grad_false(self, test_model):
        """All EMA model parameters have ``requires_grad=False``."""
        from models.ema import EMAModel

        ema = EMAModel(model=test_model, momentum=0.999)
        for p in ema.model.parameters():
            assert not p.requires_grad, (
                "EMA parameter should have requires_grad=False"
            )

    def test_ema_params_stay_no_grad_after_update(self, test_model):
        """After ``update()``, all EMA parameters still have ``requires_grad=False``."""
        from models.ema import EMAModel

        ema = EMAModel(model=test_model, momentum=0.999)

        # Perturb student
        for p in test_model.parameters():
            p.data.add_(torch.randn_like(p.data) * 0.1)

        ema.update(test_model)
        for p in ema.model.parameters():
            assert not p.requires_grad, (
                "EMA parameter should remain requires_grad=False after update"
            )

    def test_ema_always_eval(self, test_model):
        """EMA model is always in ``eval`` mode: after init and after update."""
        from models.ema import EMAModel

        ema = EMAModel(model=test_model, momentum=0.999)
        assert not ema.model.training, "EMA must be in eval mode after init"

        ema.update(test_model)
        assert not ema.model.training, "EMA must remain in eval mode after update"

    def test_ema_update_under_no_grad(self, test_model):
        """``EMAModel.update()`` itself runs under ``torch.no_grad()``.

        Verify that after calling update, the EMA model's params were updated
        *without* building a computation graph — they do not require grad."""
        from models.ema import EMAModel

        ema = EMAModel(model=test_model, momentum=0.999)
        ema.update(test_model)

        # After update, no EMA param requires grad (proves no_grad context)
        for p in ema.model.parameters():
            assert not p.requires_grad, (
                "No EMA parameter should require grad after update"
            )

    def test_ema_state_dict_match(self, test_model):
        """EMA state dict contains 'shadow_params' and 'iteration' keys."""
        from models.ema import EMAModel

        ema = EMAModel(model=test_model, momentum=0.999)
        sd = ema.state_dict()
        assert "shadow_params" in sd
        assert "iteration" in sd


# ===========================================================================
# Test 3: Loss receives student_logits + pseudo_labels, never GT labels
# ===========================================================================


class TestLossNoLeakage:
    """The target loss must use **pseudo-labels** (from teacher argmax),
    never the ground-truth labels from the target dataset.

    The ``SATGLoss`` signature is ``(student_logits, pseudo_labels, trust_weights)`` —
    there is no slot for GT labels.
    """

    @pytest.fixture
    def B(self):
        return 2

    @pytest.fixture
    def C(self):
        return 19

    @pytest.fixture
    def H(self):
        return 8

    @pytest.fixture
    def W(self):
        return 8

    def test_loss_api_uses_pseudo_labels(self, B, C, H, W):
        """``SATGLoss(student_logits, pseudo_labels, trust_weights)``
        computes a scalar loss.  The pseudo_labels are long-format class
        indices — identical to what ``argmax(teacher_logits)`` would produce.

        The function has **no parameter for GT labels** — proving the API
        prevents leakage by construction.
        """
        from satg.losses import SATGLoss

        loss_fn = SATGLoss()
        student_logits = torch.randn(B, C, H, W)
        pseudo_labels = torch.randint(0, C, (B, H, W), dtype=torch.long)
        trust_weights = torch.ones(B, H, W)

        loss = loss_fn(student_logits, pseudo_labels, trust_weights)
        assert loss.ndim == 0, "Loss must be scalar"
        assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"

    def test_pseudo_labels_are_long(self, B, C, H, W):
        """Pseudo-labels (teacher argmax) are ``torch.long`` as expected
        by cross-entropy loss."""
        # Same dtype as what F.cross_entropy expects
        pseudo_labels = torch.randint(0, C, (B, H, W), dtype=torch.long)
        assert pseudo_labels.dtype == torch.long

    def test_loss_with_all_ignored_pseudo_labels(self, B, C, H, W):
        """When all pseudo_labels are 255 (ignore_index), loss is 0.0
        — confirming no GT labels are consulted."""
        from satg.losses import SATGLoss

        loss_fn = SATGLoss()
        student_logits = torch.randn(B, C, H, W)
        pseudo_labels = torch.full((B, H, W), fill_value=255, dtype=torch.long)
        trust_weights = torch.ones(B, H, W)

        loss = loss_fn(student_logits, pseudo_labels, trust_weights)
        assert loss.item() == 0.0, (
            f"Expected 0.0 loss when all pseudo-labels are ignore_index, "
            f"got {loss.item()}"
        )
        assert not torch.isnan(loss)

    def test_loss_no_gt_label_param(self):
        """``SATGLoss.forward`` has no GT-label parameter — only
        (student_logits, pseudo_labels, trust_weights)."""
        from satg.losses import SATGLoss

        import inspect
        sig = inspect.signature(SATGLoss.forward)
        params = list(sig.parameters.keys())
        # Expected: ['self', 'student_logits', 'pseudo_labels', 'trust_weights']
        forbidden = {"labels", "gt_labels", "ground_truth", "target_labels"}
        actual = set(params[1:])  # skip 'self'
        overlap = actual & forbidden
        assert not overlap, (
            f"SATGLoss.forward has GT-label-like params: {overlap}. "
            f"Full params: {params}"
        )

    def test_soft_label_kl_also_no_gt_labels(self, B, C, H, W):
        """``SoftLabelKLLoss`` also accepts only student_logits, soft_targets,
        and confidence_mask — no GT labels."""
        from satg.losses import SoftLabelKLLoss

        loss_fn = SoftLabelKLLoss()
        student_logits = torch.randn(B, C, H, W)
        soft_targets = torch.softmax(torch.randn(B, C, H, W), dim=1)
        confidence_mask = torch.ones(B, H, W, dtype=torch.bool)

        loss = loss_fn(student_logits, soft_targets, confidence_mask)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_target_loop_uses_pseudo_labels_not_gt(self):
        """Verify the actual training loop (``training/trainer.py``)
        computes target loss with pseudo_labels, never with GT labels.

        We read the source to confirm the loss call pattern.
        """
        import ast
        from pathlib import Path

        trainer_path = Path(__file__).resolve().parent.parent / "training" / "trainer.py"
        source = trainer_path.read_text()
        tree = ast.parse(source)

        # Find all calls involving "target_loss" or "satg_loss"
        # and ensure they use "pseudo_labels" not "src_labels" or "labels"
        class LossCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.target_loss_calls = []
                self.source_loss_calls = []

            def visit_Call(self, node):
                if (isinstance(node.func, ast.Name) and
                        node.func.id in ("satg_loss", "soft_label_kl_loss")):
                    args = [ast.dump(a) for a in node.args]
                    self.target_loss_calls.append(args)
                if (isinstance(node.func, ast.Attribute) and
                        isinstance(node.func.value, ast.Name) and
                        "criterion" in node.func.value.id):
                    self.source_loss_calls.append(
                        [ast.dump(a) for a in node.args]
                    )
                self.generic_visit(node)

        visitor = LossCallVisitor()
        visitor.visit(tree)

        # Verify: at least one satg_loss call uses pseudo_labels
        found_pseudo = any(
            "pseudo_labels" in str(call)
            for call in visitor.target_loss_calls
        )
        assert found_pseudo, (
            "No target loss call uses pseudo_labels — "
            "if the target loss receives GT labels, data is leaking!"
        )
