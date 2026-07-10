"""Tests for mIoU evaluation (RED phase — all should fail until ``Evaluator`` class exists).

The evaluation module currently exposes only a ``compute_miou()`` function.
These tests expect an ``Evaluator`` class (with ``__init__``, ``__call__``,
``per_class_iou``, and ``mean_iou`` methods) that does not exist yet.
"""

import torch
import pytest


# ===========================================================================
# T034: Evaluator class tests
# ===========================================================================


class TestEvaluator:
    """Expected ``Evaluator`` class-based API tests.

    The current implementation has a standalone ``compute_miou()`` function.
    These tests define the class-based interface that should wrap or replace it.
    """

    def _evaluator(self, num_classes=19, ignore_index=255):
        """Import and instantiate the non-existent ``Evaluator`` — raises ImportError."""
        from evaluation.evaluator import Evaluator  # noqa — expected ImportError
        return Evaluator(num_classes=num_classes, ignore_index=ignore_index)

    @pytest.fixture
    def B(self):
        return 2

    @pytest.fixture
    def C(self):
        return 19

    @pytest.fixture
    def H(self):
        return 2

    @pytest.fixture
    def W(self):
        return 2

    def test_miou_perfect_prediction(self, B, C, H, W):
        """When every pixel is correctly classified, mIoU = 100%."""
        labels = torch.randint(0, C, (B, H, W))
        model = _PerfectModel(labels)

        from torch.utils.data import DataLoader, TensorDataset
        imgs = torch.randn(B, 3, H, W)
        dataset = TensorDataset(imgs, labels)
        loader = DataLoader(dataset, batch_size=B)

        evaluator = self._evaluator(num_classes=C)
        mean_iou, per_class = evaluator(loader)
        assert mean_iou == 100.0

    def test_miou_all_wrong(self, B, C, H, W):
        """When all pixels are misclassified, mIoU = ``(1/C) * 100``."""
        labels = torch.zeros(B, H, W, dtype=torch.long)
        model = _ConstantModel(C - 1)

        from torch.utils.data import DataLoader, TensorDataset
        imgs = torch.randn(B, 3, H, W)
        dataset = TensorDataset(imgs, labels)
        loader = DataLoader(dataset, batch_size=B)

        evaluator = self._evaluator(num_classes=C)
        mean_iou, per_class = evaluator(loader)

        expected = 100.0 / C
        assert abs(mean_iou - expected) < 1e-4

    def test_per_class_iou_19_classes(self, B, H, W):
        """per_class_iou returns a dict with exactly 19 entries."""
        from evaluation.evaluator import CITYSCAPES_19

        labels = torch.randint(0, 19, (B, H, W))
        model = _PerfectModel(labels)

        from torch.utils.data import DataLoader, TensorDataset
        imgs = torch.randn(B, 3, H, W)
        dataset = TensorDataset(imgs, labels)
        loader = DataLoader(dataset, batch_size=B)

        evaluator = self._evaluator(num_classes=19)
        mean_iou, per_class = evaluator(loader)

        assert isinstance(per_class, dict)
        assert len(per_class) == 19
        for cls_name in CITYSCAPES_19:
            assert cls_name in per_class
            assert 0.0 <= per_class[cls_name] <= 100.0

    def test_ignore_index_excluded(self, B, C, H, W):
        """Pixels with ``ignore_index=255`` do not affect mIoU."""
        labels = torch.randint(0, C, (B, H, W))
        labels[:, 0, 0] = 255

        model = _PerfectModel(labels)

        from torch.utils.data import DataLoader, TensorDataset
        imgs = torch.randn(B, 3, H, W)
        dataset = TensorDataset(imgs, labels)
        loader = DataLoader(dataset, batch_size=B)

        evaluator = self._evaluator(num_classes=C, ignore_index=255)
        mean_iou, per_class = evaluator(loader)
        assert mean_iou == 100.0

    def test_ignore_index_all_ignored(self, B, C, H, W):
        """All pixels ignored → mIoU = 0.0."""
        labels = torch.full((B, H, W), fill_value=255, dtype=torch.long)
        model = _ConstantModel(0)

        from torch.utils.data import DataLoader, TensorDataset
        imgs = torch.randn(B, 3, H, W)
        dataset = TensorDataset(imgs, labels)
        loader = DataLoader(dataset, batch_size=B)

        evaluator = self._evaluator(num_classes=C, ignore_index=255)
        mean_iou, per_class = evaluator(loader)
        assert mean_iou == 0.0

    def test_empty_predictions(self, B, C, H, W):
        """Class with zero pixels gets 0.0 IoU (not NaN)."""
        labels = torch.zeros(B, H, W, dtype=torch.long)
        model = _ConstantModel(0)

        from torch.utils.data import DataLoader, TensorDataset
        imgs = torch.randn(B, 3, H, W)
        dataset = TensorDataset(imgs, labels)
        loader = DataLoader(dataset, batch_size=B)

        evaluator = self._evaluator(num_classes=C)
        mean_iou, per_class = evaluator(loader)

        for cls_name, iou in per_class.items():
            assert not torch.isnan(torch.tensor(iou)), f"{cls_name} IoU is NaN"

    def test_miou_formula(self, B, C, H, W):
        """Verify mIoU = mean(IoU_c) with known confusion."""
        labels = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).unsqueeze(0)
        model = _PerfectModel(labels)

        from torch.utils.data import DataLoader, TensorDataset
        imgs = torch.randn(1, 3, 2, 2)
        dataset = TensorDataset(imgs, labels)
        loader = DataLoader(dataset, batch_size=1)

        evaluator = self._evaluator(num_classes=3)
        mean_iou, per_class = evaluator(loader)
        assert mean_iou == 100.0

    def test_scalar_miou_output(self, B, C, H, W):
        """``mean_iou`` is a plain Python float."""
        labels = torch.randint(0, C, (B, H, W))
        model = _PerfectModel(labels)

        from torch.utils.data import DataLoader, TensorDataset
        imgs = torch.randn(B, 3, H, W)
        dataset = TensorDataset(imgs, labels)
        loader = DataLoader(dataset, batch_size=B)

        evaluator = self._evaluator(num_classes=C)
        mean_iou, per_class = evaluator(loader)
        assert isinstance(mean_iou, float)


# ===========================================================================
# Helper models
# ===========================================================================


class _PerfectModel(torch.nn.Module):
    """Model that returns logits perfectly matching the provided labels."""

    def __init__(self, labels: torch.Tensor):
        super().__init__()
        self._labels = labels  # (B, H, W)

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        C = 19
        logits = torch.full((B, C, H, W), -100.0)
        labels_upsampled = self._labels.unsqueeze(1).expand(B, 1, H, W)
        # Put very high logit at the correct class index
        logits.scatter_(1, labels_upsampled, 100.0)
        # Return (main_logits, aux) as the model normally does
        return logits, logits


class _ConstantModel(torch.nn.Module):
    """Model that always predicts the same class for every pixel."""

    def __init__(self, class_idx: int):
        super().__init__()
        self._class_idx = class_idx

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        C = 19
        logits = torch.full((B, C, H, W), -100.0)
        logits[:, self._class_idx, :, :] = 100.0
        return logits, logits
