"""Cityscapes dataset loader for UDA semantic segmentation.

Supports two modes:
- Train split: returns (image, heatmap) with precomputed structural heatmaps.
- Val split: returns (image, label) with ground-truth labels for evaluation.
- skip_heatmap flag allows disabling heatmap loading (baseline_mean_teacher).
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.augmentations import TargetAugment

_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

_LABEL_SUFFIX = "_gtFine_labelIds.png"
_IMG_SUFFIX = "_leftImg8bit.png"
_HEATMAP_SUFFIX = "_satg_heatmap.npy"


class CityscapesDataset(Dataset):
    """Cityscapes dataset for UDA training and evaluation.

    For the **train** split returns ``(image, heatmap)`` where heatmap
    is a precomputed structural complexity map loaded from ``heatmap_root``,
    or an all-ones dummy if ``skip_heatmap=True``.
    For the **val** split returns ``(image, label)`` with ground-truth trainIDs.

    Args:
        root: Root directory of Cityscapes dataset (leftImg8bit + gtFine).
        split: ``"train"`` or ``"val"``.
        heatmap_root: Path to directory containing precomputed .npy heatmap
            files mirroring the leftImg8bit/train structure. Required when
            ``split="train"`` and ``skip_heatmap=False``.
        crop_size: Target (H, W). If ``None``, original image size is kept.
        augment: Whether to apply target-domain augmentation (flip + crop).
        skip_heatmap: If ``True``, return an all-ones dummy heatmap instead
            of loading from disk (used by source_only and mean_teacher baselines).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        heatmap_root: Optional[str] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
        skip_heatmap: bool = False,
    ) -> None:
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.augment = augment
        self.skip_heatmap = skip_heatmap

        img_dir = self.root / "leftImg8bit" / split

        if split == "train":
            # For training, discover images; heatmaps loaded on demand
            self._samples: list[Path] = []
            for img_path in sorted(img_dir.rglob(f"*{_IMG_SUFFIX}")):
                self._samples.append(img_path)

            if len(self._samples) == 0:
                raise FileNotFoundError(f"No training images found in {img_dir}")

            if heatmap_root is not None:
                self._heatmap_root = Path(heatmap_root)
            else:
                self._heatmap_root = None
        else:
            # For validation, discover image-label pairs
            label_dir = self.root / "gtFine" / split
            self._samples: list[Tuple[Path, Path]] = []
            for img_path in sorted(img_dir.rglob(f"*{_IMG_SUFFIX}")):
                stem = img_path.name[: -len(_IMG_SUFFIX)]
                label_path = (
                    label_dir / img_path.relative_to(img_dir).parent / f"{stem}{_LABEL_SUFFIX}"
                )
                if label_path.exists():
                    self._samples.append((img_path, label_path))

            if len(self._samples) == 0:
                raise FileNotFoundError(
                    f"No validation images found in {img_dir} "
                    f"with matching labels in {label_dir}"
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.split == "train":
            return self._get_train_item(idx)
        else:
            return self._get_val_item(idx)

    def _get_train_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self._samples[idx]

        # --- Load image (BGR → RGB float32) ---
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # --- Load heatmap ---
        if self.skip_heatmap:
            img_tensor = torch.from_numpy((img_rgb - _MEAN) / _STD).permute(2, 0, 1).float()
            return img_tensor, torch.zeros(1)

        stem = img_path.name[: -len(_IMG_SUFFIX)]
        rel = img_path.relative_to(
            img_path.parent.parent.parent
        )  # leftImg8bit/train/city/stem_leftImg8bit.png
        if self._heatmap_root is not None:
            heat_path = (
                self._heatmap_root
                / rel.parent.parent
                / rel.parent.name
                / f"{stem}{_HEATMAP_SUFFIX}"
            )
        else:
            heat_path = img_path.with_name(f"{stem}{_HEATMAP_SUFFIX}")

        if not heat_path.exists():
            raise FileNotFoundError(f"Heatmap not found: {heat_path}")

        heatmap = np.load(str(heat_path)).astype(np.float32)

        # --- Augmentation (spatial only: flip + crop) ---
        if self.augment:
            aug = TargetAugment(crop_size=self.crop_size)
            img_rgb, heatmap = aug(img_rgb, heatmap)
        elif self.crop_size is not None:
            h, w = self.crop_size
            img_rgb = _centre_crop(img_rgb, h, w)
            heatmap = _centre_crop(heatmap, h, w)

        # --- Normalise image ---
        img_rgb = (img_rgb - _MEAN) / _STD

        # --- Convert to tensors ---
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        heatmap_tensor = torch.from_numpy(heatmap).float()

        return img_tensor, heatmap_tensor

    def _get_val_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self._samples[idx]

        # --- Load image (BGR → RGB float32) ---
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # --- Load label (single-channel, already trainIDs) ---
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label is None:
            raise FileNotFoundError(f"Failed to load label: {label_path}")

        # --- Centre crop for consistent evaluation ---
        if self.crop_size is not None:
            h, w = self.crop_size
            img_rgb = _centre_crop(img_rgb, h, w)
            label = _centre_crop(label, h, w)

        # --- Normalise image ---
        img_rgb = (img_rgb - _MEAN) / _STD

        # --- Convert to tensors ---
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        label_tensor = torch.from_numpy(label).long()

        return img_tensor, label_tensor


def _centre_crop(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Centre-crop a 2D or 3D numpy array."""
    cur_h, cur_w = arr.shape[:2]
    y_start = max(0, (cur_h - h) // 2)
    x_start = max(0, (cur_w - w) // 2)
    return arr[y_start : y_start + h, x_start : x_start + w]
