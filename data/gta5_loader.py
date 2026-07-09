"""GTA5 dataset loader for UDA semantic segmentation.

Loads synthetic GTA5 images and their preprocessed 19-class index-map labels.
Applies source-domain augmentation (spatial + colour jitter).
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.augmentations import SourceAugment
from data.label_mapping import map_gta5_label

# ImageNet normalisation (RGB order)
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class GTA5Dataset(Dataset):
    """GTA5 dataset returning (image, label) pairs for UDA training.

    The dataset expects preprocessed labels: either RGB palette-encoded PNGs
    (converted on-the-fly via `map_gta5_label`) or uint8 index maps with
    values in {0..32} (converted via ``GTA5_TO_CITYSCAPES``).

    Args:
        root: Root directory containing image and label subdirectories.
        img_subdir: Subdirectory for images (e.g. ``images/train``).
        label_subdir: Subdirectory for labels (e.g. ``labels/train``).
        crop_size: Target (H, W) for random crop during training.
            If ``None``, the original image size is kept.
        augment: Whether to apply source-domain augmentation
            (spatial transforms + colour jitter).
    """

    def __init__(
        self,
        root: str,
        img_subdir: str = "images/train",
        label_subdir: str = "labels/train",
        crop_size: Optional[Tuple[int, int]] = None,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.img_dir = self.root / img_subdir
        self.label_dir = self.root / label_subdir
        self.crop_size = crop_size
        self.augment = augment

        # Recursively discover image files
        self._samples: list[Tuple[Path, Path]] = []
        for img_path in sorted(self.img_dir.rglob("*.png")):
            # Derive the corresponding label path by replacing img_subdir
            # with label_subdir in the relative path
            rel = img_path.relative_to(self.img_dir)
            label_path = self.label_dir / rel
            if label_path.exists():
                self._samples.append((img_path, label_path))

        if len(self._samples) == 0:
            raise FileNotFoundError(
                f"No PNG files found in {self.img_dir} " f"with matching labels in {self.label_dir}"
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self._samples[idx]

        # --- Load image (BGR → RGB float32) ---
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # --- Load label ---
        label_raw = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label_raw is None:
            raise FileNotFoundError(f"Failed to load label: {label_path}")

        # If label is RGB palette-encoded, convert to trainIDs
        if label_raw.ndim == 3:
            label = map_gta5_label(label_raw)
        else:
            # Already an index map — map GTA5 IDs → Cityscapes trainIDs
            from data.label_mapping import GTA5_TO_CITYSCAPES

            label = np.vectorize(GTA5_TO_CITYSCAPES.__getitem__)(label_raw).astype(np.uint8)

        # --- Source-domain augmentation ---
        if self.augment and self.crop_size is not None:
            aug = SourceAugment(crop_size=self.crop_size)
            img_rgb, label = aug(img_rgb, label)

        # --- Normalise image ---
        img_rgb = (img_rgb - _MEAN) / _STD  # (H, W, 3)

        # --- Convert to tensors ---
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()  # (C,H,W)
        label_tensor = torch.from_numpy(label).long()

        return img_tensor, label_tensor


def _centre_crop(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Centre-crop a 2D or 3D numpy array."""
    cur_h, cur_w = arr.shape[:2]
    y_start = max(0, (cur_h - h) // 2)
    x_start = max(0, (cur_w - w) // 2)
    return arr[y_start : y_start + h, x_start : x_start + w]
