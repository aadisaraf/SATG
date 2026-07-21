"""Data augmentation for UDA semantic segmentation.

Provides deterministic, seed-reproducible augmentation pipelines for
source (GTA5) and target (Cityscapes) domains. All spatial transforms
are applied consistently to both image and label arrays.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class _BaseAugment:
    """Base class for seedable augmentation pipelines."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.RandomState(seed)

    def _maybe_flip(
        self, image: np.ndarray, label: np.ndarray, p: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Horizontally flip both image and label with probability p."""
        if self._rng.rand() < p:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        return image, label


class SourceAugment(_BaseAugment):
    """Source-domain augmentation for GTA5 training.

    Applies spatial transforms and colour jitter consistently,
    with seed reproducibility.

    Args:
        crop_size: Target (H, W) after random crop. Default (512, 512).
        min_scale: Minimum scale factor for random scaling. Default 0.5.
        max_scale: Maximum scale factor for random scaling. Default 1.5.
        brightness: Colour jitter brightness max delta. Default 0.4.
        contrast: Colour jitter contrast max delta. Default 0.4.
        saturation: Colour jitter saturation max delta. Default 0.4.
        hue: Colour jitter hue max delta. Default 0.1.
        seed: Random seed for reproducibility. If None, no fixed seed.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int] = (512, 512),
        min_scale: float = 0.5,
        max_scale: float = 1.5,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation and return (image, label) as numpy arrays."""
        # 1. Random horizontal flip (both image and label)
        image, label = self._maybe_flip(image, label)

        # 2. Random scale then crop (both)
        image, label = self._random_scale_crop(image, label)

        # 3. Colour jitter (image only)
        image = self._colour_jitter(image)

        return image, label

    def _random_scale_crop(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly scale then centre-crop to crop_size."""
        h, w = image.shape[:2]
        target_h, target_w = self.crop_size

        # Random scale factor
        scale = self._rng.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))

        if scale != 1.0:
            interpolation = cv2.INTER_LINEAR
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Centre crop to target size
        return self._centre_crop(image, label, target_h, target_w)

    @staticmethod
    def _centre_crop(
        image: np.ndarray, label: np.ndarray, h: int, w: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Centre-crop image and label consistently to exactly (h, w).

        If the (possibly down-scaled) input is smaller than the crop in any
        dimension, reflect-pad it up first — otherwise plain slicing returns an
        undersized crop, and batching tensors of different sizes crashes the
        DataLoader collate ("Trying to resize storage that is not resizable").
        """
        cur_h, cur_w = image.shape[:2]
        pad_h = max(0, h - cur_h)
        pad_w = max(0, w - cur_w)
        if pad_h or pad_w:
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
            label = cv2.copyMakeBorder(
                label, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
            cur_h, cur_w = image.shape[:2]
        y_start = max(0, (cur_h - h) // 2)
        x_start = max(0, (cur_w - w) // 2)
        return (
            image[y_start : y_start + h, x_start : x_start + w],
            label[y_start : y_start + h, x_start : x_start + w],
        )

    def _colour_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness, contrast, saturation, and hue jitter.

        Operates on uint8 RGB images and returns uint8.
        """
        img = image.astype(np.float32)

        # Brightness
        if self.brightness > 0:
            delta = self._rng.uniform(-self.brightness, self.brightness)
            img += delta * 255.0

        # Contrast
        if self.contrast > 0:
            factor = self._rng.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
            mean = img.mean(axis=(0, 1), keepdims=True)
            img = mean + factor * (img - mean)

        # Saturation and hue via HSV
        if self.saturation > 0 or self.hue > 0:
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

            if self.saturation > 0:
                s_factor = self._rng.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)

            if self.hue > 0:
                h_delta = self._rng.uniform(-self.hue, self.hue) * 255.0
                hsv[:, :, 0] = (hsv[:, :, 0] + h_delta) % 180.0

            img_rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(
                np.float32
            )
            img = img_rgb

        return np.clip(img, 0, 255).astype(np.uint8)


class TargetAugment(_BaseAugment):
    """Target-domain augmentation for Cityscapes training.

    Milder than source: only random horizontal flip by default,
    optionally random scale + crop.

    Args:
        crop_size: Target (H, W) after centre crop. If None, keep original size.
        seed: Random seed for reproducibility. If None, no fixed seed.
    """

    def __init__(
        self,
        crop_size: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.crop_size = crop_size

    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation and return (image, label) as numpy arrays."""
        # 1. Random horizontal flip
        image, label = self._maybe_flip(image, label)

        # 2. Centre crop if specified
        if self.crop_size is not None:
            h, w = self.crop_size
            image, label = SourceAugment._centre_crop(image, label, h, w)

        return image, label
