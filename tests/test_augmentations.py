"""Tests for data augmentation utilities (RED → GREEN).

Verifies that spatial transforms are applied consistently to
image and label pairs, and that augmentation is reproducible.
"""

import numpy as np
import pytest


@pytest.fixture(scope="function")
def image_label_pair():
    """Return a synthetic (image, label) pair as numpy arrays."""
    np.random.seed(42)
    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    label = np.random.randint(0, 19, (64, 64), dtype=np.uint8)
    # Put a distinctive pattern in the top-left corner to verify spatial consistency
    image[:16, :16] = 255
    label[:16, :16] = 7  # Class 7 (traffic sign)
    return image, label


class TestSourceAugment:
    """Tests for SourceAugment — GTA5 domain augmentation."""

    def test_module_exists(self):
        from data.augmentations import SourceAugment

        assert SourceAugment is not None

    def test_augment_returns_tuple(self, image_label_pair):
        from data.augmentations import SourceAugment

        img, label = image_label_pair
        aug = SourceAugment(crop_size=(32, 32))
        result = aug(img, label)
        assert isinstance(result, tuple) and len(result) == 2

    def test_augment_returns_numpy(self, image_label_pair):
        from data.augmentations import SourceAugment

        img, label = image_label_pair
        aug = SourceAugment(crop_size=(32, 32))
        out_img, out_label = aug(img, label)
        assert isinstance(out_img, np.ndarray)
        assert isinstance(out_label, np.ndarray)

    def test_output_shapes(self, image_label_pair):
        from data.augmentations import SourceAugment

        img, label = image_label_pair
        aug = SourceAugment(crop_size=(32, 32))
        out_img, out_label = aug(img, label)
        assert out_img.shape == (32, 32, 3), f"Got {out_img.shape}"
        assert out_label.shape == (32, 32), f"Got {out_label.shape}"

    def test_image_dtype_preserved(self, image_label_pair):
        from data.augmentations import SourceAugment

        img, label = image_label_pair
        aug = SourceAugment(crop_size=(32, 32))
        out_img, _ = aug(img, label)
        assert out_img.dtype == img.dtype

    def test_label_dtype_preserved(self, image_label_pair):
        from data.augmentations import SourceAugment

        img, label = image_label_pair
        aug = SourceAugment(crop_size=(32, 32))
        _, out_label = aug(img, label)
        assert out_label.dtype == label.dtype

    def test_spatial_transform_consistency(self, image_label_pair):
        """Centre crop applies same coordinates to image and label."""
        from data.augmentations import SourceAugment

        img, label = image_label_pair
        h, w = 32, 32
        out_img, out_label = SourceAugment._centre_crop(img, label, h, w)

        cur_h, cur_w = img.shape[:2]
        y_start = (cur_h - h) // 2
        x_start = (cur_w - w) // 2

        np.testing.assert_array_equal(out_img, img[y_start : y_start + h, x_start : x_start + w])
        np.testing.assert_array_equal(
            out_label, label[y_start : y_start + h, x_start : x_start + w]
        )

    def test_label_values_preserved(self, image_label_pair):
        """Augmentation must not introduce invalid label values."""
        from data.augmentations import SourceAugment

        img, label = image_label_pair
        aug = SourceAugment(crop_size=(32, 32))
        _, out_label = aug(img, label)
        valid = set(range(19)) | {255}
        invalid = set(out_label.flatten().tolist()) - valid
        assert not invalid, f"Invalid label values: {invalid}"

    def test_seed_reproducibility(self, image_label_pair):
        """Same seed must produce same augmentation result."""
        from data.augmentations import SourceAugment

        img, label = image_label_pair
        aug1 = SourceAugment(crop_size=(32, 32), seed=42)
        aug2 = SourceAugment(crop_size=(32, 32), seed=42)
        r1_img, r1_lbl = aug1(img.copy(), label.copy())
        r2_img, r2_lbl = aug2(img.copy(), label.copy())
        np.testing.assert_array_equal(r1_img, r2_img)
        np.testing.assert_array_equal(r1_lbl, r2_lbl)

    def test_different_seed_different_result(self, image_label_pair):
        """Different seeds should produce different results."""
        from data.augmentations import SourceAugment

        img, label = image_label_pair
        aug1 = SourceAugment(crop_size=(32, 32), seed=42)
        aug2 = SourceAugment(crop_size=(32, 32), seed=99)
        r1_img, _ = aug1(img.copy(), label.copy())
        r2_img, _ = aug2(img.copy(), label.copy())
        assert not np.array_equal(r1_img, r2_img), "Different seeds produced identical result"

    def test_color_jitter_only_on_image(self, image_label_pair):
        """Color jitter must only affect image, not label."""
        from data.augmentations import SourceAugment

        img, label = image_label_pair
        # Use seed=1 for spatial; spatial affects both, colour only image.
        # After spatial transform, the label values are the same as before.
        aug = SourceAugment(crop_size=(32, 32), seed=1)
        _, out_label = aug(img.copy(), label.copy())
        # Label values should still be valid trainIDs (no colour shift on labels)
        assert set(np.unique(out_label)).issubset(set(range(19)) | {255})


class TestTargetAugment:
    """Tests for TargetAugment — Cityscapes training augmentation."""

    def test_module_exists(self):
        from data.augmentations import TargetAugment

        assert TargetAugment is not None

    def test_flip_only_default(self, image_label_pair):
        """TargetAugment with default args should only flip, keeping size."""
        from data.augmentations import TargetAugment

        img, label = image_label_pair
        aug = TargetAugment()
        out_img, out_label = aug(img, label)
        assert out_img.shape[:2] == img.shape[:2]
        assert out_label.shape == label.shape

    def test_seed_reproducibility(self, image_label_pair):
        from data.augmentations import TargetAugment

        img, label = image_label_pair
        aug1 = TargetAugment(seed=42)
        aug2 = TargetAugment(seed=42)
        r1_img, r1_lbl = aug1(img.copy(), label.copy())
        r2_img, r2_lbl = aug2(img.copy(), label.copy())
        np.testing.assert_array_equal(r1_img, r2_img)
        np.testing.assert_array_equal(r1_lbl, r2_lbl)
