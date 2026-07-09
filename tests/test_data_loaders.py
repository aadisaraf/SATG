"""Tests for data loaders and label mapping.

Test-first (RED → GREEN) per TDD workflow.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


# =============================================================================
# T006: Label mapping tests (RED phase — module not yet implemented)
# =============================================================================


class TestLabelMapping:
    """Tests for GTA5 → Cityscapes 19-class label mapping."""

    def test_mapping_covers_all_19_cityscapes_classes(self):
        """The mapping dict must contain at least one GTA5 class mapping
        to each of the 19 Cityscapes train IDs."""
        from data.label_mapping import GTA5_TO_CITYSCAPES

        cityscapes_ids = set(range(19))
        mapped_ids = set(v for v in GTA5_TO_CITYSCAPES.values() if v != 255)
        missing = cityscapes_ids - mapped_ids
        assert not missing, f"Missing Cityscapes train IDs in mapping: {missing}"

    def test_mapping_dict_completeness(self):
        """All 33 GTA5 class IDs (0–32) must have a mapping entry."""
        from data.label_mapping import GTA5_TO_CITYSCAPES

        gta5_ids = set(range(33))
        mapped_ids = set(GTA5_TO_CITYSCAPES.keys())
        missing = gta5_ids - mapped_ids
        extra = mapped_ids - gta5_ids
        assert not missing, f"Missing GTA5 IDs in mapping: {missing}"
        assert not extra, f"Extra GTA5 IDs not in 0–32 range: {extra}"

    def test_unmapped_classes_raise_error(self):
        """A GTA5 class ID not in the dict should raise KeyError
        (though the dict covers all 33 IDs, testing defensive coding)."""
        from data.label_mapping import GTA5_TO_CITYSCAPES

        with pytest.raises(KeyError):
            GTA5_TO_CITYSCAPES[99]

    def test_mapped_to_valid_ids(self):
        """All mapped values must be in {0..18} ∪ {255}."""
        from data.label_mapping import GTA5_TO_CITYSCAPES

        valid = set(range(19)) | {255}
        for gta5_id, cs_id in GTA5_TO_CITYSCAPES.items():
            assert cs_id in valid, f"GTA5 ID {gta5_id} maps to invalid Cityscapes ID {cs_id}"

    def test_map_gta5_label_function(self):
        """map_gta5_label converts an RGB label map to Cityscapes trainIDs.
        Verified in detail by shape, dtype, and value tests below.
        """
        from data.label_mapping import map_gta5_label

        assert callable(map_gta5_label)

    def test_map_gta5_label_output_shape(self):
        """Output shape must match input spatial dims."""
        from data.label_mapping import map_gta5_label

        label_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        result = map_gta5_label(label_rgb)
        assert result.shape == (64, 64), f"Expected shape (64, 64), got {result.shape}"

    def test_map_gta5_label_output_dtype(self):
        """Output dtype must be uint8 for efficient storage."""
        from data.label_mapping import map_gta5_label

        label_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        result = map_gta5_label(label_rgb)
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"

    def test_map_gta5_label_values_in_range(self):
        """Output values must be in {0..18} ∪ {255}."""
        from data.label_mapping import map_gta5_label

        label_rgb = np.ones((16, 16, 3), dtype=np.uint8) * 128
        result = map_gta5_label(label_rgb)
        valid = set(range(19)) | {255}
        unique_vals = set(result.ravel().tolist())
        invalid = unique_vals - valid
        assert not invalid, f"Invalid label values found: {invalid}"


# =============================================================================
# Fixtures: synthetic GTA5 data for loader tests
# =============================================================================


@pytest.fixture(scope="function")
def gta5_data_root():
    """Create a temporary GTA5-like directory with synthetic images and labels.

    Structure:
        tmpdir/
          images/train/cityA/AAA_1.png
          images/train/cityB/BBB_1.png
          labels/train/cityA/AAA_1.png    (index map, uint8, 0..32)
          labels/train/cityB/BBB_1.png
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        img_dir = root / "images" / "train"
        lbl_dir = root / "labels" / "train"

        cities = {
            "cityA": ["AAA_1", "AAA_2"],
            "cityB": ["BBB_1"],
        }

        for city, stems in cities.items():
            (img_dir / city).mkdir(parents=True)
            (lbl_dir / city).mkdir(parents=True)
            for stem in stems:
                # Create synthetic RGB image (H=64, W=64)
                img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(img_dir / city / f"{stem}.png"), img)

                # Create synthetic label index map (H=64, W=64, values 0..32)
                label = np.random.randint(0, 19, (64, 64), dtype=np.uint8)
                cv2.imwrite(str(lbl_dir / city / f"{stem}.png"), label)

        yield root


# =============================================================================
# T008: GTA5 loader tests (RED phase — module not yet implemented)
# =============================================================================


class TestGTA5Loader:
    """Tests for GTA5Dataset class."""

    def test_len_returns_correct_count(self, gta5_data_root):
        """Dataset.__len__ must return the number of image-label pairs."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root=str(gta5_data_root),
            img_subdir="images/train",
            label_subdir="labels/train",
        )
        assert len(dataset) == 3, f"Expected 3 samples, got {len(dataset)}"

    def test_getitem_returns_tuple(self, gta5_data_root):
        """__getitem__ must return a (image, label) tuple."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root=str(gta5_data_root),
            img_subdir="images/train",
            label_subdir="labels/train",
        )
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor), f"Expected Tensor, got {type(img)}"
        assert isinstance(label, torch.Tensor), f"Expected Tensor, got {type(label)}"

    def test_getitem_image_shape(self, gta5_data_root):
        """Image tensor must have shape (C, H, W)."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root=str(gta5_data_root),
            img_subdir="images/train",
            label_subdir="labels/train",
        )
        img, _ = dataset[0]
        assert img.dim() == 3, f"Expected 3D tensor, got {img.dim()}D"
        c, h, w = img.shape
        assert c == 3, f"Expected 3 channels, got {c}"
        assert h > 0 and w > 0, f"Invalid spatial dims: {h}x{w}"

    def test_getitem_image_dtype(self, gta5_data_root):
        """Image tensor must be float32."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root=str(gta5_data_root),
            img_subdir="images/train",
            label_subdir="labels/train",
        )
        img, _ = dataset[0]
        assert img.dtype == torch.float32, f"Expected float32, got {img.dtype}"

    def test_getitem_label_shape(self, gta5_data_root):
        """Label tensor must be 2D (H, W)."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root=str(gta5_data_root),
            img_subdir="images/train",
            label_subdir="labels/train",
        )
        _, label = dataset[0]
        assert label.dim() == 2, f"Expected 2D label, got {label.dim()}D"

    def test_getitem_label_dtype(self, gta5_data_root):
        """Label tensor must be int64 (torch.long) for cross-entropy."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root=str(gta5_data_root),
            img_subdir="images/train",
            label_subdir="labels/train",
        )
        _, label = dataset[0]
        assert label.dtype == torch.long, f"Expected torch.long, got {label.dtype}"

    def test_label_values_in_valid_range(self, gta5_data_root):
        """Label values must be in {0..18} ∪ {255} after mapping."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root=str(gta5_data_root),
            img_subdir="images/train",
            label_subdir="labels/train",
        )
        _, label = dataset[0]
        valid = set(range(19)) | {255}
        unique_vals = set(label.unique().tolist())
        invalid = unique_vals - valid
        assert not invalid, f"Invalid label values found: {invalid}"

    def test_multiple_samples_unique(self, gta5_data_root):
        """Different indices must return different samples."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root=str(gta5_data_root),
            img_subdir="images/train",
            label_subdir="labels/train",
        )
        img0, _ = dataset[0]
        img1, _ = dataset[1]
        # Very unlikely that two random images are identical
        assert not torch.equal(img0, img1), "Two samples returned identical images"


# =============================================================================
# T010: Cityscapes loader tests (RED phase)
# =============================================================================


@pytest.fixture(scope="function")
def cityscapes_data_root():
    """Create a temporary Cityscapes-like directory with synthetic data.

    Structure:
        tmpdir/
          leftImg8bit/
            train/cityA/cityA_000001_leftImg8bit.png
            train/cityB/cityB_000001_leftImg8bit.png
            val/cityC/cityC_000001_leftImg8bit.png
          gtFine/
            train/cityA/cityA_000001_gtFine_labelIds.png
            train/cityB/cityB_000001_gtFine_labelIds.png
            val/cityC/cityC_000001_gtFine_labelIds.png
          heatmaps/ (mirrors leftImg8bit/train structure with .npy files)
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        cities_train = {
            "cityA": ["000001_000000", "000002_000000"],
            "cityB": ["000003_000000"],
        }
        cities_val = {
            "cityC": ["000004_000000"],
        }

        # Training images + labels + heatmaps
        for city, stems in cities_train.items():
            img_dir = root / "leftImg8bit" / "train" / city
            lbl_dir = root / "gtFine" / "train" / city
            heat_dir = root / "heatmaps" / "train" / city
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            heat_dir.mkdir(parents=True)
            for stem in stems:
                img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(img_dir / f"{stem}_leftImg8bit.png"), img)
                # label values in 0..18 ∪ {255}
                label = np.random.choice(list(range(19)) + [255], (64, 64)).astype(np.uint8)
                cv2.imwrite(str(lbl_dir / f"{stem}_gtFine_labelIds.png"), label)
                # heatmap .npy
                heatmap = np.random.rand(64, 64).astype(np.float32)
                np.save(str(heat_dir / f"{stem}_satg_heatmap.npy"), heatmap)

        # Validation images + labels (no heatmaps)
        for city, stems in cities_val.items():
            img_dir = root / "leftImg8bit" / "val" / city
            lbl_dir = root / "gtFine" / "val" / city
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            for stem in stems:
                img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(img_dir / f"{stem}_leftImg8bit.png"), img)
                label = np.random.choice(list(range(19)) + [255], (64, 64)).astype(np.uint8)
                cv2.imwrite(str(lbl_dir / f"{stem}_gtFine_labelIds.png"), label)

        yield root


class TestCityscapesLoader:
    """Tests for CityscapesDataset class."""

    # ------------------------------------------------------------------ #
    # Train split — returns (image, heatmap)
    # ------------------------------------------------------------------ #

    def test_train_len(self, cityscapes_data_root):
        """Dataset.__len__ returns correct count for train split."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root=str(cityscapes_data_root),
            split="train",
            heatmap_root=str(cityscapes_data_root / "heatmaps"),
        )
        assert len(dataset) == 3, f"Expected 3, got {len(dataset)}"

    def test_train_getitem_returns_tuple(self, cityscapes_data_root):
        """Train __getitem__ returns (image, heatmap)."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root=str(cityscapes_data_root),
            split="train",
            heatmap_root=str(cityscapes_data_root / "heatmaps"),
        )
        img, heatmap = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(heatmap, torch.Tensor)

    def test_train_heatmap_shape(self, cityscapes_data_root):
        """Heatmap shape matches image spatial dims."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root=str(cityscapes_data_root),
            split="train",
            heatmap_root=str(cityscapes_data_root / "heatmaps"),
        )
        img, heatmap = dataset[0]
        assert heatmap.dim() == 2, f"Expected 2D, got {heatmap.dim()}D"
        _, h, w = img.shape
        assert heatmap.shape == (h, w), f"Heatmap shape {heatmap.shape} != image spatial {h}x{w}"

    def test_train_heatmap_values_in_01(self, cityscapes_data_root):
        """Heatmap values must be in [0, 1]."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root=str(cityscapes_data_root),
            split="train",
            heatmap_root=str(cityscapes_data_root / "heatmaps"),
        )
        _, heatmap = dataset[0]
        assert heatmap.min() >= 0.0, f"Min heatmap value {heatmap.min()}"
        assert heatmap.max() <= 1.0, f"Max heatmap value {heatmap.max()}"

    def test_train_heatmap_dtype_float32(self, cityscapes_data_root):
        """Heatmap dtype must be float32."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root=str(cityscapes_data_root),
            split="train",
            heatmap_root=str(cityscapes_data_root / "heatmaps"),
        )
        _, heatmap = dataset[0]
        assert heatmap.dtype == torch.float32, f"Expected float32, got {heatmap.dtype}"

    def test_train_skip_heatmap_returns_none(self, cityscapes_data_root):
        """When skip_heatmap=True, train __getitem__ returns dummy ones_like heatmap."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root=str(cityscapes_data_root),
            split="train",
            skip_heatmap=True,
        )
        _, heatmap = dataset[0]
        # Should still return a valid float32 [0,1] tensor
        assert isinstance(heatmap, torch.Tensor)
        assert heatmap.dtype == torch.float32
        assert heatmap.min() >= 0.0

    # ------------------------------------------------------------------ #
    # Val split — returns (image, label)
    # ------------------------------------------------------------------ #

    def test_val_len(self, cityscapes_data_root):
        """Dataset.__len__ returns correct count for val split."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(root=str(cityscapes_data_root), split="val")
        assert len(dataset) == 1, f"Expected 1, got {len(dataset)}"

    def test_val_getitem_returns_tuple(self, cityscapes_data_root):
        """Val __getitem__ returns (image, label)."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(root=str(cityscapes_data_root), split="val")
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_val_label_shape(self, cityscapes_data_root):
        """Val label is 2D (H, W)."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(root=str(cityscapes_data_root), split="val")
        _, label = dataset[0]
        assert label.dim() == 2

    def test_val_label_dtype_long(self, cityscapes_data_root):
        """Val label dtype is torch.long for cross-entropy."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(root=str(cityscapes_data_root), split="val")
        _, label = dataset[0]
        assert label.dtype == torch.long

    def test_val_label_values_in_range(self, cityscapes_data_root):
        """Val label values in {0..18} ∪ {255}."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(root=str(cityscapes_data_root), split="val")
        _, label = dataset[0]
        valid = set(range(19)) | {255}
        unique_vals = set(label.unique().tolist())
        invalid = unique_vals - valid
        assert not invalid, f"Invalid label values: {invalid}"

    def test_image_shape_and_dtype(self, cityscapes_data_root):
        """Common: image is 3D float32 (C, H, W)."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root=str(cityscapes_data_root),
            split="train",
            heatmap_root=str(cityscapes_data_root / "heatmaps"),
        )
        img, _ = dataset[0]
        assert img.dim() == 3
        c, h, w = img.shape
        assert c == 3
        assert img.dtype == torch.float32
        assert h > 0 and w > 0

    def test_invalid_split_raises(self, cityscapes_data_root):
        """Invalid split raises ValueError."""
        from data.cityscapes_loader import CityscapesDataset

        with pytest.raises(ValueError, match="split"):
            CityscapesDataset(root=str(cityscapes_data_root), split="invalid")

    # ------------------------------------------------------------------ #
    # T010b: Augmentation consistency test (RISK-04)
    # ------------------------------------------------------------------ #

    def test_spatial_transform_identity_image_heatmap(self, cityscapes_data_root):
        """Spatial transforms (flip, resize, crop) applied identically to
        image and heatmap for target augmentation (RISK-04)."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root=str(cityscapes_data_root),
            split="train",
            heatmap_root=str(cityscapes_data_root / "heatmaps"),
            crop_size=(32, 32),
            augment=True,
        )
        img, heatmap = dataset[0]
        # Spatial dims must match after augmentation
        assert (
            img.shape[1:] == heatmap.shape
        ), f"Image spatial {img.shape[1:]} != heatmap {heatmap.shape}"

    def test_no_color_jitter_on_target(self, cityscapes_data_root):
        """Target augmentation must NOT apply color jitter (RISK-04).
        Verify by checking that label/heatmap values remain unchanged
        when pixel values change due to spatial transforms."""
        from data.cityscapes_loader import CityscapesDataset

        # Load with explicit crop but no random flip (seed-controlled)
        dataset = CityscapesDataset(
            root=str(cityscapes_data_root),
            split="train",
            heatmap_root=str(cityscapes_data_root / "heatmaps"),
            crop_size=(32, 32),
            augment=False,  # no augmentation = baseline
        )
        img_no_aug, heatmap_no_aug = dataset[0]

        dataset_aug = CityscapesDataset(
            root=str(cityscapes_data_root),
            split="train",
            heatmap_root=str(cityscapes_data_root / "heatmaps"),
            crop_size=(32, 32),
            augment=True,
        )
        img_aug, heatmap_aug = dataset_aug[0]

        # Heatmap values stay in [0,1] regardless of augmentation
        assert heatmap_aug.min() >= 0.0
        assert heatmap_aug.max() <= 1.0

    def test_heatmap_values_preserved_after_augmentation(self, cityscapes_data_root):
        """Heatmap values remain in [0,1] after spatial augmentation."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root=str(cityscapes_data_root),
            split="train",
            heatmap_root=str(cityscapes_data_root / "heatmaps"),
            crop_size=(32, 32),
            augment=True,
        )
        for i in range(len(dataset)):
            _, heatmap = dataset[i]
            assert heatmap.min() >= 0.0, f"Sample {i}: heatmap min < 0"
            assert heatmap.max() <= 1.0, f"Sample {i}: heatmap max > 1"
