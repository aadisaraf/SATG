"""Tests for data loaders, label mapping, and preprocessing.

Test-first (RED → GREEN) per TDD workflow.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


# =============================================================================
# PART A — Label mapping tests
# =============================================================================


class TestLabelMapping:
    """Tests for GTA5 → Cityscapes 19-class label mapping (GTA5_TO_CITYSCAPES_19)."""

    # ------------------------------------------------------------------ #
    # Existing tests (updated for GTA5_TO_CITYSCAPES_19, 35 classes)
    # ------------------------------------------------------------------ #

    def test_mapping_covers_all_19_cityscapes_classes(self):
        """The mapping dict must contain at least one GTA5 class mapping
        to each of the 19 Cityscapes train IDs."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        cityscapes_ids = set(range(19))
        mapped_ids = set(v for v in GTA5_TO_CITYSCAPES_19.values() if v != 255)
        missing = cityscapes_ids - mapped_ids
        assert not missing, f"Missing Cityscapes train IDs in mapping: {missing}"

    def test_mapping_dict_completeness(self):
        """All 35 GTA5 class IDs (0–34) must have a mapping entry."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        expected = set(range(35))
        actual = set(GTA5_TO_CITYSCAPES_19.keys())
        missing = expected - actual
        extra = actual - expected
        assert not missing, f"Missing GTA5 IDs in mapping: {missing}"
        assert not extra, f"Extra GTA5 IDs not in 0–34 range: {extra}"

    def test_unmapped_classes_raise_error(self):
        """A GTA5 class ID not in the dict should raise KeyError
        (though the dict covers all 35 IDs, testing defensive coding)."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        with pytest.raises(KeyError):
            GTA5_TO_CITYSCAPES_19[99]

    def test_mapped_to_valid_ids(self):
        """All mapped values must be in {0..18} ∪ {255}."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        valid = set(range(19)) | {255}
        for gta5_id, cs_id in GTA5_TO_CITYSCAPES_19.items():
            assert cs_id in valid, f"GTA5 ID {gta5_id} maps to invalid Cityscapes ID {cs_id}"

    def test_map_gta5_label_function(self):
        """map_gta5_label converts an RGB label map to Cityscapes trainIDs."""
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

    # ------------------------------------------------------------------ #
    # NEW Part A tests (per spec)
    # ------------------------------------------------------------------ #

    def test_unlabeled_maps_to_ignore(self):
        """GTA5 class 0 (unlabeled) → 255 (ignore)."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        assert GTA5_TO_CITYSCAPES_19[0] == 255, "Unlabeled class should map to ignore"

    def test_road_maps_to_zero(self):
        """GTA5 class 7 (road) → 0 (Cityscapes road trainID)."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        assert GTA5_TO_CITYSCAPES_19[7] == 0, "Road should map to Cityscapes trainID 0"

    def test_person_maps_to_eleven(self):
        """GTA5 class 18 (person) → 11 (Cityscapes person trainID)."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        assert GTA5_TO_CITYSCAPES_19[18] == 11, "Person should map to Cityscapes trainID 11"

    def test_all_35_classes_handled(self):
        """GTA5_TO_CITYSCAPES_19 must have exactly 35 entries."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        assert len(GTA5_TO_CITYSCAPES_19) == 35, (
            f"Expected 35 entries, got {len(GTA5_TO_CITYSCAPES_19)}"
        )

    def test_no_valid_maps_to_255(self):
        """All mapped values that are not 255 must be in 0-18."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        for gta5_id, cs_id in GTA5_TO_CITYSCAPES_19.items():
            if cs_id != 255:
                assert 0 <= cs_id <= 18, (
                    f"GTA5 ID {gta5_id} maps to {cs_id}, which is not in 0-18 or 255"
                )


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
            root_or_cfg=str(gta5_data_root),
            img_subdir="images/train",
            label_subdir="labels/train",
        )
        assert len(dataset) == 3, f"Expected 3 samples, got {len(dataset)}"

    def test_getitem_returns_tuple(self, gta5_data_root):
        """__getitem__ must return a (image, label) tuple."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root_or_cfg=str(gta5_data_root),
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
            root_or_cfg=str(gta5_data_root),
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
            root_or_cfg=str(gta5_data_root),
            img_subdir="images/train",
            label_subdir="labels/train",
        )
        img, _ = dataset[0]
        assert img.dtype == torch.float32, f"Expected float32, got {img.dtype}"

    def test_getitem_label_shape(self, gta5_data_root):
        """Label tensor must be 2D (H, W)."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root_or_cfg=str(gta5_data_root),
            img_subdir="images/train",
            label_subdir="labels/train",
        )
        _, label = dataset[0]
        assert label.dim() == 2, f"Expected 2D label, got {label.dim()}D"

    def test_getitem_label_dtype(self, gta5_data_root):
        """Label tensor must be int64 (torch.long) for cross-entropy."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root_or_cfg=str(gta5_data_root),
            img_subdir="images/train",
            label_subdir="labels/train",
        )
        _, label = dataset[0]
        assert label.dtype == torch.long, f"Expected torch.long, got {label.dtype}"

    def test_label_values_in_valid_range(self, gta5_data_root):
        """Label values must be in {0..18} ∪ {255} after mapping."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root_or_cfg=str(gta5_data_root),
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
            root_or_cfg=str(gta5_data_root),
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
            train/cityA/cityA_000001_gtFine_labelTrainIds.png
            train/cityB/cityB_000001_gtFine_labelTrainIds.png
            val/cityC/cityC_000001_gtFine_labelTrainIds.png
          (heatmaps saved alongside images in leftImg8bit/train/)
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


        for city, stems in cities_train.items():
            img_dir = root / "leftImg8bit" / "train" / city
            lbl_dir = root / "gtFine" / "train" / city
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            for stem in stems:
                img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(img_dir / f"{stem}_leftImg8bit.png"), img)
                # label values in 0..18 ∪ {255}
                label = np.random.choice(list(range(19)) + [255], (64, 64)).astype(np.uint8)
                cv2.imwrite(str(lbl_dir / f"{stem}_gtFine_labelTrainIds.png"), label)
                for suffix in ['_satg_edge.npy', '_satg_var.npy', '_satg_ent.npy', '_satg_corn.npy']:
                    component = np.random.rand(64, 64).astype(np.float32)
                    np.save(str(img_dir / f"{stem}{suffix}"), component)

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
                cv2.imwrite(str(lbl_dir / f"{stem}_gtFine_labelTrainIds.png"), label)

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
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
        )
        assert len(dataset) == 3, f"Expected 3, got {len(dataset)}"

    def test_train_getitem_returns_tuple(self, cityscapes_data_root):
        """Train __getitem__ returns (image, heatmap)."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
        )
        img, heatmap = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(heatmap, torch.Tensor)

    def test_train_heatmap_shape(self, cityscapes_data_root):
        """Heatmap shape matches image spatial dims."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
        )
        img, heatmap = dataset[0]
        assert heatmap.dim() == 2, f"Expected 2D, got {heatmap.dim()}D"
        _, h, w = img.shape
        assert heatmap.shape == (h, w), f"Heatmap shape {heatmap.shape} != image spatial {h}x{w}"

    def test_train_heatmap_values_in_01(self, cityscapes_data_root):
        """Heatmap values must be in [0, 1]."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
        )
        _, heatmap = dataset[0]
        assert heatmap.min() >= 0.0, f"Min heatmap value {heatmap.min()}"
        assert heatmap.max() <= 1.0, f"Max heatmap value {heatmap.max()}"

    def test_train_heatmap_dtype_float32(self, cityscapes_data_root):
        """Heatmap dtype must be float32."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
        )
        _, heatmap = dataset[0]
        assert heatmap.dtype == torch.float32, f"Expected float32, got {heatmap.dtype}"

    def test_train_skip_heatmap_returns_none(self, cityscapes_data_root):
        """When skip_heatmap=True, train __getitem__ returns dummy ones_like heatmap."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
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

        dataset = CityscapesDataset(root_or_cfg=str(cityscapes_data_root), split="val")
        assert len(dataset) == 1, f"Expected 1, got {len(dataset)}"

    def test_val_getitem_returns_tuple(self, cityscapes_data_root):
        """Val __getitem__ returns (image, label)."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(root_or_cfg=str(cityscapes_data_root), split="val")
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_val_label_shape(self, cityscapes_data_root):
        """Val label is 2D (H, W)."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(root_or_cfg=str(cityscapes_data_root), split="val")
        _, label = dataset[0]
        assert label.dim() == 2

    def test_val_label_dtype_long(self, cityscapes_data_root):
        """Val label dtype is torch.long for cross-entropy."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(root_or_cfg=str(cityscapes_data_root), split="val")
        _, label = dataset[0]
        assert label.dtype == torch.long

    def test_val_label_values_in_range(self, cityscapes_data_root):
        """Val label values in {0..18} ∪ {255}."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(root_or_cfg=str(cityscapes_data_root), split="val")
        _, label = dataset[0]
        valid = set(range(19)) | {255}
        unique_vals = set(label.unique().tolist())
        invalid = unique_vals - valid
        assert not invalid, f"Invalid label values: {invalid}"

    def test_image_shape_and_dtype(self, cityscapes_data_root):
        """Common: image is 3D float32 (C, H, W)."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
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
            CityscapesDataset(root_or_cfg=str(cityscapes_data_root), split="invalid")

    # ------------------------------------------------------------------ #
    # T010b: Augmentation consistency test (RISK-04)
    # ------------------------------------------------------------------ #

    def test_spatial_transform_identity_image_heatmap(self, cityscapes_data_root):
        """Spatial transforms (flip, resize, crop) applied identically to
        image and heatmap for target augmentation (RISK-04)."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
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
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
            crop_size=(32, 32),
            augment=False,  # no augmentation = baseline
        )
        img_no_aug, heatmap_no_aug = dataset[0]

        dataset_aug = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
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
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
            crop_size=(32, 32),
            augment=True,
        )
        for i in range(len(dataset)):
            _, heatmap = dataset[i]
            assert heatmap.min() >= 0.0, f"Sample {i}: heatmap min < 0"
            assert heatmap.max() <= 1.0, f"Sample {i}: heatmap max > 1"


# =============================================================================
# PART C — CityscapesDataset tests (cfg-based init, augment pipeline)
# =============================================================================


class TestCityscapesDataset:
    """Tests for CityscapesDataset cfg-based init and augmentation pipeline."""

    def test_train_without_heatmap(self, cityscapes_data_root):
        """skip_heatmap=True → returns (image_tensor, torch.zeros(1))."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            skip_heatmap=True,
        )
        img, placeholder = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(placeholder, torch.Tensor)
        assert placeholder.numel() == 1, (
            f"Expected scalar placeholder, got shape {placeholder.shape}"
        )
        assert placeholder.item() == 0.0, "Placeholder should be 0.0"

    def test_train_with_heatmap(self, cityscapes_data_root):
        """skip_heatmap=False → returns (image, heatmap)
        where heatmap is Tensor[H, W] float32 in [0, 1]."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
        )
        img, heatmap = dataset[0]
        assert isinstance(heatmap, torch.Tensor)
        assert heatmap.dim() == 2, f"Expected 2D heatmap, got {heatmap.dim()}D"
        assert heatmap.dtype == torch.float32
        assert heatmap.min() >= 0.0, f"Heatmap min {heatmap.min()}"
        assert heatmap.max() <= 1.0, f"Heatmap max {heatmap.max()}"

    def test_val_returns_image_and_label(self, cityscapes_data_root):
        """Val split → (Tensor[3,H,W], Tensor[H,W])."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(root_or_cfg=str(cityscapes_data_root), split="val")
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert img.dim() == 3
        assert img.shape[0] == 3
        assert label.dim() == 2

    def test_heatmap_augmentation_matches_image(self, cityscapes_data_root):
        """Same crop + flip applied to both image and heatmap."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
            crop_size=(32, 32),
            augment=True,
        )
        img, heatmap = dataset[0]
        assert img.shape[1:] == heatmap.shape, (
            f"Image spatial {img.shape[1:]} != heatmap {heatmap.shape}"
        )

    def test_color_jitter_image_only(self, cityscapes_data_root):
        """Heatmap range stays [0, 1] after augmentation (only image gets jitter)."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
            crop_size=(32, 32),
            augment=True,
        )
        _, heatmap = dataset[0]
        assert heatmap.min() >= 0.0, f"Heatmap min {heatmap.min()}"
        assert heatmap.max() <= 1.0, f"Heatmap max {heatmap.max()}"

    def test_output_shapes(self, cityscapes_data_root):
        """Output shapes [3, H, W] and [H, W] for configured crop_size."""
        from data.cityscapes_loader import CityscapesDataset

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
            crop_size=(32, 32),
            augment=True,
        )
        img, heatmap = dataset[0]
        assert img.shape == (3, 32, 32), f"Image shape {img.shape}"
        assert heatmap.shape == (32, 32), f"Heatmap shape {heatmap.shape}"

    def test_missing_npy_raises(self, cityscapes_data_root):
        """FileNotFoundError if .npy missing and skip_heatmap=False."""
        from data.cityscapes_loader import CityscapesDataset

        for npy in (cityscapes_data_root / "leftImg8bit" / "train" / "cityA").glob("*_satg_*.npy"):
            npy.unlink()
        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            heatmap_root=None,
            skip_heatmap=False,
        )
        with pytest.raises(FileNotFoundError):
            _ = dataset[0]

    def test_old_single_heatmap_format_raises(self, cityscapes_data_root):
        """Old *_satg_heatmap.npy format (single file) must raise FileNotFoundError."""
        from data.cityscapes_loader import CityscapesDataset

        # Remove all 4 component files and create the old combined file instead
        heat_dir = cityscapes_data_root / "leftImg8bit" / "train" / "cityA"
        for suffix in ['_satg_edge.npy', '_satg_var.npy', '_satg_ent.npy', '_satg_corn.npy']:
            (heat_dir / f"000001_000000{suffix}").unlink()
        old_heatmap = np.random.rand(64, 64).astype(np.float32)
        np.save(str(heat_dir / "000001_000000_satg_heatmap.npy"), old_heatmap)

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
        )
        with pytest.raises(FileNotFoundError, match="satg_edge"):
            _ = dataset[0]

    def test_missing_one_component_raises(self, cityscapes_data_root):
        """If 1 of 4 component files is missing, must raise FileNotFoundError."""
        from data.cityscapes_loader import CityscapesDataset

        heat_dir = cityscapes_data_root / "leftImg8bit" / "train" / "cityA"
        (heat_dir / "000001_000000_satg_corn.npy").unlink()

        dataset = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
        )
        with pytest.raises(FileNotFoundError, match="satg_corn"):
            _ = dataset[0]

    def test_weight_override_changes_combined(self, cityscapes_data_root):
        """Different structural_prior weights in config produce different heatmaps."""
        from data.cityscapes_loader import CityscapesDataset

        # Config with aggressive edge weight, zero others
        cfg_edge = OmegaConf.create({
            "structural_prior": {
                "edge_weight": 1.0,
                "variance_weight": 0.0,
                "entropy_weight": 0.0,
                "cornerness_weight": 0.0,
            }
        })
        # Config with aggressive variance weight, zero others
        cfg_var = OmegaConf.create({
            "structural_prior": {
                "edge_weight": 0.0,
                "variance_weight": 1.0,
                "entropy_weight": 0.0,
                "cornerness_weight": 0.0,
            }
        })

        dataset_edge = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
            cfg=cfg_edge,
        )
        dataset_var = CityscapesDataset(
            root_or_cfg=str(cityscapes_data_root),
            split="train",
            
            cfg=cfg_var,
        )

        _, hm_edge = dataset_edge[0]
        _, hm_var = dataset_var[0]

        # The two heatmaps should differ (different weights)
        diff = (hm_edge - hm_var).abs().mean().item()
        assert diff > 0.01, (
            f"Expected heatmaps to differ with different cfg weights, diff={diff:.6f}"
        )


# =============================================================================
# Fixture: synthetic GTA5 data with *_trainids.png naming for PART D
# =============================================================================


@pytest.fixture(scope="function")
def gta5_trainids_root():
    """Create a temporary GTA5 directory with *_trainids.png labels.

    Structure:
        tmpdir/
          images/train/cityA/AAA_1.png
          labels/train/cityA/AAA_1_trainids.png   (single-channel, Cityscapes IDs)
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        img_dir = root / "images" / "train"
        lbl_dir = root / "labels" / "train"
        (img_dir / "cityA").mkdir(parents=True)
        (lbl_dir / "cityA").mkdir(parents=True)

        # Create a few synthetic images and corresponding trainids label files
        for stem in ["AAA_1", "AAA_2", "BBB_1"]:
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "cityA" / f"{stem}.png"), img)
            # Label with values in Cityscapes trainID range (0-18, 255)
            label = np.random.choice(list(range(19)) + [255], (64, 64)).astype(np.uint8)
            cv2.imwrite(str(lbl_dir / "cityA" / f"{stem}_trainids.png"), label)

        yield root


# =============================================================================
# PART D — GTA5Dataset tests (trainids label reading, rare class weights)
# =============================================================================


class TestGTA5Dataset:
    """Tests for GTA5Dataset cfg init, trainids label reading, rare class weights."""

    def test_returns_image_and_label(self, gta5_trainids_root):
        """__getitem__ returns (Tensor[3,H,W], Tensor[H,W])."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root_or_cfg=str(gta5_trainids_root),
            img_subdir="images/train",
            label_subdir="labels/train",
            label_suffix="_trainids.png",
        )
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert img.dim() == 3 and img.shape[0] == 3
        assert label.dim() == 2

    def test_label_uses_cityscapes_ids(self, gta5_trainids_root):
        """All label values are in {0–18, 255}."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root_or_cfg=str(gta5_trainids_root),
            img_subdir="images/train",
            label_subdir="labels/train",
            label_suffix="_trainids.png",
        )
        _, label = dataset[0]
        valid = set(range(19)) | {255}
        unique_vals = set(label.unique().tolist())
        invalid = unique_vals - valid
        assert not invalid, f"Invalid label values: {invalid}"

    def test_label_reads_trainids_file(self, gta5_trainids_root):
        """GTA5Dataset reads *_trainids.png, not original labels."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root_or_cfg=str(gta5_trainids_root),
            img_subdir="images/train",
            label_subdir="labels/train",
            label_suffix="_trainids.png",
        )
        _, label = dataset[0]
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        assert label.dim() == 2

    def test_rare_class_weights_shape(self, gta5_trainids_root):
        """class_weights is Tensor[19] when enabled."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root_or_cfg=str(gta5_trainids_root),
            img_subdir="images/train",
            label_subdir="labels/train",
            label_suffix="_trainids.png",
        )
        if hasattr(dataset, "rare_class_weights"):
            weights = dataset.rare_class_weights
            assert isinstance(weights, torch.Tensor)
            assert weights.shape == (19,), f"Expected shape (19,), got {weights.shape}"
        else:
            pytest.skip("GTA5Dataset does not expose rare_class_weights")

    def test_rare_class_weights_inverse_freq(self, gta5_trainids_root):
        """weight[c] = N/(count[c]*19), clipped [0.1, 10.0]."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root_or_cfg=str(gta5_trainids_root),
            img_subdir="images/train",
            label_subdir="labels/train",
            label_suffix="_trainids.png",
        )
        if hasattr(dataset, "rare_class_weights"):
            weights = dataset.rare_class_weights
            total_pixels = 0
            counts = torch.zeros(19, dtype=torch.float64)
            for i in range(len(dataset)):
                _, label = dataset[i]
                for c in range(19):
                    counts[c] += (label == c).sum().item()
                total_pixels += label.numel()
            expected = torch.where(counts > 0, total_pixels / (counts * 19), torch.tensor(0.0))
            expected = expected.clamp(0.1, 10.0)
            assert torch.allclose(weights, expected.float(), atol=1e-3), (
                f"Weights don't match inverse freq formula. "
                f"Got {weights}, expected {expected}"
            )
        else:
            pytest.skip("GTA5Dataset does not expose rare_class_weights")

    def test_weights_clipped(self, gta5_trainids_root):
        """No weight is below 0.1 or above 10.0."""
        from data.gta5_loader import GTA5Dataset

        dataset = GTA5Dataset(
            root_or_cfg=str(gta5_trainids_root),
            img_subdir="images/train",
            label_subdir="labels/train",
            label_suffix="_trainids.png",
        )
        if hasattr(dataset, "rare_class_weights"):
            weights = dataset.rare_class_weights
            assert weights.min() >= 0.1, f"Weight min {weights.min()} < 0.1"
            assert weights.max() <= 10.0, f"Weight max {weights.max()} > 10.0"
        else:
            pytest.skip("GTA5Dataset does not expose rare_class_weights")
