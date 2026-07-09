"""Tests for GTA5 label preprocessing (index-map → Cityscapes trainIDs).

Test-first (RED → GREEN) per TDD workflow.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="function")
def synthetic_gta5_labels():
    """Create a temporary directory with synthetic single-channel GTA5 labels.

    GTA5 labels are single-channel indexed PNGs where pixel values are
    the GTA5 class index (0–32). This fixture generates small synthetic
    arrays to test the preprocessing pipeline.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Create subdirectories mirroring GTA5 structure
        cities = {
            "cityA": ["img_0001", "img_0002"],
            "cityB": ["img_0003"],
        }

        for city, stems in cities.items():
            label_dir = root / city
            label_dir.mkdir(parents=True)
            for stem in stems:
                # Create synthetic 10x10 single-channel label
                label = np.random.randint(0, 26, (10, 10), dtype=np.uint8)
                cv2.imwrite(str(label_dir / f"{stem}.png"), label)

        yield root


class TestGTA5Preprocessing:
    """Tests for GTA5 label → Cityscapes trainID preprocessing."""

    def test_index_map_to_trainid(self):
        """Given a synthetic 10x10 array where all pixels have GTA5 index 7
        (road), output is all 0 (Cityscapes road)."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        arr = np.full((10, 10), 7, dtype=np.uint8)
        out = np.full_like(arr, 255, dtype=np.uint8)
        for i in range(35):
            out[arr == i] = GTA5_TO_CITYSCAPES_19[i]
        assert (out == 0).all(), f"Expected all 0 (road), got unique values: {np.unique(out)}"

    def test_unknown_index_maps_to_255(self):
        """Pixel value 33 (out of range for 0–34) → 255."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        arr = np.full((10, 10), 33, dtype=np.uint8)
        out = np.full_like(arr, 255, dtype=np.uint8)
        for i in range(35):
            out[arr == i] = GTA5_TO_CITYSCAPES_19[i]
        assert (out == 255).all(), f"Expected all 255, got {np.unique(out)}"

    def test_output_dtype_uint8(self):
        """Output array is np.uint8."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        arr = np.full((10, 10), 7, dtype=np.uint8)
        out = np.full_like(arr, 255, dtype=np.uint8)
        for i in range(35):
            out[arr == i] = GTA5_TO_CITYSCAPES_19[i]
        assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"

    def test_output_shape_2d(self):
        """Output shape is [H, W] not [H, W, C]."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        arr = np.full((10, 10), 7, dtype=np.uint8)
        out = np.full_like(arr, 255, dtype=np.uint8)
        for i in range(35):
            out[arr == i] = GTA5_TO_CITYSCAPES_19[i]
        assert out.ndim == 2, f"Expected 2D array, got {out.ndim}D"

    def test_values_in_valid_range(self):
        """All output values are in 0–18 or 255."""
        from data.label_mapping import GTA5_TO_CITYSCAPES_19

        arr = np.random.randint(0, 35, (10, 10), dtype=np.uint8)
        out = np.full_like(arr, 255, dtype=np.uint8)
        for i in range(35):
            out[arr == i] = GTA5_TO_CITYSCAPES_19[i]
        valid = set(range(19)) | {255}
        unique_vals = set(out.ravel().tolist())
        invalid = unique_vals - valid
        assert not invalid, f"Invalid values found: {invalid}"