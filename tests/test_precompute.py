"""Tests for heatmap precomputation (RED phase — should fail until module gets a public API).

The precompute module has private helper functions (``_find_images``, ``_heatmap_stem``,
``_heatmap_paths``) but no public API yet.  These tests expect public functions named
``find_images``, ``heatmap_stem``, and ``heatmap_paths`` that do not currently exist.
"""

from pathlib import Path

import numpy as np
import pytest


# ===========================================================================
# T024: CLI discovery and naming convention
# ===========================================================================


class TestPrecomputeCLI:
    """Expected CLI discovery and heatmap naming convention tests.

    These tests verify the expected public API for the precompute module.
    All should fail (red phase) until ``find_images``, ``heatmap_stem``,
    and ``heatmap_paths`` are exposed as public functions.
    """

    def _import(self):
        """Import the non-existent public API — raises ImportError."""
        from precompute.compute_heatmaps import find_images, heatmap_stem, heatmap_paths  # noqa
        return find_images, heatmap_stem, heatmap_paths

    def test_discover_images_recursive(self, tmp_path):
        """find_images discovers ``.png`` files recursively in a dataset tree."""
        find_images, _, _ = self._import()

        train_dir = tmp_path / "leftImg8bit" / "train"
        city_a = train_dir / "aachen"
        city_b = train_dir / "bonn"
        city_a.mkdir(parents=True)
        city_b.mkdir(parents=True)

        (city_a / "aaa_000001_leftImg8bit.png").write_bytes(b"fake")
        (city_a / "aaa_000002_leftImg8bit.png").write_bytes(b"fake")
        (city_b / "bbb_000001_leftImg8bit.png").write_bytes(b"fake")

        images = find_images(str(tmp_path))
        assert len(images) == 3, f"Expected 3 images, got {len(images)}"
        assert all(p.suffix == ".png" for p in images)
        assert images == sorted(images)

    def test_discover_excludes_non_png(self, tmp_path):
        """Only ``.png`` files are returned — other extensions ignored."""
        find_images, _, _ = self._import()

        train_dir = tmp_path / "leftImg8bit" / "train" / "city"
        train_dir.mkdir(parents=True)
        (train_dir / "img_000001_leftImg8bit.png").write_bytes(b"fake")
        (train_dir / "note.txt").write_text("not an image")
        (train_dir / "mask.npy").write_bytes(b"\x00" * 16)

        images = find_images(str(tmp_path))
        assert len(images) == 1, f"Expected 1 .png only, got {len(images)}"
        assert images[0].suffix == ".png"

    def test_no_images_returns_empty_list(self, tmp_path):
        """Empty directory tree returns an empty list, no crash."""
        find_images, _, _ = self._import()

        (tmp_path / "leftImg8bit" / "train" / "empty_city").mkdir(parents=True)
        images = find_images(str(tmp_path))
        assert images == []

    def test_output_naming_edge(self):
        """heatmap_stem strips Cityscapes suffix."""
        _, heatmap_stem, _ = self._import()

        img_path = Path("aaa_000001_leftImg8bit.png")
        stem = heatmap_stem(img_path)
        assert stem == "aaa_000001", f"Expected 'aaa_000001', got '{stem}'"

    def test_output_naming_four_files(self, tmp_path):
        """heatmap_paths returns 4 paths: ``{stem}_satg_{edge,var,ent,corn}.npy``."""
        _, _, heatmap_paths = self._import()

        img_path = tmp_path / "aaa_000001_leftImg8bit.png"
        img_path.write_bytes(b"fake")

        paths = heatmap_paths(img_path)
        assert len(paths) == 4, f"Expected 4 paths, got {len(paths)}"

        names = [p.name for p in paths]
        expected = [
            "aaa_000001_satg_edge.npy",
            "aaa_000001_satg_var.npy",
            "aaa_000001_satg_ent.npy",
            "aaa_000001_satg_corn.npy",
        ]
        for ename in expected:
            assert ename in names, f"Missing expected file: {ename}"

    def test_heatmap_paths_same_parent(self, tmp_path):
        """heatmap_paths preserves the parent directory."""
        _, _, heatmap_paths = self._import()

        img_path = tmp_path / "leftImg8bit" / "train" / "city" / "img_000001_leftImg8bit.png"
        img_path.parent.mkdir(parents=True)
        img_path.write_bytes(b"fake")

        paths = heatmap_paths(img_path)
        for p in paths:
            assert p.parent == img_path.parent

    def test_output_dtype_float32(self, tmp_path):
        """Generated ``.npy`` component files have dtype float32."""
        _, _, heatmap_paths = self._import()

        img_path = tmp_path / "img_000001_leftImg8bit.png"
        heatmap_data = np.random.rand(64, 64).astype(np.float32)

        out_paths = heatmap_paths(img_path)
        for p in out_paths:
            np.save(str(p), heatmap_data)

        for p in out_paths:
            loaded = np.load(str(p))
            assert loaded.dtype == np.float32, (
                f"{p.name}: expected float32, got {loaded.dtype}"
            )

    def test_output_shape_matches_input(self, tmp_path):
        """Each component file shape matches the input image ``(H, W)``."""
        _, _, heatmap_paths = self._import()

        img_path = tmp_path / "img_000001_leftImg8bit.png"
        H, W = 128, 256
        heatmap_data = np.random.rand(H, W).astype(np.float32)

        out_paths = heatmap_paths(img_path)
        for p in out_paths:
            np.save(str(p), heatmap_data)

        for p in out_paths:
            loaded = np.load(str(p))
            assert loaded.shape == (H, W), (
                f"{p.name}: expected ({H}, {W}), got {loaded.shape}"
            )

    def test_all_four_components_saved(self, tmp_path):
        """After processing one image, exactly 4 ``.npy`` files exist."""
        _, _, heatmap_paths = self._import()

        img_path = tmp_path / "img_000001_leftImg8bit.png"
        data = np.random.rand(32, 32).astype(np.float32)

        out_paths = heatmap_paths(img_path)
        for p in out_paths:
            np.save(str(p), data)

        npy_files = sorted(tmp_path.glob("*.npy"))
        assert len(npy_files) == 4, f"Expected 4 .npy files, got {len(npy_files)}"
        assert all(p.suffix == ".npy" for p in npy_files)

    def test_non_cityscapes_fallback_stem(self):
        """heatmap_stem on a non-Cityscapes filename strips just the extension."""
        _, heatmap_stem, _ = self._import()

        img_path = Path("some_random_image.png")
        stem = heatmap_stem(img_path)
        assert stem == "some_random_image", f"Expected 'some_random_image', got '{stem}'"

    def test_find_images_handles_alternate_layout(self, tmp_path):
        """find_images also handles ``leftImg8bit_trainvaltest/train/`` layout."""
        find_images, _, _ = self._import()

        train_dir = tmp_path / "leftImg8bit_trainvaltest" / "train" / "city"
        train_dir.mkdir(parents=True)
        (train_dir / "img_000001_leftImg8bit.png").write_bytes(b"fake")

        images = find_images(str(tmp_path))
        assert len(images) == 1

    def test_no_npy_for_non_existent_image(self, tmp_path):
        """heatmap_paths on a non-existent ``Path`` still returns 4 paths."""
        _, _, heatmap_paths = self._import()

        img_path = tmp_path / "nonexistent_leftImg8bit.png"
        paths = heatmap_paths(img_path)
        assert len(paths) == 4
        assert not any(p.exists() for p in paths)
