#!/usr/bin/env python3
"""Offline structural heatmap precomputation for SATG.

Usage::

    python precompute/compute_heatmaps.py --data_root /path/to/cityscapes \\
                                          --num_workers 8

Finds all ``*.png`` images under ``<data_root>/leftImg8bit/train/``
recursively.  For each image:

1. Loads via ``cv2.imread`` (RGB).
2. Computes structural complexity heatmap via ``StructuralPrior.compute()``.
3. Saves the heatmap as ``{stem}_satg_heatmap.npy`` in the *same directory*
   as the source image.

After completion, loads 20 random heatmaps and prints summary statistics.
"""

import argparse
import multiprocessing
import os
from functools import partial
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from satg.structural_prior import StructuralPrior


def _default_cfg() -> OmegaConf:
    """Return a minimal OmegaConf with default structural_prior parameters."""
    return OmegaConf.create(
        {
            "structural_prior": {
                "edge_low_threshold": 50,
                "edge_high_threshold": 150,
                "gaussian_sigma": 2.0,
                "edge_kernel_size": 15,
                "variance_kernel_size": 15,
                "edge_weight": 0.5,
                "variance_weight": 0.5,
            }
        }
    )


def _find_images(data_root: str) -> list:
    """Recursively find all ``*.png`` files under ``leftImg8bit/train/``.

    Args:
        data_root: Root directory of the Cityscapes dataset.

    Returns:
        Sorted list of absolute paths to ``.png`` files.
    """
    train_dir = Path(data_root) / "leftImg8bit" / "train"
    pattern = str(train_dir / "**" / "*.png")
    return sorted(glob(pattern, recursive=True))


def _process_one(image_path: str, prior: StructuralPrior) -> str:
    """Load a single image, compute heatmap, and save as ``.npy``.

    Args:
        image_path: Absolute path to the source ``.png`` image.
        prior: Initialized ``StructuralPrior`` instance.

    Returns:
        The image path (for progress tracking).
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = prior.compute(img_rgb)

    stem = Path(image_path).stem  # e.g. "frankfurt_000000_000294_leftImg8bit"
    out_path = Path(image_path).parent / f"{stem}_satg_heatmap.npy"
    np.save(str(out_path), heatmap)

    return image_path


def _print_statistics(data_root: str, all_paths: list) -> None:
    """Load 20 random heatmaps and print min/max/mean/std.

    Args:
        data_root: Root directory (used to construct heatmap paths from image paths).
        all_paths: List of image paths used for heatmap name resolution.
    """
    if len(all_paths) == 0:
        print("No heatmaps found. Skipping statistics.")
        return

    rng = np.random.RandomState(42)
    sample_paths = rng.choice(all_paths, size=min(20, len(all_paths)), replace=False)

    stats = {"min": [], "max": [], "mean": [], "std": []}
    for img_path_str in sample_paths:
        img_path = Path(img_path_str)
        heatmap_path = img_path.parent / f"{img_path.stem}_satg_heatmap.npy"
        if not heatmap_path.exists():
            continue
        hm = np.load(str(heatmap_path))
        stats["min"].append(float(hm.min()))
        stats["max"].append(float(hm.max()))
        stats["mean"].append(float(hm.mean()))
        stats["std"].append(float(hm.std()))

    if len(stats["min"]) == 0:
        print("No heatmap files found for statistics.")
        return

    print("\n=== Heatmap Statistics (20 random samples) ===")
    print(f"  min:  {np.mean(stats['min']):.4f} ± {np.std(stats['min']):.4f}")
    print(f"  max:  {np.mean(stats['max']):.4f} ± {np.std(stats['max']):.4f}")
    print(f"  mean: {np.mean(stats['mean']):.4f} ± {np.std(stats['mean']):.4f}")
    print(f"  std:  {np.mean(stats['std']):.4f} ± {np.std(stats['std']):.4f}")
    print(f"  (based on {len(stats['min'])} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute SATG structural heatmaps for Cityscapes training images."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to Cityscapes dataset root (contains leftImg8bit/).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8).",
    )
    args = parser.parse_args()

    # Discover images
    print(f"Scanning for images under {args.data_root}/leftImg8bit/train/ ...")
    all_images = _find_images(args.data_root)
    print(f"Found {len(all_images)} PNG images.")

    if len(all_images) == 0:
        print("No images found. Exiting.")
        return

    # Build prior (share the same config across workers via initialisation)
    cfg = _default_cfg()
    prior = StructuralPrior(cfg)

    # Run multiprocessing
    worker = partial(_process_one, prior=prior)
    print(f"Processing with {args.num_workers} workers ...")
    with multiprocessing.Pool(args.num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(worker, all_images),
                total=len(all_images),
                desc="Heatmaps",
                unit="img",
            )
        )

    print("Done. All heatmaps saved.")
    _print_statistics(args.data_root, all_images)


if __name__ == "__main__":
    main()
