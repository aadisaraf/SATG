#!/usr/bin/env python3
"""Offline structural heatmap precomputation for SATG.

Usage::

    python -m precompute.compute_heatmaps --data_root /path/to/cityscapes \\
                                          --num_workers 8

Finds all ``*.png`` images under ``<data_root>/leftImg8bit/train/``
recursively.  For each image:

1. Loads via ``cv2.imread`` (RGB).
2. Computes structural complexity heatmap components via
   ``StructuralPrior.compute()``.
3. Saves four component files ``{stem}_satg_{edge,var,ent,corn}.npy``
   in the *same directory* as the source image.

After completion, loads 20 random component sets and prints per-component
summary statistics plus an optional verification pass.
"""

import argparse
import math
import multiprocessing
import os
import shutil
import sys
from functools import partial
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Ensure project root is on sys.path (works when invoked both as `-m` and directly)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from satg.structural_prior import StructuralPrior


def _default_cfg() -> OmegaConf:
    """Return a minimal OmegaConf with default structural_prior parameters."""
    return OmegaConf.create(
        {
            "structural_prior": {
                "norm_percentile": 95.0,
                "gaussian_sigma": 2.0,
                "edge_kernel_size": 15,
                "variance_kernel_size": 15,
                "edge_weight": 0.25,
                "variance_weight": 0.25,
                "entropy_kernel_radius": 7,
                "entropy_n_bins": 32,
                "entropy_weight": 0.25,
                "cornerness_kernel_size": 15,
                "cornerness_sigma": 2.0,
                "cornerness_weight": 0.25,
            }
        }
    )


def _find_images(data_root: str) -> list:
    """Recursively find all ``*.png`` training images under the dataset.

    Accepts:
    - A city subdirectory directly (e.g. ``/path/to/train/aachen``)
    - Cityscapes root with ``leftImg8bit/train/`` or
      ``leftImg8bit_trainvaltest/train/``.

    Args:
        data_root: Path to a city subdirectory or Cityscapes root.

    Returns:
        Sorted list of absolute paths to ``.png`` files.
    """
    candidate = Path(data_root)

    # Check standard Cityscapes layouts first — handles the case where
    # `data_root` is the Cityscapes root and avoids picking up val/test
    # or gtFine images from sibling directories.
    for img_root in ("leftImg8bit", "leftImg8bit_trainvaltest"):
        train_dir = candidate / img_root / "train"
        if train_dir.is_dir():
            images = sorted(train_dir.rglob("*.png"))
            if images:
                return images

    # Fallback: point directly at a city subdirectory that has PNGs.
    if list(candidate.rglob("*.png")):
        return sorted(candidate.rglob("*.png"))

    print(
        f"Error: no images found at {candidate}.\n"
        f"  Tried: {candidate}/{{leftImg8bit,leftImg8bit_trainvaltest}}/train/\n"
        f"  Hint: Point --data_root to your Cityscapes directory "
        f"(e.g. /Users/aadisaraf/Documents/research/SATG/data/cityscapes)."
    )
    return []


_CITYSCAPES_IMG_SUFFIX = "_leftImg8bit.png"

_HEATMAP_SUFFIXES = ['_satg_edge.npy', '_satg_var.npy', '_satg_ent.npy', '_satg_corn.npy']


def _heatmap_stem(img_path: Path) -> str:
    """Extract the base stem for heatmap files from a Cityscapes image path.

    Cityscapes image filenames are ``{id}_leftImg8bit.png``.  The heatmap
    stem strips ``_leftImg8bit.png`` so component files are named
    ``{id}_satg_edge.npy`` etc.
    """
    name = img_path.name
    if name.endswith(_CITYSCAPES_IMG_SUFFIX):
        return name[: -len(_CITYSCAPES_IMG_SUFFIX)]
    # Fallback for non-standard filenames: strip the extension
    return name[: -len(img_path.suffix)]


def _heatmap_paths(img_path: Path) -> list[Path]:
    """Return the 4 expected ``.npy`` paths for a given Cityscapes image."""
    stem = _heatmap_stem(img_path)
    return [img_path.parent / f"{stem}{suf}" for suf in _HEATMAP_SUFFIXES]


def _has_all_components(img_path: Path) -> bool:
    """Check whether all 4 heatmap component files already exist."""
    return all(p.exists() for p in _heatmap_paths(img_path))


def _cleanup_incomplete(img_path: Path) -> None:
    """Remove any partial heatmap files (not all 4 present)."""
    paths = _heatmap_paths(img_path)
    if not all(p.exists() for p in paths):
        for p in paths:
            p.unlink(missing_ok=True)


def _process_one(image_path: str, prior: StructuralPrior) -> str:
    """Load a single image, compute heatmap components, and save as 4 ``.npy``.

    Args:
        image_path: Absolute path to the source ``.png`` image.
        prior: Initialized ``StructuralPrior`` instance.

    Returns:
        The image path (for progress tracking).
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  WARNING: failed to load {image_path}, skipping")
        return image_path
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    components = prior.compute(img_rgb)  # returns dict with 'edge', 'var', 'ent', 'corn'

    img_path = Path(image_path)
    # Write each component; if any save fails, clean up so resume doesn't
    # see a partial set.
    out_paths = _heatmap_paths(img_path)
    try:
        np.save(str(out_paths[0]), components['edge'])
        np.save(str(out_paths[1]), components['var'])
        np.save(str(out_paths[2]), components['ent'])
        np.save(str(out_paths[3]), components['corn'])
    except OSError:
        for p in out_paths:
            p.unlink(missing_ok=True)
        raise

    return image_path


def _print_statistics(data_root: str, all_paths: list) -> None:
    """Load 20 random component sets and print per-component statistics.

    Args:
        data_root: Root directory (used to construct heatmap paths from image paths).
        all_paths: List of image paths used for heatmap name resolution.
    """
    if len(all_paths) == 0:
        print("No heatmaps found. Skipping statistics.")
        return

    rng = np.random.RandomState(42)
    sample_paths = rng.choice(all_paths, size=min(20, len(all_paths)), replace=False)

    comp_keys = ['edge', 'var', 'ent', 'corn']
    stats = {k: {"min": [], "max": [], "mean": []} for k in comp_keys}
    combined_means = []

    for img_path_str in sample_paths:
        img_path = Path(img_path_str)
        stem = _heatmap_stem(img_path)
        out_dir = img_path.parent
        comp_data = {}
        ok = True
        for key in comp_keys:
            comp_path = out_dir / f"{stem}_satg_{key}.npy"
            if not comp_path.exists():
                ok = False
                break
            comp_data[key] = np.load(str(comp_path))
        if not ok:
            continue

        for key in comp_keys:
            hm = comp_data[key]
            stats[key]["min"].append(float(hm.min()))
            stats[key]["max"].append(float(hm.max()))
            stats[key]["mean"].append(float(hm.mean()))

        # Combined using default equal weights 0.25
        combined = 0.25 * sum(comp_data[k] for k in comp_keys)
        combined = np.clip(combined, 0, 1)
        combined_means.append(float(combined.mean()))

    if len(combined_means) == 0:
        print("No heatmap files found for statistics.")
        return

    print("\n=== Heatmap Component Statistics (20 random samples) ===")
    for key in comp_keys:
        if len(stats[key]["min"]) > 0:
            print(f"  {key}:  min={np.mean(stats[key]['min']):.4f}  "
                  f"max={np.mean(stats[key]['max']):.4f}  "
                  f"mean={np.mean(stats[key]['mean']):.4f}")
    if len(combined_means) > 0:
        print(f"  combined:  mean={np.mean(combined_means):.4f}")
    print(f"  (based on {len(combined_means)} samples)")


def verify_components(data_root: str) -> None:
    """Load 50 random component sets and verify correctness.

    Checks:
    - All values in [0, 1] for all components.
    - Warns if any component has mean < 0.02 (likely all zeros).
    - Warns if any component has mean > 0.95 (likely normalisation failure).
    - Prints pairwise correlation matrix to confirm components are not redundant.

    Args:
        data_root: Root directory of the Cityscapes dataset.
    """
    all_images = _find_images(data_root)
    if len(all_images) == 0:
        print("No images found. Skipping verification.")
        return

    rng = np.random.RandomState(123)
    sample_paths = rng.choice(all_images, size=min(50, len(all_images)), replace=False)

    comp_keys = ['edge', 'var', 'ent', 'corn']
    all_comps = {k: [] for k in comp_keys}

    n_loaded = 0
    n_warn_zero = 0
    n_warn_saturated = 0

    for img_path_str in sample_paths:
        img_path = Path(img_path_str)
        stem = _heatmap_stem(img_path)
        out_dir = img_path.parent
        comp_data = {}
        ok = True
        for key in comp_keys:
            comp_path = out_dir / f"{stem}_satg_{key}.npy"
            if not comp_path.exists():
                ok = False
                break
            comp_data[key] = np.load(str(comp_path))
        if not ok:
            continue

        n_loaded += 1
        for key in comp_keys:
            arr = comp_data[key]
            # Range check
            assert arr.min() >= 0.0, f"{key}: min {arr.min()} < 0 in {img_path.name}"
            assert arr.max() <= 1.0, f"{key}: max {arr.max()} > 1 in {img_path.name}"
            # Zero-warning
            if arr.mean() < 0.02:
                n_warn_zero += 1
            if arr.mean() > 0.95:
                n_warn_saturated += 1
            all_comps[key].append(arr.flatten())

    print(f"\n=== Component Verification (loaded {n_loaded} / {len(sample_paths)} samples) ===")
    if n_warn_zero > 0:
        print(f"  WARNING: {n_warn_zero} component(s) have mean < 0.02 (likely all zeros)")
    if n_warn_saturated > 0:
        print(f"  WARNING: {n_warn_saturated} component(s) have mean > 0.95 "
              f"(likely normalisation failure)")

    if n_loaded > 0:
        # Pairwise correlation
        print("\n  Pairwise correlations (flattened per image, averaged):")
        corr_sum = {f"{k1}-{k2}": [] for k1 in comp_keys for k2 in comp_keys if k1 < k2}
        for i in range(n_loaded):
            for k1 in comp_keys:
                for k2 in comp_keys:
                    if k1 < k2:
                        c = np.corrcoef(all_comps[k1][i], all_comps[k2][i])[0, 1]
                        corr_sum[f"{k1}-{k2}"].append(c)
        for pair, vals in corr_sum.items():
            if len(vals) > 0:
                avg_corr = np.mean(vals)
                print(f"    {pair}:  mean corr = {avg_corr:.4f}")


_COMPONENT_BYTES_ESTIMATE = 4 * 2048 * 1024 * 4  # 4 × float32 × 2048 × 1024


def _check_disk_space(
    out_dirs: set[Path], num_images: int, data_root: str
) -> None:
    """Print a warning if fewer than 2× the estimated space is free."""
    needed = num_images * _COMPONENT_BYTES_ESTIMATE
    # Check each unique output directory; warn for the worst one.
    min_free = min(
        shutil.disk_usage(d).free for d in _parent_filesystems(out_dirs)
    )
    if min_free < needed:
        free_gb = min_free / 2 ** 30
        need_gb = math.ceil(needed / 2 ** 30)
        print(
            f"  ⚠  Low disk space: ~{need_gb} GiB needed, "
            f"{free_gb:.1f} GiB free on the tightest volume.\n"
            f"     Heatmaps will be saved alongside source images "
            f"under {data_root}.\n"
            f"     Free up space or use a symlink to a larger volume."
        )


def _parent_filesystems(dirs: set[Path]) -> set[Path]:
    """Return the closest existing ancestor for each dir (disk_usage needs it)."""
    result: set[Path] = set()
    for d in dirs:
        d = d.resolve()
        while not d.exists():
            d = d.parent
        result.add(d)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute SATG structural heatmap components for Cityscapes training images."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=(
            "Path to Cityscapes dataset root (contains leftImg8bit/train/ or "
            "leftImg8bit_trainvaltest/train/), or directly to a city "
            "subdirectory.  Example: "
            "/Users/aadisaraf/Documents/research/SATG/data/cityscapes"
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images that already have all 4 heatmap component files.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification pass after precomputation.",
    )
    args = parser.parse_args()

    # Discover images
    print(f"Scanning for images under {args.data_root}/leftImg8bit/train/ ...")
    all_images = _find_images(args.data_root)
    print(f"Found {len(all_images)} PNG images.")

    if len(all_images) == 0:
        print("No images found. Exiting.")
        return

    # Resume: filter to images that are missing ≥1 component, and clean up
    # any partial files left by a prior interrupted run.
    if args.resume:
        missing = []
        for p in all_images:
            if not _has_all_components(p):
                _cleanup_incomplete(p)
                missing.append(p)
        skipped = len(all_images) - len(missing)
        all_images = missing
        if skipped:
            print(f"Resume mode: {skipped} already done, "
                  f"{len(all_images)} remaining.")
        if len(all_images) == 0:
            print("All images already processed. Nothing to do.")
            return

    # Disk space check
    _check_disk_space(
        {Path(p).parent for p in all_images},
        len(all_images),
        args.data_root,
    )

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

    print("Done. All heatmap components saved.")
    _print_statistics(args.data_root, all_images)

    if args.verify:
        verify_components(args.data_root)


if __name__ == "__main__":
    main()
