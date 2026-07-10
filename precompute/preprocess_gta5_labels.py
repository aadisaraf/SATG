"""Preprocess GTA5 ground-truth labels into Cityscapes 19-class trainID maps.

GTA5 labels are stored as single-channel indexed PNG files where each pixel
value is the GTA5 class index (0–32). This script converts them to Cityscapes
19-class trainIDs (0–18 or 255) using the standard GTA5→Cityscapes mapping
and saves the result as ``{stem}_trainids.png`` alongside the original file.

Usage::
    python -m precompute.preprocess_gta5_labels --label_root /Users/aadisaraf/Documents/research/SATG/data/GTA5/labels
"""

import argparse
import multiprocessing
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure project root is on sys.path (works when invoked both as `-m` and directly)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.label_mapping import GTA5_TO_CITYSCAPES_19


def _process_single_label(label_path: Path) -> str:
    """Load a single GTA5 label PNG, map to Cityscapes trainIDs, save.

    Args:
        label_path: Path to the GTA5 label PNG.

    Returns:
        Status string: ``"OK: {path}"`` or ``"ERR: {path} {reason}"``.
    """
    try:
        arr = np.array(Image.open(label_path))
    except Exception as e:
        return f"ERR: {label_path} — {e}"

    if arr.ndim != 2:
        return f"ERR: {label_path} has {arr.ndim} channels (expected 2)"

    out = np.full_like(arr, 255, dtype=np.uint8)
    for gta5_idx, cs_id in GTA5_TO_CITYSCAPES_19.items():
        out[arr == gta5_idx] = cs_id

    stem = label_path.stem
    out_path = label_path.with_name(f"{stem}_trainids.png")
    Image.fromarray(out, mode="L").save(out_path)
    return f"OK: {out_path}"


def _discover_labels(label_root: Path) -> list[Path]:
    """Recursively discover all PNG label files under *label_root*."""
    paths = sorted(label_root.rglob("*.png"))
    paths = [p for p in paths if not p.name.endswith("_trainids.png")]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess GTA5 labels into Cityscapes 19-class trainID maps."
    )
    parser.add_argument(
        "--label_root",
        type=str,
        required=True,
        help="Root directory of GTA5 label PNGs (recursively scanned).",
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=multiprocessing.cpu_count(),
        help=f"Number of parallel workers (default: {multiprocessing.cpu_count()}).",
    )
    args = parser.parse_args()

    label_root = Path(args.label_root)
    if not label_root.is_dir():
        print(f"Error: '{label_root}' is not a directory.")
        return

    label_files = _discover_labels(label_root)
    if not label_files:
        print(f"No label PNG files found under {label_root}")
        return

    print(f"Found {len(label_files)} label files. Processing with {args.pool_size} workers...")

    with multiprocessing.Pool(args.pool_size) as pool:
        results = list(tqdm(
            pool.imap_unordered(_process_single_label, label_files),
            total=len(label_files),
            desc="Preprocessing labels",
            unit="file",
        ))

    ok_count = sum(1 for r in results if r.startswith("OK"))
    err_count = sum(1 for r in results if r.startswith("ERR"))
    print(f"\nDone. Processed {ok_count}/{len(results)} files successfully.")
    if err_count > 0:
        for r in results:
            if r.startswith("ERR"):
                print(f"  {r}")


if __name__ == "__main__":
    main()