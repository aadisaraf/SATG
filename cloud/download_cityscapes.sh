#!/usr/bin/env bash
# =============================================================================
# download_cityscapes.sh — Download Cityscapes from Kaggle (no registration flow)
# =============================================================================
# Pulls the two mirrored Kaggle datasets and lays them out in the exact
# structure SATG expects:
#
#   leftImg8bit  ->  https://www.kaggle.com/datasets/chrisviviers/cityscapes-leftimg8bit-trainvaltest
#   gtFine       ->  https://www.kaggle.com/datasets/kclaude/gtfine-trainvaltest
#
# Final layout produced under <cityscapes_root>:
#   <root>/leftImg8bit/{train,val,test}/<city>/*_leftImg8bit.png
#   <root>/gtFine/{train,val}/<city>/*_gtFine_labelIds.png
#   <root>/gtFine/{train,val}/<city>/*_gtFine_labelTrainIds.png   (generated if missing)
#
# The SATG loaders require:
#   - training images:  leftImg8bit/train/<city>/*_leftImg8bit.png   (2,975)
#   - val label suffix: _gtFine_labelTrainIds.png                     (500)
#     (see data/cityscapes_loader.py:26)
#
# Credentials (you provide — this script never asks for them):
#   Put your Kaggle API token at ~/.kaggle/kaggle.json (chmod 600), OR export
#   KAGGLE_USERNAME and KAGGLE_KEY. Get the token from:
#   https://www.kaggle.com/settings/account  ->  "Create New API Token"
#
# Resume-safe: skips a dataset whose target dir already looks populated.
#
# Usage:
#   ./cloud/download_cityscapes.sh                      # -> ./data/cityscapes
#   ./cloud/download_cityscapes.sh /mnt/data/cityscapes # custom root
#   SKIP_TRAINIDS=1 ./cloud/download_cityscapes.sh      # don't auto-generate trainIds
# =============================================================================

set -euo pipefail

trap 'echo "=== CITYSCAPES DOWNLOAD FAILED at line $LINENO ===" >&2; exit 1' ERR

LEFT_SLUG="chrisviviers/cityscapes-leftimg8bit-trainvaltest"
GT_SLUG="kclaude/gtfine-trainvaltest"

CITYSCAPES_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)/data/cityscapes}"
# Resolve to absolute
case "$CITYSCAPES_ROOT" in
    /*) ;;
    *) CITYSCAPES_ROOT="$(pwd)/$CITYSCAPES_ROOT" ;;
esac

mkdir -p "$CITYSCAPES_ROOT"

echo "=========================================="
echo "  Cityscapes Download (Kaggle)"
echo "=========================================="
echo "  Target root: $CITYSCAPES_ROOT"
echo "  leftImg8bit: $LEFT_SLUG"
echo "  gtFine:      $GT_SLUG"
echo ""

# ------------------------------------------------------------------
# 0. Ensure kaggle CLI + credentials
# ------------------------------------------------------------------
if ! command -v kaggle &> /dev/null; then
    echo "--- Installing kaggle CLI ---"
    pip install --quiet kaggle
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ] && { [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; }; then
    echo "ERROR: No Kaggle credentials found." >&2
    echo "  Provide ONE of the following, then re-run:" >&2
    echo "    1) ~/.kaggle/kaggle.json   (from https://www.kaggle.com/settings/account)" >&2
    echo "         mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json" >&2
    echo "    2) export KAGGLE_USERNAME=<you> KAGGLE_KEY=<token>" >&2
    exit 1
fi
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    chmod 600 "$HOME/.kaggle/kaggle.json" 2>/dev/null || true
fi

# ------------------------------------------------------------------
# Helper: move a discovered subtree (leftImg8bit / gtFine) into ROOT
# ------------------------------------------------------------------
# $1 = extraction dir, $2 = component name (leftImg8bit|gtFine)
normalize_component() {
    local src_root="$1"
    local comp="$2"
    local dest="$CITYSCAPES_ROOT/$comp"

    # Find the component directory anywhere in the extracted tree
    local found
    found="$(find "$src_root" -type d -iname "$comp" -print -quit 2>/dev/null || true)"

    if [ -z "$found" ]; then
        # Some mirrors extract split dirs (train/val/test) directly at top level.
        if [ -d "$src_root/train" ] || [ -d "$src_root/val" ]; then
            found="$src_root"
        else
            echo "ERROR: could not locate a '$comp' directory in $src_root" >&2
            echo "  Extracted top level was:" >&2
            ls -la "$src_root" >&2 || true
            exit 1
        fi
    fi

    echo "  Normalizing $comp: $found -> $dest"
    mkdir -p "$dest"
    # Move split dirs (train/val/test) into place; merge if partially present.
    for split in train val test; do
        if [ -d "$found/$split" ]; then
            mkdir -p "$dest/$split"
            # cp -rl uses hardlinks when possible (fast, low space); falls back to cp
            cp -rl "$found/$split/." "$dest/$split/" 2>/dev/null || cp -r "$found/$split/." "$dest/$split/"
        fi
    done
}

# ------------------------------------------------------------------
# Helper: download + unzip a Kaggle dataset into a temp dir, normalize
# ------------------------------------------------------------------
# $1 = slug, $2 = component name
fetch_dataset() {
    local slug="$1"
    local comp="$2"
    local dest="$CITYSCAPES_ROOT/$comp"

    # Resume: skip if the component already has train images/labels
    local existing
    existing="$(find "$dest/train" -type f -name "*.png" 2>/dev/null | head -1 || true)"
    if [ -n "$existing" ]; then
        echo "--- $comp already present under $dest — skipping download ---"
        return 0
    fi

    local tmp
    tmp="$(mktemp -d)"
    echo "--- Downloading $comp from $slug ---"
    # --unzip extracts and removes the archive; large (~11GB left / ~1.5GB gt)
    kaggle datasets download -d "$slug" -p "$tmp" --unzip
    normalize_component "$tmp" "$comp"
    rm -rf "$tmp"
    echo "  $comp ready."
    echo ""
}

# ------------------------------------------------------------------
# 1. leftImg8bit
# ------------------------------------------------------------------
fetch_dataset "$LEFT_SLUG" "leftImg8bit"

# ------------------------------------------------------------------
# 2. gtFine
# ------------------------------------------------------------------
fetch_dataset "$GT_SLUG" "gtFine"

# ------------------------------------------------------------------
# 3. Ensure *_gtFine_labelTrainIds.png exist (loader requires them)
# ------------------------------------------------------------------
TRAINID_COUNT="$(find "$CITYSCAPES_ROOT/gtFine" -name "*_gtFine_labelTrainIds.png" 2>/dev/null | wc -l | tr -d ' ')"
if [ "${SKIP_TRAINIDS:-0}" = "1" ]; then
    echo "--- SKIP_TRAINIDS=1 set — not generating trainIds (found $TRAINID_COUNT) ---"
elif [ "$TRAINID_COUNT" -ge 3475 ]; then
    echo "--- labelTrainIds already present ($TRAINID_COUNT files) — skipping generation ---"
else
    echo "--- Generating *_gtFine_labelTrainIds.png ($TRAINID_COUNT found, expected ~3475) ---"
    if ! python -c "import cityscapesscripts" &> /dev/null; then
        pip install --quiet cityscapesscripts
    fi

    # Preferred path: official script (needs *_gtFine_polygons.json).
    POLYGONS="$(find "$CITYSCAPES_ROOT/gtFine" -name "*_gtFine_polygons.json" 2>/dev/null | head -1 || true)"
    if [ -n "$POLYGONS" ] && command -v csCreateTrainIdLabelImgs &> /dev/null; then
        echo "  polygons present — using csCreateTrainIdLabelImgs"
        CITYSCAPES_DATASET="$CITYSCAPES_ROOT" csCreateTrainIdLabelImgs || true
    fi

    # Fallback: map *_gtFine_labelIds.png -> trainIds directly from the PNGs.
    # Works even when the Kaggle mirror stripped the polygon JSONs.
    TRAINID_COUNT="$(find "$CITYSCAPES_ROOT/gtFine" -name "*_gtFine_labelTrainIds.png" 2>/dev/null | wc -l | tr -d ' ')"
    if [ "$TRAINID_COUNT" -lt 3475 ]; then
        echo "  polygons missing/insufficient — converting labelIds -> trainIds from PNGs"
        CITYSCAPES_ROOT="$CITYSCAPES_ROOT" python - <<'PY'
import os, glob
import numpy as np
from PIL import Image
from cityscapesscripts.helpers.labels import id2label

root = os.environ["CITYSCAPES_ROOT"]
# Build a 256-entry LUT: labelId -> trainId (255 = ignore)
lut = np.full(256, 255, dtype=np.uint8)
for lid, lab in id2label.items():
    if 0 <= lid < 256:
        t = lab.trainId
        lut[lid] = 255 if t in (-1, 255) else t

made = 0
for f in glob.glob(os.path.join(root, "gtFine", "*", "*", "*_gtFine_labelIds.png")):
    out = f.replace("_gtFine_labelIds.png", "_gtFine_labelTrainIds.png")
    if os.path.exists(out):
        continue
    arr = np.array(Image.open(f))
    Image.fromarray(lut[arr]).save(out)
    made += 1
print(f"  converted {made} labelTrainIds via LUT")
PY
    fi
    TRAINID_COUNT="$(find "$CITYSCAPES_ROOT/gtFine" -name "*_gtFine_labelTrainIds.png" 2>/dev/null | wc -l | tr -d ' ')"
    echo "  labelTrainIds now: $TRAINID_COUNT"
fi

# ------------------------------------------------------------------
# 4. Verify structure + counts
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Verifying Cityscapes layout"
echo "=========================================="
TRAIN_IMGS="$(find "$CITYSCAPES_ROOT/leftImg8bit/train" -name "*_leftImg8bit.png" 2>/dev/null | wc -l | tr -d ' ')"
VAL_IMGS="$(find "$CITYSCAPES_ROOT/leftImg8bit/val" -name "*_leftImg8bit.png" 2>/dev/null | wc -l | tr -d ' ')"
VAL_TRAINIDS="$(find "$CITYSCAPES_ROOT/gtFine/val" -name "*_gtFine_labelTrainIds.png" 2>/dev/null | wc -l | tr -d ' ')"

echo "  train images (expect 2975): $TRAIN_IMGS"
echo "  val images   (expect  500): $VAL_IMGS"
echo "  val trainIds (expect  500): $VAL_TRAINIDS"
echo ""

if [ "$TRAIN_IMGS" -eq 2975 ] && [ "$VAL_IMGS" -eq 500 ] && [ "$VAL_TRAINIDS" -eq 500 ]; then
    echo "  >>> CITYSCAPES READY <<<"
else
    echo "  WARNING: counts differ from stock Cityscapes. Inspect the layout:"
    echo "    ls $CITYSCAPES_ROOT/leftImg8bit/train/"
    echo "    ls $CITYSCAPES_ROOT/gtFine/val/"
    echo "  (Non-fatal if the Kaggle mirror is a subset — but heatmap validation"
    echo "   in prepare_data.sh expects the full 2975 training images.)"
fi

echo ""
echo "  Next: precompute heatmaps —"
echo "    python -m precompute.compute_heatmaps \\"
echo "        --data_root \"$CITYSCAPES_ROOT\" --num_workers 8 --resume --verify"
