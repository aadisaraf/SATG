#!/usr/bin/env bash
# =============================================================================
# prepare_data.sh — Download datasets and precompute SATG heatmaps
# =============================================================================
# Downloads GTA5 (automated), prints instructions for Cityscapes (manual
# download due to registration), precomputes structural-prior heatmap
# components, and validates the output.
#
# Usage:
#   ./cloud/prepare_data.sh                                           # defaults
#   ./cloud/prepare_data.sh --data_root /mnt/data                     # custom root
#   ./cloud/prepare_data.sh --data_root /mnt/data --cityscapes_root /mnt/data/cityscapes
#
# Arguments:
#   --data_root        Parent directory for all datasets  (default: ./data)
#   --cityscapes_root  Cityscapes directory                (default: <data_root>/cityscapes)
# =============================================================================

set -euo pipefail

trap 'echo "=== DATA PREP FAILED at line $LINENO ===" >&2; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- Defaults ---------------------------------------------------------------
DATA_ROOT="$(pwd)/data"
CITYSCAPES_ROOT="$DATA_ROOT/cityscapes"

# ---- Argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --cityscapes_root)
            CITYSCAPES_ROOT="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument '$1'"
            echo "Usage: $0 [--data_root <path>] [--cityscapes_root <path>]"
            exit 1
            ;;
    esac
done

# Resolve relative paths if needed
case "$DATA_ROOT" in
    /*) ;;
    *) DATA_ROOT="$(pwd)/$DATA_ROOT" ;;
esac
case "$CITYSCAPES_ROOT" in
    /*) ;;
    *) CITYSCAPES_ROOT="$(pwd)/$CITYSCAPES_ROOT" ;;
esac

mkdir -p "$DATA_ROOT"

echo "=========================================="
echo "  SATG Data Preparation"
echo "=========================================="
echo "  Data root:        $DATA_ROOT"
echo "  Cityscapes root:  $CITYSCAPES_ROOT"
echo ""

# =============================================================================
# Step 1 — Download GTA5
# =============================================================================
echo "=========================================="
echo "  Step 1 / 4: Download GTA5"
echo "=========================================="
if [ -f "cloud/download_gta5.sh" ]; then
    bash cloud/download_gta5.sh "$DATA_ROOT/GTA5"
else
    echo "WARNING: cloud/download_gta5.sh not found — skipping GTA5 download."
    echo "  Download manually from:"
    echo "  https://download.visinf.tu-darmstadt.de/data/from_games/"
fi

# ------------------------------------------------------------------
# Preprocess GTA5 labels (map to Cityscapes trainIds)
# ------------------------------------------------------------------
if [ -d "$DATA_ROOT/GTA5/labels" ]; then
    echo ""
    echo "--- Preprocessing GTA5 labels ---"
    python -m precompute.preprocess_gta5_labels --label_root "$DATA_ROOT/GTA5/labels"
else
    echo ""
    echo "WARNING: GTA5 labels not found — skipping label preprocessing."
    echo "  Run './cloud/download_gta5.sh' first to download GTA5."
fi

# =============================================================================
# Step 2 — Cityscapes download (Kaggle, automated)
# =============================================================================
echo ""
echo "=========================================="
echo "  Step 2 / 4: Cityscapes Download (Kaggle)"
echo "=========================================="
echo ""

if [ -d "$CITYSCAPES_ROOT/leftImg8bit/train" ] && [ -d "$CITYSCAPES_ROOT/gtFine/train" ]; then
    echo "  Cityscapes already present at $CITYSCAPES_ROOT — skipping download."
elif [ -f "cloud/download_cityscapes.sh" ]; then
    echo "  Fetching leftImg8bit + gtFine from Kaggle mirrors."
    echo "  Requires Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)."
    echo ""
    bash cloud/download_cityscapes.sh "$CITYSCAPES_ROOT"
else
    echo "  WARNING: cloud/download_cityscapes.sh not found."
    echo "  Download manually from Kaggle:"
    echo "    kaggle datasets download -d chrisviviers/cityscapes-leftimg8bit-trainvaltest -p \"$CITYSCAPES_ROOT\" --unzip"
    echo "    kaggle datasets download -d kclaude/gtfine-trainvaltest              -p \"$CITYSCAPES_ROOT\" --unzip"
fi
echo ""

# =============================================================================
# Step 3 — Precompute SATG heatmaps
# =============================================================================
echo "=========================================="
echo "  Step 3 / 4: Precompute SATG Heatmaps"
echo "=========================================="

if [ -d "$CITYSCAPES_ROOT/leftImg8bit/train" ]; then
    echo "  Found Cityscapes images at $CITYSCAPES_ROOT/leftImg8bit/train"
    echo "  Running heatmap precomputation (this takes ∼10-20 min on 8 cores)..."
    echo ""
    python -m precompute.compute_heatmaps \
        --data_root "$CITYSCAPES_ROOT" \
        --num_workers 8 \
        --resume \
        --verify
    echo "  Heatmap precomputation complete."
else
    echo "  Cityscapes training images not found at $CITYSCAPES_ROOT/leftImg8bit/train"
    echo "  Skipping heatmap precomputation."
    echo ""
    echo "  After downloading Cityscapes, run:"
    echo "    python -m precompute.compute_heatmaps \\"
    echo "        --data_root \"$CITYSCAPES_ROOT\" --num_workers 8 --resume --verify"
fi

# =============================================================================
# Step 4 — Validate heatmap count
# =============================================================================
echo ""
echo "=========================================="
echo "  Step 4 / 4: Validate Heatmaps"
echo "=========================================="

TRAIN_DIR="$CITYSCAPES_ROOT/leftImg8bit/train"
if [ -d "$TRAIN_DIR" ]; then
    HEATMAP_COUNT=$(find "$TRAIN_DIR" -name "*_satg_*.npy" 2>/dev/null | wc -l | tr -d ' ')
    IMAGE_COUNT=$(find "$TRAIN_DIR" -name "*_leftImg8bit.png" 2>/dev/null | wc -l | tr -d ' ')
    EXPECTED_HEATMAPS=$((IMAGE_COUNT * 4))

    echo "  Training images found:      $IMAGE_COUNT"
    echo "  Heatmap files found:        $HEATMAP_COUNT"
    echo "  Expected (images × 4):      $EXPECTED_HEATMAPS"

    if [ "$IMAGE_COUNT" -eq 2975 ] && [ "$HEATMAP_COUNT" -eq 11900 ]; then
        echo ""
        echo "  >>> HEATMAP VALIDATION PASSED <<<"
        echo "  All $HEATMAP_COUNT heatmap components ($IMAGE_COUNT images × 4) present."
        echo ""
    elif [ "$HEATMAP_COUNT" -eq "$EXPECTED_HEATMAPS" ] && [ "$EXPECTED_HEATMAPS" -gt 0 ]; then
        echo ""
        echo "  >>> HEATMAP VALIDATION PASSED ($HEATMAP_COUNT files) <<<"
        echo ""
    else
        echo ""
        echo "  WARNING: Heatmap count mismatch."
        echo "  Expected $EXPECTED_HEATMAPS heatmaps for $IMAGE_COUNT images."
        echo "  Re-run with: python -m precompute.compute_heatmaps \\"
        echo "      --data_root \"$CITYSCAPES_ROOT\" --num_workers 8 --resume"
    fi
else
    echo "  Cannot validate heatmaps — Cityscapes training dir not found."
fi

echo ""
echo "=========================================="
echo "  Data preparation complete"
echo "=========================================="
echo ""
echo "  Next step: run training experiments"
echo "    ./cloud/run_phase8.sh                    # Phase 8 baselines"
echo "    ./cloud/run_phase9.sh                    # Phase 9 SATG experiments"
echo ""
