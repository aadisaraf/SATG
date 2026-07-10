#!/usr/bin/env bash
# =============================================================================
# download_gta5.sh — Download + extract GTA5 "Playing for Data" dataset
# =============================================================================
# Downloads all 10 parts of images (~57GB) and labels (~700MB) from
# TU Darmstadt, then extracts into:
#   <output_dir>/images/   (24,966 PNGs)
#   <output_dir>/labels/   (24,966 single-channel PNGs, indices 0-34)
#
# Resume-safe: counts existing images to determine which part to start from.
#
# Usage:
#   ./cloud/download_gta5.sh                              # -> ./data/GTA5
#   ./cloud/download_gta5.sh /path/to/GTA5                 # custom path
#   nohup ./cloud/download_gta5.sh > gta5.log 2>&1 &       # survive terminal close
#   tail -f gta5.log                                       # check progress
# =============================================================================

set -uo pipefail

BASE_URL="https://download.visinf.tu-darmstadt.de/data/from_games/data"
OUT_DIR="${1:-$(dirname "$0")/../data/GTA5}"
TOTAL_IMAGES=24966
IMAGES_PER_PART=$((TOTAL_IMAGES / 10))   # ~2497

mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR/images"
mkdir -p "$OUT_DIR/labels"

echo "=== GTA5 Download Script ==="
echo "Output: $OUT_DIR"

# Count what we already have (resume support)
CUR_IMAGES=$(find "$OUT_DIR/images" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
CUR_LABELS=$(find "$OUT_DIR/labels" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
echo "Existing: $CUR_IMAGES images / $CUR_LABELS labels"

# Calculate starting part
START_PART=$((CUR_IMAGES / IMAGES_PER_PART + 1))
if [ "$START_PART" -gt 10 ]; then START_PART=10; fi
if [ "$START_PART" -lt 1 ]; then START_PART=1; fi

echo "Starting from part $(printf '%02d' $START_PART)"
echo ""

for i in $(seq -w "$START_PART" 10); do
    IMG_ZIP="${i}_images.zip"
    LAB_ZIP="${i}_labels.zip"

    echo "--- Part $i ---"

    # Download + validate images zip (auto-resume on failure)
    for attempt in 1 2 3; do
        echo "  Downloading $IMG_ZIP ..."
        curl -L -C - --retry 5 --retry-delay 10 -o "$OUT_DIR/$IMG_ZIP" "$BASE_URL/$IMG_ZIP" && break
        echo "  Attempt $attempt failed, retrying in 30s ..."
        rm -f "$OUT_DIR/$IMG_ZIP"
        sleep 30
    done

    # Verify zip integrity
    if ! unzip -t "$OUT_DIR/$IMG_ZIP" &>/dev/null; then
        echo "  FATAL: $IMG_ZIP is corrupt after 3 download attempts."
        rm -f "$OUT_DIR/$IMG_ZIP"
        exit 1
    fi

    # Download + validate labels zip
    for attempt in 1 2 3; do
        echo "  Downloading $LAB_ZIP ..."
        curl -L -C - --retry 5 --retry-delay 10 -o "$OUT_DIR/$LAB_ZIP" "$BASE_URL/$LAB_ZIP" && break
        echo "  Attempt $attempt failed, retrying in 30s ..."
        rm -f "$OUT_DIR/$LAB_ZIP"
        sleep 30
    done

    if ! unzip -t "$OUT_DIR/$LAB_ZIP" &>/dev/null; then
        echo "  FATAL: $LAB_ZIP is corrupt after 3 download attempts."
        rm -f "$OUT_DIR/$LAB_ZIP"
        exit 1
    fi

    # Extract into temp, then move to flat dirs
    TMP=$(mktemp -d)

    echo "  Extracting $IMG_ZIP ..."
    unzip -q -o "$OUT_DIR/$IMG_ZIP" -d "$TMP"
    echo "  Extracting $LAB_ZIP ..."
    unzip -q -o "$OUT_DIR/$LAB_ZIP" -d "$TMP"

    PNG_MOVED=0
    if [ -d "$TMP/images" ]; then
        mv "$TMP/images/"*.png "$OUT_DIR/images/"
        PNG_MOVED=$((PNG_MOVED + $(ls "$TMP/images/"*.png 2>/dev/null | wc -l)))
    fi
    if [ -d "$TMP/labels" ]; then
        mv "$TMP/labels/"*.png "$OUT_DIR/labels/"
    fi

    rm -rf "$TMP"
    rm "$OUT_DIR/$IMG_ZIP" "$OUT_DIR/$LAB_ZIP"

    CUR_IMAGES=$(find "$OUT_DIR/images" -maxdepth 1 -name "*.png" | wc -l)
    echo "  Done part $i  (total images: $CUR_IMAGES / $TOTAL_IMAGES)"
    echo ""
done

# Final verification
echo "=== Verifying ==="
IMG_COUNT=$(find "$OUT_DIR/images" -maxdepth 1 -name "*.png" | wc -l)
LAB_COUNT=$(find "$OUT_DIR/labels" -maxdepth 1 -name "*.png" | wc -l)
echo "Images: $IMG_COUNT   (expected: $TOTAL_IMAGES)"
echo "Labels: $LAB_COUNT   (expected: $TOTAL_IMAGES)"

if [ "$IMG_COUNT" -eq "$TOTAL_IMAGES" ] && [ "$LAB_COUNT" -eq "$TOTAL_IMAGES" ]; then
    echo ""
    echo "=== Download complete! ==="
    echo ""
    echo "Next step:"
    echo "  python -m precompute.preprocess_gta5_labels --label_root $OUT_DIR/labels"
else
    echo ""
    echo "WARNING: File counts don't match expected $TOTAL_IMAGES."
    exit 1
fi
