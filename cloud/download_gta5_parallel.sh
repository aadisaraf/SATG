#!/usr/bin/env bash
# =============================================================================
# download_gta5_parallel.sh — Parallel GTA5 downloader
# =============================================================================
# TU Darmstadt throttles each connection to ~1.3 MB/s, so the sequential
# download_gta5.sh takes ~14 h. This fetches all 10 image + 10 label zips
# CONCURRENTLY (default 10 at a time) to saturate the node's bandwidth, then
# verifies + extracts into the same layout download_gta5.sh produces:
#   <out>/images/  (24,966 PNGs)
#   <out>/labels/  (24,966 PNGs)
#
# Resume-safe: a zip that already passes `unzip -t` is skipped; partials resume
# with curl -C -. Re-run freely.
#
# Usage:
#   bash cloud/download_gta5_parallel.sh                 # -> ./data/GTA5
#   bash cloud/download_gta5_parallel.sh /path/to/GTA5
#   PAR=6 bash cloud/download_gta5_parallel.sh           # fewer parallel conns
# =============================================================================

set -uo pipefail

BASE_URL="https://download.visinf.tu-darmstadt.de/data/from_games/data"
OUT_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/data/GTA5}"
PAR="${PAR:-10}"
TOTAL_IMAGES=24966

mkdir -p "$OUT_DIR/images" "$OUT_DIR/labels"
cd "$OUT_DIR"

echo "=========================================="
echo "  Parallel GTA5 Download (concurrency=$PAR)"
echo "=========================================="
echo "  Output: $OUT_DIR"
echo ""

# Skip work if already complete
HAVE_IMG=$(find "$OUT_DIR/images" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
if [ "$HAVE_IMG" -eq "$TOTAL_IMAGES" ]; then
    echo "  All $TOTAL_IMAGES images already present — nothing to do."
    exit 0
fi

# ---- Build the URL list (image + label zips for parts 01..10) --------------
urls=()
for i in $(seq -w 1 10); do
    urls+=("$BASE_URL/${i}_images.zip")
    urls+=("$BASE_URL/${i}_labels.zip")
done

# ---- Parallel download (resume-safe, skips already-complete zips) ----------
fetch_one() {
    local url="$1" f="${url##*/}"
    if unzip -t "$f" >/dev/null 2>&1; then
        echo "  have   $f"
        return 0
    fi
    echo "  get    $f"
    if ! curl -L -C - --retry 5 --retry-delay 10 -sS -o "$f" "$url"; then
        # -C - can fail on a stale/complete file; retry from scratch once
        rm -f "$f"
        curl -L --retry 5 --retry-delay 10 -sS -o "$f" "$url"
    fi
}
export -f fetch_one

echo "--- Downloading ${#urls[@]} zips, $PAR at a time ---"
printf '%s\n' "${urls[@]}" | xargs -P "$PAR" -I {} bash -c 'fetch_one "$@"' _ {}
echo ""

# ---- Verify + extract each part (sequential; cheap vs. the download) -------
echo "--- Verifying + extracting ---"
for i in $(seq -w 1 10); do
    for kind in images labels; do
        zip="${i}_${kind}.zip"
        [ -f "$zip" ] || continue
        if ! unzip -t "$zip" >/dev/null 2>&1; then
            echo "  FATAL: $zip is corrupt — re-run this script to re-fetch it." >&2
            exit 1
        fi
        tmp="$(mktemp -d)"
        unzip -q -o "$zip" -d "$tmp"
        if [ -d "$tmp/$kind" ]; then
            mv "$tmp/$kind/"*.png "$OUT_DIR/$kind/" 2>/dev/null || true
        fi
        rm -rf "$tmp"
        rm -f "$zip"
    done
    echo "  extracted part $i"
done

# ---- Verify counts ---------------------------------------------------------
IMG=$(find "$OUT_DIR/images" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ')
LAB=$(find "$OUT_DIR/labels" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ')
echo ""
echo "Images: $IMG / $TOTAL_IMAGES   Labels: $LAB / $TOTAL_IMAGES"
if [ "$IMG" -eq "$TOTAL_IMAGES" ] && [ "$LAB" -eq "$TOTAL_IMAGES" ]; then
    echo "=== GTA5 download complete ==="
    echo "Next: python -m precompute.preprocess_gta5_labels --label_root $OUT_DIR/labels"
else
    echo "WARNING: count mismatch — re-run this script to fill gaps."
    exit 1
fi
