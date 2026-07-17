#!/usr/bin/env bash
# =============================================================================
# download_gta5_parallel.sh — Parallel GTA5 downloader (local-staged)
# =============================================================================
# TU Darmstadt throttles each connection to ~1.3 MB/s, so sequential download
# takes ~14 h. This downloads parts CONCURRENTLY to saturate the node's
# bandwidth.
#
# IMPORTANT: zips are staged on FAST LOCAL disk ($STAGE, default /mnt/gta5_zips)
# and only the extracted PNGs are moved to $OUT_DIR (which may be a blob mount).
# This keeps blobfuse out of the high-throughput download path — writing many
# large files straight to a blob mount fails ("curl (23)") because the local
# cache can't drain fast enough. Each part's zips are deleted right after
# extraction so $STAGE never accumulates more than ~PAR parts at once.
#
# Resume-safe: a staged zip that passes `unzip -t` is reused; extracted parts
# (already moved to $OUT_DIR) are detected by image count and skipped.
#
# Usage:
#   sudo mkdir -p /mnt/gta5_zips && sudo chown "$USER" /mnt/gta5_zips
#   PAR=3 bash cloud/download_gta5_parallel.sh                 # -> ./data/GTA5
#   PAR=4 STAGE=/mnt/gta5_zips bash cloud/download_gta5_parallel.sh /path/GTA5
# =============================================================================

set -uo pipefail

BASE_URL="https://download.visinf.tu-darmstadt.de/data/from_games/data"
OUT_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/data/GTA5}"
STAGE="${STAGE:-/mnt/gta5_zips}"
PAR="${PAR:-3}"
TOTAL_IMAGES=24966
IMAGES_PER_PART=$((TOTAL_IMAGES / 10))   # ~2496

mkdir -p "$OUT_DIR/images" "$OUT_DIR/labels"
if ! mkdir -p "$STAGE" 2>/dev/null || [ ! -w "$STAGE" ]; then
    echo "ERROR: staging dir '$STAGE' is not writable." >&2
    echo "  Create it first:  sudo mkdir -p '$STAGE' && sudo chown \"\$USER\" '$STAGE'" >&2
    exit 1
fi

echo "=========================================="
echo "  Parallel GTA5 Download (concurrency=$PAR)"
echo "=========================================="
echo "  Output (final PNGs): $OUT_DIR"
echo "  Staging (local zips): $STAGE"
echo "  Free on staging fs:"; df -h "$STAGE" | tail -1
echo ""

HAVE_IMG=$(find "$OUT_DIR/images" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
if [ "$HAVE_IMG" -eq "$TOTAL_IMAGES" ]; then
    echo "  All $TOTAL_IMAGES images already present — nothing to do."
    exit 0
fi

# ---- Per-part pipeline: download -> verify -> extract -> move -> delete -----
# Each part is independent, so xargs runs PAR of these at once.
do_part() {
    local i="$1"
    # Resume: a part that finished extraction leaves a marker on the output fs
    # (blob), so restarts/preemptions skip it instead of re-downloading.
    local marker="$OUT_DIR/.part_${i}_done"
    if [ -f "$marker" ]; then
        echo "  [$i] already done (marker) — skipping"
        return 0
    fi
    for kind in images labels; do
        local url="$BASE_URL/${i}_${kind}.zip"
        local z="$STAGE/${i}_${kind}.zip"
        if ! unzip -t "$z" >/dev/null 2>&1; then
            echo "  [$i] downloading ${kind} ..."
            if ! curl -L -C - --retry 5 --retry-delay 10 -sS -o "$z" "$url"; then
                rm -f "$z"
                curl -L --retry 5 --retry-delay 10 -sS -o "$z" "$url" || {
                    echo "  [$i] FAILED to download $kind" >&2; return 1; }
            fi
        fi
        if ! unzip -t "$z" >/dev/null 2>&1; then
            echo "  [$i] $kind zip corrupt after download" >&2; return 1
        fi
        # Extract into a temp dir ON THE OUTPUT FS (blob), not /mnt, so the only
        # thing /mnt holds is the zip itself — lets more parts run concurrently.
        local tmp; tmp="$(mktemp -d -p "$OUT_DIR")"
        unzip -q -o "$z" -d "$tmp"
        if [ -d "$tmp/$kind" ]; then
            mv "$tmp/$kind/"*.png "$OUT_DIR/$kind/" 2>/dev/null || true
        fi
        rm -rf "$tmp" "$z"
    done
    touch "$marker"
    echo "  [$i] done"
}
export -f do_part
export BASE_URL OUT_DIR STAGE

echo "--- Fetching 10 parts, $PAR at a time (zips staged on $STAGE) ---"
seq -w 1 10 | xargs -P "$PAR" -I {} bash -c 'do_part "$@"' _ {}
echo ""

# ---- Verify counts ---------------------------------------------------------
IMG=$(find "$OUT_DIR/images" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ')
LAB=$(find "$OUT_DIR/labels" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ')
echo "Images: $IMG / $TOTAL_IMAGES   Labels: $LAB / $TOTAL_IMAGES"
if [ "$IMG" -eq "$TOTAL_IMAGES" ] && [ "$LAB" -eq "$TOTAL_IMAGES" ]; then
    echo "=== GTA5 download complete ==="
    echo "Next: bash cloud/prepare_data.sh   (skips GTA5 download, does labels + Cityscapes + heatmaps)"
else
    echo "WARNING: count mismatch — re-run this script to fill gaps (it resumes)."
    exit 1
fi
