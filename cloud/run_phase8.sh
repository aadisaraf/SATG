#!/usr/bin/env bash
# =============================================================================
# run_phase8.sh — SATG Phase 8 Baseline Experiment Launcher
# =============================================================================
# Runs all 6 baseline training commands sequentially:
#   - Source Only  × 3 seeds {42, 1337, 2024}
#   - Mean Teacher × 3 seeds {42, 1337, 2024}
#
# Each run logs to cloud/logs/phase8_<name>_seed<N>.log
# Prints a summary table on completion.
#
# Usage:
#   ./cloud/run_phase8.sh                           # full run
#   nohup ./cloud/run_phase8.sh > cloud/logs/phase8_full.log 2>&1 &
#
# Estimated total time: ~5–9 hours on A100
# =============================================================================

set -euo pipefail

trap 'echo "=== PHASE 8 ABORTED at line $LINENO ===" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="cloud/logs"
mkdir -p "$LOG_DIR"

SEEDS=(42 1337 2024)
RESULTS_FILE="$LOG_DIR/phase8_results.txt"
START_TIME=$(date +%s)

echo "============================================="
echo "  SATG Phase 8 — Baseline Experiments"
echo "  Started: $(date)"
echo "============================================="
echo ""

# Clear previous results
> "$RESULTS_FILE"

run_experiment() {
    local name="$1"
    local config="$2"
    local seed="$3"
    local logfile="$4"

    echo "-----------------------------------------------------------------"
    echo "  [$name] seed=$seed"
    echo "  Config: $config"
    echo "  Log:    $logfile"
    echo "-----------------------------------------------------------------"

    # Record start
    local run_start=$(date +%s)

    # Run training
    python -m training.trainer --config "$config" "seed=$seed" 2>&1 | tee "$logfile"
    local exit_code=$?

    local run_end=$(date +%s)
    local duration=$((run_end - run_start))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))

    if [ "$exit_code" -eq 0 ]; then
        echo "  ✓ [$name] seed=$seed completed in ${minutes}m${seconds}s"
        # Extract best mIoU from log
        BEST_MIOU=$(grep "★ New best" "$logfile" | tail -1 | grep -oP '\d+\.\d+(?=%)' || echo "N/A")
        echo "$name | seed=$seed | ${minutes}m${seconds}s | best_mIoU=$BEST_MIOU" >> "$RESULTS_FILE"
    else
        echo "  ✗ [$name] seed=$seed FAILED (exit code $exit_code)"
        echo "$name | seed=$seed | FAILED" >> "$RESULTS_FILE"
    fi
    echo ""
}

# =============================================================================
# Source Only (lambda_target=0.0, skip_heatmap=true)
# =============================================================================
echo "====== Source Only Baseline ======"
for seed in "${SEEDS[@]}"; do
    run_experiment \
        "source_only" \
        "configs/source_only.yaml" \
        "$seed" \
        "$LOG_DIR/phase8_source_only_seed${seed}.log"
done

# =============================================================================
# Mean Teacher (hard gate, tau_conf=0.90, tau_struct=1.01, skip_heatmap=true)
# =============================================================================
echo "====== Mean Teacher Baseline ======"
for seed in "${SEEDS[@]}"; do
    run_experiment \
        "mean_teacher" \
        "configs/baseline_mean_teacher.yaml" \
        "$seed" \
        "$LOG_DIR/phase8_mean_teacher_seed${seed}.log"
done

# =============================================================================
# Summary
# =============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_HOURS=$((TOTAL_MINUTES / 60))
TOTAL_MINUTES_REMAINDER=$((TOTAL_MINUTES % 60))

echo "============================================="
echo "  Phase 8 Complete"
echo "  Finished: $(date)"
echo "  Total duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES_REMAINDER}m"
echo "============================================="
echo ""
echo "  Summary:"
echo "  --------"
if [ -f "$RESULTS_FILE" ]; then
    column -t -s '|' "$RESULTS_FILE" 2>/dev/null || cat "$RESULTS_FILE"
fi
echo ""
echo "  Logs: $LOG_DIR/phase8_*.log"
echo "  Checkpoints: checkpoints/"
echo ""

# Extract all best mIoU values for quick reference
echo "Quick mIoU summary:"
for log in "$LOG_DIR"/phase8_*.log; do
    NAME=$(basename "$log" .log)
    BEST=$(grep "★ New best" "$log" | tail -1 | grep -oP '\d+\.\d+(?=%)' || echo "N/A")
    echo "  $NAME: $BEST"
done
