#!/usr/bin/env bash
# =============================================================================
# run_phase9.sh — SATG Phase 9 Main Experiments + Ablations Launcher
# =============================================================================
# Runs all SATG experiments and ablations sequentially:
#
#   Section A — Main Experiments
#     - SATG Hard        × 3 seeds  {42, 1337, 2024}
#     - SATG Soft-Weight × 3 seeds  {42, 1337, 2024}
#     - SATG Soft-Label  × 5 seeds  {42, 1337, 2024, 7, 99}
#
#   Section B — Ablation A: Prior Component Isolation  (seed=42)
#   Section C — Ablation B: tau_conf Sweep              (seed=42)
#   Section D — Ablation C: tau_struct Sweep             (seed=42)
#   Section E — Ablation E: Kernel Size & Sigma          (seed=42)
#   Section F — Ablation G: Soft-Label k Sensitivity     (seed=42)
#   Section G — Grid Search: tau_conf × tau_struct       (seed=42)
#
# Each run logs to cloud/logs/phase9_<name>_seed<N>.log
# Prints a summary table on completion.
#
# Usage:
#   ./cloud/run_phase9.sh
#   nohup ./cloud/run_phase9.sh > cloud/logs/phase9_full.log 2>&1 &
#
# Estimated total time: ~40–60 hours on A100
# =============================================================================

set -euo pipefail

trap 'echo "=== PHASE 9 ABORTED at line $LINENO ===" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="cloud/logs"
mkdir -p "$LOG_DIR"

RESULTS_FILE="$LOG_DIR/phase9_results.txt"
START_TIME=$(date +%s)

echo "============================================="
echo "  SATG Phase 9 — Main Experiments + Ablations"
echo "  Started: $(date)"
echo "============================================="
echo ""

> "$RESULTS_FILE"

# ---- Helper ----------------------------------------------------------------
run_experiment() {
    local section="$1"
    local name="$2"
    local config="$3"
    local overrides="$4"
    local seed="$5"
    local logfile="$LOG_DIR/phase9_${name}_seed${seed}.log"

    echo "-----------------------------------------------------------------"
    echo "  [$section] $name  seed=$seed"
    echo "  Config: $config"
    echo "  Log:    $logfile"
    echo "-----------------------------------------------------------------"

    local run_start=$(date +%s)

    # Build command: config + overrides + seed
    # shellcheck disable=SC2086
    python -m training.trainer \
        --config "$config" \
        $overrides \
        "seed=$seed" \
        2>&1 | tee "$logfile"
    local exit_code=$?

    local run_end=$(date +%s)
    local duration=$((run_end - run_start))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))

    if [ "$exit_code" -eq 0 ]; then
        echo "  ✓ [$section] $name seed=$seed completed in ${minutes}m${seconds}s"
        BEST_MIOU=$(grep "★ New best" "$logfile" | tail -1 | grep -oP '\d+\.\d+(?=%)' || echo "N/A")
        echo "$section | $name | seed=$seed | ${minutes}m${seconds}s | best_mIoU=$BEST_MIOU" >> "$RESULTS_FILE"
    else
        echo "  ✗ [$section] $name seed=$seed FAILED (exit code $exit_code)"
        echo "$section | $name | seed=$seed | FAILED" >> "$RESULTS_FILE"
    fi
    echo ""
}

# =============================================================================
# Section A — Main SATG Experiments
# =============================================================================
echo "============================================="
echo "  Section A: SATG Main Experiments"
echo "============================================="
echo ""

# --- SATG Hard --------------------------------------------------------------
echo "--- SATG Hard Rejection ---"
for seed in 42 1337 2024; do
    run_experiment "A" "satg_hard" "configs/satg_hard.yaml" "" "$seed"
done

# --- SATG Soft-Weight -------------------------------------------------------
echo "--- SATG Soft-Weight ---"
for seed in 42 1337 2024; do
    run_experiment "A" "satg_soft_weight" "configs/satg_soft_weight.yaml" "" "$seed"
done

# --- SATG Soft-Label (5 seeds) ----------------------------------------------
echo "--- SATG Soft-Label ---"
for seed in 42 1337 2024 7 99; do
    run_experiment "A" "satg_soft_label" "configs/satg_soft_label.yaml" "" "$seed"
done

# =============================================================================
# Section B — Ablation A: Prior Component Isolation
# =============================================================================
echo ""
echo "============================================="
echo "  Section B: Ablation A — Prior Component"
echo "============================================="
echo ""

run_experiment "B" "ablation_a_edge_only" "configs/satg_hard.yaml" \
    "structural_prior.edge_weight=1.0 structural_prior.variance_weight=0.0 structural_prior.entropy_weight=0.0 structural_prior.cornerness_weight=0.0" \
    42

run_experiment "B" "ablation_a_variance_only" "configs/satg_hard.yaml" \
    "structural_prior.edge_weight=0.0 structural_prior.variance_weight=1.0 structural_prior.entropy_weight=0.0 structural_prior.cornerness_weight=0.0" \
    42

run_experiment "B" "ablation_a_entropy_only" "configs/satg_hard.yaml" \
    "structural_prior.edge_weight=0.0 structural_prior.variance_weight=0.0 structural_prior.entropy_weight=1.0 structural_prior.cornerness_weight=0.0" \
    42

run_experiment "B" "ablation_a_cornerness_only" "configs/satg_hard.yaml" \
    "structural_prior.edge_weight=0.0 structural_prior.variance_weight=0.0 structural_prior.entropy_weight=0.0 structural_prior.cornerness_weight=1.0" \
    42

# =============================================================================
# Section C — Ablation B: tau_conf Sweep (tau_struct=0.60 fixed)
# =============================================================================
echo ""
echo "============================================="
echo "  Section C: Ablation B — tau_conf Sweep"
echo "============================================="
echo ""

run_experiment "C" "ablation_b_tauconf_0.80" "configs/satg_hard.yaml" \
    "trust_gate.tau_conf=0.80" \
    42

# tau_conf=0.90 = default satg_hard — skip (already run in Section A)

run_experiment "C" "ablation_b_tauconf_0.95" "configs/satg_hard.yaml" \
    "trust_gate.tau_conf=0.95" \
    42

# =============================================================================
# Section D — Ablation C: tau_struct Sweep (tau_conf=0.90 fixed)
# =============================================================================
echo ""
echo "============================================="
echo "  Section D: Ablation C — tau_struct Sweep"
echo "============================================="
echo ""

run_experiment "D" "ablation_c_taustruct_0.40" "configs/satg_hard.yaml" \
    "trust_gate.tau_struct=0.40" \
    42

# tau_struct=0.60 = default satg_hard — skip (already run in Section A)

run_experiment "D" "ablation_c_taustruct_0.70" "configs/satg_hard.yaml" \
    "trust_gate.tau_struct=0.70" \
    42

# =============================================================================
# Section E — Ablation E: Kernel Size & Sigma
# =============================================================================
echo ""
echo "============================================="
echo "  Section E: Ablation E — Kernel Parameters"
echo "============================================="
echo ""

# 3 sigmas × 2 window sizes = 6 configs (window=15 default omitted since it
# matches the default satg_hard run from Section A)

for sigma in 0.5 1.0 2.0; do
    for ksize in 7 31; do
        # Skip the (2.0, 15) combo since that's the default — already run
        if [ "$sigma" = "2.0" ] && [ "$ksize" = "15" ]; then
            continue
        fi
        tag="sigma${sigma}_ksize${ksize}"
        run_experiment "E" "ablation_e_${tag}" "configs/satg_hard.yaml" \
            "structural_prior.gaussian_sigma=${sigma} structural_prior.cornerness_sigma=${sigma} structural_prior.edge_kernel_size=${ksize} structural_prior.variance_kernel_size=${ksize} structural_prior.cornerness_kernel_size=${ksize}" \
            42
    done
done

# =============================================================================
# Section F — Ablation G: Soft-Label k Sensitivity
# =============================================================================
echo ""
echo "============================================="
echo "  Section F: Ablation G — Soft-Label k"
echo "============================================="
echo ""

run_experiment "F" "ablation_g_k2.0" "configs/satg_soft_label.yaml" \
    "trust_gate.soft_label_k=2.0" \
    42

run_experiment "F" "ablation_g_k6.0" "configs/satg_soft_label.yaml" \
    "trust_gate.soft_label_k=6.0" \
    42

# k=4.0 = default soft_label — skip (already run in Section A)

# =============================================================================
# Section G — Tau Grid Search (tau_conf × tau_struct)
# =============================================================================
echo ""
echo "============================================="
echo "  Section G: tau_conf × tau_struct Grid"
echo "============================================="
echo ""

# Sweep tau_conf ∈ {0.85, 0.90, 0.95} × tau_struct ∈ {0.50, 0.60, 0.70}
# tau_conf=0.90 × tau_struct=0.60 = default — skip (already run in Section A)
for tau_conf in 0.85 0.90 0.95; do
    for tau_struct in 0.50 0.60 0.70; do
        if [ "$tau_conf" = "0.90" ] && [ "$tau_struct" = "0.60" ]; then
            continue
        fi
        tag="grid_tc${tau_conf}_ts${tau_struct}"
        run_experiment "G" "${tag}" "configs/satg_hard.yaml" \
            "trust_gate.tau_conf=${tau_conf} trust_gate.tau_struct=${tau_struct}" \
            42
    done
done

# =============================================================================
# Summary
# =============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_HOURS=$((TOTAL_MINUTES / 60))
TOTAL_MINUTES_REMAINDER=$((TOTAL_MINUTES % 60))

# Count runs
SUCCESS_COUNT=$(grep -c "completed" "$RESULTS_FILE" 2>/dev/null || echo 0)
FAIL_COUNT=$(grep -c "FAILED" "$RESULTS_FILE" 2>/dev/null || echo 0)

echo ""
echo "============================================="
echo "  Phase 9 Complete"
echo "  Finished: $(date)"
echo "  Total duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES_REMAINDER}m"
echo "  Runs: $SUCCESS_COUNT succeeded, $FAIL_COUNT failed"
echo "============================================="
echo ""
echo "  Summary Table:"
echo "  --------------"
if [ -f "$RESULTS_FILE" ]; then
    echo ""
    printf "  %-8s %-30s %-8s %-12s %s\n" "Section" "Experiment" "Seed" "Duration" "Best mIoU"
    printf "  %-8s %-30s %-8s %-12s %s\n" "-------" "----------" "----" "--------" "---------"
    while IFS='|' read -r section name seed duration miou; do
        # Trim whitespace
        section=$(echo "$section" | xargs)
        name=$(echo "$name" | xargs)
        seed=$(echo "$seed" | xargs)
        duration=$(echo "$duration" | xargs)
        miou=$(echo "$miou" | xargs)
        printf "  %-8s %-30s %-8s %-12s %s\n" "$section" "$name" "$seed" "$duration" "$miou"
    done < "$RESULTS_FILE"
fi
echo ""
echo "  Logs: $LOG_DIR/phase9_*.log"
echo "  Checkpoints: checkpoints/"
echo ""

# Quick mIoU summary grouped by experiment
echo "Quick mIoU summary (grouped):"
echo "-----------------------------"
for exp_type in "satg_hard" "satg_soft_weight" "satg_soft_label"; do
    echo ""
    echo "  $exp_type:"
    for log in "$LOG_DIR/phase9_${exp_type}"*.log; do
        if [ -f "$log" ]; then
            NAME=$(basename "$log" .log)
            BEST=$(grep "★ New best" "$log" | tail -1 | grep -oP '\d+\.\d+(?=%)' || echo "N/A")
            echo "    $NAME: $BEST"
        fi
    done
done
