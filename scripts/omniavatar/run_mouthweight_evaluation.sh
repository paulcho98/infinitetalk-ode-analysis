#!/bin/bash
# Evaluate MouthWeight ODE trajectories + Euler variants.
# Runs sequentially on 1 GPU. Designed to run on a SEPARATE machine
# from training to avoid CPU contention.
#
# Metrics computed: LPIPS, SyncNet (Sync-C, Sync-D) only.
# Skips: pixel_mse, ssim, lmd, sharpness (saves ~33% eval time).
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/omniavatar/run_mouthweight_evaluation.sh
#
# Total wall time: ~22h (6 full-trajectory evals × ~3.7h each)
#                + ~10 min (2 endpoint-only evals for Pareto)

set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$REPO_ROOT/configs/machine.env" ] && source "$REPO_ROOT/configs/machine.env"
PY_OMNI="${PY_OMNI:-/home/work/.local/miniconda3/envs/omniavatar/bin/python}"

PY="$PY_OMNI"
SCRIPT="$REPO_ROOT/scripts/omniavatar/eval_ode_perceptual_v2.py"
VAE_PATH="${VAE_PATH:-/home/work/.local/OmniAvatar/pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth}"
MASK_PATH="${MASK_PATH:-$REPO_ROOT/data/mask.png}"

TRAJ_ROOT="${ODE_TRAJ_ROOT_OMNI:-/home/work/.local/ode_full_trajectories}/14B_mouthweight"
ANALYSIS_ROOT="${ODE_ANALYSIS_ROOT_OMNI:-/home/work/.local/ode_analysis}/14B_mouthweight"

STATE_DIR="$ANALYSIS_ROOT/_eval_state"
mkdir -p "$STATE_DIR"

run_full_eval() {
    local idx="$1"; local label="$2"; local traj_dir="$3"; local out_dir="$4"
    local marker="$STATE_DIR/${idx}.done"
    if [ -f "$marker" ]; then
        echo "[$(date '+%F %T')] [$idx] SKIP (already done): $label"
        return 0
    fi
    echo ""
    echo "=========================================="
    echo "[$(date '+%F %T')] [$idx] START: $label"
    echo "  traj_dir=$traj_dir"
    echo "  output_dir=$out_dir"
    echo "=========================================="
    mkdir -p "$out_dir"
    local t0=$(date +%s)

    # Phase 1: Decode
    echo "[$(date '+%F %T')] Decode..."
    "$PY" "$SCRIPT" --phase decode --vae_type wan \
        --traj_dir "$traj_dir" \
        --vae_path "$VAE_PATH" \
        --output_dir "$out_dir"

    # Phase 2: Metrics (LPIPS + SSIM + SyncNet only — skip pixel_mse, lmd, sharpness)
    echo "[$(date '+%F %T')] Metrics (lpips + ssim + syncnet)..."
    "$PY" "$SCRIPT" --phase metrics \
        --traj_dir "$traj_dir" \
        --mask_path "$MASK_PATH" \
        --output_dir "$out_dir" \
        --sync_min_track 50 \
        --skip_metrics pixel_mse,lmd,sharpness

    # Phase 3: Merge
    echo "[$(date '+%F %T')] Merge..."
    "$PY" "$SCRIPT" --merge \
        --traj_dir "$traj_dir" \
        --output_dir "$out_dir"

    local t1=$(date +%s)
    touch "$marker"
    echo "[$(date '+%F %T')] [$idx] DONE in $((t1 - t0))s: $label"
}

run_endpoint_eval() {
    local idx="$1"; local label="$2"; local traj_dir="$3"; local out_dir="$4"
    local marker="$STATE_DIR/${idx}.done"
    if [ -f "$marker" ]; then
        echo "[$(date '+%F %T')] [$idx] SKIP (already done): $label"
        return 0
    fi
    echo ""
    echo "=========================================="
    echo "[$(date '+%F %T')] [$idx] START (endpoint only): $label"
    echo "=========================================="
    mkdir -p "$out_dir"
    local t0=$(date +%s)

    # Decode all steps (required by script), then metrics, then merge
    echo "[$(date '+%F %T')] Decode..."
    "$PY" "$SCRIPT" --phase decode --vae_type wan \
        --traj_dir "$traj_dir" \
        --vae_path "$VAE_PATH" \
        --output_dir "$out_dir"

    echo "[$(date '+%F %T')] Metrics..."
    "$PY" "$SCRIPT" --phase metrics \
        --traj_dir "$traj_dir" \
        --mask_path "$MASK_PATH" \
        --output_dir "$out_dir" \
        --sync_min_track 50

    echo "[$(date '+%F %T')] Merge..."
    "$PY" "$SCRIPT" --merge \
        --traj_dir "$traj_dir" \
        --output_dir "$out_dir"

    local t1=$(date +%s)
    touch "$marker"
    echo "[$(date '+%F %T')] [$idx] DONE in $((t1 - t0))s: $label"
}

echo "============================================"
echo "MouthWeight ODE Evaluation"
echo "Sequential on GPU ${CUDA_VISIBLE_DEVICES:-0}"
echo "Metrics: LPIPS + SyncNet (Sync-C, Sync-D)"
echo "============================================"
echo ""

# ── Fig 3: Full per-step eval (6 variants) ──

# Trajectories
run_full_eval 01_traj_cfg45 "trajectory CFG=4.5" \
    "$TRAJ_ROOT/cfg4.5" "$ANALYSIS_ROOT/perceptual_v2"

run_full_eval 02_traj_nocfg "trajectory noCFG" \
    "$TRAJ_ROOT/nocfg" "$ANALYSIS_ROOT/trajectory_nocfg"

# Euler factorial
run_full_eval 03_euler_cfg45_cfg45 "Euler CFG→CFG" \
    "$TRAJ_ROOT/cfg4.5" "$ANALYSIS_ROOT/euler_cfg45_cfg45"

run_full_eval 04_euler_nocfg_cfg45 "Euler noCFG→CFG" \
    "$TRAJ_ROOT/cfg4.5" "$ANALYSIS_ROOT/euler_nocfg_cfg45"

run_full_eval 05_euler_nocfg_nocfg "Euler noCFG→noCFG" \
    "$TRAJ_ROOT/cfg4.5" "$ANALYSIS_ROOT/euler_nocfg_nocfg"

run_full_eval 06_euler_cfg45_nocfg "Euler CFG→noCFG" \
    "$TRAJ_ROOT/cfg4.5" "$ANALYSIS_ROOT/euler_cfg45_nocfg"

# ── Fig 4: Endpoint eval (2 additional trajectories) ──
# Note: CFG=4.5, noCFG, and noCFG→CFG endpoints come from runs 01, 02, 04 above.

run_endpoint_eval 07_traj_cfg30 "trajectory CFG=3.0 (endpoint)" \
    "$TRAJ_ROOT/cfg3.0" "$ANALYSIS_ROOT/trajectory_cfg3.0"

run_endpoint_eval 08_traj_cfg60 "trajectory CFG=6.0 (endpoint)" \
    "$TRAJ_ROOT/cfg6.0" "$ANALYSIS_ROOT/trajectory_cfg6.0"

echo ""
echo "============================================"
echo "[$(date '+%F %T')] ALL EVALUATION COMPLETE"
echo ""
echo "CSVs for Figure 3:"
for V in perceptual_v2 trajectory_nocfg euler_cfg45_cfg45 euler_nocfg_cfg45 euler_nocfg_nocfg euler_cfg45_nocfg; do
    F="$ANALYSIS_ROOT/$V/metrics.csv"
    [ -f "$F" ] && echo "  $V: $(wc -l < "$F") rows" || echo "  $V: MISSING"
done
echo ""
echo "CSVs for Figure 4 (endpoint rows at step=49):"
for V in perceptual_v2 trajectory_nocfg trajectory_cfg3.0 trajectory_cfg6.0 euler_nocfg_cfg45; do
    F="$ANALYSIS_ROOT/$V/metrics.csv"
    [ -f "$F" ] && echo "  $V: $(wc -l < "$F") rows" || echo "  $V: MISSING"
done
echo "============================================"
