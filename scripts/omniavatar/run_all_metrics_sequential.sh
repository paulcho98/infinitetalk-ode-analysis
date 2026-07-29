#!/bin/bash
# Sequential metrics run on a single GPU (no sharding, one job at a time).
# Processes all 11 trajectories/variants: 3 trajectories + 4 OmniAvatar + 4 LatentSync variants.
# Estimated total wall time: ~36-40 hrs on 1 GPU.
#
# Usage:  bash scripts/omniavatar/run_all_metrics_sequential.sh
# GPU is set via CUDA_VISIBLE_DEVICES env var (default: 3).

set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$REPO_ROOT/configs/machine.env" ] && source "$REPO_ROOT/configs/machine.env"
PY_OMNI="${PY_OMNI:-/home/work/.local/miniconda3/envs/omniavatar/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

PY="$PY_OMNI"
MASK_PATH="${MASK_PATH:-$REPO_ROOT/data/mask.png}"
SCRIPT="$REPO_ROOT/scripts/omniavatar/eval_ode_perceptual_v2.py"

STATE_DIR="${ODE_ANALYSIS_ROOT_OMNI:-/home/work/.local/ode_analysis}/_seq_state"
mkdir -p "$STATE_DIR"

run_job() {
    local idx="$1"; local label="$2"; local traj="$3"; local out="$4"
    local marker="$STATE_DIR/${idx}_${label// /_}.done"
    if [ -f "$marker" ]; then
        echo "[$(date '+%F %T')] [$idx] SKIP (already done): $label"
        return 0
    fi
    echo ""
    echo "=========================================="
    echo "[$(date '+%F %T')] [$idx] START: $label"
    echo "  traj_dir=$traj"
    echo "  output_dir=$out"
    echo "=========================================="
    local t0=$(date +%s)
    "$PY" "$SCRIPT" --phase metrics --traj_dir "$traj" --mask_path "$MASK_PATH" --output_dir "$out"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date '+%F %T')] [$idx] FAIL metrics phase (rc=$rc): $label"
        return $rc
    fi
    "$PY" "$SCRIPT" --merge --traj_dir "$traj" --output_dir "$out"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date '+%F %T')] [$idx] FAIL merge phase (rc=$rc): $label"
        return $rc
    fi
    local t1=$(date +%s)
    touch "$marker"
    echo "[$(date '+%F %T')] [$idx] DONE in $((t1 - t0))s: $label"
}

TRAJ_OMNI="${ODE_TRAJ_ROOT_OMNI:-/home/work/.local/ode_full_trajectories}/14B_audio_only_cfg"
TRAJ_LS="${ODE_TRAJ_ROOT_OMNI:-/home/work/.local/ode_full_trajectories}/latentsync_1.6"
TRAJ_LS_NOCFG="${ODE_TRAJ_ROOT_OMNI:-/home/work/.local/ode_full_trajectories}/latentsync_1.6_nocfg"
OUT_OMNI="${ODE_ANALYSIS_ROOT_OMNI:-/home/work/.local/ode_analysis}/14B_audio_only_cfg"
OUT_LS="${ODE_ANALYSIS_ROOT_OMNI:-/home/work/.local/ode_analysis}/latentsync_1.6"

# --- Phase 3: trajectory metrics ---
run_job 01 "omni audio-only CFG trajectory" "$TRAJ_OMNI" "$OUT_OMNI/perceptual_v2"
run_job 02 "latentsync CFG=1.5 trajectory" "$TRAJ_LS" "$OUT_LS/perceptual_v2"
run_job 03 "latentsync noCFG trajectory" "$TRAJ_LS_NOCFG" "$OUT_LS/trajectory_nocfg"

# --- Phase 5: OmniAvatar variants ---
for V in fresh_noise euler_cfg45_cfg45 euler_nocfg_cfg45 euler_nocfg_nocfg; do
    run_job "04_$V" "omni $V" "$TRAJ_OMNI" "$OUT_OMNI/$V"
done

# --- Phase 5: LatentSync variants ---
for V in fresh_noise euler_cfg15_cfg15 euler_nocfg_cfg15 euler_nocfg_nocfg; do
    run_job "05_$V" "latentsync $V" "$TRAJ_LS" "$OUT_LS/$V"
done

echo ""
echo "=========================================="
echo "[$(date '+%F %T')] ALL METRICS DONE"
echo "=========================================="
