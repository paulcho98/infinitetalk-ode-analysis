#!/bin/bash
# Stage 2 for the Euler-jump factorial (runs AFTER scripts/run_infinitetalk_euler_jump.sh).
#
# The Euler-jump driver emits trajectory-shaped dirs, so this is the same 2a/2b pipeline as
# run_stage2_infinitetalk.sh, just pointed at $OUT/euler_{step0}_{teacher}/ instead of the
# sequential trajectories. 7 cells, one per GPU.
#
# Usage:
#   run_stage2_euler_jump.sh euler_on_noaudio [gpu]   # ONE cell — validation
#   run_stage2_euler_jump.sh all                      # all 7 cells, one per GPU
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
REF=/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk
export INFINITETALK_ROOT="$REF/InfiniteTalk"
export METRICS_ROOT=/data/karlo-research_715/workspace/kinemaar/paul/eval_metrics
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$METRICS_ROOT:${PYTHONPATH:-}"
VENV=/data/karlo-research_715/workspace/kinemadae/projects/paul/videos/lip/.venvs/infinitetalk-ode
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python"

EULER_ROOT="${EULER_ROOT:-$REPO/ode_euler_jump_infinitetalk}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$REPO/ode_analysis_euler_jump}"
MASKCACHE="${MASKCACHE:-$REPO/ode_analysis_infinitetalk/_mouth_mask_cache}"
mkdir -p "$ANALYSIS_ROOT" "$MASKCACHE"

CELLS=(euler_on_on euler_on_noaudio euler_noaudio_on euler_noaudio_noaudio
       euler_on_nocfg euler_nocfg_on euler_nocfg_nocfg)

run_one() {   # $1 = cell name, $2 = gpu id
    local cell="$1" gpu="$2"
    local traj="$EULER_ROOT/$cell"
    local outp="$ANALYSIS_ROOT/$cell/perceptual_v2"
    local outg="$ANALYSIS_ROOT/$cell/trajectory"
    mkdir -p "$outp" "$outg"
    local log="$ANALYSIS_ROOT/$cell/stage2.log"
    echo "[$cell gpu$gpu] traj=$traj" | tee "$log"
    if [ ! -e "$traj" ]; then echo "[$cell] MISSING dir, skip" | tee -a "$log"; return 1; fi

    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/eval_ode_perceptual_v2_infinitetalk.py \
        --phase decode  --traj_dir "$traj" --output_dir "$outp"  >> "$log" 2>&1
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/eval_ode_perceptual_v2_infinitetalk.py \
        --phase metrics --traj_dir "$traj" --output_dir "$outp"  >> "$log" 2>&1
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/eval_ode_perceptual_v2_infinitetalk.py \
        --merge         --traj_dir "$traj" --output_dir "$outp"  >> "$log" 2>&1

    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/analyze_ode_trajectory_infinitetalk.py \
        --traj_dir "$traj" --output_dir "$outg" --gt_mode encode \
        --mask_source ref_decode --mouth_mask_cache "$MASKCACHE" >> "$log" 2>&1
    echo "[$cell] done -> $outp , $outg" | tee -a "$log"
}

if [ "${1:-}" = "all" ]; then
    gpu=0
    for c in "${CELLS[@]}"; do
        run_one "$c" "$gpu" &
        gpu=$((gpu + 1))
    done
    wait
    echo "=== Stage 2 done for all Euler-jump cells ==="
    echo "Next: figures"
    echo "  $PY scripts/plot_euler_jump_factorial.py \\"
    echo "      --euler_analysis_root $ANALYSIS_ROOT \\"
    echo "      --sequential_analysis_root $REPO/ode_analysis_infinitetalk \\"
    echo "      --output_dir results/figures/euler_jump"
else
    run_one "${1:?usage: run_stage2_euler_jump.sh <cell>|all}" "${2:-0}"
fi
