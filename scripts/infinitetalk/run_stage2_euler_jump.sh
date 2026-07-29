#!/bin/bash
# Stage 2 for the Euler-jump factorial (runs AFTER scripts/infinitetalk/run_infinitetalk_euler_jump.sh).
#
# The Euler-jump driver emits trajectory-shaped dirs, so 2a runs on them unchanged, just pointed at
# $EULER_ROOT/euler_{step0}_{teacher}/ instead of the sequential trajectories. 7 cells, one per GPU.
#
# WHAT RUNS PER CELL:
#   1. straightness  ‖x0_euler − x0_seq‖ per step  — THE curvature number. Seconds, no GPU/VAE/GT.
#   2. Stage 2a      decode → perceptual/lip/sync metrics vs GT.
#   3. Stage 2b      latent geometry vs the GT latent — SKIPPED BY DEFAULT (RUN_2B=1 to enable).
#
# 2b is off by default because it is the expensive leg (VAE-encodes GT per sample) and measures
# "how close to GT", which for this experiment is secondary to "how far did the jump miss" — that
# is what step 1 gives you, far more cheaply.
#
# Usage:
#   run_stage2_euler_jump.sh euler_on_noaudio [gpu]   # ONE cell — validation
#   run_stage2_euler_jump.sh all                      # all 7 cells, one per GPU
#   RUN_2B=1 run_stage2_euler_jump.sh all             # ...including the GT-latent geometry
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
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

TRAJ_ROOT="${TRAJ_ROOT:-$REPO/ode_full_trajectories_infinitetalk}"   # sequential, for straightness
STRAIGHT_OUT="${STRAIGHT_OUT:-$REPO/results/infinitetalk/data}"
RUN_2B="${RUN_2B:-0}"

CELLS=(euler_on_on euler_on_noaudio euler_noaudio_on euler_noaudio_noaudio
       euler_on_nocfg euler_nocfg_on euler_nocfg_nocfg)

# cell -> the sequential trajectory to compare against (matches the cell's TEACHER leg, so the
# only difference is HOW the state was reached: one Euler jump vs the full sequential path).
seq_for_cell() {
    case "${1#euler_*_}" in
        on)      echo "$TRAJ_ROOT/infinitetalk_t5.0_a4.0" ;;
        noaudio) echo "$TRAJ_ROOT/infinitetalk_t5.0_a1.0" ;;
        nocfg)   echo "$TRAJ_ROOT/infinitetalk_t1.0_a1.0" ;;
        *)       echo "" ;;
    esac
}

run_one() {   # $1 = cell name, $2 = gpu id
    local cell="$1" gpu="$2"
    local traj="$EULER_ROOT/$cell"
    local outp="$ANALYSIS_ROOT/$cell/perceptual_v2"
    local outg="$ANALYSIS_ROOT/$cell/trajectory"
    mkdir -p "$outp" "$outg" "$STRAIGHT_OUT"
    local log="$ANALYSIS_ROOT/$cell/stage2.log"
    echo "[$cell gpu$gpu] traj=$traj" | tee "$log"
    if [ ! -e "$traj" ]; then echo "[$cell] MISSING dir, skip" | tee -a "$log"; return 1; fi

    # 1. straightness vs the sequential path at the same teacher CFG (cheap, CPU, no GT)
    local seq; seq="$(seq_for_cell "$cell")"
    if [ -n "$seq" ] && [ -d "$seq" ]; then
        "$PY" scripts/common/measure_euler_straightness.py \
            --euler_dir "$traj" --sequential_dir "$seq" \
            --mouth_mask_cache "$MASKCACHE" \
            --output_dir "$STRAIGHT_OUT" --tag "${cell#euler_}" >> "$log" 2>&1 \
            && echo "[$cell] straightness -> $STRAIGHT_OUT/straightness_${cell#euler_}.json" | tee -a "$log"
    else
        echo "[$cell] no sequential dir for straightness ($seq)" | tee -a "$log"
    fi

    # 2. Stage 2a — perceptual/lip/sync vs GT
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/infinitetalk/eval_ode_perceptual_v2_infinitetalk.py \
        --phase decode  --traj_dir "$traj" --output_dir "$outp"  >> "$log" 2>&1
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/infinitetalk/eval_ode_perceptual_v2_infinitetalk.py \
        --phase metrics --traj_dir "$traj" --output_dir "$outp"  >> "$log" 2>&1
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/infinitetalk/eval_ode_perceptual_v2_infinitetalk.py \
        --merge         --traj_dir "$traj" --output_dir "$outp"  >> "$log" 2>&1

    # 3. Stage 2b — GT-latent geometry (opt-in)
    if [ "$RUN_2B" = "1" ]; then
        CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/infinitetalk/analyze_ode_trajectory_infinitetalk.py \
            --traj_dir "$traj" --output_dir "$outg" --gt_mode encode \
            --mask_source ref_decode --mouth_mask_cache "$MASKCACHE" >> "$log" 2>&1
    else
        echo "[$cell] Stage 2b skipped (RUN_2B=1 to enable)" | tee -a "$log"
    fi
    echo "[$cell] done -> $outp" | tee -a "$log"
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
    echo "  $PY scripts/infinitetalk/plot_euler_jump_factorial.py \\"
    echo "      --euler_analysis_root $ANALYSIS_ROOT \\"
    echo "      --sequential_analysis_root $REPO/ode_analysis_infinitetalk \\"
    echo "      --output_dir results/infinitetalk/figures/euler_jump"
else
    run_one "${1:?usage: run_stage2_euler_jump.sh <cell>|all}" "${2:-0}"
fi
