#!/bin/bash
# Stage 2 orchestration for the InfiniteTalk ODE analysis (runs AFTER the Stage-1 sweep).
# Single venv has WanVAE + dlib + lpips + syncnet, so decode(Phase1) + metrics(Phase2) run together.
#
# Per config: 2a decode -> 2a metrics -> 2a merge -> 2b latent-trajectory analysis.
# Usage:
#   run_stage2_infinitetalk.sh <t>_<a>          # ONE config on GPU 0 (e.g. 5.0_4.0) — validation
#   run_stage2_infinitetalk.sh all              # all 7 configs, one per GPU (0..6), in parallel
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
REF=/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk
export INFINITETALK_ROOT="$REF/InfiniteTalk"
export METRICS_ROOT=/data/karlo-research_715/workspace/kinemaar/paul/eval_metrics
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$METRICS_ROOT:${PYTHONPATH:-}"     # so `from eval.syncnet import SyncNetEval` resolves
VENV=/data/karlo-research_715/workspace/kinemadae/projects/paul/videos/lip/.venvs/infinitetalk-ode
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python"

TRAJ_ROOT="${TRAJ_ROOT:-$REPO/ode_full_trajectories_infinitetalk}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$REPO/ode_analysis_infinitetalk}"
MASKCACHE="$ANALYSIS_ROOT/_mouth_mask_cache"
mkdir -p "$ANALYSIS_ROOT" "$MASKCACHE"

CONFIGS=("5.0_4.0" "5.0_1.0" "5.0_2.0" "5.0_6.0" "1.0_1.0" "2.5_2.0" "7.5_6.0")

run_one() {   # $1 = "T_A", $2 = gpu id
    local ta="$1" gpu="$2"
    local T="${ta%_*}" A="${ta#*_}"
    local traj="$TRAJ_ROOT/infinitetalk_t${T}_a${A}"
    # dir naming matches what plot_cfg_grid / plot_ode_curves expect: infinitetalk_t{T}_a{A}/perceptual_v2
    local outp="$ANALYSIS_ROOT/infinitetalk_t${T}_a${A}/perceptual_v2"
    local outg="$ANALYSIS_ROOT/infinitetalk_t${T}_a${A}/trajectory"
    mkdir -p "$outp" "$outg"
    local log="$ANALYSIS_ROOT/infinitetalk_t${T}_a${A}/stage2.log"
    echo "[cfg $ta gpu$gpu] traj=$traj" | tee "$log"
    if [ ! -e "$traj" ]; then echo "[cfg $ta] MISSING traj dir, skip" | tee -a "$log"; return 1; fi

    # 2a: decode -> metrics -> merge  (single-GPU, all 10 samples)
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/infinitetalk/eval_ode_perceptual_v2_infinitetalk.py \
        --phase decode  --traj_dir "$traj" --output_dir "$outp"  >> "$log" 2>&1
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/infinitetalk/eval_ode_perceptual_v2_infinitetalk.py \
        --phase metrics --traj_dir "$traj" --output_dir "$outp"  >> "$log" 2>&1
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/infinitetalk/eval_ode_perceptual_v2_infinitetalk.py \
        --merge         --traj_dir "$traj" --output_dir "$outp"  >> "$log" 2>&1

    # 2b: latent trajectory geometry
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/infinitetalk/analyze_ode_trajectory_infinitetalk.py \
        --traj_dir "$traj" --output_dir "$outg" --gt_mode encode \
        --mask_source ref_decode --mouth_mask_cache "$MASKCACHE" >> "$log" 2>&1
    echo "[cfg $ta] done -> $outp , $outg" | tee -a "$log"
}

if [ "${1:-}" = "all" ]; then
    gpu=0
    for ta in "${CONFIGS[@]}"; do
        run_one "$ta" "$gpu" &
        gpu=$((gpu + 1))
    done
    wait
    echo "=== Stage 2a/2b done for all configs. Now build figures: ==="
    echo "  $PY scripts/infinitetalk/plot_ode_curves_infinitetalk.py --analysis_root <perceptual parent> ..."
else
    run_one "${1:?usage: run_stage2_infinitetalk.sh <T_A>|all}" "${2:-0}"
fi
