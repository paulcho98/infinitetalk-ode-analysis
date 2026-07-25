#!/bin/bash
# Stage-2b: latent trajectory geometry (straightness/velocity/x0-vs-GT) for all 7 configs in parallel.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
REF=/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk
export INFINITETALK_ROOT="$REF/InfiniteTalk"
export METRICS_ROOT=/data/karlo-research_715/workspace/kinemaar/paul/eval_metrics
export TORCHDYNAMO_DISABLE=1
VENV=/data/karlo-research_715/workspace/kinemadae/projects/paul/videos/lip/.venvs/infinitetalk-ode
export PATH="$VENV/bin:$PATH"; PY="$VENV/bin/python"

TRAJ_ROOT="$REPO/ode_full_trajectories_infinitetalk"
ANALYSIS_ROOT="$REPO/ode_analysis_infinitetalk"
CONFIGS=("5.0_4.0" "5.0_1.0" "5.0_2.0" "5.0_6.0" "1.0_1.0" "2.5_2.0" "7.5_6.0")

gpu=0
for ta in "${CONFIGS[@]}"; do
    T="${ta%_*}" A="${ta#*_}"
    traj="$TRAJ_ROOT/infinitetalk_t${T}_a${A}"
    outg="$ANALYSIS_ROOT/infinitetalk_t${T}_a${A}/trajectory"
    cache="$outg/mask_cache"; mkdir -p "$outg" "$cache"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/analyze_ode_trajectory_infinitetalk.py \
        --traj_dir "$traj" --output_dir "$outg" --gt_mode encode \
        --mask_source ref_decode --mouth_mask_cache "$cache" \
        > "$outg/analyze.log" 2>&1 &
    gpu=$(((gpu + 1) % 8))
done
wait
echo "=== Stage-2b done for all configs ==="
for ta in "${CONFIGS[@]}"; do
    T="${ta%_*}" A="${ta#*_}"
    outg="$ANALYSIS_ROOT/infinitetalk_t${T}_a${A}/trajectory"
    echo "$ta: $(ls "$outg"/*.png "$outg"/*.json "$outg"/*.csv 2>/dev/null | wc -l) output files"
done
