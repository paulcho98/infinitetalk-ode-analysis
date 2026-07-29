#!/bin/bash
# Fast Stage-2a METRICS phase: shard by sample (num_shards per config) across 8 GPUs.
# Decode is already cached on disk; this only recomputes metrics (the syncnet-bound part) in parallel.
# Writes metrics_shard{i}.csv per config, then merges to metrics.csv.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"                                  # CWD=repo so scripts/ + sfd symlink resolve
REF=/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk
export INFINITETALK_ROOT="$REF/InfiniteTalk"
export METRICS_ROOT=/data/karlo-research_715/workspace/kinemaar/paul/eval_metrics
export PYTHONPATH="$METRICS_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
VENV=/data/karlo-research_715/workspace/kinemadae/projects/paul/videos/lip/.venvs/infinitetalk-ode
export PATH="$VENV/bin:$PATH"; PY="$VENV/bin/python"

TRAJ_ROOT="$REPO/ode_full_trajectories_infinitetalk"
ANALYSIS_ROOT="$REPO/ode_analysis_infinitetalk"
CONFIGS=("5.0_4.0" "5.0_1.0" "5.0_2.0" "5.0_6.0" "1.0_1.0" "2.5_2.0" "7.5_6.0")
NS=${NS:-10}                                # shards per config (10 = one sample each)
NGPU=8

gpu=0
declare -a PIDS=()
for ta in "${CONFIGS[@]}"; do
    T="${ta%_*}" A="${ta#*_}"
    traj="$TRAJ_ROOT/infinitetalk_t${T}_a${A}"
    outp="$ANALYSIS_ROOT/infinitetalk_t${T}_a${A}/perceptual_v2"
    mkdir -p "$outp"
    rm -f "$outp"/metrics_shard*.csv        # clear stale shards
    for i in $(seq 0 $((NS - 1))); do
        CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/infinitetalk/eval_ode_perceptual_v2_infinitetalk.py \
            --phase metrics --traj_dir "$traj" --output_dir "$outp" \
            --shard_id "$i" --num_shards "$NS" \
            > "$outp/metrics_shard${i}.log" 2>&1 &
        PIDS+=($!)
        gpu=$(((gpu + 1) % NGPU))
        sleep 1
    done
done
echo "launched ${#PIDS[@]} metric shards across $NGPU GPUs; waiting ..."
wait
echo "=== all shards done; merging per config ==="
for ta in "${CONFIGS[@]}"; do
    T="${ta%_*}" A="${ta#*_}"
    traj="$TRAJ_ROOT/infinitetalk_t${T}_a${A}"
    outp="$ANALYSIS_ROOT/infinitetalk_t${T}_a${A}/perceptual_v2"
    "$PY" scripts/infinitetalk/eval_ode_perceptual_v2_infinitetalk.py \
        --merge --traj_dir "$traj" --output_dir "$outp" > "$outp/merge.log" 2>&1
    echo "merged $ta: $(wc -l < "$outp/metrics.csv" 2>/dev/null) rows"
done
echo "=== Stage-2a metrics complete ==="
