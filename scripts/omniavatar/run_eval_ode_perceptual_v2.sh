#!/bin/bash
# Decode all ODE steps + compute perceptual/lip metrics on 4 GPUs.
# Phase 1: VAE decode → save videos (4 GPUs parallel)
# Phase 2: Compute metrics on saved videos (4 GPUs parallel)
# Phase 3: Merge CSVs + plot
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$REPO_ROOT/configs/machine.env" ] && source "$REPO_ROOT/configs/machine.env"
PY_OMNI="${PY_OMNI:-/home/work/.local/miniconda3/envs/omniavatar/bin/python}"

PYTHON="$PY_OMNI"
SCRIPT="$REPO_ROOT/scripts/omniavatar/eval_ode_perceptual_v2.py"
TRAJ_DIR="${ODE_TRAJ_ROOT_OMNI:-/home/work/.local/ode_full_trajectories}/14B"
VAE_PATH="${VAE_PATH:-/home/work/.local/OmniAvatar/pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth}"
MASK_PATH="${MASK_PATH:-$REPO_ROOT/data/mask.png}"
OUTPUT_DIR="${ODE_ANALYSIS_ROOT_OMNI:-/home/work/.local/ode_analysis}/14B/perceptual_v2"
NUM_SHARDS=4

echo "=== Phase 1: Decode (4 GPUs) ==="
for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD $PYTHON $SCRIPT \
        --phase decode \
        --traj_dir "$TRAJ_DIR" \
        --vae_path "$VAE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --shard_id $SHARD \
        --num_shards $NUM_SHARDS &
done
wait
echo "Phase 1 done."

echo ""
echo "=== Phase 2: Metrics (4 GPUs) ==="
for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD $PYTHON $SCRIPT \
        --phase metrics \
        --traj_dir "$TRAJ_DIR" \
        --mask_path "$MASK_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --shard_id $SHARD \
        --num_shards $NUM_SHARDS &
done
wait
echo "Phase 2 done."

echo ""
echo "=== Phase 3: Merge + Plot ==="
$PYTHON $SCRIPT --merge \
    --traj_dir "$TRAJ_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Done! Results in $OUTPUT_DIR"
echo "  Videos:  $OUTPUT_DIR/videos/"
echo "  CSV:     $OUTPUT_DIR/metrics.csv"
echo "  Plots:   $OUTPUT_DIR/reference_metrics.png, noref_metrics.png"
