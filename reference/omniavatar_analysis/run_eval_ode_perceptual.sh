#!/bin/bash
# Run perceptual ODE evaluation on 4 GPUs in parallel, then merge + plot.
set -euo pipefail

PYTHON=/home/work/.local/miniconda3/envs/omniavatar/bin/python
SCRIPT=scripts/eval_ode_perceptual.py
TRAJ_DIR=/home/work/ode_full_trajectories/14B
VAE_PATH=pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth
MASK_PATH=/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png
OUTPUT_DIR=/home/work/ode_analysis/14B/perceptual
NUM_SHARDS=4

cd /home/work/.local/OmniAvatar

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD $PYTHON $SCRIPT \
        --traj_dir "$TRAJ_DIR" \
        --vae_path "$VAE_PATH" \
        --mask_path "$MASK_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --shard_id $SHARD \
        --num_shards $NUM_SHARDS &
done

echo "Waiting for all shards..."
wait
echo "All shards done. Merging..."

$PYTHON $SCRIPT --merge \
    --output_dir "$OUTPUT_DIR" \
    --traj_dir "$TRAJ_DIR"

echo "Done! Results in $OUTPUT_DIR"
