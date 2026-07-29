#!/bin/bash
# Generate InfiniteTalk ODE trajectories for the 7-config CFG sweep, samples sharded across 4 GPUs.
#
# 7 distinct configs (text:audio):
#   Family 1 (text fixed=5): 5:1  5:2  5:4  5:6
#   Family 2 (paired):       1:1  2.5:2  5:4(shared)  7.5:6
#   -> 5:4, 5:1, 5:2, 5:6, 1:1, 2.5:2, 7.5:6   (1:1 = no-CFG baseline, 1 forward pass)
#
# Output: $OUT/infinitetalk_t{T}_a{A}/<sample>/{step_NNN_xt.pt, step_NNN_x0.pt, ode_schedule.json, input_latents.pt}
set -euo pipefail

IT=/home/work/.local/InfiniteTalk/weights   # EDIT: InfiniteTalk weights dir on this machine
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repo root (holds scripts/, data/)
PY=/home/work/.local/miniconda3/envs/infinitetalk/bin/python   # EDIT: infinitetalk env python
OUT=/home/work/.local/ode_full_trajectories_infinitetalk
NAMES="$REPO/data/recon_sample_names.txt"
VID="$REPO/data/recon_clips"    # bundled recon clips (<hash>.mp4)
AUD="$REPO/data/recon_clips"    # bundled recon clips (<hash>.wav)
CONFIGS="5.0:4.0,5.0:1.0,5.0:2.0,5.0:6.0,1.0:1.0,2.5:2.0,7.5:6.0"
STEPS=${1:-50}

mkdir -p "$OUT"
cd "$REPO"
echo "Configs: $CONFIGS | steps=$STEPS | out=$OUT"

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD LOCAL_RANK=0 $PY scripts/infinitetalk/generate_infinitetalk_ode_pairs_full.py \
        --checkpoint_dir $IT/Wan2.1-I2V-14B-480P \
        --infinitetalk_dir $IT/InfiniteTalk/single/infinitetalk.safetensors \
        --wav2vec_dir $IT/chinese-wav2vec2-base \
        --video_dir "$VID" --audio_dir "$AUD" --sample_names_file "$NAMES" \
        --output_root "$OUT" --configs "$CONFIGS" \
        --num_inference_steps $STEPS --shift 7.0 --size infinitetalk-480 \
        --frame_num 81 --seed 42 \
        --max_samples 10 --num_shards 4 --shard_id $SHARD --skip_existing \
        > "$OUT/gen_shard${SHARD}.log" 2>&1 &
done
wait
echo "All shards done. Trajectories in $OUT/infinitetalk_t*_a*/"
