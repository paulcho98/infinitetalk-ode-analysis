#!/bin/bash
# Generate InfiniteTalk ODE trajectories for the 7-config CFG sweep, JOB-sharded across 8 GPUs.
#
# vs run_infinitetalk_ode_sweep.sh (4-GPU, sample-sharded): this shards the 70 (sample x config)
# trajectories across 8 GPUs for balanced utilization (~3.5h vs ~5.5h), and precomputes the wav2vec
# audio cache once up front (CPU) so the 8 workers only READ it (no torch.save race per hash).
#
# 7 distinct configs (text:audio): 5:4, 5:1, 5:2, 5:6, 1:1(no-CFG 1-pass), 2.5:2, 7.5:6
# Output: $OUT/infinitetalk_t{T}_a{A}/<sample>/{step_NNN_xt.pt, step_NNN_x0.pt, ode_schedule.json, input_latents.pt}
set -euo pipefail

# ── paths on THIS machine ──
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
REF=/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk
export INFINITETALK_ROOT="$REF/InfiniteTalk"
export TORCHDYNAMO_DISABLE=1   # run eager: no python3.10 dev headers for Triton/inductor gcc compile
CKPT="$INFINITETALK_ROOT/weights/Wan2.1-I2V-14B-480P"          # base Wan2.1-I2V-14B-480P
ITALK="$REF/weights/InfiniteTalk/single/infinitetalk.safetensors"  # audio-condition weights
WAV2VEC="$REF/weights/chinese-wav2vec2-base"                    # audio encoder
PY=/data/karlo-research_715/workspace/kinemadae/projects/paul/videos/lip/.venvs/infinitetalk-ode/bin/python

OUT="${OUT:-$REPO/ode_full_trajectories_infinitetalk}"         # ~18GB dump (gitignored)
NAMES="$REPO/data/recon_sample_names.txt"
VID="$REPO/data/recon_clips"
AUD="$REPO/data/recon_clips"
CONFIGS="5.0:4.0,5.0:1.0,5.0:2.0,5.0:6.0,1.0:1.0,2.5:2.0,7.5:6.0"
STEPS=${1:-50}
NGPU=${NGPU:-8}

mkdir -p "$OUT"
cd "$REPO"
echo "REPO=$REPO"
echo "OUT=$OUT | configs=$CONFIGS | steps=$STEPS | ngpu=$NGPU"
echo "CKPT=$CKPT"
echo "ITALK=$ITALK"

COMMON=(--checkpoint_dir "$CKPT" --infinitetalk_dir "$ITALK" --wav2vec_dir "$WAV2VEC"
        --video_dir "$VID" --audio_dir "$AUD" --sample_names_file "$NAMES"
        --output_root "$OUT" --configs "$CONFIGS"
        --num_inference_steps "$STEPS" --shift 7.0 --size infinitetalk-480
        --frame_num 81 --seed 42 --max_samples 10)

# ── 1) precompute audio cache once (CPU-only pass) ──
echo "=== [precompute] wav2vec audio cache ==="
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 "$PY" scripts/infinitetalk/generate_infinitetalk_ode_pairs_full.py \
    "${COMMON[@]}" --precompute_audio_only 2>&1 | tee "$OUT/precompute_audio.log"

# ── 1.5) prewarm shared weights into the page cache (all 8 workers read the same ~100GB;
#         warm it once so they hit RAM instead of stampeding the NAS) ──
echo "=== [prewarm] reading weights into page cache ==="
cat "$CKPT"/*.safetensors "$CKPT"/*.pth "$ITALK" > /dev/null 2>&1 || true
echo "=== [prewarm] done ==="

# ── 2) launch NGPU job-sharded workers (small stagger so cache stays warm, GPUs ramp cleanly) ──
echo "=== [sweep] launching $NGPU job-sharded workers ==="
for SHARD in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$SHARD LOCAL_RANK=0 "$PY" scripts/infinitetalk/generate_infinitetalk_ode_pairs_full.py \
        "${COMMON[@]}" --shard_unit job --num_shards "$NGPU" --shard_id "$SHARD" --skip_existing \
        > "$OUT/gen_shard${SHARD}.log" 2>&1 &
    sleep 8
done
wait
echo "All $NGPU shards done. Trajectories in $OUT/infinitetalk_t*_a*/"
