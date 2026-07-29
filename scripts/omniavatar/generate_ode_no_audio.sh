#!/bin/bash
# Generate no-audio ODE trajectories for ablation analysis.
# Identical to run_ode_full_trajectory.sh but zeroes out audio embeddings.
#
# Usage:
#   bash scripts/omniavatar/generate_ode_no_audio.sh              # 14B (default)
#   bash scripts/omniavatar/generate_ode_no_audio.sh 1.3B          # 1.3B
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$REPO_ROOT/configs/machine.env" ] && source "$REPO_ROOT/configs/machine.env"
PY_FASTGEN="${PY_FASTGEN:-/home/work/.local/miniconda3/envs/fastgen/bin/python}"

MODEL_SIZE="${1:-14B}"

PRETRAINED="${WEIGHTS_ROOT:-/home/work/.local/OmniAvatar/pretrained_models}"
DATA_DIR="${RECON_DATA_DIR:-/home/work/stableavatar_data/v2v_validation_data/recon}"
MASK_PATH="${MASK_PATH:-$REPO_ROOT/data/mask.png}"
NEG_TEXT_EMB="${NEG_TEXT_EMB:-/home/work/stableavatar_data/neg_text_emb.pt}"
OUTPUT_ROOT="${ODE_TRAJ_ROOT_OMNI:-/home/work/.local/ode_full_trajectories}"

if [ "$MODEL_SIZE" = "14B" ]; then
    BASE_PATHS="${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors"
    CKPT="${TEACHER_CKPT:-/home/work/output_omniavatar_v2v_phase2/step-10500.pt}"
    OUTPUT_DIR="${OUTPUT_ROOT}/14B_no_audio"
elif [ "$MODEL_SIZE" = "1.3B" ]; then
    BASE_PATHS="${PRETRAINED}/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
    CKPT="/home/work/output_omniavatar_v2v_1.3B_phase2/step-19500.pt"  # 1.3B teacher, no env override
    OUTPUT_DIR="${OUTPUT_ROOT}/1.3B_no_audio"
else
    echo "ERROR: model_size must be 14B or 1.3B, got: $MODEL_SIZE"
    exit 1
fi

echo "Model:  ${MODEL_SIZE} (NO AUDIO)"
echo "Output: ${OUTPUT_DIR}"

cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES=0 "$PY_FASTGEN" "$REPO_ROOT/scripts/omniavatar/generate_omniavatar_ode_pairs_full.py" \
    --model_size "$MODEL_SIZE" \
    --in_dim 65 \
    --base_model_paths "$BASE_PATHS" \
    --omniavatar_ckpt_path "$CKPT" \
    --data_dir "$DATA_DIR" \
    --latentsync_mask_path "$MASK_PATH" \
    --neg_text_emb_path "$NEG_TEXT_EMB" \
    --output_dir "$OUTPUT_DIR" \
    --num_inference_steps 50 \
    --guidance_scale 4.5 \
    --shift 5.0 \
    --max_samples 10 \
    --skip_existing \
    --zero_audio
