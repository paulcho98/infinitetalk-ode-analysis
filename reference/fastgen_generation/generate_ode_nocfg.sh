#!/bin/bash
# Generate 50-step ODE trajectory with NO CFG (guidance_scale=1.0).
# Only 1 forward pass per step (no unconditional branch), so ~2x faster.
set -euo pipefail

MODEL_SIZE="${1:-14B}"

PRETRAINED="/home/work/.local/OmniAvatar/pretrained_models"
DATA_DIR="/home/work/stableavatar_data/v2v_validation_data/recon"
MASK_PATH="/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png"
NEG_TEXT_EMB="/home/work/stableavatar_data/neg_text_emb.pt"
OUTPUT_ROOT="/home/work/.local/ode_full_trajectories"
FASTGEN_ROOT="/home/work/.local/hyunbin/FastGen"

if [ "$MODEL_SIZE" = "14B" ]; then
    BASE_PATHS="${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors"
    BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors"
    CKPT="/home/work/output_omniavatar_v2v_phase2/step-10500.pt"
    OUTPUT_DIR="${OUTPUT_ROOT}/14B_nocfg"
elif [ "$MODEL_SIZE" = "1.3B" ]; then
    BASE_PATHS="${PRETRAINED}/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
    CKPT="/home/work/output_omniavatar_v2v_1.3B_phase2/step-19500.pt"
    OUTPUT_DIR="${OUTPUT_ROOT}/1.3B_nocfg"
else
    echo "ERROR: model_size must be 14B or 1.3B, got: $MODEL_SIZE"
    exit 1
fi

echo "Model:  ${MODEL_SIZE} (NO CFG, guidance_scale=1.0)"
echo "Ckpt:   ${CKPT}"
echo "Output: ${OUTPUT_DIR}"

cd "$FASTGEN_ROOT"

CUDA_VISIBLE_DEVICES=0 /home/work/.local/miniconda3/envs/fastgen/bin/python scripts/generate_omniavatar_ode_pairs_full.py \
    --model_size "$MODEL_SIZE" \
    --in_dim 65 \
    --base_model_paths "$BASE_PATHS" \
    --omniavatar_ckpt_path "$CKPT" \
    --data_dir "$DATA_DIR" \
    --latentsync_mask_path "$MASK_PATH" \
    --neg_text_emb_path "$NEG_TEXT_EMB" \
    --output_dir "$OUTPUT_DIR" \
    --num_inference_steps 50 \
    --guidance_scale 1.0 \
    --shift 5.0 \
    --max_samples 10 \
    --skip_existing
