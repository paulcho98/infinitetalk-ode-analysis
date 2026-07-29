#!/bin/bash
# Generate all 8 ODE trajectory/Euler runs for MouthWeight checkpoint.
# Two batches on 4 GPUs. Pause training before running.
#
# Usage: bash scripts/omniavatar/run_mouthweight_generation.sh
#
# Total wall time: ~1.5h (batch 1 ~40 min + batch 2 ~60 min)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$REPO_ROOT/configs/machine.env" ] && source "$REPO_ROOT/configs/machine.env"
PY_FASTGEN="${PY_FASTGEN:-/home/work/.local/miniconda3/envs/fastgen/bin/python}"
PY_OMNI="${PY_OMNI:-/home/work/.local/miniconda3/envs/omniavatar/bin/python}"

CKPT="${MOUTHWEIGHT_CKPT:-/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/step-6000.pt}"
PRETRAINED="${WEIGHTS_ROOT:-/home/work/.local/OmniAvatar/pretrained_models}"
BASE_PATHS="${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors"
BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors"
BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors"
BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors"
BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors"
BASE_PATHS="${BASE_PATHS},${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors"
DATA_DIR="${RECON_DATA_DIR:-/home/work/stableavatar_data/v2v_validation_data/recon}"
MASK_PATH="${MASK_PATH:-$REPO_ROOT/data/mask.png}"
NEG_TEXT_EMB="${NEG_TEXT_EMB:-/home/work/stableavatar_data/neg_text_emb.pt}"

TRAJ_ROOT="${ODE_TRAJ_ROOT_OMNI:-/home/work/.local/ode_full_trajectories}/14B_mouthweight"
ANALYSIS_ROOT="${ODE_ANALYSIS_ROOT_OMNI:-/home/work/.local/ode_analysis}/14B_mouthweight"
FASTGEN_SCRIPT="$REPO_ROOT/scripts/omniavatar/generate_omniavatar_ode_pairs_full.py"
EULER_SCRIPT="$REPO_ROOT/scripts/omniavatar/generate_single_step_predictions.py"

COMMON_TRAJ_FLAGS="--model_size 14B --in_dim 65 \
    --base_model_paths $BASE_PATHS \
    --omniavatar_ckpt_path $CKPT \
    --data_dir $DATA_DIR \
    --latentsync_mask_path $MASK_PATH \
    --neg_text_emb_path $NEG_TEXT_EMB \
    --num_inference_steps 50 --shift 5.0 \
    --max_samples 10 --skip_existing \
    --cfg_drop_text true"

echo "============================================"
echo "MouthWeight ODE Generation"
echo "Checkpoint: $CKPT"
echo "Output: $TRAJ_ROOT"
echo "============================================"
echo ""

# ── BATCH 1: 4 trajectories on 4 GPUs (~40 min) ──
echo "[$(date '+%F %T')] BATCH 1: 4 trajectories"

mkdir -p "$TRAJ_ROOT/cfg4.5" "$TRAJ_ROOT/nocfg" "$TRAJ_ROOT/cfg3.0" "$TRAJ_ROOT/cfg6.0"

CUDA_VISIBLE_DEVICES=0 "$PY_FASTGEN" "$FASTGEN_SCRIPT" \
    $COMMON_TRAJ_FLAGS --guidance_scale 4.5 \
    --output_dir "$TRAJ_ROOT/cfg4.5" \
    > "$TRAJ_ROOT/cfg4.5.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 "$PY_FASTGEN" "$FASTGEN_SCRIPT" \
    $COMMON_TRAJ_FLAGS --guidance_scale 1.0 \
    --output_dir "$TRAJ_ROOT/nocfg" \
    > "$TRAJ_ROOT/nocfg.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 "$PY_FASTGEN" "$FASTGEN_SCRIPT" \
    $COMMON_TRAJ_FLAGS --guidance_scale 3.0 \
    --output_dir "$TRAJ_ROOT/cfg3.0" \
    > "$TRAJ_ROOT/cfg3.0.log" 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 "$PY_FASTGEN" "$FASTGEN_SCRIPT" \
    $COMMON_TRAJ_FLAGS --guidance_scale 6.0 \
    --output_dir "$TRAJ_ROOT/cfg6.0" \
    > "$TRAJ_ROOT/cfg6.0.log" 2>&1 &
PID3=$!

echo "  GPU 0: CFG=4.5 (PID $PID0)"
echo "  GPU 1: noCFG   (PID $PID1)"
echo "  GPU 2: CFG=3.0 (PID $PID2)"
echo "  GPU 3: CFG=6.0 (PID $PID3)"
wait $PID0 $PID1 $PID2 $PID3
echo "[$(date '+%F %T')] BATCH 1 DONE"
echo ""

# ── BATCH 2: 4 Euler variants on 4 GPUs (~60 min) ──
echo "[$(date '+%F %T')] BATCH 2: 4 Euler variants"

TRAJ_DIR="$TRAJ_ROOT/cfg4.5"
SAMPLES=$(ls -d ${TRAJ_DIR}/*/ | xargs -I{} basename {} | tr '\n' ',' | sed 's/,$//')

CUDA_VISIBLE_DEVICES=0 "$PY_OMNI" "$EULER_SCRIPT" \
    --mode euler_jump --traj_dir "$TRAJ_DIR" \
    --output_dir "$ANALYSIS_ROOT/euler_cfg45_cfg45" \
    --samples "$SAMPLES" \
    --cfg_step0 4.5 --cfg_teacher 4.5 --cfg_drop_text true --skip_existing \
    > "$ANALYSIS_ROOT/euler_cfg45_cfg45.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 "$PY_OMNI" "$EULER_SCRIPT" \
    --mode euler_jump --traj_dir "$TRAJ_DIR" \
    --output_dir "$ANALYSIS_ROOT/euler_nocfg_cfg45" \
    --samples "$SAMPLES" \
    --cfg_step0 1.0 --cfg_teacher 4.5 --cfg_drop_text true --skip_existing \
    > "$ANALYSIS_ROOT/euler_nocfg_cfg45.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 "$PY_OMNI" "$EULER_SCRIPT" \
    --mode euler_jump --traj_dir "$TRAJ_DIR" \
    --output_dir "$ANALYSIS_ROOT/euler_nocfg_nocfg" \
    --samples "$SAMPLES" \
    --cfg_step0 1.0 --cfg_teacher 1.0 --cfg_drop_text true --skip_existing \
    > "$ANALYSIS_ROOT/euler_nocfg_nocfg.log" 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 "$PY_OMNI" "$EULER_SCRIPT" \
    --mode euler_jump --traj_dir "$TRAJ_DIR" \
    --output_dir "$ANALYSIS_ROOT/euler_cfg45_nocfg" \
    --samples "$SAMPLES" \
    --cfg_step0 4.5 --cfg_teacher 1.0 --cfg_drop_text true --skip_existing \
    > "$ANALYSIS_ROOT/euler_cfg45_nocfg.log" 2>&1 &
PID3=$!

echo "  GPU 0: CFG→CFG     (PID $PID0)"
echo "  GPU 1: noCFG→CFG   (PID $PID1)"
echo "  GPU 2: noCFG→noCFG (PID $PID2)"
echo "  GPU 3: CFG→noCFG   (PID $PID3)"
wait $PID0 $PID1 $PID2 $PID3
echo "[$(date '+%F %T')] BATCH 2 DONE"
echo ""

echo "============================================"
echo "[$(date '+%F %T')] ALL GENERATION COMPLETE"
echo ""
echo "Outputs:"
echo "  Trajectories: $TRAJ_ROOT/{cfg4.5,nocfg,cfg3.0,cfg6.0}/"
echo "  Euler:        $ANALYSIS_ROOT/{euler_cfg45_cfg45,euler_nocfg_cfg45,euler_nocfg_nocfg,euler_cfg45_nocfg}/"
echo ""
echo "Next: run evaluation (can be on a different machine)"
echo "  bash scripts/omniavatar/run_mouthweight_evaluation.sh"
echo "============================================"
