#!/bin/bash
# Run 4 single-step prediction variants in parallel on 4 GPUs, then metrics.
#
# GPU 0: fresh_noise          — x_t from fresh noise, teacher with CFG 4.5
# GPU 1: euler_jump           — Euler from step 0 (CFG 4.5), teacher with CFG 4.5
# GPU 2: euler_nocfg_first    — Euler from step 0 (no CFG), teacher with CFG 4.5
# GPU 3: euler_nocfg_both     — Euler from step 0 (no CFG), teacher with no CFG
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$REPO_ROOT/configs/machine.env" ] && source "$REPO_ROOT/configs/machine.env"
PY_OMNI="${PY_OMNI:-/home/work/.local/miniconda3/envs/omniavatar/bin/python}"
PY_FASTGEN="${PY_FASTGEN:-/home/work/.local/miniconda3/envs/fastgen/bin/python}"

SCRIPT="$REPO_ROOT/scripts/omniavatar/generate_single_step_predictions.py"
METRICS="$REPO_ROOT/scripts/omniavatar/eval_ode_perceptual_v2.py"
TRAJ_DIR="${ODE_TRAJ_ROOT_OMNI:-/home/work/.local/ode_full_trajectories}/14B"
MASK_PATH="${MASK_PATH:-$REPO_ROOT/data/mask.png}"
SAMPLES="$(paste -sd, "$REPO_ROOT/data/recon_sample_names.txt")"
BASE_OUT="${ODE_ANALYSIS_ROOT_OMNI:-/home/work/.local/ode_analysis}/14B"

echo "=== Phase 1: Generate predictions (4 GPUs parallel) ==="

# 1. Fresh noise (CFG 4.5)
CUDA_VISIBLE_DEVICES=0 "$PY_FASTGEN" "$SCRIPT" \
    --mode fresh_noise \
    --traj_dir "$TRAJ_DIR" \
    --output_dir "$BASE_OUT/fresh_noise" \
    --samples "$SAMPLES" \
    --guidance_scale 4.5 \
    --skip_existing &

# 2. Euler jump: CFG 4.5 → CFG 4.5 (original)
CUDA_VISIBLE_DEVICES=1 "$PY_FASTGEN" "$SCRIPT" \
    --mode euler_jump \
    --traj_dir "$TRAJ_DIR" \
    --output_dir "$BASE_OUT/euler_cfg45_cfg45" \
    --samples "$SAMPLES" \
    --cfg_step0 4.5 --cfg_teacher 4.5 \
    --skip_existing &

# 3. Euler jump: no CFG → CFG 4.5
CUDA_VISIBLE_DEVICES=2 "$PY_FASTGEN" "$SCRIPT" \
    --mode euler_jump \
    --traj_dir "$TRAJ_DIR" \
    --output_dir "$BASE_OUT/euler_nocfg_cfg45" \
    --samples "$SAMPLES" \
    --cfg_step0 1.0 --cfg_teacher 4.5 \
    --skip_existing &

# 4. Euler jump: no CFG → no CFG
CUDA_VISIBLE_DEVICES=3 "$PY_FASTGEN" "$SCRIPT" \
    --mode euler_jump \
    --traj_dir "$TRAJ_DIR" \
    --output_dir "$BASE_OUT/euler_nocfg_nocfg" \
    --samples "$SAMPLES" \
    --cfg_step0 1.0 --cfg_teacher 1.0 \
    --skip_existing &

wait
echo "Phase 1 done."

echo ""
echo "=== Phase 2: Metrics (4 GPUs parallel) ==="

for VARIANT in fresh_noise euler_cfg45_cfg45 euler_nocfg_cfg45 euler_nocfg_nocfg; do
    GPU_ID=$((RANDOM % 4))
    CUDA_VISIBLE_DEVICES=$GPU_ID "$PY_OMNI" "$METRICS" \
        --phase metrics \
        --traj_dir "$TRAJ_DIR" \
        --mask_path "$MASK_PATH" \
        --output_dir "$BASE_OUT/$VARIANT" &
done

wait
echo "Phase 2 done."

echo ""
echo "=== Phase 3: Merge + Plot ==="

for VARIANT in fresh_noise euler_cfg45_cfg45 euler_nocfg_cfg45 euler_nocfg_nocfg; do
    "$PY_OMNI" "$METRICS" --merge \
        --traj_dir "$TRAJ_DIR" \
        --output_dir "$BASE_OUT/$VARIANT"
done

echo ""
echo "Done! Results:"
echo "  Fresh noise:         $BASE_OUT/fresh_noise/"
echo "  Euler CFG→CFG:       $BASE_OUT/euler_cfg45_cfg45/"
echo "  Euler noCFG→CFG:     $BASE_OUT/euler_nocfg_cfg45/"
echo "  Euler noCFG→noCFG:   $BASE_OUT/euler_nocfg_nocfg/"
