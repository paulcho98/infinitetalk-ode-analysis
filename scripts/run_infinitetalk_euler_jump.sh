#!/bin/bash
# Euler-jump (ODE-straightness) factorial for InfiniteTalk — port of the OmniAvatar 2x2.
#
# TWO overlapping factorials over (step-0 CFG) x (teacher CFG), where "on" is the default
# config and "off" comes in two flavours:
#
#   A) audio-off :  on = (t5,a4)   off = (t5,a1)   -> isolates the AUDIO guidance term
#   B) all-off   :  on = (t5,a4)   off = (t1,a1)   -> isolates guidance as a whole
#
# The on/on cell is SHARED by both factorials, so the two 2x2s are 7 distinct runs, not 8:
#
#            teacher ->    on          noaudio      nocfg
#   step0 on            euler_on_on   on_noaudio   on_nocfg
#   step0 noaudio       noaudio_on    noaudio_noaudio    -
#   step0 nocfg         nocfg_on           -        nocfg_nocfg
#
# Factorial A = {on_on, on_noaudio, noaudio_on, noaudio_noaudio}
# Factorial B = {on_on, on_nocfg,   nocfg_on,   nocfg_nocfg}
#
# PREREQUISITE: the Stage-1 sweep must already have produced trajectories for t5.0_a4.0,
# t5.0_a1.0 and t1.0_a1.0 — the step-0 leg is READ from them (no recomputation).
#
# Output: $OUT/euler_{step0}_{teacher}/<sample>/{step_NNN_xt.pt, step_NNN_x0.pt, ...}
#         i.e. trajectory-shaped, so Stage 2a AND Stage 2b run on it unchanged.
set -euo pipefail

# ── paths on THIS machine (mirror run_infinitetalk_ode_sweep_8gpu.sh) ──
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REF=/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk
export INFINITETALK_ROOT="$REF/InfiniteTalk"
export TORCHDYNAMO_DISABLE=1
CKPT="$INFINITETALK_ROOT/weights/Wan2.1-I2V-14B-480P"
ITALK="$REF/weights/InfiniteTalk/single/infinitetalk.safetensors"
WAV2VEC="$REF/weights/chinese-wav2vec2-base"
PY=/data/karlo-research_715/workspace/kinemadae/projects/paul/videos/lip/.venvs/infinitetalk-ode/bin/python

TRAJ="${TRAJ:-$REPO/ode_full_trajectories_infinitetalk}"   # Stage-1 output (source of step 0)
OUT="${OUT:-$REPO/ode_euler_jump_infinitetalk}"            # this experiment's output (gitignored)
NAMES="$REPO/data/recon_sample_names.txt"
VID="$REPO/data/recon_clips"
AUD="$REPO/data/recon_clips"
STEPS=${1:-50}
NGPU=${NGPU:-8}

# alias -> "text audio"
declare -A CFG=( [on]="5.0 4.0" [noaudio]="5.0 1.0" [nocfg]="1.0 1.0" )
# alias -> Stage-1 trajectory dir supplying step 0
declare -A TRAJDIR=(
    [on]="$TRAJ/infinitetalk_t5.0_a4.0"
    [noaudio]="$TRAJ/infinitetalk_t5.0_a1.0"
    [nocfg]="$TRAJ/infinitetalk_t1.0_a1.0"
)

# the 7 distinct (step0, teacher) cells
CELLS=(
    "on on"
    "on noaudio"
    "noaudio on"
    "noaudio noaudio"
    "on nocfg"
    "nocfg on"
    "nocfg nocfg"
)

mkdir -p "$OUT"
cd "$REPO"
echo "REPO=$REPO"
echo "TRAJ=$TRAJ (step-0 source)"
echo "OUT=$OUT | steps=$STEPS | ngpu=$NGPU | cells=${#CELLS[@]}"

# ── preflight: every step-0 trajectory we depend on must exist ──
for alias in on noaudio nocfg; do
    d="${TRAJDIR[$alias]}"
    if [ ! -d "$d" ]; then
        echo "[FATAL] missing Stage-1 trajectory for '$alias': $d" >&2
        echo "        Run scripts/run_infinitetalk_ode_sweep_8gpu.sh first." >&2
        exit 1
    fi
done

COMMON=(--checkpoint_dir "$CKPT" --infinitetalk_dir "$ITALK" --wav2vec_dir "$WAV2VEC"
        --video_dir "$VID" --audio_dir "$AUD" --sample_names_file "$NAMES"
        --audio_cache_dir "$TRAJ/_audio_cache"
        --num_inference_steps "$STEPS" --shift 7.0 --size infinitetalk-480
        --seed 42 --max_samples 10 --skip_existing)

# One cell per GPU; with 7 cells and 8 GPUs every cell runs concurrently.
gpu=0
pids=()
for cell in "${CELLS[@]}"; do
    read -r s0 tch <<< "$cell"
    read -r t_t a_t <<< "${CFG[$tch]}"
    name="euler_${s0}_${tch}"
    outdir="$OUT/$name"
    mkdir -p "$outdir"
    echo "  GPU $gpu: $name   step0=${CFG[$s0]}  teacher=${CFG[$tch]}"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/generate_infinitetalk_euler_jump.py \
        "${COMMON[@]}" \
        --step0_traj_dir "${TRAJDIR[$s0]}" \
        --text_cfg_teacher "$t_t" --audio_cfg_teacher "$a_t" \
        --output_dir "$outdir" \
        > "$OUT/$name.log" 2>&1 &
    pids+=($!)
    gpu=$(((gpu + 1) % NGPU))
done

echo "waiting on ${#pids[@]} jobs ..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo "=== Euler-jump generation done (fail=$fail) ==="
for cell in "${CELLS[@]}"; do
    read -r s0 tch <<< "$cell"
    name="euler_${s0}_${tch}"
    n=$(find "$OUT/$name" -name "step_*_x0.pt" 2>/dev/null | wc -l)
    echo "  $name: $n x0 tensors"
done
echo ""
echo "Next: Stage-2 metrics + geometry over these dirs"
echo "  bash scripts/run_stage2_euler_jump.sh"
exit $fail
