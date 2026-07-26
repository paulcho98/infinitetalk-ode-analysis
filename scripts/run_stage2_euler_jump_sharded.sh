#!/bin/bash
# Stage 2 for the Euler-jump factorial — SHARDED metrics variant of run_stage2_euler_jump.sh.
#
# WHY THIS EXISTS: run_stage2_euler_jump.sh runs the Stage-2a *metrics* phase unsharded (one cell
# per GPU). Metrics are SyncNet-bound, not GPU-bound — on the Stage-1 sweep that was the difference
# between ~7 h and ~90 min, which is why scripts/run_stage2a_metrics_sharded.sh exists for the
# sequential configs. This is the same trick applied to the 7 Euler cells.
#
# PHASES
#   1. straightness  ‖x0_euler − x0_seq‖ per step — THE curvature number. CPU, seconds, no GT/VAE.
#   2. decode        one process per cell, one GPU each (~55 min). Already parallel across cells.
#   3. metrics       NS shards per cell, round-robined over 8 GPUs (7*NS processes). The slow leg.
#   4. merge         fold metrics_shard*.csv -> metrics.csv per cell, then plot.
#
# Stage 2b (GT-latent geometry) is NOT run here — same rationale as run_stage2_euler_jump.sh:
# it measures distance-to-GT, while this experiment is about distance-to-sequential (phase 1).
#
# Usage:
#   bash scripts/run_stage2_euler_jump_sharded.sh            # all 7 cells, NS=5 -> 35 procs
#   NS=10 bash scripts/run_stage2_euler_jump_sharded.sh      # 70 procs (heavier, faster)
#   CELLS="euler_on_on euler_on_nocfg" bash scripts/...      # subset (e.g. cells that finished early)
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"                                  # CWD=repo so scripts/ + the sfd_face.pth symlink resolve
REF=/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk
export INFINITETALK_ROOT="$REF/InfiniteTalk"
export METRICS_ROOT=/data/karlo-research_715/workspace/kinemaar/paul/eval_metrics
export PYTHONPATH="$METRICS_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
VENV=/data/karlo-research_715/workspace/kinemadae/projects/paul/videos/lip/.venvs/infinitetalk-ode
export PATH="$VENV/bin:$PATH"; PY="$VENV/bin/python"

EULER_ROOT="${EULER_ROOT:-$REPO/ode_euler_jump_infinitetalk}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$REPO/ode_analysis_euler_jump}"
TRAJ_ROOT="${TRAJ_ROOT:-$REPO/ode_full_trajectories_infinitetalk}"
MASKCACHE="${MASKCACHE:-$REPO/ode_analysis_infinitetalk/_mouth_mask_cache}"
STRAIGHT_OUT="${STRAIGHT_OUT:-$REPO/results/data}"
NS=${NS:-5}                                 # metric shards per cell (5 -> 35 procs over 8 GPUs)
# GPUS: which GPUs to use. Defaults to all 8. Set it to the FREE ones when generation is still
# running on the others — a 14B generation process holds ~60 GB, so landing a decode/metrics
# process on the same GPU risks OOM.  e.g. GPUS="4 6 7"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
mkdir -p "$ANALYSIS_ROOT" "$STRAIGHT_OUT"

CELLS="${CELLS:-euler_on_on euler_on_noaudio euler_noaudio_on euler_noaudio_noaudio euler_on_nocfg euler_nocfg_on euler_nocfg_nocfg}"

# cell -> sequential trajectory matching its TEACHER leg, so the only difference between the two
# is HOW the state was reached: one Euler jump from step 0 vs the full sequential path.
seq_for_cell() {
    case "${1#euler_*_}" in
        on)      echo "$TRAJ_ROOT/infinitetalk_t5.0_a4.0" ;;
        noaudio) echo "$TRAJ_ROOT/infinitetalk_t5.0_a1.0" ;;
        nocfg)   echo "$TRAJ_ROOT/infinitetalk_t1.0_a1.0" ;;
        *)       echo "" ;;
    esac
}

# Only operate on cells that actually finished generating (500 x0 tensors = 10 samples x 50 steps).
READY=()
for c in $CELLS; do
    n=$(find "$EULER_ROOT/$c" -name "step_*_x0.pt" 2>/dev/null | wc -l)
    if [ "$n" -ge 500 ]; then READY+=("$c"); else echo "[skip] $c — only $n/500 x0 tensors"; fi
done
[ ${#READY[@]} -eq 0 ] && { echo "[FATAL] no complete cells"; exit 1; }
echo "=== cells ready: ${READY[*]} (NS=$NS shards each) ==="

# ── 1. straightness (CPU, seconds) ──
echo "--- phase 1: straightness ---"
for c in "${READY[@]}"; do
    s="$(seq_for_cell "$c")"
    [ -d "$s" ] || { echo "[$c] no sequential dir"; continue; }
    "$PY" scripts/measure_euler_straightness.py \
        --euler_dir "$EULER_ROOT/$c" --sequential_dir "$s" \
        --mouth_mask_cache "$MASKCACHE" \
        --output_dir "$STRAIGHT_OUT" --tag "${c#euler_}" 2>&1 | grep -vE "FutureWarning|@amp"
done

# ── 2. decode: one process per cell, one GPU each ──
echo "--- phase 2: decode (one cell per GPU) ---"
gpu=0
pids=()
for c in "${READY[@]}"; do
    outp="$ANALYSIS_ROOT/$c/perceptual_v2"; mkdir -p "$outp"
    CUDA_VISIBLE_DEVICES=${GPU_ARR[$gpu]} "$PY" scripts/eval_ode_perceptual_v2_infinitetalk.py \
        --phase decode --traj_dir "$EULER_ROOT/$c" --output_dir "$outp" \
        > "$outp/decode.log" 2>&1 &
    pids+=($!); gpu=$(((gpu + 1) % NGPU))
done
for p in "${pids[@]}"; do wait "$p"; done
echo "decode done"

# ── 3. metrics: NS shards per cell, round-robin over GPUs (the SyncNet-bound leg) ──
echo "--- phase 3: metrics ($((${#READY[@]} * NS)) shards over GPUs [$GPUS]) ---"
gpu=0
pids=()
for c in "${READY[@]}"; do
    outp="$ANALYSIS_ROOT/$c/perceptual_v2"
    for i in $(seq 0 $((NS - 1))); do
        CUDA_VISIBLE_DEVICES=${GPU_ARR[$gpu]} "$PY" scripts/eval_ode_perceptual_v2_infinitetalk.py \
            --phase metrics --traj_dir "$EULER_ROOT/$c" --output_dir "$outp" \
            --shard_id "$i" --num_shards "$NS" \
            > "$outp/metrics_shard${i}.log" 2>&1 &
        pids+=($!); gpu=$(((gpu + 1) % NGPU)); sleep 1
    done
done
echo "launched ${#pids[@]} metric shards; waiting ..."
for p in "${pids[@]}"; do wait "$p"; done

# ── 4. merge ──
echo "--- phase 4: merge ---"
for c in "${READY[@]}"; do
    outp="$ANALYSIS_ROOT/$c/perceptual_v2"
    "$PY" scripts/eval_ode_perceptual_v2_infinitetalk.py \
        --merge --traj_dir "$EULER_ROOT/$c" --output_dir "$outp" > "$outp/merge.log" 2>&1
    echo "  $c: $(wc -l < "$outp/metrics.csv" 2>/dev/null) rows"
done

echo "=== Stage 2 complete for ${#READY[@]} Euler cells ==="
echo "Next: figures"
echo "  $PY scripts/plot_euler_jump_factorial.py \\"
echo "      --euler_analysis_root $ANALYSIS_ROOT \\"
echo "      --sequential_analysis_root $REPO/ode_analysis_infinitetalk \\"
echo "      --output_dir results/figures/euler_jump"
