# Environment setup

## Conda env (`infinitetalk`, python 3.10)

Per InfiniteTalk's README, BUT with fixes for a 2026-era package ecosystem (see gotchas below).

```bash
conda create -n infinitetalk python=3.10 -y
conda activate infinitetalk

# CRITICAL: unset the NVIDIA container's global pip constraint or torch 2.4.1 is rejected.
unset PIP_CONSTRAINT

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
pip install ninja psutil packaging wheel

# flash-attn: use the PREBUILT wheel (source build is slow/fragile)
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

pip install -r <INFINITETALK_REPO>/requirements.txt

# PIN these two DOWN (the README's loose >= pulls versions too new for torch 2.4.1):
pip install "transformers==4.49.0" "diffusers==0.33.1"

conda install -c conda-forge librosa ffmpeg -y

# Stage-2 only: face landmarks (prebuilt dlib wheel)
pip install dlib-bin        # -> import dlib works; needed for the mouth bbox
```

### Gotchas (each of these actually bit us)
1. **`PIP_CONSTRAINT=/etc/pip/constraint.txt`** (NVIDIA container) force-pins `torch==2.7.0a0+nv25.3`
   onto every pip call → rejects `torch==2.4.1`. **`unset PIP_CONSTRAINT`** before installing.
2. **`transformers` must be 4.49.0.** 5.x imports `DTensor` from the torch≥2.5 path
   `torch.distributed.tensor` (torch 2.4 only has `torch.distributed._tensor`). This also breaks
   `optimum.quanto` transitively.
3. **`diffusers` must be 0.33.1** — a squeeze: `xfuser 0.4.5` needs `diffusers>=0.33.0`, but `diffusers>=0.35`
   registers a flash-attn-3 op with PEP-604 `X|None` hints that torch 2.4's `infer_schema` rejects.
4. **`latentsync-metrics` env lacks `lpips`** — for Stage-2 Phase-2 metrics use the `omniavatar` env
   (has dlib + lpips + syncnet), NOT latentsync-metrics.
5. **Inference is slow** (~10 s/forward pass at 480p) — this is inherent to the 14B model; teacache is
   deliberately OFF for clean ODE trajectories.

### Known-good versions
torch 2.4.1+cu121 · torchvision 0.19.1 · torchaudio 2.4.1 · xformers 0.0.28 · flash_attn 2.7.4.post1 ·
transformers 4.49.0 · diffusers 0.33.1 · optimum-quanto 0.2.6 · xfuser 0.4.5 · numpy 1.26.4 · dlib-bin 20.0.1.
CUDA available; verified on 4× H200 (140 GB).

## Checkpoints (~190 GB → `<INFINITETALK_REPO>/weights/`)

```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
```
(Note the extra `model.safetensors` pull for wav2vec — main branch only ships `pytorch_model.bin`.)

## Metrics models (Stage-2 Phase-2)
On the original machine these live under `/home/work/.local/eval_metrics/`:
- `shape_predictor_68_face_landmarks.dat` (dlib mouth landmarks)
- `checkpoints/auxiliary/syncnet_v2.model` (SyncNet lip-sync)
- `lpips` is a pip package (in the `omniavatar` env). SyncNet inference code comes from the `eval` /
  `syncnet_python` packages on the original machine.

The Stage-2 scripts reference `METRICS_ROOT=/home/work/.local/eval_metrics` — re-point this.

## The InfiniteTalk repo itself
The Stage-1 driver does `sys.path.insert(0, "/home/work/.local/InfiniteTalk")` and
`from wan.multitalk import InfiniteTalkPipeline, timestep_transform, resize_and_centercrop`. Clone
`github.com/MeiGen-AI/InfiniteTalk`, set it up per its README, and update `INFINITETALK_ROOT` in the
driver + the Stage-2 scripts.
