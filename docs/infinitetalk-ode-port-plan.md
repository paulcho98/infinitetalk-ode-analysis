# InfiniteTalk ODE-Trajectory Port — Plan

**Goal:** Replicate the OmniAvatar ODE-trajectory analysis (`/home/work/.local/ode_full_trajectories`
+ `/home/work/.local/ode_analysis`) on the **InfiniteTalk** baseline (`/home/work/.local/InfiniteTalk`,
Wan2.1-I2V-14B-480P + audio cross-attn). Env: conda `infinitetalk`. **Path A only — no training/distillation.**

## Scope decision: Path A (minimal fork)
Fork InfiniteTalk's own denoising loop into `FastGen/scripts/generate_infinitetalk_ode_pairs_full.py`.
Reuse `InfiniteTalkPipeline` for ALL setup (model/VAE/CLIP/T5 load, input prep, conditioning dicts,
timesteps). We do NOT build a `FastGenNetwork` wrapper (that would only be needed for training).
InfiniteTalk's audio is its OWN cross-attention (AudioProjModel → SingleStreamMutiAttention); we do
not reuse OmniAvatar's AudioPack.

## Key facts (from wan/multitalk.py)
- No x0 head: hand-written flow-matching **Euler on velocity**. `latent` IS x_t; the `x0` var is just
  the final latent. DiT returns velocity; code does `noise_pred = -noise_pred` (multitalk.py:758).
- **Derive x0 per step:** `sigma_i = timesteps[i]/1000`; `x0_pred = latent + sigma_i * noise_pred`
  (using the code's post-:758 noise_pred). Equivalent to OmniAvatar's x0-space save.
- **Euler update ≡ OmniAvatar's x0→eps→forward_process re-noise** (both are the RF deterministic step),
  and **CFG-in-velocity ≡ CFG-in-x0** (affine) → InfiniteTalk trajectories are directly comparable to
  OmniAvatar's.
- **3-call CFG** (default text=5, audio=4): `noise_pred = uncond + text_s*(cond-drop_text) + audio_s*(drop_text-uncond)`.
- Conditioning is constant across steps (arg_c/arg_null_text/arg_null built once) → precompute once.
- **shift=7** for 480p (generate_infinitetalk.py CLI default `sample_shift`).
- Gotchas: motion-frame pinning (multitalk.py:711,773 — first latent frame clamped, deterministic);
  keep teacache OFF and APG OFF (already CLI defaults).

## CFG sweep matrix (7 distinct configs)
Shared audio axis {1,2,4,6}. Dir naming `infinitetalk_t{T}_a{A}`.

| audio | Family 1 (text fixed=5) | Family 2 (text scaled, ratio≈1.25) |
|---|---|---|
| 1 | (5,1) | (1,1) ← **no-CFG baseline** = cond only, 1 forward pass |
| 2 | (5,2) | (2.5,2) |
| 4 | (5,4) ← **default, shared** | (5,4) same point |
| 6 | (5,6) | (7.5,6) |

Special-case (1,1): collapses to `noise_pred = cond` → run 1 forward pass, skip CFG combine.

## Data / resolution
- Same 10 recon identities + audio as OmniAvatar's ODE runs.
- **Square 480p bucket ≈ 640×640 → latent [16, 21, 80, 80]** (from `ASPECT_RATIO_627`; pick exact
  square entry). (OmniAvatar was 512²→[…,64,64].)

## Output layout (match OmniAvatar so Stage-2 scripts consume it)
Per sample dir: `step_NNN_xt.pt`, `step_NNN_x0.pt`, `ode_schedule.json` (timesteps, shift, text/audio
scales, latent_shape), `input_latents.pt` (GT ref latent). ~7 configs × 10 samples × 50 steps × 2 ≈
7k files, ~18 GB. Shard configs across GPUs 0–3.

## Stage-2 analysis adaptation (small)
- Same `Wan2.1_VAE.pth` → decode compatible.
- Parameterize latent spatial dims (bucket-dependent, not hardcoded 64×64).
- No LatentSync mouth-mask → masked perceptual metrics need a face-crop / full-frame variant.

## Data sources (finalized)
The 10 recon samples are **Hallo3** (0/10 in HDTF); Hallo3 names clips by content hash, so the hash
IS the ID. We keep the same `<hash>_shot_001_000` names as the OmniAvatar ODE runs for 1:1 comparison.
The canonical Hallo3 path in metadata is deleted; live copies used instead:
- reference video (frame 0 = I2V ref): `/home/work/.local/Hallo3_validation/validation_set_for_benchmark/<hash>.mp4` (722×722)
- audio: `/home/work/.local/Hallo3_validation/processed/audios/<hash>.wav`
- sample list: `/home/work/.local/ode_full_trajectories_infinitetalk/recon_sample_names.txt`
Prompt: generic `"A person is talking."` (analog of OmniAvatar's common prompt).

## Stage-1 driver (DONE + running)
- `FastGen/scripts/generate_infinitetalk_ode_pairs_full.py` (subclass `ODEInfiniteTalkPipeline`).
  Multi-config per model load; shard by SAMPLE across GPUs; audio wav2vec embs cached per hash.
- Launcher: `FastGen/scripts/run_infinitetalk_ode_sweep.sh` (7 configs, 50 steps, 4 GPUs).
- Output: `/home/work/.local/ode_full_trajectories_infinitetalk/infinitetalk_t{T}_a{A}/<name>/`.
- Validated: derived x0 decodes to a coherent talking-head; shapes [16,21,80,80]; (1,1) 1-pass shortcut works.

## Mouth-region metric (replaces LatentSync mask)
dlib 68-pt landmarks → mouth points **48–67** → padded bbox → crop just the mouth (stable per-clip bbox).
`eval_ode_perceptual_v2.py` ALREADY has `_extract_mouth_landmarks` using pts[48:68] → localized swap.
Regions become **mouth + full** only; `upper_face` (mask complement) has no InfiniteTalk analogue → dropped.

## Stage-2 script port classification (audit)
- TRIVIAL (1): `decode_ode_trajectory.py` (mask overlay auto-skips at 640px).
- PARAM (10): `eval_ode_perceptual{,_v2}.py`, `visualize_ode_stepwise.py`, `analyze_ode_trajectory.py`
  (dims 512→640 / latent 64→80; mask→dlib mouth-bbox), + 6 plotters (repoint config-dir lists to the 7 dirs).
- REWORK/NA (3): `simulate_euler_and_decode.py` (re-runs the OmniAvatar teacher — model-specific),
  `plot_mouthweight_ode_results.py` (1D-cfg frontier → 2D t×a grid), `fix_euler_nocfg_cfg15_step0.py` (LatentSync one-shot, N/A).
- NEW: a true 2D `t×a` heatmap/Pareto view (natural for the grid) — build fresh, don't bend the 1D frontier.

## Status
- ✅ Env + checkpoints; smoke-test coherent clip.
- ✅ Stage-1 driver written, validated (decode → coherent face), **7-config × 10-sample × 50-step sweep RUNNING**.
- ▶ Next (Stage-2): adapt the metric engine (`eval_ode_perceptual_v2.py`) for 640px + dlib mouth-bbox,
  then trajectory analysis + plotters (repoint to 7 dirs), then the new 2D-grid view.
