# Data

## The 10 recon samples (Hallo3, NOT HDTF)

The evaluation reuses the **same 10 identities** as the OmniAvatar ODE runs, for a 1:1 comparison. We
verified these are **Hallo3** samples (0/10 appear in the HDTF validation set; all 10 are in Hallo3).
Hallo3 identifies clips by content **hash**, so the hash *is* the canonical ID — there is no
HDTF-style name to substitute. Sample names keep the `<hash>_shot_001_000` form the OmniAvatar runs
used (the `_shot_001_000` suffix came from StableAvatar's clip segmentation; the video/audio are keyed
by the bare `<hash>`).

Sample list: `data/recon_sample_names.txt` (10 lines).

## Per-sample source files (original machine)

For each sample, `hash = name.split("_shot")[0]`:
- **Reference video** (frame 0 = the I2V reference image):
  `/home/work/.local/Hallo3_validation/validation_set_for_benchmark/<hash>.mp4`
  (native ~720–1020 px, VARYING aspect: some 1:1, some 2:3 portrait — see force-square below)
- **Audio**: `/home/work/.local/Hallo3_validation/processed/audios/<hash>.wav`

The canonical Hallo3 path recorded in `hallo3_metadata.csv`
(`/home/work/.local/Hallo3/validation/{videos_cfr,audios}/`) was **deleted** on the original machine;
the `validation_set_for_benchmark` + `processed/audios` copies above are the live sources. If neither
exists on the new machine, obtain the Hallo3 validation set and match by hash.

**All 10 recon clips (video + audio) are BUNDLED at `data/recon_clips/<hash>.{mp4,wav}`** — point Stage-1's `--video_dir` and `--audio_dir` both there. (`examples/inputs/` keeps 2 as quick examples.)

## FORCE-SQUARE (critical)

InfiniteTalk's `generate_infinitetalk` selects the aspect-ratio bucket **closest to each reference
image's aspect ratio**. Because Hallo3 references vary (1:1 and 2:3), that yields NON-uniform latents:
square refs → `[16,21,80,80]`, 2:3 refs → `[16,21,64,96]`. That violates the "square" requirement and
breaks the uniform-dims analysis. The driver therefore **hardcodes the square 640×640 bucket**
(`target_h=target_w=640`) so every sample is `[16,21,80,80]`. Reference frames are center-cropped to
square. Stage-2 GT frame readers must use the SAME center-crop-to-square (matching InfiniteTalk's
`resize_and_centercrop`, which does cover-scale-shorter-side-then-center-crop) — NOT a stretch resize.

## Prompt

All samples use the generic prompt `"A person is talking."` (analog of OmniAvatar's common
"a person is talking" text embedding). This is the positive text; InfiniteTalk's long default negative
prompt is used for the unconditional branch. The prompt affects the text-CFG direction, so keep it
consistent across any regeneration.

## Output layout (what Stage 1 writes)

```
<OUT_ROOT>/infinitetalk_t{T}_a{A}/<hash>_shot_001_000/
    step_000_xt.pt … step_049_xt.pt     # noisy state x_t, [16,21,80,80] bf16
    step_000_x0.pt … step_049_x0.pt     # denoised prediction x0, [16,21,80,80] bf16
    ode_schedule.json                   # t_list, shift(=7), text/audio scales, latent_shape, seed
    input_latents.pt                    # VAE-encoded reference-frame latent [16,1,80,80]
<OUT_ROOT>/_audio_cache/<hash>.pt       # wav2vec embedding [N,12,768], computed once per hash
```
~100 tensors/sample/config → 70 trajectories → ~7k `.pt` files, ~18 GB.

## OmniAvatar results to compare against (original machine)
- Generation: `/home/work/.local/ode_full_trajectories/` (dirs `14B_cfg1.0/3.0/6.0`, `14B_nocfg`,
  `14B_audio_only_cfg*`, `14B_schedule25`, …).
- Analysis: `/home/work/.local/ode_analysis/` (`14B/perceptual_v2/metrics.csv`, etc.).
