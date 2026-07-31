# Reference — the ORIGINAL OmniAvatar code (for cross-checking the InfiniteTalk port)

These are the unmodified originals from the source machine. Nothing here runs as-is in this repo (paths
are absolute; the generation half needs the full FastGen package). They are provided so the port in
`../scripts/` can be verified against what the OmniAvatar study actually did.

## `omniavatar_analysis/` — original analysis (OmniAvatar repo `scripts/`)
The InfiniteTalk analysis scripts derive directly from two of these:
- `eval_ode_perceptual_v2.py`  → `../scripts/infinitetalk/eval_ode_perceptual_v2_infinitetalk.py` (2a)
- `analyze_ode_trajectory.py`  → `../scripts/infinitetalk/analyze_ode_trajectory_infinitetalk.py` (2b)
Also included for completeness / reference:
- `eval_ode_perceptual.py` (v1 of the metric engine)
- `decode_ode_trajectory.py` (decode-to-video; TRIVIAL to port)
- `simulate_euler_and_decode.py` (RE-RUNS the OmniAvatar teacher — model-specific, REWORK/N-A)
- `visualize_ode_stepwise.py` (stepwise mouth-crop viz; PARAM)
- `run_eval_ode_perceptual*.sh` (the 4-GPU decode→metrics→merge launchers — the pattern to mirror)

To check the port: diff the InfiniteTalk version against its origin here; the intended changes are
dims 512→640 / latent 64→80, LatentSync-mask → dlib mouth-bbox (regions mouth+full, drop upper_face),
InfiniteTalk `WanVAE` for decode, and Hallo3 GT paths. See `../docs/stage2-audit.md`.

## `fastgen_generation/` — original generation (FastGen repo)
- `generate_omniavatar_ode_pairs_full.py` — the ORIGINAL ODE driver. `../scripts/infinitetalk/generate_infinitetalk_ode_pairs_full.py`
  mirrors its loop structure (build shifted timesteps → save x_t → forward to x0 → CFG in x0-space →
  save x0 → re-noise). **Key cross-check:** the OmniAvatar driver gets x0 directly from
  `OmniAvatarWan.forward(..., fwd_pred_type="x0")`; InfiniteTalk has no x0 head, so our driver DERIVES
  it as `x0 = x_t + sigma·noise_pred` — verify these are equivalent (they are: both are the
  rectified-flow x0, InfiniteTalk just exposes velocity).
- `OmniAvatarWan_model/` — the model wrapper the original driver calls (`network.py` = `OmniAvatarWan`,
  `wan_model.py` = the DiT, `audio_pack.py` = OmniAvatar's additive-residual audio). Read to understand
  the ORIGINAL conditioning/CFG/noise-schedule. NOTE: the InfiniteTalk generation does NOT use this —
  it uses InfiniteTalk's own `wan.multitalk` model with cross-attention audio. This is here only as the
  reference the port was validated against.
- `fastgen_core/` — `network.py` (the `FastGenNetwork` ABC: `forward`/`sample`/noise-scheduler surface)
  and `noise_schedule.py` (`RFNoiseSchedule`: `max_t`, `latents`, `x0_to_eps`, `forward_process`,
  `flow_to_x0` — the exact rectified-flow math the derivation must match).
- `generate_ode_{trajectories,nocfg,no_audio}.sh` — the original OmniAvatar-side launchers (show how
  the 1-D cfg sweep + audio/no-audio ablations were invoked).
