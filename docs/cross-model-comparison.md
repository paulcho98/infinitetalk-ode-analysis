# Cross-model comparison vs OmniAvatar (original step 6) — handoff

**Status (2026-07-29): UNBLOCKED, not started.** Every doc in this repo written before this date says
step 6 is "blocked on the OmniAvatar `ode_analysis/` results, which are not on this machine." That was
true on the **sweep machine**. It is **not** true on the **OmniAvatar machine** — the box whose repo
root is `/home/work/.local/OmniAvatar`, where the original study was run. All OmniAvatar results are
there, intact, and I verified they are actually diffable against this repo's CSVs.

Read this doc before doing anything on step 6. It records what exists where, what was verified, what is
genuinely missing, and what is *not* comparable.

---

## Machine map (this is the thing that keeps getting lost)

| | **OmniAvatar machine** (`/home/work/.local/OmniAvatar`) | **Sweep machine** (where the InfiniteTalk runs happened) |
|---|---|---|
| OmniAvatar ODE results | ✅ all of it (see inventory below) | ❌ none |
| InfiniteTalk trajectories (`.pt`, ~29 GB) | ❌ only an abandoned partial — see below | ✅ complete |
| InfiniteTalk Stage-2 analysis roots (decoded videos, per-cell `metrics.csv`) | ❌ | ✅ complete |
| InfiniteTalk aggregate results | ✅ via this git repo (`results/`) | ✅ |
| `METRICS_ROOT` | `/home/work/.local/eval_metrics` | `/data/karlo-research_715/workspace/kinemaar/paul/eval_metrics` |

`ode_analysis*/`, `ode_euler_jump*/` and `ode_full_trajectories*/` are **gitignored**, so everything
except the committed `results/` tree stays on whichever machine produced it. That is the entire reason
step 6 has a transfer problem rather than being done already.

**The partial copy on the OmniAvatar machine is junk.**
`/home/work/.local/ode_full_trajectories_infinitetalk/` (2.7 GB, Jul-24) holds an abandoned pre-handoff
run: 2 of 7 configs, 4 of 10 samples, 200 / 124 of 500 `x0` files. Do not mistake it for the sweep.

---

## OmniAvatar-side inventory (all paths on the OmniAvatar machine)

### `/home/work/.local/ode_analysis/all_csvs/` — the merged CSV bundle, 29 files
This is the comparison surface. `all_csvs/README.md` is a full experiment inventory (teacher ckpt,
schedule, CFG drop modes, per-experiment status, generation commands, plot scripts) — read it.

The four **replication targets** for our Factorial B, 6031 rows each:

| CSV | trace | our cell |
|---|---|---|
| `14B_textaudio_euler_cfg45_cfg45.csv` | 4.5 → 4.5 | `euler_on_on` |
| `14B_textaudio_euler_nocfg_cfg45.csv` | 1.0 → 4.5 | `euler_nocfg_on` |
| `14B_textaudio_euler_nocfg_nocfg.csv` | 1.0 → 1.0 | `euler_nocfg_nocfg` |
| `14B_textaudio_euler_cfg45_nocfg.csv` | 4.5 → 1.0 | `euler_on_nocfg` |

Sequential counterparts: `14B_textaudio_perceptual_v2.csv` (cfg 4.5), `14B_textaudio_trajectory_nocfg.csv`
(cfg 1.0), `14B_textaudio_cfg{1.0,3.0,6.0}_trajectory.csv` (the scalar-CFG sweep).
Also present, with no InfiniteTalk counterpart: `14B_textaudio_fresh_noise.csv`,
`14B_textaudio_schedule25.csv`, the whole `14B_audioonly_*` group (7), `latentsync_*` (6, a third
model), `spatial_probe_*` (3).

### Decoded per-step videos (no re-decode needed)
- `ode_analysis/14B/perceptual_v2/videos/<sample>/step_NNN.mp4` — sequential cfg 4.5, 826 MB
- `ode_analysis/14B/euler_*/videos/<sample>/step_NNN.mp4` — 4 euler cells, ~3.0 GB
- 102 files per sample (`gt`, `gt_audio`, `step_000..049`, each with an `_audio` variant)

### Raw `.pt` trajectories — `/home/work/.local/ode_full_trajectories/` (42 GB)
Present: `14B_cfg1.0`, `14B_cfg3.0`, `14B_cfg6.0`, `14B_nocfg`, `14B_audio_only_cfg{,1.0,3.0,6.0,_schedule25}`,
`14B_schedule25`, `14B_mouthweight`, `latentsync_1.6{,_nocfg}`, `videopainter_*`.
Layout per sample: `input_latents.pt`, `ode_schedule.json`, `step_NNN_{xt,x0}.pt` ×50.

**Absent — and this is the one that hurts:** the default **cfg 4.5** sequential dump. `all_csvs/README.md`
places it at `ode_full_trajectories/14B/` (no `.local`); that root no longer exists, and
`find /home/work -maxdepth 4 -type d -name '14B'` returns only `ode_analysis/14B`. Its *decoded videos
and metrics survive*, but the latents are gone.

**Also absent: euler latents, for every cell.** The OmniAvatar euler runs decoded to mp4 inline and
never wrote `.pt` (`all_csvs/README.md` lists their trajectory dir as "—"). Consequence in gap C below.

---

## Verified compatible (checked 2026-07-29, not assumed)

- **Same 10 samples.** Sample-ID sets in `14B_textaudio_perceptual_v2.csv` and
  `results/infinitetalk/data/perceptual_t5.0_a4.0.csv` are identical, string for string.
- **Same schema.** `step,t,sample,metric,region,value`, with a `step=-1` GT baseline row and steps 0–49.
- **Row counts reconcile exactly.** OmniAvatar 6031 vs InfiniteTalk 5027. The delta is the dropped
  `upper_face` region (500 `pixel_mse` + 500 `ssim`) plus SyncNet dropouts (`sync_c`/`sync_d` 510 vs 508).
  Nothing unexplained.

So a per-step, per-sample, per-metric join is straightforward — for the sequential configs. The euler
cells are gap A.

---

## Gaps

### A. InfiniteTalk's per-step euler metrics CSVs — ✅ CLOSED 2026-07-30

**Resolved from the sweep machine: the 7 CSVs are committed to this repo**, relocated during the
`results/data/` → `results/infinitetalk/data/` restructure:

```
results/infinitetalk/data/euler_perceptual_{on_on,on_noaudio,on_nocfg,noaudio_on,noaudio_noaudio,nocfg_on,nocfg_nocfg}.csv
```

2.7 MB total. Verbatim copies of `ode_analysis_euler_jump/euler_<cell>/perceptual_v2/metrics.csv`
(still gitignored at source), named to mirror the existing `straightness_<cell>.json` convention.
Verified on copy: schema is byte-identical to the sequential
`results/infinitetalk/data/perceptual_t{T}_a{A}.csv` (`step,t,sample,metric,region,value`), each
file carries all 10 samples × steps 0–49 plus 30 `step=-1` GT rows, 4750–4962 rows per cell (the
spread is SyncNet face-track dropouts on heavily blurred late-landing jumps, same cause as the
510-vs-508 delta noted above).

*Why this mattered:* the only euler perceptual data previously in git was
`results/infinitetalk/figures/euler_jump/euler_terminal_values.csv` — **terminal step only, mouth
only, 4 metrics** (`plot_euler_jump_factorial.py` hardcodes `REGION = "mouth"` and reads the last
step). That is precisely the wrong slice, since Sync-C peaks at landing **step 11–15** and decays
~3× by step 49; a terminal-only diff would have discarded the headline finding.

**`configs/registry.yaml` updated to match**: the 4 per-step euler entries (`it_euler_on_on`,
`it_euler_nocfg_on`, `it_euler_nocfg_nocfg`, `it_euler_on_nocfg`) now point at the real `csv:`
paths above and no longer carry `status: missing_sweep_machine`. Each stays paired with its
OmniAvatar replication target under `comparisons:` in the same file (`euler_on_on`,
`euler_nocfg_on`, `euler_nocfg_nocfg`, `euler_on_nocfg`). The registry also gained 3 new
`it_euler_noaudio_noaudio` / `it_euler_noaudio_on` / `it_euler_on_noaudio` entries for the
remaining audio-knob cells — no OmniAvatar counterpart exists for those, so they carry no
`comparisons:` pairing. Also logged in `docs/status-and-todo.md` § Deferred re-runs (c).

### B. Stage 2b is stale and its re-run lives on the sweep machine
`results/infinitetalk/data/geometry_*.json` predate the true `{region}_velocity` metric and the `delta_cosine[0]`
fix. Re-running needs the 29 GB trajectory dump. Either re-run there and commit fresh JSONs (cheap:
VAE-encode per sample), or ship the trajectories — `/home/work/.local` has ~549 GB free.

### C. There is no OmniAvatar counterpart to the straightness number
`measure_euler_straightness.py` computes `‖x0_euler − x0_seq‖` in **latent** space. On the OmniAvatar
side both operands are missing: the euler runs never saved latents, and the cfg-4.5 sequential dump is
gone. Options:

1. **Regenerate** OmniAvatar's cfg-4.5 sequential trajectory + the 4 euler cells with latent dumps
   (`FastGen/scripts/generate_omniavatar_ode_pairs_full.py`, `OmniAvatar/scripts/generate_single_step_predictions.py`
   — commands are in `all_csvs/README.md`). Exact, expensive.
2. **Pixel-space proxy** from the per-step mp4s, which exist on the OmniAvatar side for both the
   sequential run and all 4 euler cells. Cheap and no generation — but to be apples-to-apples it must be
   computed the same way on the InfiniteTalk side, whose decoded videos are also gitignored and multi-GB.
   Compute it on each machine and ship only the numbers.

Note this is an **OmniAvatar-side** gap. The sweep machine did not miss it; the data was never saved.

### D. OmniAvatar experiment families with no InfiniteTalk counterpart
- `fresh_noise` — deliberately not ported (documented in `euler-jump-experiment.md`).
- Scheduled CFG τ=25 (`14B_schedule25`) — no analogue.
- The **audio-only-CFG** group (`cfg_drop_text=false`): its analogue would be guidance on the audio axis
  alone, i.e. `(t1, a>1)`. **Our 7-config grid contains no such cell**, so this cannot be derived from
  existing runs — it needs new sweeps if we want the supplementary comparison.
- Spatial CFG probe (Exp 2), and the LatentSync 1.6 group (a third model) — OmniAvatar-side only.

---

## Not comparable in absolute terms — normalize first

The two studies measure the same metrics on **differently preprocessed frames**: OmniAvatar uses
512×512 pre-aligned crops with LatentSync-mask regions (`mouth`/`upper_face`/`full`); we use
force-square 640² frames with a dlib mouth bbox (`mouth`/`full`). The GT baselines diverge accordingly
(`step=-1` rows, mouth region):

| GT metric | OmniAvatar | InfiniteTalk |
|---|---|---|
| sharpness | 11.77 | 44.11 |
| Sync-C | 6.64 | 6.22 |
| Sync-D | 7.78 | 8.52 |

Sharpness and pixel-MSE are **not** cross-model comparable at face value; Sync-C/D and the perceptual
metrics are close enough to compare in trend. Normalize every claim against each model's own GT row or
its own no-CFG baseline before making it. Also remember `on = (t5,a4) ↔ cfg 4.5` is a stated mapping
assumption, not an equivalence — OmniAvatar's CFG is scalar, ours is 2-D.

---

## Suggested order of work

1. ~~Fetch the 7 euler `metrics.csv` files (gap A).~~ **Done — `git pull`.** They are committed as
   `results/infinitetalk/data/euler_perceptual_<cell>.csv`. Everything else for Factorial B was
   already on the OmniAvatar machine, so **step 2 is now unblocked and can start immediately, there.**
2. Write the diff: per-step overlays of our 4 Factorial-B cells against the 4 `14B_textaudio_euler_*.csv`,
   normalized per model. The specific questions worth answering — does OmniAvatar also peak at a
   mid landing step, and is its guided-origin curvature penalty the same sign and rough magnitude.
3. Same treatment for the sequential CFG sweep: our diagonal family `(1,1) (2.5,2) (5,4) (7.5,6)` against
   OmniAvatar's scalar `1.0 / 3.0 / 4.5 / 6.0`.
4. Decide whether the straightness number needs an OmniAvatar twin (gap C) — regenerate, or proxy, or
   report it as InfiniteTalk-only.
5. Gap B (Stage-2b re-run) is independent of all of the above and can proceed in parallel on the sweep
   machine.
