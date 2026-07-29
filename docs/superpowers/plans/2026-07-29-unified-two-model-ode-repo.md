# Unified Two-Model ODE Repo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand `infinitetalk-ode-analysis` into `talking-head-ode-analysis` — one repo that runs the full OmniAvatar and InfiniteTalk ODE experiment suites and joins their results for cross-model comparison.

**Architecture:** Port-only restructure per the approved spec (`docs/superpowers/specs/2026-07-29-unified-two-model-ode-repo-design.md`). Sequential foundation (skeleton → restructure → vendored model packages), then parallel fan-out for ~20 independent script ports, then results/registry/docs, a new TDD'd comparison stage, GPU-sharded verification against existing outputs, and cleanup commits in the two source repos.

**Tech Stack:** Python (torch, einops, matplotlib, pandas, pyyaml), bash launchers, three conda envs (`omniavatar`, `fastgen`, `infinitetalk`), pytest for new comparison code.

## Global Constraints

- **Repo root:** `/home/work/.local/infinitetalk-ode-analysis` (renamed to `talking-head-ode-analysis` in Task 22 — all earlier tasks use the current path).
- **Port-only: numerical behavior of every ported script must be unchanged.** Allowed edits: import lines, path constants → env/CLI, `sys.path` surgery, the new `--save_latents` flag (Task 6). Nothing else.
- **Canonical sources:** InfiniteTalk scripts → THIS repo's copies (the ones in `/home/work/.local/OmniAvatar/scripts` and `/home/work/.local/hyunbin/FastGen/scripts` are stale — never copy from them). OmniAvatar scripts → `/home/work/.local/OmniAvatar/scripts/` and `/home/work/.local/hyunbin/FastGen/scripts/generate_omniavatar_ode_pairs_full.py`.
- **Env-var vocabulary** (exact names, used everywhere):
  - `ODE_TRAJ_ROOT_OMNI` default `/home/work/.local/ode_full_trajectories`
  - `ODE_ANALYSIS_ROOT_OMNI` default `/home/work/.local/ode_analysis`
  - `ODE_TRAJ_ROOT_IT` default `/home/work/.local/ode_full_trajectories_infinitetalk`
  - `ODE_ANALYSIS_ROOT_IT` default `/home/work/.local/ode_analysis_infinitetalk`
  - `WEIGHTS_ROOT` default `/home/work/.local/OmniAvatar/pretrained_models`
  - `VAE_PATH` default `${WEIGHTS_ROOT}/Wan2.1-T2V-14B/Wan2.1_VAE.pth`
  - `TEACHER_CKPT` default `/home/work/output_omniavatar_v2v_phase2/step-10500.pt`
  - `MOUTHWEIGHT_CKPT` default `/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/step-6000.pt`
  - `RECON_DATA_DIR` default `/home/work/stableavatar_data/v2v_validation_data/recon`
  - `NEG_TEXT_EMB` default `/home/work/stableavatar_data/neg_text_emb.pt`
  - `MASK_PATH` default `<repo>/data/mask.png`
  - `METRICS_ROOT` default `/home/work/.local/eval_metrics`
  - `INFINITETALK_ROOT` (already used by InfiniteTalk scripts — keep as-is)
  - `PY_OMNI` default `/home/work/.local/miniconda3/envs/omniavatar/bin/python`
  - `PY_FASTGEN` default `/home/work/.local/miniconda3/envs/fastgen/bin/python`
  - `PY_IT` default `/home/work/.local/miniconda3/envs/infinitetalk/bin/python`
- **Python parameterization pattern** (top of each ported script):
  ```python
  REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  MASK_PATH = os.environ.get("MASK_PATH", os.path.join(REPO_ROOT, "data", "mask.png"))
  ```
  Module-level absolute constants become `os.environ.get("VAR", default)`; argparse defaults reference these; existing CLI flags stay and override.
- **Bash parameterization pattern** (top of each ported launcher):
  ```bash
  REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
  [ -f "$REPO_ROOT/configs/machine.env" ] && source "$REPO_ROOT/configs/machine.env"
  PY_OMNI="${PY_OMNI:-/home/work/.local/miniconda3/envs/omniavatar/bin/python}"
  ```
  All `cd /home/work/.local/OmniAvatar` lines are deleted (scripts are now in-repo); script invocations become `"$REPO_ROOT/scripts/omniavatar/<name>.py"`.
- **Dead-path rule:** any `/home/work/ode_full_trajectories` or `/home/work/ode_analysis` (missing `.local`) is a DEAD path — always replace with the env-var live root, including in docstrings.
- **Grep gate (every port task):** `grep -n "/home/work" <file> | grep -v "environ.get\|:-\|^ *#"` must output nothing (env-var defaults and comments are the only allowed occurrences).
- **Syntax gate (every Python port task):** `python3 -c "import ast; ast.parse(open('<file>').read())"` — runs with system python, no env needed.
- **One commit per task**, conventional prefix (`feat:`, `refactor:`, `docs:`, `chore:`). Work directly on `main` (this is a solo research repo; the repo's existing convention).

## Execution waves (parallelization map)

| Wave | Tasks | Parallel? | Suggested agent tier |
|---|---|---|---|
| 0 | 1 → 2 | sequential | sonnet |
| 1 | 3, 4 | parallel pair | sonnet |
| 2 | 5, 6, 7, 8, 9, 10, 11, 12, 13 | parallel fan-out (11 after 5-7 only for names; safe concurrently since files are disjoint) | 5-7,10: sonnet; 8,9,11-13: haiku/sonnet |
| 3 | 14 → 15 | sequential | 14: sonnet; 15: sonnet |
| 4 | 16 → 17 | sequential | opus |
| 5 | 18, 19, 20, 21 | parallel across GPUs 0-3 | sonnet (opus reviews results) |
| 6 | 22 | sequential, last | sonnet |

---

### Task 1: Directory skeleton, machine env files, committed assets

**Files:**
- Create: `configs/machine-omniavatar.env.example`, `configs/machine-sweep.env.example`
- Create: `data/mask.png` (copy), `models/__init__.py`, dirs `scripts/{common,infinitetalk,omniavatar,comparison}`, `results/{infinitetalk,omniavatar,comparison}`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `configs/machine.env` sourcing convention; `data/mask.png` at the `MASK_PATH` default; directory tree all later tasks write into.

- [ ] **Step 1: Create directories and copy the mask**

```bash
cd /home/work/.local/infinitetalk-ode-analysis
mkdir -p configs models scripts/common scripts/infinitetalk scripts/omniavatar scripts/comparison \
         results/infinitetalk results/omniavatar/data results/omniavatar/figures results/comparison
touch models/__init__.py
cp /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png data/mask.png
```

- [ ] **Step 2: Write `configs/machine-omniavatar.env.example`** (exact content — this box's live values, matching the Global Constraints defaults)

```bash
# Machine config for the OmniAvatar box. Copy to configs/machine.env (gitignored) and edit.
export ODE_TRAJ_ROOT_OMNI=/home/work/.local/ode_full_trajectories
export ODE_ANALYSIS_ROOT_OMNI=/home/work/.local/ode_analysis
export ODE_TRAJ_ROOT_IT=/home/work/.local/ode_full_trajectories_infinitetalk
export ODE_ANALYSIS_ROOT_IT=/home/work/.local/ode_analysis_infinitetalk
export WEIGHTS_ROOT=/home/work/.local/OmniAvatar/pretrained_models
export VAE_PATH=$WEIGHTS_ROOT/Wan2.1-T2V-14B/Wan2.1_VAE.pth
export TEACHER_CKPT=/home/work/output_omniavatar_v2v_phase2/step-10500.pt
export MOUTHWEIGHT_CKPT=/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/step-6000.pt
export RECON_DATA_DIR=/home/work/stableavatar_data/v2v_validation_data/recon
export NEG_TEXT_EMB=/home/work/stableavatar_data/neg_text_emb.pt
export METRICS_ROOT=/home/work/.local/eval_metrics
export INFINITETALK_ROOT=/home/work/.local/InfiniteTalk
export PY_OMNI=/home/work/.local/miniconda3/envs/omniavatar/bin/python
export PY_FASTGEN=/home/work/.local/miniconda3/envs/fastgen/bin/python
export PY_IT=/home/work/.local/miniconda3/envs/infinitetalk/bin/python
```

- [ ] **Step 3: Write `configs/machine-sweep.env.example`** — same variable list; values from the sweep machine as recorded in `docs/status-and-todo.md`'s re-point checklist: `INFINITETALK_ROOT=/data/karlo-research_715/personal/hyunbin/reference_FastGen_InfiniteTalk/InfiniteTalk`, `PY_IT` pointing at the `.venvs/infinitetalk-ode` venv, `METRICS_ROOT` per that doc; OmniAvatar-side vars present but commented out with `# not on this machine`.

- [ ] **Step 4: Append to `.gitignore`**

```
configs/machine.env
```

- [ ] **Step 5: Verify and commit**

```bash
git add -A && git status --short   # expect: new configs/, data/mask.png, models/__init__.py, .gitignore
git commit -m "feat: repo skeleton for two-model layout (configs, dirs, committed mask)"
```

---

### Task 2: Restructure existing InfiniteTalk content (git mv, no logic changes)

**Files:**
- Move: all 17 files in `scripts/` → `scripts/infinitetalk/`, EXCEPT `measure_euler_straightness.py` → `scripts/common/`
- Delete: `scripts/plotters_to_adapt/` (7 files), `scripts/__pycache__/`
- Move: `results/data` → `results/infinitetalk/data`, `results/figures` → `results/infinitetalk/figures`, `results/findings.md` → `results/infinitetalk/findings.md`

**Interfaces:**
- Produces: `scripts/infinitetalk/<name>.py` and `scripts/common/measure_euler_straightness.py` paths that launchers and docs reference.

- [ ] **Step 1: git mv scripts**

```bash
cd /home/work/.local/infinitetalk-ode-analysis
rm -rf scripts/__pycache__
git rm -r scripts/plotters_to_adapt
git mv scripts/measure_euler_straightness.py scripts/common/
for f in scripts/*.py scripts/*.sh; do git mv "$f" scripts/infinitetalk/; done
git mv results/data results/infinitetalk/data
git mv results/figures results/infinitetalk/figures
git mv results/findings.md results/infinitetalk/findings.md
```

- [ ] **Step 2: Fix intra-repo references.** The launchers invoke sibling scripts by relative path and the plotters read `results/data/`. Find every reference and update:

```bash
grep -rn "scripts/\|results/data\|results/figures\|measure_euler_straightness" scripts/infinitetalk/ | grep -v "\.pyc"
```

For each hit: `scripts/<name>.py` → `scripts/infinitetalk/<name>.py` (or `scripts/common/measure_euler_straightness.py`); `results/data` → `results/infinitetalk/data`; `results/figures` → `results/infinitetalk/figures`. Launchers that compute their own dir (`$(dirname "$0")`) and call scripts next to themselves need no change for siblings — verify per file, don't assume.

- [ ] **Step 3: Update doc references.** Same grep over `docs/ README.md`; update moved paths. Do NOT rewrite content — path strings only.

- [ ] **Step 4: Syntax gate on every moved Python file**

```bash
for f in scripts/infinitetalk/*.py scripts/common/*.py; do python3 -c "import ast; ast.parse(open('$f').read())" || echo "FAIL $f"; done
```

Expected: no FAIL lines.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: model-scoped layout — scripts/{infinitetalk,common}, results/infinitetalk"
```

---

### Task 3: Vendored package `models/omniavatar_wan/`

**Files:**
- Create: `models/omniavatar_wan/{__init__.py,network.py,wan_model.py,audio_pack.py,fastgen_network.py,noise_schedule.py,logging_utils.py,utils.py}`
- Source: `reference/fastgen_generation/OmniAvatarWan_model/{network,wan_model,audio_pack}.py`, `reference/fastgen_generation/fastgen_core/{network,noise_schedule}.py` (byte-identical to live FastGen — verified)

**Interfaces:**
- Produces: `from models.omniavatar_wan import OmniAvatarWan` (and `WanModel`, `AudioPack`) — consumed by Tasks 5, 6, 10, 18. `network_causal.py` is deliberately NOT vendored (needs causal distillation machinery; unused by the ODE study; stays in `reference/`).

- [ ] **Step 1: Copy files**

```bash
cd /home/work/.local/infinitetalk-ode-analysis
cp reference/fastgen_generation/OmniAvatarWan_model/network.py models/omniavatar_wan/network.py
cp reference/fastgen_generation/OmniAvatarWan_model/wan_model.py models/omniavatar_wan/wan_model.py
cp reference/fastgen_generation/OmniAvatarWan_model/audio_pack.py models/omniavatar_wan/audio_pack.py
cp reference/fastgen_generation/fastgen_core/network.py models/omniavatar_wan/fastgen_network.py
cp reference/fastgen_generation/fastgen_core/noise_schedule.py models/omniavatar_wan/noise_schedule.py
```

- [ ] **Step 2: Write `models/omniavatar_wan/logging_utils.py`** (shim — same call surface as `fastgen.utils.logging_utils`)

```python
"""Minimal stand-in for fastgen.utils.logging_utils (print-based, rank-agnostic)."""
import sys

def _emit(level, msg):
    print(f"[{level}] {msg}", file=sys.stderr if level in ("WARNING", "ERROR", "CRITICAL") else sys.stdout)

def trace(msg): _emit("TRACE", msg)
def debug(msg): _emit("DEBUG", msg)
def info(msg): _emit("INFO", msg)
def success(msg): _emit("SUCCESS", msg)
def warning(msg): _emit("WARNING", msg)
def error(msg): _emit("ERROR", msg)
def critical(msg): _emit("CRITICAL", msg)
def set_log_level(level): pass
```

- [ ] **Step 3: Write `models/omniavatar_wan/utils.py`** — copy the real bodies verbatim: `expand_like` from `/home/work/.local/hyunbin/FastGen/fastgen/utils/__init__.py:22` and `PRECISION_MAP` from `/home/work/.local/hyunbin/FastGen/fastgen/utils/basic_utils.py:31-36`:

```python
import torch

PRECISION_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

def expand_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # copy the full body verbatim from FastGen/fastgen/utils/__init__.py (def at line 22)
    ...
```

(The `...` above means: paste the verbatim function body from the source file — it is ~5 lines reshaping `x` with singleton dims to match `target.ndim`.)

- [ ] **Step 4: Rewrite imports** — exact substitutions, nothing else changes:

| File | Old | New |
|---|---|---|
| `network.py` | `from fastgen.networks.network import FastGenNetwork` | `from .fastgen_network import FastGenNetwork` |
| `network.py` | `from fastgen.networks.noise_schedule import NET_PRED_TYPES` | `from .noise_schedule import NET_PRED_TYPES` |
| `network.py` | `from fastgen.networks.OmniAvatar.wan_model import WanModel` | `from .wan_model import WanModel` |
| `network.py` | `import fastgen.utils.logging_utils as logger` | `from . import logging_utils as logger` |
| `wan_model.py` | `from fastgen.networks.OmniAvatar.audio_pack import AudioPack` | `from .audio_pack import AudioPack` |
| `fastgen_network.py` | `from fastgen.networks.noise_schedule import get_noise_schedule, NET_PRED_TYPES` | `from .noise_schedule import get_noise_schedule, NET_PRED_TYPES` |
| `fastgen_network.py` | `import fastgen.utils.logging_utils as logger` | `from . import logging_utils as logger` |
| `noise_schedule.py` | `import fastgen.utils.logging_utils as logger` | `from . import logging_utils as logger` |
| `noise_schedule.py` | `from fastgen.utils import expand_like` | `from .utils import expand_like` |
| `noise_schedule.py` | `from fastgen.utils.basic_utils import PRECISION_MAP` | `from .utils import PRECISION_MAP` |

- [ ] **Step 5: Write `models/omniavatar_wan/__init__.py`**

```python
from .wan_model import WanModel
from .audio_pack import AudioPack
from .network import OmniAvatarWan
```

- [ ] **Step 6: Verify no fastgen imports remain, then import-test in the fastgen env** (it has torch/einops/scipy/diffusers)

```bash
grep -rn "fastgen\." models/omniavatar_wan/ && echo "LEFTOVER IMPORTS" || echo "clean"
cd /home/work/.local/infinitetalk-ode-analysis
/home/work/.local/miniconda3/envs/fastgen/bin/python -c "from models.omniavatar_wan import OmniAvatarWan, WanModel, AudioPack; print('import ok')"
```

Expected: `clean` then `import ok`.

- [ ] **Step 7: Commit**

```bash
git add models/ && git commit -m "feat: vendor OmniAvatarWan model package (from FastGen snapshot, imports localized)"
```

---

### Task 4: Vendored package `models/wan_vae/`

**Files:**
- Create: `models/wan_vae/{__init__.py,wan_video_vae.py,loader.py}`
- Source: `/home/work/.local/OmniAvatar/OmniAvatar/models/wan_video_vae.py` (self-contained: imports only torch/einops/tqdm — verified)

**Interfaces:**
- Produces: `from models.wan_vae import load_wan_vae`; `vae = load_wan_vae(vae_path, dtype=torch.bfloat16, device="cuda")` returning a `WanVideoVAE` whose `.decode(latents, device=device, tiled=False)` matches the ModelManager-loaded one bit-for-bit. Consumed by Tasks 6, 7, 9, 10, 19.

- [ ] **Step 1: Copy the VAE file unchanged**

```bash
cp /home/work/.local/OmniAvatar/OmniAvatar/models/wan_video_vae.py \
   /home/work/.local/infinitetalk-ode-analysis/models/wan_vae/wan_video_vae.py
```

- [ ] **Step 2: Write `models/wan_vae/loader.py`.** Replicate exactly what `ModelManager` does for a civitai-format VAE: read `OmniAvatar/models/model_manager.py` lines 1-60 (the `load_model_from_single_file` path: `state_dict_converter().from_civitai(state_dict)` → instantiate → `load_state_dict`) and mirror it. Skeleton (adjust to match what those lines actually do — e.g. `from_civitai` may return `(state_dict, kwargs)`):

```python
import torch
from .wan_video_vae import WanVideoVAE

def load_wan_vae(vae_path, dtype=torch.bfloat16, device="cpu"):
    state_dict = torch.load(vae_path, map_location="cpu", weights_only=False)
    result = WanVideoVAE.state_dict_converter().from_civitai(state_dict)
    if isinstance(result, tuple):
        converted, extra_kwargs = result
    else:
        converted, extra_kwargs = result, {}
    vae = WanVideoVAE(**extra_kwargs)
    vae.load_state_dict(converted)
    return vae.to(dtype=dtype, device=device).eval()
```

- [ ] **Step 3: Write `models/wan_vae/__init__.py`**

```python
from .wan_video_vae import WanVideoVAE
from .loader import load_wan_vae
```

- [ ] **Step 4: Import-test in the omniavatar env**

```bash
cd /home/work/.local/infinitetalk-ode-analysis
/home/work/.local/miniconda3/envs/omniavatar/bin/python -c "from models.wan_vae import load_wan_vae; print('import ok')"
```

(Bit-for-bit decode equivalence against ModelManager is Task 19 — GPU.)

- [ ] **Step 5: Commit**

```bash
git add models/wan_vae && git commit -m "feat: vendor Wan VAE decoder + direct loader (replaces ModelManager dependency)"
```

---

### Task 5: Port OmniAvatar Stage-1 driver + 3 generation launchers

**Files:**
- Create: `scripts/omniavatar/generate_omniavatar_ode_pairs_full.py` (from `/home/work/.local/hyunbin/FastGen/scripts/generate_omniavatar_ode_pairs_full.py`)
- Create: `scripts/omniavatar/generate_ode_trajectories.sh`, `generate_ode_nocfg.sh`, `generate_ode_no_audio.sh` (from `/home/work/.local/OmniAvatar/scripts/`)

**Interfaces:**
- Consumes: `from models.omniavatar_wan import OmniAvatarWan` (Task 3).
- Produces: driver CLI unchanged (`--data_dir --output_dir --guidance_scale --cfg_drop_text --cfg_crossover ...`); launchers runnable as `bash scripts/omniavatar/generate_ode_trajectories.sh`.

- [ ] **Step 1: Copy driver; replace its FastGen coupling** — delete `sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))` (line 53) and replace:

```python
# old (lines 53-56)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastgen.networks.OmniAvatar.network import OmniAvatarWan
import fastgen.utils.logging_utils as logger
# new
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.omniavatar_wan import OmniAvatarWan
from models.omniavatar_wan import logging_utils as logger
```

- [ ] **Step 2: Parameterize the docstring examples** (lines 21, 31 reference `RECON_DATA_DIR`'s literal value — fine to keep as examples, but apply the dead-path rule if any `/home/work/ode_*` non-.local path appears).

- [ ] **Step 3: Copy the 3 launchers; apply the bash pattern.** For each: add the `REPO_ROOT`/`machine.env` preamble; replace the variable block, e.g. `generate_ode_trajectories.sh` lines 11-16 become:

```bash
PRETRAINED="${WEIGHTS_ROOT:-/home/work/.local/OmniAvatar/pretrained_models}"
DATA_DIR="${RECON_DATA_DIR:-/home/work/stableavatar_data/v2v_validation_data/recon}"
MASK_PATH="${MASK_PATH:-$REPO_ROOT/data/mask.png}"
NEG_TEXT_EMB="${NEG_TEXT_EMB:-/home/work/stableavatar_data/neg_text_emb.pt}"
OUTPUT_ROOT="${ODE_TRAJ_ROOT_OMNI:-/home/work/.local/ode_full_trajectories}"
```

Delete `FASTGEN_ROOT` entirely; the python invocation becomes `"$PY_FASTGEN" "$REPO_ROOT/scripts/omniavatar/generate_omniavatar_ode_pairs_full.py"`. `CKPT` lines → `"${TEACHER_CKPT:-/home/work/output_omniavatar_v2v_phase2/step-10500.pt}"` (14B branch; keep the 1.3B literal as-is with a comment). **`generate_ode_no_audio.sh` line 16 has the DEAD `/home/work/ode_full_trajectories` — fix to the env default.**

- [ ] **Step 4: Gates**

```bash
cd /home/work/.local/infinitetalk-ode-analysis
python3 -c "import ast; ast.parse(open('scripts/omniavatar/generate_omniavatar_ode_pairs_full.py').read())"
/home/work/.local/miniconda3/envs/fastgen/bin/python scripts/omniavatar/generate_omniavatar_ode_pairs_full.py --help >/dev/null && echo "help ok"
for f in scripts/omniavatar/generate_ode_*.sh; do bash -n "$f" && echo "bash ok $f"; done
grep -n "/home/work" scripts/omniavatar/generate_omniavatar_ode_pairs_full.py scripts/omniavatar/generate_ode_*.sh | grep -v "environ.get\|:-\|^ *#\|^[^:]*:[0-9]*: *#" 
```

Expected: `help ok`, 3× `bash ok`, final grep empty.

- [ ] **Step 5: Commit** — `git add scripts/omniavatar && git commit -m "feat: port OmniAvatar Stage-1 driver + launchers (vendored model, env roots)"`

---

### Task 6: Port euler-jump/fresh-noise generator + `--save_latents`

**Files:**
- Create: `scripts/omniavatar/generate_single_step_predictions.py` (from `/home/work/.local/OmniAvatar/scripts/`)
- Create: `scripts/omniavatar/run_single_step_both.sh`

**Interfaces:**
- Consumes: `models.omniavatar_wan.OmniAvatarWan`, `models.wan_vae.load_wan_vae`.
- Produces: same CLI plus new `--save_latents` (store_true) writing `step_NNN_x0.pt` per jump into `<output_dir>/latents/<sample>/`.

- [ ] **Step 1: Copy; replace coupling.** Line 43 `sys.path.insert(0, "/home/work/.local/hyunbin/FastGen")` → repo-root insert (as Task 5). FastGen import → `from models.omniavatar_wan import OmniAvatarWan`. Module constants (lines 50-57): `PRETRAINED` → `os.environ.get("WEIGHTS_ROOT", ...)`, `CKPT_14B` → `os.environ.get("TEACHER_CKPT", ...)`, `MASK_PATH` → env pattern with repo `data/mask.png` default. Docstring lines 18/25 have DEAD `/home/work/ode_full_trajectories/14B` → live root. If the script loads the VAE via `OmniAvatar.models.model_manager`, replace with `from models.wan_vae import load_wan_vae; vae = load_wan_vae(vae_path, device=device)` (decode call sites keep `vae.decode(latents, device=device, tiled=False)`).

- [ ] **Step 2: Add `--save_latents`.** In argparse: `parser.add_argument("--save_latents", action="store_true", help="Persist per-jump x0 latents as step_NNN_x0.pt (for straightness analysis)")`. At the point where each jump's `x0_pred` exists (immediately before decode), add:

```python
if args.save_latents:
    lat_dir = os.path.join(args.output_dir, "latents", sample_name)
    os.makedirs(lat_dir, exist_ok=True)
    torch.save(x0_pred.detach().to(torch.bfloat16).cpu(), os.path.join(lat_dir, f"step_{step_idx:03d}_x0.pt"))
```

(Match the actual local variable names in the loop — the pattern is fixed, the names come from the file.) Default off → behavior unchanged.

- [ ] **Step 3: Port `run_single_step_both.sh`** — bash pattern; `TRAJ_DIR` line 13 is DEAD → `"${ODE_TRAJ_ROOT_OMNI:-...}/14B"`; `BASE_OUT` → `"${ODE_ANALYSIS_ROOT_OMNI:-...}/14B"`; delete `cd /home/work/.local/OmniAvatar`; replace the inlined 10-sample literal string with `SAMPLES="$(paste -sd, "$REPO_ROOT/data/recon_sample_names.txt")"` (same 10 names — verify with `diff <(tr ',' '\n' <<< "$OLD_LITERAL") data/recon_sample_names.txt` before deleting the literal).

- [ ] **Step 4: Gates** — syntax gate, `--help` gate under `$PY_FASTGEN`, `bash -n`, grep gate (as Task 5 Step 4).

- [ ] **Step 5: Commit** — `git commit -m "feat: port euler-jump/fresh-noise generator with --save_latents"`

---

### Task 7: Port Stage-2a engine `eval_ode_perceptual_v2.py` + launcher

**Files:**
- Create: `scripts/omniavatar/eval_ode_perceptual_v2.py`, `scripts/omniavatar/run_eval_ode_perceptual_v2.sh` (from `/home/work/.local/OmniAvatar/scripts/`)

**Interfaces:**
- Consumes: `models.wan_vae.load_wan_vae`.
- Produces: identical CLI (`--traj_dir --output_dir --mask_path --vae_path --vae_type --shard_id --num_shards --sync_min_track --sync_only --skip_metrics ...`); identical `metrics.csv` schema `step,t,sample,metric,region,value` with `step=-1` GT row.

- [ ] **Step 1: Copy; replace VAE loading** (lines 157-163):

```python
# old
from OmniAvatar.models.model_manager import ModelManager
...
vae = model_manager.model[vae_idx].to(device)
# new
from models.wan_vae import load_wan_vae
vae = load_wan_vae(args.vae_path, dtype=torch.bfloat16, device=device)
```

Also change line 58's `sys.path.insert` (which pointed at the OmniAvatar repo root) to the repo-root insert pattern. The `sys.path.insert(0, METRICS_ROOT)` calls at lines 383/461 stay — parameterize `METRICS_ROOT` via env pattern if it's a hardcoded constant.

- [ ] **Step 2: Parameterize remaining constants.** `grep -n "/home/work\|AUDIO_BASE_DIR" scripts/omniavatar/eval_ode_perceptual_v2.py` — `AUDIO_BASE_DIR` → `os.environ.get("RECON_DATA_DIR", ...)`; any VAE/mask defaults → env pattern; dead paths → live roots.

- [ ] **Step 3: Port the launcher** — bash pattern; `TRAJ_DIR` (DEAD path, line 10) → `"${ODE_TRAJ_ROOT_OMNI:-...}/14B"`; `OUTPUT_DIR` → `"${ODE_ANALYSIS_ROOT_OMNI:-...}/14B/perceptual_v2"`; `PYTHON` → `"$PY_OMNI"`; delete the `cd`.

- [ ] **Step 4: Gates** — syntax, `--help` under `$PY_OMNI`, `bash -n`, grep gate.

- [ ] **Step 5: Commit** — `git commit -m "feat: port Stage-2a perceptual engine (vendored VAE, env roots)"`

---

### Task 8: Port Stage-2b engine `analyze_ode_trajectory.py`

**Files:**
- Create: `scripts/omniavatar/analyze_ode_trajectory.py` (from `/home/work/.local/OmniAvatar/scripts/`)

**Interfaces:**
- Produces: identical CLI (`--traj_dir --mask_path --output_dir --no_audio_traj_dir ...`); identical JSON/CSV outputs.

- [ ] **Step 1: Copy.** No package imports (pure torch/numpy/PIL/matplotlib — verified). Fix docstring DEAD paths (lines 11-20: `/home/work/ode_full_trajectories/14B`, `/home/work/ode_analysis/14B` → live env-root examples). Apply env pattern to the mask default if one exists (`grep -n "mask" | grep default`).
- [ ] **Step 2: Gates** — syntax, `--help` under `$PY_OMNI`, grep gate.
- [ ] **Step 3: Commit** — `git commit -m "feat: port Stage-2b trajectory-geometry engine"`

---

### Task 9: Port `decode_ode_trajectory.py` + `visualize_ode_stepwise.py`

**Files:**
- Create: `scripts/omniavatar/decode_ode_trajectory.py`, `scripts/omniavatar/visualize_ode_stepwise.py`

**Interfaces:**
- Consumes: `models.wan_vae.load_wan_vae` (decode script only).

- [ ] **Step 1: `decode_ode_trajectory.py`:** replace module-level `from OmniAvatar.models.model_manager import ModelManager` (line 30) + the load block (lines ~108-117) with `load_wan_vae` (exactly as Task 7 Step 1). Repo-root `sys.path` insert. Env pattern for mask/VAE defaults.
- [ ] **Step 2: `visualize_ode_stepwise.py`:** reads mp4s only — docstring paths (lines 13-15) get env-root examples; mask default → env pattern. No import surgery.
- [ ] **Step 3: Gates** — syntax, `--help` under `$PY_OMNI` for both, grep gate.
- [ ] **Step 4: Commit** — `git commit -m "feat: port trajectory decode + stepwise visualization scripts"`

---

### Task 10: Port `simulate_euler_and_decode.py` + `spatial_cfg_probe.py` + 2 spatial plotters

**Files:**
- Create: `scripts/omniavatar/simulate_euler_and_decode.py`, `scripts/omniavatar/spatial_cfg_probe.py`, `scripts/omniavatar/plot_spatial_cfg_probe.py`, `scripts/omniavatar/plot_spatial_cfg_heatmaps.py`

**Interfaces:**
- Consumes: `models.omniavatar_wan.OmniAvatarWan`, `models.wan_vae.load_wan_vae`.

- [ ] **Step 1: `simulate_euler_and_decode.py`:** line 30 FastGen `sys.path` → repo-root insert + vendored import; module constants (lines 37-46: `PRETRAINED`, `CKPT_14B`, `MASK_PATH`, `DATA_DIR`, `NEG_TEXT_EMB`) → env pattern; it imports BOTH FastGen and `OmniAvatar.models.model_manager` — replace the latter with `load_wan_vae`; docstring DEAD paths (lines 12-13) → live roots.
- [ ] **Step 2: `spatial_cfg_probe.py`:** line 55 FastGen `sys.path` → repo-root insert + vendored import; docstring paths (lines 24-38) already use live `.local` roots — convert to env-root examples.
- [ ] **Step 3: The 2 plotters:** `--probe_dir` defaults → `os.environ.get("ODE_ANALYSIS_ROOT_OMNI", ...) + "/spatial_cfg_probe"`.
- [ ] **Step 4: Gates** — syntax ×4, `--help` ×4 (`$PY_FASTGEN` for the two generators since they load the teacher, `$PY_OMNI` for plotters), grep gate ×4.
- [ ] **Step 5: Commit** — `git commit -m "feat: port euler-simulate, spatial CFG probe + spatial plotters"`

---

### Task 11: Port sequential-metrics + mouthweight launchers

**Files:**
- Create: `scripts/omniavatar/run_all_metrics_sequential.sh`, `scripts/omniavatar/run_mouthweight_generation.sh`, `scripts/omniavatar/run_mouthweight_evaluation.sh`

**Interfaces:**
- Consumes: script names/CLIs from Tasks 5-8 (`scripts/omniavatar/eval_ode_perceptual_v2.py`, `analyze_ode_trajectory.py`, `generate_omniavatar_ode_pairs_full.py`, `generate_single_step_predictions.py`).

- [ ] **Step 1: `run_all_metrics_sequential.sh`:** bash pattern; lines 13-18 (`cd`, `PY`, `MASK`, `STATE_DIR`) and 52-55 (`TRAJ_*`, `OUT_*`) → env-root derived (`STATE_DIR="${ODE_ANALYSIS_ROOT_OMNI:-...}/_seq_state"`); every `python scripts/<name>.py` → `"$PY_OMNI" "$REPO_ROOT/scripts/omniavatar/<name>.py"`. The latentsync TRAJ vars stay (results-only scope — the launcher may still reference them; keep entries but they're only used if those dirs exist).
- [ ] **Step 2: Mouthweight pair:** env pattern (`MOUTHWEIGHT_CKPT`, `WEIGHTS_ROOT`, `RECON_DATA_DIR`, `MASK_PATH`, `NEG_TEXT_EMB`, roots); `FASTGEN_PY` → `"$PY_FASTGEN"`; generation script path → `"$REPO_ROOT/scripts/omniavatar/generate_omniavatar_ode_pairs_full.py"`; delete `cd` lines.
- [ ] **Step 3: Gates** — `bash -n` ×3, grep gate ×3.
- [ ] **Step 4: Commit** — `git commit -m "feat: port sequential-metrics + mouthweight launchers"`

---

### Task 12: Port plotters batch A (5 files)

**Files:**
- Create in `scripts/omniavatar/`: `plot_combined_ode_comparison.py`, `plot_combined_ode_comparison_audio_only.py`, `plot_combined_ode_comparison_latentsync.py`, `plot_trajectory_cfg_comparison.py`, `plot_cfg_mode_compare.py`

**Interfaces:**
- Produces: same figures; all CSV paths resolved as `os.path.join(ANALYSIS_ROOT, "<relative>")` where `ANALYSIS_ROOT = os.environ.get("ODE_ANALYSIS_ROOT_OMNI", "/home/work/.local/ode_analysis")`, overridable via `--analysis_root`.

- [ ] **Step 1: Per plotter, transform the module-level dict.** Pattern (from `plot_combined_ode_comparison.py` lines 21-57):

```python
# old
"path": "/home/work/.local/ode_analysis/14B/perceptual_v2/metrics.csv",
# new
"path": "14B/perceptual_v2/metrics.csv",   # joined with ANALYSIS_ROOT at load time
```

Add once per file:

```python
ANALYSIS_ROOT = os.environ.get("ODE_ANALYSIS_ROOT_OMNI", "/home/work/.local/ode_analysis")
```

plus `parser.add_argument("--analysis_root", default=ANALYSIS_ROOT)` and join at the single place paths are consumed. `TRAJ_DIR` constants → `ODE_TRAJ_ROOT_OMNI` env pattern. Dead paths → live.

- [ ] **Step 2: Gates** — syntax ×5, `--help` ×5 under `$PY_OMNI`, grep gate ×5.
- [ ] **Step 3: Commit** — `git commit -m "feat: port comparison plotters batch A (env-rooted CSV paths)"`

---

### Task 13: Port plotters batch B (3 files)

**Files:**
- Create in `scripts/omniavatar/`: `plot_all_models_compare.py`, `plot_exp1_schedule_compare.py`, `plot_mouthweight_ode_results.py`

Same recipe as Task 12 (env-rooted relative paths, `--analysis_root` override; `plot_mouthweight_ode_results.py` lines 28-29 `ANALYSIS_ROOT`/`TRAJ_DIR` → env pattern). Gates ×3, then:

```bash
git commit -m "feat: port comparison plotters batch B"
```

---

### Task 14: Results migration + experiment registry

**Files:**
- Create: `results/omniavatar/data/` ← the 28 files from `/home/work/.local/ode_analysis/all_csvs/` (27 CSVs + `README.md`)
- Create: `configs/registry.yaml`

**Interfaces:**
- Produces: `configs/registry.yaml` consumed by Task 16 (`load_registry()`); committed CSVs at `results/omniavatar/data/<name>.csv`.

- [ ] **Step 1: Copy and commit the CSVs**

```bash
cp /home/work/.local/ode_analysis/all_csvs/*.csv /home/work/.local/ode_analysis/all_csvs/README.md \
   /home/work/.local/infinitetalk-ode-analysis/results/omniavatar/data/
```

- [ ] **Step 2: Write `configs/registry.yaml`.** Full content — schema plus one entry per committed CSV (all 27 OmniAvatar-side) and the 7 InfiniteTalk perceptual CSVs + 7 straightness JSONs already in `results/infinitetalk/data/`:

```yaml
schema_version: 1
models:
  omniavatar:
    preprocessing: {resolution: 512, crop: pre-aligned latentsync crop, mask: latentsync_png,
                    regions: [mouth, upper_face, full], gt_row: -1}
  infinitetalk:
    preprocessing: {resolution: 640, crop: force-square resize, mask: dlib_bbox,
                    regions: [mouth, full], gt_row: -1}
  latentsync:      # results-only: no generation/analysis scripts in this repo
    preprocessing: {resolution: 512, crop: pre-aligned latentsync crop, mask: latentsync_png,
                    regions: [mouth, upper_face, full], gt_row: -1}
metric_rules:      # cross-model comparability
  sync_c: face_value
  sync_d: face_value
  mse: normalized_only
  ssim: normalized_only
  lpips: normalized_only
  lmd: normalized_only
  sharpness: normalized_only
experiments:
  # --- OmniAvatar, text+audio CFG drop mode ---
  - {id: omni_ta_default,        model: omniavatar, family: trajectory,  guidance: {cfg: 4.5}, drop: text+audio, csv: results/omniavatar/data/14B_textaudio_perceptual_v2.csv}
  - {id: omni_ta_nocfg,          model: omniavatar, family: trajectory,  guidance: {cfg: 1.0}, drop: text+audio, csv: results/omniavatar/data/14B_textaudio_trajectory_nocfg.csv}
  - {id: omni_ta_cfg1.0,         model: omniavatar, family: trajectory,  guidance: {cfg: 1.0}, drop: text+audio, csv: results/omniavatar/data/14B_textaudio_cfg1.0_trajectory.csv}
  - {id: omni_ta_cfg3.0,         model: omniavatar, family: trajectory,  guidance: {cfg: 3.0}, drop: text+audio, csv: results/omniavatar/data/14B_textaudio_cfg3.0_trajectory.csv}
  - {id: omni_ta_cfg6.0,         model: omniavatar, family: trajectory,  guidance: {cfg: 6.0}, drop: text+audio, csv: results/omniavatar/data/14B_textaudio_cfg6.0_trajectory.csv}
  - {id: omni_ta_euler_on_on,    model: omniavatar, family: euler_jump,  knobs: {step0: cfg45, teacher: cfg45}, drop: text+audio, csv: results/omniavatar/data/14B_textaudio_euler_cfg45_cfg45.csv}
  - {id: omni_ta_euler_nocfg_on, model: omniavatar, family: euler_jump,  knobs: {step0: nocfg, teacher: cfg45}, drop: text+audio, csv: results/omniavatar/data/14B_textaudio_euler_nocfg_cfg45.csv}
  - {id: omni_ta_euler_nocfg_nocfg, model: omniavatar, family: euler_jump, knobs: {step0: nocfg, teacher: nocfg}, drop: text+audio, csv: results/omniavatar/data/14B_textaudio_euler_nocfg_nocfg.csv}
  - {id: omni_ta_euler_on_nocfg, model: omniavatar, family: euler_jump,  knobs: {step0: cfg45, teacher: nocfg}, drop: text+audio, csv: results/omniavatar/data/14B_textaudio_euler_cfg45_nocfg.csv}
  - {id: omni_ta_fresh_noise,    model: omniavatar, family: fresh_noise, guidance: {cfg: 4.5}, drop: text+audio, csv: results/omniavatar/data/14B_textaudio_fresh_noise.csv}
  - {id: omni_ta_schedule25,     model: omniavatar, family: scheduled_cfg, knobs: {tau: 25},   drop: text+audio, csv: results/omniavatar/data/14B_textaudio_schedule25.csv}
  # --- OmniAvatar, audio-only CFG drop mode (same 7 shapes) ---
  - {id: omni_ao_default,        model: omniavatar, family: trajectory,  guidance: {cfg: 4.5}, drop: audio-only, csv: results/omniavatar/data/14B_audioonly_perceptual_v2.csv}
  - {id: omni_ao_euler_on_on,    model: omniavatar, family: euler_jump,  knobs: {step0: cfg45, teacher: cfg45}, drop: audio-only, csv: results/omniavatar/data/14B_audioonly_euler_cfg45_cfg45.csv}
  - {id: omni_ao_euler_nocfg_on, model: omniavatar, family: euler_jump,  knobs: {step0: nocfg, teacher: cfg45}, drop: audio-only, csv: results/omniavatar/data/14B_audioonly_euler_nocfg_cfg45.csv}
  - {id: omni_ao_euler_nocfg_nocfg, model: omniavatar, family: euler_jump, knobs: {step0: nocfg, teacher: nocfg}, drop: audio-only, csv: results/omniavatar/data/14B_audioonly_euler_nocfg_nocfg.csv}
  - {id: omni_ao_euler_on_nocfg, model: omniavatar, family: euler_jump,  knobs: {step0: cfg45, teacher: nocfg}, drop: audio-only, csv: results/omniavatar/data/14B_audioonly_euler_cfg45_nocfg.csv}
  - {id: omni_ao_fresh_noise,    model: omniavatar, family: fresh_noise, guidance: {cfg: 4.5}, drop: audio-only, csv: results/omniavatar/data/14B_audioonly_fresh_noise.csv}
  - {id: omni_ao_schedule25,     model: omniavatar, family: scheduled_cfg, knobs: {tau: 25},   drop: audio-only, csv: results/omniavatar/data/14B_audioonly_schedule25.csv}
  # --- Spatial probe (OmniAvatar diagnostics, results-only for comparison purposes) ---
  - {id: omni_spatial_fresh,     model: omniavatar, family: spatial_probe, results_only: true, csv: results/omniavatar/data/spatial_probe_fresh_noise.csv}
  - {id: omni_spatial_cfg,       model: omniavatar, family: spatial_probe, results_only: true, csv: results/omniavatar/data/spatial_probe_trajectory_cfg.csv}
  - {id: omni_spatial_nocfg,     model: omniavatar, family: spatial_probe, results_only: true, csv: results/omniavatar/data/spatial_probe_trajectory_nocfg.csv}
  # --- LatentSync (results-only) ---
  - {id: ls_default,             model: latentsync, family: trajectory,  results_only: true, csv: results/omniavatar/data/latentsync_perceptual_v2.csv}
  - {id: ls_nocfg,               model: latentsync, family: trajectory,  results_only: true, csv: results/omniavatar/data/latentsync_trajectory_nocfg.csv}
  - {id: ls_euler_on_on,         model: latentsync, family: euler_jump,  results_only: true, csv: results/omniavatar/data/latentsync_euler_cfg15_cfg15.csv}
  - {id: ls_euler_nocfg_on,      model: latentsync, family: euler_jump,  results_only: true, csv: results/omniavatar/data/latentsync_euler_nocfg_cfg15.csv}
  - {id: ls_euler_nocfg_nocfg,   model: latentsync, family: euler_jump,  results_only: true, csv: results/omniavatar/data/latentsync_euler_nocfg_nocfg.csv}
  - {id: ls_fresh_noise,         model: latentsync, family: fresh_noise, results_only: true, csv: results/omniavatar/data/latentsync_fresh_noise.csv}
  # --- InfiniteTalk 7-config grid ---
  - {id: it_t1.0_a1.0,           model: infinitetalk, family: trajectory, guidance: {text: 1.0, audio: 1.0}, csv: results/infinitetalk/data/perceptual_t1.0_a1.0.csv}
  - {id: it_t2.5_a2.0,           model: infinitetalk, family: trajectory, guidance: {text: 2.5, audio: 2.0}, csv: results/infinitetalk/data/perceptual_t2.5_a2.0.csv}
  - {id: it_t5.0_a1.0,           model: infinitetalk, family: trajectory, guidance: {text: 5.0, audio: 1.0}, csv: results/infinitetalk/data/perceptual_t5.0_a1.0.csv}
  - {id: it_t5.0_a2.0,           model: infinitetalk, family: trajectory, guidance: {text: 5.0, audio: 2.0}, csv: results/infinitetalk/data/perceptual_t5.0_a2.0.csv}
  - {id: it_t5.0_a4.0,           model: infinitetalk, family: trajectory, guidance: {text: 5.0, audio: 4.0}, csv: results/infinitetalk/data/perceptual_t5.0_a4.0.csv}
  - {id: it_t5.0_a6.0,           model: infinitetalk, family: trajectory, guidance: {text: 5.0, audio: 6.0}, csv: results/infinitetalk/data/perceptual_t5.0_a6.0.csv}
  - {id: it_t7.5_a6.0,           model: infinitetalk, family: trajectory, guidance: {text: 7.5, audio: 6.0}, csv: results/infinitetalk/data/perceptual_t7.5_a6.0.csv}
  # InfiniteTalk per-step euler CSVs: GAP A — on the sweep machine, not yet committed
  - {id: it_euler_on_on,         model: infinitetalk, family: euler_jump, knobs: {step0: on, teacher: on},       csv: results/infinitetalk/data/euler_on_on_metrics.csv, status: missing_sweep_machine}
  - {id: it_euler_nocfg_on,      model: infinitetalk, family: euler_jump, knobs: {step0: nocfg, teacher: on},    csv: results/infinitetalk/data/euler_nocfg_on_metrics.csv, status: missing_sweep_machine}
  - {id: it_euler_nocfg_nocfg,   model: infinitetalk, family: euler_jump, knobs: {step0: nocfg, teacher: nocfg}, csv: results/infinitetalk/data/euler_nocfg_nocfg_metrics.csv, status: missing_sweep_machine}
  - {id: it_euler_on_nocfg,      model: infinitetalk, family: euler_jump, knobs: {step0: on, teacher: nocfg},    csv: results/infinitetalk/data/euler_on_nocfg_metrics.csv, status: missing_sweep_machine}
comparisons:   # cross-model pairs; assumption on=(t5,a4) <-> cfg4.5 carried here, not hardcoded in code
  - {name: default_trajectory, omniavatar: omni_ta_default, infinitetalk: it_t5.0_a4.0}
  - {name: nocfg_trajectory,   omniavatar: omni_ta_nocfg,   infinitetalk: it_t1.0_a1.0}
  - {name: euler_on_on,        omniavatar: omni_ta_euler_on_on,       infinitetalk: it_euler_on_on}
  - {name: euler_nocfg_on,     omniavatar: omni_ta_euler_nocfg_on,    infinitetalk: it_euler_nocfg_on}
  - {name: euler_nocfg_nocfg,  omniavatar: omni_ta_euler_nocfg_nocfg, infinitetalk: it_euler_nocfg_nocfg}
  - {name: euler_on_nocfg,     omniavatar: omni_ta_euler_on_nocfg,    infinitetalk: it_euler_on_nocfg}
```

Before committing, cross-check the euler knob→CSV mapping and metric names against `results/omniavatar/data/README.md` and one actual CSV header; fix any mismatch (e.g. metric name `sync_c` vs `Sync-C`) so `metric_rules` keys match the real CSV `metric` column values.

- [ ] **Step 3: Validate**

```bash
/home/work/.local/miniconda3/envs/omniavatar/bin/python -c "
import yaml; r = yaml.safe_load(open('configs/registry.yaml'))
import os
missing = [e['id'] for e in r['experiments'] if 'status' not in e and not os.path.exists(e['csv'])]
print('missing:', missing); assert not missing"
```

- [ ] **Step 4: Commit** — `git add results/omniavatar configs/registry.yaml && git commit -m "feat: commit OmniAvatar result CSVs + unified experiment registry"`

---

### Task 15: Docs — README rewrite, omniavatar-experiments.md, doc path updates

**Files:**
- Modify: `README.md`; Create: `docs/omniavatar-experiments.md`; Modify: `docs/status-and-todo.md`, `docs/cross-model-comparison.md`

- [ ] **Step 1: Rewrite `README.md`** to the two-model scope. Required sections: (1) what the repo is (both models, two-stage pipeline); (2) repo map (the tree from the spec); (3) env matrix table (the 3 envs, exact pins for `infinitetalk`: torch 2.4.1, transformers 4.49.0, diffusers 0.33.1, `TORCHDYNAMO_DISABLE=1`); (4) machine setup (`cp configs/machine-omniavatar.env.example configs/machine.env`); (5) per-model quickstart — the exact launcher commands for generate → analyze → plot on each side; (6) external dependencies: weights list (Wan2.1-T2V-14B shards, `Wan2.1_VAE.pth`, `step-10500.pt`, wav2vec2), InfiniteTalk upstream (pin the commit currently checked out at `/home/work/.local/InfiniteTalk` — record it with `git -C /home/work/.local/InfiniteTalk rev-parse HEAD`), `METRICS_ROOT` tooling. Preserve the five load-bearing InfiniteTalk facts from the current README (no-x0-head derivation, 3-call CFG, force-square 640, latent shape, timing) — move, don't delete.
- [ ] **Step 2: Write `docs/omniavatar-experiments.md`** — adapt `results/omniavatar/data/README.md` (the all_csvs inventory): the 6 experiment groups, per-experiment registry id, CSV, and regeneration command using the new launchers.
- [ ] **Step 3: Update `docs/status-and-todo.md`** — add a "Deferred re-runs" section: (a) OmniAvatar cfg4.5 sequential with latents + euler `--save_latents` runs (unblocks cross-model straightness), (b) InfiniteTalk Stage-2b re-run on sweep machine, (c) gap A commit from sweep machine. Update `docs/cross-model-comparison.md` transfer-gap section to point at the registry `status: missing_sweep_machine` entries.
- [ ] **Step 4: Commit** — `git commit -m "docs: two-model README, OmniAvatar experiment inventory, deferred-runs status"`

---

### Task 16: Comparison stage — `compare_models.py` (TDD)

**Files:**
- Create: `scripts/comparison/compare_models.py`, `tests/test_compare_models.py`, `tests/__init__.py`

**Interfaces:**
- Consumes: `configs/registry.yaml` (Task 14 schema).
- Produces: `load_registry(path) -> dict`; `build_comparison(registry, pair_name) -> pandas.DataFrame` with columns `[pair, model, experiment_id, step, sample, metric, region, value, value_norm, comparability]`; CLI `compare_models.py --registry configs/registry.yaml --out results/comparison/` writing `comparison_<pair>.csv` per pair plus `comparison_long.csv` (all pairs concatenated). Pairs whose CSV has `status: missing_sweep_machine` are skipped with a printed warning, not an error.

- [ ] **Step 1: Verify deps** — `$PY_OMNI -c "import pandas, yaml; print('ok')"` (if pandas missing: `$PY_OMNI -m pip install pandas`).

- [ ] **Step 2: Write the failing test** (`tests/test_compare_models.py`) — build two tiny fixture CSVs in `tmp_path` with the real schema `step,t,sample,metric,region,value` (2 samples × steps {-1, 0, 49} × metrics {mse, sync_c} × region mouth), a minimal registry dict pointing at them, and assert:

```python
import pandas as pd
from scripts.comparison.compare_models import load_registry, build_comparison

def _write_csv(path, model_tag):
    rows = ["step,t,sample,metric,region,value"]
    for sample in ("s1", "s2"):
        rows.append(f"-1,,{sample},mse,mouth,2.0")        # GT row
        rows.append(f"-1,,{sample},sync_c,mouth,8.0")
        for step, v in ((0, 4.0), (49, 1.0)):
            rows.append(f"{step},0.5,{sample},mse,mouth,{v}")
            rows.append(f"{step},0.5,{sample},sync_c,mouth,{v + (1 if model_tag == 'b' else 0)}")
    path.write_text("\n".join(rows))

def test_build_comparison_joins_and_normalizes(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_csv(a, "a"); _write_csv(b, "b")
    registry = {
        "metric_rules": {"sync_c": "face_value", "mse": "normalized_only"},
        "experiments": [
            {"id": "ea", "model": "omniavatar", "csv": str(a)},
            {"id": "eb", "model": "infinitetalk", "csv": str(b)},
        ],
        "comparisons": [{"name": "p", "omniavatar": "ea", "infinitetalk": "eb"}],
    }
    df = build_comparison(registry, "p")
    assert set(df["model"]) == {"omniavatar", "infinitetalk"}
    assert len(df) == 16                       # 2 models x 2 samples x 2 steps x 2 metrics (GT rows consumed, not emitted)
    mse0 = df[(df.model == "omniavatar") & (df.step == 0) & (df.metric == "mse")]
    assert (mse0["value_norm"] == 2.0).all()   # 4.0 / GT 2.0
    assert (df[df.metric == "sync_c"]["comparability"] == "face_value").all()
    assert (df[df.metric == "mse"]["comparability"] == "normalized_only").all()

def test_missing_csv_is_skipped(tmp_path, capsys):
    registry = {
        "metric_rules": {}, "comparisons": [{"name": "p", "omniavatar": "ea", "infinitetalk": "eb"}],
        "experiments": [
            {"id": "ea", "model": "omniavatar", "csv": str(tmp_path / "nope.csv"), "status": "missing_sweep_machine"},
            {"id": "eb", "model": "infinitetalk", "csv": str(tmp_path / "also_nope.csv")},
        ],
    }
    assert build_comparison(registry, "p") is None
    assert "skip" in capsys.readouterr().out.lower()
```

- [ ] **Step 3: Run to verify failure** — `cd /home/work/.local/infinitetalk-ode-analysis && $PY_OMNI -m pytest tests/test_compare_models.py -v` → FAIL (module not found).

- [ ] **Step 4: Implement `compare_models.py`.** `load_registry` = `yaml.safe_load`. `build_comparison`: resolve the pair's two experiment entries (skip → return `None` + print if either has `status:` or missing file); read each CSV with pandas; extract GT rows (`step == -1`) into a per-`(sample, metric, region)` lookup; drop GT rows from the body; `value_norm = value / gt_value` where a GT value exists, else `NaN`; `comparability` from `metric_rules` (default `normalized_only`); tag `pair/model/experiment_id`; concat both models. CLI `main()` iterates `registry["comparisons"]`, writes per-pair CSVs + `comparison_long.csv` into `--out`.

- [ ] **Step 5: Run tests** — expected PASS. Then run against the real registry:

```bash
$PY_OMNI scripts/comparison/compare_models.py --registry configs/registry.yaml --out results/comparison/
```

Expected: `default_trajectory` and `nocfg_trajectory` produce CSVs; the 4 euler pairs print skip warnings (gap A). Sanity: OmniAvatar side of `default_trajectory` has 6031 − 1000 (upper_face kept; join is on shared regions only — mouth/full) rows before region filtering; assert non-empty and both models present.

- [ ] **Step 6: Commit** — `git add scripts/comparison tests && git commit -m "feat: cross-model comparison join with normalization rules (TDD)"`

---

### Task 17: Comparison plots

**Files:**
- Create: `scripts/comparison/plot_comparison.py`

**Interfaces:**
- Consumes: `results/comparison/comparison_<pair>.csv` (Task 16 schema).
- Produces: per-pair per-metric per-region figures `results/comparison/figures/<pair>_<metric>_<region>.png` — two lines (one per model), x = step, y = `value_norm` (or raw `value` for `face_value` metrics), model-colored, titled with the pair name and a `(normalized)` / `(face value)` suffix.

- [ ] **Step 1: Write the plotter** — pandas groupby over `(metric, region)`, mean over samples per step, one matplotlib figure per group. CLI: `--comparison_dir results/comparison --out results/comparison/figures`. Regions limited to those present for BOTH models in the frame (inner join semantics — `upper_face` drops out naturally).
- [ ] **Step 2: Run on real output of Task 16** — expect figures for `default_trajectory` and `nocfg_trajectory` (mouth + full × available metrics). Visually spot-check one figure (Sync-C should show OmniAvatar and InfiniteTalk curves on comparable 0-10-ish scale).
- [ ] **Step 3: Commit** — `git add scripts/comparison results/comparison/figures && git commit -m "feat: cross-model comparison plots"`

---

### Task 18: Verify vendored model — load test + minimal Stage-1 smoke (GPU)

**Files:**
- Create: `tests/test_omniavatar_wan_load.py` (a script, run manually — the fastgen env may lack pytest; plain `python` script with asserts)

- [ ] **Step 1: Write the load test**

```python
"""Run: CUDA_VISIBLE_DEVICES=0 $PY_FASTGEN tests/test_omniavatar_wan_load.py"""
import os, sys, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.omniavatar_wan import OmniAvatarWan

# Mirror the constructor call used by scripts/omniavatar/generate_omniavatar_ode_pairs_full.py
# (open that file, copy its OmniAvatarWan(...) instantiation + checkpoint-load lines verbatim here).
# Assert: it constructs, loads TEACHER_CKPT without missing/unexpected keys, and
# sum(p.numel() for p in net.parameters()) > 14e9.
```

- [ ] **Step 2: Run it** — `CUDA_VISIBLE_DEVICES=0 $PY_FASTGEN tests/test_omniavatar_wan_load.py` → prints param count, exits 0.
- [ ] **Step 3: Minimal Stage-1 smoke.** Check `scripts/omniavatar/generate_omniavatar_ode_pairs_full.py --help` for a step-count flag; if none exists, add `--max_steps` (int, default None) that slices the timestep list after construction (`timesteps = timesteps[:args.max_steps]` — one line + argparse). Then:

```bash
CUDA_VISIBLE_DEVICES=0 $PY_FASTGEN scripts/omniavatar/generate_omniavatar_ode_pairs_full.py \
  --data_dir "$RECON_DATA_DIR" --output_dir /tmp/claude-1100/-home-work--local-OmniAvatar/2673fc09-e4c6-4c51-a460-b9624acf269e/scratchpad/ode_smoke \
  --max_steps 2   # plus whatever sample-limit flag the driver already has (limit to 1 sample)
```

Expected: `step_000_{xt,x0}.pt` + `step_001_{xt,x0}.pt` + `ode_schedule.json` for 1 sample, shapes `[16, 21, 64, 64]`.
- [ ] **Step 4: Commit** — `git add tests scripts/omniavatar && git commit -m "test: vendored OmniAvatarWan load test + Stage-1 smoke (--max_steps)"`

---

### Task 19: Verify vendored VAE — decode equivalence (GPU)

- [ ] **Step 1: Write + run a one-off equivalence check** (scratchpad script, not committed):

```python
# CUDA_VISIBLE_DEVICES=1 $PY_OMNI <this file>   (needs OmniAvatar repo importable for the reference side)
import sys, torch
sys.path.insert(0, "/home/work/.local/infinitetalk-ode-analysis")
sys.path.insert(0, "/home/work/.local/OmniAvatar")
from models.wan_vae import load_wan_vae
from OmniAvatar.models.model_manager import ModelManager
import os
vae_path = os.environ.get("VAE_PATH", "/home/work/.local/OmniAvatar/pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth")
mm = ModelManager(device="cpu", infer=True); mm.load_models([vae_path], torch_dtype=torch.bfloat16, device="cpu")
ref = mm.model[mm.model_name.index("wan_video_vae")].to("cuda")
ours = load_wan_vae(vae_path, dtype=torch.bfloat16, device="cuda")
# pick any real latent from the live analysis data:
lat = torch.load(sorted(__import__("glob").glob(
    os.environ.get("ODE_TRAJ_ROOT_OMNI", "/home/work/.local/ode_full_trajectories") + "/14B_cfg1.0/*/step_049_x0.pt"))[0],
    map_location="cuda").to(torch.bfloat16)
with torch.no_grad():
    a = ref.decode(lat, device="cuda", tiled=False)
    b = ours.decode(lat, device="cuda", tiled=False)
print("max abs diff:", (a - b).abs().max().item()); assert torch.equal(a, b), "NOT bit-identical"
```

Expected: `max abs diff: 0.0`. If the state-dict conversion differs, fix `loader.py` until bit-identical — the ModelManager path (model_manager.py lines 12-40) is the ground truth.
- [ ] **Step 2: Record the result** in the Task-20 commit message (no separate commit; the script lives in scratchpad).

---

### Task 20: Verify Stage-2 engines reproduce committed results (GPU-sharded)

- [ ] **Step 1: Stage-2a reproduction (GPU 2).** Run the ported engine on ONE existing trajectory config into a scratch dir, then diff against the live metrics for the same samples:

```bash
CUDA_VISIBLE_DEVICES=2 $PY_OMNI scripts/omniavatar/eval_ode_perceptual_v2.py \
  --traj_dir "${ODE_TRAJ_ROOT_OMNI}/14B_cfg1.0" \
  --output_dir /tmp/claude-1100/-home-work--local-OmniAvatar/2673fc09-e4c6-4c51-a460-b9624acf269e/scratchpad/verify_2a \
  --mask_path data/mask.png --vae_path "$VAE_PATH" --num_shards 10 --shard_id 0   # 1 sample
```

Then compare (`pandas` merge on `step,sample,metric,region`) against `${ODE_ANALYSIS_ROOT_OMNI}/14B_cfg1.0`'s metrics for that sample: MSE/SSIM/LPIPS/sharpness identical to ~1e-6; Sync-C/D allowed small tolerance (SyncNet is deterministic given same env — expect exact; investigate if not).
- [ ] **Step 2: Stage-2b reproduction (GPU 3, fast).** `analyze_ode_trajectory.py --traj_dir ${ODE_TRAJ_ROOT_OMNI}/14B_cfg1.0 --mask_path data/mask.png --output_dir <scratch>` → diff JSON/CSV outputs against the live `14B_cfg1.0` analysis dir (exact match expected — pure tensor math).
- [ ] **Step 3: Plotter regeneration.** Run `plot_combined_ode_comparison.py` and `plot_cfg_mode_compare.py` with default env roots into a scratch `--output_dir`; open the PNGs and confirm they are structurally identical to the existing figures under `${ODE_ANALYSIS_ROOT_OMNI}/14B/combined`.
- [ ] **Step 4: Whole-repo grep gate**

```bash
grep -rn "/home/work" scripts/ models/ | grep -v "environ.get\|:-\|^ *#\|Binary" | grep -v "^[^:]*:[0-9]*: *#"
```

Expected: empty (docs/ and reference/ are exempt).
- [ ] **Step 5: Commit** any fixes surfaced — `git commit -m "test: Stage-2 engines reproduce committed metrics (verification fixes)"` — and record the Task-19 VAE equivalence result in this message.

---

### Task 21: InfiniteTalk-side sanity (parallel with 18-20)

- [ ] **Step 1: Syntax + `--help` gates** on all `scripts/infinitetalk/*.py` under `$PY_IT` (generation scripts) and `$PY_OMNI` (plotters). The two Stage-2 engines' `--help` must work under `$PY_OMNI`.
- [ ] **Step 2: Straightness script check** — `$PY_OMNI scripts/common/measure_euler_straightness.py --help` works.
- [ ] **Step 3: InfiniteTalk plotter regeneration** — rerun `plot_default_vs_baseline.py` (reads committed `results/infinitetalk/data/`) into scratch; confirm it still produces its figures after the Task-2 path updates.
- [ ] **Step 4: Commit** fixes if any — `git commit -m "test: InfiniteTalk scripts pass gates post-restructure"`

---

### Task 22: Finalization — cleanup commits, rename, push, memory

- [ ] **Step 1: Cleanup commit in OmniAvatar-Train** (`/home/work/.local/OmniAvatar`):

```bash
cd /home/work/.local/OmniAvatar
git rm scripts/analyze_ode_trajectory_infinitetalk.py scripts/eval_ode_perceptual_v2_infinitetalk.py \
       scripts/eval_ode_perceptual_v2.py scripts/eval_ode_perceptual.py scripts/analyze_ode_trajectory.py \
       scripts/generate_single_step_predictions.py scripts/simulate_euler_and_decode.py \
       scripts/decode_ode_trajectory.py scripts/visualize_ode_stepwise.py scripts/spatial_cfg_probe.py \
       scripts/fix_euler_nocfg_cfg15_step0.py scripts/run_eval_ode_perceptual.sh scripts/run_eval_ode_perceptual_v2.sh \
       scripts/run_single_step_both.sh scripts/run_all_metrics_sequential.sh \
       scripts/generate_ode_trajectories.sh scripts/generate_ode_nocfg.sh scripts/generate_ode_no_audio.sh \
       scripts/run_mouthweight_generation.sh scripts/run_mouthweight_evaluation.sh \
       scripts/plot_combined_ode_comparison.py scripts/plot_combined_ode_comparison_audio_only.py \
       scripts/plot_combined_ode_comparison_latentsync.py scripts/plot_trajectory_cfg_comparison.py \
       scripts/plot_cfg_mode_compare.py scripts/plot_all_models_compare.py scripts/plot_exp1_schedule_compare.py \
       scripts/plot_mouthweight_ode_results.py scripts/plot_spatial_cfg_probe.py scripts/plot_spatial_cfg_heatmaps.py
git commit -m "chore: ODE analysis suite moved to talking-head-ode-analysis repo"
```

(Before running: `git ls-files` each name — some may be untracked (`rm` instead) or already absent (skip). Untracked ODE scripts listed in the repo's git status, e.g. `analyze_ode_trajectory_infinitetalk.py`, may need plain `rm`.)
- [ ] **Step 2: Cleanup in FastGen** (`/home/work/.local/hyunbin/FastGen`): `rm scripts/generate_infinitetalk_ode_pairs_full.py` (untracked); `git rm scripts/run_infinitetalk_ode_sweep.sh scripts/generate_omniavatar_ode_pairs_full.py scripts/verify_ode_trajectory.py scripts/run_ode_full_trajectory.sh`; KEEP `generate_omniavatar_ode_pairs.py` and `generate_ode_trajectories.py` (distillation assets) and the `fastgen/networks/OmniAvatar/` package. Commit: `"chore: ODE study drivers moved to talking-head-ode-analysis repo"`. Push per repo norms (SSH — see memory `git-push-auth`; FastGen remote is HTTPS with dead token → switch to `git@github.com:paulcho98/FastGen.git` first).
- [ ] **Step 3: Rename.** ASK THE USER to rename the repo on GitHub (Settings → General → Rename to `talking-head-ode-analysis`) — no `gh` CLI on this box. After confirmation:

```bash
cd /home/work/.local/infinitetalk-ode-analysis
git remote set-url origin git@github.com:paulcho98/talking-head-ode-analysis.git
cd /home/work/.local && mv infinitetalk-ode-analysis talking-head-ode-analysis
```

(If the user defers the GitHub rename, skip `set-url` — pushes to the old URL keep working, and redirect after a later rename.)
- [ ] **Step 4: Push the unified repo** — `git push origin main` (SSH key `~/.ssh/id_rsa`).
- [ ] **Step 5: Update memory** — edit `ode-analysis-pipeline.md` in the Claude memory dir: new repo name/path, "unified repo now canonical for BOTH models' ODE code", stale-copy cleanup done, deferred re-runs list. Update `git-push-auth.md` if FastGen's remote changed to SSH.

---

## Self-review checklist (run after writing, fixed inline)

1. **Spec coverage** — layout ✓(T1,T2), vendoring ✓(T3,T4), Stage-1 ✓(T5), euler+`--save_latents` ✓(T6), engines ✓(T7,T8), aux ✓(T9,T10), launchers ✓(T5,T6,T7,T11), 10 plotters ✓(T10×2,T12×5,T13×3), results+registry ✓(T14), docs ✓(T15), comparison ✓(T16,T17), verification ✓(T18-T21), cleanup+rename ✓(T22), excluded files never referenced ✓.
2. **Placeholders** — Task 3 Step 3 contains a deliberate verbatim-copy instruction (source file + line given); Task 18 Step 1 likewise (instantiation copied from the driver). These reference exact sources, not unwritten designs.
3. **Type consistency** — env var names, `load_wan_vae` signature, registry schema keys (`id/model/family/csv/status`), and comparison DataFrame columns are used identically across tasks.
