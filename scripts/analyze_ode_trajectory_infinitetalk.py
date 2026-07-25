"""Analyze InfiniteTalk ODE trajectory data to identify which timesteps drive lip/audio generation.

Adapted from ``scripts/analyze_ode_trajectory.py`` (OmniAvatar) for InfiniteTalk trajectories.

Key differences vs the OmniAvatar base script (see CHANGELOG at bottom of this docstring):
  * Latent grid is 80x80 (640px pixel frame), not 64x64 (512px).
  * InfiniteTalk has NO LatentSync latent mask. The mouth region is derived per-sample from
    dlib 68-pt facial landmarks on a decoded reference frame (or a GT video frame), then the
    mouth bounding box is down-projected to the 80x80 latent grid. Regions are "mouth" + "full"
    (the OmniAvatar "upper_face" region has no analogue and is dropped).
  * Ground truth for the x0-vs-GT comparison is either (a) the VAE-encoding of the original
    Hallo3 clip (``--gt_mode encode``, the requested default) or (b) the FINAL x0 used as a
    convergence proxy (``--gt_mode final_x0``). ``input_latents.pt`` is a single reference frame
    here and is NOT usable as GT.

Analyses (unchanged in spirit):
  1. x0_pred vs GT similarity (MSE, cosine) in mouth vs full regions across timesteps
  2. Δ-contribution per step (similarity improvement over previous step)
  3. Audio ablation: ||x0_with_audio - x0_no_audio|| per step (optional; skipped if no no-audio run)
  4. Inter-sample variance per step

Environment note:
  WanVAE lives in the ``infinitetalk`` conda env; dlib does NOT (it is available in ``omniavatar``
  and ``latentsync-metrics``). See ``--mask_source`` / ``--mouth_mask_cache`` for a decoupled path.

Usage:
    # Single env (requires dlib installed in the infinitetalk env — see notes):
    python scripts/analyze_ode_trajectory_infinitetalk.py \
        --traj_dir /home/work/.local/ode_full_trajectories_infinitetalk/infinitetalk_t5.0_a4.0 \
        --output_dir /home/work/ode_analysis_infinitetalk/t5.0_a4.0

    # With audio ablation (requires a matching no-audio trajectory dir — we currently have none):
    python scripts/analyze_ode_trajectory_infinitetalk.py \
        --traj_dir .../infinitetalk_t5.0_a4.0 \
        --no_audio_traj_dir .../infinitetalk_t5.0_a4.0_no_audio \
        --output_dir /home/work/ode_analysis_infinitetalk/t5.0_a4.0
"""

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Defaults / constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_INFINITETALK_ROOT = os.environ.get(
    "INFINITETALK_ROOT",
    "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk",
)
DEFAULT_VAE_PTH = os.path.join(DEFAULT_INFINITETALK_ROOT, "weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth")
DEFAULT_SHAPE_PREDICTOR = os.path.join(
    os.environ.get("METRICS_ROOT", "/data/karlo-research_715/workspace/kinemaar/paul/eval_metrics"),
    "shape_predictor_68_face_landmarks.dat",
)
DEFAULT_GT_VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "recon_clips")  # bundled recon clips

FRAME_SIZE = 640          # pixel frame is 640x640
LATENT_H = LATENT_W = 80  # latent grid is 80x80 (640 / 8)
VAE_SPATIAL_STRIDE = 8    # latent → pixel factor
NUM_FRAMES = 81           # 81 video frames → 21 latent frames ((81-1)/4 + 1)

# Regions analysed (OmniAvatar had "mouth"/"upper_face"; InfiniteTalk has "mouth"/"full").
REGIONS = ["mouth", "full"]


# ─────────────────────────────────────────────────────────────────────────────
# WanVAE (lazy singleton — only loaded when we actually need to decode/encode)
# ─────────────────────────────────────────────────────────────────────────────

_VAE_CACHE = {"vae": None}


def get_vae(args) -> "object":
    """Load InfiniteTalk's WanVAE once and cache it. Requires the ``infinitetalk`` env."""
    if _VAE_CACHE["vae"] is None:
        if args.infinitetalk_root not in sys.path:
            sys.path.insert(0, args.infinitetalk_root)
        from wan.modules.vae import WanVAE  # noqa: E402  (deferred import, needs sys.path)
        print(f"[VAE] Loading WanVAE from {args.vae_pth} on {args.device} ...")
        _VAE_CACHE["vae"] = WanVAE(vae_pth=args.vae_pth, device=args.device, dtype=torch.float)
    return _VAE_CACHE["vae"]


# ─────────────────────────────────────────────────────────────────────────────
# Mouth-mask derivation (dlib landmarks → padded bbox → latent grid)
# ─────────────────────────────────────────────────────────────────────────────

_DLIB_CACHE = {"detector": None, "predictor": None}


def _get_dlib(shape_predictor_path: str):
    """Lazily construct dlib detector + 68-pt predictor. dlib import deferred so that a
    fully-cached run (all masks precomputed) needs no dlib at all."""
    if _DLIB_CACHE["detector"] is None:
        import dlib  # noqa: E402  (deferred — not in the infinitetalk env by default)
        _DLIB_CACHE["detector"] = dlib.get_frontal_face_detector()
        _DLIB_CACHE["predictor"] = dlib.shape_predictor(shape_predictor_path)
    return _DLIB_CACHE["detector"], _DLIB_CACHE["predictor"]


def detect_mouth_points(img_rgb_uint8: np.ndarray, shape_predictor_path: str) -> Optional[np.ndarray]:
    """Run dlib 68-pt landmarks on a HxWx3 uint8 RGB image; return mouth pts[48:68] as [20,2].

    Returns None if no face is detected (caller falls back to a default lower-face box)."""
    detector, predictor = _get_dlib(shape_predictor_path)
    dets = detector(img_rgb_uint8, 1)
    if len(dets) == 0:
        return None
    # Pick the largest detected face.
    rect = max(dets, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
    shape = predictor(img_rgb_uint8, rect)
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)
    return pts[48:68]  # 20 mouth landmarks


def mouth_bbox_to_latent_mask(
    mouth_pts: Optional[np.ndarray],
    pad_frac: float,
    frame_size: int = FRAME_SIZE,
    latent_h: int = LATENT_H,
    latent_w: int = LATENT_W,
) -> torch.Tensor:
    """Convert mouth landmark points (pixel space) → padded bbox → boolean [H, W] latent mask.

    If ``mouth_pts`` is None, fall back to a centred lower-face box.
    Returns: [latent_h, latent_w] bool tensor, True = mouth region.
    """
    if mouth_pts is None or len(mouth_pts) == 0:
        # Fallback: centred lower-face box (x∈[0.30,0.70], y∈[0.58,0.88] of the frame).
        x0, x1 = 0.30 * frame_size, 0.70 * frame_size
        y0, y1 = 0.58 * frame_size, 0.88 * frame_size
    else:
        x0, y0 = float(mouth_pts[:, 0].min()), float(mouth_pts[:, 1].min())
        x1, y1 = float(mouth_pts[:, 0].max()), float(mouth_pts[:, 1].max())
        pad_x = pad_frac * max(x1 - x0, 1.0)
        pad_y = pad_frac * max(y1 - y0, 1.0)
        x0, x1 = x0 - pad_x, x1 + pad_x
        y0, y1 = y0 - pad_y, y1 + pad_y

    # Clamp to frame, then down-project to the latent grid (divide by VAE spatial stride).
    x0 = max(0.0, min(x0, frame_size))
    x1 = max(0.0, min(x1, frame_size))
    y0 = max(0.0, min(y0, frame_size))
    y1 = max(0.0, min(y1, frame_size))

    lx0 = int(np.floor(x0 / VAE_SPATIAL_STRIDE))
    lx1 = int(np.ceil(x1 / VAE_SPATIAL_STRIDE))
    ly0 = int(np.floor(y0 / VAE_SPATIAL_STRIDE))
    ly1 = int(np.ceil(y1 / VAE_SPATIAL_STRIDE))

    lx0 = max(0, min(lx0, latent_w))
    lx1 = max(0, min(lx1, latent_w))
    ly0 = max(0, min(ly0, latent_h))
    ly1 = max(0, min(ly1, latent_h))
    # Guarantee at least one latent cell.
    if lx1 <= lx0:
        lx1 = min(latent_w, lx0 + 1)
    if ly1 <= ly0:
        ly1 = min(latent_h, ly0 + 1)

    mask = torch.zeros(latent_h, latent_w, dtype=torch.bool)
    mask[ly0:ly1, lx0:lx1] = True
    return mask


def _to_uint8_rgb(frame_chw_m1p1: torch.Tensor) -> np.ndarray:
    """[3, H, W] float in [-1, 1] → HxWx3 uint8 RGB numpy array."""
    f = ((frame_chw_m1p1.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255)
    return f.permute(1, 2, 0).to(torch.uint8).cpu().numpy()


def reference_frame_from_latent(args, sample_name: str) -> np.ndarray:
    """Decode ``input_latents.pt`` (the [16, 1, 80, 80] reference latent) via WanVAE → 640px RGB frame."""
    ref_latent = load_tensor(args.traj_dir, sample_name, "input_latents.pt")  # [16, 1, 80, 80]
    vae = get_vae(args)
    with torch.no_grad():
        dec = vae.decode([ref_latent.to(args.device)])[0]  # [3, 1, 640, 640]
    return _to_uint8_rgb(dec[:, 0])


def reference_frame_from_gt_video(args, sample_name: str) -> np.ndarray:
    """Read the first frame of the GT Hallo3 clip, resized to 640x640 → RGB uint8. No WanVAE needed.

    Enables deriving masks in an env that has dlib but not WanVAE (e.g. omniavatar/latentsync-metrics)."""
    import cv2  # noqa: E402
    gt_path = gt_video_path(args, sample_name)
    cap = cv2.VideoCapture(gt_path)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame of GT video: {gt_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Match resize_and_centercrop: cover-scale (shorter side -> FRAME_SIZE) then center-crop.
    h0, w0 = frame_rgb.shape[:2]
    scale = FRAME_SIZE / min(h0, w0)
    nh, nw = int(np.ceil(scale * h0)), int(np.ceil(scale * w0))
    frame_rgb = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    top, left = (nh - FRAME_SIZE) // 2, (nw - FRAME_SIZE) // 2
    frame_rgb = frame_rgb[top:top + FRAME_SIZE, left:left + FRAME_SIZE]
    return frame_rgb.astype(np.uint8)


def build_mouth_masks(args, samples: List[str]) -> Dict[str, torch.Tensor]:
    """Build (or load from cache) a per-sample [80, 80] boolean mouth mask.

    Per-sample (not shared) because InfiniteTalk faces are NOT canonically aligned across clips.
    """
    masks: Dict[str, torch.Tensor] = {}
    cache_dir = args.mouth_mask_cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    for sample_name in samples:
        cache_path = os.path.join(cache_dir, f"{sample_name}.npy") if cache_dir else None
        if cache_path and os.path.isfile(cache_path):
            masks[sample_name] = torch.from_numpy(np.load(cache_path)).bool()
            continue

        # Derive the reference RGB frame, then landmarks → latent mask.
        if args.mask_source == "ref_decode":
            img = reference_frame_from_latent(args, sample_name)
        elif args.mask_source == "gt_frame":
            img = reference_frame_from_gt_video(args, sample_name)
        else:
            raise ValueError(f"Unknown --mask_source: {args.mask_source}")

        mouth_pts = detect_mouth_points(img, args.shape_predictor)
        if mouth_pts is None:
            print(f"[Mask] {sample_name}: no face detected — using fallback lower-face box")
        mask = mouth_bbox_to_latent_mask(mouth_pts, args.mouth_pad)
        masks[sample_name] = mask

        if cache_path:
            np.save(cache_path, mask.cpu().numpy())

        frac = mask.float().mean().item()
        print(f"[Mask] {sample_name}: mouth covers {frac:.1%} of the {LATENT_H}x{LATENT_W} latent grid")

    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Region selection + metrics
# ─────────────────────────────────────────────────────────────────────────────

def region_spatial_mask(mouth_mask: torch.Tensor, region: str) -> torch.Tensor:
    """Return an [H, W] bool mask selecting a region.

    region: "mouth" (mouth bbox) or "full" (entire latent grid).
    """
    if region == "mouth":
        return mouth_mask
    elif region == "full":
        return torch.ones_like(mouth_mask, dtype=torch.bool)
    raise ValueError(f"Unknown region: {region}")


def apply_region_mask(tensor: torch.Tensor, mouth_mask: torch.Tensor, region: str) -> torch.Tensor:
    """Extract values from a [C, T, H, W] tensor for a spatial region → flat tensor."""
    spatial_mask = region_spatial_mask(mouth_mask, region)
    expanded = spatial_mask.unsqueeze(0).unsqueeze(0).expand_as(tensor)
    return tensor[expanded]


def compute_mse(a: torch.Tensor, b: torch.Tensor, mouth_mask: torch.Tensor, region: str) -> float:
    va = apply_region_mask(a, mouth_mask, region)
    vb = apply_region_mask(b, mouth_mask, region)
    return (va - vb).pow(2).mean().item()


def compute_cosine_sim(a: torch.Tensor, b: torch.Tensor, mouth_mask: torch.Tensor, region: str) -> float:
    va = apply_region_mask(a, mouth_mask, region).float()
    vb = apply_region_mask(b, mouth_mask, region).float()
    if va.norm() < 1e-8 or vb.norm() < 1e-8:
        return 0.0
    return F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()


def compute_l2_norm(a: torch.Tensor, mouth_mask: torch.Tensor, region: str) -> float:
    va = apply_region_mask(a, mouth_mask, region).float()
    return va.norm().item()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def discover_samples(traj_dir: str) -> List[str]:
    """Find sample subdirectories that have complete trajectory data."""
    samples = []
    for name in sorted(os.listdir(traj_dir)):
        d = os.path.join(traj_dir, name)
        if not os.path.isdir(d):
            continue
        if os.path.isfile(os.path.join(d, "ode_schedule.json")) and os.path.isfile(os.path.join(d, "input_latents.pt")):
            samples.append(name)
    return samples


def load_schedule(traj_dir: str, sample_name: str) -> dict:
    with open(os.path.join(traj_dir, sample_name, "ode_schedule.json")) as f:
        return json.load(f)


def load_tensor(traj_dir: str, sample_name: str, filename: str) -> torch.Tensor:
    path = os.path.join(traj_dir, sample_name, filename)
    return torch.load(path, map_location="cpu", weights_only=True).float()


def sample_hash(sample_name: str) -> str:
    """Trajectory dir name → Hallo3 GT hash (strip the ``_shot...`` suffix)."""
    return sample_name.split("_shot")[0]


def gt_video_path(args, sample_name: str) -> str:
    return os.path.join(args.gt_video_dir, f"{sample_hash(sample_name)}.mp4")


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth latent provider
# ─────────────────────────────────────────────────────────────────────────────

def read_gt_video(path: str, num_frames: int = NUM_FRAMES, size: int = FRAME_SIZE) -> torch.Tensor:
    """Read the first ``num_frames`` frames, resize to size×size, → [3, T, H, W] float in [-1, 1]."""
    import cv2  # noqa: E402
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < num_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Match InfiniteTalk's resize_and_centercrop: cover-scale (shorter side -> size)
        # then center-crop, so GT aligns with the center-cropped reference the model saw.
        h0, w0 = frame_rgb.shape[:2]
        scale = size / min(h0, w0)
        nh, nw = int(np.ceil(scale * h0)), int(np.ceil(scale * w0))
        frame_rgb = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        top, left = (nh - size) // 2, (nw - size) // 2
        frame_rgb = frame_rgb[top:top + size, left:left + size]
        frames.append(frame_rgb)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    # Pad by repeating the last frame if the clip is shorter than num_frames.
    while len(frames) < num_frames:
        frames.append(frames[-1])
    arr = np.stack(frames, axis=0).astype(np.float32)          # [T, H, W, 3]
    vid = torch.from_numpy(arr).permute(3, 0, 1, 2)            # [3, T, H, W]
    vid = vid / 127.5 - 1.0                                    # → [-1, 1]
    return vid


def encode_gt_latent(args, sample_name: str) -> torch.Tensor:
    """VAE-encode the first 81 frames of the GT clip → [16, 21, 80, 80] latent (CPU float)."""
    vid = read_gt_video(gt_video_path(args, sample_name), NUM_FRAMES, FRAME_SIZE)
    vae = get_vae(args)
    with torch.no_grad():
        lat = vae.encode([vid.to(args.device)])[0]  # [16, 21, 80, 80]
    return lat.float().cpu()


def make_gt_provider(args, num_steps: int) -> Callable[[str], torch.Tensor]:
    """Return a function sample_name → GT latent [16, 21, 80, 80], per --gt_mode."""
    if args.gt_mode == "final_x0":
        final_idx = num_steps - 1
        return lambda s: load_tensor(args.traj_dir, s, f"step_{final_idx:03d}_x0.pt")
    elif args.gt_mode == "encode":
        return lambda s: encode_gt_latent(args, s)
    raise ValueError(f"Unknown --gt_mode: {args.gt_mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1 & 2: x0_pred vs GT similarity + Δ-contribution
# ─────────────────────────────────────────────────────────────────────────────

def analyze_gt_similarity(
    traj_dir: str,
    samples: List[str],
    masks: Dict[str, torch.Tensor],
    num_steps: int,
    get_gt: Callable[[str], torch.Tensor],
) -> Dict[str, np.ndarray]:
    """Compute x0_pred vs GT similarity at each step, for mouth and full regions (per-sample masks)."""
    metrics = {
        f"{region}_{metric}": np.zeros(num_steps)
        for region in REGIONS
        for metric in ["mse", "cosine"]
    }

    for sample_name in samples:
        gt = get_gt(sample_name)
        mouth_mask = masks[sample_name]
        for step_i in range(num_steps):
            x0 = load_tensor(traj_dir, sample_name, f"step_{step_i:03d}_x0.pt")
            for region in REGIONS:
                metrics[f"{region}_mse"][step_i] += compute_mse(x0, gt, mouth_mask, region)
                metrics[f"{region}_cosine"][step_i] += compute_cosine_sim(x0, gt, mouth_mask, region)

    n = len(samples)
    for k in metrics:
        metrics[k] /= n

    # Δ-contribution: improvement at step i relative to step i-1.
    for region in REGIONS:
        cosine_vals = metrics[f"{region}_cosine"]
        delta = np.zeros(num_steps)
        delta[0] = cosine_vals[0]
        delta[1:] = cosine_vals[1:] - cosine_vals[:-1]
        metrics[f"{region}_delta_cosine"] = delta

        mse_vals = metrics[f"{region}_mse"]
        delta_mse = np.zeros(num_steps)
        delta_mse[0] = 0.0
        delta_mse[1:] = mse_vals[:-1] - mse_vals[1:]  # MSE decrease = improvement
        metrics[f"{region}_delta_mse"] = delta_mse

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3: Audio ablation (optional — requires a matching no-audio trajectory dir)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_audio_ablation(
    traj_dir: str,
    no_audio_traj_dir: str,
    samples: List[str],
    masks: Dict[str, torch.Tensor],
    num_steps: int,
) -> Dict[str, np.ndarray]:
    """||x0_with_audio - x0_no_audio|| per step, in mouth and full regions."""
    metrics = {f"{region}_audio_diff_l2": np.zeros(num_steps) for region in REGIONS}
    metrics.update({f"{region}_audio_diff_mse": np.zeros(num_steps) for region in REGIONS})

    valid_count = 0
    for sample_name in samples:
        no_audio_dir = os.path.join(no_audio_traj_dir, sample_name)
        if not os.path.isdir(no_audio_dir):
            print(f"[Ablation] Skipping {sample_name}: no-audio trajectory not found")
            continue

        valid_count += 1
        mouth_mask = masks[sample_name]
        for step_i in range(num_steps):
            x0_audio = load_tensor(traj_dir, sample_name, f"step_{step_i:03d}_x0.pt")
            x0_noaudio = load_tensor(no_audio_traj_dir, sample_name, f"step_{step_i:03d}_x0.pt")
            diff = x0_audio - x0_noaudio

            for region in REGIONS:
                metrics[f"{region}_audio_diff_l2"][step_i] += compute_l2_norm(diff, mouth_mask, region)
                vals = apply_region_mask(diff, mouth_mask, region)
                metrics[f"{region}_audio_diff_mse"][step_i] += vals.pow(2).mean().item()

    if valid_count > 0:
        for k in metrics:
            metrics[k] /= valid_count
    else:
        print("[Ablation] WARNING: No matching no-audio samples found!")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 4: Inter-sample variance
# ─────────────────────────────────────────────────────────────────────────────

def analyze_inter_sample_variance(
    traj_dir: str,
    samples: List[str],
    masks: Dict[str, torch.Tensor],
    num_steps: int,
) -> Dict[str, np.ndarray]:
    """Variance of x0_pred across samples at each step.

    - scalar_variance (mouth + full): variance of per-sample region means. Works with per-sample masks.
    - pixelwise_variance (full only): mean of per-pixel variances across samples on the common
      80x80 grid. NOT computed for "mouth" because per-sample mouth masks differ in size/location,
      so mouth pixels do not align across samples.
    """
    metrics = {f"{region}_x0_variance": np.zeros(num_steps) for region in REGIONS}
    metrics["full_x0_pixelwise_var"] = np.zeros(num_steps)

    n_samples = len(samples)
    if n_samples < 2:
        print("[Variance] Need at least 2 samples for variance analysis")
        return metrics

    for step_i in range(num_steps):
        # Scalar variance per region (reduce each sample to a scalar first).
        for region in REGIONS:
            scalar_values = []
            for sample_name in samples:
                x0 = load_tensor(traj_dir, sample_name, f"step_{step_i:03d}_x0.pt")
                region_vals = apply_region_mask(x0, masks[sample_name], region)
                scalar_values.append(region_vals.mean().item())
            metrics[f"{region}_x0_variance"][step_i] = float(np.var(scalar_values))

        # Pixelwise variance on the full grid (aligned across samples).
        full_stack = []
        for sample_name in samples:
            x0 = load_tensor(traj_dir, sample_name, f"step_{step_i:03d}_x0.pt")
            full_stack.append(x0.flatten())  # [C*T*H*W]
        stacked = torch.stack(full_stack, dim=0)  # [N, C*T*H*W]
        metrics["full_x0_pixelwise_var"][step_i] = stacked.var(dim=0).mean().item()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _tick_positions(num_steps: int) -> List[int]:
    """Guarded tick list — avoids IndexError when num_steps != 50."""
    return [t for t in [0, 10, 20, 30, 40, 49] if t < num_steps]


def _add_timestep_axis(ax, t_values: np.ndarray, num_steps: int):
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = _tick_positions(num_steps)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"t={t_values[i]:.3f}" for i in ticks])
    ax2.set_xlabel("Timestep t")


def plot_gt_similarity(metrics: dict, t_list: list, output_dir: str, gt_mode: str):
    """Plot x0_pred vs GT similarity curves (Analysis 1 & 2)."""
    num_steps = len(t_list) - 1
    steps = np.arange(num_steps)
    t_values = np.array(t_list[:num_steps])
    gt_label = "GT (encoded clip)" if gt_mode == "encode" else "final x0 (convergence proxy)"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.plot(steps, metrics["mouth_cosine"], "r-o", markersize=2, label="Mouth region")
    ax.plot(steps, metrics["full_cosine"], "b-s", markersize=2, label="Full frame")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel(f"Cosine Similarity to {gt_label}")
    ax.set_title("(a) x0_pred vs GT Cosine Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _add_timestep_axis(ax, t_values, num_steps)

    ax = axes[0, 1]
    ax.plot(steps, metrics["mouth_mse"], "r-o", markersize=2, label="Mouth region")
    ax.plot(steps, metrics["full_mse"], "b-s", markersize=2, label="Full frame")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel(f"MSE to {gt_label}")
    ax.set_title("(b) x0_pred vs GT MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes[1, 0]
    ax.bar(steps - 0.15, metrics["mouth_delta_cosine"], width=0.3, color="red", alpha=0.7, label="Mouth")
    ax.bar(steps + 0.15, metrics["full_delta_cosine"], width=0.3, color="blue", alpha=0.7, label="Full")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("Δ Cosine Similarity")
    ax.set_title("(c) Per-Step Improvement (Cosine)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax = axes[1, 1]
    ax.bar(steps - 0.15, metrics["mouth_delta_mse"], width=0.3, color="red", alpha=0.7, label="Mouth")
    ax.bar(steps + 0.15, metrics["full_delta_mse"], width=0.3, color="blue", alpha=0.7, label="Full")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("Δ MSE (decrease = improvement)")
    ax.set_title("(d) Per-Step MSE Improvement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "gt_similarity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved {path}")


def plot_audio_ablation(metrics: dict, t_list: list, output_dir: str):
    """Plot audio ablation analysis (Analysis 3)."""
    num_steps = len(t_list) - 1
    steps = np.arange(num_steps)
    t_values = np.array(t_list[:num_steps])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(steps, metrics["mouth_audio_diff_l2"], "r-o", markersize=3, label="Mouth region")
    ax.plot(steps, metrics["full_audio_diff_l2"], "b-s", markersize=3, label="Full frame")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("L2 Norm of Audio Difference")
    ax.set_title("(a) Audio Influence: ||x0_audio - x0_no_audio|| (L2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _add_timestep_axis(ax, t_values, num_steps)

    ax = axes[1]
    ax.plot(steps, metrics["mouth_audio_diff_mse"], "r-o", markersize=3, label="Mouth region")
    ax.plot(steps, metrics["full_audio_diff_mse"], "b-s", markersize=3, label="Full frame")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("Per-Element MSE")
    ax.set_title("(b) Audio Influence: per-element MSE (size-normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    mouth_mse = np.array(metrics["mouth_audio_diff_mse"])
    full_mse = np.array(metrics["full_audio_diff_mse"])
    ratio = np.where(full_mse > 1e-10, mouth_mse / full_mse, 0)
    ax_ratio = ax.twinx()
    ax_ratio.plot(steps, ratio, "g--", alpha=0.5, linewidth=1, label="Mouth/Full ratio")
    ax_ratio.set_ylabel("Mouth/Full Ratio", color="green")
    ax_ratio.tick_params(axis="y", labelcolor="green")

    plt.tight_layout()
    path = os.path.join(output_dir, "audio_ablation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved {path}")


def plot_inter_sample_variance(metrics: dict, t_list: list, output_dir: str):
    """Plot inter-sample variance (Analysis 4)."""
    num_steps = len(t_list) - 1
    steps = np.arange(num_steps)
    t_values = np.array(t_list[:num_steps])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(steps, metrics["mouth_x0_variance"], "r-o", markersize=3, label="Mouth region")
    ax.plot(steps, metrics["full_x0_variance"], "b-s", markersize=3, label="Full frame")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("Variance of per-sample region mean")
    ax.set_title("(a) Scalar Variance (overall level)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _add_timestep_axis(ax, t_values, num_steps)

    ax = axes[1]
    ax.plot(steps, metrics["full_x0_pixelwise_var"], "b-s", markersize=3, label="Full frame")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("Mean per-pixel variance across samples")
    ax.set_title("(b) Pixelwise Variance (full grid; per-sample mouth masks don't align)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "inter_sample_variance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved {path}")


def plot_combined_summary(all_metrics: dict, t_list: list, output_dir: str):
    """Single-page summary combining the most informative curves."""
    num_steps = len(t_list) - 1
    steps = np.arange(num_steps)

    has_ablation = "mouth_audio_diff_mse" in all_metrics
    ncols = 3 if has_ablation else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    ax = axes[0]
    ax.plot(steps, all_metrics["mouth_cosine"], "r-", linewidth=2, label="Mouth")
    ax.plot(steps, all_metrics["full_cosine"], "b-", linewidth=2, label="Full")
    ax.fill_between(steps, all_metrics["mouth_cosine"], all_metrics["full_cosine"],
                    alpha=0.1, color="purple")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("Cosine Similarity to GT")
    ax.set_title("Convergence to Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(steps - 0.15, all_metrics["mouth_delta_cosine"], width=0.3, color="red", alpha=0.7, label="Mouth")
    ax.bar(steps + 0.15, all_metrics["full_delta_cosine"], width=0.3, color="blue", alpha=0.7, label="Full")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("Δ Cosine Similarity")
    ax.set_title("Per-Step Quality Improvement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    if has_ablation:
        ax = axes[2]
        ax.plot(steps, all_metrics["mouth_audio_diff_mse"], "r-o", markersize=3, label="Mouth")
        ax.plot(steps, all_metrics["full_audio_diff_mse"], "b-s", markersize=3, label="Full")
        ax.set_xlabel("ODE Step")
        ax.set_ylabel("Audio Influence (MSE)")
        ax.set_title("Audio Conditioning Impact")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze InfiniteTalk ODE trajectories for lip generation timing")
    parser.add_argument("--traj_dir", type=str, required=True,
                        help="An infinitetalk_t*_a* config dir with per-sample trajectory subdirs")
    parser.add_argument("--no_audio_traj_dir", type=str, default=None,
                        help="Optional no-audio trajectory dir (for ablation). Skipped if absent.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for analysis outputs (plots, JSON)")

    # Ground truth
    parser.add_argument("--gt_mode", type=str, default="encode", choices=["encode", "final_x0"],
                        help="'encode' = VAE-encode the Hallo3 GT clip (default); "
                             "'final_x0' = use the final x0 as a convergence proxy (no VAE/GT needed).")
    parser.add_argument("--gt_video_dir", type=str, default=DEFAULT_GT_VIDEO_DIR,
                        help="Directory of Hallo3 GT mp4s (hash = sample_name.split('_shot')[0]).")

    # Mouth-mask derivation
    parser.add_argument("--mask_source", type=str, default="ref_decode", choices=["ref_decode", "gt_frame"],
                        help="'ref_decode' = decode input_latents.pt via WanVAE (spec default, most "
                             "spatially faithful, needs WanVAE+dlib together); "
                             "'gt_frame' = read GT mp4 frame 0 (no WanVAE; lets masks be derived in a "
                             "dlib-capable env like omniavatar).")
    parser.add_argument("--mouth_pad", type=float, default=0.30,
                        help="Padding around the mouth bbox as a fraction of bbox width/height.")
    parser.add_argument("--mouth_mask_cache", type=str, default=None,
                        help="Optional dir to save/load per-sample latent mouth masks (.npy). "
                             "If all masks are cached, dlib is not needed at run time.")
    parser.add_argument("--shape_predictor", type=str, default=DEFAULT_SHAPE_PREDICTOR,
                        help="dlib 68-pt shape_predictor .dat path.")

    # WanVAE
    parser.add_argument("--infinitetalk_root", type=str, default=DEFAULT_INFINITETALK_ROOT,
                        help="InfiniteTalk repo root (added to sys.path for `from wan.modules.vae import WanVAE`).")
    parser.add_argument("--vae_pth", type=str, default=DEFAULT_VAE_PTH,
                        help="Wan2.1 VAE checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for WanVAE encode/decode.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Discover samples ──
    samples = discover_samples(args.traj_dir)
    print(f"[Data] Found {len(samples)} samples in {args.traj_dir}")
    if not samples:
        print("ERROR: No valid samples found")
        return

    # ── Schedule ──
    schedule = load_schedule(args.traj_dir, samples[0])
    num_steps = schedule["num_steps"]
    t_list = schedule["t_list"]  # length num_steps + 1
    print(f"[Schedule] {num_steps} steps, shift={schedule.get('shift')}, "
          f"t range=[{t_list[0]:.4f}, {t_list[-1]:.4f}]")

    # ── Per-sample mouth masks ──
    print(f"\n[Mask] Deriving per-sample mouth masks (source={args.mask_source}) ...")
    masks = build_mouth_masks(args, samples)

    all_metrics = {}

    # ── GT provider ──
    get_gt = make_gt_provider(args, num_steps)
    # For encode mode, restrict GT similarity to samples that actually have a GT video.
    if args.gt_mode == "encode":
        gt_samples = [s for s in samples if os.path.isfile(gt_video_path(args, s))]
        missing = [s for s in samples if s not in gt_samples]
        if missing:
            print(f"[GT] {len(missing)} samples have no GT video and are skipped for GT similarity: {missing}")
    else:
        gt_samples = samples

    # ── Analysis 1 & 2 ──
    print(f"\n[Analysis 1&2] x0_pred vs GT ({args.gt_mode}) similarity "
          f"({len(gt_samples)} samples × {num_steps} steps)...")
    if gt_samples:
        gt_metrics = analyze_gt_similarity(args.traj_dir, gt_samples, masks, num_steps, get_gt)
        all_metrics.update(gt_metrics)
        plot_gt_similarity(gt_metrics, t_list, args.output_dir, args.gt_mode)

        mouth_delta = gt_metrics["mouth_delta_cosine"]
        top5 = np.argsort(mouth_delta)[::-1][:5]
        print("[Analysis 2] Top 5 Δ-cosine steps for MOUTH:")
        for rank, i in enumerate(top5):
            print(f"  #{rank+1}: step {i} (t={t_list[i]:.4f}) → Δcos={mouth_delta[i]:.6f}")
    else:
        print("[Analysis 1&2] Skipped (no GT-eligible samples)")

    # ── Analysis 3: Audio ablation (optional) ──
    if args.no_audio_traj_dir and os.path.isdir(args.no_audio_traj_dir):
        no_audio_samples = discover_samples(args.no_audio_traj_dir)
        common_samples = [s for s in samples if s in no_audio_samples]
        print(f"\n[Analysis 3] Audio ablation: {len(common_samples)} common samples")
        if common_samples:
            ablation_metrics = analyze_audio_ablation(
                args.traj_dir, args.no_audio_traj_dir, common_samples, masks, num_steps
            )
            all_metrics.update(ablation_metrics)
            plot_audio_ablation(ablation_metrics, t_list, args.output_dir)

            mouth_audio = ablation_metrics["mouth_audio_diff_mse"]
            top5_audio = np.argsort(mouth_audio)[::-1][:5]
            print("[Analysis 3] Top 5 audio-influence steps for MOUTH (MSE):")
            for rank, i in enumerate(top5_audio):
                print(f"  #{rank+1}: step {i} (t={t_list[i]:.4f}) → audio_mse={mouth_audio[i]:.6f}")
    else:
        print("\n[Analysis 3] Skipped (no --no_audio_traj_dir or dir missing)")

    # ── Analysis 4: Inter-sample variance ──
    print(f"\n[Analysis 4] Computing inter-sample variance ({len(samples)} samples)...")
    var_metrics = analyze_inter_sample_variance(args.traj_dir, samples, masks, num_steps)
    all_metrics.update(var_metrics)
    plot_inter_sample_variance(var_metrics, t_list, args.output_dir)

    mouth_var = var_metrics["mouth_x0_variance"]
    peak_step = int(np.argmax(mouth_var))
    print(f"[Analysis 4] Peak mouth variance at step {peak_step} (t={t_list[peak_step]:.4f})")

    # ── Combined summary plot (only if GT similarity ran) ──
    if "mouth_cosine" in all_metrics:
        plot_combined_summary(all_metrics, t_list, args.output_dir)

    # ── Save all metrics as JSON ──
    json_metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in all_metrics.items()}
    json_metrics["t_list"] = t_list
    json_metrics["num_steps"] = num_steps
    json_metrics["samples"] = samples
    json_metrics["gt_mode"] = args.gt_mode
    json_metrics["mask_source"] = args.mask_source
    json_metrics["mouth_pad"] = args.mouth_pad

    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"\n[Done] All metrics saved to {json_path}")
    print(f"[Done] Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
