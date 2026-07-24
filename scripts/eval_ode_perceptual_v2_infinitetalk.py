"""Decode InfiniteTalk x0_pred at all ODE steps, save videos, compute perceptual + lip metrics.

InfiniteTalk-adapted copy of scripts/eval_ode_perceptual_v2.py. See the CHANGELOG at the
bottom of this docstring for the full list of differences vs the OmniAvatar base.

Data (input trajectories):
  Root:  /home/work/.local/ode_full_trajectories_infinitetalk/
  Config dirs: infinitetalk_t{T}_a{A}/  (pass one as --traj_dir)
  Per sample dir (name = <hash>_shot_001_000):
    step_000_x0.pt .. step_049_x0.pt, step_000_xt.pt .., ode_schedule.json, input_latents.pt
  Each x0/xt latent: torch [16, 21, 80, 80] bf16 → decodes to [3, 81, 640, 640] pixels.

Two phases:
  Phase 1 (--phase decode): InfiniteTalk-VAE-decode all steps → save mp4 videos (640x640, with audio)
                            Ground truth comes from the real validation mp4 (NOT from latents).
  Phase 2 (--phase metrics): Compute metrics on saved videos
    - Pixel MSE, SSIM, LPIPS (mouth, full) — reference, vs GT
    - LMD (Lip Landmark Distance) — reference, dlib 68-pt landmarks on mouth
    - Sharpness (Laplacian variance) — no-reference, mouth crop
    - SyncNet Sync-C, Sync-D — no-reference (uses audio)
  Phase 3 (--merge): Merge shard CSVs and plot

Regions: "mouth" (per-clip dlib mouth bbox) and "full" (whole 640x640 frame). There is NO
"upper_face" region (InfiniteTalk has no LatentSync mask to split by).

Both phases support --shard_id / --num_shards for multi-GPU parallelism.

CSV column schema (unchanged from base, so existing plotters stay compatible):
    step, t, sample, metric, region, value

────────────────────────────────────────────────────────────────────────────────────────
RUN COMMANDS
────────────────────────────────────────────────────────────────────────────────────────
Phase 1 (decode) — run in the `infinitetalk` conda env (needs InfiniteTalk's wan.modules.vae,
torch+CUDA, cv2, imageio, ffmpeg). REQUIRES A GPU.

    CUDA_VISIBLE_DEVICES=0 /home/work/.local/miniconda3/envs/infinitetalk/bin/python \
        scripts/eval_ode_perceptual_v2_infinitetalk.py \
        --phase decode \
        --traj_dir /home/work/.local/ode_full_trajectories_infinitetalk/infinitetalk_t5.0_a4.0 \
        --output_dir /home/work/.local/ode_analysis_infinitetalk/t5.0_a4.0/perceptual_v2

Phase 2 (metrics) — run in the `omniavatar` conda env (has the full set: cv2, dlib, lpips,
skimage, plus eval_metrics syncnet). NOTE: `latentsync-metrics` is MISSING the `lpips`
package, so either use `omniavatar`, or use latentsync-metrics with `--skip_metrics lpips`.

    CUDA_VISIBLE_DEVICES=0 /home/work/.local/miniconda3/envs/omniavatar/bin/python \
        scripts/eval_ode_perceptual_v2_infinitetalk.py \
        --phase metrics \
        --traj_dir /home/work/.local/ode_full_trajectories_infinitetalk/infinitetalk_t5.0_a4.0 \
        --output_dir /home/work/.local/ode_analysis_infinitetalk/t5.0_a4.0/perceptual_v2

Phase 3 (--merge) — any env with pandas + matplotlib (e.g. omniavatar). No GPU.

    /home/work/.local/miniconda3/envs/omniavatar/bin/python \
        scripts/eval_ode_perceptual_v2_infinitetalk.py --merge \
        --traj_dir /home/work/.local/ode_full_trajectories_infinitetalk/infinitetalk_t5.0_a4.0 \
        --output_dir /home/work/.local/ode_analysis_infinitetalk/t5.0_a4.0/perceptual_v2

Multi-GPU: add --shard_id/--num_shards to phase 1 and phase 2 (one process per GPU), then
run --merge once at the end.

────────────────────────────────────────────────────────────────────────────────────────
CHANGELOG (vs scripts/eval_ode_perceptual_v2.py)
────────────────────────────────────────────────────────────────────────────────────────
1. Pixel/latent dims: 512px / 64-latent  →  640px / 80-latent everywhere (docstrings,
   default frame size, plot comments). Latents are [16,21,80,80]; decoded frames are 640x640.
2. VAE: OmniAvatar Wan VAE (ModelManager) / SD-VAE branch REMOVED. Now decodes with
   InfiniteTalk's VAE:
       sys.path.insert(0, "/home/work/.local/InfiniteTalk")
       from wan.modules.vae import WanVAE
       vae = WanVAE(vae_pth=<InfiniteTalk Wan2.1_VAE.pth>, device="cuda")
       frames = vae.decode([latent.float().cuda()])[0]   # [3,81,640,640] in [-1,1]
   Removed --vae_type flag. --vae_path now defaults to the InfiniteTalk VAE weights.
3. Ground truth: was decoded from input_latents.pt. InfiniteTalk's input_latents.pt is only
   [16,1,80,80] (a single reference-frame latent), so GT is instead loaded from the real
   validation mp4, center-cropped to square + resized to 640x640, and trimmed to the
   generated frame count (81). GT audio taken from the processed audios dir.
       GT video : /home/work/.local/Hallo3_validation/validation_set_for_benchmark/<hash>.mp4
       GT audio : /home/work/.local/Hallo3_validation/processed/audios/<hash>.wav
       hash = sample_name.split("_shot")[0]
   (old AUDIO_BASE_DIR = /home/work/stableavatar_data/v2v_validation_data/recon → removed)
4. Regions: LatentSync mask.png → dlib per-clip mouth bbox. load_pixel_mask()/get_mouth_bbox()
   REMOVED. Added compute_clip_mouth_bbox(): dlib 68-pt landmarks (pts[48:68]) on GT frames,
   median mouth center + 1.6x median mouth width/height → one stable per-clip bbox. Regions
   reduced from {mouth, upper_face, full} to {mouth (bbox crop), full}. --mask_path REMOVED.
5. Metric-tool paths: METRICS_ROOT = /home/work/.local/latentsync-metrics-가짜  →
   /home/work/.local/eval_metrics. SHAPE_PREDICTOR and SYNCNET_MODEL rebased accordingly.
6. Imports made env-aware: dlib, skimage, lpips, matplotlib, pandas, imageio are imported
   lazily inside the phase that needs them, so Phase 1 (infinitetalk env) and Phase 2
   (omniavatar env) each only require their own deps.
7. CLI (--phase / --shard_id / --num_shards / --merge / --sync_only / --skip_metrics /
   --sync_min_track) preserved. CSV column schema preserved.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# InfiniteTalk repo (for wan.modules.vae in Phase 1)
INFINITETALK_ROOT = "/home/work/.local/InfiniteTalk"
DEFAULT_VAE_PATH = os.path.join(
    INFINITETALK_ROOT, "weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth"
)

# Ground-truth video + audio sources (Hallo3 validation set)
GT_VIDEO_DIR = "/home/work/.local/Hallo3_validation/validation_set_for_benchmark"
GT_AUDIO_DIR = "/home/work/.local/Hallo3_validation/processed/audios"

# Generated frame geometry
FRAME_SIZE = 640  # decoded pixel resolution (640x640); latent is 80x80

# Metrics tools paths
METRICS_ROOT = "/home/work/.local/eval_metrics"
SHAPE_PREDICTOR = os.path.join(METRICS_ROOT, "shape_predictor_68_face_landmarks.dat")
SYNCNET_MODEL = os.path.join(METRICS_ROOT, "checkpoints/auxiliary/syncnet_v2.model")


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def sample_to_hash(sample_name):
    """<hash>_shot_001_000 → <hash>."""
    return sample_name.split("_shot")[0]


def discover_samples(traj_dir):
    return sorted([
        d for d in os.listdir(traj_dir)
        if os.path.isdir(os.path.join(traj_dir, d))
        and os.path.isfile(os.path.join(traj_dir, d, "ode_schedule.json"))
    ])


def shard_list(items, shard_id, num_shards):
    if shard_id is not None and num_shards is not None:
        return items[shard_id::num_shards]
    return items


def load_schedule(traj_dir, sample_names):
    with open(os.path.join(traj_dir, sample_names[0], "ode_schedule.json")) as f:
        schedule = json.load(f)
    return schedule["t_list"], schedule["num_steps"]


def load_full_schedule(traj_dir, sample_names):
    with open(os.path.join(traj_dir, sample_names[0], "ode_schedule.json")) as f:
        return json.load(f)


def generated_frame_count(schedule):
    """Pixel frame count produced by the VAE from the latent temporal dim.

    Wan VAE upsamples time by 4x with a +1 offset: F_pix = (F_lat - 1) * 4 + 1.
    For latent_shape [16, 21, 80, 80] → 81 frames.
    """
    f_lat = schedule["latent_shape"][1]
    return (f_lat - 1) * 4 + 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Decode  (run in the `infinitetalk` conda env, needs a GPU)
# ─────────────────────────────────────────────────────────────────────────────

def decode_latents_to_numpy(vae, latents, device):
    """Decode a [16, 21, 80, 80] latent → [T, H, W, 3] uint8 RGB numpy using InfiniteTalk VAE.

    vae.decode([z])[0] returns [3, 81, 640, 640] float in [-1, 1].
    """
    lat = latents.float().to(device)  # [16, 21, 80, 80]
    with torch.no_grad():
        frames = vae.decode([lat])[0]  # [3, 81, 640, 640] in [-1, 1]
    frames = ((frames.clamp(-1, 1) + 1) / 2 * 255).clamp(0, 255).byte()  # [3, T, H, W]
    return frames.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, 3]


def center_crop_resize(frame_bgr, size=FRAME_SIZE):
    """Center-crop to square (min side) then resize to size×size."""
    h, w = frame_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = frame_bgr[y0:y0 + s, x0:x0 + s]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def load_gt_frames_rgb(hash_id, n_frames, size=FRAME_SIZE):
    """Read the real GT mp4, center-crop+resize to size, trim to n_frames → [T,H,W,3] uint8 RGB."""
    path = os.path.join(GT_VIDEO_DIR, f"{hash_id}.mp4")
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = center_crop_resize(frame, size)
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        return None
    return np.stack(out)  # [T, H, W, 3] RGB uint8


def save_video(frames_np, path, fps=25):
    import imageio
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()


def mux_audio(video_path, audio_path, output_path):
    """Mux audio into video, trimming to shortest."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path, "-i", audio_path,
        "-c:v", "libx264", "-crf", "18", "-c:a", "aac",
        "-shortest", output_path,
    ], capture_output=True)


def run_decode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_samples = discover_samples(args.traj_dir)
    sample_names = shard_list(all_samples, args.shard_id, args.num_shards)
    schedule = load_full_schedule(args.traj_dir, all_samples)
    t_list, num_steps = schedule["t_list"], schedule["num_steps"]
    n_gt = generated_frame_count(schedule)

    if args.shard_id is not None:
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(sample_names)} samples")

    # Load InfiniteTalk VAE
    sys.path.insert(0, INFINITETALK_ROOT)
    from wan.modules.vae import WanVAE
    print(f"Loading InfiniteTalk WanVAE from {args.vae_path} ...")
    vae = WanVAE(vae_pth=args.vae_path, device=str(device))

    videos_dir = os.path.join(args.output_dir, "videos")
    total_decodes = len(sample_names) * (num_steps + 1)
    decode_count = 0
    t_start = time.time()

    for si, sample_name in enumerate(sample_names):
        print(f"\n[{si+1}/{len(sample_names)}] {sample_name}")
        sample_vid_dir = os.path.join(videos_dir, sample_name)
        os.makedirs(sample_vid_dir, exist_ok=True)

        hash_id = sample_to_hash(sample_name)
        audio_path = os.path.join(GT_AUDIO_DIR, f"{hash_id}.wav")
        has_audio = os.path.exists(audio_path)

        # Ground truth: real validation mp4 (NOT decoded from latents)
        gt_path = os.path.join(sample_vid_dir, "gt.mp4")
        if not os.path.exists(gt_path):
            gt_frames = load_gt_frames_rgb(hash_id, n_gt, size=FRAME_SIZE)
            if gt_frames is None:
                print(f"  WARNING: GT video missing for {hash_id}, skipping sample")
                continue
            save_video(gt_frames, gt_path)
            if has_audio:
                mux_audio(gt_path, audio_path, gt_path.replace(".mp4", "_audio.mp4"))
        decode_count += 1

        # Decode each ODE step's x0 prediction
        for step_i in range(num_steps):
            step_path = os.path.join(sample_vid_dir, f"step_{step_i:03d}.mp4")
            if os.path.exists(step_path):
                decode_count += 1
                continue

            x0 = torch.load(
                os.path.join(args.traj_dir, sample_name, f"step_{step_i:03d}_x0.pt"),
                map_location="cpu", weights_only=True,
            )
            frames = decode_latents_to_numpy(vae, x0, device)
            save_video(frames, step_path)

            # Mux audio for SyncNet
            if has_audio:
                mux_audio(step_path, audio_path, step_path.replace(".mp4", "_audio.mp4"))

            decode_count += 1

            if step_i % 10 == 0 or step_i == num_steps - 1:
                elapsed = time.time() - t_start
                rate = decode_count / elapsed if elapsed > 0 else 0
                remaining = (total_decodes - decode_count) / rate if rate > 0 else 0
                print(f"  step {step_i:2d} | decoded {decode_count}/{total_decodes} | "
                      f"ETA {remaining/60:.1f} min")

    print(f"\nDecode done! {decode_count} videos in {(time.time()-t_start)/60:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Metrics  (run in the `omniavatar` conda env)
# ─────────────────────────────────────────────────────────────────────────────

def bbox_to_mask(bbox, H, W):
    """(y0, y1, x0, x1) → boolean [H, W] mask, True inside the box."""
    m = torch.zeros((H, W), dtype=torch.bool)
    y0, y1, x0, x1 = bbox
    m[y0:y1, x0:x1] = True
    return m


def default_mouth_bbox(H, W):
    """Fallback lower-center-face box when no face is detected in the GT clip."""
    return (int(H * 0.55), int(H * 0.95), int(W * 0.30), int(W * 0.70))


def _mouth_points_abs(image_bgr, detector, predictor):
    """Absolute (x, y) coords of the 20 mouth landmarks (dlib points 48-67). None if no face."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if not rects:
        return None
    rect = max(rects, key=lambda r: r.width() * r.height())
    shape = predictor(gray, rect)
    pts = np.asarray([(pt.x, pt.y) for pt in shape.parts()], dtype=np.float64)
    return pts[48:68]


def compute_clip_mouth_bbox(gt_frames_bgr, detector, predictor, H, W,
                            pad_scale=1.6, max_probe=40):
    """One stable per-clip mouth bbox: median mouth center + pad_scale × median mouth size.

    Returns (y0, y1, x0, x1) or None if no face is found in any probed frame.
    Probes up to max_probe evenly-spaced frames to keep dlib cost bounded.
    """
    n = len(gt_frames_bgr)
    if n == 0:
        return None
    idxs = np.unique(np.linspace(0, n - 1, min(max_probe, n)).astype(int))

    centers, widths, heights = [], [], []
    for i in idxs:
        mouth = _mouth_points_abs(gt_frames_bgr[int(i)], detector, predictor)
        if mouth is None:
            continue
        centers.append(mouth.mean(axis=0))                       # (cx, cy)
        widths.append(mouth[:, 0].max() - mouth[:, 0].min())     # mouth width
        heights.append(mouth[:, 1].max() - mouth[:, 1].min())    # mouth height

    if not centers:
        return None

    cx, cy = np.median(np.stack(centers), axis=0)
    w = float(np.median(widths))
    h = float(np.median(heights))
    half_w = max(pad_scale * w / 2.0, 4.0)
    half_h = max(pad_scale * h / 2.0, 4.0)

    x0 = int(max(0, np.floor(cx - half_w)))
    x1 = int(min(W, np.ceil(cx + half_w)))
    y0 = int(max(0, np.floor(cy - half_h)))
    y1 = int(min(H, np.ceil(cy + half_h)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (y0, y1, x0, x1)


def read_video_frames(path):
    """Read video → list of BGR numpy frames."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def frames_to_tensor(frames_bgr):
    """BGR uint8 list → [T, 3, H, W] float32 in [0, 1] (RGB)."""
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    arr = np.stack(rgb).astype(np.float32) / 255.0  # [T, H, W, 3]
    return torch.from_numpy(arr).permute(0, 3, 1, 2)  # [T, 3, H, W]


# -- Pixel MSE --
def compute_masked_mse(pred, gt, mask):
    mask_exp = mask.unsqueeze(0).unsqueeze(0).expand_as(pred)
    diff_sq = (pred - gt).pow(2)
    return diff_sq[mask_exp].mean().item()


# -- SSIM --
_ssim_fn = None


def compute_masked_ssim(pred, gt, mask):
    global _ssim_fn
    if _ssim_fn is None:
        from skimage.metrics import structural_similarity as _s
        _ssim_fn = _s
    pred_np = pred.permute(0, 2, 3, 1).numpy()
    gt_np = gt.permute(0, 2, 3, 1).numpy()
    mask_np = mask.numpy()
    vals = []
    for t in range(pred_np.shape[0]):
        _, ssim_map = _ssim_fn(
            gt_np[t], pred_np[t],
            channel_axis=2, data_range=1.0, full=True,
        )
        ssim_spatial = ssim_map.mean(axis=2)
        vals.append(ssim_spatial[mask_np].mean())
    return float(np.mean(vals))


# -- LPIPS --
def compute_lpips(lpips_model, pred, gt, bbox, device):
    y0, y1, x0, x1 = bbox
    pred_crop = pred[:, :, y0:y1, x0:x1] * 2 - 1
    gt_crop = gt[:, :, y0:y1, x0:x1] * 2 - 1
    vals = []
    for i in range(0, pred_crop.shape[0], 16):
        p = pred_crop[i:i+16].to(device)
        g = gt_crop[i:i+16].to(device)
        with torch.no_grad():
            d = lpips_model(p, g)
        vals.append(d.cpu().reshape(-1))
    return torch.cat(vals).mean().item()


def compute_lpips_full(lpips_model, pred, gt, device):
    pred_l = pred * 2 - 1
    gt_l = gt * 2 - 1
    vals = []
    for i in range(0, pred_l.shape[0], 16):
        p = pred_l[i:i+16].to(device)
        g = gt_l[i:i+16].to(device)
        with torch.no_grad():
            d = lpips_model(p, g)
        vals.append(d.cpu().reshape(-1))
    return torch.cat(vals).mean().item()


# -- Sharpness (Laplacian variance) --
def compute_mouth_sharpness(frames_bgr, bbox):
    """Laplacian variance in mouth crop, averaged over frames. No reference needed."""
    y0, y1, x0, x1 = bbox
    vals = []
    for frame in frames_bgr:
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        vals.append(lap.var())
    return float(np.mean(vals))


# -- LMD --
def compute_video_lmd(pred_frames_bgr, gt_frames_bgr, detector, predictor):
    """LMD between pred and GT videos, matching the latentsync-metrics implementation."""
    T = min(len(pred_frames_bgr), len(gt_frames_bgr))
    total_lmd = 0.0
    count = 0
    for t in range(T):
        pred_land = _extract_mouth_landmarks(pred_frames_bgr[t], detector, predictor)
        gt_land = _extract_mouth_landmarks(gt_frames_bgr[t], detector, predictor)
        if pred_land is None or gt_land is None:
            continue
        diff = pred_land - gt_land
        lmd = float(np.sum(np.linalg.norm(diff, axis=1)) / pred_land.shape[0])
        total_lmd += lmd
        count += 1
    if count == 0:
        return None
    return total_lmd / count


def _extract_mouth_landmarks(image_bgr, detector, predictor):
    """Extract 20 mouth landmarks (dlib points 48-67), mean-centered."""
    mouth = _mouth_points_abs(image_bgr, detector, predictor)
    if mouth is None:
        return None
    mouth = mouth.copy()
    mouth -= mouth.mean(axis=0, keepdims=True)
    return mouth


def run_metrics(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_samples = discover_samples(args.traj_dir)
    sample_names = shard_list(all_samples, args.shard_id, args.num_shards)
    t_list, num_steps = load_schedule(args.traj_dir, all_samples)

    if args.shard_id is not None:
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(sample_names)} samples")

    videos_dir = os.path.join(args.output_dir, "videos")

    # Load LPIPS (skip if not needed)
    if "lpips" not in args._skip_set and not args.sync_only:
        import lpips
        print("Loading LPIPS...")
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    else:
        lpips_model = None

    # Load dlib — needed for the per-clip mouth bbox AND for LMD (any non-sync run)
    if not args.sync_only:
        import dlib
        print("Loading dlib landmark detector...")
        dlib_detector = dlib.get_frontal_face_detector()
        dlib_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
    else:
        dlib_detector = None
        dlib_predictor = None

    # SyncNet: load once, reuse
    print("Loading SyncNet...")
    sys.path.insert(0, METRICS_ROOT)
    from eval.syncnet import SyncNetEval
    from eval.syncnet_detect import SyncNetDetector
    syncnet = SyncNetEval(device=str(device))
    syncnet.loadParameters(SYNCNET_MODEL)

    # CSV
    shard_suffix = f"_shard{args.shard_id}" if args.shard_id is not None else ""
    if args.sync_only:
        # Append-only sync output; filename differs so we don't clobber existing metrics.csv.
        csv_path = os.path.join(args.output_dir, f"metrics_sync_only{shard_suffix}.csv")
    else:
        csv_path = os.path.join(args.output_dir, f"metrics{shard_suffix}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "t", "sample", "metric", "region", "value"])
    sync_min_track = args.sync_min_track

    t_start = time.time()

    for si, sample_name in enumerate(sample_names):
        print(f"\n[{si+1}/{len(sample_names)}] {sample_name}")
        sample_vid_dir = os.path.join(videos_dir, sample_name)

        # Read GT video once
        gt_path = os.path.join(sample_vid_dir, "gt.mp4")
        gt_bgr = read_video_frames(gt_path)
        if not gt_bgr:
            print(f"  WARNING: no GT frames at {gt_path}, skipping sample")
            continue
        gt_tensor = frames_to_tensor(gt_bgr)
        H, W = gt_bgr[0].shape[:2]
        full_mask = torch.ones((H, W), dtype=torch.bool)

        # Per-clip mouth bbox / mask (from GT frames via dlib)
        mouth_bbox = None
        mouth_mask = None
        if not args.sync_only:
            mouth_bbox = compute_clip_mouth_bbox(
                gt_bgr, dlib_detector, dlib_predictor, H, W)
            if mouth_bbox is None:
                mouth_bbox = default_mouth_bbox(H, W)
                print(f"  WARNING: no face detected in GT; using default mouth bbox "
                      f"{mouth_bbox}")
            mouth_mask = bbox_to_mask(mouth_bbox, H, W)

        # GT baselines (sharpness, SyncNet) — written with step=-1
        if not args.sync_only:
            gt_sharp = compute_mouth_sharpness(gt_bgr, mouth_bbox)
            writer.writerow([-1, "gt", sample_name, "sharpness", "mouth", f"{gt_sharp:.4f}"])

        gt_audio_path = os.path.join(sample_vid_dir, "gt_audio.mp4")
        if os.path.exists(gt_audio_path):
            detect_dir = tempfile.mkdtemp(prefix="syncdet_gt_")
            temp_dir_sync = tempfile.mkdtemp(prefix="synctemp_gt_")
            try:
                detector_sync = SyncNetDetector(
                    device=str(device), detect_results_dir=detect_dir
                )
                detector_sync(video_path=gt_audio_path, min_track=sync_min_track)
                crop_dir = os.path.join(detect_dir, "crop")
                if os.path.exists(crop_dir) and os.listdir(crop_dir):
                    sd_list, sc_list = [], []
                    for vid in os.listdir(crop_dir):
                        vtemp = tempfile.mkdtemp(prefix="sv_", dir=temp_dir_sync)
                        try:
                            _, md, cf = syncnet.evaluate(
                                video_path=os.path.join(crop_dir, vid), temp_dir=vtemp)
                            sd_list.append(md)
                            sc_list.append(cf)
                        except Exception:
                            pass
                        finally:
                            shutil.rmtree(vtemp, ignore_errors=True)
                    if sd_list:
                        from statistics import fmean
                        writer.writerow([-1, "gt", sample_name, "sync_d", "mouth", f"{fmean(sd_list):.6f}"])
                        writer.writerow([-1, "gt", sample_name, "sync_c", "mouth", f"{fmean(sc_list):.6f}"])
            except Exception as e:
                print(f"  GT SyncNet error: {e}")
            finally:
                shutil.rmtree(detect_dir, ignore_errors=True)
                shutil.rmtree(temp_dir_sync, ignore_errors=True)

        csv_file.flush()

        for step_i in range(num_steps):
            t_val = t_list[step_i]
            step_path = os.path.join(sample_vid_dir, f"step_{step_i:03d}.mp4")
            step_audio_path = step_path.replace(".mp4", "_audio.mp4")

            if not args.sync_only:
                pred_bgr = read_video_frames(step_path)
                pred_tensor = frames_to_tensor(pred_bgr)

                T = min(pred_tensor.shape[0], gt_tensor.shape[0])
                pred_t = pred_tensor[:T]
                gt_t = gt_tensor[:T]
                pred_b = pred_bgr[:T]
                gt_b = gt_bgr[:T]

                # -- Pixel MSE --
                if "pixel_mse" not in args._skip_set:
                    for region, mask in [("mouth", mouth_mask), ("full", full_mask)]:
                        v = compute_masked_mse(pred_t, gt_t, mask)
                        writer.writerow([step_i, f"{t_val:.6f}", sample_name, "pixel_mse", region, f"{v:.8f}"])

                # -- SSIM --
                if "ssim" not in args._skip_set:
                    for region, mask in [("mouth", mouth_mask), ("full", full_mask)]:
                        v = compute_masked_ssim(pred_t, gt_t, mask)
                        writer.writerow([step_i, f"{t_val:.6f}", sample_name, "ssim", region, f"{v:.8f}"])

                # -- LPIPS --
                if "lpips" not in args._skip_set:
                    lp_mouth = compute_lpips(lpips_model, pred_t, gt_t, mouth_bbox, device)
                    writer.writerow([step_i, f"{t_val:.6f}", sample_name, "lpips", "mouth", f"{lp_mouth:.8f}"])
                    lp_full = compute_lpips_full(lpips_model, pred_t, gt_t, device)
                    writer.writerow([step_i, f"{t_val:.6f}", sample_name, "lpips", "full", f"{lp_full:.8f}"])

                # -- Sharpness --
                if "sharpness" not in args._skip_set:
                    sharp = compute_mouth_sharpness(pred_b, mouth_bbox)
                    writer.writerow([step_i, f"{t_val:.6f}", sample_name, "sharpness", "mouth", f"{sharp:.4f}"])

                # -- LMD --
                if "lmd" not in args._skip_set:
                    lmd = compute_video_lmd(pred_b, gt_b, dlib_detector, dlib_predictor)
                    if lmd is not None:
                        writer.writerow([step_i, f"{t_val:.6f}", sample_name, "lmd", "mouth", f"{lmd:.6f}"])

            # -- SyncNet --
            if os.path.exists(step_audio_path):
                detect_dir = tempfile.mkdtemp(prefix="syncdet_")
                temp_dir_sync = tempfile.mkdtemp(prefix="synctemp_")
                try:
                    detector_sync = SyncNetDetector(
                        device=str(device), detect_results_dir=detect_dir
                    )
                    detector_sync(video_path=step_audio_path, min_track=sync_min_track)
                    crop_dir = os.path.join(detect_dir, "crop")
                    if os.path.exists(crop_dir) and os.listdir(crop_dir):
                        sync_d_list, sync_c_list = [], []
                        for vid in os.listdir(crop_dir):
                            vtemp = tempfile.mkdtemp(prefix="sv_", dir=temp_dir_sync)
                            try:
                                _, min_dist, conf = syncnet.evaluate(
                                    video_path=os.path.join(crop_dir, vid),
                                    temp_dir=vtemp,
                                )
                                sync_d_list.append(min_dist)
                                sync_c_list.append(conf)
                            except Exception:
                                pass
                            finally:
                                shutil.rmtree(vtemp, ignore_errors=True)
                        if sync_d_list:
                            from statistics import fmean
                            writer.writerow([step_i, f"{t_val:.6f}", sample_name,
                                             "sync_d", "mouth", f"{fmean(sync_d_list):.6f}"])
                            writer.writerow([step_i, f"{t_val:.6f}", sample_name,
                                             "sync_c", "mouth", f"{fmean(sync_c_list):.6f}"])
                except Exception as e:
                    print(f"    SyncNet error step {step_i}: {e}")
                finally:
                    shutil.rmtree(detect_dir, ignore_errors=True)
                    shutil.rmtree(temp_dir_sync, ignore_errors=True)

            csv_file.flush()

            if step_i % 10 == 0 or step_i == num_steps - 1:
                elapsed = time.time() - t_start
                print(f"  step {step_i:2d} (t={t_val:.3f}) | "
                      f"elapsed {elapsed/60:.1f} min")

    csv_file.close()
    print(f"\nMetrics done! CSV: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Merge + Plot
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_plot(args):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shard_files = sorted(glob.glob(os.path.join(args.output_dir, "metrics_shard*.csv")))
    if shard_files:
        dfs = [pd.read_csv(f) for f in shard_files]
        df = pd.concat(dfs, ignore_index=True)
        merged_path = os.path.join(args.output_dir, "metrics.csv")
        df.to_csv(merged_path, index=False)
        print(f"Merged {len(shard_files)} shards → {merged_path} ({len(df)} rows)")
    else:
        merged_path = os.path.join(args.output_dir, "metrics.csv")
        if not os.path.exists(merged_path):
            print("No shard or merged CSV found.")
            return
        df = pd.read_csv(merged_path)

    # Load t_list
    all_samples = discover_samples(args.traj_dir)
    t_list, num_steps = load_schedule(args.traj_dir, all_samples)
    t_values = np.array(t_list[:num_steps])

    # Separate GT baselines (step=-1) from trajectory data
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    gt_rows = df[df["step"] == -1]
    gt_baselines = gt_rows.groupby(["metric", "region"])["value"].mean().reset_index()
    df_steps = df[df["step"] >= 0].copy()
    agg = df_steps.groupby(["step", "metric", "region"])["value"].mean().reset_index()

    # ── Reference metrics (vs GT): MSE, SSIM, LPIPS, LMD ──
    # Regions: mouth (dlib bbox) and full. No "upper_face".
    ref_metrics = [
        ("pixel_mse", "Pixel MSE", True, ["mouth", "full"]),
        ("ssim", "SSIM", False, ["mouth", "full"]),
        ("lpips", "LPIPS", False, ["mouth", "full"]),
        ("lmd", "LMD (lip landmarks)", False, ["mouth"]),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(28, 10))

    region_style = {
        "mouth": ("red", "o"),
        "full": ("gray", "^"),
    }
    delta_offsets = {"mouth": -0.2, "full": 0.2}

    for col, (metric_name, title, use_log, regions) in enumerate(ref_metrics):
        ax = axes[0, col]
        sub = agg[agg["metric"] == metric_name]
        for region in regions:
            data = sub[sub["region"] == region].sort_values("step")
            if len(data) == 0:
                continue
            color, marker = region_style[region]
            ax.plot(data["step"], data["value"],
                    f"-{marker}", markersize=2, color=color,
                    label=region.replace("_", " ").title())
        ax.set_xlabel("ODE Step")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs GT")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

        # Secondary x-axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"t={t_values[i]:.1f}" for i in tick_pos])
        ax2.set_xlabel("Timestep t", fontsize=8)

        # Delta
        ax_d = axes[1, col]
        lower_better = metric_name in ("pixel_mse", "lpips", "lmd")
        for region in regions:
            data = sub[sub["region"] == region].sort_values("step")
            if len(data) == 0:
                continue
            vals = data["value"].values
            delta = np.zeros(len(vals))
            if lower_better:
                delta[1:] = vals[:-1] - vals[1:]
            else:
                delta[1:] = vals[1:] - vals[:-1]
            color, _ = region_style[region]
            ax_d.bar(data["step"].values + delta_offsets.get(region, 0),
                     delta, width=0.4, color=color, alpha=0.7,
                     label=region.replace("_", " ").title())
        ax_d.set_xlabel("ODE Step")
        ax_d.set_ylabel(f"Δ {title}")
        ax_d.set_title(f"Per-Step Δ ({title})")
        ax_d.legend()
        ax_d.grid(True, alpha=0.3)
        ax_d.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    path1 = os.path.join(args.output_dir, "reference_metrics.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"Saved {path1}")

    # ── No-reference metrics: Sharpness, SyncNet ──
    noref_metrics = [
        ("sharpness", "Mouth Sharpness (Laplacian var)", False),
        ("sync_d", "Sync-D (lower=better sync)", False),
        ("sync_c", "Sync-C (higher=better sync)", False),
    ]

    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 5))

    for col, (metric_name, title, use_log) in enumerate(noref_metrics):
        ax = axes2[col]
        sub = agg[agg["metric"] == metric_name]
        data = sub[sub["region"] == "mouth"].sort_values("step")
        if len(data) == 0:
            ax.set_title(f"{title} (no data)")
            continue
        ax.plot(data["step"], data["value"], "r-o", markersize=3, label="Prediction")
        ax.set_xlabel("ODE Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

        # GT baseline as horizontal dashed line
        gt_val = gt_baselines[
            (gt_baselines["metric"] == metric_name) & (gt_baselines["region"] == "mouth")
        ]["value"]
        if len(gt_val) > 0:
            ax.axhline(y=gt_val.values[0], color="green", linestyle="--",
                        linewidth=2, label=f"GT ({gt_val.values[0]:.2f})")
        ax.legend()

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"t={t_values[i]:.1f}" for i in tick_pos])
        ax2.set_xlabel("Timestep t", fontsize=8)

    plt.tight_layout()
    path2 = os.path.join(args.output_dir, "noref_metrics.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved {path2}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, choices=["decode", "metrics"],
                        help="Which phase to run")
    parser.add_argument("--merge", action="store_true", help="Merge shards and plot")
    parser.add_argument("--traj_dir", type=str, required=True,
                        help="An infinitetalk_t*_a* trajectory directory")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default=DEFAULT_VAE_PATH,
                        help="Path to the InfiniteTalk Wan2.1 VAE .pth (decode phase)")
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--sync_min_track", type=int, default=50,
                        help="Min frames for SyncNet face track. Lower for short videos "
                             "(default 50 works for 81-frame InfiniteTalk clips).")
    parser.add_argument("--sync_only", action="store_true",
                        help="Compute ONLY SyncNet sync_c/sync_d (skip pixel_mse/ssim/lpips/lmd/sharpness). "
                             "Writes to metrics_sync_only.csv so existing metrics.csv is preserved.")
    parser.add_argument("--skip_metrics", type=str, default=None,
                        help="Comma-separated metrics to skip (e.g. 'pixel_mse,lmd,sharpness'). "
                             "Remaining metrics are computed normally.")
    args = parser.parse_args()
    args._skip_set = set(args.skip_metrics.split(",")) if args.skip_metrics else set()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.merge:
        merge_and_plot(args)
    elif args.phase == "decode":
        run_decode(args)
    elif args.phase == "metrics":
        run_metrics(args)
    else:
        parser.error("Specify --phase decode, --phase metrics, or --merge")


if __name__ == "__main__":
    main()
