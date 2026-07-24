"""Decode x0_pred at all ODE steps and compute perceptual metrics vs GT.

For each sample and each ODE step, VAE-decodes x0_pred to 81 frames at 512x512,
then computes pixel-space metrics against the decoded GT (input_latents.pt).

Metrics (per step, averaged over frames then samples):
  - Pixel MSE  (mouth, upper_face, full)
  - SSIM       (mouth, upper_face, full)
  - LPIPS      (mouth_crop, full)

Outputs:
  - perceptual_metrics.csv   — per-sample per-step values
  - perceptual_curves.png    — summary plot (mouth vs upper_face vs full)
  - perceptual_delta.png     — per-step improvement (delta)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_ode_perceptual.py \
        --traj_dir /home/work/ode_full_trajectories/14B \
        --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
        --mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
        --output_dir /home/work/ode_analysis/14B/perceptual
"""

import argparse
import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OmniAvatar.models.model_manager import ModelManager


# ─────────────────────────────────────────────────────────────────────────────
# Mask utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_pixel_mask(mask_path: str, H: int = 512, W: int = 512) -> torch.Tensor:
    """Load LatentSync mask, resize to pixel resolution.

    Returns:
        mask_keep: [H, W] bool tensor. True=upper face (keep), False=mouth.
    """
    mask_img = Image.open(mask_path)
    mask_arr = np.array(mask_img, dtype=np.float32)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    mask_arr = mask_arr / 255.0
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_t, size=(H, W), mode="bilinear", align_corners=False)
    return mask_resized.squeeze() > 0.5


def get_mouth_bbox(mask_keep: torch.Tensor, pad: int = 4):
    """Get bounding box of mouth region (where mask_keep is False).

    Returns (y_min, y_max, x_min, x_max) with padding, clamped to image bounds.
    """
    mouth = ~mask_keep
    ys, xs = torch.where(mouth)
    H, W = mask_keep.shape
    y_min = max(0, ys.min().item() - pad)
    y_max = min(H, ys.max().item() + 1 + pad)
    x_min = max(0, xs.min().item() - pad)
    x_max = min(W, xs.max().item() + 1 + pad)
    return y_min, y_max, x_min, x_max


# ─────────────────────────────────────────────────────────────────────────────
# VAE decode
# ─────────────────────────────────────────────────────────────────────────────

def decode_latents(vae, latents: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Decode [16, 21, 64, 64] latents -> [T, 3, H, W] float32 in [0, 1]."""
    latents = latents.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        frames = vae.decode(latents, device=device, tiled=False)
    if frames.dim() == 5:
        frames = frames[0]  # [3, T, H, W]
    frames = ((frames.float() + 1) / 2).clamp(0, 1)
    return frames.permute(1, 0, 2, 3).cpu()  # [T, 3, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_masked_mse(pred: torch.Tensor, gt: torch.Tensor,
                       mask: torch.Tensor) -> float:
    """Pixel MSE within a spatial mask. pred, gt: [T, 3, H, W], mask: [H, W] bool."""
    mask_exp = mask.unsqueeze(0).unsqueeze(0).expand_as(pred)
    diff_sq = (pred - gt).pow(2)
    return diff_sq[mask_exp].mean().item()


def compute_masked_ssim(pred: torch.Tensor, gt: torch.Tensor,
                        mask: torch.Tensor) -> float:
    """Per-pixel SSIM averaged within a spatial mask, then averaged over frames."""
    pred_np = pred.permute(0, 2, 3, 1).numpy()  # [T, H, W, 3]
    gt_np = gt.permute(0, 2, 3, 1).numpy()
    mask_np = mask.numpy()

    vals = []
    for t in range(pred_np.shape[0]):
        _, ssim_map = ssim_fn(
            gt_np[t], pred_np[t],
            channel_axis=2, data_range=1.0, full=True,
        )
        # ssim_map: [H, W, 3] — average over channels then mask
        ssim_spatial = ssim_map.mean(axis=2)  # [H, W]
        vals.append(ssim_spatial[mask_np].mean())
    return float(np.mean(vals))


def compute_lpips_crop(lpips_model, pred: torch.Tensor, gt: torch.Tensor,
                       bbox, device: torch.device) -> float:
    """LPIPS on a cropped region. pred, gt: [T, 3, H, W] in [0, 1]."""
    y0, y1, x0, x1 = bbox
    pred_crop = pred[:, :, y0:y1, x0:x1]
    gt_crop = gt[:, :, y0:y1, x0:x1]

    # LPIPS expects [-1, 1]
    pred_lpips = pred_crop * 2 - 1
    gt_lpips = gt_crop * 2 - 1

    vals = []
    batch_size = 16
    for i in range(0, pred_lpips.shape[0], batch_size):
        p = pred_lpips[i:i + batch_size].to(device)
        g = gt_lpips[i:i + batch_size].to(device)
        with torch.no_grad():
            d = lpips_model(p, g)
        vals.append(d.cpu().reshape(-1))
    return torch.cat(vals).mean().item()


def compute_lpips_full(lpips_model, pred: torch.Tensor, gt: torch.Tensor,
                       device: torch.device) -> float:
    """LPIPS on full frames. pred, gt: [T, 3, H, W] in [0, 1]."""
    pred_lpips = pred * 2 - 1
    gt_lpips = gt * 2 - 1

    vals = []
    batch_size = 16
    for i in range(0, pred_lpips.shape[0], batch_size):
        p = pred_lpips[i:i + batch_size].to(device)
        g = gt_lpips[i:i + batch_size].to(device)
        with torch.no_grad():
            d = lpips_model(p, g)
        vals.append(d.cpu().reshape(-1))
    return torch.cat(vals).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(csv_path: str, t_list: list, output_dir: str):
    """Read CSV and plot per-step metric curves."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    num_steps = df["step"].max() + 1
    steps = np.arange(num_steps)
    t_values = np.array(t_list[:num_steps])

    # Aggregate: mean over samples
    agg = df.groupby(["step", "metric", "region"])["value"].mean().reset_index()

    metrics_to_plot = [
        ("pixel_mse", "Pixel MSE", True),
        ("ssim", "SSIM", False),
        ("lpips", "LPIPS", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for col, (metric_name, title, use_log) in enumerate(metrics_to_plot):
        ax = axes[0, col]
        sub = agg[agg["metric"] == metric_name]

        for region, color, marker in [
            ("mouth", "red", "o"),
            ("upper_face", "blue", "s"),
            ("full", "gray", "^"),
        ]:
            data = sub[sub["region"] == region].sort_values("step")
            if len(data) == 0:
                continue
            ax.plot(data["step"], data["value"],
                    f"{color[0]}-{marker}", markersize=2, color=color,
                    label=region.replace("_", " ").title())

        ax.set_xlabel("ODE Step")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs GT")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

        # Secondary x-axis with timestep values
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"t={t_values[i]:.3f}" for i in tick_pos])
        ax2.set_xlabel("Timestep t")

        # Delta plot (per-step improvement)
        ax_d = axes[1, col]
        for region, color in [("mouth", "red"), ("upper_face", "blue"), ("full", "gray")]:
            data = sub[sub["region"] == region].sort_values("step")
            if len(data) == 0:
                continue
            vals = data["value"].values
            if metric_name in ("pixel_mse", "lpips"):
                # Lower is better → improvement = decrease
                delta = np.zeros(len(vals))
                delta[1:] = vals[:-1] - vals[1:]
            else:
                # Higher is better → improvement = increase
                delta = np.zeros(len(vals))
                delta[1:] = vals[1:] - vals[:-1]

            ax_d.bar(data["step"].values + {"mouth": -0.25, "upper_face": 0.0, "full": 0.25}[region],
                     delta, width=0.25, color=color, alpha=0.7,
                     label=region.replace("_", " ").title())

        ax_d.set_xlabel("ODE Step")
        ax_d.set_ylabel(f"Δ {title} (improvement)")
        ax_d.set_title(f"Per-Step Improvement ({title})")
        ax_d.legend()
        ax_d.grid(True, alpha=0.3)
        ax_d.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perceptual_curves.png"), dpi=150)
    plt.close()
    print(f"  Saved {os.path.join(output_dir, 'perceptual_curves.png')}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated sample names (default: all)")
    parser.add_argument("--skip_lpips", action="store_true",
                        help="Skip LPIPS computation (faster)")
    parser.add_argument("--shard_id", type=int, default=None,
                        help="Shard index for multi-GPU parallelism")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Total number of shards")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Discover samples
    if args.samples:
        sample_names = args.samples.split(",")
    else:
        sample_names = sorted([
            d for d in os.listdir(args.traj_dir)
            if os.path.isdir(os.path.join(args.traj_dir, d))
            and os.path.isfile(os.path.join(args.traj_dir, d, "ode_schedule.json"))
        ])

    # Shard
    if args.shard_id is not None and args.num_shards is not None:
        sample_names = sample_names[args.shard_id::args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(sample_names)} samples")

    # Load schedule
    with open(os.path.join(args.traj_dir, sample_names[0], "ode_schedule.json")) as f:
        schedule = json.load(f)
    t_list = schedule["t_list"]
    num_steps = schedule["num_steps"]

    print(f"Samples: {len(sample_names)}")
    print(f"Steps: {num_steps}, t range: [{t_list[0]:.4f}, {t_list[-2]:.4f}]")

    # Load VAE
    print("Loading VAE...")
    model_manager = ModelManager(device="cpu", infer=True)
    model_manager.load_models([args.vae_path], torch_dtype=torch.bfloat16, device="cpu")
    vae_idx = model_manager.model_name.index("wan_video_vae")
    vae = model_manager.model[vae_idx].to(device)

    # Load LPIPS model
    lpips_model = None
    if not args.skip_lpips:
        import lpips
        print("Loading LPIPS model...")
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    # Load pixel mask and compute mouth bbox
    mask_keep = load_pixel_mask(args.mask_path, H=512, W=512)
    mouth_bbox = get_mouth_bbox(mask_keep)
    print(f"Mouth bbox: y=[{mouth_bbox[0]}, {mouth_bbox[1]}), x=[{mouth_bbox[2]}, {mouth_bbox[3]})")

    # CSV output
    shard_suffix = f"_shard{args.shard_id}" if args.shard_id is not None else ""
    csv_path = os.path.join(args.output_dir, f"perceptual_metrics{shard_suffix}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "t", "sample", "metric", "region", "value"])

    total_decodes = len(sample_names) * (num_steps + 1)  # +1 for GT
    decode_count = 0
    t_start = time.time()

    for si, sample_name in enumerate(sample_names):
        print(f"\n[{si+1}/{len(sample_names)}] {sample_name}")

        # Decode GT once
        gt_latents = torch.load(
            os.path.join(args.traj_dir, sample_name, "input_latents.pt"),
            map_location="cpu", weights_only=True,
        ).float()
        gt_frames = decode_latents(vae, gt_latents, device)  # [T, 3, H, W]
        decode_count += 1

        for step_i in range(num_steps):
            t_val = t_list[step_i]

            # Decode x0_pred
            x0 = torch.load(
                os.path.join(args.traj_dir, sample_name, f"step_{step_i:03d}_x0.pt"),
                map_location="cpu", weights_only=True,
            ).float()
            pred_frames = decode_latents(vae, x0, device)  # [T, 3, H, W]
            decode_count += 1

            # Truncate to same length
            T = min(pred_frames.shape[0], gt_frames.shape[0])
            pred = pred_frames[:T]
            gt = gt_frames[:T]

            # — Pixel MSE —
            for region, mask in [
                ("mouth", ~mask_keep),
                ("upper_face", mask_keep),
                ("full", torch.ones_like(mask_keep)),
            ]:
                mse = compute_masked_mse(pred, gt, mask)
                writer.writerow([step_i, f"{t_val:.6f}", sample_name, "pixel_mse", region, f"{mse:.8f}"])

            # — SSIM —
            for region, mask in [
                ("mouth", ~mask_keep),
                ("upper_face", mask_keep),
                ("full", torch.ones_like(mask_keep)),
            ]:
                s = compute_masked_ssim(pred, gt, mask)
                writer.writerow([step_i, f"{t_val:.6f}", sample_name, "ssim", region, f"{s:.8f}"])

            # — LPIPS —
            if lpips_model is not None:
                lp_mouth = compute_lpips_crop(lpips_model, pred, gt, mouth_bbox, device)
                writer.writerow([step_i, f"{t_val:.6f}", sample_name, "lpips", "mouth", f"{lp_mouth:.8f}"])

                lp_full = compute_lpips_full(lpips_model, pred, gt, device)
                writer.writerow([step_i, f"{t_val:.6f}", sample_name, "lpips", "full", f"{lp_full:.8f}"])

            # Progress
            elapsed = time.time() - t_start
            rate = decode_count / elapsed
            remaining = (total_decodes - decode_count) / rate if rate > 0 else 0
            if step_i % 10 == 0 or step_i == num_steps - 1:
                print(f"  step {step_i:2d} (t={t_val:.3f}) | "
                      f"decoded {decode_count}/{total_decodes} | "
                      f"ETA {remaining/60:.1f} min")

            csv_file.flush()

    csv_file.close()
    elapsed_total = time.time() - t_start
    print(f"\nDone! {decode_count} VAE decodes in {elapsed_total/60:.1f} min")
    print(f"CSV: {csv_path}")

    # Plot (only when not sharded — use merge_and_plot for multi-GPU)
    if args.shard_id is None:
        print("Plotting...")
        plot_curves(csv_path, t_list, args.output_dir)


def merge_and_plot(output_dir: str, traj_dir: str):
    """Merge shard CSVs and produce plots. Run after all shards finish."""
    import glob
    import pandas as pd

    shard_files = sorted(glob.glob(os.path.join(output_dir, "perceptual_metrics_shard*.csv")))
    if not shard_files:
        print("No shard files found")
        return

    dfs = [pd.read_csv(f) for f in shard_files]
    merged = pd.concat(dfs, ignore_index=True)
    merged_path = os.path.join(output_dir, "perceptual_metrics.csv")
    merged.to_csv(merged_path, index=False)
    print(f"Merged {len(shard_files)} shards → {merged_path} ({len(merged)} rows)")

    # Load t_list for plotting
    sample_name = sorted([
        d for d in os.listdir(traj_dir)
        if os.path.isdir(os.path.join(traj_dir, d))
        and os.path.isfile(os.path.join(traj_dir, d, "ode_schedule.json"))
    ])[0]
    with open(os.path.join(traj_dir, sample_name, "ode_schedule.json")) as f:
        t_list = json.load(f)["t_list"]

    plot_curves(merged_path, t_list, output_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        # Usage: python eval_ode_perceptual.py --merge --output_dir DIR --traj_dir DIR
        parser = argparse.ArgumentParser()
        parser.add_argument("--merge", action="store_true")
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--traj_dir", type=str, required=True)
        merge_args = parser.parse_args()
        merge_and_plot(merge_args.output_dir, merge_args.traj_dir)
    else:
        main()
