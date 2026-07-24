"""Analyze ODE trajectory data to identify which timesteps drive lip/audio generation.

Four analyses:
  1. x0_pred vs GT similarity (MSE, cosine) in mask vs non-mask regions across timesteps
  2. Δ-contribution per step (similarity improvement over previous step)
  3. Audio ablation: ||x0_with_audio - x0_no_audio|| in mask region per step
  4. Inter-sample variance in mask region at each step

Usage:
    python scripts/analyze_ode_trajectory.py \
        --traj_dir /home/work/ode_full_trajectories/14B \
        --mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
        --output_dir /home/work/ode_analysis/14B

    # With audio ablation (requires no-audio trajectories):
    python scripts/analyze_ode_trajectory.py \
        --traj_dir /home/work/ode_full_trajectories/14B \
        --no_audio_traj_dir /home/work/ode_full_trajectories/14B_no_audio \
        --mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
        --output_dir /home/work/ode_analysis/14B
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Mask utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_latent_mask(mask_path: str, latent_h: int = 64, latent_w: int = 64) -> torch.Tensor:
    """Load LatentSync mask and resize to latent resolution.

    Returns:
        mask_keep: [H, W] bool tensor. True=keep (upper face), False=mask (mouth).
    """
    mask_img = Image.open(mask_path)
    mask_arr = np.array(mask_img, dtype=np.float32)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    mask_arr = mask_arr / 255.0
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_t, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
    return mask_resized.squeeze() > 0.5  # True=keep, False=mouth


def apply_region_mask(tensor: torch.Tensor, mask_keep: torch.Tensor, region: str) -> torch.Tensor:
    """Extract values from a [C, T, H, W] tensor for a spatial region.

    Args:
        tensor: [C, T, H, W]
        mask_keep: [H, W] bool (True=keep/upper face)
        region: "mouth" or "upper_face"

    Returns:
        Flat tensor of selected values.
    """
    if region == "mouth":
        spatial_mask = ~mask_keep  # Invert: select mouth region
    elif region == "upper_face":
        spatial_mask = mask_keep
    else:
        raise ValueError(f"Unknown region: {region}")
    # Broadcast: [H, W] → [1, 1, H, W] → [C, T, H, W]
    expanded = spatial_mask.unsqueeze(0).unsqueeze(0).expand_as(tensor)
    return tensor[expanded]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_mse(a: torch.Tensor, b: torch.Tensor, mask_keep: torch.Tensor, region: str) -> float:
    """MSE between two [C, T, H, W] tensors in a spatial region."""
    va = apply_region_mask(a, mask_keep, region)
    vb = apply_region_mask(b, mask_keep, region)
    return (va - vb).pow(2).mean().item()


def compute_cosine_sim(a: torch.Tensor, b: torch.Tensor, mask_keep: torch.Tensor, region: str) -> float:
    """Cosine similarity between two [C, T, H, W] tensors in a spatial region."""
    va = apply_region_mask(a, mask_keep, region).float()
    vb = apply_region_mask(b, mask_keep, region).float()
    if va.norm() < 1e-8 or vb.norm() < 1e-8:
        return 0.0
    return F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()


def compute_l2_norm(a: torch.Tensor, mask_keep: torch.Tensor, region: str) -> float:
    """L2 norm of values in a spatial region of a [C, T, H, W] tensor."""
    va = apply_region_mask(a, mask_keep, region).float()
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


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1 & 2: x0_pred vs GT similarity + Δ-contribution
# ─────────────────────────────────────────────────────────────────────────────

def analyze_gt_similarity(
    traj_dir: str, samples: List[str], mask_keep: torch.Tensor, num_steps: int
) -> Dict[str, np.ndarray]:
    """Compute x0_pred vs GT similarity at each step, for mouth and upper_face regions."""
    # Accumulators: [num_steps] per metric per region
    metrics = {
        f"{region}_{metric}": np.zeros(num_steps)
        for region in ["mouth", "upper_face"]
        for metric in ["mse", "cosine"]
    }

    for sample_name in samples:
        gt = load_tensor(traj_dir, sample_name, "input_latents.pt")
        for step_i in range(num_steps):
            x0 = load_tensor(traj_dir, sample_name, f"step_{step_i:03d}_x0.pt")
            for region in ["mouth", "upper_face"]:
                metrics[f"{region}_mse"][step_i] += compute_mse(x0, gt, mask_keep, region)
                metrics[f"{region}_cosine"][step_i] += compute_cosine_sim(x0, gt, mask_keep, region)

    # Average over samples
    n = len(samples)
    for k in metrics:
        metrics[k] /= n

    # Δ-contribution: improvement at step i relative to step i-1
    for region in ["mouth", "upper_face"]:
        cosine_vals = metrics[f"{region}_cosine"]
        delta = np.zeros(num_steps)
        delta[0] = cosine_vals[0]  # First step improvement from "nothing"
        delta[1:] = cosine_vals[1:] - cosine_vals[:-1]
        metrics[f"{region}_delta_cosine"] = delta

        mse_vals = metrics[f"{region}_mse"]
        delta_mse = np.zeros(num_steps)
        delta_mse[0] = 0.0
        delta_mse[1:] = mse_vals[:-1] - mse_vals[1:]  # MSE decrease = improvement
        metrics[f"{region}_delta_mse"] = delta_mse

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3: Audio ablation
# ─────────────────────────────────────────────────────────────────────────────

def analyze_audio_ablation(
    traj_dir: str, no_audio_traj_dir: str, samples: List[str],
    mask_keep: torch.Tensor, num_steps: int
) -> Dict[str, np.ndarray]:
    """||x0_with_audio - x0_no_audio|| per step, in mouth and upper_face regions."""
    metrics = {
        f"{region}_audio_diff_l2": np.zeros(num_steps)
        for region in ["mouth", "upper_face"]
    }

    # Also track per-element MSE (normalized by region size)
    for region in ["mouth", "upper_face"]:
        metrics[f"{region}_audio_diff_mse"] = np.zeros(num_steps)

    valid_count = 0
    for sample_name in samples:
        no_audio_dir = os.path.join(no_audio_traj_dir, sample_name)
        if not os.path.isdir(no_audio_dir):
            print(f"[Ablation] Skipping {sample_name}: no-audio trajectory not found")
            continue

        valid_count += 1
        for step_i in range(num_steps):
            x0_audio = load_tensor(traj_dir, sample_name, f"step_{step_i:03d}_x0.pt")
            x0_noaudio = load_tensor(no_audio_traj_dir, sample_name, f"step_{step_i:03d}_x0.pt")
            diff = x0_audio - x0_noaudio

            for region in ["mouth", "upper_face"]:
                metrics[f"{region}_audio_diff_l2"][step_i] += compute_l2_norm(diff, mask_keep, region)
                vals = apply_region_mask(diff, mask_keep, region)
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
    traj_dir: str, samples: List[str], mask_keep: torch.Tensor, num_steps: int
) -> Dict[str, np.ndarray]:
    """Variance of x0_pred across samples at each step, in mouth vs upper_face.

    Computes two complementary measures:
    - scalar_variance: variance of per-sample region means (captures overall level shifts)
    - pixelwise_variance: mean of per-pixel variances across samples (captures spatial detail diversity)

    High variance in mouth = audio is driving diverse lip shapes.
    Low variance = model hasn't resolved audio-specific content yet.
    """
    metrics = {
        f"{region}_x0_variance": np.zeros(num_steps)
        for region in ["mouth", "upper_face"]
    }
    metrics.update({
        f"{region}_x0_pixelwise_var": np.zeros(num_steps)
        for region in ["mouth", "upper_face"]
    })

    n_samples = len(samples)
    if n_samples < 2:
        print("[Variance] Need at least 2 samples for variance analysis")
        return metrics

    for step_i in range(num_steps):
        for region in ["mouth", "upper_face"]:
            scalar_values = []
            all_region_vals = []
            for sample_name in samples:
                x0 = load_tensor(traj_dir, sample_name, f"step_{step_i:03d}_x0.pt")
                region_vals = apply_region_mask(x0, mask_keep, region)
                scalar_values.append(region_vals.mean().item())
                all_region_vals.append(region_vals)
            metrics[f"{region}_x0_variance"][step_i] = np.var(scalar_values)
            # Per-pixel variance: stack all samples, compute var along sample dim, then mean
            stacked = torch.stack(all_region_vals, dim=0)  # [N_samples, num_pixels]
            metrics[f"{region}_x0_pixelwise_var"][step_i] = stacked.var(dim=0).mean().item()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_gt_similarity(metrics: dict, t_list: list, output_dir: str):
    """Plot x0_pred vs GT similarity curves (Analysis 1 & 2)."""
    num_steps = len(t_list) - 1  # t_list has num_steps+1 entries
    steps = np.arange(num_steps)
    t_values = np.array(t_list[:num_steps])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) Cosine similarity to GT
    ax = axes[0, 0]
    ax.plot(steps, metrics["mouth_cosine"], "r-o", markersize=2, label="Mouth region")
    ax.plot(steps, metrics["upper_face_cosine"], "b-s", markersize=2, label="Upper face")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("Cosine Similarity to GT")
    ax.set_title("(a) x0_pred vs GT Cosine Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Secondary x-axis: timestep
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_positions = [0, 10, 20, 30, 40, 49]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"t={t_values[i]:.3f}" for i in tick_positions])
    ax2.set_xlabel("Timestep t")

    # (b) MSE to GT
    ax = axes[0, 1]
    ax.plot(steps, metrics["mouth_mse"], "r-o", markersize=2, label="Mouth region")
    ax.plot(steps, metrics["upper_face_mse"], "b-s", markersize=2, label="Upper face")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("MSE to GT")
    ax.set_title("(b) x0_pred vs GT MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # (c) Δ cosine similarity (per-step improvement)
    ax = axes[1, 0]
    ax.bar(steps - 0.15, metrics["mouth_delta_cosine"], width=0.3, color="red", alpha=0.7, label="Mouth")
    ax.bar(steps + 0.15, metrics["upper_face_delta_cosine"], width=0.3, color="blue", alpha=0.7, label="Upper face")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("Δ Cosine Similarity")
    ax.set_title("(c) Per-Step Improvement (Cosine)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # (d) Δ MSE (per-step MSE decrease)
    ax = axes[1, 1]
    ax.bar(steps - 0.15, metrics["mouth_delta_mse"], width=0.3, color="red", alpha=0.7, label="Mouth")
    ax.bar(steps + 0.15, metrics["upper_face_delta_mse"], width=0.3, color="blue", alpha=0.7, label="Upper face")
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

    # (a) L2 norm of (x0_audio - x0_noaudio)
    ax = axes[0]
    ax.plot(steps, metrics["mouth_audio_diff_l2"], "r-o", markersize=3, label="Mouth region")
    ax.plot(steps, metrics["upper_face_audio_diff_l2"], "b-s", markersize=3, label="Upper face")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("L2 Norm of Audio Difference")
    ax.set_title("(a) Audio Influence: ||x0_audio - x0_no_audio|| (L2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_positions = [0, 10, 20, 30, 40, 49]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"t={t_values[i]:.3f}" for i in tick_positions])
    ax2.set_xlabel("Timestep t")

    # (b) Per-element MSE of (x0_audio - x0_noaudio) — normalized by region size
    ax = axes[1]
    ax.plot(steps, metrics["mouth_audio_diff_mse"], "r-o", markersize=3, label="Mouth region")
    ax.plot(steps, metrics["upper_face_audio_diff_mse"], "b-s", markersize=3, label="Upper face")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("Per-Element MSE")
    ax.set_title("(b) Audio Influence: per-element MSE (size-normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add ratio subplot as annotation
    mouth_mse = np.array(metrics["mouth_audio_diff_mse"])
    upper_mse = np.array(metrics["upper_face_audio_diff_mse"])
    ratio = np.where(upper_mse > 1e-10, mouth_mse / upper_mse, 0)
    ax_ratio = ax.twinx()
    ax_ratio.plot(steps, ratio, "g--", alpha=0.5, linewidth=1, label="Mouth/Upper ratio")
    ax_ratio.set_ylabel("Mouth/Upper Ratio", color="green")
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

    # (a) Scalar variance (mean-level)
    ax = axes[0]
    ax.plot(steps, metrics["mouth_x0_variance"], "r-o", markersize=3, label="Mouth region")
    ax.plot(steps, metrics["upper_face_x0_variance"], "b-s", markersize=3, label="Upper face")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("Variance of per-sample region mean")
    ax.set_title("(a) Scalar Variance (overall level)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_positions = [0, 10, 20, 30, 40, 49]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"t={t_values[i]:.3f}" for i in tick_positions])
    ax2.set_xlabel("Timestep t")

    # (b) Pixelwise variance (spatial detail diversity)
    if "mouth_x0_pixelwise_var" in metrics:
        ax = axes[1]
        ax.plot(steps, metrics["mouth_x0_pixelwise_var"], "r-o", markersize=3, label="Mouth region")
        ax.plot(steps, metrics["upper_face_x0_pixelwise_var"], "b-s", markersize=3, label="Upper face")
        ax.set_xlabel("ODE Step")
        ax.set_ylabel("Mean per-pixel variance across samples")
        ax.set_title("(b) Pixelwise Variance (spatial detail diversity)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "inter_sample_variance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved {path}")


def plot_combined_summary(all_metrics: dict, t_list: list, output_dir: str):
    """Single-page summary combining the most informative curves."""
    num_steps = len(t_list) - 1
    steps = np.arange(num_steps)
    t_values = np.array(t_list[:num_steps])

    has_ablation = "mouth_audio_diff_mse" in all_metrics
    ncols = 3 if has_ablation else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    # (1) Cosine similarity convergence
    ax = axes[0]
    ax.plot(steps, all_metrics["mouth_cosine"], "r-", linewidth=2, label="Mouth")
    ax.plot(steps, all_metrics["upper_face_cosine"], "b-", linewidth=2, label="Upper face")
    ax.fill_between(steps, all_metrics["mouth_cosine"], all_metrics["upper_face_cosine"],
                     alpha=0.1, color="purple")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("Cosine Similarity to GT")
    ax.set_title("Convergence to Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) Δ-contribution (cosine improvement)
    ax = axes[1]
    ax.bar(steps - 0.15, all_metrics["mouth_delta_cosine"], width=0.3, color="red", alpha=0.7, label="Mouth")
    ax.bar(steps + 0.15, all_metrics["upper_face_delta_cosine"], width=0.3, color="blue", alpha=0.7, label="Upper face")
    ax.set_xlabel("ODE Step")
    ax.set_ylabel("Δ Cosine Similarity")
    ax.set_title("Per-Step Quality Improvement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # (3) Audio ablation (if available)
    if has_ablation:
        ax = axes[2]
        ax.plot(steps, all_metrics["mouth_audio_diff_mse"], "r-o", markersize=3, label="Mouth")
        ax.plot(steps, all_metrics["upper_face_audio_diff_mse"], "b-s", markersize=3, label="Upper face")
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
    parser = argparse.ArgumentParser(description="Analyze ODE trajectories for lip generation timing")
    parser.add_argument("--traj_dir", type=str, required=True,
                        help="Directory with ODE trajectory data (with audio)")
    parser.add_argument("--no_audio_traj_dir", type=str, default=None,
                        help="Directory with no-audio ODE trajectory data (for ablation)")
    parser.add_argument("--mask_path", type=str, required=True,
                        help="Path to LatentSync spatial mask PNG")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for analysis outputs (plots, JSON)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load mask
    mask_keep = load_latent_mask(args.mask_path)
    mouth_frac = (~mask_keep).float().mean().item()
    upper_frac = mask_keep.float().mean().item()
    print(f"[Mask] Loaded {mask_keep.shape}: upper_face={upper_frac:.1%}, mouth={mouth_frac:.1%}")

    # Discover samples
    samples = discover_samples(args.traj_dir)
    print(f"[Data] Found {len(samples)} samples in {args.traj_dir}")
    if not samples:
        print("ERROR: No valid samples found")
        return

    # Load schedule from first sample
    schedule = load_schedule(args.traj_dir, samples[0])
    num_steps = schedule["num_steps"]
    t_list = schedule["t_list"]
    print(f"[Schedule] {num_steps} steps, shift={schedule['shift']}, "
          f"t range=[{t_list[0]:.4f}, {t_list[-1]:.4f}]")

    all_metrics = {}

    # ── Analysis 1 & 2: GT similarity + Δ-contribution ──
    print(f"\n[Analysis 1&2] Computing x0_pred vs GT similarity ({len(samples)} samples × {num_steps} steps)...")
    gt_metrics = analyze_gt_similarity(args.traj_dir, samples, mask_keep, num_steps)
    all_metrics.update(gt_metrics)
    plot_gt_similarity(gt_metrics, t_list, args.output_dir)

    # Print top-5 most productive steps for mouth region
    mouth_delta = gt_metrics["mouth_delta_cosine"]
    top5 = np.argsort(mouth_delta)[::-1][:5]
    print(f"[Analysis 2] Top 5 Δ-cosine steps for MOUTH:")
    for rank, i in enumerate(top5):
        print(f"  #{rank+1}: step {i} (t={t_list[i]:.4f}) → Δcos={mouth_delta[i]:.6f}")

    # ── Analysis 3: Audio ablation ──
    if args.no_audio_traj_dir and os.path.isdir(args.no_audio_traj_dir):
        no_audio_samples = discover_samples(args.no_audio_traj_dir)
        common_samples = [s for s in samples if s in no_audio_samples]
        print(f"\n[Analysis 3] Audio ablation: {len(common_samples)} common samples")
        if common_samples:
            ablation_metrics = analyze_audio_ablation(
                args.traj_dir, args.no_audio_traj_dir, common_samples,
                mask_keep, num_steps
            )
            all_metrics.update(ablation_metrics)
            plot_audio_ablation(ablation_metrics, t_list, args.output_dir)

            # Print peak audio influence steps
            mouth_audio = ablation_metrics["mouth_audio_diff_mse"]
            top5_audio = np.argsort(mouth_audio)[::-1][:5]
            print(f"[Analysis 3] Top 5 audio-influence steps for MOUTH (MSE):")
            for rank, i in enumerate(top5_audio):
                print(f"  #{rank+1}: step {i} (t={t_list[i]:.4f}) → audio_mse={mouth_audio[i]:.6f}")
    else:
        print(f"\n[Analysis 3] Skipped (no --no_audio_traj_dir or dir missing)")

    # ── Analysis 4: Inter-sample variance ──
    print(f"\n[Analysis 4] Computing inter-sample variance ({len(samples)} samples)...")
    var_metrics = analyze_inter_sample_variance(args.traj_dir, samples, mask_keep, num_steps)
    all_metrics.update(var_metrics)
    plot_inter_sample_variance(var_metrics, t_list, args.output_dir)

    # Peak variance step
    mouth_var = var_metrics["mouth_x0_variance"]
    peak_step = np.argmax(mouth_var)
    print(f"[Analysis 4] Peak mouth variance at step {peak_step} (t={t_list[peak_step]:.4f})")

    # ── Combined summary plot ──
    plot_combined_summary(all_metrics, t_list, args.output_dir)

    # ── Save all metrics as JSON ──
    json_metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in all_metrics.items()}
    json_metrics["t_list"] = t_list
    json_metrics["num_steps"] = num_steps
    json_metrics["samples"] = samples

    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"\n[Done] All metrics saved to {json_path}")
    print(f"[Done] Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
