"""Decode x0_pred at representative ODE steps to video frames for visual comparison.

For selected samples, decodes x0_pred at steps [0, 13, 25, 38, 49] + GT input_latents.
Produces:
  - Per-sample frame grids (single representative frame from each step)
  - Per-sample stitched videos (full 81-frame comparison, side-by-side)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/decode_ode_trajectory.py \
        --traj_dir /home/work/ode_full_trajectories/14B \
        --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
        --mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
        --output_dir /home/work/ode_analysis/14B/decoded \
        --samples 17ef723a912e46713e84fc2b7dd74e23_shot_001_000,283979598869ec6d8c5cdbe66eb5ecb8_shot_001_000
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OmniAvatar.models.model_manager import ModelManager


def decode_latents(vae, latents, device):
    """Decode [16, 21, 64, 64] latents → [81, H, W, 3] uint8 numpy."""
    latents = latents.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        frames = vae.decode(latents, device=device, tiled=False)
    # frames: [1, 3, T_video, H, W] float in [-1, 1] (or [3, T, H, W])
    if frames.dim() == 5:
        frames = frames[0]
    frames = ((frames.float() + 1) / 2 * 255).clamp(0, 255).byte()
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, 3]
    return frames


def save_video(frames_np, path, fps=25):
    """Save [T, H, W, 3] uint8 numpy array as mp4 via imageio."""
    import imageio
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()


def add_label_to_frame(frame_np, label, font_size=18):
    """Add text label to top-left of a frame."""
    img = Image.fromarray(frame_np)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    # Black background for readability
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle([0, 0, bbox[2] + 6, bbox[3] + 6], fill="black")
    draw.text((3, 3), label, fill="white", font=font)
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated sample names (default: first 2)")
    parser.add_argument("--steps", type=str, default="0,13,25,38,49",
                        help="Comma-separated ODE steps to decode")
    parser.add_argument("--display_frames", type=str, default="10,40,60",
                        help="Video frame indices for the grid image")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = [int(s) for s in args.steps.split(",")]
    display_frames = [int(f) for f in args.display_frames.split(",")]

    # Discover samples
    if args.samples:
        sample_names = args.samples.split(",")
    else:
        sample_names = sorted([
            d for d in os.listdir(args.traj_dir)
            if os.path.isdir(os.path.join(args.traj_dir, d))
            and os.path.isfile(os.path.join(args.traj_dir, d, "ode_schedule.json"))
        ])[:2]

    print(f"Samples: {sample_names}")
    print(f"Steps to decode: {steps}")

    # Load schedule for labels
    with open(os.path.join(args.traj_dir, sample_names[0], "ode_schedule.json")) as f:
        schedule = json.load(f)
    t_list = schedule["t_list"]

    # Load VAE only
    print("Loading VAE...")
    model_manager = ModelManager(device="cpu", infer=True)
    model_manager.load_models(
        [args.vae_path],
        torch_dtype=torch.bfloat16, device="cpu",
    )
    # ModelManager stores models in .model list; VAE is the only one loaded
    vae_idx = model_manager.model_name.index("wan_video_vae")
    vae = model_manager.model[vae_idx]
    vae.to(device)
    print(f"VAE loaded on {device}")

    # Load mask for overlay (optional)
    mask_overlay = None
    if args.mask_path and os.path.exists(args.mask_path):
        mask_img = Image.open(args.mask_path).convert("L").resize((512, 512))
        mask_arr = np.array(mask_img)
        # Create a red overlay for the mouth region
        mask_overlay = np.zeros((512, 512, 3), dtype=np.uint8)
        mask_overlay[:, :, 0] = ((mask_arr < 128) * 60).astype(np.uint8)  # Light red on mouth

    for sample_name in sample_names:
        print(f"\n=== Processing {sample_name} ===")
        sample_dir = os.path.join(args.traj_dir, sample_name)
        sample_out = os.path.join(args.output_dir, sample_name)
        os.makedirs(sample_out, exist_ok=True)

        # Decode GT
        print("  Decoding GT...")
        gt_latents = torch.load(
            os.path.join(sample_dir, "input_latents.pt"),
            map_location="cpu", weights_only=True,
        ).float()
        gt_frames = decode_latents(vae, gt_latents, device)
        save_video(gt_frames, os.path.join(sample_out, "gt.mp4"))

        # Decode x0_pred at each step
        all_decoded = {"GT": gt_frames}
        for step_i in steps:
            t_val = t_list[step_i]
            label = f"step_{step_i:02d}_t={t_val:.3f}"
            print(f"  Decoding x0 at step {step_i} (t={t_val:.4f})...")
            x0 = torch.load(
                os.path.join(sample_dir, f"step_{step_i:03d}_x0.pt"),
                map_location="cpu", weights_only=True,
            ).float()
            frames = decode_latents(vae, x0, device)
            save_video(frames, os.path.join(sample_out, f"x0_step{step_i:03d}.mp4"))
            all_decoded[label] = frames

        # === Frame grid: selected frames from each step ===
        print("  Creating frame grids...")
        for frame_idx in display_frames:
            grid_imgs = []
            for name, frames in all_decoded.items():
                fi = min(frame_idx, len(frames) - 1)
                frame = frames[fi].copy()
                frame = add_label_to_frame(frame, name, font_size=16)
                if mask_overlay is not None:
                    h, w = frame.shape[:2]
                    if h == 512 and w == 512:
                        frame = np.clip(frame.astype(np.int16) + mask_overlay.astype(np.int16), 0, 255).astype(np.uint8)
                grid_imgs.append(frame)

            # Arrange in a row
            grid = np.concatenate(grid_imgs, axis=1)
            grid_path = os.path.join(sample_out, f"grid_frame{frame_idx:03d}.png")
            Image.fromarray(grid).save(grid_path)
            print(f"    Saved {grid_path}")

        # === Stitched video: side-by-side all steps ===
        print("  Creating stitched video...")
        # Label each video's frames
        labeled_videos = {}
        for name, frames in all_decoded.items():
            labeled = np.stack([add_label_to_frame(f, name, font_size=14) for f in frames])
            labeled_videos[name] = labeled

        # Concatenate horizontally: [T, H, W*N, 3]
        keys = list(labeled_videos.keys())
        min_T = min(v.shape[0] for v in labeled_videos.values())
        stitched_frames = np.concatenate(
            [labeled_videos[k][:min_T] for k in keys], axis=2
        )
        stitched_path = os.path.join(sample_out, "stitched_all_steps.mp4")
        save_video(stitched_frames, stitched_path, fps=25)
        print(f"    Saved {stitched_path}")

        # Also save a narrower comparison: just GT vs step 0 vs step 49
        if "GT" in labeled_videos and len(steps) >= 2:
            first_step_key = f"step_{steps[0]:02d}_t={t_list[steps[0]]:.3f}"
            last_step_key = f"step_{steps[-1]:02d}_t={t_list[steps[-1]]:.3f}"
            if first_step_key in labeled_videos and last_step_key in labeled_videos:
                trio = np.concatenate([
                    labeled_videos["GT"][:min_T],
                    labeled_videos[first_step_key][:min_T],
                    labeled_videos[last_step_key][:min_T],
                ], axis=2)
                trio_path = os.path.join(sample_out, "stitched_gt_vs_step0_vs_final.mp4")
                save_video(trio, trio_path, fps=25)
                print(f"    Saved {trio_path}")

        # Mux with audio if available
        audio_path = os.path.join(
            "/home/work/stableavatar_data/v2v_validation_data/recon",
            sample_name, "audio.wav"
        )
        if os.path.exists(audio_path):
            for vid_name in ["stitched_all_steps.mp4", "stitched_gt_vs_step0_vs_final.mp4"]:
                vid_path = os.path.join(sample_out, vid_name)
                if os.path.exists(vid_path):
                    out_path = vid_path.replace(".mp4", "_audio.mp4")
                    subprocess.run([
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", vid_path, "-i", audio_path,
                        "-c:v", "libx264", "-crf", "18", "-c:a", "aac",
                        "-shortest", out_path,
                    ], capture_output=True)
                    print(f"    Muxed audio: {out_path}")

    print(f"\nDone! Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
