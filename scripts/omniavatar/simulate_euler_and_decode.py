"""Simulate 1-step Euler jumps from step 0, run teacher on simulated x_t, decode & compare.

For each sample and each landing step:
  1. Load x_t and x0 from step 0
  2. Euler-jump to the landing step's noise level → x_t_simulated
  3. Run the teacher on x_t_simulated → x0_from_sim
  4. Decode both x0_from_sim and x0_from_real (already saved) via VAE
  5. Save side-by-side frame grids and stitched videos

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/simulate_euler_and_decode.py \
        --traj_dir /home/work/.local/ode_full_trajectories/14B \
        --output_dir /home/work/.local/ode_analysis/14B/euler_decoded \
        --samples "17ef723a912e46713e84fc2b7dd74e23_shot_001_000,283979598869ec6d8c5cdbe66eb5ecb8_shot_001_000" \
        --landing_steps "13,25,38,49"
"""

import argparse
import json
import os
import sys
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.omniavatar_wan import OmniAvatarWan
from models.wan_vae import load_wan_vae


# ── Paths ──
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRETRAINED = os.environ.get("WEIGHTS_ROOT", "/home/work/.local/OmniAvatar/pretrained_models")
BASE_14B = ",".join([
    f"{PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-{i:05d}-of-00006.safetensors"
    for i in range(1, 7)
])
CKPT_14B = os.environ.get("TEACHER_CKPT", "/home/work/output_omniavatar_v2v_phase2/step-10500.pt")
VAE_PATH = f"{PRETRAINED}/Wan2.1-T2V-14B/Wan2.1_VAE.pth"   # derived, no separate env var (matches source)
MASK_PATH = os.environ.get("MASK_PATH", os.path.join(REPO_ROOT, "data", "mask.png"))
DATA_DIR = os.environ.get("RECON_DATA_DIR", "/home/work/stableavatar_data/v2v_validation_data/recon")
NEG_TEXT_EMB = os.environ.get("NEG_TEXT_EMB", "/home/work/stableavatar_data/neg_text_emb.pt")


def load_mask(latent_h=64, latent_w=64):
    mask_arr = np.array(Image.open(MASK_PATH), dtype=np.float32)[:, :, 0] / 255.0
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_t, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
    return (mask_resized.squeeze() > 0.5).float()


def load_condition(sample_name, mask, neg_text_embeds, device, dtype):
    """Load condition dict for a sample (same as generate_omniavatar_ode_pairs_full.py)."""
    sample_dir = os.path.join(DATA_DIR, sample_name)

    vae_path = os.path.join(sample_dir, "vae_latents_mask_all.pt")
    if not os.path.exists(vae_path):
        vae_path = os.path.join(sample_dir, "vae_latents.pt")
    vae_data = torch.load(vae_path, map_location="cpu", weights_only=False)
    input_latents = vae_data["input_latents"].to(dtype)
    masked_latents = vae_data["masked_latents"].to(dtype)

    audio_data = torch.load(os.path.join(sample_dir, "audio_emb_omniavatar.pt"), map_location="cpu", weights_only=False)
    audio_emb = audio_data["audio_emb"][:81].to(dtype)

    text_emb = torch.load(os.path.join(sample_dir, "text_emb.pt"), map_location="cpu", weights_only=False)
    if isinstance(text_emb, dict):
        text_emb = next(v for v in text_emb.values() if isinstance(v, torch.Tensor))
    text_emb = text_emb.to(dtype)
    if text_emb.dim() == 2:
        text_emb = text_emb.unsqueeze(0)

    ref_latent = input_latents[:, 0:1, :, :]

    ref_seq = None
    ref_path = os.path.join(sample_dir, "ref_latents.pt")
    if os.path.exists(ref_path):
        ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
        ref_seq = ref_data["ref_sequence_latents"].to(dtype)

    condition = {
        "text_embeds": text_emb.unsqueeze(0).to(device),
        "audio_emb": audio_emb.unsqueeze(0).to(device),
        "ref_latent": ref_latent.unsqueeze(0).to(device),
        "mask": mask.to(device),
        "masked_video": masked_latents.unsqueeze(0).to(device),
        "ref_sequence": ref_seq.unsqueeze(0).to(device) if ref_seq is not None else torch.zeros_like(masked_latents.unsqueeze(0)).to(device),
    }
    if condition["text_embeds"].dim() == 4:
        condition["text_embeds"] = condition["text_embeds"].squeeze(1)

    neg_text = neg_text_embeds.to(device=device, dtype=dtype)
    if neg_text.dim() == 2:
        neg_text = neg_text.unsqueeze(0)
    neg_condition = {
        "text_embeds": neg_text,
        "audio_emb": torch.zeros_like(condition["audio_emb"]),
        "ref_latent": condition["ref_latent"],
        "mask": condition["mask"],
        "masked_video": condition["masked_video"],
        "ref_sequence": condition["ref_sequence"],
    }

    return condition, neg_condition


def teacher_predict_x0(teacher, x_t, t_val, condition, neg_condition, guidance_scale=4.5):
    """Run teacher with CFG to get x0_pred."""
    t_tensor = torch.tensor([t_val], device=x_t.device, dtype=torch.float32)
    x0_cond = teacher(x_t, t_tensor, condition=condition, fwd_pred_type="x0")
    if guidance_scale != 1.0:
        x0_uncond = teacher(x_t, t_tensor, condition=neg_condition, fwd_pred_type="x0")
        x0_pred = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
    else:
        x0_pred = x0_cond
    return x0_pred


def decode_latents(vae, latents, device):
    latents = latents.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        frames = vae.decode(latents, device=device, tiled=False)
    if frames.dim() == 5:
        frames = frames[0]
    frames = ((frames.float() + 1) / 2 * 255).clamp(0, 255).byte()
    return frames.permute(1, 2, 3, 0).cpu().numpy()


def add_label(frame_np, label, font_size=16):
    img = Image.fromarray(frame_np)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle([0, 0, bbox[2] + 6, bbox[3] + 6], fill="black")
    draw.text((3, 3), label, fill="white", font=font)
    return np.array(img)


def save_video(frames_np, path, fps=25):
    import imageio
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--samples", type=str, required=True)
    parser.add_argument("--landing_steps", type=str, default="13,25,38,49")
    parser.add_argument("--display_frames", type=str, default="10,40,60")
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    sample_names = args.samples.split(",")
    landing_steps = [int(s) for s in args.landing_steps.split(",")]
    display_frames = [int(f) for f in args.display_frames.split(",")]

    # Load schedule
    with open(os.path.join(args.traj_dir, sample_names[0], "ode_schedule.json")) as f:
        schedule = json.load(f)
    t_list = schedule["t_list"]

    # Load teacher
    print("Loading 14B teacher...")
    teacher = OmniAvatarWan(
        model_size="14B", in_dim=65, mode="v2v", use_audio=True,
        base_model_paths=BASE_14B, omniavatar_ckpt_path=CKPT_14B,
        merge_lora=True, net_pred_type="flow", schedule_type="rf",
    ).to(device, dtype=dtype).eval()
    teacher.requires_grad_(False)
    print("Teacher loaded.")

    # Load VAE
    print("Loading VAE...")
    vae = load_wan_vae(VAE_PATH, device=device)
    print("VAE loaded.")

    # Load mask and neg text
    mask = load_mask()
    neg_text_embeds = torch.load(NEG_TEXT_EMB, map_location="cpu", weights_only=False)
    if isinstance(neg_text_embeds, dict):
        neg_text_embeds = next(v for v in neg_text_embeds.values() if isinstance(v, torch.Tensor))
    neg_text_embeds = neg_text_embeds.to(dtype)

    for sample_name in sample_names:
        print(f"\n=== {sample_name} ===")
        sample_out = os.path.join(args.output_dir, sample_name)
        os.makedirs(sample_out, exist_ok=True)

        # Load condition
        condition, neg_condition = load_condition(sample_name, mask, neg_text_embeds, device, dtype)

        # Load step 0 data
        x_t_0 = torch.load(os.path.join(args.traj_dir, sample_name, "step_000_xt.pt"),
                            map_location="cpu", weights_only=True).float()
        x0_0 = torch.load(os.path.join(args.traj_dir, sample_name, "step_000_x0.pt"),
                           map_location="cpu", weights_only=True).float()
        gt_latents = torch.load(os.path.join(args.traj_dir, sample_name, "input_latents.pt"),
                                map_location="cpu", weights_only=True).float()
        t_0 = t_list[0]

        # Decode GT and final output once
        print("  Decoding GT...")
        gt_frames = decode_latents(vae, gt_latents, device)

        print("  Decoding final output (step 49 real x0)...")
        x0_final = torch.load(
            os.path.join(args.traj_dir, sample_name, "step_049_x0.pt"),
            map_location="cpu", weights_only=True,
        ).float()
        final_frames = decode_latents(vae, x0_final, device)

        # Decode step 0's x0 once (same for all landings)
        x0_step0_frames = decode_latents(vae, x0_0, device)

        for landing in landing_steps:
            t_land = t_list[landing]
            print(f"  Landing step {landing} (t={t_land:.4f})...")

            # 1. Euler jump from step 0
            eps = (x_t_0 - (1.0 - t_0) * x0_0) / t_0
            if t_land > 1e-6:
                x_t_sim = (1.0 - t_land) * x0_0 + t_land * eps
            else:
                x_t_sim = x0_0
            x_t_sim = x_t_sim.unsqueeze(0).to(device=device, dtype=dtype)

            # 2. Run teacher on simulated x_t
            print(f"    Running teacher on simulated x_t...")
            with torch.no_grad():
                x0_from_sim = teacher_predict_x0(
                    teacher, x_t_sim, t_land, condition, neg_condition, args.guidance_scale
                )

            # 3. Load teacher's actual x0 at this step (from saved trajectory)
            x0_from_real = torch.load(
                os.path.join(args.traj_dir, sample_name, f"step_{landing:03d}_x0.pt"),
                map_location="cpu", weights_only=True,
            ).float()

            # 4. Decode both
            print(f"    Decoding x0_from_sim and x0_from_real...")
            sim_frames = decode_latents(vae, x0_from_sim.squeeze(0).float().cpu(), device)
            real_frames = decode_latents(vae, x0_from_real, device)

            # 5. Compute amplified difference maps against final output
            DIFF_AMP = 5  # Amplification factor for visibility
            def diff_map(a, b, amp=DIFF_AMP):
                """Amplified absolute difference between two uint8 frame arrays."""
                diff = np.abs(a.astype(np.float32) - b.astype(np.float32)) * amp
                return np.clip(diff, 0, 255).astype(np.uint8)

            # 6. Save frame grids with diff columns:
            # GT | x0@step0 | diff(step0,final) | Teacher(Euler) | diff(Euler,final) | Teacher(real) | diff(real,final) | Final
            for fi in display_frames:
                fi_safe = min(fi, len(gt_frames) - 1)
                row = np.concatenate([
                    add_label(gt_frames[fi_safe], "GT"),
                    add_label(x0_step0_frames[fi_safe], "x0@step0 (1-step)"),
                    add_label(diff_map(x0_step0_frames[fi_safe], final_frames[fi_safe]), f"|step0 - final| x{DIFF_AMP}"),
                    add_label(sim_frames[fi_safe], f"Teacher(Euler @s{landing})"),
                    add_label(diff_map(sim_frames[fi_safe], final_frames[fi_safe]), f"|Euler - final| x{DIFF_AMP}"),
                    add_label(real_frames[fi_safe], f"Teacher(real @s{landing})"),
                    add_label(diff_map(real_frames[fi_safe], final_frames[fi_safe]), f"|real - final| x{DIFF_AMP}"),
                    add_label(final_frames[fi_safe], "Final (s49 real)"),
                ], axis=1)
                grid_path = os.path.join(sample_out, f"euler_s{landing:03d}_frame{fi:03d}.png")
                Image.fromarray(row).save(grid_path)

            # 7. Save stitched video (compact: skip diff maps, they don't read well in video)
            min_T = min(len(gt_frames), len(sim_frames), len(real_frames),
                        len(x0_step0_frames), len(final_frames))
            stitched = np.concatenate([
                np.stack([add_label(gt_frames[i], "GT") for i in range(min_T)]),
                np.stack([add_label(x0_step0_frames[i], "x0@step0 (1-step)") for i in range(min_T)]),
                np.stack([add_label(sim_frames[i], f"Teacher(Euler @s{landing})") for i in range(min_T)]),
                np.stack([add_label(real_frames[i], f"Teacher(real @s{landing})") for i in range(min_T)]),
                np.stack([add_label(final_frames[i], "Final (s49)") for i in range(min_T)]),
            ], axis=2)
            vid_path = os.path.join(sample_out, f"euler_s{landing:03d}_stitched.mp4")
            save_video(stitched, vid_path)

            # Mux audio
            audio_path = os.path.join(DATA_DIR, sample_name, "audio.wav")
            if os.path.exists(audio_path):
                out_path = vid_path.replace(".mp4", "_audio.mp4")
                subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", vid_path, "-i", audio_path,
                    "-c:v", "libx264", "-crf", "18", "-c:a", "aac",
                    "-shortest", out_path,
                ], capture_output=True)

            print(f"    Saved grid + video for landing step {landing}")

            # Cleanup GPU
            del x_t_sim, x0_from_sim
            torch.cuda.empty_cache()

    print(f"\nDone! Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
