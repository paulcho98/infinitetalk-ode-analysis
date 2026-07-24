"""Visualize per-step changes in x0_pred across the ODE trajectory.

For selected samples, generates mouth-crop visualizations:
  1. Per-step change map: |x0[i] - x0[i-1]| amplified
  2. Error vs GT: |x0[i] - GT| amplified
  3. Improvement/degradation map: |x0[i-1] - GT| - |x0[i] - GT| (green=better, red=worse)
  4. Mouth crop side-by-side at key steps

Reads from pre-decoded videos (output of eval_ode_perceptual_v2.py phase 1).

Usage:
    python scripts/visualize_ode_stepwise.py \
        --videos_dir /home/work/.local/ode_analysis/14B/perceptual_v2/videos \
        --output_dir /home/work/.local/ode_analysis/14B/stepwise_vis \
        --mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
        --samples "1ec7a45d803ab472e7fe5c6625667289_shot_001_000,17ef723a912e46713e84fc2b7dd74e23_shot_001_000"
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_pixel_mask(mask_path, H=512, W=512):
    mask_img = Image.open(mask_path)
    mask_arr = np.array(mask_img, dtype=np.float32)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    mask_arr = mask_arr / 255.0
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_t, size=(H, W), mode="bilinear", align_corners=False)
    return (mask_resized.squeeze() > 0.5).numpy()


def get_mouth_bbox(mask_keep, pad=16):
    mouth = ~mask_keep
    ys, xs = np.where(mouth)
    H, W = mask_keep.shape
    return (
        max(0, ys.min() - pad),
        min(H, ys.max() + 1 + pad),
        max(0, xs.min() - pad),
        min(W, xs.max() + 1 + pad),
    )


def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def crop_mouth(frame, bbox):
    y0, y1, x0, x1 = bbox
    return frame[y0:y1, x0:x1]


def add_label(img, text, font_size=14, position="top"):
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if position == "top":
        draw.rectangle([0, 0, tw + 6, th + 6], fill="black")
        draw.text((3, 3), text, fill="white", font=font)
    return np.array(pil)


def amplified_abs_diff(a, b, amp=5):
    """Amplified absolute difference, uint8."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)) * amp
    return np.clip(diff, 0, 255).astype(np.uint8)


def improvement_map(prev, curr, gt):
    """Per-pixel improvement map: green=better, red=worse.
    Computes |prev - gt| - |curr - gt| per channel, then visualizes.
    """
    prev_err = np.abs(prev.astype(np.float32) - gt.astype(np.float32))
    curr_err = np.abs(curr.astype(np.float32) - gt.astype(np.float32))
    # Positive = improvement (error decreased), negative = degradation
    delta = (prev_err - curr_err).mean(axis=2)  # [H, W] average across RGB

    # Map to green (improved) / red (worsened)
    out = np.zeros((*delta.shape, 3), dtype=np.uint8)
    amp = 10  # Amplification for visibility
    improved = np.clip(delta * amp, 0, 255).astype(np.uint8)
    worsened = np.clip(-delta * amp, 0, 255).astype(np.uint8)
    out[:, :, 1] = improved  # Green channel = better
    out[:, :, 0] = worsened  # Red channel = worse
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 1: Per-step change map (consecutive differences)
# ─────────────────────────────────────────────────────────────────────────────

def vis_consecutive_changes(all_frames, gt_frames, bbox, display_frame, steps, output_path):
    """Grid showing |x0[i] - x0[i-1]| for consecutive step pairs, mouth crop."""
    cols = []
    for i, step in enumerate(steps):
        if i == 0:
            # No previous step — show x0[step] itself
            crop = crop_mouth(all_frames[step][display_frame], bbox)
            cols.append(add_label(crop, f"x0 @ step {step}"))
        else:
            prev_step = steps[i - 1]
            curr_crop = all_frames[step][display_frame]
            prev_crop = all_frames[prev_step][display_frame]
            diff = amplified_abs_diff(
                crop_mouth(curr_crop, bbox),
                crop_mouth(prev_crop, bbox),
                amp=8,
            )
            cols.append(add_label(diff, f"|s{step} - s{prev_step}| x8"))

    row = np.concatenate(cols, axis=1)
    Image.fromarray(row).save(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 2: Error vs GT at each step
# ─────────────────────────────────────────────────────────────────────────────

def vis_error_vs_gt(all_frames, gt_frames, bbox, display_frame, steps, output_path):
    """Grid showing |x0[step] - GT| for each step, mouth crop."""
    gt_crop = crop_mouth(gt_frames[display_frame], bbox)
    cols = [add_label(gt_crop, "GT")]

    for step in steps:
        pred_crop = crop_mouth(all_frames[step][display_frame], bbox)
        diff = amplified_abs_diff(pred_crop, gt_crop, amp=5)
        cols.append(add_label(diff, f"|s{step} - GT| x5"))

    row = np.concatenate(cols, axis=1)
    Image.fromarray(row).save(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 3: Improvement/degradation map
# ─────────────────────────────────────────────────────────────────────────────

def vis_improvement_map(all_frames, gt_frames, bbox, display_frame, steps, output_path):
    """Grid showing where each step improved (green) or worsened (red) vs GT."""
    gt_crop = crop_mouth(gt_frames[display_frame], bbox)
    cols = [add_label(gt_crop, "GT")]

    for i, step in enumerate(steps):
        if i == 0:
            # Compare against zero (all improvement)
            prev_frame = np.full_like(all_frames[step][display_frame], 128)
        else:
            prev_frame = all_frames[steps[i - 1]][display_frame]

        curr_crop = crop_mouth(all_frames[step][display_frame], bbox)
        prev_crop = crop_mouth(prev_frame, bbox)
        imp = improvement_map(prev_crop, curr_crop, gt_crop)
        cols.append(add_label(imp, f"s{steps[i-1] if i > 0 else '?'}→s{step} G=better R=worse"))

    row = np.concatenate(cols, axis=1)
    Image.fromarray(row).save(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization 4: Mouth crops at key steps
# ─────────────────────────────────────────────────────────────────────────────

def vis_mouth_crops(all_frames, gt_frames, bbox, display_frame, steps, output_path):
    """Side-by-side mouth crops at key steps + GT."""
    gt_crop = crop_mouth(gt_frames[display_frame], bbox)
    cols = [add_label(gt_crop, "GT")]

    for step in steps:
        crop = crop_mouth(all_frames[step][display_frame], bbox)
        cols.append(add_label(crop, f"Step {step}"))

    row = np.concatenate(cols, axis=1)
    Image.fromarray(row).save(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Combined grid: all 4 visualizations stacked
# ─────────────────────────────────────────────────────────────────────────────

def vis_combined(all_frames, gt_frames, bbox, display_frame, steps, output_path):
    """4 rows stacked: mouth crops, error vs GT, consecutive changes, improvement map."""
    gt_crop = crop_mouth(gt_frames[display_frame], bbox)
    h, w = gt_crop.shape[:2]

    def make_row(cols_list):
        """Pad all columns to same height, concatenate."""
        max_h = max(c.shape[0] for c in cols_list)
        padded = []
        for c in cols_list:
            if c.shape[0] < max_h:
                pad = np.zeros((max_h - c.shape[0], c.shape[1], 3), dtype=np.uint8)
                c = np.concatenate([c, pad], axis=0)
            padded.append(c)
        return np.concatenate(padded, axis=1)

    # Row 1: Mouth crops
    r1_cols = [add_label(gt_crop, "GT")]
    for step in steps:
        r1_cols.append(add_label(crop_mouth(all_frames[step][display_frame], bbox), f"Step {step}"))

    # Row 2: Error vs GT
    r2_cols = [add_label(gt_crop, "GT")]
    for step in steps:
        diff = amplified_abs_diff(crop_mouth(all_frames[step][display_frame], bbox), gt_crop, amp=5)
        r2_cols.append(add_label(diff, f"|s{step}-GT| x5"))

    # Row 3: Consecutive change
    r3_cols = [add_label(np.zeros_like(gt_crop), "Δ consec.")]
    for i, step in enumerate(steps):
        if i == 0:
            r3_cols.append(add_label(crop_mouth(all_frames[step][display_frame], bbox), f"x0@s{step}"))
        else:
            diff = amplified_abs_diff(
                crop_mouth(all_frames[step][display_frame], bbox),
                crop_mouth(all_frames[steps[i-1]][display_frame], bbox),
                amp=8,
            )
            r3_cols.append(add_label(diff, f"|s{step}-s{steps[i-1]}| x8"))

    # Row 4: Improvement map
    r4_cols = [add_label(np.zeros_like(gt_crop), "Improve")]
    for i, step in enumerate(steps):
        if i == 0:
            prev_crop = np.full_like(gt_crop, 128)
        else:
            prev_crop = crop_mouth(all_frames[steps[i-1]][display_frame], bbox)
        curr_crop = crop_mouth(all_frames[step][display_frame], bbox)
        imp = improvement_map(prev_crop, curr_crop, gt_crop)
        r4_cols.append(add_label(imp, f"s{steps[i-1] if i>0 else '?'}→{step}"))

    rows = [make_row(r) for r in [r1_cols, r2_cols, r3_cols, r4_cols]]
    # Pad rows to same width
    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded_rows.append(r)

    grid = np.concatenate(padded_rows, axis=0)
    Image.fromarray(grid).save(output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# All-steps consecutive difference strip (for identifying transition)
# ─────────────────────────────────────────────────────────────────────────────

def vis_all_steps_strip(all_frames, gt_frames, bbox, display_frame, output_path,
                        num_steps=50, amp=8):
    """Horizontal strip: |x0[i] - x0[i-1]| for ALL consecutive steps (mouth crop).
    Plus a second row: improvement map for each step vs GT.
    """
    gt_crop = crop_mouth(gt_frames[display_frame], bbox)

    change_cols = []
    improve_cols = []

    for step in range(num_steps):
        if step == 0:
            curr_crop = crop_mouth(all_frames[0][display_frame], bbox)
            # Resize to smaller thumbnails for the strip
            th, tw = 64, 80  # thumbnail size
            curr_small = cv2.resize(curr_crop, (tw, th))
            change_cols.append(add_label(curr_small, f"s0", font_size=9))
            improve_cols.append(add_label(np.zeros((th, tw, 3), dtype=np.uint8), f"s0", font_size=9))
        else:
            curr = crop_mouth(all_frames[step][display_frame], bbox)
            prev = crop_mouth(all_frames[step-1][display_frame], bbox)

            diff = amplified_abs_diff(curr, prev, amp=amp)
            diff_small = cv2.resize(diff, (80, 64))
            change_cols.append(add_label(diff_small, f"s{step}", font_size=9))

            imp = improvement_map(prev, curr, gt_crop)
            imp_small = cv2.resize(imp, (80, 64))
            improve_cols.append(add_label(imp_small, f"s{step}", font_size=9))

    change_row = np.concatenate(change_cols, axis=1)
    improve_row = np.concatenate(improve_cols, axis=1)
    grid = np.concatenate([change_row, improve_row], axis=0)
    Image.fromarray(grid).save(output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str, required=True,
                        help="Directory with decoded videos (from eval_ode_perceptual_v2.py)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--samples", type=str, required=True,
                        help="Comma-separated sample names")
    parser.add_argument("--key_steps", type=str, default="0,5,10,15,20,25,30,35,40,45,49",
                        help="Key steps for the combined grid")
    parser.add_argument("--display_frames", type=str, default="10,40,60",
                        help="Video frame indices to visualize")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mask_keep = load_pixel_mask(args.mask_path)
    bbox = get_mouth_bbox(mask_keep, pad=16)
    print(f"Mouth bbox: y=[{bbox[0]}, {bbox[1]}), x=[{bbox[2]}, {bbox[3]})")

    key_steps = [int(s) for s in args.key_steps.split(",")]
    display_frames = [int(f) for f in args.display_frames.split(",")]
    sample_names = args.samples.split(",")

    # Find how many steps we have
    test_dir = os.path.join(args.videos_dir, sample_names[0])
    num_steps = len([f for f in os.listdir(test_dir)
                     if f.startswith("step_") and f.endswith(".mp4") and "_audio" not in f])
    print(f"Found {num_steps} steps")

    for sample_name in sample_names:
        print(f"\n=== {sample_name} ===")
        sample_vid_dir = os.path.join(args.videos_dir, sample_name)
        sample_out = os.path.join(args.output_dir, sample_name)
        os.makedirs(sample_out, exist_ok=True)

        # Load GT
        print("  Loading GT...")
        gt_frames = read_video_frames(os.path.join(sample_vid_dir, "gt.mp4"))

        # Load all steps
        print(f"  Loading {num_steps} step videos...")
        all_frames = {}
        for step in range(num_steps):
            path = os.path.join(sample_vid_dir, f"step_{step:03d}.mp4")
            if os.path.exists(path):
                all_frames[step] = read_video_frames(path)

        for fi in display_frames:
            fi_safe = min(fi, len(gt_frames) - 1)
            print(f"  Frame {fi_safe}:")

            # Combined 4-row grid at key steps
            p = vis_combined(all_frames, gt_frames, bbox, fi_safe, key_steps,
                             os.path.join(sample_out, f"combined_frame{fi:03d}.png"))
            print(f"    {p}")

            # All-steps strip (50 thumbnails)
            p = vis_all_steps_strip(all_frames, gt_frames, bbox, fi_safe,
                                    os.path.join(sample_out, f"all_steps_frame{fi:03d}.png"),
                                    num_steps=num_steps)
            print(f"    {p}")

        # Also generate individual visualizations at key frame
        mid_frame = min(40, len(gt_frames) - 1)
        vis_mouth_crops(all_frames, gt_frames, bbox, mid_frame, key_steps,
                        os.path.join(sample_out, "mouth_crops.png"))
        vis_error_vs_gt(all_frames, gt_frames, bbox, mid_frame, key_steps,
                        os.path.join(sample_out, "error_vs_gt.png"))
        vis_consecutive_changes(all_frames, gt_frames, bbox, mid_frame, key_steps,
                                os.path.join(sample_out, "consecutive_changes.png"))
        vis_improvement_map(all_frames, gt_frames, bbox, mid_frame, key_steps,
                            os.path.join(sample_out, "improvement_map.png"))
        print(f"  Individual vis saved.")

    print(f"\nDone! Output: {args.output_dir}")


if __name__ == "__main__":
    main()
