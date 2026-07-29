#!/usr/bin/env python
"""THE straightness number: ‖x0_euler(t) − x0_sequential(t)‖ per ODE step.

The Euler-jump experiment asks how straight the ODE path is. The direct measurement is the gap
between the jumped prediction and the sequential one AT THE SAME STEP — if the trajectory were
perfectly straight, a single jump from step 0 would reproduce the sequential result and this gap
would be zero everywhere.

Neither Stage 2a nor Stage 2b computes this: 2a compares against GT *pixels*, 2b against the GT
*latent*. Both are "how good is it", not "how far did the jump miss". This script fills that gap.

CHEAP BY DESIGN — it only reads the saved x0 tensors. No VAE, no GT video, no model, no dlib
(mouth masks are read from the Stage-2b cache if present; otherwise it reports the full region
only). Seconds per cell, so it does not depend on the Stage-2b re-run.

Reports, per step, for each region:
    abs_l2   ‖x0_euler − x0_seq‖
    mse      mean((x0_euler − x0_seq)^2)
    rel_l2   ‖x0_euler − x0_seq‖ / ‖x0_seq‖      <- scale-free; the headline curvature number

Usage:
    python scripts/common/measure_euler_straightness.py \
        --euler_dir      ode_euler_jump_infinitetalk/euler_on_on \
        --sequential_dir ode_full_trajectories_infinitetalk/infinitetalk_t5.0_a4.0 \
        --mouth_mask_cache ode_analysis_infinitetalk/_mouth_mask_cache \
        --output_dir results/infinitetalk/data --tag on_on
"""
import argparse
import json
import os

import numpy as np
import torch


def discover_samples(d):
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d)
                  if os.path.isfile(os.path.join(d, n, "step_000_x0.pt")))


def num_steps_of(d, sample):
    n = 0
    while os.path.isfile(os.path.join(d, sample, f"step_{n:03d}_x0.pt")):
        n += 1
    return n


def region_values(t, mask, region):
    """t: [C,T,H,W]; mask: [H,W] bool. region 'full' -> all, 'mouth' -> masked."""
    if region == "full" or mask is None:
        return t.reshape(-1)
    m = mask.unsqueeze(0).unsqueeze(0).expand_as(t)
    return t[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euler_dir", required=True, help="one euler_{step0}_{teacher} cell")
    ap.add_argument("--sequential_dir", required=True,
                    help="the sequential trajectory to compare against — normally the config whose "
                         "CFG matches this cell's TEACHER leg")
    ap.add_argument("--mouth_mask_cache", default=None,
                    help="Stage-2b mask cache (<sample>.npy). Without it, only 'full' is reported.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tag", required=True, help="short label, e.g. on_on")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    es = set(discover_samples(args.euler_dir))
    ss = set(discover_samples(args.sequential_dir))
    samples = sorted(es & ss)
    if not samples:
        raise SystemExit(f"[FATAL] no overlapping samples.\n  euler={len(es)} seq={len(ss)}")
    if es - ss or ss - es:
        print(f"[warn] using {len(samples)} common samples "
              f"(euler-only={len(es - ss)}, seq-only={len(ss - es)})")

    n_steps = min(num_steps_of(args.euler_dir, samples[0]),
                  num_steps_of(args.sequential_dir, samples[0]))
    print(f"[{args.tag}] {len(samples)} samples x {n_steps} steps")

    masks = {}
    if args.mouth_mask_cache:
        for s in samples:
            p = os.path.join(args.mouth_mask_cache, f"{s}.npy")
            if os.path.isfile(p):
                masks[s] = torch.from_numpy(np.load(p)).bool()
    regions = ["full"] + (["mouth"] if len(masks) == len(samples) else [])
    if "mouth" not in regions:
        print(f"[warn] mouth masks incomplete ({len(masks)}/{len(samples)}) — full region only")

    acc = {f"{r}_{m}": np.zeros(n_steps) for r in regions for m in ("abs_l2", "mse", "rel_l2")}

    for s in samples:
        mask = masks.get(s)
        for i in range(n_steps):
            xe = torch.load(os.path.join(args.euler_dir, s, f"step_{i:03d}_x0.pt"),
                            map_location="cpu", weights_only=True).float()
            xq = torch.load(os.path.join(args.sequential_dir, s, f"step_{i:03d}_x0.pt"),
                            map_location="cpu", weights_only=True).float()
            for r in regions:
                ve, vq = region_values(xe, mask, r), region_values(xq, mask, r)
                d = ve - vq
                acc[f"{r}_abs_l2"][i] += d.norm().item()
                acc[f"{r}_mse"][i] += d.pow(2).mean().item()
                nq = vq.norm().item()
                acc[f"{r}_rel_l2"][i] += (d.norm().item() / nq) if nq > 1e-8 else 0.0

    for k in acc:
        acc[k] /= len(samples)

    out = {
        "tag": args.tag,
        "euler_dir": args.euler_dir,
        "sequential_dir": args.sequential_dir,
        "samples": samples,
        "num_steps": n_steps,
        "regions": regions,
        **{k: v.tolist() for k, v in acc.items()},
    }
    jp = os.path.join(args.output_dir, f"straightness_{args.tag}.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", jp)

    for r in regions:
        rel = acc[f"{r}_rel_l2"]
        print(f"  [{r}] rel_l2  step0={rel[0]:.4f}  mid={rel[n_steps // 2]:.4f}  "
              f"final={rel[-1]:.4f}  max={rel.max():.4f} @ step {int(rel.argmax())}")


if __name__ == "__main__":
    main()
