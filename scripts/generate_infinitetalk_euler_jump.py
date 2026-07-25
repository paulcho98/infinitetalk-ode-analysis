#!/usr/bin/env python
"""Euler-jump (ODE-straightness) probe for InfiniteTalk — port of the OmniAvatar experiment.

Port of /home/work/.local/OmniAvatar/scripts/generate_single_step_predictions.py --mode euler_jump.

THE QUESTION: how straight is the ODE path? Instead of denoising sequentially, take the step-0
prediction, extrapolate along a SINGLE Euler jump to every landing noise level, and ask the
teacher to re-predict x0 there. If the trajectory were perfectly straight, the jumped x0 would
match the sequential one at every step. The gap measures curvature.

    sigma      = t_i / num_timesteps
    eps_euler  = (x_t_0 - (1 - sigma_0) * x0_0) / sigma_0     # noise implied by step 0
    x_t        = (1 - sigma) * x0_0 + sigma * eps_euler        # the jump
    x0_pred    = x_t + sigma * noise_pred(x_t; teacher CFG)    # teacher re-prediction

TWO INDEPENDENT CFG SETTINGS (this is the factorial):
  step-0 CFG   — the guidance used for the prediction we jump FROM. Supplied implicitly by
                 --step0_traj_dir: we read that trajectory's saved step_000_{xt,x0}.pt, so no
                 recomputation is needed for any config we already swept.
  teacher CFG  — the guidance used to re-predict at each landing step (--text_cfg_teacher /
                 --audio_cfg_teacher).

Unlike the OmniAvatar original — which hardcoded `if args.cfg_step0 == 4.5` to decide whether the
saved x0 could be reused, silently loading the wrong tensor if --traj_dir pointed at a different
scale — this reads the source trajectory's OWN scales from its ode_schedule.json and verifies them.

OUTPUT is a trajectory-shaped dir, so the whole existing Stage-2 stack runs on it UNCHANGED:
    step_{i:03d}_xt.pt / step_{i:03d}_x0.pt / ode_schedule.json / input_latents.pt

    Stage 2a:  eval_ode_perceptual_v2_infinitetalk.py --phase decode|metrics --traj_dir <out>
    Stage 2b:  analyze_ode_trajectory_infinitetalk.py --traj_dir <out>

Usage:
    python scripts/generate_infinitetalk_euler_jump.py \
        --checkpoint_dir ... --infinitetalk_dir ... --wav2vec_dir ... \
        --video_dir ... --audio_dir ... --sample_names_file ... \
        --step0_traj_dir <traj_root>/infinitetalk_t5.0_a4.0 \
        --text_cfg_teacher 5.0 --audio_cfg_teacher 1.0 \
        --output_dir <out_root>/euler_on_noaudio
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# Match the trajectory driver: eager, no dynamo (see generate_infinitetalk_ode_pairs_full.py).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_infinitetalk_ode_pairs_full import (  # noqa: E402
    ODEInfiniteTalkPipeline,
    WAN_CONFIGS,
    custom_init,
    get_embedding,
    audio_prepare_single,
)


def load_schedule(traj_dir: str) -> dict:
    p = os.path.join(traj_dir, "ode_schedule.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"source trajectory has no ode_schedule.json: {p}")
    with open(p) as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    # model / data (same as the trajectory driver — needed to rebuild conditioning)
    p.add_argument("--checkpoint_dir", required=True, help="Wan2.1-I2V-14B-480P dir")
    p.add_argument("--infinitetalk_dir", required=True, help="single/infinitetalk.safetensors")
    p.add_argument("--wav2vec_dir", required=True, help="chinese-wav2vec2-base dir")
    p.add_argument("--video_dir", required=True, help="dir of <hash>.mp4 reference videos")
    p.add_argument("--audio_dir", required=True, help="dir of <hash>.wav audio")
    p.add_argument("--sample_names_file", required=True, help="one <hash>_shot_xxx name per line")
    p.add_argument("--audio_cache_dir", default=None, help="shared wav2vec cache (read-only here)")
    p.add_argument("--prompt", default="A person is talking.")

    # the factorial
    p.add_argument("--step0_traj_dir", required=True,
                   help="trajectory dir supplying step_000_{xt,x0}.pt — ITS guidance scales are "
                        "the step-0 leg of the factorial")
    p.add_argument("--text_cfg_teacher", type=float, required=True,
                   help="text guidance for the teacher re-prediction at each landing step")
    p.add_argument("--audio_cfg_teacher", type=float, required=True,
                   help="audio guidance for the teacher re-prediction at each landing step")
    p.add_argument("--output_dir", required=True, help="trajectory-shaped output dir for this cell")

    # schedule — must match the source trajectory (verified below)
    p.add_argument("--size", default="infinitetalk-480")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--shift", type=float, default=7.0)
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=10)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


@torch.no_grad()
def euler_jump_sample(pipe, input_clip, out_dir, *, step0_traj_sample_dir, sched,
                      text_cfg_teacher, audio_cfg_teacher, args):
    """Run the full Euler-jump sweep for ONE sample; write a trajectory-shaped dir."""
    os.makedirs(out_dir, exist_ok=True)

    setup = pipe.prepare_conditioning(
        input_clip, size_buckget=args.size, frame_num=args.frame_num,
        sampling_steps=args.num_inference_steps, shift=args.shift, seed=args.seed,
    )
    timesteps = setup["timesteps"]
    latent_motion_frames = setup["latent_motion_frames"]
    anchor_n = setup["cur_motion_frames_latent_num"]
    target_h, target_w = setup["target_hw"]
    num_timesteps = pipe.num_timesteps

    # ── step-0 state we jump FROM (saved by the source trajectory run) ──
    x_t_0 = torch.load(os.path.join(step0_traj_sample_dir, "step_000_xt.pt"),
                       map_location="cpu", weights_only=True).float().to(pipe.device)
    x0_0 = torch.load(os.path.join(step0_traj_sample_dir, "step_000_x0.pt"),
                      map_location="cpu", weights_only=True).float().to(pipe.device)

    # Recover the noise direction implied by step 0, then jump along it.
    sigma_0 = float(timesteps[0].item()) / num_timesteps
    eps_euler = (x_t_0 - (1.0 - sigma_0) * x0_0) / sigma_0

    # persist schedule (records BOTH legs of the factorial)
    with open(os.path.join(out_dir, "ode_schedule.json"), "w") as f:
        json.dump({
            "model": "infinitetalk-14B",
            "mode": "euler_jump",
            "num_steps": args.num_inference_steps,
            "shift": args.shift,
            "step0_traj_dir": args.step0_traj_dir,
            "text_guide_scale_step0": sched.get("text_guide_scale"),
            "audio_guide_scale_step0": sched.get("audio_guide_scale"),
            "text_guide_scale": text_cfg_teacher,     # teacher leg, under the key Stage-2 reads
            "audio_guide_scale": audio_cfg_teacher,
            "num_timesteps": num_timesteps,
            "size_buckget": args.size,
            "target_hw": [int(target_h), int(target_w)],
            "latent_shape": list(setup["noise"].shape),
            "seed": args.seed,
            "sigma_0": sigma_0,
            "t_list": [float(t.item()) for t in timesteps],
        }, f, indent=2)
    torch.save(latent_motion_frames.detach().to(torch.bfloat16).cpu(),
               os.path.join(out_dir, "input_latents.pt"))

    for i in range(len(timesteps) - 1):
        x0_path = os.path.join(out_dir, f"step_{i:03d}_x0.pt")
        if args.skip_existing and os.path.exists(x0_path):
            continue
        timestep = timesteps[i]
        sigma = (timestep / num_timesteps).view(-1, 1, 1, 1)

        # ── the Euler jump: land directly at this noise level from step 0 ──
        x_t = (1.0 - sigma) * x0_0 + sigma * eps_euler
        x_t[:, :anchor_n] = latent_motion_frames  # pin I2V anchor, as the sequential loop does
        torch.save(x_t.detach().to(torch.bfloat16).cpu(),
                   os.path.join(out_dir, f"step_{i:03d}_xt.pt"))

        # ── teacher re-prediction at the jumped state ──
        noise_pred = pipe.predict_noise(x_t, timestep, setup, text_cfg_teacher, audio_cfg_teacher)
        x0_pred = x_t + sigma * noise_pred
        torch.save(x0_pred.detach().to(torch.bfloat16).cpu(), x0_path)

        del noise_pred, x0_pred, x_t
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    device = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    # ── verify the source trajectory's schedule matches ours; a mismatch makes the jump
    #    geometrically meaningless (different t_list / seed / latent shape). ──
    sched = load_schedule(args.step0_traj_dir)
    mismatches = []
    if int(sched.get("num_steps", -1)) != args.num_inference_steps:
        mismatches.append(f"num_steps {sched.get('num_steps')} != {args.num_inference_steps}")
    if not math.isclose(float(sched.get("shift", -1)), args.shift):
        mismatches.append(f"shift {sched.get('shift')} != {args.shift}")
    if int(sched.get("seed", -1)) != args.seed:
        mismatches.append(f"seed {sched.get('seed')} != {args.seed}")
    if sched.get("size_buckget") != args.size:
        mismatches.append(f"size {sched.get('size_buckget')} != {args.size}")
    if mismatches:
        raise SystemExit(
            "[FATAL] --step0_traj_dir schedule disagrees with this run:\n  "
            + "\n  ".join(mismatches)
            + "\nThe Euler jump is only meaningful on an identical schedule."
        )

    s_t, s_a = sched.get("text_guide_scale"), sched.get("audio_guide_scale")
    print(f"[euler_jump] step0 CFG (from trajectory): text={s_t} audio={s_a}")
    print(f"[euler_jump] teacher CFG:                 text={args.text_cfg_teacher} "
          f"audio={args.audio_cfg_teacher}")

    with open(args.sample_names_file) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    names = names[: args.max_samples]
    names = names[args.shard_id :: args.num_shards]
    if not names:
        print(f"[shard {args.shard_id}] no samples")
        return

    print(f"[shard {args.shard_id}/{args.num_shards}] cuda:{device} samples={len(names)} "
          f"building pipeline ...", flush=True)
    cfg = WAN_CONFIGS["infinitetalk-14B"]
    pipe = ODEInfiniteTalkPipeline(
        config=cfg,
        checkpoint_dir=args.checkpoint_dir,
        device_id=device,
        rank=0,
        t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False,
        infinitetalk_dir=args.infinitetalk_dir,
    )
    wav2vec_feat, audio_encoder = custom_init("cpu", args.wav2vec_dir)
    audio_cache = args.audio_cache_dir or os.path.join(
        os.path.dirname(args.step0_traj_dir.rstrip("/")), "_audio_cache")
    os.makedirs(audio_cache, exist_ok=True)

    for name in names:
        h = name.split("_shot")[0]
        vid = os.path.join(args.video_dir, f"{h}.mp4")
        wav = os.path.join(args.audio_dir, f"{h}.wav")
        step0_dir = os.path.join(args.step0_traj_dir, name)
        if not os.path.exists(os.path.join(step0_dir, "step_000_x0.pt")):
            print(f"[MISSING] {name}: no step_000_x0.pt under {step0_dir}", flush=True)
            continue
        if not os.path.exists(vid) or not os.path.exists(wav):
            print(f"[MISSING] {name}: vid={os.path.exists(vid)} wav={os.path.exists(wav)}", flush=True)
            continue

        out_dir = os.path.join(args.output_dir, name)
        last = os.path.join(out_dir, f"step_{args.num_inference_steps - 1:03d}_x0.pt")
        if args.skip_existing and os.path.exists(last):
            print(f"[skip] {name}", flush=True)
            continue

        emb_path = os.path.join(audio_cache, f"{h}.pt")
        if not os.path.exists(emb_path):
            speech = audio_prepare_single(wav)
            torch.save(get_embedding(speech, wav2vec_feat, audio_encoder), emb_path)
        input_clip = {"prompt": args.prompt, "cond_video": vid,
                      "cond_audio": {"person1": emb_path}}

        t0 = time.time()
        try:
            euler_jump_sample(
                pipe, input_clip, out_dir,
                step0_traj_sample_dir=step0_dir, sched=sched,
                text_cfg_teacher=args.text_cfg_teacher,
                audio_cfg_teacher=args.audio_cfg_teacher,
                args=args,
            )
            print(f"[done] {name} {time.time() - t0:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[FAIL] {name}: {e}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
