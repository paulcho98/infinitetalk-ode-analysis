#!/usr/bin/env python
"""
Generate FULL ODE trajectories from the InfiniteTalk teacher (Wan2.1-I2V-14B + audio cross-attn).

Path-A port of scripts/generate_omniavatar_ode_pairs_full.py: forks InfiniteTalk's own
flow-matching Euler denoising loop (wan/multitalk.py::generate_infinitetalk) and, for each of
the `num_steps` iterations, saves the noisy state x_t and the derived denoised prediction x0:

  step_{i:03d}_xt.pt   noisy latent x_t at (shifted) timestep t_i   [16, 21, lat_h, lat_w] bf16
  step_{i:03d}_x0.pt   denoised prediction x0_pred                  [16, 21, lat_h, lat_w] bf16
  ode_schedule.json    timesteps, shift, guidance scales, latent shape, seed
  input_latents.pt     VAE-encoded reference-frame latent (I2V anchor)

InfiniteTalk has NO x0 head: the DiT predicts a velocity v; the loop stores `noise_pred = -v`
(multitalk.py:758) and integrates `latent += noise_pred*dt`. With x_t = (1-sigma)*x0 + sigma*noise
and sigma = timesteps[i]/num_timesteps, the denoised prediction is:

    x0_pred = x_t - sigma * v = x_t + sigma * noise_pred    (noise_pred already negated)

CFG (3-call by default): noise_pred = uncond + text_s*(cond-drop_text) + audio_s*(drop_text-uncond).
Special cases mirror InfiniteTalk exactly: text_s==1 -> 2-call drop_audio formula; (1,1) -> cond only.

One (text_guide_scale, audio_guide_scale) pair per invocation (like the OmniAvatar driver).
Samples are sharded across processes via --shard_id/--num_shards.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

INFINITETALK_ROOT = "/home/work/.local/InfiniteTalk"
sys.path.insert(0, INFINITETALK_ROOT)

import wan  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.multitalk import (  # noqa: E402
    InfiniteTalkPipeline,
    timestep_transform,
    resize_and_centercrop,
)
from wan.utils.utils import extract_specific_frames  # noqa: E402
from wan.utils.multitalk_utils import ASPECT_RATIO_627, ASPECT_RATIO_960  # noqa: E402
# audio preprocessing helpers (same ones generate_infinitetalk.py uses)
from generate_infinitetalk import custom_init, get_embedding, audio_prepare_single  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# ODE trajectory extractor (subclass reuses all of InfiniteTalkPipeline's loading)
# ─────────────────────────────────────────────────────────────────────────────
class ODEInfiniteTalkPipeline(InfiniteTalkPipeline):

    @torch.no_grad()
    def extract_ode_trajectory(
        self,
        input_clip: dict,
        out_dir: str,
        *,
        text_guide_scale: float,
        audio_guide_scale: float,
        size_buckget: str = "infinitetalk-480",
        frame_num: int = 81,
        sampling_steps: int = 50,
        shift: float = 7.0,
        seed: int = 42,
    ):
        os.makedirs(out_dir, exist_ok=True)
        self.model.disable_teacache()  # never cache across CFG passes for a clean trajectory

        # ── setup: reference image (frame 0) + aspect-ratio bucket ──
        input_prompt = input_clip["prompt"]
        cond_file_path = input_clip["cond_video"]
        cond_image = extract_specific_frames(cond_file_path, 0)

        # FORCE SQUARE bucket (per requirement: same square aspect for all samples).
        # InfiniteTalk normally picks the aspect-ratio bucket closest to the reference image,
        # which yields NON-square latents for non-square references (e.g. 2:3 -> [16,21,64,96]).
        # Center-crop every reference to the square 480p (640x640) / 720p (960x960) bucket so all
        # latents are uniformly [16, 21, 80, 80].
        target_h = target_w = 640 if size_buckget == "infinitetalk-480" else 960
        cond_image = resize_and_centercrop(cond_image, (target_h, target_w))
        cond_image = cond_image / 255
        cond_image = (cond_image - 0.5) * 2  # [-1, 1], shape [1, C, 1, H, W]
        cond_image = cond_image.to(self.device)

        # ── audio embedding (precomputed wav2vec, [N, 12, 768]) ──
        full_audio_emb = torch.load(input_clip["cond_audio"]["person1"])
        assert not torch.isnan(full_audio_emb).any(), "audio emb has NaNs"
        assert full_audio_emb.shape[0] > frame_num, (
            f"audio emb len {full_audio_emb.shape[0]} <= frame_num {frame_num}")
        full_audio_embs = [full_audio_emb]

        # ── text ──
        n_prompt = self.sample_neg_prompt
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context, context_null = self.text_encoder([input_prompt, n_prompt], self.device)
        else:
            context = [t.to(self.device) for t in self.text_encoder([input_prompt], torch.device("cpu"))]
            context_null = [t.to(self.device) for t in self.text_encoder([n_prompt], torch.device("cpu"))]

        # ── seeding (match InfiniteTalk) ──
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # single clip, single human, first clip (is_first_clip=True)
        indices = (torch.arange(2 * 2 + 1) - 2) * 1  # 5-frame window [-2..2]
        clip_length = frame_num
        cur_motion_frames_num = 1
        audio_start_idx, audio_end_idx = 0, clip_length

        # audio window -> [1, frame_num, 5, 12, 768]
        center_indices = torch.arange(audio_start_idx, audio_end_idx).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_embs[0].shape[0] - 1)
        audio_embs = full_audio_embs[0][center_indices][None, ...].to(self.device).to(self.param_dtype)

        # latent geometry
        h, w = cond_image.shape[-2], cond_image.shape[-1]
        lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
        max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        noise = torch.randn(16, (frame_num - 1) // 4 + 1, lat_h, lat_w,
                            dtype=torch.float32, device=self.device)

        # temporal mask (first frame conditioned) -> [1, 4, T_lat, lat_h, lat_w]
        msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2).to(self.param_dtype)

        # CLIP + VAE-encoded conditioning y = [msk(4) | vae_ref(16)]
        self.clip.model.to(self.device)
        clip_context = self.clip.visual(cond_image[:, :, -1:, :, :]).to(self.param_dtype)
        video_frames = torch.zeros(1, cond_image.shape[1], frame_num - cond_image.shape[2],
                                   target_h, target_w, device=self.device)
        padding = torch.concat([cond_image, video_frames], dim=2)
        y = self.vae.encode(padding)
        y = torch.stack(y).to(self.param_dtype)
        latent_motion_frames = self.vae.encode(cond_image)[0]  # I2V anchor latent [16, 1, lat_h, lat_w]
        cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num - 1) // 4)
        y = torch.concat([msk, y], dim=1)  # [1, 20, T_lat, lat_h, lat_w]

        # single-human ref_target_masks = ones over (num_classes=3, lat_h, lat_w)
        ref_target_masks = torch.ones(3, lat_h, lat_w, device=self.device).float()

        # ── timesteps (shifted flow-matching) ──
        timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
        timesteps.append(0.0)
        timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps)
                         for t in timesteps]

        latent = noise
        common = dict(clip_fea=clip_context, seq_len=max_seq_len, y=y, ref_target_masks=ref_target_masks)
        arg_c = dict(context=[context], audio=audio_embs, **common)
        arg_null_text = dict(context=[context_null], audio=audio_embs, **common)
        arg_null_audio = dict(context=[context], audio=torch.zeros_like(audio_embs)[-1:], **common)
        arg_null = dict(context=[context_null], audio=torch.zeros_like(audio_embs)[-1:], **common)

        self.model.to(self.device)

        # persist schedule + reference latent
        with open(os.path.join(out_dir, "ode_schedule.json"), "w") as f:
            json.dump({
                "model": "infinitetalk-14B",
                "num_steps": sampling_steps,
                "shift": shift,
                "text_guide_scale": text_guide_scale,
                "audio_guide_scale": audio_guide_scale,
                "num_timesteps": self.num_timesteps,
                "size_buckget": size_buckget,
                "target_hw": [int(target_h), int(target_w)],
                "latent_shape": list(noise.shape),
                "seed": seed,
                "t_list": [float(t.item()) for t in timesteps],
            }, f, indent=2)
        torch.save(latent_motion_frames.detach().to(torch.bfloat16).cpu(),
                   os.path.join(out_dir, "input_latents.pt"))

        no_cfg = math.isclose(text_guide_scale, 1.0) and math.isclose(audio_guide_scale, 1.0)

        for i in range(len(timesteps) - 1):
            timestep = timesteps[i]
            latent[:, :cur_motion_frames_latent_num] = latent_motion_frames  # pin I2V anchor
            # SAVE x_t (state fed to the model this step)
            torch.save(latent.detach().to(torch.bfloat16).cpu(),
                       os.path.join(out_dir, f"step_{i:03d}_xt.pt"))
            latent_model_input = [latent.to(self.device)]

            # ── CFG (faithful to InfiniteTalk branch logic) ──
            if no_cfg:
                noise_pred = self.model(latent_model_input, t=timestep, **arg_c)[0]
            elif math.isclose(text_guide_scale, 1.0):
                noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_drop_audio = self.model(latent_model_input, t=timestep, **arg_null_audio)[0]
                noise_pred = noise_pred_drop_audio + audio_guide_scale * (
                    noise_pred_cond - noise_pred_drop_audio)
            else:
                noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_drop_text = self.model(latent_model_input, t=timestep, **arg_null_text)[0]
                noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + text_guide_scale * (
                    noise_pred_cond - noise_pred_drop_text) + audio_guide_scale * (
                    noise_pred_drop_text - noise_pred_uncond)
            noise_pred = -noise_pred  # velocity sign flip (multitalk.py:758)

            # SAVE x0_pred = x_t + sigma * noise_pred   (sigma = timestep / num_timesteps)
            sigma = (timestep / self.num_timesteps).view(-1, 1, 1, 1)
            x0_pred = latent + sigma * noise_pred
            torch.save(x0_pred.detach().to(torch.bfloat16).cpu(),
                       os.path.join(out_dir, f"step_{i:03d}_x0.pt"))

            # Euler update -> x_{t-1}
            dt = (timesteps[i] - timesteps[i + 1]) / self.num_timesteps
            latent = latent + noise_pred * dt.view(-1, 1, 1, 1)
            latent[:, :cur_motion_frames_latent_num] = latent_motion_frames  # re-pin

        return {"num_steps": sampling_steps, "latent_shape": list(noise.shape)}


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True, help="Wan2.1-I2V-14B-480P dir")
    p.add_argument("--infinitetalk_dir", required=True, help="single/infinitetalk.safetensors")
    p.add_argument("--wav2vec_dir", required=True, help="chinese-wav2vec2-base dir")
    p.add_argument("--video_dir", required=True, help="dir of <hash>.mp4 reference videos")
    p.add_argument("--audio_dir", required=True, help="dir of <hash>.wav audio")
    p.add_argument("--sample_names_file", required=True, help="one <hash>_shot_xxx name per line")
    p.add_argument("--output_root", required=True, help="root; per-config subdir infinitetalk_t{T}_a{A}")
    p.add_argument("--configs", required=True,
                   help="comma list of text:audio guidance pairs, e.g. '5:4,5:1,1:1,2.5:2,7.5:6'")
    p.add_argument("--audio_cache_dir", default=None, help="where to cache wav2vec embs (shared)")
    p.add_argument("--prompt", default="A person is talking.")
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


def main():
    args = parse_args()
    device = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    configs = []
    for tok in args.configs.split(","):
        t_s, a_s = tok.split(":")
        configs.append((float(t_s), float(a_s)))

    with open(args.sample_names_file) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    names = names[: args.max_samples]
    names = names[args.shard_id :: args.num_shards]  # shard by SAMPLE (audio cached once per hash)
    if not names:
        print(f"[shard {args.shard_id}] no samples")
        return

    audio_cache = args.audio_cache_dir or os.path.join(args.output_root, "_audio_cache")
    os.makedirs(audio_cache, exist_ok=True)

    print(f"[shard {args.shard_id}/{args.num_shards}] cuda:{device} "
          f"samples={len(names)} configs={len(configs)} building pipeline ...", flush=True)
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

    for name in names:
        h = name.split("_shot")[0]
        vid = os.path.join(args.video_dir, f"{h}.mp4")
        wav = os.path.join(args.audio_dir, f"{h}.wav")
        if not os.path.exists(vid) or not os.path.exists(wav):
            print(f"[MISSING] {name}: vid={os.path.exists(vid)} wav={os.path.exists(wav)}", flush=True)
            continue

        # precompute (or reuse cached) audio embedding once per sample
        emb_path = os.path.join(audio_cache, f"{h}.pt")
        if not os.path.exists(emb_path):
            speech = audio_prepare_single(wav)
            emb = get_embedding(speech, wav2vec_feat, audio_encoder)
            torch.save(emb, emb_path)
        input_clip = {"prompt": args.prompt, "cond_video": vid, "cond_audio": {"person1": emb_path}}

        for (T, A) in configs:
            out_dir = os.path.join(args.output_root, f"infinitetalk_t{T}_a{A}", name)
            last = os.path.join(out_dir, f"step_{args.num_inference_steps - 1:03d}_x0.pt")
            if args.skip_existing and os.path.exists(last):
                print(f"[skip] {name} t{T}/a{A}", flush=True)
                continue
            t0 = time.time()
            try:
                pipe.extract_ode_trajectory(
                    input_clip, out_dir,
                    text_guide_scale=T, audio_guide_scale=A,
                    size_buckget=args.size, frame_num=args.frame_num,
                    sampling_steps=args.num_inference_steps, shift=args.shift, seed=args.seed,
                )
                print(f"[done] {name} t{T}/a{A} {time.time() - t0:.1f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback
                print(f"[FAIL] {name} t{T}/a{A}: {e}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
