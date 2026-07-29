# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base-weight load test for the vendored OmniAvatarWan model (Task 18, reduced scope).

REDUCED SCOPE: the teacher checkpoint (`$TEACHER_CKPT`, step-10500.pt) no longer exists on
this box (backed up to HuggingFace and cleaned locally), so this test does NOT load the
OmniAvatar LoRA+audio checkpoint overlay and does NOT run a generation smoke. It only verifies
that the vendored architecture (models/omniavatar_wan/) can construct and load the base
Wan2.1-T2V-14B weights. See docs/status-and-todo.md "Deferred re-runs" for the checkpoint-
restoration follow-up.

`scripts/omniavatar/generate_omniavatar_ode_pairs_full.py` instantiates the model as:

    teacher = OmniAvatarWan(
        model_size=args.model_size, in_dim=args.in_dim, mode="v2v", use_audio=True,
        base_model_paths=args.base_model_paths, omniavatar_ckpt_path=args.omniavatar_ckpt_path,
        merge_lora=True, net_pred_type="flow", schedule_type="rf",
    ).to(device, dtype=dtype).eval()

`OmniAvatarWan._load_weights()` (models/omniavatar_wan/network.py) shows `omniavatar_ckpt_path`
is optional -- Stage 1 (base weights) and Stage 2 (checkpoint overlay) are independent, and
Stage 2 is skipped entirely when the path is None. So the base-weight part CAN be exercised
standalone; but the class API only *logs* missing/unexpected key counts from
`_smart_load_weights()`, it doesn't return them to the caller. To get counts we can assert on,
this test reconstructs the same `WanModel` (the module `OmniAvatarWan.__init__` builds
internally) and drives Stage 1 of `_load_weights()` directly via the module's own helpers
(`_load_state_dict`, `_smart_load_weights`, and the same diffusers-vs-original-format
detection block) -- i.e. exactly the base-weight code path, minus the checkpoint overlay.

Run (GPU 3 only -- check `nvidia-smi -i 3` shows < 10 GB used first):
    CUDA_VISIBLE_DEVICES=3 env LD_LIBRARY_PATH= \
        /home/work/.local/miniconda3/envs/fastgen/bin/python tests/test_omniavatar_wan_load.py
"""

import glob
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.omniavatar_wan.wan_model import WanModel
from models.omniavatar_wan.network import (
    MODEL_CONFIGS,
    _COMMON_CFG,
    _convert_diffusers_state_dict,
    _load_state_dict,
    _smart_load_weights,
)

# Mirrors the OmniAvatarWan(...) call in generate_omniavatar_ode_pairs_full.py, minus
# omniavatar_ckpt_path (deferred -- checkpoint is off-box).
MODEL_SIZE = "14B"
IN_DIM = 65  # V2V + ref_sequence, matches the generate script's default
USE_AUDIO = True
AUDIO_HIDDEN_SIZE = 32

WEIGHTS_ROOT = os.environ.get("WEIGHTS_ROOT", "/home/work/.local/hyunbin/LipForcing-release/weights")
BASE_DIR = os.path.join(WEIGHTS_ROOT, "Wan2.1-T2V-14B")
SHARD_PATHS = sorted(glob.glob(os.path.join(BASE_DIR, "diffusion_pytorch_model-*-of-00006.safetensors")))

# Name patterns legitimately absent from base T2V weights when no OmniAvatar checkpoint is
# loaded: audio conditioning modules are new-to-OmniAvatar (never in base Wan 2.1), and LoRA
# adapters / expanded patch_embedding channels only exist in the teacher checkpoint overlay
# (which this reduced-scope test intentionally skips).
EXPECTED_MISSING_PATTERNS = ("lora_A", "lora_B", "audio_proj", "audio_cond_projs", "patch_embedding")


def _load_base_state_dict(shard_paths, dtype):
    """Replicates Stage 1 of OmniAvatarWan._load_weights() (network.py) verbatim:
    load + merge shards, then detect diffusers vs. original-Wan key format and convert."""
    base_sd = {}
    for p in shard_paths:
        base_sd.update(_load_state_dict(p, dtype=dtype))

    sample_key = next(iter(base_sd.keys()), "")
    is_diffusers = any(
        marker in sample_key for marker in ("condition_embedder", "attn1", "attn2", "ffn.net")
    )
    if is_diffusers:
        print("[test] Detected diffusers-format base weights, converting keys")
        base_sd = _convert_diffusers_state_dict(base_sd)
    else:
        print("[test] Detected original-Wan-format base weights (already WanModel-compatible keys)")
        cleaned = {}
        for k, v in base_sd.items():
            clean_k = k
            for prefix in ("model.", "module.", "transformer."):
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
            cleaned[clean_k] = v
        base_sd = cleaned
    return base_sd


def main():
    assert len(SHARD_PATHS) == 6, (
        f"Expected 6 safetensors shards under {BASE_DIR}, found {len(SHARD_PATHS)}: {SHARD_PATHS}"
    )
    print(f"[test] WEIGHTS_ROOT={WEIGHTS_ROOT}")
    print(f"[test] Found {len(SHARD_PATHS)} base shards under {BASE_DIR}")

    cfg = MODEL_CONFIGS[MODEL_SIZE]
    model = WanModel(
        dim=cfg["dim"],
        in_dim=IN_DIM,
        ffn_dim=cfg["ffn_dim"],
        out_dim=_COMMON_CFG["out_dim"],
        text_dim=_COMMON_CFG["text_dim"],
        freq_dim=_COMMON_CFG["freq_dim"],
        eps=_COMMON_CFG["eps"],
        patch_size=_COMMON_CFG["patch_size"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        use_audio=USE_AUDIO,
        audio_hidden_size=AUDIO_HIDDEN_SIZE,
        has_image_input=False,  # OmniAvatar always uses T2V base, not I2V
    )
    print(f"[test] Constructed WanModel(model_size={MODEL_SIZE}, in_dim={IN_DIM}, use_audio={USE_AUDIO})")

    dtype = torch.bfloat16
    base_sd = _load_base_state_dict(SHARD_PATHS, dtype=dtype)
    print(f"[test] Loaded+converted {len(base_sd)} base weight tensors (dtype={dtype})")

    missing, unexpected = _smart_load_weights(model, base_sd)
    model = model.to(dtype)  # mirrors OmniAvatarWan.__init__'s post-_load_weights() cast

    print(f"[test] Base-weight load: missing={len(missing)} keys, unexpected={len(unexpected)} keys")
    if unexpected:
        print(f"[test]   unexpected (first 10): {unexpected[:10]}")

    bad_missing = [k for k in missing if not any(pat in k for pat in EXPECTED_MISSING_PATTERNS)]
    print(
        f"[test] missing keys NOT matching expected patterns {EXPECTED_MISSING_PATTERNS}: "
        f"{len(bad_missing)}"
    )
    if missing:
        print(f"[test]   missing (first 10): {missing[:10]}")
    if bad_missing:
        print(f"[test]   unexpected-missing (first 10): {bad_missing[:10]}")

    assert len(unexpected) == 0, f"Unexpected keys from base load: {unexpected[:20]}"
    assert len(missing) > 0, "Expected at least the audio modules to be missing from base-only weights"
    assert len(bad_missing) == 0, f"Missing keys outside expected patterns: {bad_missing[:20]}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[test] Total parameter count: {n_params:,}")
    assert n_params > 14e9, f"Expected >14e9 params for a 14B model, got {n_params:,}"

    # Move to GPU to confirm the loaded tensors are well-formed and transfer cleanly.
    if torch.cuda.is_available():
        model = model.to("cuda", dtype=dtype)
        torch.cuda.synchronize()
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        print(f"[test] Moved model to cuda (bf16); {allocated_gb:.2f} GB allocated on device")
        device_used = "GPU (cuda:0 within CUDA_VISIBLE_DEVICES restriction)"
    else:
        print("[test] CUDA not available; kept model on CPU")
        device_used = "CPU"

    print(f"[test] Device used: {device_used}")
    print("[test] PASSED")


if __name__ == "__main__":
    main()
