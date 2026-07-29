"""Direct loader for the vendored WanVideoVAE, replacing the
OmniAvatar.models.model_manager.ModelManager dependency for VAE decode.

This mirrors OmniAvatar/models/model_manager.py's actual code path for
loading a civitai-format single-file checkpoint (the `Wan2.1_VAE.pth` case),
specifically `load_model_from_single_file()` (model_manager.py:8-41) as
invoked by `ModelManager.load_model()` with `infer=True` -- every OmniAvatar
inference entry point (scripts/inference.py, inference_v2v.py,
precompute_vae_latents_masked.py, etc.) constructs `ModelManager(infer=True)`,
so that is the only code path relevant here; the `infer=False` xavier-init
training branch (model_manager.py:27-33) never applies to VAE loading and is
intentionally not replicated.

Traced code path (OmniAvatar/models/model_manager.py):
  1. `state_dict = load_state_dict(file_path)` (ModelManager.load_model,
     line 376) -> since Wan2.1_VAE.pth is not `.safetensors`, this is
     `load_state_dict_from_bin` (utils/io_utils.py:94-100), i.e.
     `torch.load(file_path, map_location="cpu", weights_only=True)`.
  2. `state_dict_converter = model_class.state_dict_converter()` then
     `state_dict_converter.from_civitai(state_dict)` (line 12-14), since the
     VAE is registered with `model_resource="civitai"`
     (configs/model_config.py:15-16). For `WanVideoVAE`,
     `from_civitai` (wan_video_vae.py) returns a plain dict, NOT a tuple, so
     `extra_kwargs = {}` (line 17-21) -- there are no extra constructor kwargs
     for this VAE (z_dim stays at its default of 16).
  3. `torch_dtype = torch.float32 if extra_kwargs.get("upcast_to_float32", False)
     else torch_dtype` (line 22) -- a no-op here since extra_kwargs is always
     `{}` for the VAE, but replicated verbatim in case that ever changes.
  4. `model = model_class(**extra_kwargs)` then `model.eval()` (lines 23-26).
     ModelManager constructs this inside `init_weights_on_device()` (meta
     device) purely as a memory optimization for huge models (DiT); for the
     VAE (~127M params) this is unnecessary, and since step 6 below overwrites
     every parameter anyway, skipping the meta/`to_empty` dance is numerically
     identical to replicating it -- we just construct directly on CPU.
  5. Since `infer=True`, `model = model.to_empty(device=device)`
     (line 34-35) -- also a memory-allocation detail superseded by step 6;
     not replicated for the same reason as step 4.
  6. `model, _, _ = smart_load_weights(model, model_state_dict)` (line 36).
     NOTE: this is NOT a plain `model.load_state_dict(...)` call -- that line
     is present but commented out in ModelManager (line 37, `# model.
     load_state_dict(model_state_dict, assign=True, strict=False)`).
     `smart_load_weights` (utils/io_utils.py:102-125) builds a filtered
     state dict (truncating any checkpoint tensor that's larger than the
     model's declared shape along every dim, skipping ones that are smaller)
     and applies it via `model.load_state_dict(new_state_dict, assign=True,
     strict=False)`. Reimplemented verbatim below (`_smart_load_weights`) so
     this package has no import-time dependency on the OmniAvatar package.
     For this unchanged VAE class every checkpoint key/shape matches the
     model exactly, so this reduces to a complete overlay with no truncation
     or skipping -- but we still call the real algorithm, not a plain
     `load_state_dict`, for fidelity.
  7. `model = model.to(dtype=torch_dtype, device=device)` (line 38).

Deviation from ModelManager (deliberate, documented): ModelManager discards
`missing_keys`/`unexpected_keys` (`model, _, _ = smart_load_weights(...)`) and
never raises on a mismatch -- appropriate for its generic use across models
where partial-shape overlays are legitimate (e.g. DiT patch_embedding
channel expansion). This loader is VAE-only, where there is no legitimate
reason for a mismatch, so `load_wan_vae` raises `RuntimeError` if any missing
or unexpected key is reported, to fail loudly instead of silently on a wrong
checkpoint file.
"""

import torch

from .wan_video_vae import WanVideoVAE


def _smart_load_weights(model, ckpt_state_dict):
    """Verbatim port of OmniAvatar.utils.io_utils.smart_load_weights."""
    model_state_dict = model.state_dict()
    new_state_dict = {}

    for name, param in model_state_dict.items():
        if name in ckpt_state_dict:
            ckpt_param = ckpt_state_dict[name]
            if param.shape == ckpt_param.shape:
                new_state_dict[name] = ckpt_param
            else:
                # Auto-truncate to fit, matching smart_load_weights exactly.
                if all(p >= c for p, c in zip(param.shape, ckpt_param.shape)):
                    print(f"[Truncate] {name}: ckpt {ckpt_param.shape} -> model {param.shape}")
                    new_param = param.clone()
                    slices = tuple(slice(0, s) for s in ckpt_param.shape)
                    new_param[slices] = ckpt_param
                    new_state_dict[name] = new_param
                else:
                    print(f"[Skip] {name}: ckpt {ckpt_param.shape} is larger than model {param.shape}")

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, assign=True, strict=False)
    return model, missing_keys, unexpected_keys


def load_wan_vae(vae_path, dtype=torch.bfloat16, device="cpu"):
    """Load a civitai-format Wan VAE checkpoint (e.g. Wan2.1_VAE.pth),
    replicating ModelManager's load_model_from_single_file() code path for
    infer=True. Returns a WanVideoVAE in eval mode, cast to `dtype` on
    `device`, with `.decode(latents, device=device, tiled=False)` matching
    the ModelManager-loaded VAE bit-for-bit (verified on GPU in Task 19).
    """
    # Step 1: mirrors utils/io_utils.py load_state_dict -> load_state_dict_from_bin
    # for non-.safetensors files.
    state_dict = torch.load(vae_path, map_location="cpu", weights_only=True)

    # Step 2: state_dict_converter().from_civitai(...) + tuple/non-tuple handling
    # (model_manager.py:12-21).
    result = WanVideoVAE.state_dict_converter().from_civitai(state_dict)
    if isinstance(result, tuple):
        model_state_dict, extra_kwargs = result
    else:
        model_state_dict, extra_kwargs = result, {}

    # Step 3: upcast_to_float32 extra_kwargs check (model_manager.py:22).
    load_dtype = torch.float32 if extra_kwargs.get("upcast_to_float32", False) else dtype

    # Step 4: model_class(**extra_kwargs) + eval() (model_manager.py:23-26).
    vae = WanVideoVAE(**extra_kwargs)
    vae = vae.eval()

    # Steps 5-6: smart_load_weights overlay (model_manager.py:34-36).
    vae, missing_keys, unexpected_keys = _smart_load_weights(vae, model_state_dict)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "load_wan_vae: state_dict mismatch loading "
            f"{vae_path!r} -- missing_keys={missing_keys}, "
            f"unexpected_keys={unexpected_keys}"
        )

    # Step 7: final dtype/device cast (model_manager.py:38).
    return vae.to(dtype=load_dtype, device=device).eval()
