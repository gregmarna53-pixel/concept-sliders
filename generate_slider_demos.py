"""
Generate labelled alpha-sweep comparison strips for every trained concept slider.

For each slider we keep the prompt, seed and sampler identical and only vary the
LoRA multiplier (alpha), so the panels differ *only* by the slider.

Outputs: images/demo_<name>.png  (one horizontal strip per slider)

Usage:
    python generate_slider_demos.py
    python generate_slider_demos.py --sliders fatcat weather
"""

import argparse
import gc
import os
from pathlib import Path

def _base_model_is_cached() -> bool:
    """True when CompVis/stable-diffusion-v1-4 already has a local snapshot."""
    from huggingface_hub import constants

    snapshots = (
        Path(constants.HF_HUB_CACHE) / "models--CompVis--stable-diffusion-v1-4" / "snapshots"
    )
    return snapshots.is_dir() and any(snapshots.iterdir())


# Only pin to the cache when the base model is already downloaded. On a fresh
# checkout it must stay free to fetch the model, otherwise the first run fails.
if _base_model_is_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
import trainscripts.textsliders.lora as lora_mod
from trainscripts.textsliders.lora import LoRANetwork

# Upstream's LoRANetwork hardcodes DEFAULT_TARGET_REPLACE (Attention only) and
# silently ignores the `modules` list its own training script computes, so a
# plain LoRANetwork only builds 64 of the 150 modules our c3lier checkpoints
# contain and load_state_dict fails. Widen the global it reads before we build.
lora_mod.DEFAULT_TARGET_REPLACE = (
    lora_mod.UNET_TARGET_REPLACE_MODULE_TRANSFORMER + lora_mod.UNET_TARGET_REPLACE_MODULE_CONV
)

PRETRAINED_MODEL = "CompVis/stable-diffusion-v1-4"
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "images"

# scale 0.0 is the untouched base model and sits in the middle of the strip
SCALES = [-4.0, -2.0, 0.0, 2.0, 4.0]

STEPS = 50
GUIDANCE_SCALE = 7.5
START_NOISE = 800  # slider is only applied after this timestep
NEGATIVE_PROMPT = "split screen, missing limbs, bad anatomy, worst quality"
RESOLUTION = 512
SEED = 888

# name -> (weights path, prompt, caption)
SLIDERS = {
    "fatcat": (
        "models/fatcatslider_alpha1.0_rank4_noxattn/fatcatslider_alpha1.0_rank4_noxattn_500steps.safetensors",
        "a highly detailed photorealistic full body shot of a cat standing on a wooden floor, 8k resolution",
        "cat body weight",
    ),
    "smile": (
        "models/smileslider_alpha1.0_rank4_noxattn/smileslider_alpha1.0_rank4_noxattn_500steps.safetensors",
        "a highly detailed photorealistic portrait photo of a person, upper body, looking at the camera, 8k resolution",
        "smile intensity",
    ),
    "weather": (
        "models/weatherslider_alpha1.0_rank4_noxattn/weatherslider_alpha1.0_rank4_noxattn_500steps.safetensors",
        "a highly detailed photorealistic wide landscape photo of a street with sky, 8k resolution",
        "weather mood",
    ),
}


def load_font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_strip(images, scales, caption, font):
    """Stitch the per-scale images into one strip with a labelled header bar."""
    w, h = images[0].size
    bar = 56
    strip = Image.new("RGB", (w * len(images), h + bar), "white")
    draw = ImageDraw.Draw(strip)

    title = font
    for i, (img, scale) in enumerate(zip(images, scales)):
        strip.paste(img, (i * w, bar))
        label = f"alpha = {scale:+.1f}"
        if scale == 0.0:
            label += "  (base)"
        tw = draw.textlength(label, font=title)
        draw.text((i * w + (w - tw) / 2, (bar - 30) / 2), label, fill="black", font=title)

    # thin separators between panels
    for i in range(1, len(images)):
        draw.line([(i * w, bar), (i * w, h + bar)], fill=(220, 220, 220), width=2)

    return strip


def build_network(unet_path, device, weight_dtype):
    unet = UNet2DConditionModel.from_pretrained(
        PRETRAINED_MODEL, subfolder="unet", revision=None
    )
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)

    rank = 8 if "rank8" in unet_path else 4
    network = LoRANetwork(
        unet, rank=rank, multiplier=1.0, alpha=1.0, train_method="noxattn"
    ).to(device, dtype=weight_dtype)
    return unet, network


@torch.no_grad()
def render(prompt, scales, tokenizer, text_encoder, vae, unet, network, device, weight_dtype):
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    uncond_input = tokenizer(
        [NEGATIVE_PROMPT],
        padding="max_length",
        max_length=text_input.input_ids.shape[-1],
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    cond_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    images = []
    for scale in scales:
        generator = torch.manual_seed(SEED)
        latents = torch.randn(
            (1, unet.in_channels, RESOLUTION // 8, RESOLUTION // 8), generator=generator
        ).to(device)

        scheduler.set_timesteps(STEPS)
        latents = (latents * scheduler.init_noise_sigma).to(weight_dtype)

        for t in scheduler.timesteps:
            # below START_NOISE the slider is active; above it we let the base model
            # lay down the structure first so the edit stays clean
            network.set_lora_slider(scale=scale if t <= START_NOISE else 0)
            latent_input = scheduler.scale_model_input(torch.cat([latents] * 2), timestep=t)

            with network:
                noise_pred = unet(latent_input, t, encoder_hidden_states=cond_embeddings).sample

            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_text - noise_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        images.append(Image.fromarray((image[0] * 255).round().astype("uint8")))

    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sliders", nargs="*", default=list(SLIDERS), choices=list(SLIDERS))
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    font = load_font(26)

    tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_MODEL, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)

    vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL, subfolder="vae")
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    for name in args.sliders:
        rel_path, prompt, caption = SLIDERS[name]
        weights = ROOT / rel_path
        if not weights.exists():
            print(f"[skip] {name}: {weights} not found")
            continue

        print(f"\n=== {name} ({caption}) ===")
        print(f"    prompt: {prompt}")

        unet, network = build_network(str(weights), device, weight_dtype)
        state = load_file(str(weights)) if weights.suffix == ".safetensors" else torch.load(weights, map_location=device)
        network.load_state_dict(state)

        images = render(prompt, SCALES, tokenizer, text_encoder, vae, unet, network, device, weight_dtype)

        strip = build_strip(images, SCALES, caption, font)
        out = OUT_DIR / f"demo_{name}.png"
        strip.save(out)
        print(f"    saved -> {out.relative_to(ROOT)}  ({strip.size[0]}x{strip.size[1]})")

        del network, unet
        gc.collect()
        torch.cuda.empty_cache()

    print("\ndone.")


if __name__ == "__main__":
    main()
