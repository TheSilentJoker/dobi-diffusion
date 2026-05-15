from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from diffusers import DDPMScheduler, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DOBI NIR images from prompts.")
    parser.add_argument("--model-dir", default="outputs/text_ddpm")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-csv", default="data/manifests/dobi_prompts.csv")
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--num-per-prompt", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--output-dir", default="outputs/generated")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _encode_prompts(
    prompts: list[str],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
) -> torch.Tensor:
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        return text_encoder(**tokens).last_hidden_state


def _tensor_to_dobi_image(sample: torch.Tensor) -> Image.Image:
    image = ((sample.clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    pil = Image.fromarray(image).convert("L")
    if pil.size == (128, 128):
        pil = pil.crop((0, 13, 128, 115))
    return pil


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    unet = UNet2DConditionModel.from_pretrained(model_dir / "unet").to(device)
    scheduler = DDPMScheduler.from_pretrained(model_dir / "scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_dir / "tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_dir / "text_encoder").to(device)
    unet.eval()
    text_encoder.eval()

    if args.prompt:
        prompt_rows = pd.DataFrame([{"prompt": args.prompt, "label": args.label if args.label is not None else -1}])
    else:
        prompt_rows = pd.read_csv(args.prompt_csv)
        if args.label is not None:
            prompt_rows = prompt_rows[prompt_rows["label"] == args.label]
        prompt_rows = prompt_rows.sample(min(args.num_prompts, len(prompt_rows)), random_state=args.seed)

    records: list[dict[str, object]] = []
    scheduler.set_timesteps(args.steps, device=device)
    image_size = int(json.loads((model_dir / "training_config.json").read_text(encoding="utf-8")).get("image_size", 128))

    for row_index, row in tqdm(list(prompt_rows.iterrows()), desc="generating"):
        prompt = str(row["prompt"])
        label = int(row.get("label", -1))
        for sample_index in range(args.num_per_prompt):
            sample = torch.randn((1, 3, image_size, image_size), generator=generator, device=device)
            cond = _encode_prompts([prompt], tokenizer, text_encoder, device)
            uncond = _encode_prompts([""], tokenizer, text_encoder, device)

            for timestep in scheduler.timesteps:
                with torch.no_grad():
                    if args.guidance_scale > 1.0:
                        model_input = torch.cat([sample, sample], dim=0)
                        text_states = torch.cat([uncond, cond], dim=0)
                        noise_pred = unet(model_input, timestep, encoder_hidden_states=text_states).sample
                        noise_uncond, noise_cond = noise_pred.chunk(2)
                        noise_pred = noise_uncond + args.guidance_scale * (noise_cond - noise_uncond)
                    else:
                        noise_pred = unet(sample, timestep, encoder_hidden_states=cond).sample
                    sample = scheduler.step(noise_pred, timestep, sample).prev_sample

            filename = f"generated_label{label}_{row_index}_{sample_index}.png"
            image_path = output_dir / filename
            _tensor_to_dobi_image(sample[0]).save(image_path)
            records.append(
                {
                    "filename": image_path.stem,
                    "image_path": str(image_path),
                    "label": label,
                    "split": "train",
                    "prompt": prompt,
                    "source_row": int(row_index) if isinstance(row_index, int) else str(row_index),
                }
            )

    manifest_path = output_dir / "generated_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(records)} generated images and manifest to {manifest_path}")


if __name__ == "__main__":
    main()
