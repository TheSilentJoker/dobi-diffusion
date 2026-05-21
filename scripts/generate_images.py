from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import DDPMScheduler, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breastdiffusion.dobi_masks import canonical_dobi_mask, extract_foreground_mask, resize_pad_gray
from breastdiffusion.models import MetadataConditioner


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
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeu-s1", type=float, default=0.9)
    parser.add_argument("--freeu-s2", type=float, default=0.2)
    parser.add_argument("--freeu-b1", type=float, default=1.2)
    parser.add_argument("--freeu-b2", type=float, default=1.4)
    parser.add_argument("--mask-source", choices=["canonical", "source"], default="canonical")
    parser.add_argument("--output-dir", default="outputs/generated")
    parser.add_argument("--run-name", default=None, help="Optional suffix for the timestamped output subdirectory.")
    parser.add_argument("--no-run-subdir", action="store_true", help="Write directly into output-dir instead of a timestamped subdirectory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-cols", type=int, default=8, help="Columns in the generated preview grid.")
    parser.add_argument("--no-grid", action="store_true", help="Do not save preview_grid.png.")
    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-threshold", type=int, default=12, help="Pixels below this value are treated as black background.")
    parser.add_argument("--contrast-low", type=float, default=1.0, help="Low percentile for generated image contrast stretching.")
    parser.add_argument("--contrast-high", type=float, default=99.0, help="High percentile for generated image contrast stretching.")
    return parser.parse_args()


def _resolve_output_dir(base_dir: str | Path, run_name: str | None, no_run_subdir: bool) -> Path:
    output_dir = Path(base_dir)
    if no_run_subdir:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_name}" if run_name else ""
    run_dir = output_dir / f"{timestamp}{suffix}"
    index = 1
    while run_dir.exists():
        run_dir = output_dir / f"{timestamp}{suffix}_{index:02d}"
        index += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _resolve_model_dir(model_dir: str | Path) -> Path:
    model_path = Path(model_dir)
    if (model_path / "unet").exists() or (model_path / "unet_ema").exists():
        return model_path
    if (model_path / "latest" / "unet").exists() or (model_path / "latest" / "unet_ema").exists():
        return model_path / "latest"

    candidates: list[Path] = []
    if model_path.exists():
        for child in model_path.iterdir():
            if not child.is_dir():
                continue
            if (child / "latest" / "unet").exists() or (child / "latest" / "unet_ema").exists():
                candidates.append(child / "latest")
            elif (child / "unet").exists() or (child / "unet_ema").exists():
                candidates.append(child)
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return model_path


def _encode_prompts(
    prompts: list[str],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
    metadata: torch.Tensor | None = None,
    metadata_conditioner: MetadataConditioner | None = None,
) -> torch.Tensor:
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        text_states = text_encoder(**tokens).last_hidden_state
        if metadata is not None and metadata_conditioner is not None:
            text_states = torch.cat([text_states, metadata_conditioner(metadata)], dim=1)
        return text_states


def _mask_for_row(row: pd.Series, image_size: int, source: str, threshold: int, device: torch.device) -> torch.Tensor:
    if source == "source" and "image_path" in row and Path(str(row["image_path"])).exists():
        image = resize_pad_gray(str(row["image_path"]), image_size)
        mask = extract_foreground_mask(image, threshold=threshold)
    else:
        mask = canonical_dobi_mask(image_size=image_size)
    array = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array[None, None, :, :]).float().to(device)


def _metadata_for_row(row: pd.Series, config: dict[str, object], device: torch.device) -> torch.Tensor | None:
    columns = list(config.get("metadata_columns", []))
    if not config.get("use_metadata", False) or not columns:
        return None
    mean = np.asarray(config.get("metadata_mean", [0.0] * len(columns)), dtype=np.float32)
    std = np.asarray(config.get("metadata_std", [1.0] * len(columns)), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    values = []
    for index, column in enumerate(columns):
        if column in row and not pd.isna(row[column]):
            values.append(float(row[column]))
        else:
            values.append(float(mean[index]))
    normalized = (np.asarray(values, dtype=np.float32) - mean) / std
    return torch.from_numpy(normalized[None, :]).float().to(device)


def _tensor_to_dobi_image(sample: torch.Tensor) -> Image.Image:
    image = ((sample.clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    pil = Image.fromarray(image).convert("L")
    if pil.size == (128, 128):
        pil = pil.crop((0, 13, 128, 115))
    return pil


def _postprocess_dobi_image(
    image: Image.Image,
    mask_threshold: int,
    contrast_low: float,
    contrast_high: float,
) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    foreground = array > mask_threshold
    if foreground.any():
        values = array[foreground]
        lo = float(np.percentile(values, contrast_low))
        hi = float(np.percentile(values, contrast_high))
        if hi > lo:
            array = (array - lo) / (hi - lo) * 255.0
        array[~foreground] = 0.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def _save_preview_grid(images: list[Image.Image], output_path: Path, cols: int) -> None:
    if not images:
        return
    cols = max(1, min(cols, len(images)))
    rows = (len(images) + cols - 1) // cols
    width, height = images[0].size
    grid = Image.new("L", (cols * width, rows * height), color=0)
    for index, image in enumerate(images):
        x = (index % cols) * width
        y = (index // cols) * height
        grid.paste(image, (x, y))
    grid.save(output_path)


def main() -> None:
    args = parse_args()
    model_dir = _resolve_model_dir(args.model_dir)
    print(f"Loading model from {model_dir}")
    output_dir = _resolve_output_dir(args.output_dir, args.run_name, args.no_run_subdir)
    print(f"Writing generated images to {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    config = json.loads((model_dir / "training_config.json").read_text(encoding="utf-8"))
    unet_subdir = "unet_ema" if args.use_ema and (model_dir / "unet_ema").exists() else "unet"
    unet = UNet2DConditionModel.from_pretrained(model_dir / unet_subdir).to(device)
    scheduler = DDPMScheduler.from_pretrained(model_dir / "scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_dir / "tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_dir / "text_encoder").to(device)
    metadata_conditioner = None
    if config.get("use_metadata", False) and (model_dir / "metadata_conditioner.pt").exists():
        metadata_conditioner = MetadataConditioner(
            input_dim=len(config.get("metadata_columns", [])),
            hidden_size=text_encoder.config.hidden_size,
            num_tokens=int(config.get("metadata_tokens", 4)),
            dropout=0.0,
        ).to(device)
        metadata_conditioner.load_state_dict(torch.load(model_dir / "metadata_conditioner.pt", map_location=device))
        metadata_conditioner.eval()
    if args.freeu and hasattr(unet, "enable_freeu"):
        unet.enable_freeu(s1=args.freeu_s1, s2=args.freeu_s2, b1=args.freeu_b1, b2=args.freeu_b2)
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
    preview_images: list[Image.Image] = []
    scheduler.set_timesteps(args.steps, device=device)
    image_size = int(config.get("image_size", 128))
    use_mask = bool(config.get("use_mask", False))

    for row_index, row in tqdm(list(prompt_rows.iterrows()), desc="generating"):
        prompt = str(row["prompt"])
        label = int(row.get("label", -1))
        mask_values = _mask_for_row(row, image_size, args.mask_source, args.mask_threshold, device) if use_mask else None
        metadata_values = _metadata_for_row(row, config, device)
        empty_metadata = torch.zeros_like(metadata_values) if metadata_values is not None else None
        for sample_index in range(args.num_per_prompt):
            sample = torch.randn((1, 3, image_size, image_size), generator=generator, device=device)
            cond = _encode_prompts([prompt], tokenizer, text_encoder, device, metadata_values, metadata_conditioner)
            uncond = _encode_prompts([""], tokenizer, text_encoder, device, empty_metadata, metadata_conditioner)

            for timestep in scheduler.timesteps:
                with torch.no_grad():
                    if args.guidance_scale > 1.0:
                        step_input = torch.cat([sample, mask_values], dim=1) if use_mask and mask_values is not None else sample
                        model_input = torch.cat([step_input, step_input], dim=0)
                        text_states = torch.cat([uncond, cond], dim=0)
                        noise_pred = unet(model_input, timestep, encoder_hidden_states=text_states).sample
                        noise_uncond, noise_cond = noise_pred.chunk(2)
                        noise_pred = noise_uncond + args.guidance_scale * (noise_cond - noise_uncond)
                    else:
                        model_input = torch.cat([sample, mask_values], dim=1) if use_mask and mask_values is not None else sample
                        noise_pred = unet(model_input, timestep, encoder_hidden_states=cond).sample
                    sample = scheduler.step(noise_pred, timestep, sample).prev_sample

            filename = f"generated_label{label}_{row_index}_{sample_index}.png"
            image_path = output_dir / filename
            image = _tensor_to_dobi_image(sample[0])
            raw_image_path = output_dir / f"{image_path.stem}_raw.png"
            image.save(raw_image_path)
            if args.postprocess:
                image = _postprocess_dobi_image(image, args.mask_threshold, args.contrast_low, args.contrast_high)
            image.save(image_path)
            preview_images.append(image)
            records.append(
                {
                    "filename": image_path.stem,
                    "image_path": str(image_path),
                    "raw_image_path": str(raw_image_path),
                    "label": label,
                    "split": "train",
                    "prompt": prompt,
                    "source_row": int(row_index) if isinstance(row_index, int) else str(row_index),
                }
            )

    manifest_path = output_dir / "generated_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    if not args.no_grid:
        _save_preview_grid(preview_images, output_dir / "preview_grid.png", args.grid_cols)
    print(f"Saved {len(records)} generated images and manifest to {manifest_path}")
    if not args.no_grid:
        print(f"Saved preview grid to {output_dir / 'preview_grid.png'}")


if __name__ == "__main__":
    main()
