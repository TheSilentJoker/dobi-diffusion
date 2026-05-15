from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class DobiPromptDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, image_size: int, prompt_dropout: float = 0.0) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.image_size = image_size
        self.prompt_dropout = prompt_dropout

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_image(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("L")
        image = ImageOps.contain(image, (self.image_size, self.image_size), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (self.image_size, self.image_size), color=0)
        offset = ((self.image_size - image.width) // 2, (self.image_size - image.height) // 2)
        canvas.paste(image, offset)
        rgb = canvas.convert("RGB")
        array = torch.from_numpy(np.asarray(rgb, dtype="float32")).permute(2, 0, 1)
        return array / 127.5 - 1.0

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.manifest.iloc[index]
        prompt = str(row["prompt"])
        if self.prompt_dropout and random.random() < self.prompt_dropout:
            prompt = ""
        return {
            "pixel_values": self._load_image(str(row["image_path"])),
            "prompt": prompt,
            "label": int(row["label"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text-conditioned DDPM for DOBI NIR images.")
    parser.add_argument("--manifest", default="data/manifests/dobi_prompts.csv")
    parser.add_argument("--output-dir", default="outputs/text_ddpm")
    parser.add_argument("--text-encoder", default="openai/clip-vit-base-patch32")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--prompt-dropout", type=float, default=0.1)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def _build_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    labels = df["label"].astype(int).to_numpy()
    counts = {label: max(int((labels == label).sum()), 1) for label in set(labels)}
    weights = torch.as_tensor([1.0 / counts[int(label)] for label in labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _save_checkpoint(
    output_dir: Path,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(output_dir / "unet")
    scheduler.save_pretrained(output_dir / "scheduler")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    text_encoder.save_pretrained(output_dir / "text_encoder")
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(args.manifest)
    train_df = manifest[manifest["split"] == "train"].copy()

    tokenizer = CLIPTokenizer.from_pretrained(args.text_encoder)
    text_encoder = CLIPTextModel.from_pretrained(args.text_encoder).to(device)
    text_encoder.eval()
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)

    unet = UNet2DConditionModel(
        sample_size=args.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=text_encoder.config.hidden_size,
    ).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    dataset = DobiPromptDataset(train_df, args.image_size, args.prompt_dropout)
    sampler = _build_sampler(train_df) if args.balanced_sampling else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    output_dir = Path(args.output_dir)
    config = vars(args) | {"device": str(device), "train_rows": int(len(train_df))}

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        unet.train()
        losses: list[float] = []
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            pixel_values = batch["pixel_values"].to(device)
            noise = torch.randn_like(pixel_values)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (pixel_values.shape[0],),
                device=device,
            ).long()
            noisy_images = scheduler.add_noise(pixel_values, noise, timesteps)
            tokens = tokenizer(
                list(batch["prompt"]),
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                text_states = text_encoder(**tokens).last_hidden_state

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                prediction = unet(noisy_images, timesteps, encoder_hidden_states=text_states).sample
                loss = F.mse_loss(prediction, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            _save_checkpoint(output_dir, unet, scheduler, tokenizer, text_encoder, config | {"epoch": epoch, "global_step": global_step})

    _save_checkpoint(output_dir, unet, scheduler, tokenizer, text_encoder, config | {"epoch": args.epochs, "global_step": global_step})


if __name__ == "__main__":
    main()
