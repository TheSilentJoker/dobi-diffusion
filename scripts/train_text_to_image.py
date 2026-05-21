from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breastdiffusion.dobi_masks import image_and_mask_tensors
from breastdiffusion.models import MetadataConditioner, min_snr_weights


METADATA_COLUMNS = [
    "Breast_Ratio_Global(%)",
    "Illum_Coverage_Local(%)",
    "Leak_Ratio_Local(%)",
]


def _init_swanlab(args: argparse.Namespace, config: dict[str, object]):
    if not args.swanlab:
        return None
    try:
        import swanlab
    except ImportError as exc:
        raise ImportError(
            "SwanLab logging was requested, but the 'swanlab' package is not installed. "
            "Install it with: pip install swanlab"
        ) from exc

    return swanlab.init(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        experiment_name=args.swanlab_experiment,
        description="Text-conditioned DDPM training for DOBI NIR breast images.",
        config=config,
        logdir=args.swanlab_logdir,
        mode=args.swanlab_mode,
    )


def _swanlab_log(run, data: dict[str, object], step: int) -> None:
    if run is None:
        return
    run.log(data, step=step)


def _swanlab_finish(run) -> None:
    if run is None:
        return
    run.finish()


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


class DobiPromptDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        image_size: int,
        prompt_dropout: float = 0.0,
        use_mask: bool = True,
        mask_threshold: int = 12,
        metadata_columns: list[str] | None = None,
        metadata_mean: list[float] | None = None,
        metadata_std: list[float] | None = None,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.image_size = image_size
        self.prompt_dropout = prompt_dropout
        self.use_mask = use_mask
        self.mask_threshold = mask_threshold
        self.metadata_columns = metadata_columns or []
        self.metadata_mean = np.asarray(metadata_mean or [0.0] * len(self.metadata_columns), dtype=np.float32)
        self.metadata_std = np.asarray(metadata_std or [1.0] * len(self.metadata_columns), dtype=np.float32)
        self.metadata_std = np.where(self.metadata_std < 1e-6, 1.0, self.metadata_std)

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_image_and_mask(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        image_array, mask_array = image_and_mask_tensors(path, self.image_size, threshold=self.mask_threshold)
        rgb = np.repeat(image_array[None, :, :], 3, axis=0)
        image = torch.from_numpy(rgb).float() * 2.0 - 1.0
        mask = torch.from_numpy(mask_array[None, :, :]).float()
        if not self.use_mask:
            mask = torch.zeros_like(mask)
        return image, mask

    def _metadata(self, row: pd.Series) -> torch.Tensor:
        if not self.metadata_columns:
            return torch.zeros(0, dtype=torch.float32)
        values = row[self.metadata_columns].astype(float).to_numpy(dtype=np.float32)
        values = (values - self.metadata_mean) / self.metadata_std
        return torch.from_numpy(values.astype(np.float32))

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.manifest.iloc[index]
        prompt = str(row["prompt"])
        if self.prompt_dropout and random.random() < self.prompt_dropout:
            prompt = ""
        image, mask = self._load_image_and_mask(str(row["image_path"]))
        return {
            "pixel_values": image,
            "mask_values": mask,
            "metadata_values": self._metadata(row),
            "prompt": prompt,
            "label": int(row["label"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text-conditioned DDPM for DOBI NIR images.")
    parser.add_argument("--manifest", default="data/manifests/dobi_prompts.csv")
    parser.add_argument("--output-dir", default="outputs/text_ddpm")
    parser.add_argument("--run-name", default=None, help="Optional suffix for the timestamped training run directory.")
    parser.add_argument("--no-run-subdir", action="store_true", help="Write directly into output-dir instead of a timestamped run directory.")
    parser.add_argument("--text-encoder", default="openai/clip-vit-base-patch32")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--prompt-dropout", type=float, default=0.1)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--condition-mode",
        choices=["text", "text_mask", "text_metadata", "text_mask_metadata"],
        default="text_mask_metadata",
        help="Condition set used by the DOBI diffusion model.",
    )
    parser.add_argument("--mask-threshold", type=int, default=12)
    parser.add_argument("--foreground-loss-weight", type=float, default=2.0)
    parser.add_argument("--background-loss-weight", type=float, default=1.5)
    parser.add_argument("--background-black-loss-weight", type=float, default=0.05)
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument("--metadata-tokens", type=int, default=4)
    parser.add_argument("--metadata-dropout", type=float, default=0.0)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10, help="Log training step metrics every N steps.")
    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=False, help="Enable SwanLab tracking.")
    parser.add_argument("--swanlab-project", default="BreastDiffusion-ai")
    parser.add_argument("--swanlab-workspace", default=None)
    parser.add_argument("--swanlab-experiment", default=None)
    parser.add_argument("--swanlab-mode", default=None, choices=[None, "cloud", "local", "offline", "disabled"])
    parser.add_argument("--swanlab-logdir", default="swanlog")
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
    metadata_conditioner: MetadataConditioner | None = None,
    ema_unet: UNet2DConditionModel | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(output_dir / "unet")
    if ema_unet is not None:
        ema_unet.save_pretrained(output_dir / "unet_ema")
    scheduler.save_pretrained(output_dir / "scheduler")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    text_encoder.save_pretrained(output_dir / "text_encoder")
    if metadata_conditioner is not None:
        torch.save(metadata_conditioner.state_dict(), output_dir / "metadata_conditioner.pt")
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def _checkpoint_dir(output_dir: Path, epoch: int) -> Path:
    return output_dir / f"checkpoint-epoch-{epoch:03d}"


def _save_training_state(
    output_dir: Path,
    epoch: int,
    global_step: int,
    epoch_loss: float,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    metadata_conditioner: MetadataConditioner | None,
    ema_unet: UNet2DConditionModel | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, object],
) -> dict[str, str]:
    state_config = config | {"epoch": epoch, "global_step": global_step, "epoch_loss": epoch_loss}
    checkpoint_path = _checkpoint_dir(output_dir, epoch)
    latest_path = output_dir / "latest"
    _save_checkpoint(checkpoint_path, unet, scheduler, tokenizer, text_encoder, state_config, metadata_conditioner, ema_unet)
    _save_checkpoint(latest_path, unet, scheduler, tokenizer, text_encoder, state_config, metadata_conditioner, ema_unet)
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "epoch_loss": epoch_loss,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": state_config,
    }
    if metadata_conditioner is not None:
        training_state["metadata_conditioner"] = metadata_conditioner.state_dict()
    torch.save(training_state, checkpoint_path / "training_state.pt")
    torch.save(training_state, latest_path / "training_state.pt")
    return {
        "checkpoint_path": str(checkpoint_path),
        "latest_path": str(latest_path),
    }


@torch.no_grad()
def _update_ema(ema_unet: UNet2DConditionModel, unet: UNet2DConditionModel, decay: float) -> None:
    ema_state = ema_unet.state_dict()
    model_state = unet.state_dict()
    for key, ema_value in ema_state.items():
        if not torch.is_floating_point(ema_value):
            ema_value.copy_(model_state[key])
        else:
            ema_value.mul_(decay).add_(model_state[key], alpha=1.0 - decay)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(args.manifest)
    train_df = manifest[manifest["split"] == "train"].copy()
    use_mask = "mask" in args.condition_mode
    use_metadata = "metadata" in args.condition_mode
    metadata_columns = [column for column in METADATA_COLUMNS if column in train_df.columns] if use_metadata else []
    metadata_mean = train_df[metadata_columns].astype(float).mean().tolist() if metadata_columns else []
    metadata_std = train_df[metadata_columns].astype(float).std().replace(0, 1).fillna(1).tolist() if metadata_columns else []

    tokenizer = CLIPTokenizer.from_pretrained(args.text_encoder)
    text_encoder = CLIPTextModel.from_pretrained(args.text_encoder).to(device)
    text_encoder.eval()
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)

    unet = UNet2DConditionModel(
        sample_size=args.image_size,
        in_channels=4 if use_mask else 3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=text_encoder.config.hidden_size,
    ).to(device)
    metadata_conditioner = None
    if use_metadata:
        metadata_conditioner = MetadataConditioner(
            input_dim=len(metadata_columns),
            hidden_size=text_encoder.config.hidden_size,
            num_tokens=args.metadata_tokens,
            dropout=args.metadata_dropout,
        ).to(device)
    ema_unet = copy.deepcopy(unet).eval() if args.ema else None
    if ema_unet is not None:
        for parameter in ema_unet.parameters():
            parameter.requires_grad_(False)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    dataset = DobiPromptDataset(
        train_df,
        args.image_size,
        args.prompt_dropout,
        use_mask=use_mask,
        mask_threshold=args.mask_threshold,
        metadata_columns=metadata_columns,
        metadata_mean=metadata_mean,
        metadata_std=metadata_std,
    )
    sampler = _build_sampler(train_df) if args.balanced_sampling else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    trainable_parameters = list(unet.parameters())
    if metadata_conditioner is not None:
        trainable_parameters += list(metadata_conditioner.parameters())
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir = _resolve_output_dir(args.output_dir, args.run_name, args.no_run_subdir)
    print(f"Writing training run to {output_dir}")
    config = vars(args) | {
        "device": str(device),
        "train_rows": int(len(train_df)),
        "train_positive": int(train_df["label"].sum()),
        "train_negative": int((train_df["label"] == 0).sum()),
        "steps_per_epoch": int(len(loader)),
        "use_mask": use_mask,
        "use_metadata": use_metadata,
        "metadata_columns": metadata_columns,
        "metadata_mean": metadata_mean,
        "metadata_std": metadata_std,
        "unet_in_channels": 4 if use_mask else 3,
        "resolved_output_dir": str(output_dir),
    }
    if args.swanlab and args.swanlab_experiment is None:
        args.swanlab_experiment = output_dir.name
        config["swanlab_experiment"] = args.swanlab_experiment
    (output_dir / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    run = _init_swanlab(args, config)

    global_step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            unet.train()
            losses: list[float] = []
            noise_losses: list[float] = []
            background_losses: list[float] = []
            progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
            for batch_index, batch in enumerate(progress, start=1):
                pixel_values = batch["pixel_values"].to(device)
                mask_values = batch["mask_values"].to(device)
                metadata_values = batch["metadata_values"].to(device)
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
                if metadata_conditioner is not None:
                    metadata_tokens = metadata_conditioner(metadata_values)
                    text_states = torch.cat([text_states, metadata_tokens], dim=1)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    model_input = torch.cat([noisy_images, mask_values], dim=1) if use_mask else noisy_images
                    prediction = unet(model_input, timesteps, encoder_hidden_states=text_states).sample
                    pixel_loss = F.mse_loss(prediction, noise, reduction="none")
                    if use_mask:
                        weights = (
                            1.0
                            + args.foreground_loss_weight * mask_values
                            + args.background_loss_weight * (1.0 - mask_values)
                        )
                        pixel_loss = pixel_loss * weights
                    sample_loss = pixel_loss.mean(dim=(1, 2, 3))
                    snr_weight = min_snr_weights(timesteps, scheduler.alphas_cumprod, args.min_snr_gamma)
                    noise_loss = (sample_loss * snr_weight).mean()
                    loss = noise_loss
                    bg_black_loss = torch.zeros((), device=device)
                    if use_mask and args.background_black_loss_weight > 0:
                        alpha = scheduler.alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)
                        predicted_x0 = (noisy_images - (1.0 - alpha).sqrt() * prediction) / alpha.sqrt().clamp(min=1e-8)
                        bg_black_loss = ((predicted_x0 * (1.0 - mask_values)) ** 2).mean()
                        loss = loss + args.background_black_loss_weight * bg_black_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if ema_unet is not None:
                    _update_ema(ema_unet, unet, args.ema_decay)

                global_step += 1
                step_loss = float(loss.detach().cpu())
                step_noise_loss = float(noise_loss.detach().cpu())
                step_background_loss = float(bg_black_loss.detach().cpu())
                losses.append(step_loss)
                noise_losses.append(step_noise_loss)
                background_losses.append(step_background_loss)
                epoch_loss = float(sum(losses) / len(losses))
                epoch_noise_loss = float(sum(noise_losses) / len(noise_losses))
                epoch_background_loss = float(sum(background_losses) / len(background_losses))
                lr = float(optimizer.param_groups[0]["lr"])
                progress.set_postfix(step_loss=f"{step_loss:.4f}", epoch_loss=f"{epoch_loss:.4f}", lr=f"{lr:.2e}")

                if global_step == 1 or global_step % args.log_every == 0:
                    log_data: dict[str, object] = {
                        "step/loss": step_loss,
                        "step/noise_loss": step_noise_loss,
                        "step/background_black_loss": step_background_loss,
                        "step/lr": lr,
                    }
                    _swanlab_log(run, log_data, step=global_step)

            final_epoch_loss = float(sum(losses) / max(len(losses), 1))
            final_epoch_noise_loss = float(sum(noise_losses) / max(len(noise_losses), 1))
            final_epoch_background_loss = float(sum(background_losses) / max(len(background_losses), 1))
            checkpoint_saved = epoch % args.save_every == 0 or epoch == args.epochs
            checkpoint_paths: dict[str, str] = {}
            if checkpoint_saved:
                checkpoint_paths = _save_training_state(
                    output_dir,
                    epoch,
                    global_step,
                    final_epoch_loss,
                    unet,
                    scheduler,
                    tokenizer,
                    text_encoder,
                    metadata_conditioner,
                    ema_unet,
                    optimizer,
                    scaler,
                    config,
                )
                tqdm.write(f"Saved checkpoint to {checkpoint_paths['checkpoint_path']}")

            _swanlab_log(
                run,
                {
                    "epoch/loss": final_epoch_loss,
                    "epoch/noise_loss": final_epoch_noise_loss,
                    "epoch/background_black_loss": final_epoch_background_loss,
                    "epoch/lr": float(optimizer.param_groups[0]["lr"]),
                },
                step=epoch,
            )

        if not (args.epochs % args.save_every == 0):
            checkpoint_paths = _save_training_state(
                output_dir,
                args.epochs,
                global_step,
                final_epoch_loss,
                unet,
                scheduler,
                tokenizer,
                text_encoder,
                metadata_conditioner,
                ema_unet,
                optimizer,
                scaler,
                config,
            )
            tqdm.write(f"Saved final checkpoint to {checkpoint_paths['checkpoint_path']}")
    finally:
        _swanlab_finish(run)


if __name__ == "__main__":
    main()
