from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


PRESETS = {
    "text_only": {
        "condition_mode": "text",
        "min_snr_gamma": 0.0,
        "background_black_loss_weight": 0.0,
        "description": "Text-conditioned DDPM baseline.",
    },
    "text_mask": {
        "condition_mode": "text_mask",
        "min_snr_gamma": 0.0,
        "background_black_loss_weight": 0.05,
        "description": "Adds DOBI foreground mask condition and background black loss.",
    },
    "text_mask_metadata": {
        "condition_mode": "text_mask_metadata",
        "min_snr_gamma": 5.0,
        "background_black_loss_weight": 0.05,
        "description": "Main method: mask + numeric DOBI metadata + Min-SNR.",
    },
    "no_metadata": {
        "condition_mode": "text_mask",
        "min_snr_gamma": 5.0,
        "background_black_loss_weight": 0.05,
        "description": "Ablates numeric DOBI metadata while keeping mask and Min-SNR.",
    },
    "no_min_snr": {
        "condition_mode": "text_mask_metadata",
        "min_snr_gamma": 0.0,
        "background_black_loss_weight": 0.05,
        "description": "Ablates Min-SNR loss weighting.",
    },
    "no_bg_loss": {
        "condition_mode": "text_mask_metadata",
        "min_snr_gamma": 5.0,
        "background_black_loss_weight": 0.0,
        "description": "Ablates DOBI background black constraint.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or execute DOBI diffusion ablation experiments.")
    parser.add_argument("--manifest", default="data/manifests/dobi_prompts.csv")
    parser.add_argument("--output-root", default="outputs/ablations")
    parser.add_argument("--generated-root", default="outputs/ablation_generated")
    parser.add_argument("--reports-root", default="reports/ablations")
    parser.add_argument("--presets", nargs="+", default=list(PRESETS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-prompts", type=int, default=25)
    parser.add_argument("--num-per-prompt", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--stage", choices=["train", "generate", "evaluate", "classify", "all"], default="all")
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run commands. Without this flag, only writes the plan JSON.")
    return parser.parse_args()


def _python_command(script: str, *args: str) -> list[str]:
    return ["python", script, *args]


def build_plan(args: argparse.Namespace) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    for preset_name in args.presets:
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {sorted(PRESETS)}")
        preset = PRESETS[preset_name]
        model_dir = Path(args.output_root) / preset_name
        generated_dir = Path(args.generated_root) / preset_name
        reports_dir = Path(args.reports_root)

        train_cmd = _python_command(
            "scripts/train_text_to_image.py",
            "--manifest",
            args.manifest,
            "--output-dir",
            str(model_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--save-every",
            str(args.save_every),
            "--condition-mode",
            str(preset["condition_mode"]),
            "--min-snr-gamma",
            str(preset["min_snr_gamma"]),
            "--background-black-loss-weight",
            str(preset["background_black_loss_weight"]),
        )
        if args.swanlab:
            train_cmd += ["--swanlab", "--swanlab-experiment", f"ablation-{preset_name}"]

        generate_cmd = _python_command(
            "scripts/generate_images.py",
            "--model-dir",
            str(model_dir),
            "--prompt-csv",
            args.manifest,
            "--label",
            "1",
            "--num-prompts",
            str(args.num_prompts),
            "--num-per-prompt",
            str(args.num_per_prompt),
            "--steps",
            str(args.steps),
            "--guidance-scale",
            str(args.guidance_scale),
            "--output-dir",
            str(generated_dir),
            "--no-run-subdir",
        )
        if preset_name == "text_only":
            generate_cmd += ["--no-postprocess"]

        generated_manifest = generated_dir / "generated_manifest.csv"
        evaluate_cmd = _python_command(
            "scripts/evaluate_generation.py",
            "--real-manifest",
            args.manifest,
            "--generated-manifest",
            str(generated_manifest),
            "--output",
            str(reports_dir / f"{preset_name}_generation.json"),
        )
        classify_cmd = _python_command(
            "scripts/train_classifier.py",
            "--manifest",
            args.manifest,
            "--synthetic-manifest",
            str(generated_manifest),
            "--output",
            str(reports_dir / f"{preset_name}_classification.json"),
        )

        plan.append(
            {
                "preset": preset_name,
                "description": preset["description"],
                "model_dir": str(model_dir),
                "generated_dir": str(generated_dir),
                "commands": {
                    "train": train_cmd,
                    "generate": generate_cmd,
                    "evaluate": evaluate_cmd,
                    "classify": classify_cmd,
                },
            }
        )
    return plan


def execute_plan(plan: list[dict[str, object]], stage: str) -> None:
    stages = ["train", "generate", "evaluate", "classify"] if stage == "all" else [stage]
    for item in plan:
        for current_stage in stages:
            command = item["commands"][current_stage]
            print(f"\n[{item['preset']}:{current_stage}] {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    reports_root = Path(args.reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_path = reports_root / f"ablation_plan_{timestamp}.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved ablation plan to {plan_path}")
    for item in plan:
        print(f"\n# {item['preset']}: {item['description']}")
        for stage, command in item["commands"].items():
            print(f"{stage}: {' '.join(command)}")
    if args.execute:
        execute_plan(plan, args.stage)


if __name__ == "__main__":
    main()
