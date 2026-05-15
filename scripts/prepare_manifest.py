from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breastdiffusion.dataset import build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DOBI prompt manifest from labels.xlsx.")
    parser.add_argument("--labels", default="data/labels.xlsx", help="Path to labels.xlsx.")
    parser.add_argument("--image-dir", default="data/processed", help="Directory containing BMP images.")
    parser.add_argument("--output-dir", default="data/manifests", help="Directory for manifest CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args.labels, args.image_dir)
    manifest.to_csv(output_dir / "dobi_prompts.csv", index=False, encoding="utf-8-sig")
    for split, split_df in manifest.groupby("split"):
        split_df.to_csv(output_dir / f"{split}.csv", index=False, encoding="utf-8-sig")

    print(f"Saved {len(manifest)} rows to {output_dir / 'dobi_prompts.csv'}")
    print("Label distribution:")
    print(manifest.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print("\nPrompt examples:")
    for _, row in manifest.groupby("label").head(2).iterrows():
        print(f"- label={row['label']} file={row['filename']}: {row['prompt']}")


if __name__ == "__main__":
    main()
