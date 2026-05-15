from __future__ import annotations

from pathlib import Path

import pandas as pd

from .prompts import NEGATIVE_PROMPT, build_prompt


def load_labels(labels_path: str | Path) -> pd.DataFrame:
    labels_path = Path(labels_path)
    df = pd.read_excel(labels_path)
    required = {"filename", "label", "split", "side"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"labels file is missing required columns: {sorted(missing)}")
    df["label"] = df["label"].astype(int)
    df["split"] = df["split"].astype(str).str.lower()
    return df


def build_manifest(labels_path: str | Path, image_dir: str | Path) -> pd.DataFrame:
    image_dir = Path(image_dir)
    df = load_labels(labels_path).copy()
    df["image_path"] = df["filename"].map(lambda name: str(image_dir / f"{name}.bmp"))
    df["prompt"] = df.apply(build_prompt, axis=1)
    df["negative_prompt"] = NEGATIVE_PROMPT
    df["label_name"] = df["label"].map({0: "no_tumor", 1: "tumor"})
    missing_images = [path for path in df["image_path"] if not Path(path).exists()]
    if missing_images:
        examples = ", ".join(missing_images[:5])
        raise FileNotFoundError(f"{len(missing_images)} images listed in labels are missing, e.g. {examples}")
    return df
