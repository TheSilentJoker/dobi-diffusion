from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path: str | Path, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    image = Image.open(path).convert("L").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def extract_handcrafted_features(path: str | Path) -> np.ndarray:
    image = load_gray(path)
    pixels = image.reshape(-1)
    gy, gx = np.gradient(image)
    grad_mag = np.sqrt(gx * gx + gy * gy).reshape(-1)
    hist, _ = np.histogram(pixels, bins=32, range=(0.0, 1.0), density=True)
    grad_hist, _ = np.histogram(grad_mag, bins=16, range=(0.0, max(float(grad_mag.max()), 1e-6)), density=True)
    stats = np.array(
        [
            pixels.mean(),
            pixels.std(),
            np.quantile(pixels, 0.05),
            np.quantile(pixels, 0.25),
            np.quantile(pixels, 0.50),
            np.quantile(pixels, 0.75),
            np.quantile(pixels, 0.95),
            grad_mag.mean(),
            grad_mag.std(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([pixels, hist.astype(np.float32), grad_hist.astype(np.float32), stats])


def extract_feature_matrix(paths: list[str]) -> np.ndarray:
    return np.vstack([extract_handcrafted_features(path) for path in paths]).astype(np.float32)
