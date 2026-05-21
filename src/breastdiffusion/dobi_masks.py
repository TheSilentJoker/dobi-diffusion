from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def resize_pad_gray(path: str | Path, image_size: int) -> Image.Image:
    image = Image.open(path).convert("L")
    image = ImageOps.contain(image, (image_size, image_size), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (image_size, image_size), color=0)
    offset = ((image_size - image.width) // 2, (image_size - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def image_to_tensor_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def extract_foreground_mask(
    image: Image.Image,
    threshold: int = 12,
    smooth: bool = True,
) -> Image.Image:
    mask = image.convert("L").point(lambda pixel: 255 if pixel > threshold else 0)
    if smooth:
        mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5)).filter(ImageFilter.GaussianBlur(1.0))
        mask = mask.point(lambda pixel: 255 if pixel > 64 else 0)
    return mask.convert("L")


def canonical_dobi_mask(image_size: int = 128, original_height: int = 102) -> Image.Image:
    mask = Image.new("L", (image_size, image_size), color=0)
    array = np.zeros((image_size, image_size), dtype=np.uint8)
    top = max((image_size - original_height) // 2, 0)
    bottom = min(top + original_height, image_size)
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    cx = (image_size - 1) / 2.0
    cy = bottom - 1.0
    rx = image_size * 0.43
    ry = original_height * 0.46
    ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    ellipse &= yy >= top + original_height * 0.28
    array[ellipse] = 255
    return Image.fromarray(array, mode="L")


def image_and_mask_tensors(
    path: str | Path,
    image_size: int,
    threshold: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    image = resize_pad_gray(path, image_size)
    mask = extract_foreground_mask(image, threshold=threshold)
    image_array = image_to_tensor_array(image)
    mask_array = image_to_tensor_array(mask)
    return image_array, mask_array
