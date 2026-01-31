from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PadInfo:
    left: int
    top: int
    size: int  # padded image is size x size


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_image(image_path: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    return img


def save_image(image: Image.Image, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    image.save(path)


def pil_to_numpy_rgb(image: Image.Image) -> np.ndarray:
    arr = np.array(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape: {arr.shape}")
    return arr


def mask_to_pil_l(mask: np.ndarray) -> Image.Image:
    """Convert HxW bool/0-1/0-255 mask to PIL 'L' (0..255)."""
    if mask.ndim != 2:
        raise ValueError(f"Expected HxW mask, got shape: {mask.shape}")

    if mask.dtype == np.bool_:
        mask_u8 = (mask.astype(np.uint8) * 255)
    else:
        mask_f = mask.astype(np.float32)
        if mask_f.max() <= 1.0:
            mask_u8 = np.clip(mask_f * 255.0, 0, 255).astype(np.uint8)
        else:
            mask_u8 = np.clip(mask_f, 0, 255).astype(np.uint8)

    return Image.fromarray(mask_u8, mode="L")


def save_mask(mask: np.ndarray, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    mask_to_pil_l(mask).save(path)


def pad_to_square(image: Image.Image, fill: int = 0) -> Tuple[Image.Image, PadInfo]:
    """Pad image to a square canvas (size = max(W,H)), keeping content centered."""
    w, h = image.size
    size = max(w, h)

    left = (size - w) // 2
    top = (size - h) // 2

    canvas = Image.new("RGB", (size, size), color=(fill, fill, fill))
    canvas.paste(image, (left, top))

    return canvas, PadInfo(left=left, top=top, size=size)


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))
