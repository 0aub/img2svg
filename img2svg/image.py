"""Loading, with a note about alpha.

A transparent PNG already tells us where the artwork stops, so we trust it and
skip background guessing entirely.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

RGB = np.ndarray


def load_full(path: str) -> Tuple[RGB, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return ``(rgb, opaque_mask, alpha)``; both extras are None without alpha."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        arr = np.asarray(im.convert("RGBA"))
        rgb, a = arr[..., :3].astype(np.uint8), arr[..., 3].astype(np.float32) / 255.0
        if (a < 0.98).any():
            return rgb.copy(), a >= 0.5, a
        return rgb.copy(), None, None
    return np.asarray(im.convert("RGB")).astype(np.uint8), None, None


def load(path: str) -> Tuple[RGB, Optional[np.ndarray]]:
    rgb, opaque, _ = load_full(path)
    return rgb, opaque


def composite(rgb: RGB, alpha: Optional[np.ndarray], background) -> RGB:
    """Flatten a transparent source onto the same colour the renderer used."""
    if alpha is None:
        return rgb
    bg = np.asarray(background, dtype=np.float32)
    a = alpha[..., None]
    return np.clip(rgb.astype(np.float32) * a + bg * (1 - a), 0, 255).astype(np.uint8)


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def save_png(path: str, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path, format="PNG", optimize=True)
