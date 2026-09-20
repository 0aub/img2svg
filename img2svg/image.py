"""Loading, with a note about alpha.

A transparent PNG already tells us where the artwork stops, so we trust it and
skip background guessing entirely.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image


def load(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return ``(rgb, opaque_mask)``; the mask is None for images without alpha."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        a = np.asarray(im)[..., 3]
        rgb = np.asarray(im)[..., :3].astype(np.uint8)
        if (a < 250).any():
            return rgb.copy(), a >= 128
        return rgb.copy(), None
    return np.asarray(im.convert("RGB")).astype(np.uint8), None


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
