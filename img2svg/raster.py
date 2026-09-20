"""Render an SVG back to pixels so the result can be checked against the source.

resvg is preferred when it is on PATH - it is the most spec-faithful of the fast
renderers - with cairosvg as the portable fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np
from PIL import Image


def backend() -> Optional[str]:
    if os.environ.get("IMG2SVG_RENDERER") == "cairosvg":
        return "cairosvg" if _has_cairo() else None
    if shutil.which("resvg"):
        return "resvg"
    return "cairosvg" if _has_cairo() else None


def _has_cairo() -> bool:
    try:
        import cairosvg  # noqa: F401
        return True
    except Exception:
        return False


class RendererMissing(RuntimeError):
    pass


def render(svg_text: str, width: int, height: int, background=(255, 255, 255)) -> np.ndarray:
    be = backend()
    if be is None:
        raise RendererMissing(
            "no SVG renderer available - install the 'render' extra (cairosvg) "
            "or put resvg on PATH"
        )
    png = _resvg(svg_text, width, height) if be == "resvg" else _cairo(svg_text, width, height)
    im = Image.open(png).convert("RGBA") if hasattr(png, "read") else Image.frombytes(*png)
    arr = np.asarray(im).astype(np.float32)
    rgb, a = arr[..., :3], arr[..., 3:4] / 255.0
    if background is None:
        return rgb.astype(np.uint8)
    bg = np.array(background, dtype=np.float32)
    return np.clip(rgb * a + bg * (1 - a), 0, 255).astype(np.uint8)


def _cairo(svg_text: str, w: int, h: int):
    import io
    import cairosvg

    data = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"),
                            output_width=w, output_height=h)
    return io.BytesIO(data)


def _resvg(svg_text: str, w: int, h: int):
    import io

    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "a.svg"), os.path.join(td, "a.png")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(svg_text)
        subprocess.run(
            ["resvg", src, dst, "--width", str(w), "--height", str(h)],
            check=True, capture_output=True,
        )
        with open(dst, "rb") as fh:
            return io.BytesIO(fh.read())
