"""Every knob in one place.

Lengths marked *scaled* are expressed for a 1024 px image and are multiplied by
``max(w, h) / 1024`` at run time, so the same defaults behave the same way on a
256 px sprite and a 4096 px poster.  Pass ``autoscale=False`` to switch that off.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass
class Config:
    # ---- palette ------------------------------------------------------
    max_colors: int = 16
    """Hard ceiling on palette entries."""
    min_sep: float = 18.0
    """Minimum RGB distance between two palette entries."""
    min_frac: float = 0.0015
    """Drop a palette entry holding less than this share of the flat pixels."""
    purity_tol: int = 10
    """A pixel is 'flat' when its 3x3 neighbourhood varies by no more than this."""
    palette: Optional[Sequence[str]] = None
    """Explicit palette as hex strings; skips extraction entirely."""

    # ---- segmentation -------------------------------------------------
    blur: float = 6.0
    """*scaled* Box radius used to smooth the label map. 0 disables smoothing."""
    blur_passes: int = 3
    """Box passes; 3 approximates a Gaussian."""
    protect_shrink: float = 0.55
    """Restore raw labels for any region that smoothing would shrink below this."""

    # ---- regions ------------------------------------------------------
    min_area: float = 120.0
    """*scaled* Discard regions smaller than this many pixels."""
    min_density: float = 0.08
    """Discard wispy regions whose area / bounding-box area is below this."""
    min_hole_area: float = 60.0
    """*scaled* Fill holes smaller than this instead of cutting them out."""

    # ---- curves -------------------------------------------------------
    rdp: float = 2.2
    """*scaled* Ramer-Douglas-Peucker tolerance, in pixels."""
    smooth_div: float = 45.0
    """Smoothing window = contour length / this."""
    smooth_min: int = 2
    """*scaled* Floor on the smoothing window."""
    smooth_max: int = 22
    """*scaled* Ceiling on the smoothing window."""
    corner_deg: float = 50.0
    """Turns sharper than this keep a hard corner instead of a smooth tangent."""
    keep_corners: str = "auto"
    """Hold corners back from smoothing: 'auto' (only when blur is off), 'on', 'off'.

    On hard-edged art - pixel sprites, UI, anything with real right angles - this
    is what stops a square coming back as a squircle.  On organic artwork it is
    actively harmful: the raw contour of a smooth diagonal is a staircase, and
    protecting those steps facets the curve.  'auto' keys off the blur radius,
    which is already the answer to "does this artwork have soft edges".
    """
    inset: float = 0.5
    """*scaled* Pull contours inward; 0.5 undoes the half-pixel of the pixel grid."""

    # ---- output -------------------------------------------------------
    background: str = "auto"
    """'auto', 'none', or a hex colour to treat as the background."""
    precision: int = 1
    """Decimal places kept in path data."""
    regularize: Tuple[str, ...] = ()
    """Shape snapping to apply. Currently supports 'container'."""
    mono: bool = False
    """Also write a single-path silhouette using currentColor."""
    mark: bool = False
    """Also write the artwork with its container removed."""
    container: str = "auto"
    """'auto' lifts a dominant background card out of the mark; 'keep' does not."""
    title: Optional[str] = None
    desc: Optional[str] = None

    autoscale: bool = True

    # ------------------------------------------------------------------
    SCALED = ("blur", "min_area", "min_hole_area", "rdp", "inset",
              "smooth_min", "smooth_max")

    def scaled(self, width: int, height: int) -> "Config":
        """Return a copy with *scaled* fields adjusted for this image size."""
        if not self.autoscale:
            return dataclasses.replace(self)
        k = max(width, height) / 1024.0
        out = dataclasses.replace(self)
        for name in self.SCALED:
            v = getattr(out, name) * (k * k if name in ("min_area", "min_hole_area") else k)
            if name == "smooth_min":
                v = max(0, int(v))
            elif name == "smooth_max":
                v = max(1, int(v))
            setattr(out, name, v)
        return out

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
