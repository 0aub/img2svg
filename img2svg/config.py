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
    blur: float = 4.0
    """*scaled* Box radius used to smooth the label map. 0 disables smoothing.

    Chosen by sweeping seven values against three images and then looking: 2 and
    4 tie on score, but at 2 a thin highlight inside a shadow still carries a
    dark edge, so 4 is the smallest value that is also clean.
    """
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
    tolerance: float = 0.6
    """*scaled* How far a fitted curve may sit from the traced boundary, in px.

    This is a *fitting* tolerance, not a simplification one. The curve is solved
    by least squares against every boundary point at once, so noise symmetric
    about the true edge cancels rather than being interpolated through. Under
    about 1.2 the fit follows pixel jitter; much over 2 it starts cutting
    corners off genuine detail.
    """
    presmooth: int = 3
    """*scaled* Half-width of a short average applied to the contour before fitting.

    Strictly better than not doing it: swept 0 to 8 across four images it is the
    only knob here that improves the score *and* more than halves the curve
    count. On a sub-pixel contour it is nearly free - the contour already sits
    within a tenth of a pixel of the true edge, so a short average barely moves
    it, while removing the ripple that would otherwise cost a cubic every few
    points.
    """
    corner_span: int = 7
    """*scaled* How far either side to look when deciding if a point is a corner."""
    corner_deg: float = 50.0
    """Turns sharper than this keep a hard corner instead of a smooth tangent."""
    inset: float = 0.0
    """*scaled* Pull every contour inward by this much.

    The boundary walk encloses exactly the pixels it traced, and averaged over
    sub-pixel phases that is already where the true edge is, so the default is
    zero. Swept over four values on six images, zero won every column.
    """
    overlap: float = 0.5
    """*scaled* Grow each detail layer outward so neighbours cannot leave a seam.

    Only used by ``layers='flat'``; stacked layers do not need it.

    Two regions traced separately abut exactly, and two abutting anti-aliased
    edges each contribute about half coverage, so whatever is underneath shows
    through as a hairline. When that underneath is a dark card it reads as an
    outline drawn around every shape.
    """
    auto_overlap: bool = True
    """Drop the overlap to 0 when the source has no anti-aliasing.

    Growing a shape half a pixel is invisible on a 1024px illustration and
    obvious on a 16px sprite feature; art without anti-aliasing also has far
    less seam to hide.
    """
    layers: str = "stacked"
    """'stacked' draws each colour over everything above it; 'flat' draws it once.

    Stacked is the only construction that puts every visible boundary where it
    belongs. Flat layers have to be grown slightly or they leave a hairline of
    whatever is underneath at every shared edge - and growing them biases every
    boundary outward by half a pixel. Since draw order runs largest first, the
    smaller darker regions land on top and win that half pixel, so the whole
    image acquires a dark halo along every edge: a systematic -0.27 L* on the
    example icon, visible as a red outline in a difference map.

    A layer that already covers everything drawn on top of it cannot leave a
    gap, so it needs no growing and every edge sits true. Bias drops to -0.02.
    The cost is path data, because each layer carries the union of the ones
    above it; 'flat' is there when size matters more than a half pixel.
    """

    # ---- output -------------------------------------------------------
    background: str = "auto"
    """'auto', 'none', or a hex colour to treat as the background."""
    precision: int = 1
    """Decimal places kept in path data."""
    regularize: Tuple[str, ...] = ()
    """Shape snapping to apply: 'container', 'circles', or 'all'."""
    circle_tolerance: float = 0.045
    """A loop within this fraction of its radius of being round becomes a circle."""
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
    SCALED = ("blur", "min_area", "min_hole_area", "tolerance", "inset",
              "overlap", "corner_span", "presmooth")

    def scaled(self, width: int, height: int) -> "Config":
        """Return a copy with *scaled* fields adjusted for this image size."""
        if not self.autoscale:
            return dataclasses.replace(self)
        k = max(width, height) / 1024.0
        out = dataclasses.replace(self)
        for name in self.SCALED:
            v = getattr(out, name) * (k * k if name in ("min_area", "min_hole_area") else k)
            if name in ("corner_span", "presmooth"):
                v = max(0 if name == "presmooth" else 2, int(round(v)))
            setattr(out, name, v)
        return out

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
