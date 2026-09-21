"""Group labelled pixels into drawable regions and turn them into path data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .config import Config
from . import curves

CORNER_OF_EXTENT = 0.06
"""How far to look for a corner, as a share of sqrt(area).

Features do not grow with the canvas - a 1400px logo can be built from 40px dots
- so the window has to be sized from the shape, not the image.
"""



@dataclass
class Region:
    label: int
    mask: np.ndarray
    area: int
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1 inclusive

    @property
    def density(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return self.area / float((x1 - x0 + 1) * (y1 - y0 + 1))


def _thickest(sub: np.ndarray) -> float:
    """Width of the widest part of a region: twice its largest inscribed circle."""
    return 2.0 * float(ndimage.distance_transform_edt(np.pad(sub, 1)).max())


def components(mask: np.ndarray, label: int, cfg: Config) -> List[Region]:
    """Connected components of ``mask``, minus the specks, wisps and seams.

    Density screening matters more than it sounds: a run of stray pixels along an
    edge can add up to a respectable area while covering a huge bounding box, and
    tracing one produces a spindly shard that belongs to nothing.

    Thickness catches what density cannot. A sliver hugging one side of a stroke
    fills its own bounding box, so its density is 1.0, but it is two pixels wide
    and nobody drew it - it is where the quantiser cut a soft edge.
    """
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    out: List[Region] = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        sub = lab[sl] == i
        area = int(sub.sum())
        if area < cfg.min_area:
            continue
        ys, xs = sl
        box = (xs.start, ys.start, xs.stop - 1, ys.stop - 1)
        r = Region(label, np.zeros(mask.shape, dtype=bool), area, box)
        if r.density < cfg.min_density:
            continue
        if cfg.min_thickness and _thickest(sub) < cfg.min_thickness:
            continue
        r.mask[sl] = sub
        out.append(r)
    out.sort(key=lambda r: -r.area)
    return out


def trace_field(field: np.ndarray, cfg: Config, ins: float) -> Tuple[str, int]:
    """Isoline of a coverage field, fitted to cubics."""
    from . import isoline

    loops = isoline.contours(field, 0.5)
    if not loops:
        return "", 0
    parts: List[str] = []
    total = 0
    for loop in loops:
        a = abs(curves.polygon_area(loop))
        if a < cfg.min_hole_area:
            continue
        extent = math.sqrt(a)
        if "circles" in cfg.regularize:
            from . import regularize

            got = regularize.as_circle(loop, cfg.circle_tolerance)
            if got is not None:
                cx, cy, r = got
                parts.append(regularize.circle_path(cx, cy, r - ins, cfg.precision))
                total += 4
                continue
        span = int(max(2, min(cfg.corner_span, extent * CORNER_OF_EXTENT)))
        d, n = curves.loop_to_path(
            loop, tolerance=cfg.tolerance, ins=ins, corner_deg=cfg.corner_deg,
            corner_span=span, prec=cfg.precision, presmooth=cfg.presmooth,
        )
        if d:
            parts.append(d)
            total += n
    return "".join(parts), total


def region_path(mask: np.ndarray, cfg: Config, ins: float) -> Tuple[str, int]:
    """Path data for one region: outer loops plus any holes big enough to keep.

    The smoothing window is capped by the size of the shape being drawn, not the
    size of the canvas, since features do not grow with the image. The cap only
    ever tightens the global setting.
    """
    loops = curves.boundary_loops(mask)
    if not loops:
        return "", 0
    parts: List[str] = []
    total = 0
    for loop in loops:
        a = abs(curves.polygon_area(loop))
        if a < cfg.min_hole_area:
            continue
        extent = math.sqrt(a)
        if "circles" in cfg.regularize:
            from . import regularize

            got = regularize.as_circle(loop, cfg.circle_tolerance)
            if got is not None:
                cx, cy, r = got
                parts.append(regularize.circle_path(cx, cy, r - ins, cfg.precision))
                total += 4
                continue
        span = int(max(2, min(cfg.corner_span, extent * CORNER_OF_EXTENT)))
        d, n = curves.loop_to_path(
            loop, tolerance=cfg.tolerance, ins=ins, corner_deg=cfg.corner_deg,
            corner_span=span, prec=cfg.precision, presmooth=cfg.presmooth,
        )
        if d:
            parts.append(d)
            total += n
    return "".join(parts), total


def mask_path(mask: np.ndarray, cfg: Config, ins: float,
              field: Optional[np.ndarray] = None) -> Tuple[str, int, int]:
    """Path data for every surviving region of a mask.

    Region selection still happens on the crisp mask - that is a question about
    which blobs are worth drawing - but the outlines come from the coverage
    field, restricted to a dilated copy of each region so its neighbours cannot
    pull the isoline around.
    """
    regs = components(mask, -1, cfg)
    parts, total = [], 0
    for r in regs:
        if field is None:
            d, n = region_path(r.mask, cfg, ins)
        else:
            near = ndimage.binary_dilation(r.mask, iterations=2)
            d, n = trace_field(np.where(near, field, 0.0), cfg, ins)
        if d:
            parts.append(d)
            total += n
    return "".join(parts), total, len(regs)


def class_path(labels: np.ndarray, cls: int, cfg: Config, ins: float,
               field: Optional[np.ndarray] = None) -> Tuple[str, int, int]:
    """Path data covering every surviving region of one palette class."""
    return mask_path(labels == cls, cfg, ins, field)


def content_mask(labels: np.ndarray, bg) -> np.ndarray:
    return np.ones(labels.shape, dtype=bool) if bg is None else labels != bg
