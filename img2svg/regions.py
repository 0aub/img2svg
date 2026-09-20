"""Group labelled pixels into drawable regions and turn them into path data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import ndimage

from .config import Config
from . import curves


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


def components(mask: np.ndarray, label: int, cfg: Config) -> List[Region]:
    """Connected components of ``mask``, minus the specks and the wisps.

    Density screening matters more than it sounds: a run of stray pixels along an
    edge can add up to a respectable area while covering a huge bounding box, and
    tracing one produces a spindly shard that belongs to nothing.
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
        r.mask[sl] = sub
        out.append(r)
    out.sort(key=lambda r: -r.area)
    return out


def region_path(mask: np.ndarray, cfg: Config, ins: float) -> Tuple[str, int]:
    """Path data for one region: outer loops plus any holes big enough to keep."""
    loops = curves.boundary_loops(mask)
    if not loops:
        return "", 0
    parts: List[str] = []
    total = 0
    for loop in loops:
        a = abs(curves.polygon_area(loop))
        if a < cfg.min_hole_area:
            continue
        k = int(max(cfg.smooth_min, min(cfg.smooth_max, round(len(loop) / cfg.smooth_div))))
        d, n = curves.loop_to_path(
            loop, smooth_k=k, eps=cfg.rdp, ins=ins,
            corner_deg=cfg.corner_deg, prec=cfg.precision,
        )
        if d:
            parts.append(d)
            total += n
    return "".join(parts), total


def class_path(labels: np.ndarray, cls: int, cfg: Config, ins: float) -> Tuple[str, int, int]:
    """Path data covering every surviving region of one palette class."""
    regs = components(labels == cls, cls, cfg)
    parts, total = [], 0
    for r in regs:
        d, n = region_path(r.mask, cfg, ins)
        if d:
            parts.append(d)
            total += n
    return "".join(parts), total, len(regs)


def content_mask(labels: np.ndarray, bg) -> np.ndarray:
    return np.ones(labels.shape, dtype=bool) if bg is None else labels != bg
