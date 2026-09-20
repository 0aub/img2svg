"""Search a few parameters against the source image.

A word of warning that is easy to skip: the objective here is *fidelity*, and
fidelity is not the same thing as a good result.  Snapping a wonky container to a
true superellipse makes this score worse.  Flattening a gradient into one honest
hard edge can score worse than a ragged boundary that happens to straddle it.
Use the number to catch regressions, not to pick the design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .pipeline import convert
from . import image, raster, verify


@dataclass
class Trial:
    params: Dict
    mean_de: float
    segments: int
    score: float


AXES = {
    "blur": [0.0, 2.0, 4.0, 6.0, 8.0, 11.0],
    "rdp": [1.2, 1.8, 2.2, 3.0, 4.0],
    "smooth_div": [25.0, 45.0, 70.0, 120.0],
}


def score(mean_de: float, segments: int, weight: float) -> float:
    return mean_de + weight * segments / 1000.0


def run(rgb: np.ndarray, cfg: Config, opaque=None, budget: int = 18,
        curve_weight: float = 0.5, mask: Optional[np.ndarray] = None,
        log=None, alpha=None) -> Tuple[Config, List[Trial]]:
    """Coordinate descent, one axis at a time, best-so-far kept."""
    trials: List[Trial] = []
    cache: Dict[Tuple, Trial] = {}
    h, w = rgb.shape[:2]

    def evaluate(c: Config) -> Trial:
        key = (c.blur, c.rdp, c.smooth_div)
        if key in cache:
            return cache[key]
        res = convert(rgb, c, opaque)
        out = raster.render(res.svg, w, h, background=res.background_rgb)
        st = verify.compare(image.composite(rgb, alpha, res.background_rgb), out, mask=mask)
        t = Trial({"blur": c.blur, "rdp": c.rdp, "smooth_div": c.smooth_div},
                  st["mean"], res.segments, score(st["mean"], res.segments, curve_weight))
        cache[key] = t
        trials.append(t)
        if log:
            log("  blur=%-5.1f rdp=%-4.1f smooth=%-5.0f -> dE %.3f  curves %4d  score %.3f"
                % (c.blur, c.rdp, c.smooth_div, t.mean_de, t.segments, t.score))
        return t

    best_cfg = cfg
    best = evaluate(best_cfg)
    spent = 1
    for axis, values in AXES.items():
        for v in values:
            if spent >= budget:
                break
            cand = best_cfg.replace(**{axis: v})
            if (cand.blur, cand.rdp, cand.smooth_div) in cache:
                continue
            t = evaluate(cand)
            spent += 1
            if t.score < best.score:
                best, best_cfg = t, cand
    return best_cfg, trials
