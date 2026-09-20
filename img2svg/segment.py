"""Turn a noisy per-pixel labelling into regions you can actually draw.

Real illustrations are not perfectly flat.  Where the source fades one tone into
another over twenty pixels, per-pixel labelling produces a speckled mess, and
tracing that yields ragged, lumpy edges.

The fix is a smoothed majority vote: blur the one-hot label field and take the
argmax.  Two properties make this the right tool rather than a hack.  A straight
edge is unmoved - blurring a step symmetrically leaves the crossing exactly where
it was.  Only high-curvature and genuinely ambiguous boundaries move, which is
precisely the set we wanted to clean up.

The outer silhouette is held out of the vote.  The edge between artwork and page
is nearly always crisp in the source, and letting it blur would nibble away thin
drips and spikes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage


def box_blur(a: np.ndarray, radius: int, passes: int = 3) -> np.ndarray:
    """Separable box blur via cumulative sums; O(1) per pixel per pass."""
    if radius < 1 or passes < 1:
        return a
    out = a.astype(np.float32, copy=True)
    for _ in range(passes):
        for axis in (0, 1):
            out = _box1d(out, radius, axis)
    return out


def _box1d(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    n = a.shape[axis]
    r = min(r, max(n - 1, 0))
    if r < 1:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    ap = np.pad(a, pad, mode="edge")
    c = np.cumsum(ap, axis=axis, dtype=np.float32)
    zeros = np.zeros_like(np.take(c, [0], axis=axis))
    c = np.concatenate([zeros, c], axis=axis)
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / float(2 * r + 1)


def smooth_labels(
    labels: np.ndarray,
    k: int,
    radius: int,
    passes: int = 3,
    bg: Optional[int] = None,
    protect_shrink: float = 0.55,
) -> np.ndarray:
    """Smoothed majority vote over the content classes."""
    if radius < 1:
        return labels

    content = np.ones(labels.shape, dtype=bool) if bg is None else labels != bg
    if not content.any():
        return labels

    best_val = np.full(labels.shape, -1.0, dtype=np.float32)
    out = labels.copy()

    for c in range(k):
        if c == bg:
            continue
        m = ((labels == c) & content).astype(np.float32)
        if not m.any():
            continue
        b = box_blur(m, radius, passes)
        upd = content & (b > best_val)
        best_val[upd] = b[upd]
        out[upd] = c

    if protect_shrink > 0:
        _restore_shrunken(labels, out, k, bg, protect_shrink)
    return out


def _restore_shrunken(raw: np.ndarray, out: np.ndarray, k: int, bg, keep: float) -> None:
    """Put back any region the vote would have eaten.

    Small bright highlights are the usual casualty: they are narrow enough that a
    generous blur radius lets the surrounding tone outvote them.  Anything that
    loses more than ``1 - keep`` of itself is restored wholesale.
    """
    for c in range(k):
        if c == bg:
            continue
        m = raw == c
        if not m.any():
            continue
        lab, n = ndimage.label(m)
        if n == 0:
            continue
        area = np.bincount(lab.ravel(), minlength=n + 1)
        survived = np.bincount(lab[out == c].ravel(), minlength=n + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(area > 0, survived / np.maximum(area, 1), 1.0)
        doomed = np.flatnonzero((frac < keep) & (area > 0))
        doomed = doomed[doomed != 0]
        if len(doomed):
            out[np.isin(lab, doomed)] = c
