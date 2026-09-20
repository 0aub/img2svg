"""How much of each pixel each colour actually owns.

A label map answers "which colour is this pixel" and throws away the rest. At an
edge that is most of the information: a pixel that is 70% honey and 30% page is
recorded as honey, and with it goes the only evidence of where the edge really
runs inside that pixel.

Recovering it is the two-colour matting problem, which is a poor way to *label*
a pixel - it can name a colour nowhere near the pixel's own - but exactly the
right way to measure one. Here it decides nothing; the labelling is already
settled. It only says how far across the pixel the boundary sits.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage


def neighbour_and_alpha(rgb: np.ndarray, labels: np.ndarray,
                        pal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For every pixel: the label it borders, and how much of it is its own.

    Returns ``(other, alpha)``. ``alpha`` is 1 away from any boundary.
    """
    P = pal.astype(np.float32)
    k = len(P)
    h, w = labels.shape

    counts = np.zeros((k, h, w), dtype=np.float32)
    for c in range(k):
        m = (labels == c).astype(np.float32)
        counts[c] = ndimage.uniform_filter(m, size=3, mode="nearest")
    own = np.take_along_axis(counts, labels[None].astype(np.intp), axis=0)[0]
    counts[labels, np.arange(h)[:, None], np.arange(w)[None, :]] = -1.0
    other = counts.argmax(axis=0).astype(np.int16)

    interior = own >= 0.999
    Pl = P[labels]
    Po = P[other]
    d = Pl - Po
    l2 = (d * d).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = ((rgb.astype(np.float32) - Po) * d).sum(-1) / l2
    a = np.where(np.isfinite(a), a, 1.0)
    alpha = np.where(interior, 1.0, np.clip(a, 0.0, 1.0)).astype(np.float32)
    return other, alpha


def field_for(classes, labels: np.ndarray, other: np.ndarray,
              alpha: np.ndarray) -> np.ndarray:
    """Coverage of one class or a set of them, in [0, 1].

    The per-pixel coverages partition, so the coverage of a union is the sum -
    which is what lets the silhouette and the stacked layers use the same
    machinery as a single colour.
    """
    want = np.asarray([classes] if np.isscalar(classes) else list(classes))
    f = np.zeros(labels.shape, dtype=np.float32)
    mine = np.isin(labels, want)
    f[mine] = alpha[mine]
    theirs = np.isin(other, want) & ~mine
    f[theirs] = 1.0 - alpha[theirs]
    return f
