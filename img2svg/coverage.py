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

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage


#: Colour separations, in RGB units, between which alpha stops being trusted.
#: Alpha is a projection onto the segment joining two palette colours, so its
#: error is the pixel noise divided by how far apart those colours are. Ordinary
#: encoding noise is a couple of units, so a boundary between colours 250 apart
#: - ink against the page - locates itself to a hundredth of a pixel, while one
#: between two tones 25 apart is guessing to within a tenth. Below LOW the
#: estimate is worth nothing on its own; above HIGH it needs no help.
TRUST_LOW, TRUST_HIGH = 25.0, 70.0

#: Averaging an untrusted field was tried and removed. It does clear the tearing,
#: and it also bends straight edges: on a geometric mark the wobble turns from
#: pixel-scale nicks into visible waves, which is the worse of the two. The two
#: cases cannot be told apart by separation either - the marks that tear and the
#: marks that must stay straight both have their main boundaries near 25 units.
#: `trust` is still reported, for a smoothing that respects straightness.


def neighbour_and_alpha(rgb: np.ndarray, labels: np.ndarray,
                        pal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every pixel: the label it borders, how much of it is its own, and
    how far that second number can be trusted.

    Returns ``(other, alpha, trust)``. ``alpha`` is 1 away from any boundary,
    and ``trust`` runs 0 to 1 with how far apart the two colours are.
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
    sep = np.sqrt(l2, dtype=np.float32)
    trust = np.clip((sep - TRUST_LOW) / (TRUST_HIGH - TRUST_LOW), 0.0, 1.0)
    return other, alpha, trust.astype(np.float32)


def field_for(classes, labels: np.ndarray, other: np.ndarray,
              alpha: np.ndarray, trust: Optional[np.ndarray] = None) -> np.ndarray:
    """Coverage of one class or a set of them, in [0, 1].

    The per-pixel coverages partition, so the coverage of a union is the sum -
    which is what lets the silhouette and the stacked layers use the same
    machinery as a single colour.

    ``trust`` is accepted and currently unused; see the note on TRUST_BLUR.
    """
    want = np.asarray([classes] if np.isscalar(classes) else list(classes))
    f = np.zeros(labels.shape, dtype=np.float32)

    # Both contributions are added, which matters only for a set: at a boundary
    # *inside* the set the pixel is split between two members and belongs to it
    # entirely. Taking just the pixel's own share there reported half coverage
    # along every internal seam, and wherever that fell below the isoline level
    # it cut a slit clean through the shape - visible as a hairline hole
    # straight through the artwork and the card behind it.
    mine = np.isin(labels, want)
    f[mine] += alpha[mine]
    theirs = np.isin(other, want)
    f[theirs] += 1.0 - alpha[theirs]
    return np.clip(f, 0.0, 1.0, out=f)
