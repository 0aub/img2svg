"""Assign every pixel to a palette colour, un-mixing anti-aliasing as we go.

A pixel on the edge between two flat regions is a blend of those two colours.
Nearest-colour lookup sends it to whichever palette entry happens to sit closest
in RGB, which is often a *third* colour - that is where the dark fringe around
every shape in a naive trace comes from.

So instead of asking "which colour is this closest to", we ask "which *pair* of
colours, mixed in what proportion, explains this pixel", and hand the pixel to
whichever of the two owns more than half of it.  Edges land on the true 50%
boundary and no third colour gets invented.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .palette import hex_to_rgb

CHUNK = 1 << 21  # pixels per block, keeps peak memory near 100 MB


def matte(img: np.ndarray, pal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Label every pixel, and report how well the model fit it.

    Returns ``(labels, residual)`` where ``labels`` indexes ``pal`` and
    ``residual`` is the RGB distance left over - high values mean the pixel is
    not explained by any pair, i.e. a gradient or a colour we missed.
    """
    h, w, _ = img.shape
    flat = img.reshape(-1, 3).astype(np.float32)
    P = pal.astype(np.float32)
    k = len(P)

    labels = np.zeros(len(flat), dtype=np.int16)
    resid = np.full(len(flat), np.inf, dtype=np.float32)
    tol = _solid_tolerance(P)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]

    for start in range(0, len(flat), CHUNK):
        px = flat[start : start + CHUNK]

        # 1. nearest single colour, and how far off it is
        solo = np.full(len(px), np.inf, dtype=np.float32)
        solo_lab = np.zeros(len(px), dtype=np.int16)
        for i in range(k):
            r = ((px - P[i]) ** 2).sum(1)
            upd = r < solo
            solo[upd] = r[upd]
            solo_lab[upd] = i

        # 2. best two-colour mixture
        best = solo.copy()
        lab = solo_lab.copy()
        for i, j in pairs:
            d = P[i] - P[j]
            l2 = float(d @ d)
            if l2 < 1e-6:
                continue
            a = ((px - P[j]) @ d) / l2
            np.clip(a, 0.0, 1.0, out=a)
            r = ((px - (P[j] + a[:, None] * d)) ** 2).sum(1)
            upd = r < best
            if upd.any():
                best[upd] = r[upd]
                lab[upd] = np.where(a[upd] >= 0.5, i, j)

        # 3. A pixel that already *is* a palette colour keeps it. Two flat
        #    colours can always be mixed to land a hair closer to a third, and
        #    letting that win hands interior pixels to a colour they look
        #    nothing like - it repainted a dipper handle two shades too dark.
        solid = solo <= tol * tol
        lab[solid] = solo_lab[solid]
        best[solid] = solo[solid]

        labels[start : start + CHUNK] = lab
        resid[start : start + CHUNK] = np.sqrt(best)

    return labels.reshape(h, w), resid.reshape(h, w)


def _solid_tolerance(P: np.ndarray) -> float:
    """How close counts as "this pixel is that colour", from the palette itself.

    Half the distance to the nearest other entry: inside that radius no mixture
    of two other colours is a more honest explanation than the colour itself.
    """
    if len(P) < 2:
        return 1e9
    # float first: a palette is int16 by default and (255-0)**2 overflows it,
    # which turns the tolerance into NaN and silently disables the whole check
    Q = np.asarray(P, dtype=np.float64)
    d = np.sqrt(((Q[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min()) * 0.5


def detect_background(labels: np.ndarray, k: int, mode: str = "auto") -> Optional[int]:
    """Which palette index is the page behind the artwork, if any.

    Voting on a border ring sounds right and is wrong for the common case of a
    full-bleed card with rounded corners: the card owns most of every edge and
    outvotes the page that surrounds it.  The corners are the genuinely outermost
    points, so ask them first, and only accept the answer if that colour really
    does wrap the whole image.
    """
    if mode == "none":
        return None
    if mode != "auto":
        raise ValueError("resolve an explicit background colour before calling this")

    h, w = labels.shape
    corners = {labels[0, 0], labels[0, w - 1], labels[h - 1, 0], labels[h - 1, w - 1]}
    if len(corners) == 1:
        cand = int(corners.pop())
        if _wraps(labels == cand):
            return cand

    ring = np.concatenate(
        [
            labels[:2, :].ravel(),
            labels[-2:, :].ravel(),
            labels[:, :2].ravel(),
            labels[:, -2:].ravel(),
        ]
    )
    counts = np.bincount(ring, minlength=k)
    top = int(counts.argmax())
    return top if counts[top] >= 0.5 * len(ring) else None


def _wraps(mask: np.ndarray) -> bool:
    """True when ``mask`` reaches all four image edges and owns a real share of them.

    Deliberately not "one connected blob touches all four edges": a full-bleed
    card with rounded corners cuts the page into four disjoint triangles, and
    that page is still the background.  Requiring presence on every edge plus a
    minimum share is enough to rule out a stray region that merely grazes a
    corner.
    """
    h, w = mask.shape
    edges = (mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1])
    if not all(e.any() for e in edges):
        return False
    touched = sum(int(e.sum()) for e in edges)
    return touched >= 0.15 * (2 * h + 2 * w)


def resolve_background(pal: np.ndarray, labels: np.ndarray, mode: str) -> Optional[int]:
    if mode in ("auto", "none"):
        return detect_background(labels, len(pal), mode)
    want = hex_to_rgb(mode).astype(np.float32)
    d = ((pal.astype(np.float32) - want) ** 2).sum(1)
    return int(d.argmin())
