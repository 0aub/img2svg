"""Assign every pixel to a palette colour, and notice when the palette can't.

An earlier version modelled each pixel as a mixture of *two* palette colours and
handed it to whichever owned more than half, on the theory that nearest-colour
lookup sends an anti-aliased edge pixel to whatever third colour happens to sit
between the two - the classic muddy fringe.

Measured against seven images it was worse everywhere, produced twice the speckle
and tore apart junctions where three regions meet, because a mixture can name a
colour arbitrarily far from the pixel's own: honey mixed into dark brown explains
a mid-brown edge nicely, and then alpha hands the pixel to the honey. Nearest
colour cannot do that - its error is bounded by colour distance - and the blurred
majority vote downstream removes the thin fringes it does leave.

So labelling is plain nearest colour. The mixture idea survives only as a
question we ask *about* the image: how many pixels no single colour explains.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from .palette import hex_to_rgb

CHUNK = 1 << 21  # pixels per block, keeps peak memory near 100 MB
SILHOUETTE_REACH = 2  # px around the page where un-mixing is trusted


def matte(img: np.ndarray, pal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Label every pixel, and report how far it sits from the colour it got.

    Returns ``(labels, residual)``. A residual larger than :func:`pair_margin`
    means the pixel is not any palette colour - an edge blend, a gradient, or a
    colour the palette missed.
    """
    h, w, _ = img.shape
    flat = img.reshape(-1, 3).astype(np.float32)
    P = pal.astype(np.float32)

    labels = np.empty(len(flat), dtype=np.int16)
    resid = np.empty(len(flat), dtype=np.float32)

    for start in range(0, len(flat), CHUNK):
        px = flat[start : start + CHUNK]
        best = np.full(len(px), np.inf, dtype=np.float32)
        lab = np.zeros(len(px), dtype=np.int16)
        for i in range(len(P)):
            r = ((px - P[i]) ** 2).sum(1)
            upd = r < best
            best[upd] = r[upd]
            lab[upd] = i
        labels[start : start + CHUNK] = lab
        resid[start : start + CHUNK] = np.sqrt(best)

    return labels.reshape(h, w), resid.reshape(h, w)


#: How much better the two-colour model must fit before a pixel is re-labelled,
#: as a share of :func:`pair_margin`. The mixture has a free parameter the plain
#: colour does not, so it fits better almost everywhere; without a margin this
#: reassigns half the image.
UNMIX_MARGIN = 0.6

#: How far to look for the colours a pixel might be a mixture *of*. Only colours
#: that are actually nearby are considered: a global search over every pair is
#: what made an earlier attempt paint the honey dipper's handle as a blend of
#: two colours that were nowhere near it.
UNMIX_REACH = 2


def unmix(img: np.ndarray, labels: np.ndarray, resid: np.ndarray,
          pal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decide, per pixel, between "this is colour C" and "this is A mixed with B".

    Nearest-colour labelling has no idea what a blend is. Where a dark facet
    meets the white page, the anti-aliased pixels between them run along the
    line joining those two colours - and if the palette holds a mid tone,
    because some other part of the artwork really is that colour, the whole
    transition gets labelled with it. The result is a one or two pixel fringe of
    a colour that is not there, along every edge of that kind, its outer
    boundary torn because which pixel lands where is decided by noise a couple
    of units wide. That fringe is what reads as spray.

    The fix is not to filter it afterwards. A fringe is not thin enough to catch
    by thickness - it joins a real region of the same colour - and voting it
    away takes the genuinely thin features with it, a globe's grid lines and a
    frame's shaded side. It has to be decided at labelling, on the evidence:

      * a fringe pixel lies almost exactly on the segment joining the two
        colours either side of it, and only approximately on the mid tone;
      * a real thin stroke lies on its own colour and on no such segment.

    So a pixel is re-labelled only when a mixture of two colours *present in its
    neighbourhood* beats its own colour by a clear margin - and it goes to
    whichever end of the mixture it is nearer, so a stroke keeps its middle.

    Returns ``(labels, mixed)``, where ``mixed`` marks the pixels re-labelled.
    """
    h, w = labels.shape
    k = len(pal)
    P = pal.astype(np.float32)
    margin = pair_margin(pal)
    confident = resid <= margin
    size = 2 * UNMIX_REACH + 1
    near = np.empty((k, h, w), dtype=bool)
    backed = np.zeros((k, h, w), dtype=bool)
    for c in range(k):
        mine = labels == c
        near[c] = ndimage.uniform_filter(mine.astype(np.float32),
                                         size=size, mode="nearest") > 0
        backed[c] = ndimage.uniform_filter((mine & confident).astype(np.float32),
                                           size=size, mode="nearest") > 0
    # Only pixels their own colour fails to explain, and only where that colour
    # has no unambiguous pixels nearby to vouch for it. A drawn stroke two
    # pixels wide still has some: its middle sits on its own colour. A fringe
    # has none anywhere along it, because it exists only in the transition.
    own_backed = np.take_along_axis(backed, labels[None].astype(np.intp), axis=0)[0]
    cand = ~confident & ~own_backed
    mixed = np.zeros_like(cand)
    if k < 3 or not cand.any():
        return labels, mixed

    idx = np.flatnonzero(cand.ravel())
    px = img.reshape(-1, 3).astype(np.float32)[idx]
    best = resid.ravel()[idx].astype(np.float32) - UNMIX_MARGIN * margin
    out = labels.ravel()[idx].copy()
    hit = np.zeros(len(idx), dtype=bool)
    flat_near = near.reshape(k, -1)[:, idx]

    for i in range(k):
        for j in range(i + 1, k):
            sel = flat_near[i] & flat_near[j]
            if not sel.any():
                continue
            d = P[j] - P[i]
            l2 = float(d @ d)
            if l2 < 1e-6:
                continue
            q = px[sel]
            a = np.clip(((q - P[i]) @ d) / l2, 0.0, 1.0)
            r = np.sqrt(((q - (P[i] + a[:, None] * d)) ** 2).sum(1))
            where = np.flatnonzero(sel)
            win = r < best[where]
            if not win.any():
                continue
            w_idx = where[win]
            best[w_idx] = r[win]
            out[w_idx] = np.where(a[win] > 0.5, j, i)
            hit[w_idx] = True

    new = labels.copy()
    new.ravel()[idx[hit]] = out[hit]
    mixed.ravel()[idx[hit]] = True
    return new, mixed


def silhouette(img: np.ndarray, pal: np.ndarray, bg: int) -> np.ndarray:
    """Where the artwork stops, decided by un-mixing the page colour back out.

    This is the one boundary where a mixture model earns its keep. Nearest
    colour puts a half-and-half white/plum pixel wherever the palette happens to
    be densest - often a mid-brown - so the silhouette picks up a one-pixel
    fringe and grows. Inside the artwork that costs a hairline nobody sees; here
    it is the outline of the whole mark, and it moved a fitted card corner by
    three pixels.

    Two things keep it from doing harm. Only one mixture is considered per
    pixel, page against its best partner, so the answer is always "page" or "not
    page" and none of the junction damage that made a general pair model
    unusable can happen. And the question is only asked of pixels that are both
    off-palette *and* next to the page: colour alone cannot tell a mid grey from
    half a dark grey on white, so asking it of an interior edge would punch
    holes through solid artwork.
    """
    P = pal.astype(np.float32)
    B = P[bg]
    margin = pair_margin(P)
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3).astype(np.float32)

    solo = np.full(len(flat), np.inf, dtype=np.float32)
    near = np.zeros(len(flat), dtype=np.int16)
    for i in range(len(P)):
        r = ((flat - P[i]) ** 2).sum(1)
        upd = r < solo
        solo[upd] = r[upd]
        near[upd] = i

    plain = (near != bg).reshape(h, w)
    unsure = (np.sqrt(solo) > margin).reshape(h, w)
    zone = ndimage.binary_dilation(~plain, iterations=SILHOUETTE_REACH) & unsure
    if not zone.any():
        return plain

    rel = flat - B
    best = np.full(len(flat), np.inf, dtype=np.float32)
    alpha = np.zeros(len(flat), dtype=np.float32)
    for i in range(len(P)):
        if i == bg:
            continue
        d = P[i] - B
        l2 = float(d @ d)
        if l2 < 1e-6:
            continue
        a = (rel @ d) / l2
        r = ((rel - np.clip(a, 0.0, 1.0)[:, None] * d) ** 2).sum(1)
        upd = r < best
        best[upd] = r[upd]
        alpha[upd] = a[upd]

    return np.where(zone, (alpha > 0.5).reshape(h, w), plain)


def pair_margin(P: np.ndarray) -> float:
    """How far from a palette colour still counts as being that colour.

    Scaled to the palette's own resolution, with a floor for encoding noise, so
    it behaves the same on a two-colour logo and a sixteen-colour illustration.
    """
    if len(P) < 2:
        return 1e9
    # float first: a palette is int16 by default and (255-0)**2 overflows it,
    # which turns the margin into NaN and silently disables everything using it
    Q = np.asarray(P, dtype=np.float64)
    d = np.sqrt(((Q[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return max(3.0, 0.15 * float(d.min()))


def blend_fraction(img: np.ndarray, pal: np.ndarray, sample: int = 300_000) -> float:
    """Share of pixels that no single palette colour accounts for.

    Anti-aliased artwork sits near 10% - that is its edges. Art drawn without
    anti-aliasing, pixel sprites and hard-edged vector exports, sits at exactly
    zero. Worth knowing, because the half-pixel of overlap that hides seams
    between neighbouring regions is visible bloat on art whose features are only
    a few pixels across.
    """
    P = np.asarray(pal, dtype=np.float32)
    flat = img.reshape(-1, 3).astype(np.float32)
    if len(flat) > sample:
        idx = np.random.default_rng(0).choice(len(flat), sample, replace=False)
        flat = flat[idx]
    solo = np.full(len(flat), np.inf, dtype=np.float32)
    for c in P:
        np.minimum(solo, ((flat - c) ** 2).sum(1), out=solo)
    return float((np.sqrt(solo) > pair_margin(P)).mean())


def detect_background(labels: np.ndarray, k: int, mode: str = "auto") -> Optional[int]:
    """Which palette index is the page behind the artwork, if any.

    Voting on a border ring sounds right and is wrong for the common case of a
    full-bleed card with rounded corners: the card owns most of every edge and
    outvotes the page that surrounds it. The corners are the genuinely outermost
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
    that page is still the background.
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
