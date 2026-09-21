"""Find the flat colours an illustration is actually made of.

The trick that makes this reliable on anti-aliased art: only cluster *flat*
pixels - ones whose 3x3 neighbourhood is uniform.  Edge pixels are blends of two
real colours and, if you let them vote, they pull the clusters toward colours
that were never in the artwork.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .config import Config


def hex_to_rgb(s: str) -> np.ndarray:
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"not a hex colour: {s!r}")
    return np.array([int(s[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.int16)


def rgb_to_hex(c: Sequence[float]) -> str:
    return "#%02X%02X%02X" % tuple(int(round(float(v))) for v in c)


def flat_mask(img: np.ndarray, tol: int = 10) -> np.ndarray:
    """True where the 3x3 neighbourhood varies by at most ``tol`` per channel."""
    a = img.astype(np.int16)
    stack = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            stack.append(np.roll(np.roll(a, dy, axis=0), dx, axis=1))
    s = np.stack(stack, axis=0)
    spread = s.max(axis=0) - s.min(axis=0)
    ok = (spread <= tol).all(axis=-1)
    ok[:1, :] = ok[-1:, :] = ok[:, :1] = ok[:, -1:] = False  # roll wraps around
    return ok


def _seed(samples: np.ndarray, cfg: Config) -> np.ndarray:
    """Pick well-separated peaks out of a coarse histogram.

    A flat colour never lands in one histogram bin - anti-aliasing and encoding
    noise smear it across a handful - so a candidate's weight is the sum of its
    own bin and the 26 around it.  Without that, a genuinely present colour can
    look too small to bother with and get dropped before k-means ever sees it.

    Seeding is deliberately generous; starved clusters are pruned after fitting,
    where the evidence is much better.
    """
    q = (samples // 4).astype(np.int64)
    key = (q[:, 0] << 20) | (q[:, 1] << 10) | q[:, 2]
    uniq, counts = np.unique(key, return_counts=True)
    table = dict(zip(uniq.tolist(), counts.tolist()))

    def weight(k: int) -> int:
        r, g, b = (k >> 20) & 0x3FF, (k >> 10) & 0x3FF, k & 0x3FF
        total = 0
        for dr in (-1, 0, 1):
            for dg in (-1, 0, 1):
                for db in (-1, 0, 1):
                    total += table.get(((r + dr) << 20) | ((g + dg) << 10) | (b + db), 0)
        return total

    total = len(samples)
    floor = max(16, cfg.min_frac * total * 0.1)
    order = np.argsort(-counts)
    seeds: List[np.ndarray] = []
    for i in order[: 4 * cfg.max_colors + 64]:
        k = int(uniq[i])
        if weight(k) < floor:
            continue
        c = np.array([(k >> 20) & 0x3FF, (k >> 10) & 0x3FF, k & 0x3FF], dtype=np.float64) * 4 + 2
        if all(np.linalg.norm(c - s) >= cfg.min_sep for s in seeds):
            seeds.append(c)
        if len(seeds) >= cfg.max_colors:
            break
    if not seeds:
        seeds = [samples.astype(np.float64).mean(axis=0)]
    return np.array(seeds)


def _lloyd(samples: np.ndarray, seeds: np.ndarray, iters: int = 16) -> np.ndarray:
    pts = samples.astype(np.float32)
    cen = seeds.astype(np.float32)
    for _ in range(iters):
        d = ((pts[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        moved = 0.0
        for k in range(len(cen)):
            sel = pts[lab == k]
            if len(sel):
                new = sel.mean(0)
                moved = max(moved, float(np.linalg.norm(new - cen[k])))
                cen[k] = new
        if moved < 0.25:
            break
    return cen


def _merge_close(samples: np.ndarray, cen: np.ndarray, cfg: Config) -> np.ndarray:
    """Hold min_sep on the *final* palette, not just on the seeds.

    Seeding keeps candidates apart, and then Lloyd undoes it: each centroid
    walks to the mean of the pixels it owns, and inside one smoothly shaded
    region two centroids happily converge to within a few units of each other.
    Nothing downstream can recover from that. The pixels between them get
    assigned on a difference smaller than the encoding noise, so the split
    lands wherever the noise falls and paints ragged blotches across what the
    artwork draws as a single fill.

    Measured on 55 unseen logo marks: 33 palettes came back holding a pair
    closer than min_sep, the closest 6.2 apart against a setting of 18.

    Merging is weighted by how many pixels each side owns, so the survivor sits
    where the bulk of the evidence is rather than halfway between.
    """
    while len(cen) > 1:
        d = ((cen[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(d.argmin(), d.shape)
        if d[i, j] >= cfg.min_sep ** 2:
            break
        lab = ((samples[:, None, :].astype(np.float32) - cen[None]) ** 2).sum(-1).argmin(1)
        wi, wj = int((lab == i).sum()), int((lab == j).sum())
        rest = [k for k in range(len(cen)) if k != i and k != j]
        merged = (cen[i] * wi + cen[j] * wj) / max(1, wi + wj)
        cen = _lloyd(samples, np.vstack([cen[rest], merged[None, :]]))
    return cen


def _unexplained(px: np.ndarray, cen: np.ndarray) -> np.ndarray:
    """Distance from each pixel to the nearest palette colour *or blend of two*.

    The second half matters: an anti-aliased edge pixel is a mixture of two
    colours that are already in the palette, so it is fully explained even
    though it matches neither. Without that, every edge in the image looks like
    a colour we missed.
    """
    P = cen.astype(np.float32)
    best = np.full(len(px), np.inf, dtype=np.float32)
    for c in P:
        np.minimum(best, ((px - c) ** 2).sum(1), out=best)
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            d = P[i] - P[j]
            l2 = float(d @ d)
            if l2 < 1e-6:
                continue
            a = np.clip(((px - P[j]) @ d) / l2, 0.0, 1.0)
            np.minimum(best, ((px - (P[j] + a[:, None] * d)) ** 2).sum(1), out=best)
    return np.sqrt(best)


def _add_missing(img: np.ndarray, cen: np.ndarray, cfg: Config) -> np.ndarray:
    """Recover colours that own no flat pixels at all.

    Clustering only flat pixels is what keeps anti-aliasing out of the palette,
    but it has a blind spot: a three-pixel stroke has essentially no interior,
    so its colour never appears in the sample. On line art that means the
    palette comes back as the background alone and the trace is empty - 24,000
    stroke pixels in one test image, 18 of them flat.

    So afterwards, ask what the palette still cannot account for, and if a real
    share of the image is unexplained, take its dominant colour and try again.
    """
    flat = img.reshape(-1, 3).astype(np.float32)
    if len(flat) > 200_000:
        idx = np.random.default_rng(1).choice(len(flat), 200_000, replace=False)
        flat = flat[idx]

    while len(cen) < cfg.max_colors:
        dist = _unexplained(flat, cen)
        bad = dist > cfg.missing_tol
        if bad.mean() < cfg.missing_share:
            break
        seeds = _seed(flat[bad].astype(np.int16), cfg)
        if not len(seeds):
            break
        new = seeds[0]
        if min(float(np.linalg.norm(new - c)) for c in cen) < cfg.min_sep:
            break
        cen = np.vstack([cen, new[None, :]])
    return cen


def extract(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Return the palette as an (K, 3) int16 array, most common colour first."""
    if cfg.palette:
        return np.stack([hex_to_rgb(c) for c in cfg.palette]).astype(np.int16)

    ok = flat_mask(img, cfg.purity_tol)
    samples = img[ok]
    if len(samples) < 64:  # nothing flat: fall back to the whole image
        samples = img.reshape(-1, 3)
    if len(samples) > 400_000:  # cap the work; the histogram is what matters
        idx = np.random.default_rng(0).choice(len(samples), 400_000, replace=False)
        samples = samples[idx]

    cen = _lloyd(samples, _seed(samples, cfg))

    # drop starved clusters, then re-fit so the survivors absorb their pixels
    for _ in range(4):
        d = ((samples[:, None, :].astype(np.float32) - cen[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        counts = np.bincount(lab, minlength=len(cen))
        keep = counts >= cfg.min_frac * len(samples)
        if keep.all() or keep.sum() == 0:
            break
        cen = _lloyd(samples, cen[keep])

    cen = _merge_close(samples, cen, cfg)
    cen = _add_missing(img, cen, cfg)

    d = ((samples[:, None, :].astype(np.float32) - cen[None]) ** 2).sum(-1)
    counts = np.bincount(d.argmin(1), minlength=len(cen))
    cen = cen[np.argsort(-counts)]
    return np.clip(np.rint(cen), 0, 255).astype(np.int16)


def describe(pal: np.ndarray, shares: Optional[np.ndarray] = None) -> str:
    lines = []
    for i, c in enumerate(pal):
        share = "" if shares is None else "  %6.2f%%" % (100 * shares[i])
        lines.append("  %2d  %s  (%3d,%3d,%3d)%s" % (i, rgb_to_hex(c), c[0], c[1], c[2], share))
    return "\n".join(lines)
