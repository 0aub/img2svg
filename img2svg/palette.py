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
