"""Pixels to Beziers.

The chain is: walk the exact pixel boundary, smooth it, thin it with
Ramer-Douglas-Peucker, pull it half a pixel inward, then fit cubics with
tangent continuity except at detected corners.

The half-pixel inset is not cosmetic.  A boundary walk traces the *outside* of
the boundary pixels, but those pixels were chosen because their centres are
inside the shape, so the walk sits half a pixel proud all the way around.  Every
region comes out fat, and adjacent regions overlap by a full pixel.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Loop = List[Point]

# right-hand turn preference, keyed by the direction we arrived in, used to keep
# a consistent orientation where two regions touch only at a corner
_TURN = {
    (1, 0): ((0, -1), (1, 0), (0, 1), (-1, 0)),
    (0, 1): ((1, 0), (0, 1), (-1, 0), (0, -1)),
    (-1, 0): ((0, 1), (-1, 0), (0, -1), (1, 0)),
    (0, -1): ((-1, 0), (0, -1), (1, 0), (0, 1)),
}


def boundary_loops(mask: np.ndarray) -> List[Loop]:
    """Every closed boundary of ``mask``, material always on the right."""
    h, w = mask.shape
    m = mask
    up = np.ones_like(m); up[1:, :] = ~m[:-1, :]
    dn = np.ones_like(m); dn[:-1, :] = ~m[1:, :]
    lf = np.ones_like(m); lf[:, 1:] = ~m[:, :-1]
    rt = np.ones_like(m); rt[:, :-1] = ~m[:, 1:]

    edges: Dict[int, List[int]] = {}
    stride = w + 1

    def add(ys, xs, sx, sy, ex, ey):
        for y, x in zip(ys.tolist(), xs.tolist()):
            a = (y + sy) * stride + (x + sx)
            b = (y + ey) * stride + (x + ex)
            edges.setdefault(a, []).append(b)

    ys, xs = np.nonzero(m & up); add(ys, xs, 0, 0, 1, 0)
    ys, xs = np.nonzero(m & rt); add(ys, xs, 1, 0, 1, 1)
    ys, xs = np.nonzero(m & dn); add(ys, xs, 1, 1, 0, 1)
    ys, xs = np.nonzero(m & lf); add(ys, xs, 0, 1, 0, 0)

    loops: List[Loop] = []
    used = set()
    for start in list(edges):
        for slot in range(len(edges[start])):
            if (start, slot) in used:
                continue
            used.add((start, slot))
            loop = [start]
            prev, cur = start, edges[start][slot]
            guard = 0
            while cur != start:
                guard += 1
                if guard > 8 * (h + 2) * (w + 2):
                    break
                loop.append(cur)
                outs = edges.get(cur)
                if not outs:
                    break
                if len(outs) == 1:
                    pick = 0
                else:
                    d = ((cur % stride) - (prev % stride), (cur // stride) - (prev // stride))
                    pick = 0
                    for dx, dy in _TURN[d]:
                        cand = cur + dy * stride + dx
                        if cand in outs:
                            pick = outs.index(cand)
                            break
                if (cur, pick) in used:
                    break
                used.add((cur, pick))
                prev, cur = cur, outs[pick]
            if len(loop) >= 8:
                loops.append([(float(c % stride), float(c // stride)) for c in loop])
    return loops


def polygon_area(loop: Sequence[Point]) -> float:
    p = np.asarray(loop, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def smooth(loop: Sequence[Point], k: int) -> Loop:
    """Circular moving average; ``k`` points either side."""
    if k < 1:
        return list(loop)
    p = np.asarray(loop, dtype=np.float64)
    n = len(p)
    k = min(k, max(n // 2 - 1, 0))
    if k < 1:
        return list(loop)
    acc = np.zeros_like(p)
    for j in range(-k, k + 1):
        acc += np.roll(p, j, axis=0)
    acc /= (2 * k + 1)
    return [tuple(v) for v in acc]


def inset(loop: Sequence[Point], dist: float) -> Loop:
    """Shift every point toward the material, i.e. right of travel."""
    if abs(dist) < 1e-9:
        return list(loop)
    p = np.asarray(loop, dtype=np.float64)
    t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    norm = np.hypot(t[:, 0], t[:, 1])
    norm[norm < 1e-9] = 1.0
    tx, ty = t[:, 0] / norm, t[:, 1] / norm
    p[:, 0] += -ty * dist
    p[:, 1] += tx * dist
    return [tuple(v) for v in p]


def rdp(loop: Sequence[Point], eps: float) -> Loop:
    """Ramer-Douglas-Peucker on a closed loop, split at the two extreme points."""
    n = len(loop)
    if n < 4:
        return list(loop)
    p = np.asarray(loop, dtype=np.float64)
    a = int(np.argmin(p[:, 0] + p[:, 1]))
    b = int(np.argmax(p[:, 0] + p[:, 1]))
    if a == b:
        a, b = 0, n // 2
    if a > b:
        a, b = b, a
    first = _rdp_open(p[a : b + 1], eps)
    second = _rdp_open(np.vstack([p[b:], p[: a + 1]]), eps)
    out = first[:-1] + second[:-1]
    return out if len(out) >= 3 else [tuple(v) for v in p[:: max(1, n // 8)]]


def _rdp_open(pts: np.ndarray, eps: float) -> Loop:
    n = len(pts)
    if n < 3:
        return [tuple(v) for v in pts]
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        seg = b - a
        l2 = float(seg @ seg)
        rel = pts[i + 1 : j] - a
        if l2 < 1e-12:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            t = np.clip((rel @ seg) / l2, 0.0, 1.0)
            proj = t[:, None] * seg
            d = np.hypot(rel[:, 0] - proj[:, 0], rel[:, 1] - proj[:, 1])
        m = int(d.argmax())
        if d[m] > eps:
            m += i + 1
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return [tuple(v) for v in pts[keep]]


def to_bezier(pts: Sequence[Point], corner_deg: float = 50.0):
    """Catmull-Rom tangents, broken wherever the polyline turns sharply."""
    n = len(pts)
    p = np.asarray(pts, dtype=np.float64)
    nxt, prv = np.roll(p, -1, axis=0), np.roll(p, 1, axis=0)

    a_in = np.arctan2(p[:, 1] - prv[:, 1], p[:, 0] - prv[:, 0])
    a_out = np.arctan2(nxt[:, 1] - p[:, 1], nxt[:, 0] - p[:, 0])
    turn = np.degrees(np.abs(a_out - a_in)) % 360.0
    turn = np.minimum(turn, 360.0 - turn)
    corner = turn > corner_deg

    t_mid = (nxt - prv) / 2.0
    t_in = np.where(corner[:, None], p - prv, t_mid)
    t_out = np.where(corner[:, None], nxt - p, t_mid)

    segs = []
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = p[i], p[j]
        span = float(np.hypot(*(p1 - p0)))
        c1 = p0 + _cap(t_out[i], span) / 3.0
        c2 = p1 - _cap(t_in[j], span) / 3.0
        segs.append((tuple(c1), tuple(c2), tuple(p1)))
    return tuple(p[0]), segs


def _cap(t: np.ndarray, span: float) -> np.ndarray:
    m = float(np.hypot(*t))
    if m < 1e-9:
        return np.zeros(2)
    return t * (min(m, span) / m)


def fmt(v: float, prec: int = 1) -> str:
    s = f"{v:.{prec}f}".rstrip("0").rstrip(".") if prec else str(int(round(v)))
    return "0" if s in ("-0", "", "-") else s


def path_d(start: Point, segs, prec: int = 1) -> str:
    out = [f"M{fmt(start[0], prec)} {fmt(start[1], prec)}"]
    for c1, c2, p in segs:
        out.append(
            "C%s %s %s %s %s %s"
            % (fmt(c1[0], prec), fmt(c1[1], prec), fmt(c2[0], prec),
               fmt(c2[1], prec), fmt(p[0], prec), fmt(p[1], prec))
        )
    out.append("Z")
    return "".join(out)


def loop_to_path(loop: Sequence[Point], *, smooth_k: int, eps: float,
                 ins: float, corner_deg: float, prec: int) -> Tuple[str, int]:
    pts = smooth(loop, smooth_k)
    if ins:
        pts = inset(pts, ins)
    pts = rdp(pts, eps)
    if len(pts) < 3:
        return "", 0
    start, segs = to_bezier(pts, corner_deg)
    return path_d(start, segs, prec), len(segs)
