"""Least-squares cubic fitting, after Schneider (Graphics Gems, 1990).

The difference between a traced curve and a drawn one.

Interpolating a curve *through* simplified contour points locks every one of them
in place, wobble included: a threshold boundary jitters by half a pixel, the
simplifier keeps the jitter peaks because they are the extreme points, and the
interpolant dutifully passes through all of them. The result measures well - the
error never exceeds half a pixel anywhere - and looks like a blob, because a
circle that is within half a pixel of round in a hundred independent places is
visibly not round.

Fitting asks a different question: what single cubic comes closest to *all* of
these points at once. Noise that is symmetric about the true edge cancels instead
of accumulating, so the curve comes out smooth and lands closer to the real edge
than any of the points it was fitted to.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Cubic = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

MAX_REPARAM = 6
MAX_DEPTH = 24


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-12 else np.zeros(2)


def _bezier(b: Cubic, t: float) -> np.ndarray:
    m = 1.0 - t
    return (m * m * m * b[0] + 3 * m * m * t * b[1]
            + 3 * m * t * t * b[2] + t * t * t * b[3])


def _bezier_prime(b: Cubic, t: float) -> np.ndarray:
    m = 1.0 - t
    return 3 * m * m * (b[1] - b[0]) + 6 * m * t * (b[2] - b[1]) + 3 * t * t * (b[3] - b[2])


def _bezier_prime2(b: Cubic, t: float) -> np.ndarray:
    m = 1.0 - t
    return 6 * m * (b[2] - 2 * b[1] + b[0]) + 6 * t * (b[3] - 2 * b[2] + b[1])


def _chord_params(pts: np.ndarray) -> np.ndarray:
    d = np.hypot(*(pts[1:] - pts[:-1]).T)
    u = np.concatenate([[0.0], np.cumsum(d)])
    return u / u[-1] if u[-1] > 1e-12 else np.linspace(0, 1, len(pts))


def _generate(pts: np.ndarray, u: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> Cubic:
    """Solve for the two control-point distances that minimise squared error."""
    p0, p3 = pts[0], pts[-1]
    m = 1.0 - u
    b0 = m ** 3
    b1 = 3 * m * m * u
    b2 = 3 * m * u * u
    b3 = u ** 3

    a1 = t1[None, :] * b1[:, None]
    a2 = t2[None, :] * b2[:, None]
    c11 = float((a1 * a1).sum())
    c12 = float((a1 * a2).sum())
    c22 = float((a2 * a2).sum())
    tmp = pts - (p0 * (b0 + b1)[:, None] + p3 * (b2 + b3)[:, None])
    x1 = float((a1 * tmp).sum())
    x2 = float((a2 * tmp).sum())

    det = c11 * c22 - c12 * c12
    if abs(det) > 1e-12:
        alpha1 = (x1 * c22 - c12 * x2) / det
        alpha2 = (c11 * x2 - x1 * c12) / det
    else:
        alpha1 = alpha2 = 0.0

    seg = float(np.hypot(*(p3 - p0)))
    if alpha1 < 1e-6 or alpha2 < 1e-6:          # degenerate: fall back to Wu/Barsky
        alpha1 = alpha2 = seg / 3.0
    # a control arm longer than the chord folds the curve back on itself
    alpha1 = min(alpha1, seg * 1.5)
    alpha2 = min(alpha2, seg * 1.5)
    return (p0, p0 + t1 * alpha1, p3 + t2 * alpha2, p3)


def _max_error(pts: np.ndarray, b: Cubic, u: np.ndarray) -> Tuple[float, int]:
    worst, idx = 0.0, len(pts) // 2
    for i in range(1, len(pts) - 1):
        d = _bezier(b, u[i]) - pts[i]
        e = float(d @ d)
        if e > worst:
            worst, idx = e, i
    return math.sqrt(worst), idx


def _reparameterize(pts: np.ndarray, u: np.ndarray, b: Cubic) -> np.ndarray:
    """One Newton-Raphson step toward each point's true nearest parameter."""
    out = u.copy()
    for i in range(len(pts)):
        d = _bezier(b, u[i]) - pts[i]
        d1 = _bezier_prime(b, u[i])
        d2 = _bezier_prime2(b, u[i])
        den = float(d1 @ d1 + d @ d2)
        if abs(den) > 1e-12:
            out[i] = u[i] - float(d @ d1) / den
    return np.clip(out, 0.0, 1.0)


#: How far a run may stray from the straight segment that would replace it.
#: Beyond this the line is visibly in the wrong place however little it bends.
LINE_MAX = 0.8


def _bend(pts: np.ndarray) -> Tuple[float, float]:
    """How far the run bows, and how sure we are that it bows at all.

    Fits a parabola in the frame of the chord: the sagitta is the bend, and the
    quadratic coefficient's standard error says whether that bend is real or is
    the noise arranging itself. Returns ``(sagitta, significance)``.

    Both halves are needed. A threshold on the largest deviation cannot separate
    a straight edge that wobbles from an edge that genuinely curves - they reach
    the same peak. Least squares cancels the noise, which leaves the bend; but a
    short noisy run can still fit a small parabola by chance, and only its
    standard error says so.

    Measuring against the chord alone is not enough either: the chord is drawn
    between two noisy endpoints, so it arrives tilted and reports a bow that is
    not there.
    """
    a, b = pts[0], pts[-1]
    d = b - a
    L = float(np.hypot(d[0], d[1]))
    if L < 1e-9:
        return float("inf"), float("inf")
    if len(pts) < 5:
        return 0.0, 0.0
    u = d / L
    v = np.array([-u[1], u[0]])
    x = (pts - a) @ u
    y = (pts - a) @ v
    A = np.stack([np.ones_like(x), x, x * x], 1)
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        cov = np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return float("inf"), float("inf")
    resid = y - A @ coef
    dof = max(len(pts) - 3, 1)
    var = float(resid @ resid) / dof
    se = math.sqrt(max(var * float(cov[2, 2]), 1e-24))
    return abs(float(coef[2])) * L * L / 4.0, abs(float(coef[2])) / se


def _off_chord(pts: np.ndarray) -> float:
    """Furthest a point strays from the segment that would replace the run."""
    a, b = pts[0], pts[-1]
    d = b - a
    n = float(np.hypot(d[0], d[1]))
    if n < 1e-12:
        return float(np.hypot(*(pts - a).T).max())
    u = np.array([-d[1], d[0]]) / n
    return float(np.abs((pts - a) @ u).max())


def _as_line(pts: np.ndarray) -> Cubic:
    """A cubic that is exactly a straight segment, so it renders dead straight."""
    d = (pts[-1] - pts[0]) / 3.0
    return (pts[0], pts[0] + d, pts[-1] - d, pts[-1])


#: How many standard errors the bend must clear before it counts as a bend.
LINE_SIGMA = 3.0

#: ...and how many points it takes before that argument is worth making. On a
#: short run the standard error is wide enough to swallow a real bend, so a
#: quarter of a small circle would come back as a chord. Below this, a run is
#: only straight if it is straight outright.
LINE_MIN_PTS = 9


def _is_line(pts: np.ndarray, error: float) -> bool:
    """Prefer the simplest shape that fits.

    A cubic fitted to a noisy straight run always comes back as a shallow S: it
    has four control points and no reason to keep them collinear. Logos are
    mostly straight edges, so every one of them arrives gently bowed and the
    artwork looks hand-wobbled - bars that undulate, strokes that thicken and
    thin, a polygon whose facets have all gone soft. Test for a line first and
    the question never arises.
    """
    if _off_chord(pts) > LINE_MAX:
        return False
    sagitta, sigma = _bend(pts)
    if sagitta <= error:
        return True
    return len(pts) >= LINE_MIN_PTS and sigma < LINE_SIGMA


def fit_run(pts: np.ndarray, t1: np.ndarray, t2: np.ndarray,
            error: float, depth: int = 0) -> List[Cubic]:
    """Fit one open run of points, splitting only where a cubic cannot reach."""
    if len(pts) < 2:
        return []
    if len(pts) == 2:
        d = float(np.hypot(*(pts[1] - pts[0]))) / 3.0
        return [(pts[0], pts[0] + t1 * d, pts[1] + t2 * d, pts[1])]

    if _is_line(pts, error):
        return [_as_line(pts)]

    u = _chord_params(pts)
    b = _generate(pts, u, t1, t2)
    err, split = _max_error(pts, b, u)

    # Always refine before giving up. Chord length is only a guess at each
    # point's true parameter, and the error it reports is measured at that
    # guess, so an excellent fit can look like a failing one and get split in
    # half for no reason. Newton-Raphson on the parameters costs a few passes
    # and took a quarter-arc fit from 0.45px to 0.018px.
    for _ in range(MAX_REPARAM):
        if err < error:
            return [b]
        u2 = _reparameterize(pts, u, b)
        b2 = _generate(pts, u2, t1, t2)
        err2, split2 = _max_error(pts, b2, u2)
        if err2 >= err:
            break
        u, b, err, split = u2, b2, err2, split2
    if err < error:
        return [b]

    if depth >= MAX_DEPTH:
        return [b]

    split = min(max(split, 1), len(pts) - 2)
    tc = _unit(pts[split - 1] - pts[split + 1])
    return (fit_run(pts[: split + 1], t1, tc, error, depth + 1)
            + fit_run(pts[split:], -tc, t2, error, depth + 1))


def _line_dir(block: np.ndarray) -> np.ndarray:
    """Dominant direction of a run of points, by least squares.

    A single chord across a pixel staircase inherits the staircase; the
    principal axis of the whole window does not.
    """
    if len(block) < 2:
        return np.zeros(2)
    c = block - block.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    d = vt[0]
    # orient along travel
    return d if float(d @ (block[-1] - block[0])) >= 0 else -d


def _tangent(pts: np.ndarray, i: int, span: int, forward: bool) -> np.ndarray:
    n = len(pts)
    if forward:
        return _line_dir(pts[i : min(i + span + 1, n)])
    return -_line_dir(pts[max(i - span, 0) : i + 1])


def _closed_tangent(p: np.ndarray, i: int, span: int) -> np.ndarray:
    """Direction of travel at one point of a closed contour, from both sides.

    Used at every join that is not a real corner: both runs meeting there get
    the same direction, so the pieces leave and arrive along one line and the
    curve is smooth across the seam instead of kinking.
    """
    n = len(p)
    idx = (np.arange(i - span, i + span + 1)) % n
    return _line_dir(p[idx])


def fit_closed(pts: Sequence[Point], error: float, corners: Optional[Sequence[int]] = None,
               tangent_span: int = 3) -> Tuple[Point, List[Tuple[Point, Point, Point]]]:
    """Fit a closed contour, breaking the curve only at the given corner indices."""
    p = np.asarray(pts, dtype=np.float64)
    n = len(p)
    if n < 4:
        return tuple(p[0]), []

    idx = sorted(set(int(c) % n for c in (corners or [])))
    if not idx:
        idx = [0, n // 3, 2 * n // 3] if n > 24 else [0, n // 2]

    corner_set = set(idx) if corners else set()
    dirs = {i: _closed_tangent(p, i, tangent_span) for i in idx}

    segs: List[Cubic] = []
    for a, b in zip(idx, idx[1:] + [idx[0] + n]):
        run = p[np.arange(a, b + 1) % n]
        if len(run) < 2:
            continue
        end = b % n
        t1 = _tangent(run, 0, tangent_span, True) if a in corner_set else dirs[a]
        t2 = (_tangent(run, len(run) - 1, tangent_span, False)
              if end in corner_set else -dirs[end])
        segs.extend(fit_run(run, t1, t2, error))

    if not segs:
        return tuple(p[0]), []
    start = tuple(segs[0][0])
    out = [(tuple(s[1]), tuple(s[2]), tuple(s[3])) for s in segs]
    return start, out
