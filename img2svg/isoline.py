"""Sub-pixel contours from a coverage field.

Walking the pixel grid gives a staircase, and a staircase is not a quantisation
you can average away: along the flat top of a circle the boundary sits a long
way inside the true edge, and around the shoulders it sits outside, so the error
is correlated over dozens of points. Smoothing a circle's staircase leaves the
radius varying by half a pixel however hard you smooth it, and a curve fitted
through it is visibly not round.

Anti-aliasing already tells us where the edge really is: a boundary pixel that is
70% covered puts the edge 70% of the way across. Reading contours as the 0.5
level set of that coverage, with linear interpolation between pixel centres,
places them to a few hundredths of a pixel instead of half of one.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

# case -> list of (from_edge, to_edge), oriented so covered area is on the right.
# corners: TL=1, TR=2, BR=4, BL=8. Edges: T T, R R, B B, L L.
_CASES: Dict[int, Sequence[Tuple[str, str]]] = {
    0: (), 15: (),
    1: (("T", "L"),),
    2: (("R", "T"),),
    3: (("R", "L"),),
    4: (("B", "R"),),
    5: (("T", "L"), ("B", "R")),      # saddle
    6: (("B", "T"),),
    7: (("B", "L"),),
    8: (("L", "B"),),
    9: (("T", "B"),),
    10: (("R", "T"), ("L", "B")),     # saddle
    11: (("R", "B"),),
    12: (("L", "R"),),
    13: (("T", "R"),),
    14: (("L", "T"),),
}


def contours(field: np.ndarray, level: float = 0.5) -> List[List[Point]]:
    """Closed loops of the ``level`` isoline, in image coordinates.

    A pixel at index (x, y) is treated as covering [x, x+1] x [y, y+1], so its
    centre - where the field is sampled - is at (x + 0.5, y + 0.5).
    """
    f = np.asarray(field, dtype=np.float64)
    if f.shape[0] < 2 or f.shape[1] < 2:
        return []
    # Pad with empty so a shape running off the edge of the image still closes.
    # A card that bleeds to all four sides otherwise yields four open arcs -
    # its corners - and loses every straight edge in between. Padding with 0
    # against an interior of 1 puts the crossing exactly on the image edge.
    f = np.pad(f, 1, mode="constant", constant_values=0.0)
    h, w = f.shape
    above = f >= level

    tl, tr = above[:-1, :-1], above[:-1, 1:]
    bl, br = above[1:, :-1], above[1:, 1:]
    case = (tl.astype(np.uint8) | (tr.astype(np.uint8) << 1)
            | (br.astype(np.uint8) << 2) | (bl.astype(np.uint8) << 3))

    # resolve saddles by the cell average, the usual convention
    mid = (f[:-1, :-1] + f[:-1, 1:] + f[1:, :-1] + f[1:, 1:]) / 4.0
    saddle_hi = mid >= level

    def cross(a: float, b: float) -> float:
        d = b - a
        return 0.5 if abs(d) < 1e-12 else min(max((level - a) / d, 0.0), 1.0)

    # directed edge chain, keyed by the grid edge a crossing sits on
    nxt: Dict[Tuple[str, int, int], Tuple[str, int, int]] = {}
    pos: Dict[Tuple[str, int, int], Point] = {}

    ys, xs = np.nonzero((case != 0) & (case != 15))
    for y, x in zip(ys.tolist(), xs.tolist()):
        c = int(case[y, x])
        segs = _CASES[c]
        if c in (5, 10) and not saddle_hi[y, x]:
            segs = (segs[1], segs[0]) if c == 5 else (segs[1], segs[0])
        a, b, cc, d = f[y, x], f[y, x + 1], f[y + 1, x + 1], f[y + 1, x]
        ids = {
            "T": ("h", x, y),
            "B": ("h", x, y + 1),
            "L": ("v", x, y),
            "R": ("v", x + 1, y),
        }
        pts = {
            "T": (x + 0.5 + cross(a, b), y + 0.5),
            "B": (x + 0.5 + cross(d, cc), y + 1.5),
            "L": (x + 0.5, y + 0.5 + cross(a, d)),
            "R": (x + 1.5, y + 0.5 + cross(b, cc)),
        }
        for e0, e1 in segs:
            i0, i1 = ids[e0], ids[e1]
            pos[i0] = pts[e0]
            pos[i1] = pts[e1]
            nxt[i0] = i1

    loops: List[List[Point]] = []
    seen = set()
    for start in nxt:
        if start in seen:
            continue
        loop: List[Point] = []
        cur = start
        while cur in nxt and cur not in seen:
            seen.add(cur)
            loop.append(pos[cur])
            cur = nxt[cur]
        if len(loop) >= 8:
            loops.append([(x - 1.0, y - 1.0) for x, y in loop])
    return loops
