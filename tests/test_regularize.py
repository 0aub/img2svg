import numpy as np

from img2svg import regularize
from img2svg.config import Config


def _rounded(w=256, h=256, r=56, n=2.0):
    yy, xx = np.mgrid[0:h, 0:w]
    m = np.ones((h, w), dtype=bool)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        dx, dy = np.abs(xx - cx), np.abs(yy - cy)
        outside = ((xx < r) if cx == r else (xx > w - 1 - r)) & \
                  ((yy < r) if cy == r else (yy > h - 1 - r))
        m &= ~(outside & ((dx / r) ** n + (dy / r) ** n > 1.0))
    return m


def test_recovers_the_radius_of_a_rounded_square():
    m = _rounded(r=56)
    radii, box = regularize._corner_radii(m)
    assert all(abs(x - 56) <= 2 for x in radii), radii
    assert box == (0, 0, 255, 255)


def test_consensus_prefers_the_agreeing_majority():
    r, members = regularize._consensus([236, 236, 189, 187])
    assert abs(r - 236) < 1 and sorted(members) == [0, 1]
    r, members = regularize._consensus([100, 101, 99, 180])
    assert abs(r - 100) < 2 and sorted(members) == [0, 1, 2]


def test_exponent_ignores_corners_outside_the_consensus():
    """Two true corners at r=56 plus two tighter ones must not skew the fit."""
    m = _rounded(r=56, n=1.8)
    m[160:, :] = False
    m[160:, :] = _rounded(r=30, n=1.8)[160:, :]
    radii, box = regularize._corner_radii(m)
    r, members = regularize._consensus(radii)
    n_all = regularize._exponent(m, box, r, [0, 1, 2, 3])
    n_top = regularize._exponent(m, box, r, members)
    assert abs(n_top - 1.8) < abs(n_all - 1.8)


def test_emits_a_closed_path():
    d = regularize.squircle_path(0, 0, 256, 256, 56, 1.8)
    assert d.startswith("M") and d.endswith("Z")
    assert d.count("C") == 8  # two cubics per corner


def test_declines_when_the_shape_is_not_a_container():
    m = np.zeros((256, 256), dtype=bool)
    m[100:150, 100:150] = True
    d, _ = regularize.container(m, 256, 256, Config())
    assert d is None
