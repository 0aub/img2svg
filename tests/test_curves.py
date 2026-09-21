import math

import numpy as np
import pytest

from img2svg import curves


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _rect(h=20, w=30, pad=5):
    m = np.zeros((h, w), dtype=bool)
    m[pad:-pad, pad:-pad] = True
    return m


def test_boundary_walks_a_single_closed_loop():
    loops = curves.boundary_loops(_rect())
    assert len(loops) == 1
    loop = loops[0]
    assert loop[0] != loop[-1], "the loop should not repeat its first point"
    # 20x10 interior -> perimeter of 60 unit edges
    assert len(loop) == 60


def test_outer_loops_wind_with_material_on_the_right():
    (loop,) = curves.boundary_loops(_rect())
    assert curves.polygon_area(loop) > 0


def test_holes_wind_the_other_way():
    m = np.zeros((40, 40), dtype=bool)
    m[5:35, 5:35] = True
    m[15:25, 15:25] = False
    loops = curves.boundary_loops(m)
    assert len(loops) == 2
    areas = sorted(curves.polygon_area(l) for l in loops)
    assert areas[0] < 0 < areas[1]


def test_inset_shrinks_the_outline_by_the_requested_amount():
    (loop,) = curves.boundary_loops(_rect(h=40, w=40, pad=10))
    before = curves.polygon_area(loop)
    after = curves.polygon_area(curves.inset(loop, 0.5))
    side = before ** 0.5
    assert after < before
    assert abs((side - 2 * 0.5) ** 2 - after) < 1.0


def test_rdp_keeps_the_four_corners_of_a_square():
    (loop,) = curves.boundary_loops(_rect(h=40, w=40, pad=8))
    simplified = curves.rdp(loop, 1.0)
    assert 4 <= len(simplified) <= 8


def test_bezier_round_trip_starts_and_closes_on_the_same_point():
    pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    start, segs = curves.to_bezier(pts)
    assert start == pts[0]
    assert len(segs) == len(pts)
    assert segs[-1][2] == pts[0]
    assert curves.path_d(start, segs).endswith("Z")


def test_fmt_never_emits_negative_zero():
    assert curves.fmt(-0.004) == "0"
    assert curves.fmt(12.0) == "12"
    assert curves.fmt(12.25, 1) == "12.2" or curves.fmt(12.25, 1) == "12.3"


def test_smoothing_leaves_a_sharp_corner_where_it_found_one():
    """Round a blob all you like, but a right angle must survive to the fitter."""
    (loop,) = curves.boundary_loops(_rect(h=40, w=40, pad=8))
    guarded = curves.smooth(loop, 3, corner_deg=50.0)
    naive = curves.smooth(loop, 3)
    corner = min(range(len(loop)), key=lambda i: loop[i][0] + loop[i][1])
    moved_guarded = abs(guarded[corner][0] - loop[corner][0]) + \
                    abs(guarded[corner][1] - loop[corner][1])
    moved_naive = abs(naive[corner][0] - loop[corner][0]) + \
                  abs(naive[corner][1] - loop[corner][1])
    assert moved_guarded < 0.1 < moved_naive


def test_smoothing_still_rounds_a_curve():
    import numpy as np

    t = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    circle = [(50 + 20 * np.cos(a) + (0.6 if i % 2 else -0.6), 50 + 20 * np.sin(a))
              for i, a in enumerate(t)]
    out = curves.smooth(circle, 3, corner_deg=50.0)
    jitter_in = max(abs(circle[i][0] - circle[i - 1][0]) for i in range(len(circle)))
    jitter_out = max(abs(out[i][0] - out[i - 1][0]) for i in range(len(out)))
    assert jitter_out < jitter_in


def test_corners_are_found_at_two_scales():
    """A pixel staircase turns ninety degrees at every step; a corner keeps turning."""
    import numpy as np

    # a right angle, approached along two staircase edges
    a = [(float(i), 0.0) for i in range(40)]
    b = [(39.0, float(i)) for i in range(1, 40)]
    loop = a + b + [(float(i), 39.0) for i in range(38, -1, -1)] + \
        [(0.0, float(i)) for i in range(38, 0, -1)]
    found = curves.detect_corners(loop, 5, 50.0)
    assert 3 <= len(found) <= 5, found

    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    circle = [(50 + 30 * np.cos(x), 50 + 30 * np.sin(x)) for x in t]
    assert curves.detect_corners(circle, 5, 50.0) == []


def test_a_straight_run_comes_back_straight():
    """A cubic through a noisy straight run is a shallow S unless asked otherwise."""
    from img2svg import fitting

    rng = np.random.default_rng(3)
    # about what a pre-smoothed sub-pixel contour carries along a straight edge
    pts = np.stack([np.linspace(0, 60, 61), rng.normal(0, 0.12, 61)], 1)
    segs = fitting.fit_run(pts, np.array([1.0, 0.0]), np.array([-1.0, 0.0]), 0.1)
    # it may come back as more than one segment; what matters is that every one
    # of them is exactly straight, not merely close to straight
    assert segs
    for a, c1, c2, b in segs:
        for c in (c1, c2):
            assert abs(_cross(b - a, c - a)) < 1e-9


def test_a_real_arc_is_not_flattened_into_a_line():
    from img2svg import fitting

    t = np.linspace(0, math.pi / 2, 40)
    pts = np.stack([30 * np.cos(t), 30 * np.sin(t)], 1)
    assert not fitting._is_line(pts, 0.2)
    segs = fitting.fit_run(pts, np.array([0.0, 1.0]), np.array([-1.0, 0.0]), 0.2)
    a, c1, c2, b = segs[0]
    assert abs(_cross(b - a, c1 - a)) > 1.0, "an arc must keep its bend"


def test_a_gentle_bend_is_told_apart_from_a_wobble():
    """Both reach the same peak deviation; only one of them is bending."""
    from img2svg import fitting

    x = np.linspace(0, 60, 61)
    bow = np.stack([x, 0.45 * (1 - ((x - 30) / 30) ** 2)], 1)
    rng = np.random.default_rng(5)
    noise = np.stack([x, rng.normal(0, 0.18, 61)], 1)
    # they reach the same peak, so no threshold on deviation can tell them apart
    assert abs(bow[:, 1]).max() == pytest.approx(abs(noise[:, 1]).max(), rel=0.35)
    assert not fitting._is_line(bow, 0.1)
    assert fitting._is_line(noise, 0.1)


def test_a_bend_must_clear_its_own_uncertainty():
    """A short noisy run fits a small parabola by chance; that is not a bend."""
    from img2svg import fitting

    x = np.linspace(0, 60, 61)
    rng = np.random.default_rng(11)
    _, sigma = fitting._bend(np.stack([x, rng.normal(0, 0.2, 61)], 1))
    assert sigma < fitting.LINE_SIGMA
    _, sigma = fitting._bend(np.stack([x, 0.5 * (1 - ((x - 30) / 30) ** 2)], 1))
    assert sigma > fitting.LINE_SIGMA


def test_a_straight_segment_is_written_as_a_line():
    d = curves.path_d((0.0, 0.0), [((10.0, 0.0), (20.0, 0.0), (30.0, 0.0)),
                                   ((30.0, 10.0), (20.0, 25.0), (0.0, 30.0))], prec=1)
    assert d.count("L") == 1 and d.count("C") == 1
    assert "L30 0" in d
