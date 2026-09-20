import numpy as np

from img2svg import curves


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
