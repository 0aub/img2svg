"""Properties each stage claims to have, checked against ground truth.

These are the load-bearing claims in the README. If one of them stops holding,
output quality degrades quietly rather than failing, so they are worth pinning.
"""

from __future__ import annotations

import numpy as np
import pytest

from img2svg import curves, matte, palette, segment, verify
from img2svg.config import Config


# --------------------------------------------------------------------------
# boundary walking and offsetting

def _disc(r: float, size: int = 200) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    c = size / 2.0 - 0.5
    return (xx - c) ** 2 + (yy - c) ** 2 <= r * r


def test_boundary_length_equals_the_pixel_perimeter():
    m = np.zeros((20, 30), dtype=bool)
    m[5:15, 6:20] = True
    (loop,) = curves.boundary_loops(m)
    assert len(loop) == 2 * (10 + 14)


def test_traced_area_matches_the_pixel_area():
    """The corner-grid walk is exact, so the raw loop encloses exactly the mask."""
    for r in (20.0, 40.0, 70.0):
        m = _disc(r)
        (loop,) = curves.boundary_loops(m)
        assert curves.polygon_area(loop) == pytest.approx(float(m.sum()), abs=0.5)


@pytest.mark.parametrize("r", [20.0, 40.0, 60.0, 70.0])
def test_inset_removes_exactly_the_offset_of_area(r):
    """Pulling a closed curve in by d costs it perimeter*d of area.

    For a disc that is pi*r^2 - 2*pi*r*d, i.e. an effective radius of
    sqrt(r^2 - 2*r*d). Worth pinning because the offset is applied per point
    along the normal, which is only equivalent to a true offset while the
    curve stays smooth relative to the distance moved.
    """
    m = _disc(r)
    (loop,) = curves.boundary_loops(m)
    pulled = curves.inset(curves.smooth(loop, 6), 0.5)
    r_out = np.sqrt(abs(curves.polygon_area(pulled)) / np.pi)
    assert r_out == pytest.approx(np.sqrt(r * r - r), abs=0.18)


def test_a_hard_edged_source_is_not_inset_at_all():
    """The half-pixel correction assumes anti-aliasing; without it, it is a leak.

    A disc drawn with no anti-aliasing has its true edge *at* the pixel boundary,
    so the walk is already right and insetting would shave 0.5px off the radius.
    """
    from img2svg.image import load  # noqa: F401
    from img2svg.pipeline import convert

    size = 200
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[_disc(70.0, size)] = (20, 40, 160)
    res = convert(img, Config())
    assert any("no anti-aliasing" in n for n in res.notes), res.notes

    import re

    # not id="...": the attribute name ends in d too, and splitting on 'd="'
    # silently hands you the element id instead of the geometry
    paths = re.findall(r'(?<![\w-])d="([^"]+)"', res.svg)
    pts = _sample_path(paths[-1])
    rad = np.hypot(pts[:, 0] - 99.5, pts[:, 1] - 99.5)
    assert rad.mean() == pytest.approx(70.0, abs=0.4), rad.mean()


def test_a_curve_survives_the_whole_chain_within_a_pixel():
    """smooth -> inset -> rdp -> bezier must not drift off the shape."""
    m = _disc(70.0)
    (loop,) = curves.boundary_loops(m)
    cfg = Config()
    d, n = curves.loop_to_path(loop, smooth_k=8, eps=2.0, ins=0.5,
                               corner_deg=cfg.corner_deg, prec=3)
    pts = _sample_path(d)
    c = 99.5
    radii = np.hypot(pts[:, 0] - c, pts[:, 1] - c)
    assert abs(radii.mean() - np.sqrt(70.0 * 70.0 - 70.0)) < 0.25
    assert radii.std() < 0.6, "the fitted curve wobbles off the circle"
    assert n < 30


def _sample_path(d: str, per_seg: int = 12) -> np.ndarray:
    """Evaluate an M/C/Z path string; enough to check a fit, not a full parser."""
    import re

    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d)]
    start = np.array(nums[:2])
    rest = nums[2:]
    out, cur = [], start
    for i in range(0, len(rest) - 5, 6):
        c1 = np.array(rest[i:i + 2]); c2 = np.array(rest[i + 2:i + 4])
        p = np.array(rest[i + 4:i + 6])
        for t in np.linspace(0, 1, per_seg, endpoint=False):
            m = 1 - t
            out.append(m**3 * cur + 3 * m * m * t * c1 + 3 * m * t * t * c2 + t**3 * p)
        cur = p
    return np.array(out)


# --------------------------------------------------------------------------
# matting

def test_the_fifty_percent_crossing_lands_where_it_should():
    """A ramp between two palette colours must split at the midpoint."""
    pal = np.array([[39, 20, 32], [230, 167, 37]], dtype=np.int16)
    n = 101
    t = np.linspace(0, 1, n)[None, :, None]
    ramp = np.round(pal[0] * (1 - t) + pal[1] * t).astype(np.uint8)
    labels, resid = matte.matte(ramp, pal)
    flip = int(np.argmax(labels[0] == 1))
    assert abs(flip - n // 2) <= 1, f"crossing at {flip}, expected {n // 2}"


def test_residual_peaks_in_the_middle_of_a_blend():
    """The residual is the signal that a pixel is not any palette colour.

    It is what tells anti-aliased art apart from art drawn without it, which in
    turn decides whether layers are grown to hide seams.
    """
    pal = np.array([[39, 20, 32], [230, 167, 37]], dtype=np.int16)
    t = np.linspace(0, 1, 51)[None, :, None]
    ramp = np.round(pal[0] * (1 - t) + pal[1] * t).astype(np.uint8)
    _, resid = matte.matte(ramp, pal)
    r = resid[0]
    assert r[0] < 1 and r[-1] < 1, "the endpoints are palette colours"
    assert r.argmax() in range(24, 27), "the hardest pixel is the middle of the ramp"
    assert r.max() > matte.pair_margin(pal.astype(np.float32))


def test_blend_fraction_separates_anti_aliased_art_from_hard_edges():
    pal = np.array([[39, 20, 32], [230, 167, 37]], dtype=np.int16)
    hard = np.zeros((40, 40, 3), dtype=np.uint8)
    hard[:] = pal[0]
    hard[:, 20:] = pal[1]
    assert matte.blend_fraction(hard, pal) == 0.0

    t = np.clip((np.arange(40) - 19.5) / 3.0 + 0.5, 0, 1)[None, :, None]
    soft = np.round(pal[0] * (1 - t) + pal[1] * t).astype(np.uint8)
    soft = np.repeat(soft, 40, axis=0)
    assert matte.blend_fraction(soft, pal) > 0.03


def test_a_third_colour_does_not_survive_along_an_edge():
    """The guarantee, stated where it actually holds.

    Nearest-colour labelling *does* drop mid-edge pixels onto whatever third
    colour sits between the two - a 50/50 honey/plum blend really is closest to
    a dark brown. That band is one pixel wide, and the blurred majority vote is
    what removes it. Pinning this at the matte stage would pin the wrong thing;
    what has to be true is that nothing spurious reaches the document.
    """
    from img2svg.pipeline import convert

    plum, honey, deep = (39, 20, 32), (230, 167, 37), (160, 94, 28)
    h = w = 256
    img = np.empty((h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    cover = np.clip(80.0 - np.hypot(xx - 128, yy - 128) + 0.5, 0, 1)[..., None]
    img[:] = np.round(np.array(honey) * cover + np.array(plum) * (1 - cover))
    img[:6, :] = deep  # deep must stay in the palette to be a candidate

    res = convert(img, Config(background="none"))
    deep_hex = palette.rgb_to_hex(np.array(deep))
    body = [l for l in res.layers if l.hex == deep_hex]
    spurious = sum(l.pixels for l in body) - 6 * w
    assert spurious < 0.01 * h * w, f"{spurious} px of {deep_hex} leaked along the edge"


def test_residual_flags_a_colour_the_palette_is_missing():
    pal = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.int16)
    img = np.tile(np.array([200, 30, 30], dtype=np.uint8), (4, 4, 1))
    _, resid = matte.matte(img, pal)
    assert resid.min() > 60, "a saturated red is not a shade of grey"


# --------------------------------------------------------------------------
# blur

def test_box_blur_matches_a_direct_convolution():
    rng = np.random.default_rng(0)
    a = rng.random((40, 55)).astype(np.float32)
    r, passes = 3, 1
    got = segment.box_blur(a, r, passes)
    pad = np.pad(a, ((r, r), (r, r)), mode="edge")
    want = np.empty_like(a)
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            want[y, x] = pad[y:y + 2 * r + 1, x:x + 2 * r + 1].mean()
    assert np.allclose(got, want, atol=1e-5)


def test_box_blur_preserves_a_constant_field():
    a = np.full((30, 30), 0.7, dtype=np.float32)
    assert np.allclose(segment.box_blur(a, 4, 3), 0.7, atol=1e-6)


def test_blurred_vote_does_not_move_a_straight_edge():
    """The property that makes smoothed voting safe rather than lossy."""
    labels = np.zeros((60, 60), dtype=np.int16)
    labels[:, 30:] = 1
    out = segment.smooth_labels(labels, 2, radius=5, passes=3, bg=None, protect_shrink=0.0)
    assert (out == labels).all()


def test_blurred_vote_cleans_up_speckle():
    rng = np.random.default_rng(5)
    labels = np.zeros((80, 80), dtype=np.int16)
    labels[:, 40:] = 1
    noise = rng.random((80, 80)) < 0.18
    labels[noise] ^= 1
    out = segment.smooth_labels(labels, 2, radius=4, passes=3, bg=None, protect_shrink=0.0)
    clean = np.zeros((80, 80), dtype=np.int16)
    clean[:, 40:] = 1
    assert (out != clean).sum() < (labels != clean).sum() * 0.05


# --------------------------------------------------------------------------
# colour science

def test_delta_e2000_on_the_full_sharma_sample():
    cases = [
        ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
        ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
        ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
        ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
        ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
        ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
        ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
        ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
        ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
        ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
        ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
        ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
    ]
    for lab1, lab2, want in cases:
        got = float(verify.delta_e2000(np.array([lab1]), np.array([lab2]))[0])
        assert abs(got - want) < 1e-3, f"{lab1} vs {lab2}: {got:.4f} != {want}"


def test_lab_round_trips_known_colours():
    for rgb, lab in ((( 0, 0, 0), (0.0, 0.0, 0.0)),
                     ((255, 255, 255), (100.0, 0.0, 0.0)),
                     ((255, 0, 0), (53.2408, 80.0925, 67.2032))):
        got = verify.srgb_to_lab(np.array([rgb], dtype=np.uint8))[0]
        assert np.allclose(got, lab, atol=1e-3), f"{rgb} -> {got}"


# --------------------------------------------------------------------------
# palette

def test_palette_recovers_an_exact_synthetic_set():
    want = [(17, 24, 39), (220, 38, 38), (34, 197, 94), (250, 204, 21)]
    img = np.zeros((240, 240, 3), dtype=np.uint8)
    for i, c in enumerate(want):
        img[:, i * 60:(i + 1) * 60] = c
    pal = palette.extract(img, Config())
    got = {palette.rgb_to_hex(c) for c in pal}
    assert got == {palette.rgb_to_hex(np.array(c)) for c in want}


# --------------------------------------------------------------------------
# fine detail inside a large shape

def _linked_discs(size=512, r=46, rod=14, gap=120):
    """Three discs in a triangle joined by thin rods, like a molecule mark."""
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    c = size // 2
    pts = [(c, c - gap), (c - gap, c + gap // 2), (c + gap, c + gap // 2)]
    m = np.zeros((size, size), bool)
    for px, py in pts:
        m |= (xx - px) ** 2 + (yy - py) ** 2 <= r * r
    for (ax, ay), (bx, by) in ((pts[0], pts[1]), (pts[1], pts[2]), (pts[0], pts[2])):
        d = np.hypot(bx - ax, by - ay)
        t = np.clip(((xx - ax) * (bx - ax) + (yy - ay) * (by - ay)) / (d * d), 0, 1)
        m |= np.hypot(xx - (ax + t * (bx - ax)), yy - (ay + t * (by - ay))) <= rod / 2
    img[m] = (25, 99, 68)
    return img, m


def test_fine_detail_survives_the_round_trip():
    """Thin rods and the holes between them must come back the same size.

    Stated as measurable geometry rather than by eye: comparing a soft source
    against a crisp vector render at high zoom makes every edge look fatter than
    it is, and chasing that illusion wastes an afternoon. Ink area and enclosed
    hole area are not fooled. On the two real marks this was checked against,
    ink lands within 1% and holes within 1%.
    """
    from img2svg import raster
    from img2svg.pipeline import convert
    from scipy import ndimage

    if raster.backend() is None:
        pytest.skip("no SVG renderer installed")
    img, mask = _linked_discs(size=1024)
    res = convert(img, Config())
    shot = raster.render(res.svg, res.width, res.height, background=(255, 255, 255))
    drawn = np.abs(shot.astype(int) - np.array([25, 99, 68])).max(-1) < 60

    ink = mask.sum()
    assert abs(int(drawn.sum()) - ink) < 0.03 * ink, "ink area drifted"

    holes_src = int((ndimage.binary_fill_holes(mask) & ~mask).sum())
    holes_out = int((ndimage.binary_fill_holes(drawn) & ~drawn).sum())
    assert abs(holes_out - holes_src) < 0.05 * holes_src, \
        f"enclosed hole area {holes_src} -> {holes_out}"

    xor = int((mask ^ drawn).sum())
    assert xor < 0.02 * img.shape[0] * img.shape[1], "disagreement is more than an edge band"


def test_smoothing_window_is_capped_by_shape_size_not_canvas_size():
    from img2svg import regions

    small = 40.0 ** 2      # a 40px feature
    big = 400.0 ** 2
    assert regions.SMOOTH_OF_EXTENT * np.sqrt(small) < 3
    assert regions.SMOOTH_OF_EXTENT * np.sqrt(big) > 20
