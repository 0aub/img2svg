import xml.dom.minidom as minidom
from pathlib import Path

import numpy as np
import pytest

from img2svg import matte, palette, raster, verify
from img2svg.config import Config
from img2svg.pipeline import convert


def test_end_to_end_produces_parseable_svg(flat_art):
    res = convert(flat_art, Config(mono=True))
    doc = minidom.parseString(res.svg)
    assert doc.documentElement.tagName == "svg"
    assert doc.documentElement.getAttribute("viewBox") == "0 0 256 256"
    assert res.segments > 0
    assert res.mono and "currentColor" in res.mono


def test_white_page_is_detected_as_background(flat_art):
    cfg = Config()
    pal = palette.extract(flat_art, cfg)
    labels, _ = matte.matte(flat_art, pal)
    bg = matte.resolve_background(pal, labels, "auto")
    assert bg is not None
    assert palette.rgb_to_hex(pal[bg]) == "#FFFFFF"


def test_background_is_not_painted(flat_art):
    res = convert(flat_art, Config())
    assert "#FFFFFF" not in res.svg, "the page should be left transparent"
    assert res.base_hex is not None


def test_every_layer_colour_comes_from_the_palette(flat_art):
    res = convert(flat_art, Config())
    pal = {palette.rgb_to_hex(c) for c in res.palette}
    for layer in res.layers:
        assert layer.hex in pal


def test_background_none_keeps_the_page(flat_art):
    res = convert(flat_art, Config(background="none"))
    assert res.background is None


def test_alpha_channel_defines_the_silhouette(flat_art):
    opaque = np.zeros(flat_art.shape[:2], dtype=bool)
    opaque[40:200, 40:200] = True
    res = convert(flat_art, Config(), opaque=opaque)
    assert res.background is None
    assert any("alpha" in n for n in res.notes)


def test_blur_zero_is_a_valid_configuration(flat_art):
    res = convert(flat_art, Config(blur=0.0))
    assert res.segments > 0


def test_autoscale_scales_pixel_knobs():
    cfg = Config(blur=6.0, min_area=120.0)
    small = cfg.scaled(256, 256)
    assert small.blur == pytest.approx(1.5)
    assert small.min_area == pytest.approx(120 * 0.0625)
    assert cfg.scaled(1024, 1024).blur == pytest.approx(6.0)


def test_smoothing_window_shrinks_with_the_image():
    """A fixed window is proportionally huge on a small sprite and eats corners."""
    cfg = Config()
    assert cfg.scaled(1024, 1024).smooth_min == 2
    tiny = cfg.scaled(64, 64)
    assert tiny.smooth_min == 0 and tiny.smooth_max == 1


def test_square_corners_survive_on_a_small_sprite():
    """A 16x23 rect on a 64px canvas must not come back with rounded corners."""
    img = np.full((64, 64, 3), (27, 43, 52), dtype=np.uint8)
    img[16:48, 16:48] = (255, 204, 51)
    img[24:47, 24:40] = (204, 51, 68)
    res = convert(img, Config())  # blur scales to 0 here, so corners are held
    red = [l for l in res.layers if l.hex == "#CC3344"]
    assert red, [l.hex for l in res.layers]
    assert red[0].segments <= 6, "a rectangle should need four corners, not an arc"


@pytest.mark.skipif(raster.backend() is None, reason="no SVG renderer installed")
def test_trace_is_faithful_to_the_source(flat_art):
    res = convert(flat_art, Config())
    shot = raster.render(res.svg, res.width, res.height)
    stats = verify.compare(flat_art, shot)
    assert stats["mean"] < 2.0, verify.format_report(stats)


def test_a_dominant_card_is_lifted_out_of_the_mark(flat_art):
    """The plum card covers the frame; the mark is the honey disc on top of it."""
    res = convert(flat_art, Config(mono=True, mark=True))
    assert res.container is not None
    assert palette.rgb_to_hex(res.palette[res.container]) == "#271420"
    assert res.mark and "#271420" not in res.mark
    assert res.mono and "currentColor" in res.mono
    assert any("container" in n for n in res.notes)


def test_container_keep_draws_the_card_normally(flat_art):
    res = convert(flat_art, Config(mono=True, mark=True, container="keep"))
    assert res.container is None
    assert res.mark and "#271420" in res.mark


def _path_bbox(svg: str):
    import re

    d = re.search(r'(?<![\w-])d="([^"]+)"', svg).group(1)  # not id="..."
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d)]
    xs, ys = nums[0::2], nums[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def test_mono_silhouette_is_the_mark_not_the_card(flat_art):
    """The card reaches the frame edges; the honey disc sits well inside it."""
    mark = _path_bbox(convert(flat_art, Config(mono=True)).mono)
    card = _path_bbox(convert(flat_art, Config(mono=True, container="keep")).mono)
    assert card[0] < mark[0] and card[1] < mark[1]
    assert card[2] > mark[2] and card[3] > mark[3]
    assert mark[0] > 50 and mark[2] < 210


@pytest.mark.skipif(raster.backend() is None, reason="no SVG renderer installed")
def test_a_coloured_page_is_scored_against_itself():
    """The dropped background must be painted back before comparing.

    Rendering a transparent trace over white and comparing it to a source with a
    blue page measures the background we deliberately removed, not the tracing -
    it reported dE 35 on artwork that was actually near perfect.
    """
    img = np.full((160, 200, 3), (43, 108, 176), dtype=np.uint8)
    img[40:120, 50:150] = (246, 224, 94)
    res = convert(img, Config())
    assert res.background_hex == "#2B6CB0"
    assert res.background_rgb == (43, 108, 176)

    honest = raster.render(res.svg, res.width, res.height, background=res.background_rgb)
    naive = raster.render(res.svg, res.width, res.height, background=(255, 255, 255))
    assert verify.compare(img, honest)["mean"] < 1.0
    assert verify.compare(img, naive)["mean"] > 10.0


def test_empty_result_is_reported_not_hidden():
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)
    res = convert(noise, Config())
    assert res.layers == []
    assert any("no flat regions" in n for n in res.notes)


def test_composite_flattens_alpha_onto_the_chosen_ground():
    from img2svg import image as img

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    alpha = np.full((4, 4), 0.5, dtype=np.float32)
    out = img.composite(rgb, alpha, (255, 255, 255))
    assert out[0, 0].tolist() == [127, 127, 127]
    assert img.composite(rgb, None, (255, 255, 255)) is rgb


EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "honey-heart.png"


@pytest.mark.skipif(not EXAMPLE.exists(), reason="bundled example missing")
def test_organic_art_is_not_faceted_by_corner_protection():
    """On real soft-shaded artwork, holding raw corners preserves the staircase
    of every diagonal edge and inflates the curve count for nothing."""
    from img2svg.image import load

    rgb, _ = load(str(EXAMPLE))
    auto = convert(rgb, Config())
    forced = convert(rgb, Config(keep_corners="on"))
    assert forced.segments > auto.segments


@pytest.mark.skipif(not EXAMPLE.exists(), reason="bundled example missing")
@pytest.mark.skipif(raster.backend() is None, reason="no SVG renderer installed")
def test_the_bundled_example_stays_faithful():
    """Guard rail on the whole pipeline; loosen only with a reason."""
    from img2svg.image import load

    rgb, _ = load(str(EXAMPLE))
    res = convert(rgb, Config())
    shot = raster.render(res.svg, res.width, res.height, background=res.background_rgb)
    stats = verify.compare(rgb, shot)
    assert stats["mean"] < 2.5, verify.format_report(stats)
    assert res.segments < 1000


def test_verify_defaults_to_the_sources_own_page_colour():
    from img2svg.cli import _page_colour

    img = np.full((80, 80, 3), (43, 108, 176), dtype=np.uint8)
    img[20:60, 20:60] = (246, 224, 94)
    assert _page_colour(img) == (43, 108, 176)
    assert _page_colour(np.full((40, 40, 3), 255, dtype=np.uint8)) == (255, 255, 255)


def test_a_flat_colour_is_never_re_routed_by_a_hairline_better_blend():
    """Two palette colours can usually be mixed to land a shade closer to a
    third. Letting that win hands interior pixels to a colour they look nothing
    like - it repainted a dipper handle #A05E1C where the source was #B97520."""
    from img2svg.matte import matte, _solid_tolerance

    pal = np.array([[39, 20, 32], [230, 167, 37], [187, 121, 30], [160, 94, 28]],
                   dtype=np.int16)
    img = np.full((4, 4, 3), (185, 117, 32), dtype=np.uint8)  # essentially #BB791E
    labels, resid = matte(img, pal)
    assert (labels == 2).all(), "expected #BB791E, got index %r" % set(labels.ravel().tolist())
    assert resid.max() < 5.0  # the distance to #BB791E itself


def test_solid_tolerance_comes_from_the_palette_spacing():
    from img2svg.matte import _solid_tolerance

    # int16 on purpose: (255-0)**2 overflows it, and the old code returned NaN,
    # which compares False against everything and disabled the check in silence
    pal = np.array([[0, 0, 0], [100, 0, 0], [255, 0, 0]], dtype=np.int16)
    assert _solid_tolerance(pal) == pytest.approx(50.0)
    assert _solid_tolerance(pal.astype(np.float32)) == pytest.approx(50.0)
    assert _solid_tolerance(pal[:1]) > 1e6


def test_edge_blends_still_go_to_the_dominant_side():
    """The pair model must still own genuine anti-aliasing."""
    from img2svg.matte import matte

    pal = np.array([[39, 20, 32], [230, 167, 37]], dtype=np.int16)
    for frac, want in ((0.2, 0), (0.45, 0), (0.55, 1), (0.9, 1)):
        mix = np.round(np.array([39, 20, 32]) * (1 - frac)
                       + np.array([230, 167, 37]) * frac).astype(np.uint8)
        img = np.tile(mix, (3, 3, 1))
        labels, _ = matte(img, pal)
        assert (labels == want).all(), f"{frac:.2f} -> {labels[0,0]}, wanted {want}"
