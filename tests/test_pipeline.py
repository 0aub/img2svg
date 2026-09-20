import xml.dom.minidom as minidom

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
