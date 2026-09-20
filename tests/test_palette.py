import numpy as np

from img2svg import palette
from img2svg.config import Config


def test_finds_exactly_the_colours_that_are_there(flat_art):
    pal = palette.extract(flat_art, Config())
    hexes = {palette.rgb_to_hex(c) for c in pal}
    for want in ("#FFFFFF", "#271420", "#E6A725", "#F5D186"):
        assert any(_close(want, h) for h in hexes), f"{want} missing from {sorted(hexes)}"
    assert len(pal) == 4, sorted(hexes)


def _close(a, b, tol=6):
    pa, pb = palette.hex_to_rgb(a), palette.hex_to_rgb(b)
    return int(np.abs(pa - pb).max()) <= tol


def test_flat_mask_rejects_edges(flat_art):
    ok = palette.flat_mask(flat_art, 10)
    assert ok[5, 5]        # deep in the page
    assert ok[128, 128]    # deep in the disc
    assert not ok[24, 128]  # right on the card edge


def test_explicit_palette_is_used_verbatim(flat_art):
    cfg = Config(palette=["#000000", "#ffffff"])
    pal = palette.extract(flat_art, cfg)
    assert pal.shape == (2, 3)
    assert palette.rgb_to_hex(pal[0]) == "#000000"


def test_hex_round_trip():
    assert palette.rgb_to_hex(palette.hex_to_rgb("#e6a725")) == "#E6A725"
    assert palette.rgb_to_hex(palette.hex_to_rgb("#abc")) == "#AABBCC"
