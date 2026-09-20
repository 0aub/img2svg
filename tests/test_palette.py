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


def _thin_line_art(size=320, width=3):
    """Strokes too narrow to have an interior: no pixel is locally flat."""
    from PIL import Image, ImageDraw

    s = 4
    im = Image.new("RGB", (size * s, size * s), "#FFFBEB")
    d = ImageDraw.Draw(im)
    for i in range(6):
        d.arc([(30 + i * 12) * s, (30 + i * 12) * s,
               (290 - i * 12) * s, (290 - i * 12) * s], 0, 300,
              fill="#78350F", width=width * s)
    d.line([160 * s, 20 * s, 160 * s, 300 * s], fill="#78350F", width=width * s)
    return np.asarray(im.resize((size, size), Image.LANCZOS)).astype(np.uint8)


def test_a_colour_with_no_flat_pixels_is_still_found():
    """Clustering flat pixels alone is blind to anything thinner than 3 pixels.

    On real line art that meant the palette came back as the page colour by
    itself and the trace was empty: 24,000 stroke pixels in one test image, 18
    of which were flat.
    """
    img = _thin_line_art()
    ok = palette.flat_mask(img, 10)
    stroke = np.all(np.abs(img.astype(int) - palette.hex_to_rgb("#78350F")) < 45, -1)
    assert (stroke & ok).sum() < 0.02 * stroke.sum(), "the stroke should have no flat interior"

    pal = palette.extract(img, Config())
    assert len(pal) >= 2, f"only found {[palette.rgb_to_hex(c) for c in pal]}"
    d = np.abs(pal.astype(int) - palette.hex_to_rgb("#78350F")).max(1)
    assert d.min() < 50, f"stroke colour missed: {[palette.rgb_to_hex(c) for c in pal]}"


def test_recovery_does_not_fire_on_ordinary_anti_aliasing(flat_art):
    """Every edge in an image is a colour the palette does not contain.

    Judging on the distance to the nearest *blend of two* palette colours is
    what keeps those from looking like colours we missed: measured over six
    images, a palette that is genuinely short leaves 9.6% of the image
    unexplained and a complete one at most 0.12%.
    """
    base = palette.extract(flat_art, Config(missing_share=1.1))   # recovery off
    with_recovery = palette.extract(flat_art, Config())
    assert len(with_recovery) == len(base)
