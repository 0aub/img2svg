"""Assemble layers into an SVG document."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

Layer = Tuple[str, str, str]  # id, fill, path data

HEADER = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
    'width="{w}" height="{h}" role="img" aria-labelledby="i2s-title i2s-desc">\n'
    "  <title id=\"i2s-title\">{title}</title>\n"
    "  <desc id=\"i2s-desc\">{desc}</desc>\n"
)


def build(
    width: int,
    height: int,
    layers: Sequence[Layer],
    base: Optional[Tuple[str, str]] = None,
    title: str = "",
    desc: str = "",
) -> str:
    """Base shape first, then details clipped to it.

    Clipping to the silhouette lets every detail layer be drawn a hair oversized
    so neighbours overlap instead of leaving hairline gaps, while the outline of
    the artwork stays exactly where the base put it.
    """
    out = [HEADER.format(w=width, h=height, title=escape(title), desc=escape(desc))]
    if base is not None:
        base_fill, base_d = base
        out.append(f'  <defs><clipPath id="i2s-clip"><path d="{base_d}"/></clipPath></defs>\n')
        out.append(f'  <path id="i2s-base" fill="{base_fill}" fill-rule="evenodd" d="{base_d}"/>\n')
        out.append('  <g id="i2s-art" clip-path="url(#i2s-clip)">\n')
        indent = "    "
    else:
        out.append('  <g id="i2s-art">\n')
        indent = "    "
    for ident, fill, d in layers:
        if not d:
            continue
        out.append(f'{indent}<path id="{ident}" fill="{fill}" fill-rule="evenodd" d="{d}"/>\n')
    out.append("  </g>\n</svg>\n")
    return "".join(out)


def build_mono(width: int, height: int, d: str, title: str = "", desc: str = "") -> str:
    return (
        HEADER.format(w=width, h=height, title=escape(title), desc=escape(desc))
        + f'  <path id="i2s-mark" fill="currentColor" fill-rule="evenodd" d="{d}"/>\n</svg>\n'
    )
