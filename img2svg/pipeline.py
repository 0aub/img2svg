"""The whole conversion, start to finish."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from . import emit, matte, palette, regions, regularize
from .config import Config


@dataclass
class LayerInfo:
    index: int
    hex: str
    segments: int
    regions: int
    pixels: int


@dataclass
class Result:
    svg: str
    width: int
    height: int
    palette: np.ndarray
    background: Optional[int]
    base_hex: Optional[str]
    background_hex: Optional[str] = None
    layers: List[LayerInfo] = field(default_factory=list)
    mono: Optional[str] = None
    mark: Optional[str] = None
    container: Optional[int] = None
    labels: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)

    @property
    def segments(self) -> int:
        return sum(l.segments for l in self.layers)

    @property
    def background_rgb(self):
        """What the renderer must paint behind the SVG for a fair comparison."""
        if self.background is None:
            return (255, 255, 255)
        return tuple(int(v) for v in self.palette[self.background])

    def summary(self) -> Dict:
        return {
            "width": self.width,
            "height": self.height,
            "colors": len(self.palette),
            "palette": [palette.rgb_to_hex(c) for c in self.palette],
            "background": None if self.background is None else palette.rgb_to_hex(
                self.palette[self.background]),
            "container": None if self.container is None else palette.rgb_to_hex(
                self.palette[self.container]),
            "segments": self.segments,
            "bytes": len(self.svg.encode("utf-8")),
            "layers": [l.__dict__ for l in self.layers],
            "notes": self.notes,
        }


def convert(rgb: np.ndarray, cfg: Config, opaque: Optional[np.ndarray] = None) -> Result:
    h, w = rgb.shape[:2]
    cfg = cfg.scaled(w, h)
    notes: List[str] = []

    pal = palette.extract(rgb, cfg)
    labels, _resid = matte.matte(rgb, pal)

    if cfg.auto_overlap and cfg.overlap:
        aa = matte.blend_fraction(rgb, pal)
        if aa < 0.01:
            cfg = cfg.replace(overlap=0.0)
            notes.append(
                "source has no anti-aliasing (%.2f%% blended pixels), so layers "
                "are not grown to hide seams" % (100 * aa)
            )

    if opaque is not None:
        bg = None
        content = opaque
        notes.append("alpha channel used as the silhouette")
    else:
        bg = matte.resolve_background(pal, labels, cfg.background)
        content = (regions.content_mask(labels, bg) if bg is None
                   else matte.silhouette(rgb, pal, bg))
        if bg is not None:
            labels = np.where(content, labels, bg)

    radius = int(round(cfg.blur))
    if radius >= 1:
        labels = _smooth(labels, len(pal), radius, cfg, bg, content)
    labels = np.where(content, labels, -1)

    keep = np.zeros_like(content)
    for r in regions.components(content, -1, cfg):
        keep |= r.mask
    if not keep.any():
        keep = content
    labels = np.where(keep, labels, -1)

    counts = {c: int((labels == c).sum()) for c in range(len(pal)) if c != bg}
    counts = {c: n for c, n in counts.items() if n > 0}
    container = _find_container(labels, counts, w * h, cfg)
    if container is not None:
        extent = int(ndimage.binary_fill_holes(labels == container).sum())
        notes.append(
            "treated %s as a container: it fills %.0f%% of the frame, so --mark and --mono "
            "leave it out. Use --container keep to draw it as a normal layer."
            % (palette.rgb_to_hex(pal[container]), 100 * extent / float(w * h))
        )

    svg, infos, base_d = _document(labels, keep, pal, counts, cfg, drop=(), notes=notes,
                                   width=w, height=h, snap=True)

    mark = mono = None
    mark_mask = keep if container is None else (keep & (labels != container))
    if cfg.mono or cfg.mark:
        mark_counts = {c: n for c, n in counts.items() if c != container}
        if mark_counts:
            mark_svg, _, mark_d = _document(
                labels, mark_mask, pal, mark_counts, cfg,
                drop=() if container is None else (container,),
                notes=[], width=w, height=h, snap=False,
            )
            if cfg.mark:
                mark = mark_svg
            if cfg.mono:
                mono = emit.build_mono(w, h, mark_d, (cfg.title or "Vector artwork")
                                       + " silhouette", cfg.desc or "")

    base_hex = infos[0].hex if infos else None
    if not infos:
        notes.append(
            "no flat regions survived - this image may not be flat art, or "
            "--min-area / --min-frac may be too high"
        )
    return Result(svg=svg, width=w, height=h, palette=pal, background=bg,
                  base_hex=base_hex,
                  background_hex=None if bg is None else palette.rgb_to_hex(pal[bg]),
                  layers=infos, mono=mono, mark=mark,
                  container=container, labels=labels, notes=notes)


def _document(labels, silhouette, pal, counts, cfg, drop, notes, width, height, snap):
    """Build one SVG: a filled silhouette, then every other class clipped to it."""
    order = [c for c in sorted(counts, key=lambda c: -counts[c]) if c not in drop]
    stacked = cfg.layers == "stacked"
    if cfg.layers not in ("flat", "stacked"):
        raise ValueError("--layers takes 'flat' or 'stacked'")
    base_d, base_segs = regions.region_path(silhouette, cfg, cfg.inset)
    if snap and "container" in cfg.regularize:
        snapped, note = regularize.container(silhouette, width, height, cfg)
        if snapped:
            base_d, base_segs = snapped, 8
            notes.append(note)

    infos: List[LayerInfo] = []
    layers: List[emit.Layer] = []
    if order:
        base_hex = palette.rgb_to_hex(pal[order[0]])
        infos.append(LayerInfo(order[0], base_hex, base_segs, 1, counts[order[0]]))
    else:
        base_hex = None

    inner = np.where(silhouette, labels, -1)
    for i, c in enumerate(order[1:], start=1):
        if stacked:
            # cover this colour plus everything painted on top of it, so no
            # boundary in the document ever has a gap for the base to show
            d, segs, nreg = regions.mask_path(np.isin(inner, order[i:]), cfg, cfg.inset)
        else:
            d, segs, nreg = regions.class_path(inner, c, cfg, cfg.inset - cfg.overlap)
        if not d:
            continue
        hx = palette.rgb_to_hex(pal[c])
        layers.append((f"i2s-{c}", hx, d))
        infos.append(LayerInfo(c, hx, segs, nreg, counts[c]))

    svg = emit.build(width, height, layers,
                     base=(base_hex, base_d) if base_hex else None,
                     title=cfg.title or "Vector artwork",
                     desc=cfg.desc or "Traced from a raster image by img2svg.")
    return svg, infos, base_d


def _find_container(labels, counts: Dict[int, int], frame: int,
                    cfg: Config) -> Optional[int]:
    """Is the dominant colour a card the artwork sits on, or part of the artwork?

    There is no topological tell - a card surrounding a mark and a solid mark
    carrying highlights nest exactly the same way - so this is a stated rule, not
    a discovery.  A class is a container when, holes filled, it fills most of the
    frame, the rest of the artwork is small inside it, and its own outline is
    card-like: rectangles, squircles and discs pass, a heart or a leaf does not.

    Getting it wrong costs you the wrong --mark, never a wrong main SVG, and the
    conversion prints which class it lifted.  --container keep switches it off.
    """
    if cfg.container == "keep" or not counts:
        return None
    if cfg.container != "auto":
        raise ValueError("--container takes 'auto' or 'keep'")

    cand = max(counts, key=lambda c: counts[c])
    filled = ndimage.binary_fill_holes(labels == cand)
    extent = int(filled.sum())
    if extent < 0.55 * frame:
        return None

    artwork = sum(n for c, n in counts.items() if c != cand)
    if artwork > 0.60 * extent:
        return None

    ys, xs = np.nonzero(filled)
    box = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    if extent < 0.75 * box:  # too ragged to be a card
        return None
    return cand


def _smooth(labels, k, radius, cfg, bg, content):
    from .segment import smooth_labels

    if bg is None:
        tmp = np.where(content, labels, k)  # park the outside in a spare class
        out = smooth_labels(tmp, k + 1, radius, cfg.blur_passes, k, cfg.protect_shrink)
        return np.where(content, out, labels)
    return smooth_labels(labels.copy(), k, radius, cfg.blur_passes, bg, cfg.protect_shrink)
