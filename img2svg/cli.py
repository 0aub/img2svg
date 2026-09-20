"""img2svg command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np

from . import __version__, image, palette, raster, report, verify
from .config import Config
from .pipeline import convert

COMMANDS = ("convert", "palette", "verify", "tune", "version")


def _cfg_args(p: argparse.ArgumentParser) -> None:
    d = Config()
    g = p.add_argument_group("palette")
    g.add_argument("--colors", type=int, default=d.max_colors, metavar="N",
                   help="ceiling on palette size (default: %(default)s)")
    g.add_argument("--palette", metavar="HEX,HEX,...",
                   help="use this exact palette and skip extraction")
    g.add_argument("--min-sep", type=float, default=d.min_sep, metavar="D",
                   help="minimum RGB distance between palette entries (default: %(default)s)")
    g.add_argument("--min-frac", type=float, default=d.min_frac, metavar="F",
                   help="drop colours below this share of flat pixels (default: %(default)s)")

    g = p.add_argument_group("shape")
    g.add_argument("--blur", type=float, default=d.blur, metavar="R",
                   help="label smoothing radius; 0 disables (default: %(default)s)")
    g.add_argument("--tolerance", "--rdp", type=float, default=d.tolerance, metavar="PX",
                   dest="tolerance",
                   help="how far a fitted curve may sit from the traced edge "
                        "(default: %(default)s)")
    g.add_argument("--min-area", type=float, default=d.min_area, metavar="A",
                   help="discard regions under this many px (default: %(default)s)")
    g.add_argument("--corner", type=float, default=d.corner_deg, metavar="DEG",
                   help="turns sharper than this stay corners (default: %(default)s)")
    g.add_argument("--inset", type=float, default=d.inset, metavar="PX",
                   help="pull every contour inward by this many px (default: %(default)s)")
    g.add_argument("--overlap", type=float, default=None, metavar="PX",
                   help="grow detail layers outward to hide seams; omit to decide "
                        "from whether the source is anti-aliased (default %.1f when it is)"
                        % d.overlap)
    g.add_argument("--layers", default=d.layers, choices=("flat", "stacked"),
                   help="'flat' is smaller but biases every edge half a pixel "
                        "outward (default: %(default)s)")
    g.add_argument("--no-autoscale", action="store_true",
                   help="treat the px defaults literally instead of scaling to image size")

    g = p.add_argument_group("output")
    g.add_argument("--background", default=d.background, metavar="AUTO|NONE|HEX",
                   help="which colour is the page behind the art (default: %(default)s)")
    g.add_argument("--regularize", default="", metavar="LIST",
                   help="comma separated shape snapping: container, circles, or all")
    g.add_argument("--precision", type=int, default=d.precision, metavar="N",
                   help="decimals kept in path data (default: %(default)s)")
    g.add_argument("--mono", action="store_true", help="also write a currentColor silhouette")
    g.add_argument("--mark", action="store_true",
                   help="also write the artwork with its container card removed")
    g.add_argument("--container", default=d.container, choices=("auto", "keep"),
                   help="lift a dominant background card out of --mark/--mono "
                        "(default: %(default)s)")
    g.add_argument("--title", help="SVG <title>")
    g.add_argument("--desc", help="SVG <desc>")


def _build_cfg(a: argparse.Namespace) -> Config:
    return Config(
        max_colors=a.colors,
        min_sep=a.min_sep,
        min_frac=a.min_frac,
        palette=[c.strip() for c in a.palette.split(",")] if a.palette else None,
        blur=a.blur,
        tolerance=a.tolerance,
        min_area=a.min_area,
        corner_deg=a.corner,
        inset=a.inset,
        overlap=a.overlap if a.overlap is not None else Config().overlap,
        auto_overlap=a.overlap is None,
        layers=a.layers,
        background=a.background,
        regularize=_snapping(a.regularize),
        precision=a.precision,
        mono=a.mono,
        mark=a.mark,
        container=a.container,
        title=a.title,
        desc=a.desc,
        autoscale=not a.no_autoscale,
    )


def _snapping(spec: str):
    want = tuple(x.strip() for x in spec.split(",") if x.strip())
    if "all" in want:
        return ("container", "circles")
    for w in want:
        if w not in ("container", "circles"):
            raise SystemExit(f"img2svg: unknown --regularize option {w!r}")
    return want


def _out_path(inp: str, given: Optional[str], suffix: str = ".svg") -> str:
    if given:
        return given
    return os.path.splitext(inp)[0] + suffix


def cmd_convert(a) -> int:
    rgb, opaque, alpha = image.load_full(a.input)
    cfg = _build_cfg(a)
    res = convert(rgb, cfg, opaque)

    out = _out_path(a.input, a.output)
    stem = os.path.splitext(out)[0]
    image.save_text(out, res.svg)
    say = (lambda *x: None) if a.quiet else print

    say(f"{a.input}  ->  {out}")
    say("  %d x %d  ~  %d colours, %d regions, %d curves, %.1f KB"
        % (res.width, res.height, len(res.palette),
           sum(l.regions for l in res.layers), res.segments,
           len(res.svg.encode()) / 1024))
    for n in res.notes:
        say("  note: " + n)
    if not res.layers:
        print(f"img2svg: {a.input} produced an empty SVG - nothing flat enough to trace",
              file=sys.stderr)
        return 3
    for text, suffix in ((res.mark, ".mark.svg"), (res.mono, ".mono.svg")):
        if text:
            image.save_text(stem + suffix, text)
            say(f"  {suffix.strip('.').split('.')[0]:<4} -> {stem + suffix}")

    if a.report or a.verify:
        try:
            # Score against the same ground the SVG will sit on. Comparing a
            # transparent trace to an opaque source over white measures the
            # background we deliberately dropped, not the tracing.
            ground = res.background_rgb
            shot = raster.render(res.svg, res.width, res.height, background=ground)
        except raster.RendererMissing as e:
            print(f"  cannot verify: {e}", file=sys.stderr)
            return 0
        src = image.composite(rgb, alpha, ground)
        stats = verify.compare(src, shot)
        # --verify was asked for explicitly, so -q does not silence it
        if res.background_hex:
            print(f"  scored against {res.background_hex} (the detected page colour)")
        print(verify.format_report(stats))
        if a.report:
            # sits beside the SVG, not the input, so -o keeps everything together
            rp = a.report if isinstance(a.report, str) else stem + ".report.html"
            image.save_text(rp, report.build(src, res.svg, stats, res, cfg))
            say(f"  report -> {rp}")
    if a.json:
        print(json.dumps(res.summary(), indent=2))
    return 0


def cmd_palette(a) -> int:
    rgb, _ = image.load(a.input)
    cfg = _build_cfg(a).scaled(rgb.shape[1], rgb.shape[0])
    pal = palette.extract(rgb, cfg)
    from .matte import matte as run_matte

    labels, _ = run_matte(rgb, pal)
    shares = np.bincount(labels.ravel(), minlength=len(pal)) / labels.size
    print(f"{a.input}: {len(pal)} colours")
    print(palette.describe(pal, shares))
    return 0


def cmd_verify(a) -> int:
    from .palette import hex_to_rgb, rgb_to_hex

    rgb, _, alpha = image.load_full(a.input)
    with open(a.svg, encoding="utf-8") as fh:
        svg = fh.read()
    # Default to the page colour the source actually uses: scoring a transparent
    # trace over white would otherwise measure the background, not the tracing.
    ground = _page_colour(rgb) if a.against == "auto" else tuple(
        int(v) for v in hex_to_rgb(a.against))
    shot = raster.render(svg, rgb.shape[1], rgb.shape[0], background=ground)
    stats = verify.compare(image.composite(rgb, alpha, ground), shot)
    print(f"{a.svg} vs {a.input}")
    print(f"  scored against {rgb_to_hex(ground)}")
    print(verify.format_report(stats))
    return 0


def _page_colour(rgb: np.ndarray):
    from .matte import matte as run_matte, resolve_background

    cfg = Config().scaled(rgb.shape[1], rgb.shape[0])
    pal = palette.extract(rgb, cfg)
    labels, _ = run_matte(rgb, pal)
    bg = resolve_background(pal, labels, "auto")
    return (255, 255, 255) if bg is None else tuple(int(v) for v in pal[bg])


def cmd_tune(a) -> int:
    from .tune import run as tune_run

    rgb, opaque, alpha = image.load_full(a.input)
    cfg = _build_cfg(a)
    print(f"tuning on {a.input} (budget {a.budget})")
    best, trials = tune_run(rgb, cfg, opaque, budget=a.budget,
                            curve_weight=a.curve_weight, log=print, alpha=alpha)
    print("\nbest: blur=%.1f tolerance=%.2f overlap=%.2f"
          % (best.blur, best.tolerance, best.overlap))
    res = convert(rgb, best, opaque)
    out = _out_path(a.input, a.output)
    image.save_text(out, res.svg)
    print(f"wrote {out}  ({res.segments} curves, {len(res.svg.encode()) / 1024:.1f} KB)")
    print("\nreminder: this optimises fidelity, which is not the same as taste.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="img2svg",
        description="Redraw flat and cel-shaded raster art as clean, verified SVG.",
        epilog="run `img2svg convert IMAGE` or just `img2svg IMAGE`",
    )
    p.add_argument("--version", action="version", version=f"img2svg {__version__}")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("convert", help="trace an image to SVG")
    c.add_argument("input")
    c.add_argument("-o", "--output", help="output path (default: alongside the input)")
    c.add_argument("--report", nargs="?", const=True, default=False,
                   help="write a self-contained HTML comparison")
    c.add_argument("--verify", action="store_true", help="print dE stats after tracing")
    c.add_argument("--json", action="store_true", help="dump a machine-readable summary")
    c.add_argument("-q", "--quiet", action="store_true",
                   help="suppress the summary; --verify output still prints")
    _cfg_args(c)
    c.set_defaults(func=cmd_convert)

    c = sub.add_parser("palette", help="show the extracted palette")
    c.add_argument("input")
    _cfg_args(c)
    c.set_defaults(func=cmd_palette)

    c = sub.add_parser("verify", help="score an existing SVG against a raster")
    c.add_argument("input")
    c.add_argument("svg")
    c.add_argument("--against", default="auto", metavar="AUTO|HEX",
                   help="colour to composite the SVG over; auto detects the "
                        "source's page colour (default: %(default)s)")
    c.set_defaults(func=cmd_verify)

    c = sub.add_parser("tune", help="search parameters against the source")
    c.add_argument("input")
    c.add_argument("-o", "--output")
    c.add_argument("--budget", type=int, default=18, help="max trials (default: %(default)s)")
    c.add_argument("--curve-weight", type=float, default=0.5,
                   help="dE cost charged per 1000 curves (default: %(default)s)")
    _cfg_args(c)
    c.set_defaults(func=cmd_tune)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "convert")  # `img2svg logo.png` just works
    p = build_parser()
    a = p.parse_args(argv)
    if not getattr(a, "func", None):
        p.print_help()
        return 1
    try:
        return a.func(a)
    except FileNotFoundError as e:
        print(f"img2svg: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
