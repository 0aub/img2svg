"""Convert a pile of images at once, into a tree you can actually navigate.

One folder per input, so a stem's SVG, its silhouette and its report stay
together instead of forming four interleaved heaps in one directory:

    out/
      index.html          every result, worst dE first
      results.csv         the same numbers, for a spreadsheet or a diff
      <stem>/
        <stem>.svg
        <stem>.mono.svg       with --mono
        <stem>.mark.svg       with --mark
        <stem>.report.html    with --report
        compare.png           source | traced | difference

The index is sorted worst-first on purpose. A batch of forty is read by
looking at the three that went wrong, and a name-ordered gallery hides those
somewhere in the middle.
"""

from __future__ import annotations

import base64
import csv
import io
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from . import image, raster, report, verify
from .config import Config
from .pipeline import convert


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _unique(stem: str, taken: Dict[str, int]) -> str:
    """Two inputs can share a basename across folders; keep both results."""
    if stem not in taken:
        taken[stem] = 1
        return stem
    taken[stem] += 1
    return "%s-%d" % (stem, taken[stem])


def _strip(arr: np.ndarray, width: int) -> np.ndarray:
    """Source | traced | difference, side by side at a readable size."""
    tiles = [Image.fromarray(a) for a in arr]
    h = max(t.height for t in tiles)
    each = max(1, (width - 2 * 8) // len(tiles))
    scaled = [t.resize((each, max(1, round(each * t.height / t.width))), Image.LANCZOS)
              for t in tiles]
    h = max(t.height for t in scaled)
    out = Image.new("RGB", (each * len(scaled) + 8 * (len(scaled) - 1), h), (24, 22, 20))
    x = 0
    for t in scaled:
        out.paste(t, (x, 0))
        x += t.width + 8
    return np.asarray(out)


def _heat(de: np.ndarray) -> np.ndarray:
    """Difference map: black where it matches, hot where it does not."""
    t = np.clip(de / 6.0, 0, 1)
    return (np.stack([t, t ** 2.2, t ** 4.0], -1) * 255).astype(np.uint8)


def _png_uri(arr: np.ndarray, width: int = 520) -> str:
    im = Image.fromarray(arr)
    if im.width > width:
        im = im.resize((width, max(1, round(width * im.height / im.width))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run(inputs: Sequence[str], out_dir: str, cfg: Config, *, want_report: bool = False,
        compare: bool = True, log=print) -> List[Dict]:
    """Convert every input into ``out_dir``. Returns one row per input."""
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict] = []
    taken: Dict[str, int] = {}

    for path in inputs:
        name = _unique(_stem(path), taken)
        folder = os.path.join(out_dir, name)
        os.makedirs(folder, exist_ok=True)
        row: Dict = {"name": name, "input": path, "status": "ok"}
        try:
            rgb, opaque, alpha = image.load_full(path)
            res = convert(rgb, cfg, opaque)
        except Exception as e:                       # one bad file must not end the batch
            row.update(status="error", note=str(e))
            rows.append(row)
            log("  %-22s ERROR  %s" % (name, e))
            continue

        svg_path = os.path.join(folder, name + ".svg")
        image.save_text(svg_path, res.svg)
        row.update(width=res.width, height=res.height, colors=len(res.palette),
                   regions=sum(l.regions for l in res.layers), curves=res.segments,
                   kb=round(len(res.svg.encode()) / 1024, 1),
                   background=res.background_hex or "", svg=svg_path,
                   note="; ".join(res.notes))
        if not res.layers:
            row["status"] = "empty"
        # cfg.mono / cfg.mark already decide whether these exist at all
        for text, suffix in ((res.mark, ".mark.svg"), (res.mono, ".mono.svg")):
            if text:
                image.save_text(os.path.join(folder, name + suffix), text)

        try:
            ground = res.background_rgb
            shot = raster.render(res.svg, res.width, res.height, background=ground)
            src = image.composite(rgb, alpha, ground)
            stats = verify.compare(src, shot)
            row.update(mean=round(stats["mean"], 3), p95=round(stats["p95"], 3),
                       max=round(stats["max"], 2), over_2=round(stats["over_2"], 4))
            if compare:
                image.save_png(os.path.join(folder, "compare.png"),
                               _strip([src, shot, _heat(stats["map"])], 1500))
            if want_report:
                image.save_text(os.path.join(folder, name + ".report.html"),
                                report.build(src, res.svg, stats, res, cfg))
            row["_thumb"] = _png_uri(_strip([src, shot, _heat(stats["map"])], 1200))
        except raster.RendererMissing as e:
            row["note"] = (row.get("note") or "") + (" | not scored: %s" % e)

        rows.append(row)
        log("  %-22s %-5s dE %s  %4d curves  %5.1f KB"
            % (name, row["status"],
               ("%.2f" % row["mean"]) if "mean" in row else "   - ",
               row.get("curves", 0), row.get("kb", 0.0)))

    _write_csv(rows, os.path.join(out_dir, "results.csv"))
    image.save_text(os.path.join(out_dir, "index.html"), _index(rows, cfg))
    return rows


_FIELDS = ("name", "status", "mean", "p95", "max", "over_2", "colors", "regions",
           "curves", "kb", "width", "height", "background", "input", "note")


def _write_csv(rows: Sequence[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: -r.get("mean", 0.0)):
            w.writerow(r)


_INDEX_CSS = """
:root{--bg:#14110f;--fg:#f3ece2;--muted:#a2968a;--line:#2e2822;--accent:#e6a725}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:40px 24px 72px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 28px}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:0 0 30px}
.stat{border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.stat b{display:block;font-size:24px;font-weight:600;letter-spacing:-.01em}
.stat span{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.09em}
.card{border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
.card.bad{border-color:#7a3b2a}
.card.empty{border-color:#8a2f2f}
.head{display:flex;justify-content:space-between;align-items:baseline;gap:14px;
 margin-bottom:10px;flex-wrap:wrap}
.head a{color:var(--accent);text-decoration:none;font-weight:600;font-size:15px}
.head .m{color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums}
.card img{display:block;width:100%;height:auto;border-radius:8px}
.legend{color:var(--muted);font-size:12px;padding-top:7px}
.note{color:#e0a37c;font-size:12px;padding-top:6px}
"""


def _index(rows: Sequence[Dict], cfg: Config) -> str:
    scored = [r for r in rows if "mean" in r]
    ok = [r for r in rows if r["status"] == "ok"]
    mean = float(np.mean([r["mean"] for r in scored])) if scored else 0.0
    worst = max((r["mean"] for r in scored), default=0.0)
    parts = [
        "<!doctype html><meta charset=utf-8><title>img2svg batch</title>",
        "<style>%s</style><main>" % _INDEX_CSS,
        "<h1>img2svg batch</h1>",
        "<p class=sub>%d inputs &middot; %d converted &middot; sorted worst first</p>"
        % (len(rows), len(ok)),
        "<div class=stats>",
        "<div class=stat><b>%.2f</b><span>mean dE</span></div>" % mean,
        "<div class=stat><b>%.2f</b><span>worst dE</span></div>" % worst,
        "<div class=stat><b>%d</b><span>converted</span></div>" % len(ok),
        "<div class=stat><b>%d</b><span>failed or empty</span></div>"
        % (len(rows) - len(ok)),
        "</div>",
    ]
    for r in sorted(rows, key=lambda r: -r.get("mean", 1e9 if r["status"] != "ok" else 0.0)):
        cls = "card" if r["status"] == "ok" else "card " + r["status"]
        if r.get("mean", 0) > 2.0:
            cls = "card bad"
        metrics = ("dE %.2f &middot; p95 %.2f &middot; %d curves &middot; %.1f KB"
                   % (r["mean"], r["p95"], r["curves"], r["kb"])) if "mean" in r else \
                  ("%s" % r["status"].upper())
        parts.append("<div class='%s'><div class=head>"
                     "<a href='%s/%s.svg'>%s</a><span class=m>%s</span></div>"
                     % (cls, r["name"], r["name"], r["name"], metrics))
        if r.get("_thumb"):
            parts.append("<img src='%s' alt=''>" % r["_thumb"])
            parts.append("<div class=legend>source &nbsp;|&nbsp; img2svg "
                         "&nbsp;|&nbsp; difference (brighter = further off)</div>")
        if r.get("note"):
            parts.append("<div class=note>%s</div>" % r["note"])
        parts.append("</div>")
    parts.append("</main>")
    return "".join(parts)
