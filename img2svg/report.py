"""A single self-contained HTML page: source, result, and where they differ."""

from __future__ import annotations

import base64
import io
from typing import Dict

import numpy as np
from PIL import Image

_CSS = """
:root{--bg:#14110f;--fg:#f3ece2;--muted:#a2968a;--line:#2e2822;--accent:#e6a725}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1120px;margin:0 auto;padding:40px 24px 72px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 32px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
figure{margin:0}
.frame{border:1px solid var(--line);border-radius:12px;overflow:hidden;
 background:#fff;background-image:linear-gradient(45deg,#e9e4dc 25%,transparent 25%),
 linear-gradient(-45deg,#e9e4dc 25%,transparent 25%),
 linear-gradient(45deg,transparent 75%,#e9e4dc 75%),
 linear-gradient(-45deg,transparent 75%,#e9e4dc 75%);
 background-size:18px 18px;background-position:0 0,0 9px,9px -9px,-9px 0}
.frame img{display:block;width:100%;height:auto}
figcaption{color:var(--muted);padding-top:8px;font-size:13px}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:32px 0 8px}
.stat{border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.stat b{display:block;font-size:24px;font-weight:600;letter-spacing:-.01em}
.stat span{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.09em}
table{border-collapse:collapse;width:100%;margin-top:14px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.07em}
.sw{width:15px;height:15px;border-radius:4px;display:inline-block;vertical-align:-2px;
 margin-right:9px;border:1px solid rgba(255,255,255,.22)}
h2{font-size:15px;margin:36px 0 0;letter-spacing:.02em}
.note{color:var(--muted);font-size:13px;margin-top:6px}
@media(max-width:820px){.grid,.stats{grid-template-columns:1fr 1fr}}
"""


def _png_uri(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _svg_uri(text: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(text.encode("utf-8")).decode()


def _heat(de: np.ndarray) -> np.ndarray:
    t = np.clip(de / 6.0, 0, 1)
    r = np.clip(1.6 * t, 0, 1)
    g = np.clip(1.6 * t - 0.5, 0, 1)
    b = np.clip(2.2 * t - 1.3, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def build(src: np.ndarray, svg_text: str, stats: Dict, result, cfg) -> str:
    rows = "".join(
        '<tr><td><span class="sw" style="background:%s"></span>%s</td>'
        "<td>%s</td><td>%d</td><td>%d</td><td>%.2f%%</td></tr>"
        % (l.hex, l.hex, "base" if i == 0 else "layer", l.segments, l.regions,
           100 * l.pixels / float(src.shape[0] * src.shape[1]))
        for i, l in enumerate(result.layers)
    )
    worst = "".join(
        "<tr><td>%.2f</td><td>%d</td><td>%d</td></tr>" % (v, x, y)
        for v, x, y in stats["worst_blocks"][:12]
    )
    notes = ""
    if result.notes:
        notes = '<p class="note">' + "<br>".join(result.notes) + "</p>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>img2svg report</title><style>{_CSS}</style></head><body><main>
<h1>img2svg report</h1>
<p class="sub">{src.shape[1]} x {src.shape[0]} &middot; {len(result.palette)} colours &middot;
 {result.segments} curves &middot; {len(svg_text.encode()) / 1024:.1f} KB</p>
<div class="grid">
 <figure><div class="frame"><img src="{_png_uri(src)}" alt="Source raster"></div>
  <figcaption>source</figcaption></figure>
 <figure><div class="frame"><img src="{_svg_uri(svg_text)}" alt="Traced SVG"></div>
  <figcaption>svg</figcaption></figure>
 <figure><div class="frame"><img src="{_png_uri(_heat(stats['map']))}" alt="Difference heat map">
  </div><figcaption>&Delta;E heat map (black 0, white 6+)</figcaption></figure>
</div>
<div class="stats">
 <div class="stat"><span>mean &Delta;E</span><b>{stats['mean']:.2f}</b></div>
 <div class="stat"><span>p95 &Delta;E</span><b>{stats['p95']:.2f}</b></div>
 <div class="stat"><span>max &Delta;E</span><b>{stats['max']:.2f}</b></div>
 <div class="stat"><span>pixels over &Delta;E 2</span><b>{100 * stats['over_2']:.1f}%</b></div>
</div>
{notes}
<h2>Layers</h2>
<table><thead><tr><th>colour</th><th>role</th><th>curves</th><th>regions</th><th>coverage</th></tr>
</thead><tbody>{rows}</tbody></table>
<h2>Worst 32 px blocks</h2>
<table><thead><tr><th>&Delta;E</th><th>x</th><th>y</th></tr></thead><tbody>{worst}</tbody></table>
</main></body></html>
"""
