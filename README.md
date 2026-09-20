# img2svg

Redraw flat and cel-shaded raster art as clean, **verified** SVG.

Most tracers hand you a pile of paths and wish you luck. This one extracts the
palette the artwork was actually drawn with, finds the silhouette by un-mixing
the page colour back out, cleans soft shading into honest flat shapes, and then
**renders its own output back to pixels and scores it against the source** in
CIEDE2000 so you know whether to trust it.

Every design choice below was settled by measurement, and several of them
overturned what the author expected.

```bash
git clone https://github.com/0aub/img2svg && cd img2svg
./img2svg.sh logo.png --report
```

## What it is for

Logos, app icons, stickers, game sprites, cel-shaded illustration, screenshots of
flat UI — anything built from a small set of solid colours. On that material it
reliably beats general-purpose tracers, mostly because it knows what the
background is and can therefore stop inventing a dark fringe around every shape.

**Not** for photographs, painterly work, heavy noise or dithering, or small text.
Those are not "harder" for it, they are the wrong input.

## Install

**Docker** (nothing else needed). `img2svg.sh` builds the image on first run,
mounts the working directory and runs as your user, so output lands next to the
input with your ownership. Or drive Docker yourself:

```bash
docker build -t img2svg .
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/work" img2svg logo.png
```

Tagging a release (`git tag v0.1.0 && git push --tags`) publishes
`ghcr.io/0aub/img2svg` from CI, after which no clone is needed.

**pip**, if you would rather not use Docker:

```bash
pip install "img2svg[render] @ git+https://github.com/0aub/img2svg"
```

The `render` extra pulls in cairosvg, which needs the cairo shared library
(`apt install libcairo2`, `brew install cairo`). Without it everything works
except `--verify`, `--report` and `tune`. If [resvg] is on `PATH` it is used
instead, and is the better renderer of the two.

[resvg]: https://github.com/linebender/resvg

## Try it

The repo ships the icon this was built against:

```bash
make example
```

```
honey-heart.png  ->  out/honey-heart.svg
  1024 x 1024  ~  8 colours, 54 regions, 734 curves, 26.7 KB
  note: treated #271420 as a container: it fills 96% of the frame, so --mark
        and --mono leave it out. Use --container keep to draw it as a normal layer.
  note: container snapped to a superellipse: r=236, n=1.79
        (measured corner radii 236, 236, 188, 188, spread 21%)
  dE2000  mean 2.63   p95 7.69   max 99.69   7.4% of pixels over dE 2
```

The outputs are the full SVG, `.mark.svg` without the card, `.mono.svg` as a
single `currentColor` silhouette, and a self-contained HTML report with the
source, the result, a ΔE heat map, and the worst 32 px blocks ranked. The regularised version scores *worse* and looks
*better*; see [Fidelity is not taste](#fidelity-is-not-taste).

## How it works

Seven stages. Three of them are the reason this exists.

**1. Palette** — k-means, but only over *flat* pixels: ones whose 3×3
neighbourhood is uniform. Edge pixels are blends of two real colours, and letting
them vote drags clusters onto colours that were never in the artwork. Candidate
peaks are weighted over a 3×3×3 histogram neighbourhood, because a flat colour
never lands in a single bin and a real colour can otherwise look too small to
keep.

**2. Labelling** — nearest colour, which is duller than it sounds and took a
detour to arrive at. An earlier version modelled every pixel as a mixture of two
palette colours and handed it to whichever owned more than half, on the theory
that nearest-colour lookup sends an edge pixel to whatever third colour sits
between the two. Measured across seven images it was **worse everywhere**: twice
the speckle, and it tore apart junctions where three regions meet, because a
mixture can name a colour arbitrarily far from the pixel's own. Nearest colour
cannot do that — its error is bounded by colour distance — and the blurred vote
in stage 4 removes the thin fringes it does leave.

**3. Silhouette** — the one boundary where un-mixing does earn its keep. A
half-and-half page/card pixel is nearest to whatever the palette is densest
around, usually some mid-brown, so nearest colour grows the outline by a pixel.
Inside the artwork that is a hairline nobody sees; here it is the shape of the
whole mark, and it moved a fitted card corner by three pixels. So the page is
un-mixed back out — but only for pixels that are both off-palette *and* next to
the page. Colour alone cannot tell a mid grey from half a dark grey on white, and
asking that question of an interior edge punches holes through solid artwork.

The page itself is the class at all four image corners, provided it really does
wrap the frame. Voting on a border ring is the obvious approach and is wrong for
the most common case: a full-bleed card with rounded corners owns most of every
edge and outvotes the page around it.

**4. Segmentation** — real illustrations are not perfectly flat. Where the source
fades one tone into another over twenty pixels, per-pixel labelling speckles and
traces into ragged lumps. So: blur the one-hot label field, take the argmax. Two
properties make this right rather than a hack — a straight edge does not move
(blurring a step symmetrically leaves the crossing where it was), and only
high-curvature or genuinely ambiguous boundaries shift. Anything the vote would
eat more than half of is restored wholesale, which is what keeps narrow
highlights alive.

**5. Curves** — exact boundary walk on the pixel-corner grid, smoothed with a
window scaled to each contour's length, thinned with Ramer–Douglas–Peucker, then
fitted with Catmull-Rom tangents that break at detected corners. The walk encloses
exactly the pixels it traced, and averaged over sub-pixel phases that is already
where the true edge is, so nothing is inset. (An earlier version pulled every
contour half a pixel inward on a plausible-sounding argument about the walk
sitting proud. Sweeping four inset values against six images put the optimum at
zero, in every column.)

Smoothing runs before corner detection, so a hard right angle is already two soft
45° turns by the time anything looks for it and comes out as an arc. On hard-edged
art each point is held back from smoothing in proportion to how sharply the raw
contour turns there. On *organic* art that is actively harmful — the raw contour
of a smooth diagonal is a staircase, and protecting those steps facets the curve —
so it keys off the blur radius, which is already the answer to "does this artwork
have soft edges".

**6. Assembly** — the silhouette is drawn once as a base, and every detail layer
is clipped to it. That lets details be drawn a hair oversized so neighbours
overlap instead of leaving hairline seams, while the outline of the artwork stays
exactly where the base put it.

**7. Verification** — render the SVG, convert both images to Lab, score CIEDE2000
per pixel, and rank 32 px blocks. RGB distance would call a two-pixel edge shift
and a flat area being three units off the same size of mistake. They are not.

The render is composited over the *detected page colour*, not white, and a
transparent source is flattened onto the same ground. Comparing a transparent
trace to an opaque source over white measures the background you deliberately
dropped: it reported ΔE 35 on a blue-page test image whose artwork was actually
near perfect.

## Fidelity is not taste

`img2svg tune` will search parameters for you and minimise mean ΔE. Read this
before you trust it.

The honey-heart icon has corner radii of 235, 234, 187 and 188 — the bottom of
the card is visibly wrong, almost certainly a generation artefact.
`--regularize container` detects the disagreement, takes the majority (235),
recovers the superellipse exponent from the corner profile (n = 1.77) and emits
an exact shape. The mark is straightforwardly better. The score gets **worse**,
1.18 → 1.98, because it now disagrees with the source in four places where the
source was wrong.

Those fitted values are worth a second look: measured by hand from the source's
sub-pixel alpha, the corner radius is 235.8 and the exponent 1.79. The pipeline
recovers 235 and 1.77 with no knowledge of either.

The same tension shows up in shading. The heart's right rim is a genuine
gradient. Flattening it into one clean hard edge can score worse than a ragged
boundary that happens to straddle the true transition, and the clean edge is
still the one you want in a logo.

Use the number to catch regressions between runs. Do not let it pick the design.

## Commands

```
img2svg IMAGE [options]          # same as: img2svg convert IMAGE
img2svg convert IMAGE [options]
img2svg palette IMAGE            # show the extracted palette and coverage
img2svg verify IMAGE SVG         # score an existing SVG against a raster
                                 #   --against auto|HEX picks the page colour
img2svg tune IMAGE [--budget N]  # search parameters (see the warning above)
```

### Options that matter

| flag | what it does |
| --- | --- |
| `--report [PATH]` | self-contained HTML: source, result, ΔE heat map, worst blocks |
| `--verify` | print ΔE stats without writing a report |
| `--mark` | also write the artwork with its container card removed |
| `--mono` | also write a single-path `currentColor` silhouette |
| `--regularize container` | snap a card-like background to an exact superellipse |
| `--container keep` | do not lift a dominant background out of `--mark`/`--mono` |
| `--background auto\|none\|HEX` | which colour is the page behind the art |
| `--colors N` | ceiling on palette size (default 16) |
| `--min-sep D` | minimum RGB distance between palette entries (default 18) |
| `--palette HEX,HEX,…` | use an exact palette and skip extraction |
| `--blur R` | label smoothing radius; `0` disables (default 6) |
| `--rdp E` | curve simplification tolerance in px (default 2.2) |
| `--smooth-div N` | contour length ÷ N sets the smoothing window (default 45) |
| `--min-area A` | discard regions under this many px (default 120) |
| `--keep-corners auto\|on\|off` | hold corners back from smoothing (default auto) |
| `--layers flat\|stacked` | stacked cannot leak but roughly doubles path data |
| `--overlap PX` | grow detail layers to hide seams (default 0.5, auto 0 on hard edges) |
| `--inset PX` | pull every contour inward (default 0) |
| `--precision N` | decimals kept in path data (default 1) |
| `--json` | machine-readable summary on stdout |

Exit codes: `0` success, `2` input not found, `3` nothing flat enough to trace.

Pixel-valued defaults are quoted for a 1024 px image and scale with the input, so
they behave the same on a 256 px sprite and a 4096 px poster. `--no-autoscale`
turns that off.

### Tuning by hand

Output looks lumpy or speckled → raise `--blur`.
Thin highlights disappearing → lower `--blur`, or raise `--min-area` to cut the
noise instead.
Too many curves → raise `--rdp`.
Corners softened that should be sharp → lower `--corner`.
A colour that matters got merged away → lower `--min-sep`, or pass `--palette`.
Pixel art you want to stay blocky → `--blur 0 --rdp 0.4 --smooth-div 400`.
Right angles coming back rounded → `--keep-corners on`.
Smooth curves coming back faceted → `--keep-corners off`.

## Python

```python
from img2svg import Config, convert
from img2svg.image import load

rgb, opaque = load("logo.png")
result = convert(rgb, Config(mono=True, regularize=("container",)))

print(result.segments, "curves")
open("logo.svg", "w").write(result.svg)
```

`Result` carries `svg`, `mark`, `mono`, the `palette`, the label map, per-layer
statistics and the notes printed by the CLI. Scoring is separate:

```python
from img2svg import raster, verify

shot = raster.render(result.svg, result.width, result.height)
print(verify.format_report(verify.compare(rgb, shot)))
```

## Limits

- Gradients become flat regions. Faithful gradient emission is not implemented.
- Semi-transparent interiors are not modelled; alpha is used only as a silhouette.
- Strokes thinner than about 2 px may be eaten by segmentation; lower `--blur`.
- Container detection is a stated rule, not a discovery — a solid mark that fills
  most of the frame can be mistaken for a card. It only affects `--mark`/`--mono`,
  the conversion says what it did, and `--container keep` switches it off.
- Memory is roughly 8 bytes per pixel per palette entry during segmentation.
- Where three regions meet at a narrow tip, the boundary between two of them can
  step by a pixel or two instead of running smoothly into the point. Raising
  `--blur` reduces it, and `--layers stacked` usually removes it.
- Gradients become flat bands. On a six-colour gradient mark that lands at
  ΔE 1.1 with ink area within 0.7 % of the source, which reads as clean flat
  art rather than as banding — but it is a reinterpretation, not a reproduction.

## Development

```bash
make test     # pytest inside the image
make shell    # poke around
make example  # regenerate examples/out
```

Before a release, also fuzz it:

```bash
docker run --rm -v "$PWD:/src" -w /src --entrypoint python img2svg:local \
    scripts/fuzz.py 200
```

Random flat art at random sizes, with and without anti-aliasing. Each case must
not crash, must produce byte-identical output on a second run, must parse as
XML with no NaN or Inf in the path data, and must render back within a few dE.
60 cases at the time of writing: 0 failures, mean dE 0.22, max 0.89.

The output is also checked against a second renderer: a real browser engine and
cairosvg agree on it to mean dE 0.047, so the clip paths and fill rules are not
leaning on one rasteriser's quirks.

The test suite covers the geometry (winding, holes, inset distance), palette
extraction, the container fitter, and CIEDE2000 against the Sharma et al.
reference pairs.

## Licence

Apache-2.0.
