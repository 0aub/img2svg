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
  1024 x 1024  ~  8 colours, 75 regions, 1309 curves, 46.4 KB
  note: treated #271420 as a container: it fills 96% of the frame, so --mark
        and --mono leave it out. Use --container keep to draw it as a normal layer.
  note: container snapped to a superellipse: r=235, n=1.77
        (measured corner radii 235, 234, 187, 188, spread 20%)
  dE2000  mean 1.63   p95 2.23   max 99.69   5.6% of pixels over dE 2
```

The outputs are the full SVG, `.mark.svg` without the card, `.mono.svg` as a
single `currentColor` silhouette, and a self-contained HTML report with the
source, the result, a ΔE heat map, and the worst 32 px blocks ranked. That
example passes `--regularize container`, which scores *worse* (plain defaults
reach 0.79) and looks *better*; see [Fidelity is not taste](#fidelity-is-not-taste).

## Many at once

```bash
./img2svg.sh batch logos/ -o out
```

One folder per input, so a stem's files stay together instead of forming four
interleaved heaps in one directory:

```
out/
  index.html          every result, worst ΔE first, with source | traced | difference
  results.csv         the same numbers, for a spreadsheet or a diff
  <stem>/
    <stem>.svg
    <stem>.mono.svg       with --mono
    <stem>.mark.svg       with --mark
    <stem>.report.html    with --report
    compare.png           source | traced | difference
```

The index is sorted worst-first on purpose: a batch of forty is read by looking
at the three that went wrong, and a name-ordered gallery buries those in the
middle. A file that fails to convert is recorded and the batch continues.

## How it works

Eight stages. Four of them are the reason this exists.

**1. Palette** — k-means, but only over *flat* pixels: ones whose 3×3
neighbourhood is uniform. Edge pixels are blends of two real colours, and letting
them vote drags clusters onto colours that were never in the artwork. Candidate
peaks are weighted over a 3×3×3 histogram neighbourhood, because a flat colour
never lands in a single bin and a real colour can otherwise look too small to
keep.

That sampling has one blind spot, and it is fatal on line art: a three-pixel
stroke has essentially no interior, so its colour never appears among the flat
pixels at all. One test image had 24,000 stroke pixels of which 18 were flat —
the palette came back as the page colour alone and the trace was empty. So
afterwards the palette is asked what it still cannot account for, judged against
every colour *and every blend of two* so that ordinary anti-aliasing does not
look like a gap. A palette that is genuinely short leaves ~10 % of the image
unexplained; a complete one leaves under 0.2 %.

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

**5. Contours, at sub-pixel precision.** Walking the pixel grid gives a staircase,
and a staircase is not quantisation you can average away: along the flat top of a
circle the boundary sits well inside the true edge and around the shoulders it
sits outside, so the error is *correlated* over dozens of points. Smooth it as
hard as you like and the radius still varies by half a pixel.

Anti-aliasing already says where the edge is — a boundary pixel that is 70%
covered puts it 70% of the way across. Contours are read as the 0.5 level set of
that coverage with linear interpolation, which lands them within **0.03 px**
instead of **0.62 px**: a twentyfold improvement, and the difference between a
circle that is round and one that is merely close.

**6. Fitting, not interpolating.** This is the difference between a traced curve
and a drawn one. Simplifying a contour and running a spline *through* the
surviving points locks in every wobble — the simplifier keeps the jitter peaks,
because they are the extreme points, and the interpolant dutifully passes through
all of them. The result measures well and looks like a blob, because a circle
within half a pixel of round in a hundred independent places is visibly not
round.

A short average is applied first, to take off noise the source itself carries -
corners excepted, because smoothing runs before anything looks for a corner and
a crisp vertex that has already become a gentle bend comes out of the fitter as
an arc. Every angular joint in a line diagram quietly turns into a curve.

Least-squares fitting (Schneider's algorithm) then asks a different question:
what single cubic comes closest to *all* these points at once. Noise symmetric about
the true edge cancels instead of accumulating. Curves are split only where one
cubic genuinely cannot reach, and the tangent at each join is measured from both
sides at once so the pieces meet smoothly rather than kinking.

Smoothing runs before corner detection, so a hard right angle is already two soft
45° turns by the time anything looks for it and comes out as an arc. On hard-edged
art each point is held back from smoothing in proportion to how sharply the raw
contour turns there. On *organic* art that is actively harmful — the raw contour
of a smooth diagonal is a staircase, and protecting those steps facets the curve —
so it keys off the blur radius, which is already the answer to "does this artwork
have soft edges".

**7. Assembly** — the silhouette is drawn once as a base, and every detail layer
is clipped to it. That lets details be drawn a hair oversized so neighbours
overlap instead of leaving hairline seams, while the outline of the artwork stays
exactly where the base put it.

**8. Verification** — render the SVG, convert both images to Lab, score CIEDE2000
per pixel, and rank 32 px blocks. RGB distance would call a two-pixel edge shift
and a flat area being three units off the same size of mistake. They are not.

The render is composited over the *detected page colour*, not white, and a
transparent source is flattened onto the same ground. Comparing a transparent
trace to an opaque source over white measures the background you deliberately
dropped: it reported ΔE 35 on a blue-page test image whose artwork was actually
near perfect.

## Shapes the source was reaching for

Generated and hand-drawn artwork is full of shapes that are circles in *intent*
and wobble by a couple of pixels in *fact*. A soft raster edge hides that; a
crisp vector edge does not. Traced faithfully, a "circle" comes back visibly
lumpy — and the fault is in the source, which is no comfort when it is your logo.

`--regularize circles` fits a circle to each outline and, when one is within 4.5%
of its radius of fitting, emits the exact circle instead. The fit is trimmed,
because a dot in a diagram is rarely a bare dot: it has lines meeting it, and
each one takes a bite out of the outline. Discarding the worst few per cent finds
the circle the dot was drawn as — and snapping repairs the bite, which is what
the source looks like anyway.

On a six-fold node mark, a disc whose traced outline deviates 2.2 px from round
becomes exact, and the file gets *smaller*: four cubics instead of dozens.
`--regularize all` turns this on along with container snapping.

## Small images

Scaling every pixel-valued default with the canvas is right for anything
measured against the artwork and wrong for anything measured against noise.
Anti-aliasing is about a pixel wide whatever the canvas is, and so is encoding
noise. On a 190 px mark the old scaling gave `--tolerance 0.11`, finer than a
blended edge pins its own position down, and `--min-area 4`, which keeps every
speck the quantiser leaves marooned inside a neighbouring colour.

Both now stop shrinking:

- `--min-area` and `--min-hole-area` floor at 40 px and 20 px, themselves capped
  at a thousandth of the frame so a 64 px sprite keeps shapes that are small in
  pixels and large in the picture.
- `--tolerance` is **not** floored, though it was tried. Raising it does shed the
  wobble — and it sheds it by letting the fitter bow a long straight run by that
  much, so an interlocking-squares mark came back visibly melted. Tolerance is
  permission to deviate, which is the wrong instrument for suppressing noise.
- A region never as thick as two fifths of a typical feature, capped at 4.5 px,
  is dropped as a seam rather than drawn. Density cannot catch these: a sliver
  two pixels wide and a hundred long fills its own bounding box completely.

Across 55 marks between 150 and 270 px: 2,103 emitted regions → 300 and
65,503 curves → 30,638, for ΔE 1.007 → 1.036. Nothing changes at 1024 px and
above: the three larger test images come out byte-identical.

## Where an edge is, and how well it is known

Sub-pixel edges come from un-mixing: a boundary pixel is a blend of two palette
colours, and the blend fraction says how far across it the edge runs. That
fraction is a projection onto the segment joining the two colours, so **its error
is the pixel noise divided by how far apart they are**. Between ink and the page,
250 RGB units apart, the edge is placed to within a hundredth of a pixel. Between
two tones of the same green 25 apart, to within a tenth, and the boundary pixels
form a speckled band rather than a line.

`coverage.neighbour_and_alpha` reports that as a per-pixel `trust`. **Nothing
currently acts on it**, and the reason is worth recording. Averaging the coverage
field where trust is low does clear the tearing it causes — and it bends straight
edges, turning pixel-scale nicks into visible waves, which on a geometric mark is
the worse of the two. Nor can the two cases be told apart by separation: a mark
that tears and a mark that must stay straight both have their main boundaries
near 25 units. A smoothing that respects straightness would fix this; a blur does
not.

## Fidelity is not taste

`img2svg tune` will search parameters for you and minimise mean ΔE. Read this
before you trust it.

The honey-heart icon has corner radii of 235, 234, 187 and 188 — the bottom of
the card is visibly wrong, almost certainly a generation artefact.
`--regularize container` detects the disagreement, takes the majority (235),
recovers the superellipse exponent from the corner profile (n = 1.77) and emits
an exact shape. The mark is straightforwardly better. The score gets **worse**,
0.79 → 1.63, because it now disagrees with the source in four places where the
source was wrong.

So `--regularize` is a taste flag, not a quality flag, and it is off by default.
Reach for it when you want the shape the artwork was aiming at; leave it alone
when you want the shape the artwork has.

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
img2svg batch DIR|IMAGE... -o OUT  # trace many, into one organised tree
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
| `--regularize LIST` | `container`, `circles`, or `all` — snap shapes to their ideal form |
| `--container keep` | do not lift a dominant background out of `--mark`/`--mono` |
| `--background auto\|none\|HEX` | which colour is the page behind the art |
| `--colors N` | ceiling on palette size (default 16) |
| `--min-sep D` | minimum RGB distance between palette entries (default 18) |
| `--palette HEX,HEX,…` | use an exact palette and skip extraction |
| `--blur R` | label smoothing radius; `0` disables (default 4) |
| `--tolerance PX` | how far a fitted curve may sit from the traced edge (default 0.6) |
| `--corner DEG` | turns sharper than this stay corners instead of rounding (default 50) |
| `--min-area A` | discard regions under this many px (default 120) |
| `--layers flat\|stacked` | stacked cannot leak but roughly doubles path data |
| `--overlap PX` | grow detail layers to hide seams (default 0.5, auto 0 on hard edges) |
| `--inset PX` | pull every contour inward (default 0) |
| `--precision N` | decimals kept in path data (default 1) |
| `--json` | machine-readable summary on stdout |

`batch` adds `-o DIR` (the tree root), `--report` (one HTML page per input) and
`--no-compare` (skip the per-input `compare.png`). Every `convert` option above
applies to the whole batch.

Exit codes: `0` success, `2` input not found, `3` nothing flat enough to trace
(for `batch`, that at least one input failed or came out empty).

Pixel-valued defaults are quoted for a 1024 px image and scale with the input, so
they behave the same on a 256 px sprite and a 4096 px poster — except that the
knobs which fight noise stop shrinking partway down, because noise does not
shrink with the canvas. See [Small images](#small-images). `--no-autoscale`
turns the whole mechanism off.

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

Measured on 55 logo marks the tool had never seen, default flags, cut from three
contact sheets: mean ΔE 1.06, none over 2, nothing empty. **86 % of that error
sits within 2 px of a boundary**, where the source is a soft blend and the SVG is
a hard edge; away from boundaries the mean is 0.17 ΔE. So the fills are right and
the remaining disagreement is the tracer being crisper than its input.

- Gradients become flat bands. Real `linearGradient` output was prototyped and
  measured before being dropped: across shaded marks, a least-squares plane fit
  the region's colour with a residual of 15–41 RGB units, because the shading
  follows a curving ribbon or a faceted solid rather than a straight ramp. The
  one region that did fit a plane scored *worse* with a gradient than without.
  SVG has no gradient that follows a path, so flat bands are the honest output.
- Palette size is not an accuracy dial. Minimum separation was swept 18 → 8:
  mean ΔE improves 1.050 → 1.029, a 2 % gain, for 2.4× the curves. Below about
  12, extra entries start slicing smooth shading into bands whose boundaries
  are decided by encoding noise.
- Semi-transparent interiors are not modelled; alpha is used only as a silhouette.
- Strokes thinner than about 2 px may be eaten by segmentation; lower `--blur`.
- Container detection is a stated rule, not a discovery — a solid mark that fills
  most of the frame can be mistaken for a card. It only affects `--mark`/`--mono`,
  the conversion says what it did, and `--container keep` switches it off.
- Memory is roughly 8 bytes per pixel per palette entry during segmentation.
- Where three regions meet at a narrow tip, the boundary between two of them can
  step by a pixel or two instead of running smoothly into the point. Raising
  `--blur` reduces it, and `--layers stacked` usually removes it.
- Traced edges sit a touch inside the source's. Across 55 marks the result is
  +0.21 L\* lighter than the source, worst +0.51 — well under the roughly 1 L\*
  a person can see, but it is a bias, not noise.

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
