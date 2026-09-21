"""Colour that is simply wrong, as opposed to an edge that is soft.

    docker run --rm -v "$PWD:/src" -w /src --entrypoint python img2svg:local \
        scripts/wrong_regions.py 'marks/*.png'

Most of the difference between a trace and its source sits within two pixels of a
boundary, where the source is a blend and the SVG is a step. That is the tracer
working. A compact patch of the wrong colour in the middle of a flat area is not,
and no mean hides it well enough to notice. Should report zero."""
import glob, sys
import numpy as np
from scipy import ndimage
from img2svg import image, palette, raster
from img2svg.config import Config
from img2svg.matte import matte
from img2svg.pipeline import convert

MIN_PATCH = 0.0002


def patches(path):
    rgb, opaque, _ = image.load_full(path)
    h, w = rgb.shape[:2]
    res = convert(rgb, Config(), opaque)
    pal = res.palette
    want, resid = matte(rgb, pal)
    shot = raster.render(res.svg, w, h, background=res.background_rgb)
    got, _ = matte(shot, pal)

    # only where the source is unambiguous and not near one of its own edges
    solid = ndimage.binary_erosion(
        np.ones_like(want, bool) & (resid <= 12.0), np.ones((3, 3), bool))
    for ax in (0, 1):
        solid &= (np.roll(want, 1, ax) == want) & (np.roll(want, -1, ax) == want)
    solid = ndimage.binary_erosion(solid, np.ones((3, 3), bool))

    bad = solid & (got != want)
    out = []
    cc, k = ndimage.label(bad)
    for i, sl in enumerate(ndimage.find_objects(cc), start=1):
        m = cc[sl] == i
        n = int(m.sum())
        if n < MIN_PATCH * h * w:
            continue
        ys, xs = np.where(m)
        cy, cx = int(ys.mean()) + sl[0].start, int(xs.mean()) + sl[1].start
        out.append((n / (h * w), (cx, cy),
                    palette.rgb_to_hex(pal[want[cy, cx]]),
                    palette.rgb_to_hex(pal[got[cy, cx]])))
    return out


rows = []
for path in sorted(glob.glob(sys.argv[1])):
    name = path.split("/")[-1].rsplit(".", 1)[0]
    for share, at, src_c, got_c in patches(path):
        rows.append((share, name, at, src_c, got_c))
rows.sort(reverse=True)
print("%-14s %8s %12s %10s %10s" % ("tile", "size", "at", "source", "traced"))
for share, name, at, a, b in rows[:15]:
    print("  %-12s %7.2f%%  %11s %10s -> %s" % (name, 100 * share, "%d,%d" % at, a, b))
print("\n  %d patches over %.2f%% of frame, across %d images"
      % (len(rows), 100 * MIN_PATCH, len(set(r[1] for r in rows))))
