"""Do the thin features survive? A guard against fixing one defect with another.

    docker run --rm -v "$PWD:/src" -w /src --entrypoint python img2svg:local \
        scripts/thin_features.py 'marks/*.png'

Thin features that are really there, and whether the trace still draws them.

Area ratios cannot judge this: removing a spurious fringe and erasing a real
grid line both show up as "this colour lost area". So pick the probes from the
source alone, on two conditions that a fringe fails and a drawn feature passes:

  * the pixel sits at the ridge of something 2 px wide or more, and
  * its colour *is* the palette entry, not a blend that happens to land near one.

A fringe is a blend by construction, so its pixels fail the second test. Then
ask, at each probe, whether the rendered SVG still has that colour there.
"""
import glob, json, os, sys
import numpy as np
from scipy import ndimage
from img2svg import image, palette, raster
from img2svg.config import Config
from img2svg.matte import matte, pair_margin
from img2svg.pipeline import convert

PURE = 0.45          # share of pair_margin a probe pixel must be within
MIN_W, MAX_W = 2.0, 6.0


def probes_for(rgb, pal):
    lab, resid = matte(rgb, pal)
    margin = pair_margin(pal) * PURE
    out = []
    for c in range(len(pal)):
        m = lab == c
        if m.sum() < 30:
            continue
        d = ndimage.distance_transform_edt(m)
        width = 2.0 * d
        ridge = m & (width >= MIN_W) & (width <= MAX_W) & (resid <= margin)
        ridge &= d >= ndimage.maximum_filter(d, size=3) - 1e-9      # local ridge only
        ys, xs = np.where(ridge)
        if not len(ys):
            continue
        step = max(1, len(ys) // 40)
        for y, x in zip(ys[::step], xs[::step]):
            out.append((int(y), int(x), c))
    return out


def main(pattern):
    kept = total = 0
    per = []
    for path in sorted(glob.glob(pattern)):
        rgb, opaque, _ = image.load_full(path)
        h, w = rgb.shape[:2]
        res = convert(rgb, Config(), opaque)
        pal = res.palette
        pts = probes_for(rgb, pal)
        if not pts:
            continue
        shot = raster.render(res.svg, w, h, background=res.background_rgb)
        got, _ = matte(shot, pal)
        ok = sum(1 for y, x, c in pts if got[y, x] == c)
        kept += ok
        total += len(pts)
        per.append((ok / len(pts), os.path.basename(path).rsplit(".", 1)[0], len(pts)))
    per.sort()
    print("%-14s %8s %8s" % ("tile", "kept", "probes"))
    for frac, name, n in per[:12]:
        print("  %-12s %7.2f %8d%s" % (name, frac, n, "  <-- features lost" if frac < 0.8 else ""))
    print()
    print("  %d images, %d probes, %.1f%% of thin features survive"
          % (len(per), total, 100.0 * kept / max(total, 1)))
    print("  images under 80%%: %d" % sum(f < 0.8 for f, _, _ in per))


if __name__ == "__main__":
    main(sys.argv[1])
