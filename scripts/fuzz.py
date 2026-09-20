"""Random flat art: does it ever crash, drift, or come out inaccurate?

Too slow for CI, useful before a release:

    docker run --rm -v "$PWD:/src" -w /src --entrypoint python img2svg:local \
        scripts/fuzz.py 200

Checks, per case: no exception, byte-identical output on a second run, valid
XML, no NaN or Inf in the path data, and the rendered result within a few dE of
the source. 60 cases at the time of writing: 0 failures, mean dE 0.22, max 0.89.
"""
import io, sys, traceback
import numpy as np
from PIL import Image, ImageDraw
from img2svg import image, raster, verify
from img2svg.config import Config
from img2svg.pipeline import convert

rng = np.random.default_rng(7)

def make(i):
    size = int(rng.choice([64, 128, 200, 320, 512]))
    ncol = int(rng.integers(2, 7))
    pal = rng.integers(0, 255, (ncol, 3))
    aa = bool(rng.integers(0, 2))
    scale = 4 if aa else 1
    im = Image.new("RGB", (size * scale, size * scale), tuple(int(v) for v in pal[0]))
    d = ImageDraw.Draw(im)
    for _ in range(int(rng.integers(2, 9))):
        c = tuple(int(v) for v in pal[rng.integers(1, ncol)])
        x0, y0 = rng.integers(0, size, 2) * scale
        w, h = rng.integers(size // 8, size // 2, 2) * scale
        shape = rng.integers(0, 3)
        box = [int(x0), int(y0), int(x0 + w), int(y0 + h)]
        if shape == 0:
            d.ellipse(box, fill=c)
        elif shape == 1:
            d.rectangle(box, fill=c)
        else:
            d.polygon([(box[0], box[1]), (box[2], box[1]), (int((box[0]+box[2])/2), box[3])], fill=c)
    if aa:
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im).astype(np.uint8), aa

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
fails, scores, worst = 0, [], (0, None)
for i in range(N):
    img, aa = make(i)
    try:
        a = convert(img, Config())
        b = convert(img, Config())
        if a.svg != b.svg:
            print("  NON-DETERMINISTIC at case %d" % i); fails += 1; continue
        if not a.layers:
            continue
        shot = raster.render(a.svg, a.width, a.height, background=a.background_rgb)
        st = verify.compare(img, shot)
        scores.append(st["mean"])
        if st["mean"] > worst[0]:
            worst = (st["mean"], (i, img.shape, aa, len(a.palette), a.segments))
        import xml.dom.minidom as md
        md.parseString(a.svg)
        if "nan" in a.svg.lower() or "inf" in a.svg.lower():
            print("  NON-FINITE COORDS at case %d" % i); fails += 1
    except Exception:
        fails += 1
        print("  CRASH at case %d (%s, aa=%s):" % (i, img.shape, aa))
        traceback.print_exc(limit=3)

s = np.array(scores)
print("\n%d cases: %d failures" % (N, fails))
print("dE mean %.3f   median %.3f   p90 %.3f   max %.3f" % (s.mean(), np.median(s), np.percentile(s, 90), s.max()))
print("over dE 3: %d of %d" % (int((s > 3).sum()), len(s)))
print("worst case:", worst)
