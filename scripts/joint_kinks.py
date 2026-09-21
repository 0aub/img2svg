"""Joints that got sharper between two builds: a crease the artwork does not draw.

    docker run --rm -v "$PWD:/src" -w /src --entrypoint python img2svg:local \
        scripts/joint_kinks.py out-before out-after

Any change that moves where two pieces of a path meet has to leave them pointing
the same way as well. This caught a radius-sharing experiment that tightened every
number while creasing 57 joints, the worst by 34 degrees."""
import glob, math, os, re, sys
import numpy as np

NUM = re.compile(r"-?\d*\.?\d+(?:e-?\d+)?")


DATTR = re.compile(r'\bd="([^"]*)"')


def joints_in(svg):
    out = []
    for d in DATTR.findall(svg):
        out += joints(d)
    return out


def joints(d):
    """(point, tangent mismatch) at every interior joint of one path."""
    out, cur, start = [], None, None
    prev_in = None
    for m in re.finditer(r"([MLCZ])([^MLCZ]*)", d):
        cmd, body = m.group(1), m.group(2)
        v = [float(x) for x in NUM.findall(body)]
        if cmd in "LC" and len(v) < (2 if cmd == "L" else 6):
            continue
        if cmd == "M":
            cur = start = np.array(v[:2]); prev_in = None; continue
        if cmd == "Z":
            cur = start; prev_in = None; continue
        if cmd == "L":
            p = np.array(v[:2]); t_out = t_in = p - cur
        else:
            c1, c2, p = np.array(v[0:2]), np.array(v[2:4]), np.array(v[4:6])
            t_out = c1 - cur if np.hypot(*(c1 - cur)) > 1e-9 else p - cur
            t_in = p - c2 if np.hypot(*(p - c2)) > 1e-9 else p - cur
        if prev_in is not None and np.hypot(*prev_in) > 1e-9 and np.hypot(*t_out) > 1e-9:
            a = prev_in / np.hypot(*prev_in)
            b = t_out / np.hypot(*t_out)
            out.append((cur.copy(), math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))))
        prev_in, cur = t_in, p
    return out


def compare(a_path, b_path, tol=0.30):
    """Only joints that are unmistakably the same joint in both builds.

    The two builds do not segment identically, so a loose match compares a joint
    in one with a different joint in the other and invents creases that are not
    there. Require the pairing to be mutually nearest and within a third of a
    pixel; anything else is two different joints.
    """
    A, B = joints_in(open(a_path).read()), joints_in(open(b_path).read())
    if not A or not B:
        return []
    pa = np.array([p for p, _ in A])
    pb = np.array([p for p, _ in B])
    worse = []
    for i, (p, ang) in enumerate(A):
        d = np.hypot(*(pb - p).T)
        j = int(d.argmin())
        if d[j] > tol:
            continue
        back = np.hypot(*(pa - pb[j]).T)
        if int(back.argmin()) != i:          # not mutually nearest: different joints
            continue
        worse.append((B[j][1] - ang, tuple(np.round(p, 1)), ang, B[j][1]))
    return worse


base, new = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(base, "*", "*.svg"))):
    name = os.path.basename(f)[:-4]
    g = os.path.join(new, name, name + ".svg")
    if not os.path.exists(g):
        continue
    for delta, at, a, b in compare(f, g):
        rows.append((delta, name, at, a, b))
rows.sort(reverse=True)
print("%-14s %10s %9s %9s" % ("tile", "at", "before", "after"))
for delta, name, at, a, b in rows[:10]:
    print("  %-12s %10s %8.1f %8.1f  %+6.1f deg" % (name, "%g,%g" % at, a, b, delta))
over = [r for r in rows if r[0] > 5.0]
print("\n  %d joints gained more than 5 deg of mismatch; worst %+.1f"
      % (len(over), rows[0][0] if rows else 0.0))
