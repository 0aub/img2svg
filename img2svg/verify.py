"""Score a trace against its source in a colour space that matches the eye.

RGB distance says a two-pixel edge shift and a flat area being three units off
are the same size of mistake.  They are not.  CIEDE2000 fixes the colour half of
that; reporting the worst blocks rather than only the mean fixes the rest, since
the failures that matter are local.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    e, k = 216 / 24389, 24389 / 27
    f = np.where(xyz > e, np.cbrt(xyz), (k * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7 + 1e-30)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0.0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2.0))

    Lbp = (L1 + L2) / 2.0
    Cbp = (C1p + C2p) / 2.0
    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    hbp = np.where(C1p * C2p == 0, hsum,
          np.where(hdiff <= 180, hsum / 2.0,
          np.where(hsum < 360, (hsum + 360) / 2.0, (hsum - 360) / 2.0)))

    T = (1 - 0.17 * np.cos(np.radians(hbp - 30))
           + 0.24 * np.cos(np.radians(2 * hbp))
           + 0.32 * np.cos(np.radians(3 * hbp + 6))
           - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dTheta = 30 * np.exp(-(((hbp - 275) / 25.0) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7 + 1e-30))
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dTheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


def compare(src: np.ndarray, out: np.ndarray, block: int = 32,
            mask: Optional[np.ndarray] = None) -> Dict:
    if src.shape != out.shape:
        raise ValueError(f"shape mismatch: {src.shape} vs {out.shape}")
    de = delta_e2000(srgb_to_lab(src), srgb_to_lab(out))
    valid = np.ones(de.shape, dtype=bool) if mask is None else mask
    vals = de[valid]
    h, w = de.shape
    blocks: List[Tuple[float, int, int]] = []
    for by in range(0, h, block):
        for bx in range(0, w, block):
            sub = de[by : by + block, bx : bx + block]
            sv = valid[by : by + block, bx : bx + block]
            if sv.sum() < 16:
                continue
            blocks.append((float(sub[sv].mean()), bx, by))
    blocks.sort(reverse=True)
    return {
        "mean": float(vals.mean()) if vals.size else 0.0,
        "p95": float(np.percentile(vals, 95)) if vals.size else 0.0,
        "max": float(vals.max()) if vals.size else 0.0,
        "over_2": float((vals > 2.0).mean()) if vals.size else 0.0,
        "worst_blocks": blocks[:24],
        "map": de,
    }


def format_report(stats: Dict, limit: int = 8) -> str:
    lines = [
        "  dE2000  mean %.2f   p95 %.2f   max %.2f   %.1f%% of pixels over dE 2"
        % (stats["mean"], stats["p95"], stats["max"], 100 * stats["over_2"]),
    ]
    if stats["worst_blocks"]:
        lines.append("  worst 32px blocks:")
        for v, x, y in stats["worst_blocks"][:limit]:
            lines.append("    dE %5.2f at (%4d, %4d)" % (v, x, y))
    return "\n".join(lines)
