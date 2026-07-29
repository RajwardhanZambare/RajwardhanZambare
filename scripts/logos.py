"""
Logo point-cloud extraction. Each reference SVG (traced, not hand-drawn) is
rasterized and sampled down to N_POINTS positions approximating its filled
shape, normalized into a common STAGE x STAGE coordinate box so the three
logos can be morphed between via optimal transport (see grouping.py / svg_build.py).
"""
import os
import io
import numpy as np
import cairosvg
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
DATA = os.path.join(HERE, "data")

LOGOS = {
    "react": os.path.join(ASSETS, "react.svg"),
    "node": os.path.join(ASSETS, "node.svg"),
    "python": os.path.join(ASSETS, "python.svg"),
}

RASTER = 360
N_POINTS = 900
STAGE = 300.0


def rasterize_alpha(path):
    png_bytes = cairosvg.svg2png(url=path, output_width=RASTER, output_height=RASTER,
                                  background_color=None)
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return np.array(im)[:, :, 3]


def sample_points(alpha, n=N_POINTS, thresh=40, seed=42):
    rng = np.random.default_rng(seed)
    ys, xs = np.where(alpha > thresh)
    if len(xs) == 0:
        raise RuntimeError("no ink found in rasterized logo")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    scale = (STAGE * 0.86) / max(w, h)
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    px = (xs[idx] - x0 - w / 2) * scale + STAGE / 2
    py = (ys[idx] - y0 - h / 2) * scale + STAGE / 2
    pts = np.stack([px, py], axis=1)
    if len(pts) < n:
        extra = n - len(pts)
        rep_idx = rng.choice(len(pts), size=extra, replace=True)
        jitter = rng.normal(0, 0.6, size=(extra, 2))
        pts = np.vstack([pts, pts[rep_idx] + jitter])
    return pts


def build_logo_data(verbose=True):
    os.makedirs(DATA, exist_ok=True)
    clouds = {}
    for name, path in LOGOS.items():
        alpha = rasterize_alpha(path)
        pts = sample_points(alpha)
        clouds[name] = pts
        if verbose:
            print(f"[logos] {name}: {len(pts)} points sampled from {int((alpha>40).sum())} ink px")
        np.save(os.path.join(DATA, f"logo_{name}.npy"), pts)
    return clouds


if __name__ == "__main__":
    build_logo_data()
