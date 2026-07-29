"""
Groups the portrait dots two different ways for the two animation layers:

- intro groups (60): a random interleaved partition used ONCE for the shimmer
  reveal. Verified with an "evenness" metric (lower = more scattered/even)
  against a deliberately-clumped spatial-block control.
- drift bands (~90): a jittered spatial partition used by the repeating loop
  layer. Per-dot Gaussian noise (sigma=4) is added BEFORE binning so band
  boundaries are organic rather than a straight grid -- verified with a
  "straight-boundary" metric against a non-jittered control.

Neither metric's absolute scale is guaranteed to match any particular external
reference value; what's verified here is DIRECTION (real grouping vs. a bad
control), which is what the numbers below report.
"""
import os
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

GRID_W, GRID_H = 300, 340
N_INTRO_GROUPS = 60
N_BAND_X, N_BAND_Y = 10, 9  # -> 90 bands
JITTER_SIGMA = 4.0


def evenness_metric(xs, ys, group_ids, n_groups, cells=10):
    cell_w, cell_h = GRID_W / cells, GRID_H / cells
    cx = np.clip((xs / cell_w).astype(int), 0, cells - 1)
    cy = np.clip((ys / cell_h).astype(int), 0, cells - 1)
    cell_id = cy * cells + cx
    total_per_cell = np.bincount(cell_id, minlength=cells * cells).astype(float)
    valid = total_per_cell > 0
    cvs = []
    for g in range(n_groups):
        sel = group_ids == g
        grp = np.bincount(cell_id[sel], minlength=cells * cells).astype(float)
        ratio = grp[valid] / total_per_cell[valid]
        if ratio.mean() > 0:
            cvs.append(ratio.std() / ratio.mean())
    return float(np.mean(cvs))


def straight_boundary_metric(xs, ys, band_id, xbins=60, ybins=60):
    canvas = np.full((GRID_H, GRID_W), -1, dtype=int)
    xi, yi = np.clip(xs.astype(int), 0, GRID_W - 1), np.clip(ys.astype(int), 0, GRID_H - 1)
    canvas[yi, xi] = band_id
    filled = canvas.copy()
    unset = filled == -1
    if unset.any():
        ind = ndimage.distance_transform_edt(unset, return_distances=False, return_indices=True)
        filled = filled[tuple(ind)]
    ys_v, xs_v = np.where(filled[:, 1:] != filled[:, :-1])
    ys_h, xs_h = np.where(filled[1:, :] != filled[:-1, :])

    def cv_hist(vals, lo, hi, nbins):
        h, _ = np.histogram(vals, bins=nbins, range=(lo, hi))
        h = h.astype(float)
        return 0.0 if h.mean() == 0 else h.std() / h.mean()

    cv_x = cv_hist(xs_v + 0.5, 0, GRID_W, xbins)
    cv_y = cv_hist(ys_h + 0.5, 0, GRID_H, ybins)
    return float((cv_x + cv_y) / 2 / np.sqrt(xbins))


def assign_intro_groups(n_dots, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_dots)
    group_ids = np.empty(n_dots, dtype=int)
    group_ids[order] = np.arange(n_dots) % N_INTRO_GROUPS
    return group_ids


def assign_intro_groups_clumped(xs, ys):
    """Deliberately BAD control (patch-reveal) used only to sanity check direction."""
    bx = np.clip((xs / GRID_W * 10).astype(int), 0, 9)
    by = np.clip((ys / GRID_H * 6).astype(int), 0, 5)
    return (by * 10 + bx) % N_INTRO_GROUPS


def assign_drift_bands(xs, ys, jitter=True, sigma=JITTER_SIGMA, seed=7):
    rng = np.random.default_rng(seed)
    if jitter:
        jx = xs + rng.normal(0, sigma, size=xs.shape)
        jy = ys + rng.normal(0, sigma, size=ys.shape)
    else:
        jx, jy = xs, ys
    bx = np.clip((jx / GRID_W * N_BAND_X).astype(int), 0, N_BAND_X - 1)
    by = np.clip((jy / GRID_H * N_BAND_Y).astype(int), 0, N_BAND_Y - 1)
    return by * N_BAND_X + bx


def build_grouping_data(verbose=True):
    for mode in ["dark", "light"]:
        dots = np.load(os.path.join(DATA, f"{mode}_dots.npy"))
        ys, xs = np.where(dots)
        xs, ys = xs.astype(float), ys.astype(float)
        n = len(xs)

        intro_g = assign_intro_groups(n, seed=1)
        band_g = assign_drift_bands(xs, ys, jitter=True, seed=7)

        if verbose:
            ev = evenness_metric(xs, ys, intro_g, N_INTRO_GROUPS)
            ev_bad = evenness_metric(xs, ys, assign_intro_groups_clumped(xs, ys), N_INTRO_GROUPS)
            sb = straight_boundary_metric(xs, ys, band_g)
            sb_bad = straight_boundary_metric(xs, ys, assign_drift_bands(xs, ys, jitter=False))
            print(f"[grouping] {mode}: {n} dots, {N_INTRO_GROUPS} intro groups, "
                  f"{band_g.max()+1} drift bands")
            print(f"[grouping] {mode}: evenness {ev:.4f} (interleaved) vs {ev_bad:.4f} "
                  f"(clumped control)")
            print(f"[grouping] {mode}: straight-boundary {sb:.4f} (jittered) vs {sb_bad:.4f} "
                  f"(no-jitter control)")

        np.save(os.path.join(DATA, f"{mode}_intro_groups.npy"), intro_g)
        np.save(os.path.join(DATA, f"{mode}_drift_bands.npy"), band_g)


if __name__ == "__main__":
    build_grouping_data()
