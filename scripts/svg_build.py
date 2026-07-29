"""
Assembles dark.svg / light.svg from the data/ produced by portrait.py,
logos.py and grouping.py.

Animation architecture (see design notes in the class docstrings below for the
derivation): two portrait layers plus one traveler layer.

  INTRO layer  -- one-time-only shimmer reveal (60 random interleaved groups),
                  begin=0, dur=T2(4.3s), no repeat. Base opacity=0, so it simply
                  disappears once its single pass finishes (fill defaults to
                  "remove").
  LOOP layer   -- the repeating drift-band portrait, begin=T2 (4.3s) so its own
                  internal t=0 aligns with the instant the intro finishes fading
                  out. Base opacity=0 covers the [0, T2) gap before it starts.
  TRAVELER     -- the optimal-transport logo morph, begin=0, dur=LOOP_DUR,
                  hidden during the portrait phase, visible during the three
                  logo phases.

Both the portrait layers' keyframes and the traveler layers' keyframes are
reduced to the minimal set that reproduces the exact same animation under
linear interpolation (no visual difference, meaningfully smaller markup).
Redundant SMIL attributes that just restate spec defaults (calcMode="linear",
begin="0s", fill="remove", additive="replace", stroke-linecap="butt") are
omitted throughout for the same reason.
"""
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ---------------------------------------------------------------- geometry --
W, H = 1180, 610
TITLE_H = 36
PANEL_PAD = 24
PORTRAIT_W = int(W * 0.38)
GRID_W, GRID_H = 300, 340

portrait_box = dict(x0=PANEL_PAD, y0=TITLE_H + PANEL_PAD,
                     x1=PORTRAIT_W - PANEL_PAD, y1=H - PANEL_PAD)
box_w = portrait_box['x1'] - portrait_box['x0']
box_h = portrait_box['y1'] - portrait_box['y0']
scale = box_w / GRID_W
img_w, img_h = GRID_W * scale, GRID_H * scale
img_x0 = portrait_box['x0']
img_y0 = portrait_box['y0'] + (box_h - img_h) / 2


def grid_to_px(gx, gy):
    return img_x0 + gx * scale, img_y0 + gy * scale


STAGE = 300.0
logo_scale = img_w / STAGE
logo_y0 = img_y0 + (img_h - STAGE * logo_scale) / 2


def logo_to_px(lx, ly):
    return img_x0 + lx * logo_scale, logo_y0 + ly * logo_scale


info_box = dict(x0=PORTRAIT_W + PANEL_PAD, y0=TITLE_H + PANEL_PAD,
                x1=W - PANEL_PAD, y1=H - PANEL_PAD)

# ------------------------------------------------------------------ timing --
T_PORTRAIT, T_TRANS, T_LOGO = 3.0, 1.3, 2.0
SEG = [0.0]
for d in [T_PORTRAIT, T_TRANS, T_LOGO, T_TRANS, T_LOGO, T_TRANS, T_LOGO, T_TRANS]:
    SEG.append(SEG[-1] + d)
LOOP_DUR = SEG[-1]
assert abs(LOOP_DUR - 14.2) < 1e-9
T0, T1, T2, T3, T4, T5, T6, T7, T8 = SEG  # 0,3.0,4.3,6.3,7.6,9.6,10.9,12.9,14.2

# ---------------------------------------------------------------- palette --
PALETTES = {
    "dark": dict(bg="#0A101F", panel="#0D1526", chrome="#22D3EE", chrome_dim="#0E7490",
                 portrait="#A78BFA", accent="#10B981", live="#EF4444", text="#CBD5E1",
                 muted="#64748B"),
    "light": dict(bg="#F1F5F9", panel="#FFFFFF", chrome="#0891B2", chrome_dim="#67C6DA",
                  portrait="#7C3AED", accent="#0D9668", live="#DC2626", text="#1E293B",
                  muted="#94A3B8"),
}

FONT = "ui-monospace,'Cascadia Code','JetBrains Mono','Fira Code',Menlo,Consolas,monospace"
CHAR_W = 0.60


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def tw(s, size):
    return len(s) * size * CHAR_W


# ---------------------------------------------------- portrait dot runs ----
def build_group_paths(dots_xy, group_ids, n_groups):
    """One <path> per group: filled-rect runs, merged per contiguous row segment,
    drawn as short stroked hlines (compact) rather than font glyphs or rects."""
    paths = []
    xs = dots_xy[:, 0].astype(int)
    ys = dots_xy[:, 1].astype(int)
    for g in range(n_groups):
        sel = group_ids == g
        gx, gy = xs[sel], ys[sel]
        rows = defaultdict(list)
        for x, y in zip(gx, gy):
            rows[y].append(x)
        d = []
        for y, xl in rows.items():
            xl.sort()
            start = prev = xl[0]
            for x in xl[1:] + [None]:
                if x is not None and x == prev + 1:
                    prev = x
                    continue
                px0, py0 = grid_to_px(start, y)
                run_w = (prev - start + 1) * scale
                d.append(f"M{px0:.1f},{py0:.1f}h{run_w:.1f}")
                if x is not None:
                    start = prev = x
        paths.append("".join(d))
    return paths


def centroid_px(dots_xy, ids, gid, mapper):
    sel = ids == gid
    if sel.sum() == 0:
        return mapper(0, 0)
    mx, my = dots_xy[sel, 0].mean(), dots_xy[sel, 1].mean()
    return mapper(mx, my)


# -------------------------------------------------------- traveler morph ---
def load_logo_perms():
    react = np.load(os.path.join(DATA, "logo_react.npy"))
    node = np.load(os.path.join(DATA, "logo_node.npy"))
    python = np.load(os.path.join(DATA, "logo_python.npy"))
    cost12 = cdist(react, node)
    perm12 = linear_sum_assignment(cost12)[1]
    node_reordered = node[perm12]
    cost23 = cdist(node_reordered, python)
    perm23 = linear_sum_assignment(cost23)[1]
    return react, node_reordered, python[perm23]


# ------------------------------------------------------------- fmt helper --
def fmt_list(vals):
    return ";".join(f"{v:.1f}" for v in vals)


def fmt_times(times):
    return ";".join(f"{t/LOOP_DUR:.5f}" for t in times)


def fmt_times_by(times, dur):
    return ";".join(f"{t/dur:.5f}" for t in times)


# =============================================================== BUILD =====
def build_svg(theme):
    pal = PALETTES[theme]
    dots = np.load(os.path.join(DATA, f"{theme}_dots.npy"))
    ys_, xs_ = np.where(dots)
    dot_xy = np.stack([xs_, ys_], axis=1).astype(float)
    intro_g = np.load(os.path.join(DATA, f"{theme}_intro_groups.npy"))
    band_g = np.load(os.path.join(DATA, f"{theme}_drift_bands.npy"))
    n_intro = intro_g.max() + 1
    n_bands = band_g.max() + 1

    react_pts, node_pts, python_pts = load_logo_perms()
    react_centroid_px = logo_to_px(*react_pts.mean(axis=0))

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">')

    svg.append('<defs>')
    svg.append(f'<clipPath id="portraitClip-{theme}"><rect x="{portrait_box["x0"]:.1f}" '
                f'y="{portrait_box["y0"]:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" rx="6"/></clipPath>')
    svg.append(f'<clipPath id="winClip-{theme}"><rect x="0" y="0" width="{W}" height="{H}" rx="14"/></clipPath>')
    svg.append('</defs>')

    svg.append(f'<g clip-path="url(#winClip-{theme})">')
    svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{pal["bg"]}"/>')
    svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="14" fill="none" '
                f'stroke="{pal["chrome_dim"]}" stroke-opacity="0.5" stroke-width="1"/>')

    svg.append(f'<rect x="0" y="0" width="{W}" height="{TITLE_H}" fill="{pal["panel"]}"/>')
    svg.append(f'<line x1="0" y1="{TITLE_H}" x2="{W}" y2="{TITLE_H}" stroke="{pal["chrome_dim"]}" stroke-opacity="0.4"/>')
    for i, c in enumerate(["#F87171", "#FBBF24", "#34D399"]):
        svg.append(f'<circle cx="{20 + i*20}" cy="{TITLE_H/2:.1f}" r="6" fill="{c}" opacity="0.85"/>')
    svg.append(f'<text x="{W/2:.1f}" y="{TITLE_H/2+4.5:.1f}" text-anchor="middle" '
                f'font-size="13" fill="{pal["muted"]}" letter-spacing="0.5">profile.sh --live</text>')

    svg.append(f'<line x1="{PORTRAIT_W}" y1="{TITLE_H}" x2="{PORTRAIT_W}" y2="{H}" '
                f'stroke="{pal["chrome_dim"]}" stroke-opacity="0.35"/>')

    svg.append(f'<rect x="{portrait_box["x0"]:.1f}" y="{portrait_box["y0"]:.1f}" '
                f'width="{box_w:.1f}" height="{box_h:.1f}" rx="6" fill="none" '
                f'stroke="{pal["chrome_dim"]}" stroke-opacity="0.5" stroke-width="1"/>')
    svg.append(f'<text x="{portrait_box["x0"]+10:.1f}" y="{portrait_box["y0"]+18:.1f}" '
                f'font-size="11" letter-spacing="1.5" fill="{pal["chrome"]}" fill-opacity="0.85">VISUAL.MAP</text>')

    svg.append(f'<g clip-path="url(#portraitClip-{theme})">')

    # ---- INTRO LAYER ----
    intro_paths = build_group_paths(dot_xy, intro_g, n_intro)
    stagger_span, fade_dur = 1.2, 0.6
    svg.append(f'<g stroke="{pal["portrait"]}" stroke-width="{scale:.2f}" fill="none">')
    for i, d in enumerate(intro_paths):
        if not d:
            continue
        start = (i / max(n_intro - 1, 1)) * stagger_span
        end = start + fade_dur
        kt = fmt_times_by([0, start, end, T1, T2], T2)
        svg.append(f'<path d="{d}" opacity="0">'
                    f'<animate attributeName="opacity" dur="{T2:.2f}s" '
                    f'keyTimes="{kt}" values="0;0;1;1;0"/></path>')
    svg.append('</g>')

    # ---- LOOP LAYER (begin=T2 so it hands off exactly as the intro finishes) --
    band_paths = build_group_paths(dot_xy, band_g, n_bands)
    svg.append(f'<g stroke="{pal["portrait"]}" stroke-width="{scale:.2f}" fill="none">')
    for b in range(n_bands):
        d = band_paths[b]
        if not d:
            continue
        cxp, cyp = centroid_px(dot_xy, band_g, b, grid_to_px)
        dx = 0.42 * (react_centroid_px[0] - cxp)
        dy = 0.42 * (react_centroid_px[1] - cyp)
        p_hold_end = T7 - T2
        p_return = T8 - T2
        p_rest_end = p_return + T_PORTRAIT
        kt = fmt_times([0, p_hold_end, p_return, p_rest_end, LOOP_DUR])
        tr_vals = (f"translate({dx:.1f},{dy:.1f});translate({dx:.1f},{dy:.1f});"
                   f"translate(0,0);translate(0,0);translate({dx:.1f},{dy:.1f})")
        svg.append(f'<path d="{d}" opacity="0">'
                    f'<animate attributeName="opacity" dur="{LOOP_DUR:.3f}s" begin="{T2:.2f}s" '
                    f'repeatCount="indefinite" keyTimes="{kt}" values="0;0;1;1;0"/>'
                    f'<animateTransform attributeName="transform" type="translate" '
                    f'dur="{LOOP_DUR:.3f}s" begin="{T2:.2f}s" repeatCount="indefinite" '
                    f'keyTimes="{kt}" values="{tr_vals}"/></path>')
    svg.append('</g>')

    # ---- TRAVELER LAYER (minimal per-attribute keyframes; see module docstring) --
    kt_pos = fmt_times([T0, T3, T4, T5, T6, T7, T8])
    kt_op = fmt_times([T0, T2, T2 + 0.3, T7 - 0.3, T7, T8])
    svg.append(f'<g fill="{pal["accent"]}">')
    for i in range(len(react_pts)):
        p0 = logo_to_px(*react_pts[i])
        p1 = logo_to_px(*node_pts[i])
        p2 = logo_to_px(*python_pts[i])
        cxs = fmt_list([p0[0], p0[0], p1[0], p1[0], p2[0], p2[0], p0[0]])
        cys = fmt_list([p0[1], p0[1], p1[1], p1[1], p2[1], p2[1], p0[1]])
        svg.append(f'<circle r="1.6" cx="{p0[0]:.1f}" cy="{p0[1]:.1f}" opacity="0">'
                    f'<animate attributeName="cx" dur="{LOOP_DUR:.3f}s" repeatCount="indefinite" '
                    f'keyTimes="{kt_pos}" values="{cxs}"/>'
                    f'<animate attributeName="cy" dur="{LOOP_DUR:.3f}s" repeatCount="indefinite" '
                    f'keyTimes="{kt_pos}" values="{cys}"/>'
                    f'<animate attributeName="opacity" dur="{LOOP_DUR:.3f}s" repeatCount="indefinite" '
                    f'keyTimes="{kt_op}" values="0;0;1;1;0;0"/></circle>')
    svg.append('</g>')

    svg.append('</g>')  # end portrait clip

    # ---------------------------------------------------------- info panel --
    ix0, iy0, ix1 = info_box['x0'], info_box['y0'], info_box['x1']
    panel_w = ix1 - ix0
    svg.append(f'<text x="{ix0:.1f}" y="{iy0+13:.1f}" font-size="13" letter-spacing="1.5" '
                f'fill="{pal["chrome"]}">SYSTEM.INFO</text>')

    live_x = ix1 - 16
    svg.append(f'<g transform="translate({live_x:.1f},{iy0+7:.1f})">')
    svg.append(f'<circle r="4" fill="{pal["live"]}"><animate attributeName="opacity" '
                f'values="1;0.25;1" dur="1.6s" repeatCount="indefinite"/></circle>')
    svg.append(f'<text x="-11" y="4" text-anchor="end" font-size="12" letter-spacing="1" '
                f'fill="{pal["live"]}">LIVE</text></g>')

    handle = "@RajwardhanZambare"
    pill_w = tw(handle, 14) + 22
    pill_x, pill_y = ix1 - pill_w, iy0 + 20
    svg.append(f'<rect x="{pill_x:.1f}" y="{pill_y:.1f}" width="{pill_w:.1f}" height="22" rx="11" '
                f'fill="{pal["accent"]}" fill-opacity="0.15" stroke="{pal["accent"]}" stroke-width="1"/>')
    svg.append(f'<text x="{pill_x+pill_w/2:.1f}" y="{pill_y+15.5:.1f}" text-anchor="middle" '
                f'font-size="14" fill="{pal["accent"]}">{esc(handle)}</text>')

    rows_a = [("Subject", "Rajwardhan Zambare"), ("Role", "Full-Stack Developer"),
              ("Origin", "Maharashtra, India"),
              ("Education", "Pursuing B.Tech Computer Science and Engineering"),
              ("Status", "Building + Learning"),
              ("ToolChain", "VS Code \u2022 Git \u2022 GitHub \u2022 Postman \u2022 Thunder Client \u2022 MongoDB Compass")]
    rows_b = [("Core.Lang", "C \u2022 C++ \u2022 Java \u2022 Python \u2022 JavaScript \u2022 SQL"),
              ("Core.Frontend", "HTML \u2022 CSS \u2022 Tailwind CSS \u2022 JS \u2022 React"),
              ("Core.Backend", "Node.js \u2022 Express.js"),
              ("Core.Database", "MongoDB \u2022 MySQL"),
              ("Core.Infra", "Vercel \u2022 Netlify \u2022 Render \u2022 Railway \u2022 GitHub")]
    rows_c = [("Grid.Mail", "zambarerajwardhan4063@gmail.com", "mailto:zambarerajwardhan4063@gmail.com"),
              ("Grid.Portfolio", "coming soon", None),
              ("Grid.LinkedIn", "/in/rajwardhan-zambare", "https://www.linkedin.com/in/rajwardhan-zambare-704b93321"),
              ("Grid.GitHub", "@RajwardhanZambare", "https://github.com/RajwardhanZambare")]

    row_h = 23
    y = iy0 + 60
    label_size = val_size = 14
    MIN_VAL_W = 60

    def add_row(label, value, href=None):
        nonlocal y
        lbl_w = tw(label, label_size)
        val_w = max(tw(value, val_size), MIN_VAL_W)
        val_w = min(val_w, panel_w - lbl_w - 40)
        leader_x0 = ix0 + lbl_w + 6
        leader_x1 = ix1 - val_w - 4
        svg.append(f'<text x="{ix0:.1f}" y="{y:.1f}" font-size="{label_size}" '
                    f'fill="{pal["muted"]}">{esc(label)}</text>')
        if leader_x1 > leader_x0 + 4:
            svg.append(f'<line x1="{leader_x0:.1f}" y1="{y-4:.1f}" x2="{leader_x1:.1f}" y2="{y-4:.1f}" '
                        f'stroke="{pal["chrome_dim"]}" stroke-opacity="0.55" stroke-width="1.2" '
                        f'stroke-dasharray="1,3.4" stroke-linecap="round"/>')
        val_el = (f'<text x="{ix1:.1f}" y="{y:.1f}" text-anchor="end" font-size="{val_size}" '
                  f'fill="{pal["text"]}" textLength="{val_w:.1f}" lengthAdjust="spacingAndGlyphs">'
                  f'{esc(value)}</text>')
        svg.append(f'<a xlink:href="{esc(href)}" target="_blank">{val_el}</a>' if href else val_el)
        y += row_h

    for label, value in rows_a:
        add_row(label, value)
    y += 6
    for label, value in rows_b:
        add_row(label, value)
    y += 6
    for label, value, href in rows_c:
        add_row(label, value, href)

    svg.append('</g>')
    svg.append('</svg>')
    return "\n".join(svg)


def build_all(outdir=HERE, verbose=True):
    paths = {}
    for theme in ["dark", "light"]:
        out = build_svg(theme)
        path = os.path.join(outdir, f"{theme}.svg")
        with open(path, "w") as f:
            f.write(out)
        paths[theme] = path
        if verbose:
            print(f"[svg_build] {theme}.svg -> {os.path.getsize(path)/1024:.1f} KB")
    return paths


if __name__ == "__main__":
    build_all()
