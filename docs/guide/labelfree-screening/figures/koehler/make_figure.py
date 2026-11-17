"""Draw the Koehler alignment figure for the label-free screening tutorial.

Renders a cartoon inverted microscope (see scope_model.py) from one view:
an overview, then four zoomed panels that step through realigning Koehler
illumination. Each zoomed panel carries an inset of the field stop as seen
on the camera, filled with a raw brightfield crop (../data/stack_3.png).

Needs cadquery (which brings vtk); see ../../README.md for the exact command
and pinned versions. Writes fig_koehler.pdf/.png into the parent folder (or
into the directory given as the first argument) at final print width, so
its 8 pt labels match the other tutorial figures.
"""

import math
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scope_model as ts
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image, ImageDraw, ImageFilter

# Arrow geometry is laid out in pixels of a 960 px wide panel; panels are
# rendered S times larger so the print stays sharp.
S = 2
SIZE = (960 * S, 1250 * S)  # panel size in pixels
AZ, EL = 35, 10  # camera azimuth and elevation in degrees, shared by all panels
OVERVIEW = dict(focus=(-33.6, 236.7, 407), half_height=492)  # microscope fills the panel
BLACK = "#1a1a1a"
RED = "#e2574c"
ORANGE = "#f39c34"
BLUE = "#4b83e8"
N_IN = 600  # inset image size in pixels
FULL_W = 7.5  # print width (in) of a full-width figure, as in make_figures.py
LW = 0.42  # line widths below are in points for a 5-panel row 2.4x wider than print; this scales them down
N_BLADES, BLADE_TILT = 10, 11  # field stop blades and their rotation in degrees (also in make_figures.py)
HERE = Path(__file__).resolve().parent
CELL = np.asarray(
    Image.open(HERE.parent / "data" / "stack_3.png").convert("L").resize((N_IN, N_IN), Image.LANCZOS), float
)  # raw brightfield crop shown through the field stop
OFF = (0.64, 0.38)  # off-centre field stop position in the inset


def basis(az=AZ, el=EL):
    """Screen right and up unit vectors for a view direction."""
    a, e = math.radians(az), math.radians(el)
    right = np.array([math.sin(a), math.cos(a), 0.0])
    up = np.array([-math.sin(e) * math.cos(a), math.sin(e) * math.sin(a), math.cos(e)])
    return right, up


# zoomed view: shifted off-centre to leave the lower-left corner free
ZOOM = dict(focus=tuple(np.array([0, ts.AX + 8, 623]) - basis()[0] * 3), half_height=171)


def project(view, pts):
    """Project 3D points to pixel coordinates of a rendered view.

    Parameters
    ----------
    view : dict
        ``focus`` and ``half_height`` of the orthographic view.
    pts : array-like, shape (n, 3)
        Points in model coordinates.

    Returns
    -------
    ndarray, shape (n, 2)
    """
    right, up = basis()
    rel = np.atleast_2d(pts) - np.array(view["focus"])
    k = SIZE[1] / 2 / view["half_height"]
    return np.stack([SIZE[0] / 2 + rel @ right * k, SIZE[1] / 2 - rel @ up * k], axis=1)


def arc3d(center, u, v, radius, a0, a1, n=60):
    """Points on a 3D arc spanned by unit vectors u and v."""
    t = np.radians(np.linspace(a0, a1, n))[:, None]
    return np.array(center) + radius * (np.cos(t) * np.array(u) + np.sin(t) * np.array(v))


def arrow(ax, xy, color, lw=3.2, heads=(True, True), head=(45, 36), scale=S, halo=1.3, z=5):
    """Draw a polyline with solid triangular heads and a thin white halo.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on, in pixel coordinates.
    xy : ndarray, shape (n, 2)
        Path in pixels.
    color : str
        Arrow color.
    lw : float
        Line width in points (before the `LW` print scaling).
    heads : tuple of bool
        Whether to draw a head at the start and at the end.
    head : tuple of float
        Head length and width in layout pixels (multiplied by `scale`).
    scale : float
        Pixel scale of the target axes relative to the 960 px layout.
    halo : float
        White outline width in points.
    z : int
        zorder.
    """
    xy = np.asarray(xy, float)
    lw, halo = lw * LW, halo * LW
    hl, hw = head[0] * scale, head[1] * scale
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    fine = np.linspace(0, s[-1], 300)
    xy = np.stack([np.interp(fine, s, xy[:, 0]), np.interp(fine, s, xy[:, 1])], axis=1)
    s = fine
    lo = hl * 0.8 if heads[0] else 0
    hi = s[-1] - (hl * 0.8 if heads[1] else 0)
    keep = (s >= lo) & (s <= hi)
    tris = []
    for on, tip, back in ((heads[0], xy[0], s <= hl), (heads[1], xy[-1], s >= s[-1] - hl)):
        if not on:
            continue
        d = tip - xy[back].mean(axis=0)
        d /= np.linalg.norm(d)
        base = tip - d * hl
        nrm = np.array([-d[1], d[0]]) * hw / 2
        tris.append([tip, base + nrm, base - nrm])
    passes = [(color, 0, z + 1)]
    if halo:
        passes.insert(0, ("white", 2 * halo, z))
    for c, extra, zz in passes:
        ax.plot(xy[keep, 0], xy[keep, 1], color=c, lw=lw + extra, solid_capstyle="butt", zorder=zz)
        for tri in tris:
            ax.add_patch(
                Polygon(tri, closed=True, fc=c, ec=c, lw=extra if extra else 0.15, joinstyle="miter", zorder=zz)
            )


def rotation_arrow(ax, center, axis, radius, a0, a1, color=BLACK, **kw):
    """Curved arrow around a 3D axis through `center`."""
    axis = np.array(axis, float)
    axis /= np.linalg.norm(axis)
    ref = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    arrow(ax, project(ZOOM, arc3d(center, u, v, radius, a0, a1)), color, **kw)


def ring_arrow(ax, heads):
    """Black arrow turning the knurled field diaphragm ring."""
    cam = -AZ
    below = ts.FIELD_RING_Z[0] - 9  # just under the ring, on the side facing the camera
    pts = arc3d((0, ts.AX, below), (1, 0, 0), (0, 1, 0), 55, cam - 48, cam + 48)
    arrow(ax, project(ZOOM, pts), BLACK, lw=3.0, heads=heads, head=(36, 29))


# ---------------------------------------------------------------- insets
def blades(center, radius, n=N_BLADES, rot=BLADE_TILT):
    """Vertices of a bladed (polygonal) aperture, in inset pixels."""
    t = np.radians(rot + np.arange(n) * 360 / n)
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], axis=1) * N_IN


def stop_image(center, radius, blur=0, ghost=None):
    """Bright bladed field stop on a black field.

    Parameters
    ----------
    center : tuple of float
        Aperture centre in inset fractions.
    radius : float
        Aperture radius in inset fractions.
    blur : float
        Gaussian blur radius in pixels.
    ghost : tuple, optional
        ``(radius, blur, weight)`` of a fainter defocused copy underneath.

    Returns
    -------
    ndarray, shape (N_IN, N_IN)
    """

    def one(r, b):
        im = Image.new("L", (N_IN, N_IN), 0)
        ImageDraw.Draw(im).polygon([tuple(p) for p in blades(center, r)], fill=255)
        if b:
            im = im.filter(ImageFilter.GaussianBlur(b))
        return np.asarray(im, float)

    img = one(radius, blur)
    if ghost:
        img = np.maximum(img, ghost[2] * one(ghost[0], ghost[1]))
    return img


def edge_dist(radius, ang, n=N_BLADES, rot=BLADE_TILT):
    """Distance from the aperture centre to its edge along a direction."""
    half = 180 / n
    rel = (ang - rot) % (2 * half) - half
    return radius * math.cos(math.radians(half)) / math.cos(math.radians(rel))


INSET_W = 0.25  # inset width as a fraction of the panel width


def inset_axes(ax):
    """Square inset in the lower-left corner of a panel."""
    w = INSET_W
    h = w * SIZE[0] / SIZE[1]
    ia = ax.inset_axes([0.035, 0.03, w, h])
    ia.set_xticks([])
    ia.set_yticks([])
    for sp in ia.spines.values():
        sp.set_edgecolor("#555555")
        sp.set_linewidth(0.45)
    return ia


def show(ia, mask):
    """Show the brightfield crop through a field stop mask (0-255)."""
    ia.imshow(mask / 255 * CELL, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
    ia.set_xlim(0, N_IN)
    ia.set_ylim(N_IN, 0)


def inset_close(ax):
    """Blurred, off-centre stop shrinking as the field stop closes."""
    ia = inset_axes(ax)
    show(ia, stop_image(OFF, 0.27, blur=20))
    c = np.array(OFF) * N_IN
    for ang in (45, 135, 225, 315):
        d = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        tip = edge_dist(0.27, ang)
        arrow(
            ia,
            np.array([c + d * (tip + 0.25) * N_IN, c + d * tip * N_IN]),
            RED,
            lw=3.0,
            heads=(False, True),
            scale=1.6,
            halo=0,
        )


def inset_focus(ax):
    """Off-centre stop coming into focus."""
    ia = inset_axes(ax)
    show(ia, stop_image(OFF, 0.19, ghost=(0.22, 26, 0.45)))
    # outline the focused blade edges in the focus-knob color
    ia.add_patch(Polygon(blades(OFF, 0.19), closed=True, fc="none", ec=ORANGE, lw=0.6, zorder=4))


def inset_center(ax):
    """Centred, focused stop with X/Y motion arrows."""
    ia = inset_axes(ax)
    show(ia, stop_image((0.5, 0.5), 0.19))
    c = np.array([0.5, 0.5]) * N_IN
    for d in ((1, 1), (1, -1)):
        d = np.array(d) / math.sqrt(2) * 0.4 * N_IN
        arrow(ia, np.array([c - d, c + d]), BLUE, lw=3.8, scale=2.0, halo=0)


def inset_open(ax):
    """Field of view fully lit, stop opened just past its edges.

    The square field of view sits where the other insets do. The dashed
    stop surrounds it on a white patch that fades into the drawing, and is
    cut off by the panel edge at the lower left.
    """
    w = INSET_W * SIZE[0]
    x0, y1 = 0.035 * SIZE[0], SIZE[1] * (1 - 0.03)
    c = np.array([x0 + w / 2, y1 - w / 2])
    a = w / 2
    half = 1.95 * a  # radius of the white patch: just past the stop, which reaches 1.55 a
    g = np.linspace(-1, 1, 500)
    r = np.hypot(*np.meshgrid(g, g))
    t = np.clip((1.0 - r) / 0.17, 0, 1)  # opaque inside, fading over the outer 17%
    rgba = np.ones((500, 500, 4))
    rgba[..., 3] = t * t * (3 - 2 * t)
    ax.imshow(rgba, extent=(c[0] - half, c[0] + half, c[1] + half, c[1] - half), interpolation="bilinear", zorder=2)
    ax.imshow(CELL, extent=(x0, x0 + w, y1, y1 - w), cmap="gray", vmin=0, vmax=255, interpolation="bilinear", zorder=3)
    ax.add_patch(Rectangle((x0, y1 - w), w, w, fc="none", ec=BLACK, lw=0.45, zorder=3.5))
    # stop just outside the corners of the field of view (corner distance is 1.41 a)
    rad = 1.47 * a / math.cos(math.pi / N_BLADES)
    ang = np.radians(BLADE_TILT + 360 / N_BLADES * np.arange(N_BLADES))
    ax.add_patch(
        Polygon(
            c + rad * np.stack([np.cos(ang), np.sin(ang)], 1),
            closed=True,
            fc="none",
            ec="#555555",
            lw=0.45,
            ls=(0, (4, 2.5)),
            zorder=4,
        )
    )
    for deg in (45, 135, 225, 315):
        d = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        tip = edge_dist(rad, deg)
        arrow(
            ax, np.array([c + d * (tip - 0.56 * a), c + d * tip]), RED, lw=3.0, heads=(False, True), scale=1.43, halo=0
        )


# ---------------------------------------------------------------- panels
def panel_close(ax):
    """Close the field stop: turn the ring one way; the stop image shrinks."""
    ring_arrow(ax, (False, True))
    inset_close(ax)


def panel_focus(ax):
    """Focus the condenser: turn the focus knob; the condenser moves up and down."""
    knob_face = (ts.FOCUS_KNOB_X + 7, *ts.FOCUS_KNOB_YZ)
    rotation_arrow(ax, knob_face, (1, 0, 0), 24, -60, 170)
    right, _ = basis()
    base = np.array([0, ts.TURRET_Y, 0]) - right * 110  # clear of the turret, to its left
    line = np.array([base + (0, 0, 560), base + (0, 0, 664)])
    arrow(ax, project(ZOOM, line), ORANGE, lw=4.2)
    inset_focus(ax)


def panel_center(ax):
    """Center the condenser: turn both centering screws; the condenser moves in X and Y."""
    spans = {-30: (-150, 80), -150: (100, 330)}  # arc of each arrow, on the side facing the camera
    for ang in ts.SCREW_ANGLES:
        span = spans[ang]
        a = math.radians(ang)
        d = np.array([math.cos(a), math.sin(a), 0])
        c = np.array([0, ts.AX, ts.SCREW_Z]) + d * (ts.SCREW_TIP_R + 6)
        rotation_arrow(ax, c, d, 16, *span, lw=2.8, head=(32, 27))
    # X and Y in the plane perpendicular to the light path, drawn as if
    # that plane were tipped toward the viewer so all four heads read
    c = project(ZOOM, [(0, ts.AX, ts.LENS_BOTTOM_Z - 32)])[0]
    a, tip = math.radians(AZ), math.sin(math.radians(26))
    k = SIZE[1] / 2 / ZOOM["half_height"] * 44
    for d in ((math.sin(a), math.cos(a) * tip), (math.cos(a), -math.sin(a) * tip)):
        d = np.array(d) * k
        arrow(ax, np.array([c - d, c + d]), BLUE, lw=4.2)
    inset_center(ax)


def panel_open(ax):
    """Open the field stop: turn the ring back until the stop clears the field of view."""
    ring_arrow(ax, (True, False))
    inset_open(ax)


# panel titles take the color of the knob they turn
TITLE_COLORS = {"close": RED, "focus": ORANGE, "center": BLUE, "open": RED}

PANELS = [
    ("overview", "", {"A", "B", "C"}, OVERVIEW, None),
    ("close", "Close field stop", {"A"}, ZOOM, panel_close),
    ("focus", "Focus", {"B"}, ZOOM, panel_focus),
    ("center", "Center", {"C"}, ZOOM, panel_center),
    ("open", "Open field stop", {"A"}, ZOOM, panel_open),
]


def main(out=HERE.parent):
    """Render the five panels and save fig_koehler.pdf/.png.

    Parameters
    ----------
    out : path-like, optional
        Output directory; the figures folder by default.
    """
    out = Path(out)
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.size"] = 8
    fig, axes = plt.subplots(1, 5, figsize=(FULL_W, 2.055))
    fig.subplots_adjust(left=0.004, right=0.996, top=0.917, bottom=0.007, wspace=0.05)
    cache = {}
    for ax, (name, title, hl, view, draw) in zip(axes, PANELS):
        key = (frozenset(hl), draw is not None)
        if key not in cache:
            parts = ts.build(hl)
            if draw:  # keep the zoomed panels clear of the eyepiece tube
                parts = [q for q in parts if not q[0].startswith(("eyepiece", "tube_"))]
            with tempfile.TemporaryDirectory() as tmp:
                png = Path(tmp) / f"panel_{name}.png"
                ts.render(
                    ts.mesh_scene(parts), png, view["focus"], AZ, EL, view["half_height"], size=SIZE, ss=2, line_px=7
                )
                cache[key] = plt.imread(png)
        ax.imshow(cache[key], interpolation="lanczos")
        ax.set_xlim(0, SIZE[0])
        ax.set_ylim(SIZE[1], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#555555")
            sp.set_linewidth(0.5)
            sp.set_visible(draw is not None)
        if draw:
            draw(ax)
            ax.set_xlim(0, SIZE[0])
            ax.set_ylim(SIZE[1], 0)
        ax.set_title(title, fontsize=8, pad=3, color=TITLE_COLORS.get(name, BLACK))
    # zoom box on the overview, connected to the first zoomed panel
    k = SIZE[1] / 2 / OVERVIEW["half_height"]
    cx, cy = project(OVERVIEW, [ZOOM["focus"]])[0]
    hh = ZOOM["half_height"] * k
    hw = hh * SIZE[0] / SIZE[1]
    axes[0].add_patch(
        FancyBboxPatch(
            (cx - hw, cy - hh),
            2 * hw,
            2 * hh,
            boxstyle=f"round,pad=0,rounding_size={8 * S}",
            fc="none",
            ec="#555555",
            lw=0.5,
            ls=(0, (3, 2)),
            zorder=6,
        )
    )
    fig.add_artist(
        ConnectionPatch(
            (cx - hw, cy - hh),
            (0, 0),
            "data",
            "data",
            axesA=axes[0],
            axesB=axes[1],
            color="#555555",
            lw=0.5,
            ls=(0, (3, 2)),
        )
    )
    for ext, dpi in (("pdf", 300), ("png", 300)):
        # no creation date, so reruns give byte-identical PDFs
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(out / f"fig_koehler.{ext}", dpi=dpi, metadata=meta)
    print("wrote fig_koehler.pdf/.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
