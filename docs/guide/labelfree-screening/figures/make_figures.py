"""Generate figures for the label-free screening tutorial.

Draws every figure from the 8-bit, display-scaled image crops in ./data/,
so it runs anywhere::

    uv run --no-project --with matplotlib --with pillow python make_figures.py

Writes vector PDF (for the whitepaper) and PNG (for the docs site) figure
pairs into this folder. Figures are drawn at final print width so fonts
match the 9 pt document when included at 100% scale.

The crops in ./data/ come from the raw OPS position and the reconstructions
on the Biohub HPC. To rebuild them there (needs iohub)::

    python make_figures.py --from-hpc

The Koehler alignment figure has its own script, koehler/make_figure.py.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.size"] = 8

FIGURES = Path(__file__).resolve().parent
DATA = FIGURES / "data"

RAW_PATH = "/hpc/projects/icd.fast.ops/ops0105_20260106/0-convert/live_imaging/phenotyping_transform.zarr/A/1/029029"
# Biohub HPC analysis directory holding the reconstructions.
RECONS = Path("/hpc/mydata/talon.chandler/2026-11-04-wo-hcs-whitepaper/recons")

FULL_W = 7.3  # print width (in) of a full-width figure at 1.3 cm margins
HALF_W = 3.52  # print width of a boxed half-width figure

# Zoom regions chosen from full-FOV overviews: clean monolayer, no debris.
YS = slice(384, 896)
XS = slice(1152, 1664)
T_YS, T_XS = slice(256, 768), slice(1280, 1792)  # edge-tile crop
F_YS, F_XS = slice(384, 704), slice(1344, 1664)  # focus-sweep crop
S_YS, S_XS = slice(640, 1152), slice(1280, 1792)  # defocus-stack crop
PX_UM = 0.325  # yx pixel size in micrometers
N_WIDE, N_FOCUS = 512, 320  # source pixels across the wide and focus crops


def refresh_from_hpc():
    """Rebuild the crops in ./data/ from the raw data and reconstructions.

    Each crop is scaled to the contrast limits its panel is shown with and
    stored as an 8-bit grayscale PNG. Requires Biohub HPC access.
    """
    from iohub import open_ome_zarr

    def load_phase(name, position="029029"):
        with open_ome_zarr(RECONS / f"{name}.zarr" / f"A/1/{position}") as ds:
            return np.asarray(ds.data[0, 0, 0]).astype(np.float32)

    def write(name, img, lo, hi):
        scaled = np.clip((img - lo) / (hi - lo), 0, 1) * 255
        Image.fromarray(np.rint(scaled).astype(np.uint8)).save(DATA / f"{name}.png")
        print(f"wrote data/{name}.png")

    DATA.mkdir(exist_ok=True)
    with open_ome_zarr(RAW_PATH) as raw_ds:
        bf_idx = raw_ds.channel_names.index("BF")
        bf = np.asarray(raw_ds.data[0, bf_idx]).astype(np.float32)  # (7, 2048, 2048)
    # shared grayscale so the background stays consistent across slices
    stack = bf[:, S_YS, S_XS]
    stack_lo, stack_hi = np.percentile(stack, [0.5, 99.5])
    for z in range(7):
        write(f"stack_{z}", stack[z], stack_lo, stack_hi)

    nominal = load_phase("nominal")
    lo, hi = np.percentile(nominal[YS, XS], [1, 99.5])
    write("nominal", nominal[YS, XS], lo, hi)
    write("flipped", load_phase("flipped")[YS, XS], lo, hi)
    for reg in ("1e-5", "1e-1"):
        crop = load_phase(f"reg_{reg}")[YS, XS]
        write(f"reg_{reg}", crop, *np.percentile(crop, [1, 99.5]))
    for off, name in (("-2", "offset_-2.0"), ("0", "nominal"), ("+2", "defocused")):
        write(f"focus_{off}", load_phase(name)[F_YS, F_XS], lo, hi)
    for key, name in (("edge_baseline", "edge_baseline"), ("edge_tilt", "edge_apply")):
        crop = load_phase(name, position="002029")[T_YS, T_XS]
        write(key, crop, *np.percentile(crop, [1, 99.5]))


def load(name):
    """Load one display-scaled crop from ./data/.

    Parameters
    ----------
    name : str
        Basename of the PNG in ./data/ (without extension).

    Returns
    -------
    numpy.ndarray
        2D uint8 image.
    """
    return np.asarray(Image.open(DATA / f"{name}.png").convert("L"))


def show(ax, img):
    """Show a display-scaled crop in grayscale without ticks."""
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])


def field_stop_view(img, radius=0.19, n_blades=10, tilt_deg=11):
    """Show an image through a centered, focused field stop.

    Matches the "Center" inset of the Koehler figure: a bright bladed
    aperture on a dark field.

    Parameters
    ----------
    img : numpy.ndarray
        2D uint8 image filling the camera field of view.
    radius : float, optional
        Aperture radius as a fraction of the image width.
    n_blades : int, optional
        Number of aperture blades.
    tilt_deg : float, optional
        Rotation of the blades in degrees.

    Returns
    -------
    numpy.ndarray
        2D uint8 image, black outside the aperture.
    """
    n = 600
    t = np.radians(tilt_deg + np.arange(n_blades) * 360 / n_blades)
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).polygon(list(zip(n * (0.5 + radius * np.cos(t)), n * (0.5 + radius * np.sin(t)))), fill=255)
    big = np.asarray(Image.fromarray(img).resize((n, n), Image.LANCZOS), float)
    return np.rint(big * np.asarray(mask) / 255).astype(np.uint8)


def add_scalebar(ax, n_px, n_src, length_um=25):
    """Draw a scale bar in the lower-right corner of an image axis.

    The bar and label carry a thin black outline so they stay legible
    over bright cells, and the label right-aligns with the bar end so
    a label wider than the bar overhangs inward, never off the panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Image axis.
    n_px : int
        Width of the displayed image in pixels.
    n_src : int
        Width of the same field in source pixels of size ``PX_UM``.
    length_um : float, optional
        Scale bar length in micrometers.
    """
    k = n_px / n_src  # displayed pixels per source pixel
    bar_px = length_um / PX_UM * k
    x0, y0 = n_px - bar_px - 45 * k, n_px - 55 * k
    ax.plot(
        [x0, x0 + bar_px],
        [y0, y0],
        color="white",
        lw=1.5,
        path_effects=[pe.withStroke(linewidth=3, foreground="black")],
    )
    ax.text(
        x0 + bar_px,
        y0 - 10 * k,
        f"{length_um} µm",
        color="white",
        ha="right",
        va="bottom",
        fontsize=7,
        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
    )


def panel(ax, name, title, n_src=N_WIDE, scalebar=False):
    """Render one grayscale image panel from a crop in ./data/."""
    img = load(name)
    show(ax, img)
    ax.set_title(title, fontsize=8)
    if scalebar:
        add_scalebar(ax, img.shape[-1], n_src)


def save(fig, stem):
    """Save a figure as vector PDF (whitepaper) and PNG (docs site)."""
    for ext, dpi in (("pdf", 300), ("png", 220)):
        # no creation date, so reruns give byte-identical PDFs
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", transparent=True, metadata=meta)
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png")


def draw_mda_zstack_panel(ax, z_start, z_end, z_step):
    """Draw the Micro-Manager MDA "Z-stacks (slices)" group box.

    Mimics the Multi-Dimensional Acquisition dialog, laid out compactly so
    the text stays legible at print size.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes; its aspect ratio should be 440:330.
    z_start, z_end, z_step : str
        Values shown in the three text fields.
    """
    w, h = 440, 330
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    pt = ax.get_position().width * ax.figure.get_figwidth() * 72 / w
    gui = dict(family="DejaVu Sans", va="center")  # bundled with matplotlib, so it renders the same anywhere
    box = matplotlib.patches.FancyBboxPatch

    def rounded(x, y, bw, bh, fc, ec, r=4, lw=0.5, z=2):
        ax.add_patch(box((x, y), bw, bh, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))

    def checkbox(x, y, size=30):
        rounded(x, y - size / 2, size, size, "white", "#8e8e8e", r=4)
        ax.plot(
            [x + 0.22 * size, x + 0.43 * size, x + 0.80 * size],
            [y, y + 0.24 * size, y - 0.27 * size],
            color="#1f3f99",
            lw=0.11 * size * pt,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )

    ax.add_patch(plt.Rectangle((0, 0), w, h, fc="#efefef", ec="none", zorder=0))
    rounded(8, 26, w - 16, h - 34, "none", "#c4c4c4", r=5, lw=0.6, z=1)
    ax.add_patch(plt.Rectangle((22, 10), 300, 32, fc="#efefef", ec="none", zorder=1.5))
    checkbox(30, 26)
    ax.text(70, 26, "Z-stacks (slices)", fontsize=27 * pt, weight="bold", color="#1e1e1e", **gui)
    rows = (("Z-start [um]", z_start, True), ("Z-end [um]", z_end, True), ("Z-step [um]", z_step, False))
    for i, (label, value, has_set) in enumerate(rows):
        y = 82 + 50 * i
        ax.text(34, y, label, fontsize=24 * pt, color="#111111", **gui)
        rounded(192, y - 21, 112, 42, "white", "#b4b4b4", r=2)
        ax.text(202, y, value, fontsize=24 * pt, color="#111111", **gui)
        if has_set:
            rounded(312, y - 20, 102, 40, "#f5f5f5", "#c6c6c6", r=5)
            ax.text(363, y, "Set", fontsize=24 * pt, color="#9a9a9a", ha="center", **gui)
    rounded(34, 212, 236, 40, "#e2e2e2", "#a6a6a6", r=4)
    ax.text(46, 232, "relative Z", fontsize=24 * pt, color="#111111", **gui)
    ax.add_patch(plt.Polygon([[236, 226], [256, 226], [246, 239]], fc="#111111", ec="none", zorder=3))
    checkbox(96, 290, size=28)
    ax.text(136, 290, "Keep shutter open", fontsize=24 * pt, color="#111111", **gui)


def make_overview():
    """Figure 1: acquisition settings, defocus stack, WaveOrder, phase."""
    fig = plt.figure(figsize=(FULL_W, 2.1))

    def icon_axes(rect, aspect_equal=False):
        ax = fig.add_axes(rect)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if aspect_equal:
            ax.set_aspect("equal", anchor="W")
        ax.axis("off")
        return ax

    # Three icons of equal width in one column, labels left-aligned beside them.
    ICON_X, ICON_W = 0.015, 0.085
    ICON_TEXT_X = ICON_X + ICON_W + 0.012
    square = ICON_W * FULL_W / 2.1  # height of a square icon, in figure units

    # Koehler alignment on top: the centered, focused field stop of Figure 2.
    ax = fig.add_axes([ICON_X, 0.655, ICON_W, square])
    show(ax, field_stop_view(load("stack_3")))
    for spine in ax.spines.values():
        spine.set_color("0.3")
        spine.set_linewidth(0.5)
    fig.text(ICON_TEXT_X, 0.655 + square / 2, "Koehler\nalignment", fontsize=7, va="center", ha="left")

    # Transmission optics: cone angles drawn to scale for each NA.
    # half-angle = asin(NA): 23.6 deg for 0.4 (illumination, top),
    # 33.4 deg for 0.55 (detection, bottom).
    ax = icon_axes([ICON_X, 0.30, ICON_W, square], aspect_equal=True)
    tan_ill = np.tan(np.arcsin(0.4))
    tan_det = np.tan(np.arcsin(0.55))
    # trapezoids: cone sides truncated near the sample plane (y = 0.5)
    ill_near, ill_far = 0.70, 0.95  # condenser above the sample
    det_near, det_far = 0.30, 0.05  # objective below the sample
    ax.add_patch(
        plt.Polygon(
            [
                [0.5 - (ill_near - 0.5) * tan_ill, ill_near],
                [0.5 + (ill_near - 0.5) * tan_ill, ill_near],
                [0.5 + (ill_far - 0.5) * tan_ill, ill_far],
                [0.5 - (ill_far - 0.5) * tan_ill, ill_far],
            ],
            color="0.45",
        )
    )
    ax.add_patch(
        plt.Polygon(
            [
                [0.5 - (0.5 - det_near) * tan_det, det_near],
                [0.5 + (0.5 - det_near) * tan_det, det_near],
                [0.5 + (0.5 - det_far) * tan_det, det_far],
                [0.5 - (0.5 - det_far) * tan_det, det_far],
            ],
            color="0.25",
        )
    )
    ax.plot([0.08, 0.92], [0.5, 0.5], color="black", lw=1.0)
    fig.text(ICON_TEXT_X, 0.58, "0.4 NA condenser", fontsize=7, va="center", ha="left")
    fig.text(ICON_TEXT_X, 0.36, "0.55 NA, 20$\\times$\ndetection", fontsize=7, va="center", ha="left")

    # z stack: the Micro-Manager MDA settings that give 7 slices at 2 um.
    mda_h = square * 330 / 440
    ax = fig.add_axes([ICON_X, 0.03, ICON_W, mda_h])
    draw_mda_zstack_panel(ax, "-6", "6", "2")
    fig.text(ICON_TEXT_X, 0.03 + mda_h / 2, "7-slice z stack\n2 µm spacing", fontsize=7, va="center", ha="left")

    # Defocus stack: all 7 planes fanned bottom-to-top in acquisition order.
    for k in range(7):
        ax = fig.add_axes([0.29 + 0.006 * k, 0.085 + 0.053 * k, 0.135, 0.47])
        show(ax, load(f"stack_{k}"))
        for spine in ax.spines.values():
            spine.set_color("0.3")
    fig.text(0.3755, 0.025, "7-slice defocus stack", fontsize=8, ha="center")

    # WaveOrder box: config + CLI.
    ax = icon_axes([0.515, 0.05, 0.245, 0.9])
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, color="0.3", lw=1.0))
    ax.text(0.5, 0.885, "WaveOrder", fontsize=9, ha="center", weight="bold")
    ax.text(
        0.08,
        0.56,
        "$ cat config.yml\n...\nyx_pixel_size:   0.325\nz_pixel_size:    2.0\nNA_detection:    0.55\nNA_illumination: 0.4\nz_focus_offset:  auto\n...",
        fontsize=6.5,
        va="center",
    )
    ax.text(
        0.08,
        0.19,
        "$ waveorder reconstruct \\\n    -i data.zarr \\\n    -c config.yml \\\n    -o phase.zarr",
        fontsize=6.5,
        va="center",
    )

    # Reconstruction.
    show(fig.add_axes([0.815, 0.13, 0.165, 0.72]), load("nominal"))
    fig.text(0.8975, 0.025, "quantitative phase", fontsize=8, ha="center")

    # Arrows in the clear gaps between columns.
    for x0, x1 in ((0.253, 0.283), (0.475, 0.505), (0.775, 0.805)):
        fig.add_artist(
            matplotlib.patches.FancyArrowPatch(
                (x0, 0.5),
                (x1, 0.5),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=10,
                color="0.15",
                lw=1.2,
            )
        )

    save(fig, "fig_overview")


def make_failure_modes():
    """Half-width figures for the four failure-mode boxes."""
    # contrast flip
    fig, axs = plt.subplots(1, 2, figsize=(HALF_W, 2.0))
    panel(axs[0], "nominal", "correct\n(nucleoli bright)", scalebar=True)
    panel(axs[1], "flipped", "flipped z axis\n(nucleoli dark)")
    save(fig, "fig_flip")

    # regularization sweep
    fig, axs = plt.subplots(1, 3, figsize=(HALF_W, 1.45))
    for ax, reg, note in zip(axs, ["1e-5", "1e-3", "1e-1"], ["too noisy", "chosen", "too smooth"]):
        name = "nominal" if reg == "1e-3" else f"reg_{reg}"
        panel(ax, name, f"reg {reg}\n({note})", scalebar=(reg == "1e-5"))
    save(fig, "fig_reg")

    # focus offset
    fig, axs = plt.subplots(1, 3, figsize=(HALF_W, 1.45))
    for ax, off in zip(axs, ["-2", "0", "+2"]):
        note = " (chosen)" if off == "0" else ""
        panel(ax, f"focus_{off}", f"{off} slices{note}", n_src=N_FOCUS, scalebar=(off == "-2"))
    save(fig, "fig_focus")

    # illumination tilt on an edge FOV
    fig, axs = plt.subplots(1, 2, figsize=(HALF_W, 2.0))
    panel(axs[0], "edge_baseline", "tilt not modeled", scalebar=True)
    panel(axs[1], "edge_tilt", "tilt fitted\n(0.49 rad zenith)")
    save(fig, "fig_tilt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-hpc", action="store_true", help="rebuild ./data/ from the Biohub HPC stores first")
    if parser.parse_args().from_hpc:
        refresh_from_hpc()
    make_overview()
    make_failure_modes()
