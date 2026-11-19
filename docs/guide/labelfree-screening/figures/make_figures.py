"""Generate the data figures for the label-free screening tutorial.

Draws every figure from the 8-bit image crops in ./data/, so it needs no
data outside this folder. See ../README.md for the exact command and the
pinned package versions.

Writes vector PDF (for the whitepaper) and PNG (for the docs site) figure
pairs into this folder. Figures are drawn at their print size with 8 pt
labels: the overview at the full text width, the failure-mode figures at
the size they take inside their boxes, so the whitepaper includes them
unscaled.

The crops in ./data/ are cut from one raw OPS position and from
reconstructions of it on the Biohub HPC. ``--from-hpc`` rebuilds them there
(needs iohub); the paths are options because they are specific to that
system.

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

# Defaults for --from-hpc: the raw OPS position, and the folder of
# reconstructions of it (one <name>.zarr per variant, see refresh_from_hpc).
RAW_PATH = "/hpc/projects/icd.fast.ops/ops0105_20260106/0-convert/live_imaging/phenotyping_transform.zarr/A/1/029029"
RECONS = "/hpc/mydata/talon.chandler/2026-11-04-wo-hcs-whitepaper/recons"

FULL_W = 7.5  # print width (in) of a full-width figure: letter paper, 1.25 cm margins
BOX_W = 2.9  # print width (in) of a figure inside a failure-mode box
N_SLICES = 7  # slices in the defocus stack
CROP_PX = 256  # stored crop size; more than any panel needs at print resolution

# Zoom regions chosen from full-FOV overviews: clean monolayer, no debris.
YS = slice(384, 896)
XS = slice(1152, 1664)
T_YS, T_XS = slice(256, 768), slice(1280, 1792)  # edge-tile crop
F_YS, F_XS = slice(384, 704), slice(1344, 1664)  # focus-sweep crop
S_YS, S_XS = slice(640, 1152), slice(1280, 1792)  # defocus-stack crop
PX_UM = 0.325  # yx pixel size in micrometers
N_WIDE, N_FOCUS = 512, 320  # source pixels across the wide and focus crops

# Frame colors for the failure-mode panels: red = failed result, green = good result.
FAIL, GOOD = "#d62728", "#2ca02c"


def refresh_from_hpc(raw_path=RAW_PATH, recons=RECONS):
    """Rebuild the crops in ./data/ from the raw data and reconstructions.

    Each crop is scaled to the contrast limits its panel is shown with,
    resampled to ``CROP_PX`` pixels and stored as an 8-bit grayscale PNG.

    Parameters
    ----------
    raw_path : str
        Raw OME-Zarr position with a ``BF`` channel of ``N_SLICES`` slices.
    recons : str
        Folder holding one ``<name>.zarr`` phase reconstruction per panel:
        ``nominal`` (the "correct", "reg 1e-3" and "0 slices" panels),
        ``flipped``, ``reg_1e-5``, ``reg_1e-1``, ``offset_-2.0`` ("-2
        slices"), ``defocused`` ("+2 slices"), and, for edge position
        002029, ``edge_baseline`` ("tilt not modeled") and ``edge_apply``
        ("tilt fitted").
    """
    from iohub import open_ome_zarr

    def load_phase(name, position="029029"):
        with open_ome_zarr(Path(recons) / f"{name}.zarr" / f"A/1/{position}") as ds:
            return np.asarray(ds.data[0, 0, 0]).astype(np.float32)

    def write(name, img, lo, hi):
        scaled = np.rint(np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(scaled).resize((CROP_PX, CROP_PX), Image.LANCZOS).save(DATA / f"{name}.png")
        print(f"wrote data/{name}.png")

    DATA.mkdir(exist_ok=True)
    with open_ome_zarr(raw_path) as raw_ds:
        bf_idx = raw_ds.channel_names.index("BF")
        bf = np.asarray(raw_ds.data[0, bf_idx]).astype(np.float32)  # (N_SLICES, 2048, 2048)
    # shared grayscale so the background stays consistent across slices
    stack = bf[:, S_YS, S_XS]
    stack_lo, stack_hi = np.percentile(stack, [0.5, 99.5])
    for z in range(N_SLICES):
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
    # inset from the right and bottom edges by 45 and 55 source pixels
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
        y0 - 10 * k,  # label 10 source pixels above the bar
        f"{length_um} µm",
        color="white",
        ha="right",
        va="bottom",
        fontsize=7,
        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
    )


def panel(ax, name, title, n_src=N_WIDE, scalebar=False, outline=None):
    """Render one grayscale image panel from a crop in ./data/.

    ``outline`` colors the panel frame (``FAIL`` red / ``GOOD`` green in
    the failure-mode figures).
    """
    img = load(name)
    show(ax, img)
    ax.set_title(title, fontsize=8)
    if outline is not None:
        for spine in ax.spines.values():
            spine.set_color(outline)
            spine.set_linewidth(1.2)
    if scalebar:
        add_scalebar(ax, img.shape[-1], n_src)


def save(fig, stem, tight=True):
    """Save a figure as vector PDF (whitepaper) and PNG (docs site).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save and close.
    stem : str
        Output basename, without extension.
    tight : bool, optional
        Crop to the drawn content. Turn off to keep the exact figure size.
    """
    bbox = "tight" if tight else None
    # 220 dpi shows every stored pixel of the crops at print size and keeps the PDFs small
    for ext, dpi in (("pdf", 220), ("png", 220)):
        # no creation date, so reruns give byte-identical PDFs
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=dpi, bbox_inches=bbox, transparent=True, metadata=meta)
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
    step_x, step_y, size_x, size_y = 0.006, 0.053, 0.135, 0.47  # fan offsets and slice size, figure units
    for k in range(N_SLICES):
        ax = fig.add_axes([0.29 + step_x * k, 0.085 + step_y * k, size_x, size_y])
        img = load(f"stack_{k}").copy()
        if k < N_SLICES - 1:
            # blank the part hidden behind the next slice; it is never seen and only bloats the files
            h, w = img.shape
            img[: int(h * (1 - step_y / size_y)) - 2, int(w * step_x / size_x) + 2 :] = 128
        show(ax, img)
        for spine in ax.spines.values():
            spine.set_color("0.3")
    fig.text(0.3755, 0.025, "7-slice defocus stack", fontsize=8, ha="center")

    # WaveOrder box: config + CLI.
    ax = icon_axes([0.497, 0.05, 0.3, 0.9])
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, color="0.3", lw=1.0))
    ax.text(0.5, 0.885, "WaveOrder", fontsize=9, ha="center", weight="bold")
    config = [
        "$ cat config.yml",
        "...",
        "yx_pixel_size: 0.325",
        "z_pixel_size: 2.0",
        "numerical_aperture_detection: 0.55",
        "numerical_aperture_illumination: 0.4",
        "z_focus_offset: auto",
        "...",
    ]
    command = [
        "$ waveorder reconstruct \\",
        "    -i data.zarr/A/1/029029 \\",
        "    -c config.yml \\",
        "    -o phase.zarr",
    ]
    ax.text(0.045, 0.56, "\n".join(config), fontsize=6.5, va="center")
    ax.text(0.045, 0.19, "\n".join(command), fontsize=6.5, va="center")

    # Reconstruction.
    show(fig.add_axes([0.83, 0.13, 0.16, 0.72]), load("nominal"))
    fig.text(0.91, 0.025, "quantitative phase", fontsize=8, ha="center")

    # Arrows in the clear gaps between columns.
    for x0, x1 in ((0.253, 0.283), (0.466, 0.494), (0.801, 0.825)):
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

    save(fig, "fig_overview", tight=False)  # exact width, included at the full text width


def make_failure_modes():
    """Figures for the four failure-mode boxes, at their print size."""
    two, three = (BOX_W, 1.15), (BOX_W, 1.0)  # sizes of the 2- and 3-panel figures (in)

    # contrast flip
    fig, axs = plt.subplots(1, 2, figsize=two)
    panel(axs[0], "flipped", "flipped z axis\n(nucleoli dark)", scalebar=True, outline=FAIL)
    panel(axs[1], "nominal", "correct\n(nucleoli bright)", outline=GOOD)
    save(fig, "fig_flip")

    # regularization sweep
    fig, axs = plt.subplots(1, 3, figsize=three)
    for ax, reg, note in zip(axs, ["1e-5", "1e-3", "1e-1"], ["too noisy", "chosen", "too smooth"]):
        name = "nominal" if reg == "1e-3" else f"reg_{reg}"
        outline = GOOD if reg == "1e-3" else FAIL
        panel(ax, name, f"reg {reg}\n({note})", scalebar=(reg == "1e-5"), outline=outline)
    save(fig, "fig_reg")

    # focus offset
    fig, axs = plt.subplots(1, 3, figsize=three)
    for ax, off in zip(axs, ["-2", "0", "+2"]):
        note = "(chosen)" if off == "0" else ""
        outline = GOOD if off == "0" else FAIL
        panel(ax, f"focus_{off}", f"{off} slices\n{note}", n_src=N_FOCUS, scalebar=(off == "-2"), outline=outline)
    save(fig, "fig_focus")

    # illumination tilt on an edge FOV
    fig, axs = plt.subplots(1, 2, figsize=two)
    panel(axs[0], "edge_baseline", "tilt not modeled\n", scalebar=True, outline=FAIL)
    panel(axs[1], "edge_tilt", "tilt fitted\n(0.49 rad zenith)", outline=GOOD)
    save(fig, "fig_tilt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-hpc", action="store_true", help="rebuild ./data/ from the Biohub HPC stores first")
    parser.add_argument("--raw", default=RAW_PATH, help="raw OME-Zarr position used by --from-hpc")
    parser.add_argument("--recons", default=RECONS, help="folder of reconstructions used by --from-hpc")
    args = parser.parse_args()
    if args.from_hpc:
        refresh_from_hpc(args.raw, args.recons)
    make_overview()
    make_failure_modes()
