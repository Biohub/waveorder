"""Generate figures for the label-free screening tutorial.

Reads the raw OPS position and the reconstructions in ./recons/, writes
vector PDF (for the whitepaper) and PNG (for the docs site) figure pairs
into ./figures/. Figures are drawn at final print width so fonts match
the 9 pt document when included at 100% scale.
"""

from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from iohub import open_ome_zarr

matplotlib.use("Agg")
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.size"] = 8

RAW_PATH = "/hpc/projects/icd.fast.ops/ops0105_20260106/0-convert/live_imaging/phenotyping_transform.zarr/A/1/029029"
# Biohub HPC analysis directory holding the raw crops and reconstructions.
WORK = Path("/hpc/mydata/talon.chandler/2026-11-04-wo-hcs-whitepaper")
RECONS = WORK / "recons"
FIGURES = WORK / "figures"

FULL_W = 7.3  # print width (in) of a full-width figure at 1.3 cm margins
HALF_W = 3.52  # print width of a boxed half-width figure

# Zoom regions chosen from full-FOV overviews: clean monolayer, no debris.
YS = slice(384, 896)
XS = slice(1152, 1664)
T_YS, T_XS = slice(256, 768), slice(1280, 1792)  # edge-tile crop
PX_UM = 0.325  # yx pixel size in micrometers


def load_phase(name, position="029029"):
    """Load the single 2D phase image from a reconstruction zarr.

    Parameters
    ----------
    name : str
        Basename of the zarr store in ./recons/ (without extension).
    position : str, optional
        Position name inside well A/1 of the store.

    Returns
    -------
    numpy.ndarray
        2D phase image, float32.
    """
    with open_ome_zarr(RECONS / f"{name}.zarr" / f"A/1/{position}") as ds:
        return np.asarray(ds.data[0, 0, 0]).astype(np.float32)


def midband_power(img):
    """Mean squared FFT magnitude in the 0.25-0.5 Nyquist annulus."""
    f = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    n = img.shape[0]
    yy, xx = np.mgrid[-n // 2 : n // 2, -n // 2 : n // 2]
    r = np.hypot(yy, xx) / (n / 2)
    return float((f[(r > 0.25) & (r < 0.5)] ** 2).mean())


def add_scalebar(ax, n_px, length_um=25):
    """Draw a scale bar in the lower-right corner of an image axis.

    The bar and label carry a thin black outline so they stay legible
    over bright cells, and the label right-aligns with the bar end so
    a label wider than the bar overhangs inward, never off the panel.
    """
    bar_px = length_um / PX_UM
    x0, y0 = n_px - bar_px - 45, n_px - 55
    ax.plot(
        [x0, x0 + bar_px],
        [y0, y0],
        color="white",
        lw=1.5,
        path_effects=[pe.withStroke(linewidth=3, foreground="black")],
    )
    ax.text(
        x0 + bar_px,
        y0 - 10,
        f"{length_um} µm",
        color="white",
        ha="right",
        va="bottom",
        fontsize=7,
        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
    )


def panel(ax, img, title, vmin, vmax, scalebar=False):
    """Render one grayscale image panel without ticks."""
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    if scalebar:
        add_scalebar(ax, img.shape[-1])


def save(fig, stem):
    """Save a figure as vector PDF (whitepaper) and PNG (docs site)."""
    for ext, dpi in (("pdf", 300), ("png", 220)):
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png")


with open_ome_zarr(RAW_PATH) as raw_ds:
    bf_idx = raw_ds.channel_names.index("BF")
    bf = np.asarray(raw_ds.data[0, bf_idx]).astype(np.float32)  # (7, 2048, 2048)

nominal = load_phase("nominal")
lo, hi = np.percentile(nominal[YS, XS], [1, 99.5])


# --- Figure 1: overview flowchart --------------------------------------------
fig = plt.figure(figsize=(FULL_W, 2.1))


def icon_axes(rect, aspect_equal=False):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if aspect_equal:
        ax.set_aspect("equal", anchor="W")
    ax.axis("off")
    return ax


ICON_TEXT_X = 0.085  # left-aligned icon labels, all in one column

# Koehler alignment on top: centered concentric circles, text to the right.
ax = icon_axes([0.015, 0.68, 0.055, 0.27], aspect_equal=True)
for r in (0.15, 0.30, 0.46):
    ax.add_patch(plt.Circle((0.5, 0.5), r, fill=False, color="0.25", lw=1.0))
ax.plot(0.5, 0.5, marker="+", color="0.25", markersize=5)
fig.text(ICON_TEXT_X, 0.815, "Koehler\nalignment", fontsize=7, va="center", ha="left")

# Transmission optics: cone angles drawn to scale for each NA.
# half-angle = asin(NA): 23.6 deg for 0.4 (illumination, top),
# 33.4 deg for 0.55 (detection, bottom).
ax = icon_axes([0.015, 0.36, 0.055, 0.27], aspect_equal=True)
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
fig.text(ICON_TEXT_X, 0.555, "0.4 NA condenser", fontsize=7, va="center", ha="left")
fig.text(ICON_TEXT_X, 0.425, "0.55 NA, 20$\\times$\ndetection", fontsize=7, va="center", ha="left")

# z stack: 7 slices.
ax = icon_axes([0.015, 0.05, 0.055, 0.26], aspect_equal=True)
for i in range(7):
    ax.plot([0.15, 0.85], [0.08 + i * 0.14] * 2, color="0.35", lw=1.0)
fig.text(ICON_TEXT_X, 0.18, "7-slice z stack\n2 µm spacing", fontsize=7, va="center", ha="left")

# Defocus stack: all 7 planes fanned bottom-to-top in acquisition order,
# shared grayscale so the background stays consistent across slices.
stack_lo, stack_hi = np.percentile(bf[:, 640:1152, 1280:1792], [0.5, 99.5])
for k, z in enumerate(range(7)):
    ax = fig.add_axes([0.268 + 0.006 * k, 0.085 + 0.053 * k, 0.135, 0.47])
    img = bf[z, 640:1152, 1280:1792]
    ax.imshow(img, cmap="gray", vmin=stack_lo, vmax=stack_hi)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.3")
fig.text(0.345, 0.025, "7-slice defocus stack", fontsize=8, ha="center")

# WaveOrder box: config + CLI.
ax = icon_axes([0.525, 0.05, 0.245, 0.9])
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
ax = fig.add_axes([0.825, 0.13, 0.155, 0.72])
ax.imshow(nominal[YS, XS], cmap="gray", vmin=lo, vmax=hi)
ax.set_xticks([])
ax.set_yticks([])
fig.text(0.9025, 0.025, "quantitative phase", fontsize=8, ha="center")

# Arrows in the clear gaps between columns.
for x0, x1 in ((0.215, 0.255), (0.475, 0.515), (0.785, 0.818)):
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


# --- Figure: contrast flip (half-width box) -----------------------------------
flipped = load_phase("flipped")
fig, axs = plt.subplots(1, 2, figsize=(HALF_W, 2.0))
panel(axs[0], nominal[YS, XS], "correct\n(nucleoli bright)", lo, hi, scalebar=True)
panel(axs[1], flipped[YS, XS], "flipped z axis\n(nucleoli dark)", lo, hi)
save(fig, "fig_flip")

# --- Figure: regularization sweep (half-width box) ----------------------------
fig, axs = plt.subplots(1, 3, figsize=(HALF_W, 1.45))
for ax, reg, note in zip(axs, ["1e-5", "1e-3", "1e-1"], ["too noisy", "chosen", "too smooth"]):
    img = nominal if reg == "1e-3" else load_phase(f"reg_{reg}")
    plo, phi = np.percentile(img[YS, XS], [1, 99.5])
    panel(ax, img[YS, XS], f"reg {reg}\n({note})", plo, phi, scalebar=(reg == "1e-5"))
save(fig, "fig_reg")

# --- Figure: focus offset (half-width box) -------------------------------------
F_YS, F_XS = slice(384, 704), slice(1344, 1664)
fig, axs = plt.subplots(1, 3, figsize=(HALF_W, 1.45))
for ax, off, name in zip(axs, ["-2", "0", "+2"], ["offset_-2.0", None, "defocused"]):
    img = nominal if name is None else load_phase(name)
    sharp = midband_power(img[YS, XS])
    note = " (chosen)" if off == "0" else ""
    panel(ax, img[F_YS, F_XS], f"{off} slices{note}", lo, hi, scalebar=(off == "-2"))
save(fig, "fig_focus")

# --- Figure: illumination tilt on an edge FOV (half-width box) -----------------
edge_base = load_phase("edge_baseline", position="002029")
edge_tilt = load_phase("edge_apply", position="002029")
fig, axs = plt.subplots(1, 2, figsize=(HALF_W, 2.0))
for ax, img, title in zip(
    axs,
    [edge_base, edge_tilt],
    ["tilt not modeled", "tilt fitted\n(0.49 rad zenith)"],
):
    crop = img[T_YS, T_XS]
    plo, phi = np.percentile(crop, [1, 99.5])
    panel(ax, crop, title, plo, phi, scalebar=(img is edge_base))
save(fig, "fig_tilt")
