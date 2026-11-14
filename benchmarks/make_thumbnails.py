"""Build the raw/recon thumbnail grid in benchmarks/benchmark_datasets.png.

Two rows (raw input, reconstruction), one column per benchmark case, from a ``wo bm run --scope all --save-all``
run. Each case directory is self-describing: ``cli_command.sh`` holds the exact input position path (cropped input
when the case crops) and ``config.yml`` names the input channel.

Usage::

    python benchmarks/make_thumbnails.py [RUN_DIR] [OUT_PNG]

``RUN_DIR`` defaults to the newest ``*_regression`` run under ``$WAVEORDER_BENCH_OUTPUT``; ``OUT_PNG`` defaults to
``benchmark_datasets.png``. The 3D panels show the highest-contrast z-slice; flat-filled borders (e.g. sheared
oblique registrations) are trimmed with one shared window per column so raw and recon stay pixel-aligned.
"""

import glob
import logging
import os
import re
import shlex
import sys
from pathlib import Path

import numpy as np
import yaml
from iohub import open_ome_zarr

logger = logging.getLogger("make_thumbnails")

SHORT_NAMES = {
    "phase_3d_beads": "beads phase 3d",
    "fluor_3d_beads": "beads fluor 3d",
    "shift_variant_grid_beads": "shift-var grid",
    "ops_pheno_center": "ops pheno 2d",
    "ops_pheno_center_opt": "ops pheno 2d opt",
    "ops_pheno_phase_3d": "ops pheno 3d",
    "ops_track_phase_2d": "ops track 2d",
    "ops_track_phase_3d": "ops track 3d",
    "ops_fluor_gfp": "ops fluor gfp",
    "a549_phase_2d": "a549 phase 2d",
    "a549_phase_3d": "a549 phase 3d",
    "cardiomyocyte_phase_3d": "cardio phase 3d",
    "zebrafish_daxi_phase_3d": "zebrafish phase",
    "ev71_phase_3d": "ev71 phase 3d",
    "a549_fluor_3d": "a549 fluor 3d",
    "neuromast_epi_phase_3d": "neuromast epi",
    "neuromast_ls_phase_3d": "neuromast ls",
    "neuromast_tile_phase_3d": "neuromast ls tiled",
    "mantis_v1_neuromast_phase_3d": "neuromast v1",
}

STRETCH_OVERRIDES = {
    "phase_3d_beads": (1, 99.9),
    "fluor_3d_beads": (1, 99.9),
    "shift_variant_grid_beads": (1, 99.9),
}


def middle_slice(czyx: np.ndarray, c_idx: int) -> np.ndarray:
    """Return the highest-contrast z-slice of channel ``c_idx`` from a CZYX array."""
    zyx = np.asarray(czyx[c_idx])
    stds = [zyx[z].std() for z in range(zyx.shape[0])]
    return zyx[int(np.argmax(stds))]


def flat_border_slices(img: np.ndarray) -> tuple[slice, slice]:
    """Return (y, x) slices that trim flat-filled border rows/columns.

    Registered oblique acquisitions carry sheared average-filled bands (e.g. the neuromast LS channels); rows/cols
    whose std is far below the median are trimmed. Computed from the RAW image and applied to both rows of a column
    so raw and recon stay pixel-aligned.
    """
    slices = []
    for axis in (0, 1):
        std = img.std(axis=1 - axis)
        valid = np.where(std > 0.2 * np.median(std))[0]
        slices.append(slice(valid[0], valid[-1] + 1) if len(valid) else slice(None))
    return slices[0], slices[1]


def square_crop(img: np.ndarray) -> np.ndarray:
    """Center-crop a 2D image to a square."""
    y, x = img.shape
    s = min(y, x)
    y0, x0 = (y - s) // 2, (x - s) // 2
    return img[y0 : y0 + s, x0 : x0 + s]


def thumbnail(img: np.ndarray, size: int = 256, stretch: tuple[float, float] = (5, 95)) -> np.ndarray:
    """Block-average a square image down to ~size, then percentile-stretch.

    Downsampling before the stretch means the percentiles reflect the pixels actually displayed, so fine
    low-amplitude texture (e.g. 3D phase) is not washed out by full-resolution outliers.
    """
    f = max(1, img.shape[0] // size)
    m = (img.shape[0] // f) * f
    small = img[:m, :m].reshape(m // f, f, m // f, f).mean(axis=(1, 3))
    lo, hi = np.percentile(small, list(stretch))
    if hi <= lo:
        return np.zeros_like(small)
    return np.clip((small - lo) / (hi - lo), 0, 1)


def load_position(path: str) -> tuple[np.ndarray, list[str]]:
    """Open an OME-Zarr position path and return (CZYX t=0, channel names)."""
    pos = open_ome_zarr(path, layout="fov", mode="r")
    return np.array(pos["0"][0]), list(pos.channel_names)


def load_first_position(path: str) -> tuple[np.ndarray, list[str]]:
    """Open an OME-Zarr plate and return its first position's (CZYX t=0, channel names)."""
    pos = next(iter(open_ome_zarr(path, layout="hcs", mode="r").positions()))[1]
    return np.array(pos["0"][0]), list(pos.channel_names)


def raw_slice(case_dir: Path) -> np.ndarray:
    """Return the input's highest-contrast slice of the reconstructed channel (the simulation for recovery cases)."""
    command = case_dir / "cli_command.sh"
    if not command.exists():  # recovery cases skip wo rec; show the simulated input
        czyx, names = load_first_position(str(case_dir / "simulated.zarr"))
        return middle_slice(czyx, 0)
    match = re.search(r"wo rec (.*)", command.read_text())
    if match is None:
        raise ValueError(f"{command} holds no wo rec command")
    input_path = shlex.split(match.group(1))[1]
    channel = yaml.safe_load((case_dir / "config.yml").read_text())["input_channel_names"][0]
    czyx, names = load_position(input_path)
    return middle_slice(czyx, names.index(channel))


def main() -> None:
    """Draw the grid of the run in ``sys.argv[1]`` (or the newest regression run) into ``sys.argv[2]``."""
    import matplotlib.pyplot as plt
    from PIL import Image

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plt.rcParams["font.family"] = "monospace"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "benchmark_datasets.png"
    run_dir = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else sorted(glob.glob(f"{os.environ['WAVEORDER_BENCH_OUTPUT']}/*_regression"))[-1]
    )
    logger.info("run: %s", run_dir)
    case_order = list(yaml.safe_load((run_dir / "experiment.yml").read_text())["cases"])

    panels = []
    for case in case_order:
        case_dir = run_dir / "cases" / case
        raw = recon = None
        try:
            raw = raw_slice(case_dir)
        except (OSError, KeyError, ValueError) as error:
            logger.warning("%s: raw failed: %s", case, error)
        try:
            recon = middle_slice(load_first_position(str(case_dir / "reconstruction.zarr"))[0], 0)
        except (OSError, KeyError, ValueError) as error:
            logger.warning("%s: recon failed: %s", case, error)
        panels.append((case, raw, recon))
        logger.info(
            "%s: raw=%s recon=%s", case, None if raw is None else raw.shape, None if recon is None else recon.shape
        )

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(1.35 * n, 3.6))
    for j, (case, raw, recon) in enumerate(panels):
        ref = raw if raw is not None else recon
        trim = flat_border_slices(ref) if ref is not None else (slice(None), slice(None))
        for i, img in enumerate([raw, recon]):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if img is not None and ref is not None:
                view = img[trim] if img.shape == ref.shape else img
                stretch = STRETCH_OVERRIDES.get(case, (5, 95))
                ax.imshow(thumbnail(square_crop(view), stretch=stretch), cmap="gray", interpolation="auto")
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
        axes[0, j].set_title(SHORT_NAMES.get(case, case), fontsize=9, rotation=45, ha="left")
    axes[0, 0].set_ylabel("raw", fontsize=11)
    axes[1, 0].set_ylabel("recon", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.96, top=0.72, bottom=0.02, wspace=0.05, hspace=0.05)
    fig.savefig(out_png, dpi=130)
    Image.open(out_png).convert("L").save(out_png, optimize=True)
    logger.info("wrote %s", out_png)


if __name__ == "__main__":
    main()
