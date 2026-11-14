"""Run one benchmark case: phantom, simulate, ``wo config migrate``, ``wo rec``, metrics.

Every case shells out to the ``wo`` commands a user runs, so the benchmark proves the command-line workflow; the
``cli_command.sh`` saved with each case can be copied as is. Case configurations are waveorder 3.x files: each is
migrated to the current schema first, as 3.x files no longer run.
"""

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta

from benchmarks.config import (
    CropConfig,
    PhantomConfig,
    RecoveryConfig,
    ReferenceBound,
    SimulationConfig,
    infer_modality,
)
from benchmarks.metrics import compute_metrics
from benchmarks.phantoms import Phantom, grid_beads_gaussian, single_bead
from benchmarks.simulate import (
    field_coefficients,
    fluorescence_system,
    normalize_pupil_coefficients,
    simulate_fluorescence_3d,
    simulate_phase_3d,
    simulate_shift_variant_fluorescence_3d,
)
from benchmarks.utils import TimingTree
from waveorder.config.migrate import migrate
from waveorder.io.tf_store import TfStore
from waveorder.nn import (
    Free,
    Gaussian,
    ImagingModel,
    LaplacianVarianceScore,
    NormalizedVarianceScore,
    Sampling,
    Score,
    SpectralFlatnessScore,
    Tikhonov,
    TotalVariationScore,
    Wiener,
)
from waveorder.nn.scores import CentreSliceMidbandScore
from waveorder.nn.search.descent import GradientSearch
from waveorder.nn.systems.tiled import LEARNING_RATES, BeadFit, zernike_terms
from waveorder.tiling.plan import TilePlan
from waveorder.tiling.run import recover_zernikes

for _name in ("iohub", "iohub.ngff", "iohub.ngff.nodes"):
    logging.getLogger(_name).setLevel(logging.ERROR)

_PHANTOM_FUNCTIONS = {"single_bead": single_bead, "grid_beads_gaussian": grid_beads_gaussian}

CROPPED_INPUT_FILENAME = "cropped_input.zarr"
MIGRATED_CONFIG_FILENAME = "config_v4.yml"
RECONSTRUCTION_FILENAME = "reconstruction.zarr"

# Storage cap applied to per-case output zarrs. Small synthetic outputs (~4 MB for a bead phantom) pass through;
# large HPC 3D reconstructions and big simulations are deleted after their metrics are computed, unless save_all.
SIZE_LIMIT_BYTES = 25 * 1024 * 1024


def _extract_optics(tf_settings: dict) -> tuple[float, float, float]:
    """Return ``(NA_det, wavelength, pixel_size)`` of a 3.x ``transfer_function`` block, with 3.x's defaults."""
    na_det = tf_settings.get("numerical_aperture_detection", 1.2)
    wavelength = tf_settings.get("wavelength_illumination", tf_settings.get("wavelength_emission", 0.532))
    pixel_size = tf_settings.get("yx_pixel_size", 0.1)
    return na_det, wavelength, pixel_size


_SIM_KWARGS = {
    "phase": {
        "wavelength_illumination",
        "index_of_refraction_media",
        "numerical_aperture_illumination",
        "numerical_aperture_detection",
    },
    "fluorescence": {"wavelength_emission", "index_of_refraction_media", "numerical_aperture_detection"},
}

_DEFAULT_CHANNEL = {"phase": "Brightfield", "fluorescence": "GFP"}


def _sim_kwargs(modality: str, tf_settings: dict) -> dict:
    """Keep the ``tf_settings`` the ``simulate_<modality>_3d`` function takes."""
    return {k: v for k, v in tf_settings.items() if k in _SIM_KWARGS[modality]}


def _wo_rec_argv(
    position_path: Path,
    config_path: Path,
    recon_path: Path,
    *,
    tile: tuple[int, int, int] | None = None,
    tile_overlap: tuple[int, int, int] | None = None,
) -> list[str]:
    """Return the ``wo rec`` command line run and saved to ``cli_command.sh``."""
    argv = ["wo", "rec", "-i", str(position_path), "-c", str(config_path), "-o", str(recon_path)]
    if tile is not None:
        argv += ["--tile", ",".join(map(str, tile))]
    if tile_overlap is not None:
        argv += ["--tile-overlap", ",".join(map(str, tile_overlap))]
    return argv


def _wo_executable() -> str:
    """Return the ``wo`` of the running Python environment, so a run needs no activated environment."""
    sibling = Path(sys.executable).with_name("wo")
    return str(sibling) if sibling.is_file() else "wo"


def _run(argv: list[str]) -> None:
    """Run a ``wo`` command (``argv[0] == "wo"``), raising with its stderr on a non-zero exit."""
    env = {**os.environ, "PYTHONWARNINGS": "ignore::UserWarning,ignore::DeprecationWarning"}
    result = subprocess.run([_wo_executable(), *argv[1:]], capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{shlex.join(argv[:3])} failed (exit {result.returncode}):\n{result.stderr.strip()}")


def migrate_config(config_path: Path) -> Path:
    """Migrate a written 3.x ``config.yml`` with ``wo config migrate``; return the file to reconstruct from.

    A configuration already in the current schema is returned as is.

    Raises
    ------
    RuntimeError
        If ``wo config migrate`` refuses the configuration; the message holds its output.
    """
    if migrate(yaml.safe_load(config_path.read_text())).dialect == "4":
        return config_path
    migrated_path = config_path.with_name(MIGRATED_CONFIG_FILENAME)
    _run(["wo", "config", "migrate", str(config_path), "-o", str(migrated_path)])
    return migrated_path


def _dir_size_bytes(path: Path) -> int:
    """Sum the sizes of all files under ``path``."""
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _cleanup_large_outputs(case_dir: Path, *, save_all: bool) -> None:
    """Remove derived and oversized output zarrs unless ``save_all``.

    Transfer-function stores and ``cropped_input.zarr`` are always deleted (both are cheap to regenerate);
    ``simulated.zarr``, ``reconstruction.zarr`` and ``phantom.zarr`` only above :data:`SIZE_LIMIT_BYTES`.
    """
    if save_all:
        return
    for p in case_dir.glob("*_transfer_function_*.zarr"):
        if p.is_dir():
            shutil.rmtree(p)
    cropped = case_dir / CROPPED_INPUT_FILENAME
    if cropped.is_dir():
        shutil.rmtree(cropped)
    for name in ("simulated.zarr", RECONSTRUCTION_FILENAME, "phantom.zarr"):
        p = case_dir / name
        if p.is_dir() and _dir_size_bytes(p) > SIZE_LIMIT_BYTES:
            shutil.rmtree(p)


def _build_phantom(phantom_config: PhantomConfig | dict) -> Phantom:
    """Build the phantom a config describes."""
    if isinstance(phantom_config, dict):
        phantom_config = PhantomConfig(**phantom_config)
    kwargs = phantom_config.model_dump()
    return _PHANTOM_FUNCTIONS[kwargs.pop("function")](**kwargs)


def _write_zarr(data: torch.Tensor, path: Path, *, channel_name: str, pixel_sizes: tuple[float, float, float]) -> None:
    """Write a ``(Z, Y, X)`` tensor as a one-position HCS OME-Zarr (position ``0/0/0``) that ``wo rec`` reads."""
    czyx = data.numpy()[None, ...]
    dataset = open_ome_zarr(path, layout="hcs", mode="w", channel_names=[channel_name])
    position = dataset.create_position("0", "0", "0")
    position.create_zeros(
        "0",
        (1, *czyx.shape),
        dtype=np.float32,
        transform=[TransformationMeta(type="scale", scale=[1, 1, *(float(p) for p in pixel_sizes)])],
    )
    position["0"][0] = czyx
    dataset.close()


def _read_zarr(path: Path) -> torch.Tensor:
    """Read the first time point of the first position of a plate, ``(C, Z, Y, X)`` float32."""
    dataset = open_ome_zarr(path, layout="hcs", mode="r")
    position = next(iter(dataset.positions()))[1]
    data = torch.tensor(np.asarray(position["0"][0]), dtype=torch.float32)
    dataset.close()
    return data


def _run_and_measure(  # noqa: PLR0913 (the case and its optional ground truth and tiling)
    position_path: Path,
    *,
    recon_config: dict,
    case_dir: Path,
    tt: TimingTree,
    ground_truth: torch.Tensor | None = None,
    tile: tuple[int, int, int] | None = None,
    tile_overlap: tuple[int, int, int] | None = None,
) -> dict:
    """Write and migrate the config, run ``wo rec``, read channel 0 of the output and compute its metrics.

    Must be called inside ``with tt.time("total"):``; it adds the ``migrate``, ``reconstruct`` and ``metrics``
    timings.
    """
    modality = infer_modality(recon_config)
    tf_settings = recon_config.get(modality, {}).get("transfer_function", {})

    config_path = case_dir / "config.yml"
    config_path.write_text(yaml.dump(recon_config, default_flow_style=False))
    with tt.time("migrate"):
        exec_config_path = migrate_config(config_path)

    recon_path = case_dir / RECONSTRUCTION_FILENAME
    argv = _wo_rec_argv(position_path, exec_config_path, recon_path, tile=tile, tile_overlap=tile_overlap)
    (case_dir / "cli_command.sh").write_text(f"#!/bin/bash\n{shlex.join(argv)}\n")

    with tt.time("reconstruct"):
        _run(argv)

    with tt.time("metrics"):
        recon_data = _read_zarr(recon_path)[0]  # channel 0, (Z, Y, X); one plane for a 2D reconstruction
        na_det, wavelength, pixel_size = _extract_optics(tf_settings)
        metrics = compute_metrics(
            recon_data, NA_det=na_det, wavelength=wavelength, pixel_size=pixel_size, phantom=ground_truth
        )
    metrics["config_migrated"] = exec_config_path != config_path
    return metrics


def _finalize_case(case_dir: Path, *, tt: TimingTree, metrics: dict, save_all: bool) -> None:
    """Write ``timing.json`` and ``metrics.json`` and apply the storage cap."""
    tt.save(case_dir / "timing.json")
    (case_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _cleanup_large_outputs(case_dir, save_all=save_all)


def crop_input_position(input_path: str | Path, position: str, *, crop: CropConfig, case_dir: Path) -> Path:
    """Write a cropped copy of the first time point of an HCS position into ``case_dir``.

    Returns the path of the written plate; the position keeps its ``row/col/fov`` name, so
    ``{cropped} / position`` is what ``wo rec -i`` takes.
    """
    src = open_ome_zarr(Path(input_path) / position, layout="fov", mode="r")
    data = np.asarray(src.data[0])  # (C, Z, Y, X)
    z_sl, y_sl, x_sl = crop.slices()
    cropped = data[:, z_sl, y_sl, x_sl]

    out_path = case_dir / CROPPED_INPUT_FILENAME
    if out_path.exists():
        shutil.rmtree(out_path)
    row, col, fov = position.split("/")
    dst = open_ome_zarr(out_path, layout="hcs", mode="w", channel_names=list(src.channel_names))
    pos = dst.create_position(row, col, fov)
    pos.create_zeros(
        "0", (1, *cropped.shape), dtype=np.float32, transform=[TransformationMeta(type="scale", scale=src.scale)]
    )
    pos["0"][0] = cropped
    dst.close()
    src.close()
    return out_path


def _get_by_dotted_path(d: dict, path: str) -> object:
    """Return ``d[k1][k2]...`` for a dot-separated ``path``, or None if missing."""
    for key in path.split("."):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


PARAMETER_PREFIX = "parameter."

STORE_NAMES = {
    "z_focus_offset": "system.detection.z_focus_offset",
    "tilt_angle_zenith": "system.illumination.tilt.zenith",
    "tilt_angle_azimuth": "system.illumination.tilt.azimuth",
    "numerical_aperture_detection": "system.detection.numerical_aperture",
    "numerical_aperture_illumination": "system.illumination.numerical_aperture",
}
"""3.x ``transfer_function`` names of the ``parameter.*`` references and their dotted names in the store: the free
quantities of the model's imaging model, as :func:`waveorder.nn.free_specs` names them."""


def _load_calibrated_parameters(case_dir: Path) -> dict[str, float]:
    """Return the calibrated values ``wo rec`` stored beside the output, by 3.x name where it has one.

    ``wo rec`` writes them to ``reconstruction_transfer_function_<config>.zarr`` when the configuration calibrates.

    Raises
    ------
    ValueError
        If no store was written.
    """
    stores = sorted(case_dir.glob(f"{Path(RECONSTRUCTION_FILENAME).stem}_transfer_function_*.zarr"))
    if not stores:
        raise ValueError(
            f"reference includes parameter.* keys for case '{case_dir.name}' but wo rec wrote no calibration store; "
            "make at least one transfer_function value calibrated (e.g. z_focus_offset: {init: 0.0, lr: 0.05})"
        )
    row = TfStore.load(stores[0]).row()
    dotted_to_3x = {dotted: name for name, dotted in STORE_NAMES.items()}
    return {dotted_to_3x.get(name, name): value for name, value in row.items()}


def check_reference(case_dir: Path, metrics: dict, reference: dict[str, ReferenceBound] | None) -> dict | None:
    """Check the ``reference`` bounds against the metrics and the calibrated parameters.

    A ``parameter.<name>`` path is looked up among the values ``wo rec`` calibrated (:data:`STORE_NAMES` maps the
    3.x names); any other path is a dotted key into ``metrics``. Returns None without bounds.
    """
    if not reference:
        return None

    parameter_values: dict[str, float] | None = None
    per_ref = {}
    all_pass = True
    for path, bound in reference.items():
        if path.startswith(PARAMETER_PREFIX):
            if parameter_values is None:
                parameter_values = _load_calibrated_parameters(case_dir)
            param_name = path[len(PARAMETER_PREFIX) :]
            if param_name not in parameter_values:
                raise ValueError(
                    f"reference['{path}'] is not among the calibrated parameters {sorted(parameter_values)}"
                )
            value = parameter_values[param_name]
        else:
            value = _get_by_dotted_path(metrics, path)
            if value is None:
                raise ValueError(f"reference['{path}'] not found in computed metrics")

        passed = (bound.min is None or value >= bound.min) and (bound.max is None or value <= bound.max)
        per_ref[path] = {"value": float(value), "min": bound.min, "max": bound.max, "pass": passed}
        all_pass = all_pass and passed

    return {"per_ref": per_ref, "all_pass": all_pass}


def run_synthetic_case(  # noqa: PLR0913 (one argument per case field)
    phantom_config: PhantomConfig | dict,
    recon_config: dict,
    case_dir: Path,
    *,
    modality: str = "phase",
    save_all: bool = False,
    reference: dict[str, ReferenceBound] | None = None,
    simulation: SimulationConfig | None = None,
    recovery: RecoveryConfig | None = None,
) -> dict:
    """Run a synthetic case: build the phantom, simulate the data, reconstruct with ``wo rec``, measure.

    Parameters
    ----------
    phantom_config : PhantomConfig or dict
        Phantom parameters (see the experiment YAML).
    recon_config : dict
        3.x reconstruction configuration.
    case_dir : Path
        Output directory of the case.
    modality : str
        ``"phase"`` or ``"fluorescence"``.
    save_all : bool
        Keep every output; by default transfer-function stores are deleted and ``simulated.zarr`` /
        ``reconstruction.zarr`` above :data:`SIZE_LIMIT_BYTES` too, after the metrics.
    reference : dict, optional
        Bounds on metrics and calibrated parameters.
    simulation : SimulationConfig, optional
        Forward-model override (shift-variant simulation).
    recovery : RecoveryConfig, optional
        Per-tile Zernike recovery in place of ``wo rec``.

    Returns
    -------
    dict
        The case's metrics.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    tt = TimingTree()

    with tt.time("total"):
        with tt.time("phantom"):
            phantom = _build_phantom(phantom_config)
        (case_dir / "phantom_config.json").write_text(json.dumps(phantom.metadata, indent=2))
        if simulation is not None:
            (case_dir / "simulation_config.json").write_text(simulation.model_dump_json(indent=2))

        with tt.time("simulate"):
            tf_settings = recon_config.get(modality, {}).get("transfer_function", {})
            sim_kwargs = _sim_kwargs(modality, tf_settings)
            forward_model = simulation.forward_model if simulation is not None else "shift_invariant"
            if modality == "phase":
                if forward_model != "shift_invariant":
                    raise ValueError(f"forward_model={forward_model!r} is not supported for phase modality")
                data = simulate_phase_3d(phantom, **sim_kwargs)
            elif modality == "fluorescence":
                if simulation is not None and forward_model == "shift_variant":
                    data = simulate_shift_variant_fluorescence_3d(
                        phantom,
                        spatial_pupil_coefficients=simulation.spatial_pupil_coefficients,
                        n_tiles_yx=simulation.n_tiles_yx,
                        background=0,
                        **sim_kwargs,
                    )
                else:
                    data = simulate_fluorescence_3d(phantom, background=0, **sim_kwargs)
            else:
                raise ValueError(f"Unknown modality: {modality}")
            channel_name = recon_config.get("input_channel_names", [_DEFAULT_CHANNEL[modality]])[0]

        simulated_path = case_dir / "simulated.zarr"
        _write_zarr(data, simulated_path, channel_name=channel_name, pixel_sizes=phantom.pixel_sizes)
        ground_truth = phantom.phase if modality == "phase" else phantom.fluorescence
        _write_zarr(
            ground_truth,
            case_dir / "phantom.zarr",
            channel_name=f"{channel_name}_phantom",
            pixel_sizes=phantom.pixel_sizes,
        )

        if recovery is not None and recovery.enabled:
            if modality != "fluorescence":
                raise ValueError("recovery: is only supported for fluorescence modality")
            metrics = _run_recovery_and_measure(
                data=data,
                phantom=phantom,
                recon_config=recon_config,
                simulation=simulation,
                recovery=recovery,
                case_dir=case_dir,
                tt=tt,
                ground_truth=ground_truth,
            )
        else:
            metrics = _run_and_measure(
                position_path=simulated_path / "0" / "0" / "0",
                recon_config=recon_config,
                case_dir=case_dir,
                tt=tt,
                ground_truth=ground_truth,
            )
        ref_check = check_reference(case_dir, metrics, reference)
        if ref_check is not None:
            metrics["reference_check"] = ref_check

    _finalize_case(case_dir, tt=tt, metrics=metrics, save_all=save_all)
    return metrics


def recovery_score(recovered: torch.Tensor, truth: torch.Tensor, noll_indices: list[int]) -> dict:
    """Return wo-4.0's figures of merit of a recovered ``(T, M)`` coefficient table against the truth.

    ``recovery_score = 1 - RMS(recovered - truth) / RMS(truth)`` over all tiles and modes (1 is perfect, 0 no better
    than zeros), with the per-mode RMS error and Pearson correlation (None where either side is constant). Copied
    from wo-4.0 ``evaluate_recovery_against_truth``.
    """
    rec, tru = recovered.numpy(), truth.numpy()
    norm = float(np.sqrt(np.mean(tru**2)))
    score = float("nan") if norm == 0 else 1.0 - float(np.sqrt(np.mean((rec - tru) ** 2))) / norm
    rmse, correlation = {}, {}
    for m, j in enumerate(noll_indices):
        rmse[f"Z{j}"] = float(np.sqrt(np.mean((rec[:, m] - tru[:, m]) ** 2)))
        constant = tru[:, m].std() == 0 or rec[:, m].std() == 0
        correlation[f"Z{j}"] = None if constant else float(np.corrcoef(rec[:, m], tru[:, m])[0, 1])
    return {"recovery_score": score, "per_mode_rmse": rmse, "per_mode_correlation": correlation}


def _recovery_score_function(recovery: RecoveryConfig, *, cutoff: float, tile_sampling: Sampling) -> Score:
    """Return the per-tile loss of ``recovery`` as a waveorder 4 score.

    ``mse`` is :class:`~waveorder.nn.systems.tiled.BeadFit` (3.x's bead fit) and ``midband`` the 3.x recovery
    variant :class:`~waveorder.nn.scores.CentreSliceMidbandScore`; the other sharpness losses are the 3.x optimizer
    losses of :mod:`waveorder.nn.scores`.
    """
    if recovery.loss == "mse":
        bead = BeadFit.gaussian(tile_sampling, sigma=recovery.bead_template_sigma_um, like=torch.zeros(()))
        return BeadFit(bead.obj, scale_fit=recovery.scale_fit)
    if recovery.loss == "midband":
        return CentreSliceMidbandScore(solver=Wiener(), cutoff=cutoff, fractions=recovery.midband_fractions)
    if recovery.loss == "spectral_flatness":
        return SpectralFlatnessScore(solver=Wiener(), cutoff=cutoff, fractions=recovery.midband_fractions)
    sharpness = {
        "tv": TotalVariationScore,
        "laplacian_var": LaplacianVarianceScore,
        "normalized_var": NormalizedVarianceScore,
    }
    return sharpness[recovery.loss](solver=Wiener())


def _run_recovery_and_measure(  # noqa: PLR0913 (the simulated case and its settings)
    data: torch.Tensor,
    *,
    phantom: Phantom,
    recon_config: dict,
    simulation: SimulationConfig | None,
    recovery: RecoveryConfig,
    case_dir: Path,
    tt: TimingTree,
    ground_truth: torch.Tensor,
) -> dict:
    """Recover per-tile Zernike coefficients with ``waveorder.tiling.run.recover_zernikes`` in place of ``wo rec``.

    3.x tiles of size ``T`` sharing ``O`` pixels with their neighbours become tile cores ``T - O`` wide widened by
    ``O // 2`` per side; the tile centres are those of the tiles' read regions. Writes ``recovered_coefs.npy``,
    ``truth_coefs.npy``, ``tile_centers_um.npy``, ``zernike_recovery.json`` and the blended deconvolution as
    ``reconstruction.zarr``.
    """
    z_pix, yx_pix, _ = phantom.pixel_sizes
    fl = recon_config.get("fluorescence", {}).get("transfer_function", {})
    na_det = fl.get("numerical_aperture_detection", 1.2)
    wavelength = fl.get("wavelength_emission", 0.532)
    (case_dir / "recovery_config.json").write_text(recovery.model_dump_json(indent=2))

    terms = {f"z{j}": Free(0.0, bounds=(-1.0, 1.0)) for j in recovery.noll_indices}
    system = fluorescence_system(
        wavelength_emission=wavelength,
        index_of_refraction_media=fl.get("index_of_refraction_media", 1.3),
        numerical_aperture_detection=na_det,
        aberrations=terms,
    )
    model = ImagingModel(system, Tikhonov(recovery.wiener_regularization), Gaussian(1.0))
    names = zernike_terms(model)
    Z, Y, X = data.shape
    ty, tx = recovery.tile_size_yx["y"], recovery.tile_size_yx["x"]
    oy, ox = recovery.tile_overlap_yx.get("y", 0), recovery.tile_overlap_yx.get("x", 0)
    plan = TilePlan(shape=(Z, Y, X), tile=(Z, ty - oy, tx - ox), overlap=(0, oy // 2, ox // 2))
    pixel_size = (float(z_pix), float(yx_pix), float(yx_pix))
    score = _recovery_score_function(
        recovery, cutoff=2 * na_det / wavelength, tile_sampling=Sampling(shape=plan.read_shape, pixel_size=pixel_size)
    )
    # 3.x stepped the mean of the per-tile losses and kept the last iterate.
    search = GradientSearch(
        iterations=recovery.n_iter,
        learning_rates={name: LEARNING_RATES.get(j, 0.01) for name, j in names.items()},
        optimizer=recovery.optimizer,  # ty: ignore[invalid-argument-type] (3.x names; the search refuses others)
        loss_scale=1 / len(plan.regions),
        last=True,
    )

    with tt.time("reconstruct"):
        result = recover_zernikes(data, model, plan, pixel_size=pixel_size, search=search, score=score)

    coefs = torch.stack([result.calibration.parameters[name].detach().cpu().float() for name in names], dim=1)
    centers_um = [
        (((y0 + y1) / 2 - Y / 2) * yx_pix, ((x0 + x1) / 2 - X / 2) * yx_pix)
        for _, (y0, y1), (x0, x1) in (region.read for region in plan.regions)
    ]
    reconstruction = result.reconstruction[0].detach().cpu().float()
    _write_zarr(
        reconstruction,
        case_dir / RECONSTRUCTION_FILENAME,
        channel_name=recon_config["input_channel_names"][0],
        pixel_sizes=pixel_size,
    )
    np.save(case_dir / "recovered_coefs.npy", coefs.numpy())
    np.save(case_dir / "tile_centers_um.npy", np.array(centers_um))

    truth_metrics: dict | None = None
    if simulation is not None and simulation.spatial_pupil_coefficients:
        spatial = normalize_pupil_coefficients(simulation.spatial_pupil_coefficients)
        half_y, half_x = Y * yx_pix / 2, X * yx_pix / 2
        truth = torch.tensor(
            [
                [
                    field_coefficients(spatial, y_norm=yc / half_y, x_norm=xc / half_x).get(j, 0.0)
                    for j in names.values()
                ]
                for yc, xc in centers_um
            ],
            dtype=torch.float32,
        )
        np.save(case_dir / "truth_coefs.npy", truth.numpy())
        truth_metrics = recovery_score(coefs, truth, list(names.values()))

    (case_dir / "zernike_recovery.json").write_text(
        json.dumps(
            {
                "noll_indices": list(names.values()),
                "tile_centers_um": centers_um,
                "objective": result.calibration.objective.detach().cpu().tolist(),
                "truth_metrics": truth_metrics,
            },
            indent=2,
        )
    )
    with tt.time("metrics"):
        metrics = compute_metrics(
            reconstruction, NA_det=na_det, wavelength=wavelength, pixel_size=yx_pix, phantom=ground_truth
        )
    if truth_metrics is not None:
        metrics["zernike_recovery"] = truth_metrics
    (case_dir / "config.yml").write_text(yaml.dump(recon_config, default_flow_style=False))
    return metrics


def run_hpc_case(  # noqa: PLR0913 (one argument per case field)
    *,
    input_path: str,
    position: str,
    recon_config: dict,
    case_dir: Path,
    save_all: bool = False,
    reference: dict[str, ReferenceBound] | None = None,
    crop: CropConfig | None = None,
    tile: tuple[int, int, int] | None = None,
    tile_overlap: tuple[int, int, int] | None = None,
) -> dict:
    """Run a case on recorded data with ``wo rec``.

    Parameters
    ----------
    input_path : str
        Input OME-Zarr plate.
    position : str
        Position within it, e.g. ``"A/1/029029"``.
    recon_config : dict
        3.x reconstruction configuration.
    case_dir : Path
        Output directory of the case.
    save_all : bool
        Keep every output; see :func:`run_synthetic_case`.
    reference : dict, optional
        Bounds on metrics and calibrated parameters.
    crop : CropConfig, optional
        Crop the input to this box first.
    tile, tile_overlap : tuple[int, int, int], optional
        ``wo rec --tile`` and ``--tile-overlap``.

    Returns
    -------
    dict
        The case's metrics.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    tt = TimingTree()

    with tt.time("total"):
        if crop is not None:
            position_path = crop_input_position(input_path, position, crop=crop, case_dir=case_dir) / position
        else:
            position_path = Path(input_path) / position

        metrics = _run_and_measure(
            position_path=position_path,
            recon_config=recon_config,
            case_dir=case_dir,
            tt=tt,
            tile=tile,
            tile_overlap=tile_overlap,
        )
        ref_check = check_reference(case_dir, metrics, reference)
        if ref_check is not None:
            metrics["reference_check"] = ref_check

    _finalize_case(case_dir, tt=tt, metrics=metrics, save_all=save_all)
    return metrics
