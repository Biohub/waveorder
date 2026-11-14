"""The run of ``wo simulate``, which the command imports when it runs: it imports the models and torch."""

import dataclasses
from pathlib import Path

import torch
import typer
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta

from waveorder.cli._configured import ResolvedConfig, build_model, located, resolve_config
from waveorder.cli.options import expect_store_output
from waveorder.cli.render import status
from waveorder.config import V4Config
from waveorder.config.schema import FreeParameter
from waveorder.errors import Rejected
from waveorder.nn import free_specs, load_parameters
from waveorder.zernike_names import noll_indices_up_to_order, noll_to_name


def _truth(config: V4Config, generator: torch.Generator) -> dict[str, float]:
    """Return the true Zernike coefficients by term name: the fixed values, and a draw per numeric tolerance."""
    aberrations = config.imaging_system.detection.pupil_aberrations
    truth: dict[str, float] = {}
    if aberrations is None:
        return truth

    def draw(nominal: float, tolerance: float | str, bounds: tuple[float, float] | None) -> float:
        if isinstance(tolerance, str):  # free: no prior to draw from
            return float(nominal)
        value = nominal + tolerance * float(torch.randn((), generator=generator))
        return value if bounds is None else min(max(value, bounds[0]), bounds[1])

    shorthand = aberrations.zernike_up_to_order
    if shorthand is not None:
        for j in noll_indices_up_to_order(shorthand.order):
            truth[noll_to_name(j)] = draw(0.0, shorthand.tolerance, shorthand.bounds)
    for name, value in aberrations.terms.items():
        free = isinstance(value, FreeParameter)
        truth[name] = draw(value.nominal, value.tolerance, value.bounds) if free else float(value)
    return truth


def _write(path: Path, volume: torch.Tensor, *, channels: list[str], scale: tuple[float, float, float]) -> None:
    """Write a ``(C, Z, Y, X)`` volume as the one time point of position ``0/0/0`` of a new HCS plate."""
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=channels) as plate:
        position = plate.create_position("0", "0", "0")
        transform = [TransformationMeta(type="scale", scale=[1.0, 1.0, *scale])]
        position.create_image("0", volume[None].float().cpu().numpy(), transform=transform)


def _checked(resolved: ResolvedConfig, output: Path, *, overwrite: bool) -> tuple[int, tuple[float, float, float]]:
    """Return the seed and the voxel size; refuse a configuration that is no simulation and outputs that exist."""
    config = resolved.config
    where = str(resolved.path)
    if config.simulate is None:
        raise Rejected.one("has no simulate block; add simulate: {seed: 0, shape: [Z, Y, X]}", location=where)
    if resolved.pixel_size is None:
        raise Rejected.one("data.pixel_size is auto; a simulation needs the voxel size as numbers", location=where)
    for name in ("data.zarr", "object.zarr"):
        if (output / name).exists() and not overwrite:
            raise Rejected.one("exists; pass --overwrite to replace it", location=f"-o {output / name}")
    expect_store_output(output / "tf.zarr", overwrite=overwrite, location=f"-o {output / 'tf.zarr'}")
    return config.simulate.seed, resolved.pixel_size


def simulate(config_filepath: Path, output_dirpath: Path, *, overwrite: bool) -> None:
    """Draw the truth, simulate the data and write the three stores (``wo simulate``)."""
    resolved = resolve_config(config_filepath)
    seed, pixel_size = _checked(resolved, output_dirpath, overwrite=overwrite)
    model = build_model(resolved)
    generator = torch.Generator().manual_seed(seed)
    truth = _truth(resolved.config, generator)
    drawn: dict[str, float] = {}
    for name in free_specs(model.imaging_model):
        term = name.rpartition(".pupil_aberrations.")[2]
        if term not in truth or ".pupil_aberrations." not in name:
            raise Rejected.one(f"{name} is free; a simulation draws Zernike terms only, give it as a number")
        drawn[name] = truth[term]
    with located(resolved):
        load_parameters(model.imaging_model, drawn)
        data, obj = model.simulate(seed=generator)
        z, y, x = data.shape[-3:]
        store = model.to_tf_store(shapes=[(z, y, x)], pixel_size=pixel_size)
    provenance = {
        **store.provenance,
        "kind": "simulated",
        "seed": seed,
        "true_coefficients_waves": truth,
        "noise": {"kind": resolved.config.noise.kind, **resolved.config.noise.model_dump(exclude_none=True)},
    }
    output_dirpath.mkdir(parents=True, exist_ok=True)
    channels = resolved.channels[: data.shape[0]]
    _write(output_dirpath / "data.zarr", data, channels=channels, scale=pixel_size)
    names = ["object"] if obj.shape[0] == 1 else [f"object_{index}" for index in range(obj.shape[0])]
    _write(output_dirpath / "object.zarr", obj, channels=names, scale=pixel_size)
    dataclasses.replace(store, provenance=provenance).save(output_dirpath / "tf.zarr", overwrite=overwrite)
    status(f"wrote data.zarr, object.zarr, tf.zarr to {output_dirpath}")
    for name, value in truth.items():
        typer.echo(f"true {name:16s} {value:+.4f} waves")
