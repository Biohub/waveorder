"""Write the data of the two-node test: a raw plate, its reconstruction, the model's configuration and the index.

python make_data.py DATA

Sixteen positions in four wells (A/1, A/2, B/1, B/2) of 5 x 64 x 64 brightfield stacks, simulated with a thin phase
model and reconstructed with it (as ``wo rec`` would), indexed with a split by position of .5 / .25 / .25 (8 train,
4 val, 4 test fields). ``DATA`` must be on a filesystem every node reads.
"""

import sys
from pathlib import Path

import iohub
import torch

from waveorder.data import GroupSplit, index_from_plates
from waveorder.models import ReconstructionModel

MODEL = """\
version: 4
data:
  input_channel_names: [BF]
  pixel_size: {z: 0.4, y: 0.15, x: 0.15}
imaging_system:
  contrast: phase
  index_of_refraction_media: 1.33
  detection: {numerical_aperture: 0.9}
  illumination: {wavelength: 0.532, numerical_aperture: 0.52}
object: {dimension: 2, weight: 1e-3}
noise: {gaussian: {sigma: 1e-3}}
solver:
  reconstruction: closed_form
"""
SHAPE = (5, 64, 64)
PIXEL_SIZE = (0.4, 0.15, 0.15)
POSITIONS = [f"{row}/{column}/{fov}" for row in "AB" for column in (1, 2) for fov in range(4)]


def write_plate(path: Path, volumes: dict[str, torch.Tensor], channel: str) -> Path:
    """Write an HCS plate of one position per entry of ``volumes`` (``(T, C, Z, Y, X)``), one chunk per plane stack."""
    with iohub.open_ome_zarr(path, layout="hcs", mode="w-", channel_names=[channel]) as plate:
        for name, volume in volumes.items():
            position = plate.create_position(*name.split("/"))
            position.create_image("0", volume.numpy(), chunks=(1, 1, *volume.shape[2:]))
            for axis, size in zip("zyx", PIXEL_SIZE, strict=True):
                position.set_scale("0", axis, size)
    return path


def main(out: Path) -> None:
    """Write the plates, the model's configuration and the index into ``out``, a new folder."""
    out.mkdir(parents=True, exist_ok=False)
    (out / "model.yml").write_text(MODEL)
    model = ReconstructionModel.from_config(out / "model.yml")
    raw, reconstructed = {}, {}
    for number, name in enumerate(POSITIONS):
        data, _ = model.simulate(shape=SHAPE, seed=number)  # contrast (1, Z, Y, X)
        intensity = (1000 * (1 + 0.2 * data / data.std())).clamp(0, 65535)
        raw[name] = intensity.round().to(torch.int32).to(torch.uint16)[None]
        estimate = model.reconstruct(intensity, pixel_size=PIXEL_SIZE).estimate
        reconstructed[name] = estimate[:1].to(torch.float32)[None]
    inputs = write_plate(out / "bf.zarr", raw, "BF")
    targets = write_plate(out / "phase.zarr", reconstructed, "Phase2D")
    split = GroupSplit(by="position", fractions={"train": 0.5, "val": 0.25, "test": 0.25})
    index = index_from_plates([inputs], channels=["BF"], targets=[targets], target_channels=["Phase2D"], split=split)
    index.save(out / "index")
    print(f"{out}: {len(index)} fields, {index.counts}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
