<p align="center">
  <img src="https://github.com/user-attachments/assets/00367e51-0e0a-445e-8d12-2a273f9014aa" alt="Image" width="50%">
</p>

[![Python package index](https://img.shields.io/pypi/v/waveorder.svg)](https://pypi.org/project/waveorder)
[![PyPI monthly downloads](https://img.shields.io/pypi/dm/waveorder.svg)](https://pypistats.org/packages/waveorder)
[![Total downloads](https://pepy.tech/badge/waveorder)](https://pepy.tech/project/waveorder)
[![GitHub contributors](https://img.shields.io/github/contributors-anon/mehta-lab/waveorder)](https://github.com/mehta-lab/waveorder/graphs/contributors)
![GitHub Repo stars](https://img.shields.io/github/stars/mehta-lab/waveorder)
![GitHub forks](https://img.shields.io/github/forks/mehta-lab/waveorder)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/waveorder)

`waveorder` simulates microscopes and reconstructs what they image: the phase of a sample from a brightfield defocus
stack, and the density of a fluorophore from a fluorescence stack, in 2D and 3D. A model holds the optics, the object
prior, the noise model, the solver and the voxel size of the data. It reconstructs arrays in memory, volumes too
large for one solve, and every position of an OME-Zarr plate across GPUs. The apertures, the illumination tilt, the
focus offset, the pupil aberrations and the pinhole can be left free and estimated from the data; the wavelengths and
the refractive index are fixed. Models are built from PyTorch modules, so they run on CPU or GPU and differentiate
end to end. The framework is described in this [preprint](https://arxiv.org/abs/2412.09775).

## Install

```sh
pip install waveorder                   # Python 3.12 to 3.14, torch 2.15 or newer; from a source checkout: uv sync
pip install "waveorder[export,wandb]"   # adds ONNX export of trained networks and --wandb logging
```

## Quick start in Python

Simulate a brightfield stack, estimate the illumination aperture from it, reconstruct the phase, save the model:

```python
import tempfile
from pathlib import Path

import torch

import waveorder as wo
from waveorder.nn import Detection, Free, Gaussian, Illumination, Tikhonov


def phase_model(illumination_na):
    return wo.models.Phase3D(
        Illumination(wavelength=0.532, numerical_aperture=illumination_na),
        Detection(wavelength=0.532, numerical_aperture=0.9),
        index_of_refraction_media=1.33,
        pixel_size=(0.4, 0.15),  # um: (z, yx); also one number, or (z, y, x)
        object_prior=Tikhonov(weight=4e-4),
        noise=Gaussian(sigma=1e-3),
    )


# Data to work on: a random phase object imaged at illumination NA 0.52.
obj = 0.05 * torch.randn(1, 16, 64, 64, generator=torch.Generator().manual_seed(0))
data, _ = phase_model(0.52).simulate(obj, seed=0)  # brightfield contrast, (C, Z, Y, X)

# A model that starts at NA 0.45 and estimates it within its bounds.
model = phase_model(Free(0.45, bounds=(0.3, 0.7)))
calibration = model.calibrate(data, prepared=True)  # the model keeps the estimate
result = model.reconstruct(data, prepared=True)

with tempfile.TemporaryDirectory() as folder:
    again = wo.models.Phase3D.load(model.save(Path(folder, "phase")))

NA = "system.illumination.numerical_aperture"
print(f"NA = {calibration.parameters[NA].item():.3f}, estimate = {tuple(result.estimate.shape)}, {result.status}")
print(f"loaded NA = {again.system.illumination.numerical_aperture.item():.3f}")
```

`simulate` returns what the system measures, here the contrast `I / mean(I) - 1`, so `prepared=True` skips that
step; recorded intensities go in without it. The search maximizes the evidence by default, and `Free` takes a
`tolerance` (a Gaussian prior) and `bounds`. Give the noise `sigma` and the prior `weight` in the units of the data,
or free them too (`Gaussian(sigma="estimate")`, `Tikhonov(Free(1e-3, bounds=(1e-6, 1e2)))`).

## Quick start on the command line

The same optics as a configuration file, `phase.yml`:

```yaml
version: 4
data:
  input_channel_names: [BF]
  pixel_size: auto    # read from the OME-Zarr store; or a number, or {z: 0.25, y: 0.1, x: 0.1}
imaging_system:
  contrast: phase
  index_of_refraction_media: 1.33
  detection: {numerical_aperture: 0.9}
  illumination: {wavelength: 0.532, numerical_aperture: {nominal: 0.45, tolerance: 0.05, bounds: [0.3, 0.7]}}
object: {prior: tikhonov, weight: 4.0e-4}
noise: {gaussian: {sigma: 1.0e-3}}
```

```sh
wo config new --contrast phase --dimension 3 -o phase.yml   # a commented template to start from
wo config check phase.yml                                   # validate it and build the model, without data
wo rec -i 'input.zarr/*/*/*' -c phase.yml -o output.zarr    # calibrate on the first position, reconstruct all

# Or calibrate once, then reconstruct anywhere with the stored values.
wo calibrate-tf -i input.zarr/A/1/0 -c phase.yml -o tf.zarr
wo apply-inv-tf -i 'input.zarr/*/*/*' -c phase.yml -t tf.zarr -o output.zarr --device cuda:0,cuda:1
wo config show phase.yml -t tf.zarr                         # the configuration with its calibrated: block
```

With nothing free in the configuration, `wo compute-tf -i input.zarr/A/1/0 -c phase.yml -o tf.zarr` stores the
transfer functions of the input grids, and `wo apply-inv-tf -t tf.zarr` then reads them instead of computing them
wherever the grid, voxel size, imaging system, parameter values, dtype and device match.

`wo.models.ReconstructionModel.from_config("phase.yml")` builds the same `Phase3D` in Python. A configuration is
read in one direction, into a model: calibrated values go to a separate `calibrated:` block and the keys you wrote
stay as they are. `wo --help` lists the commands, and each command takes `--help`.

## What it can do

- **Contrasts.** `Phase3D` and `Phase2D` (thick and thin phase from brightfield), `Fluorescence3D` (widefield, or
  confocal with `pinhole_diameter=`) and `Fluorescence2D`, in `waveorder.models`.
- **Priors, noise models and solvers.** `Tikhonov`, `TotalVariation`, `NoPrior` and `Beads`;
  `Gaussian`, `Poisson` and `PoissonGaussian`; the closed-form `Wiener`, `FISTA`, `ADMM`, `RichardsonLucy` and
  `Distilled` (a trained network). Each is a constructor argument, or assigned later (`model.noise = Gaussian(0.1)`).
  A model without a solver picks the one that fits its parts. A combination that cannot run is refused when the
  model is built or the part assigned. A `Beads` prior makes a calibration-only model: it calibrates, and its
  reconstruction is refused.
- **Calibration.** An optical number given as `Free` is estimated: apertures, illumination tilt, focus offset, Zernike
  pupil aberrations (`ZernikeAberration({"spherical": Free(0.02)})`), the pinhole, the noise and the prior weight
  (wavelengths and the refractive index are fixed numbers). Scores (`Evidence`, `GCV`, sharpness) and search stages
  (`GridSearch`, `RandomSearch`, `GradientSearch`, `LBFGSSearch`, `NelderMeadSearch`, and `WeightSweep` for the
  `otsu_cnr` and `l_curve` rules) are in `waveorder.calibrate`. With
  `field_tiles=FieldTiles(size_yx=(256, 256))` a model is calibrated and reconstructed per tile of the field of view.
- **Volumes larger than one solve.** `model.tile_plan(shape, tile=...)` plans overlapping tiles with the halo the
  solver needs, and `model.reconstruct_tiles([(source, sink)], plan=plan)` reads any sliceable array (zarr, memory
  map, tensor) tile by tile into the sink. On the command line: `wo rec --tile 16,512,512 --tile-overlap 0,32,32`.
- **Plates.** `model.calibrate_position(position, channels=[...])` returns a `TfStore`;
  `model.reconstruct_positions(paths, output=..., channels=[...], devices=...)` fills an output plate one
  (position, time) unit at a time, over GPUs and worker processes. `wo rec --resume` (`resume=True`) skips the units
  an interrupted run finished and recomputes those whose settings changed.
- **Simulation.** `model.simulate(obj)` returns `H f + n` and `f`, or draws a random object; `wo simulate` writes
  `data.zarr`, `object.zarr` and `tf.zarr` from a configuration with a `simulate:` block.
- **A fast network from a reconstruction.** A network (the student) trained on recorded reconstructions runs as the
  model's solver (`Distilled`) and exports as ONNX; see [Training a network](#training-a-network).

## Training a network

A distilled model is a network, the student, trained to give what a reconstruction model gives, then run as that
model's solver. Training reads reconstructions recorded beforehand with `wo rec`; a training step never runs the
physics model. The model is kept for the student's shape contract and for the record in the bundle.

```sh
wo rec -i 'raw.zarr/*/*/*' -c phase_2d.yml -o phase.zarr     # the targets
wo distill index raw.zarr --channel BF --targets phase.zarr --target-channel Phase2D \
    --named-split plates=plate:train=0.8,val=0.1,test=0.1 -o index
wo distill fit --config base.yml                             # RUN = runs/lightning_logs/version_0
wo distill fit --config base.yml --config cytoland.yml       # a variant: a later file overrides an earlier one
wo distill fit --config base.yml --model.student.init_args.head SubPixelHead ...  # or flags
wo distill fit --config RUN/config.yaml --ckpt_path RUN/checkpoints/last.ckpt    # resume
wo distill validate --config RUN/config.yaml --ckpt_path RUN/checkpoints/last.ckpt  # and wo distill test
wo distill evaluate RUN/best --index index --model phase_2d.yml   # whole fields of the test split
wo distill export RUN/best -o model.onnx
```

`base.yml` names waveorder's parts by their class names:

```yaml
trainer: {max_epochs: 1, accelerator: cpu}  # a quick check; without accelerator it trains on the GPUs it finds
model:
  reconstruction: phase_2d.yml  # the configuration wo rec made the targets with
  student:
    class_path: UNet
    init_args:
      stem: {class_path: PlaneProjection, init_args: {planes: 5, channels: 8}}
      encoder: {class_path: ResNetEncoder, init_args: {widths: [8, 8, 8, 8], depths: [1, 1, 1, 1]}}
      head: PlaneHead
      decoder_channels: [8, 8, 8, 8, 8]
  losses: {l1: L1}
data: {index: index, patch_size: [32, 32], per_field: 4, batch_size: 4, workers: 0}
```

and `cytoland.yml` holds only what the variant changes (`benchmarks/train/` has these and two more):

```yaml
model:
  student:
    init_args:
      stem: {class_path: StridedVolume, init_args: {planes: 5}}
      head: SubPixelHead
      decoder_channels: [8, 8, 8]
```

`wo distill fit --help` lists every key, and `--model.student.help UNet` the arguments of one class. A mistake is
refused in one line that names its key (`wo distill fit: error: data.batch_size: input should be greater than 0`).
A layer replaces a dictionary such as `losses` whole, while a flag (`--model.losses.ssim SSIM`) adds to it.

- **The run folder** holds `config.yaml`, the resolved configuration with full class paths and absolute paths
  (`wo distill fit --config RUN/config.yaml` runs it again from any directory); `run.json`, with the hashes of the
  index and the model, the split used, the software versions and the git commit; copies of the model's configuration
  and the index's manifest; the bundles `best/` and `last/`; and `checkpoints/last.ckpt`, written every
  `checkpoint_every` steps, at each epoch's end and on SIGTERM. A resumed run continues as the uninterrupted run
  would have, from mid-epoch too.
- **The index** is a directory: one parquet file per plate and `manifest.json` (the channels, the target channels,
  the voxel size, and each file's hash and row count). `wo distill index --append` adds a plate by writing its file
  only. Every row has a target. The column `split` holds the split of `--split-by` and `--fractions`, and each
  `--named-split NAME=...` adds a column `split_NAME`. A run picks one with `data.split: split_plates` and may hold
  plates or wells out with `data.holdout: {plates: [p1], wells: [p2/A/1]}`; `run.json` records what it used.
- **A part from outside waveorder** is named by its import path, as in
  `stem: {class_path: my_lab.stems.MeanThenConv, init_args: {planes: 5}}` for a subclass of
  `waveorder.nn.networks.Stem`. It needs no registration: the bundle records the class path, so the bundle loads
  wherever `my_lab` can be imported.
- **Many GPUs and nodes.** With `trainer.devices` and `trainer.num_nodes` each rank reads its own fields.
  `benchmarks/train/multinode/` holds a two-node Slurm test with a forced requeue.

In Python, `model.train(index, ...)` builds the same configuration, runs the same classes and returns a copy of the
model that runs the trained network as its solver. `module`, `data` and `trainer` take any other key of those blocks,
with parts given as objects (`module={"losses": {"l1": L1(), "ssim": SSIM(window=7)}}`):

```python
import waveorder as wo
from waveorder.nn import Detection, Illumination
from waveorder.nn.networks import ResNetEncoder, StridedVolume, SubPixelHead, UNet

model = wo.models.Phase2D(
    Illumination(wavelength=0.532, numerical_aperture=0.52),
    Detection(wavelength=0.532, numerical_aperture=0.9),
    index_of_refraction_media=1.33,
    pixel_size=(0.4, 0.15),
)
# A small U-Net that trains in seconds on a CPU; by default, a strided U-Net over the whole stack.
encoder = ResNetEncoder(widths=(8,) * 4, depths=(1,) * 4)
student = UNet(StridedVolume(5), encoder, SubPixelHead(), decoder_channels=(8, 8, 8))
result = model.train("index", student=student, out="run", patch=32, epochs=1, trainer={"accelerator": "cpu"})
onnx = result.model.export("model.onnx")  # result.model: a copy of the model that runs the network as its solver
print(f"solver = {type(result.model.solver).__name__}, bundle = {result.bundle.name}, onnx = {onnx.name}")
```

## Migrating from 3.x

Every command refuses a 3.x configuration and prints the command that converts it:

```sh
wo config migrate phase_3x.yml -o phase.yml   # wo config migrate --help lists what each 3.x key becomes
```

| 3.x | Now |
|---|---|
| `waveorder.models.phase_thick_3d` | `waveorder.models.Phase3D`; the system alone is `waveorder.nn.ThickPhase3D` |
| `waveorder.models.isotropic_thin_3d` | `waveorder.models.Phase2D`, `waveorder.nn.ThinPhase2D` |
| `waveorder.models.isotropic_fluorescent_thick_3d` | `waveorder.models.Fluorescence3D`, `waveorder.nn.WidefieldFluorescence3D` |
| `waveorder.models.isotropic_fluorescent_thin_3d` | `waveorder.models.Fluorescence2D`, `waveorder.nn.ThinFluorescence2D` |
| `calculate_transfer_function`, `apply_inverse_transfer_function` | `model.to_tf_store(shapes=[...])`, `model.reconstruct(data)` |
| `waveorder.focus.focus_from_transverse_band` | `waveorder.focus.focus_from_transverse_band` (same call, deprecated), `waveorder.nn.focus_from_transverse_band` |
| `waveorder.zernike` | `waveorder.nn.ZernikeAberration`, `waveorder.zernike_names` |
| `wo migrate-config`, `wo tile-stitch` | `wo config migrate`, `wo rec --tile` |

Not in this version: birefringence and permittivity tensor imaging, the napari plugin and GUI (`wo interactive`),
`wo view`, plugins for configuration options, uncertainty estimates of calibrated values, and custom CUDA kernels.
Use a 3.x release for those.

## For developers

A new contrast is a system and a model class. A shift-invariant system subclasses `ShiftInvariantSystem` and returns
its transfer function from `compute_transfer`. Registering it names it, so `save` and `load` can write and rebuild
it. [CONTRIBUTING](./CONTRIBUTING.md) covers the setup, the checks every change passes, the layers of the package,
the other kinds of part and the contract checks of `waveorder.testing` (`check_system` for a system).

```python
import tempfile
from pathlib import Path
from typing import Unpack

import torch

from waveorder import registry
from waveorder.models import Parts, ReconstructionModel
from waveorder.nn import Free, Quantity, Sampling, ShiftInvariantSystem
from waveorder.nn.functional import half_spectrum_frequencies


@registry.register("gaussian_blur_3d", kind="system")
class GaussianBlur3D(ShiftInvariantSystem):
    real_transfer = True  # an even point-spread function has a real transfer
    preserves_nonnegativity = True  # so Richardson-Lucy may run on it

    def __init__(self, *, width: Quantity) -> None:
        super().__init__(fields=("density",), channel_count=1)
        self.register_quantity("width", width, positive=True)  # um; a buffer when fixed, an nn.Parameter when Free

    def compute_transfer(self, sampling: Sampling) -> torch.Tensor:
        fz, fy, fx = half_spectrum_frequencies(sampling, device=self.width.device, dtype=self.width.dtype)
        return torch.exp(-2 * torch.pi**2 * self.width**2 * (fz**2 + fy**2 + fx**2))[None, None]  # (C, F, ...)


class Blur3D(ReconstructionModel):
    system_types = (GaussianBlur3D,)  # load and for_system return this class for the system

    def __init__(self, *, width: Quantity, **parts: Unpack[Parts]) -> None:
        super().__init__(GaussianBlur3D(width=width), **parts)


model = Blur3D(width=Free(0.3, bounds=(0.1, 0.6)), pixel_size=(0.25, 0.1))
data, _ = model.simulate(shape=(8, 32, 32), seed=0)
result = model.reconstruct(data)
with tempfile.TemporaryDirectory() as folder:
    again = ReconstructionModel.load(model.save(Path(folder, "blur")))
print(f"{type(again).__name__}, solver = {type(again.solver).__name__}, estimate = {tuple(result.estimate.shape)}")
```

## Citation

Please cite this repository, along with the relevant publications and preprints, if you use or adapt this code. The
citation information can be found by clicking "Cite this repository" button in the About section in the right
sidebar. The methods: phase from defocus in [2D](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-54-28-8566) and
[3D](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-57-1-a205), [widefield deconvolution](https://doi.org/10.1016/B978-0-12-407761-4.00017-8)
and [oblique plane light-sheet microscopy](https://doi.org/10.1093/pnasnexus/pgae323).

<details>
<summary> Chandler et al. 2024 </summary>
<pre><code>
@article{chandler_2024,
    author = {Chandler, Talon and Hirata-Miyasaki, Eduardo and Ivanov, Ivan E. and Liu, Ziwen and Sundarraman, Deepika and Ryan, Allyson Quinn and Jacobo, Adrian and Balla, Keir and Mehta, Shalin B.},
	title = {waveOrder: generalist framework for label-agnostic computational microscopy},
	journal = {arXiv},
	year = {2024},
	month = dec,
	eprint = {2412.09775},
	doi = {10.48550/arXiv.2412.09775}
}
</code></pre>
</details>
