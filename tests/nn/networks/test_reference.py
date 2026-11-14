"""The built-in parts against the networks they must be able to express: timm's ResNets and the OPS U-Net.

The OPS U-Net is read from a copy pinned with these tests, ``ops_reference/``: the files ``model.py``,
``batchrenorm.py`` and ``dataset.py`` of ``ops_process/ops_analysis/benchmarks/phase_recon/unet`` in the OPS
monorepo at commit 3b6f5749875f8740266878cee370782ec4329cd7 (2026-11-13), stored as ``.txt`` so that pytest and the
linters leave them alone. Their SHA-256 are checked before they are imported. To follow a newer reference, copy the
files at its commit (``git show <commit>:<path> > ops_reference/<name>.py.txt``) and update ``PINNED``.
"""

import ast
import hashlib
import importlib.machinery
import importlib.util
import math
import sys
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from waveorder.nn import Sampling
from waveorder.nn.networks import (
    Bias,
    ConcatPlanes,
    DefocusKernel,
    Place,
    PlaneHead,
    PlaneOffsets,
    PlaneProjection,
    ResNetEncoder,
    StridedVolume,
    SubPixelHead,
    TimmEncoder,
    UNet,
)

PINNED = Path(__file__).parent / "ops_reference"
"""The pinned copy of the reference."""

REFERENCE_COMMIT = "3b6f5749875f8740266878cee370782ec4329cd7"
"""The OPS monorepo commit the pinned files are taken from."""

SHA256 = {
    "model.py.txt": "6d9be75ce0b84cdc0f90cc27e0852fdfe3c11dbec36ff8cc2856b9e600c15ffe",
    "batchrenorm.py.txt": "ad40cbc4b6895969251dc250fe0cd26b3c0643660d9162081e17900286413b6b",
    "dataset.py.txt": "d68141fdd5815334e5a48ac28f6caeede8f908f7e99cef3e1adf94d3b0f111d5",
}
"""The SHA-256 of each pinned file at ``REFERENCE_COMMIT``."""

CONDITIONS = {"well": 2, "field": 2}  # the reference's four floats, in its order: the first two, then the last two

RENAMES = (
    # CoordTimmUNet key prefix -> UNet key prefix, first match wins.
    ("zproj.0.kernel_mlp.", "stem.kernel."),  # the defocus stem's kernel network
    ("zproj.0.conv.", "stem.conv."),  # the strided stem's Conv3d
    ("zproj.0.", "stem.conv."),  # the projection stem's Conv2d
    ("zproj.1.", "stem.norm."),
    ("stem_cond.", "stem_condition."),
    ("encoder.", "encoder.model."),  # a TimmEncoder holds the timm network as `model`
    ("full_res.", "full_resolution."),
    ("head.1.", "head.refine."),  # the sub-pixel head's 3x3 block
    ("head.2.", "head.conv."),  # the sub-pixel head's 1x1 convolution
    ("head.", "head.conv."),  # the plane head, a bare Conv2d in the reference
)


def renamed(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return a CoordTimmUNet state dict under the keys of the UNet that expresses it."""
    out = {}
    for key, value in state.items():
        key = key.replace(".cond.weight", ".condition.weight") if key.startswith("decoder.") else key
        prefix = next((old for old, _ in RENAMES if key.startswith(old)), None)
        out[key if prefix is None else dict(RENAMES)[prefix] + key.removeprefix(prefix)] = value
    return out


def pinned(name: str) -> Path:
    """Return a pinned file after checking it is the one taken at ``REFERENCE_COMMIT``."""
    path = PINNED / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == SHA256[name], f"{path} is not the file of commit {REFERENCE_COMMIT}: it was changed"
    return path


@pytest.fixture(scope="module")
def reference() -> Iterator[types.ModuleType]:
    """The pinned ``model.py``, imported as a module of a stand-in package (it imports its sibling batchrenorm)."""
    package = types.ModuleType("ops_reference_unet")
    package.__path__ = []
    sys.modules["ops_reference_unet"] = package
    try:
        for name in ("batchrenorm", "model"):
            qualified = f"ops_reference_unet.{name}"
            loader = importlib.machinery.SourceFileLoader(qualified, str(pinned(f"{name}.py.txt")))
            spec = importlib.util.spec_from_loader(qualified, loader)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            loader.exec_module(module)
        yield sys.modules["ops_reference_unet.model"]
    finally:
        for name in [name for name in sys.modules if name.startswith("ops_reference_unet")]:
            del sys.modules[name]


@dataclass(frozen=True)
class ReferenceDefocus:
    """The reference's definition of the defocus: ``slice_defocus`` and ``Optics`` of the pinned ``dataset.py``."""

    slice_defocus: Callable[..., object]
    optics: type


@pytest.fixture(scope="module")
def reference_defocus() -> ReferenceDefocus:
    """``slice_defocus`` and ``Optics``, compiled from the pinned ``dataset.py`` alone (its imports are not here)."""
    import numpy as np  # noqa: TID251 (the reference computes the defocus in numpy)

    tree = ast.parse(pinned("dataset.py.txt").read_text())
    wanted = [node for node in tree.body if getattr(node, "name", None) in ("slice_defocus", "Optics")]
    assert len(wanted) == 2
    namespace: dict[str, object] = {"np": np, "dataclass": dataclass, "Path": Path}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "dataset.py.txt", "exec"), namespace)
    slice_defocus, optics = namespace["slice_defocus"], namespace["Optics"]
    assert callable(slice_defocus)
    assert isinstance(optics, type)
    return ReferenceDefocus(slice_defocus=slice_defocus, optics=optics)


def randomized(network: nn.Module, *, seed: int) -> nn.Module:
    """Give every parameter and buffer a random value at a scale that keeps the signal through the depth.

    Default initialization hides errors (residual branches start at zero, running statistics at 0 and 1), and small
    random weights make the output a constant that no longer depends on the data.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in (*network.named_parameters(), *network.named_buffers()):
            if not tensor.is_floating_point():
                continue
            noise = torch.randn(tensor.shape, generator=generator)
            if tensor.dim() >= 2:  # convolutions and linear layers, He-scaled
                tensor.copy_(noise * (2 / tensor[0].numel()) ** 0.5)
            elif name.endswith(("running_var", "weight")):  # variances and normalization scales, positive
                tensor.copy_(0.75 + 0.5 * torch.rand(tensor.shape, generator=generator))
            else:  # biases and running means
                tensor.copy_(0.1 * noise)
    return network.eval()


def inputs(planes: int, *, seed: int = 0) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(2, planes, 64, 64, generator=generator), torch.rand(2, 4, generator=generator)


def named(coords: Tensor) -> dict[str, Tensor]:
    return {"well": coords[:, :2], "field": coords[:, 2:]}


def assert_same_function(theirs: nn.Module, ours: UNet, planes: int, **extra: Tensor) -> None:
    ours.load_state_dict(renamed(randomized(theirs, seed=1).state_dict()))
    ours.eval()
    stack, coords = inputs(planes)
    condition = named(coords)
    if extra:
        condition["plane_offsets"] = extra["defocus"]
    with torch.no_grad():
        expected = theirs(stack, coords, **extra)
        estimate = ours(stack[:, None], condition)
        # The comparison means something only if the reference responds to its data and to each pair of conditions.
        for changed in (theirs(stack.flip(-1), coords, **extra), theirs(stack, coords.flip(1), **extra)):
            assert (changed - expected).abs().max() > 1e-2
    assert estimate.shape == (2, 1, 1, 64, 64)
    torch.testing.assert_close(estimate[:, :, 0], expected, rtol=0, atol=1e-5)
    swapped = {"well": condition["field"], "field": condition["well"]}
    swapped.update({name: value for name, value in condition.items() if name not in CONDITIONS})
    with torch.no_grad():  # the declared order of the conditions is part of the network
        assert (ours(stack[:, None], swapped)[:, :, 0] - expected).abs().max() > 1e-2


def test_projection_stem_with_concatenated_planes_is_the_reference(reference):
    theirs = reference.CoordTimmUNet(
        encoder="resnet34", in_z=7, n_coords=4, proj_ch=3, stem="zproj", cond="concat", pretrained=False
    )
    ours = UNet(
        PlaneProjection(planes=7, channels=3),
        TimmEncoder("resnet34"),
        PlaneHead(),
        decoder_channels=(256, 128, 64, 32, 16),
        conditioning=ConcatPlanes(sizes=CONDITIONS),
        dropout=0.1,
    )
    assert_same_function(theirs, ours, planes=7)


def test_strided_stem_with_offsets_is_the_reference(reference):
    theirs = reference.CoordTimmUNet(
        encoder="resnet34", in_z=7, n_coords=4, stem="cytoland", cond="bias", pretrained=False
    )
    ours = UNet(
        StridedVolume(planes=7),
        TimmEncoder("resnet34"),
        SubPixelHead(),
        decoder_channels=(256, 128, 64),
        conditioning=Bias(sizes=CONDITIONS),
        dropout=0.1,
    )
    assert_same_function(theirs, ours, planes=7)


@pytest.mark.parametrize(
    ("stem", "cond", "norm", "full_res_ch"),
    [
        ("zproj", "bias", "bn", 0),
        ("zproj", "concat", "bn", 8),  # the full-resolution skip
        ("zproj", "concat", "gn", 0),  # group norm, encoder batch norms frozen
        ("cytoland", "concat", "bn", 0),  # planes repeated over z into the Conv3d
        ("cytoland", "bias", "gn", 0),
    ],
)
def test_the_other_layouts_are_the_reference(reference, stem, cond, norm, full_res_ch):
    quarter = stem == "cytoland"
    theirs = reference.CoordTimmUNet(
        encoder="resnet18", in_z=6, n_coords=4, stem=stem, cond=cond, norm=norm, full_res_ch=full_res_ch,
        stem_kz=3 if quarter else 0, pretrained=False,
    )  # fmt: skip
    ours = UNet(
        StridedVolume(planes=6, kernel_planes=3) if quarter else PlaneProjection(planes=6, channels=3),
        TimmEncoder("resnet18"),
        SubPixelHead() if quarter else PlaneHead(),
        decoder_channels=(256, 128, 64) if quarter else (256, 128, 64, 32, 16),
        conditioning=(ConcatPlanes if cond == "concat" else Bias)(sizes=CONDITIONS),
        norm="batch" if norm == "bn" else "group",
        full_resolution_channels=full_res_ch,
    )
    assert_same_function(theirs, ours, planes=6)


@pytest.mark.parametrize("cond", ["concat", "bias"])
def test_defocus_kernel_stem_is_the_reference(reference, cond):
    theirs = reference.CoordTimmUNet(
        encoder="resnet18", n_coords=4, stem="optical_defocus", cond=cond, pretrained=False
    )
    ours = UNet(
        DefocusKernel(planes=5),
        TimmEncoder("resnet18"),
        SubPixelHead(),
        decoder_channels=(256, 128, 64),
        conditioning=(ConcatPlanes if cond == "concat" else Bias)(sizes=CONDITIONS),
    )
    nan = float("nan")
    # Two stacks in one batch: five planes, and three planes padded with NaN defocus; one plane of the first lies
    # beyond z_range (8 units) and gets no kernel.
    defocus = torch.tensor([[-9.0, -2.5, 0.0, 2.5, 5.0], [-1.3, 0.0, 1.3, nan, nan]])
    assert_same_function(theirs, ours, planes=5, defocus=defocus)


@pytest.mark.parametrize(("planes", "focus_offset"), [(5, 0.0), (6, 1.5), (1, -0.5)])
def test_plane_offsets_are_the_references_defocus(reference_defocus, planes, focus_offset):
    import numpy as np  # noqa: TID251 (the reference computes the defocus in numpy)

    spacing, wavelength, aperture, index = 2.0, 0.45, 0.55, 1.33
    unit = reference_defocus.optics(wavelength, aperture, index).axial_unit_um
    condition = PlaneOffsets(
        8, wavelength=wavelength, numerical_aperture=aperture, index_of_refraction=index, focus_offset=focus_offset
    )
    assert condition.unit == pytest.approx(unit, rel=1e-12)
    place = Place.of(Sampling(shape=(planes, 16, 16), pixel_size=(spacing, 0.5, 0.5)))
    values = condition.values(place, None)
    expected = reference_defocus.slice_defocus(np.arange(planes), planes, spacing, focus_offset, unit)
    torch.testing.assert_close(values[:planes], torch.from_numpy(np.asarray(expected)).reshape(-1), rtol=0, atol=1e-6)
    assert all(math.isnan(value) for value in values[planes:].tolist())


@pytest.mark.parametrize("name", ["resnet10t", "resnet18", "resnet34"])
@pytest.mark.parametrize("own_stem", [True, False])
def test_built_in_resnet_is_timms(name, own_stem):
    """Same keys, same function: a checkpoint trained with timm's ResNet runs on the built-in one, without timm."""
    in_channels = 3 if own_stem else None
    theirs = TimmEncoder(name).module(in_channels=in_channels, norm="batch")
    ours = ResNetEncoder(name).module(in_channels=in_channels, norm="batch")
    state = {key.removeprefix("model."): value for key, value in randomized(theirs, seed=2).state_dict().items()}
    ours.load_state_dict(state)
    ours.eval()
    assert ours.in_channels == theirs.in_channels
    assert (ours.channels, ours.reductions) == (theirs.channels, theirs.reductions)
    features = torch.randn(2, ours.in_channels, 64, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        for mine, expected in zip(ours(features), theirs(features), strict=True):
            torch.testing.assert_close(mine, expected, rtol=0, atol=1e-5)


def test_frozen_built_in_resnet_loads_a_batch_norm_checkpoint():
    """With norm="group" the encoder's batch norms are frozen; a checkpoint with step counters still loads."""
    trained = randomized(ResNetEncoder("resnet10t").module(in_channels=3, norm="batch"), seed=3)
    frozen = ResNetEncoder("resnet10t").module(in_channels=3, norm="group")
    frozen.load_state_dict(trained.state_dict())
    features = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    frozen.train()  # training mode changes nothing in a frozen encoder
    with torch.no_grad():
        for mine, expected in zip(frozen(features), trained(features), strict=True):
            # The frozen formula rounds otherwise than F.batch_norm: float32 through the depth differs by about 1e-6
            # of a map's scale, so the tolerance is relative to that scale, not to each (possibly near-zero) value.
            torch.testing.assert_close(mine, expected, rtol=0, atol=1e-5 * float(expected.abs().max()))
    # Nothing in a frozen encoder trains but its convolutions: every parameter is a convolution kernel.
    assert all(parameter.dim() == 4 for parameter in frozen.parameters())
