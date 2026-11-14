"""A model wraps an imaging system with its solver: built in Python or from a config, it reconstructs and calibrates."""

import copy
import inspect
import re
from collections.abc import Callable
from typing import Any, override

import pytest
import torch
import yaml
from torch import Tensor

from tests.registration import temporarily
from waveorder.config import V4Config
from waveorder.errors import (
    CalibrationWarning,
    ConfigurationError,
    IncompatibleModelError,
    ParameterError,
    Rejected,
    ShapeError,
)
from waveorder.models import FieldTiles, Fluorescence2D, Fluorescence3D, Phase2D, Phase3D, ReconstructionModel
from waveorder.nn import (
    FISTA,
    GCV,
    ConfocalFluorescence3D,
    Detection,
    Free,
    Gaussian,
    GradientSearch,
    GridSearch,
    Illumination,
    MidbandPowerScore,
    NoPrior,
    Poisson,
    PoissonGaussian,
    Quantity,
    RichardsonLucy,
    Sampling,
    ShiftInvariantSystem,
    ThickPhase3D,
    ThinFluorescence2D,
    ThinPhase2D,
    Tikhonov,
    TotalVariation,
    WidefieldFluorescence3D,
    Wiener,
    ZPadding,
    brightfield_contrast,
    free_specs,
    load_parameters,
)
from waveorder.nn.scores import transverse_cutoff
from waveorder.tiling import TilePlan

PIXEL_SIZE = (0.25, 0.1, 0.1)
SHAPE = (8, 32, 32)
FORMS = "one (isotropic), (z, yx), (z, y, x) or {z, y, x}"
"""The voxel-size forms a refusal names."""
APERTURE = "system.detection.numerical_aperture"


def _illumination() -> Illumination:
    return Illumination(wavelength=0.532, numerical_aperture=0.5)


def _detection(wavelength: float = 0.532, numerical_aperture: Quantity = 1.2) -> Detection:
    return Detection(wavelength=wavelength, numerical_aperture=numerical_aperture)


def _phase_3d(**parts) -> Phase3D:
    return Phase3D(_illumination(), _detection(), index_of_refraction_media=1.33, **parts)


def _phase_2d(**parts) -> Phase2D:
    return Phase2D(_illumination(), _detection(), index_of_refraction_media=1.33, **parts)


def _fluorescence_3d(**parts) -> Fluorescence3D:
    return Fluorescence3D(_detection(0.52), index_of_refraction_media=1.33, **parts)


def _confocal_3d(**parts) -> Fluorescence3D:
    return Fluorescence3D(_detection(0.52), index_of_refraction_media=1.33, pinhole_diameter=0.3, **parts)


def _fluorescence_2d(**parts) -> Fluorescence2D:
    return Fluorescence2D(_detection(0.52), index_of_refraction_media=1.33, **parts)


def _phase_raw(dimension: int = 3) -> dict:
    return {
        "version": 4,
        "data": {"input_channel_names": ["BF"], "pixel_size": dict(zip("zyx", PIXEL_SIZE, strict=True))},
        "imaging_system": {
            "contrast": "phase",
            "index_of_refraction_media": 1.33,
            "detection": {"numerical_aperture": 1.2},
            "illumination": {"wavelength": 0.532, "numerical_aperture": 0.5},
        },
        "object": {"dimension": dimension},
    }


def _fluorescence_raw(dimension: int = 3, pinhole: dict | None = None) -> dict:
    detection: dict = {"wavelength": 0.52, "numerical_aperture": 1.2}
    if pinhole is not None:
        detection["pinhole"] = pinhole
    return {
        "version": 4,
        "data": {"input_channel_names": ["GFP"], "pixel_size": dict(zip("zyx", PIXEL_SIZE, strict=True))},
        "imaging_system": {"contrast": "fluorescence", "index_of_refraction_media": 1.33, "detection": detection},
        "object": {"dimension": dimension},
    }


def _free_aperture_raw() -> dict:
    """A fluorescence config whose detection aperture is calibrated from 1.05 within [1.0, 1.3]."""
    raw = _fluorescence_raw()
    raw["imaging_system"]["detection"]["numerical_aperture"] = {"nominal": 1.05, "tolerance": 0.2, "bounds": [1.0, 1.3]}
    raw["object"]["weight"] = 1e-2
    raw["noise"] = {"gaussian": {"sigma": 0.05}}
    raw["solver"] = {
        "calibration": {"score": "evidence", "search": [{"grid": {"points": 4}}, {"gradient": {"max_iter": 5}}]}
    }
    return raw


def _stack(seed: int = 0) -> Tensor:
    """Recorded brightfield intensities ``(Z, Y, X)``."""
    return 1e3 * (1 + 0.01 * torch.randn(SHAPE, generator=torch.Generator().manual_seed(seed)))


def _simulated(model: ReconstructionModel, *, aperture: float) -> Tensor:
    """Noisy data of a random object, simulated with ``model``'s parts at detection aperture ``aperture``."""
    truth = copy.deepcopy(model.imaging_model).double()
    load_parameters(truth, {APERTURE: torch.tensor(aperture, dtype=torch.float64)})
    sampling = Sampling(shape=SHAPE, pixel_size=PIXEL_SIZE)
    generator = torch.Generator().manual_seed(0)
    obj = 0.5 * torch.randn(truth.system.object_shape(sampling), generator=generator, dtype=torch.float64)
    with torch.no_grad():
        return truth.noise.sample(truth(obj, sampling), generator=generator)


def _same_values(first: ReconstructionModel, second: ReconstructionModel) -> bool:
    ours, theirs = first.imaging_model.state_dict(), second.imaging_model.state_dict()
    return ours.keys() == theirs.keys() and all(torch.equal(ours[name], theirs[name]) for name in ours)


CASES = [
    (_phase_3d, _phase_raw(), Phase3D, ThickPhase3D),
    (_phase_2d, _phase_raw(2), Phase2D, ThinPhase2D),
    (_fluorescence_3d, _fluorescence_raw(), Fluorescence3D, WidefieldFluorescence3D),
    (_confocal_3d, _fluorescence_raw(pinhole={"diameter_um": 0.3}), Fluorescence3D, ConfocalFluorescence3D),
    (_fluorescence_2d, _fluorescence_raw(2), Fluorescence2D, ThinFluorescence2D),
]


@pytest.mark.parametrize(("build", "raw", "model_class", "system"), CASES)
def test_a_model_built_from_optics_equals_the_one_its_minimal_config_builds(
    build: Callable[[], ReconstructionModel], raw: dict, model_class: type, system: type
):
    model = build()
    configured = ReconstructionModel.from_config(V4Config(**raw))
    assert type(model) is type(configured) is model_class
    assert type(model.system) is type(configured.system) is system
    for part in ("object_prior", "noise", "solver"):
        assert type(getattr(model, part)) is type(getattr(configured, part))
    assert isinstance(model.object_prior, Tikhonov)
    assert isinstance(model.noise, Gaussian)
    assert isinstance(model.solver, Wiener)
    assert _same_values(model, configured)
    assert model.config is None
    assert configured.config is not None
    assert model.z_padding == configured.z_padding == 0
    assert model.device == configured.device == torch.device("cpu")


def test_fluorescence_is_widefield_without_a_pinhole_and_confocal_with_one():
    assert type(_fluorescence_3d().system) is WidefieldFluorescence3D
    confocal = _confocal_3d().system
    assert isinstance(confocal, ConfocalFluorescence3D)
    assert float(confocal.pinhole_diameter) == pytest.approx(0.3)


def test_reconstruct_prepares_recorded_data_and_solves_as_the_modules_do():
    model = _phase_3d()
    stack = _stack()
    sampling = Sampling(shape=SHAPE, pixel_size=PIXEL_SIZE)
    contrast = brightfield_contrast(stack)[None]
    expected = Wiener()(contrast, model.imaging_model, sampling)
    result = model.reconstruct(stack, pixel_size=PIXEL_SIZE)
    assert result.estimate.shape == (1, *SHAPE)
    assert result.status == expected.status
    assert torch.equal(result.estimate, expected.estimate)
    # prepared data is solved as given; recorded data that is contrast already would be normalized a second time
    assert torch.equal(model.reconstruct(contrast, pixel_size=PIXEL_SIZE, prepared=True).estimate, expected.estimate)
    assert not torch.allclose(model.reconstruct(contrast, pixel_size=PIXEL_SIZE).estimate, expected.estimate)


def test_the_thick_phase_contrast_is_formed_in_the_models_dtype():
    """3.x divided a float32 stack by its float32 mean. With a contrast of 1e-3 on a mean of 1e3 that rounding is
    1e-4 of the contrast, which a float32 model keeps (the phase_3d_beads regression case)."""
    stack = 1e3 * (1 + 1e-3 * torch.randn(SHAPE, generator=torch.Generator().manual_seed(0), dtype=torch.float64))
    single, double = stack.float(), stack / stack.mean() - 1
    model = _phase_3d()
    contrast = model.prepare(stack[None])
    assert torch.equal(contrast, (single / single.mean() - 1)[None])
    assert (contrast[0].double() - double).abs().max() > 1e-5 * double.abs().max()
    assert torch.equal(model.to(dtype=torch.float64).prepare(stack[None]), double[None])


def test_thin_phase_is_normalized_plane_by_plane_and_fluorescence_is_left_as_recorded():
    stack = _stack() * torch.linspace(1, 2, SHAPE[0])[:, None, None]
    sampling = Sampling(shape=SHAPE, pixel_size=PIXEL_SIZE)
    thin = _phase_2d()
    contrast = brightfield_contrast(stack.double(), per_plane=True).float()[None]
    expected = Wiener()(contrast, thin.imaging_model, sampling).estimate
    assert torch.equal(thin.reconstruct(stack, pixel_size=PIXEL_SIZE).estimate, expected)
    assert expected.shape == (2, 1, *SHAPE[1:])
    widefield = _fluorescence_3d()
    expected = Wiener()(stack[None], widefield.imaging_model, sampling).estimate
    assert torch.equal(widefield.reconstruct(stack, pixel_size=PIXEL_SIZE).estimate, expected)


def test_data_may_be_a_nested_list_in_any_dtype_with_or_without_its_channel_axis():
    model = _fluorescence_3d()
    stack = _stack().double()
    expected = model.reconstruct(stack[None].float(), pixel_size=PIXEL_SIZE).estimate
    assert torch.equal(model.reconstruct(stack, pixel_size=PIXEL_SIZE).estimate, expected)
    assert torch.equal(model.reconstruct(stack.float().tolist(), pixel_size=PIXEL_SIZE).estimate, expected)


@pytest.mark.parametrize("shape", [(32, 32), (2, *SHAPE), (1, 1, *SHAPE)])
def test_data_of_another_shape_is_refused_by_the_models_class_name(shape: tuple[int, ...]):
    with pytest.raises(ShapeError, match=r"Phase3D\.reconstruct data has shape"):
        _phase_3d().reconstruct(torch.zeros(shape), pixel_size=PIXEL_SIZE)


def test_z_padding_solves_the_padded_stack_and_keeps_the_measured_slices():
    model = _phase_3d(z_padding=2)
    stack = _stack()
    padding = ZPadding(2)
    sampling = padding.sampling(Sampling(shape=SHAPE, pixel_size=PIXEL_SIZE))
    contrast = brightfield_contrast(padding.pad(stack[None]))
    expected = padding.crop(Wiener()(contrast, model.imaging_model, sampling)).estimate
    estimate = model.reconstruct(stack, pixel_size=PIXEL_SIZE).estimate
    assert estimate.shape == (1, *SHAPE)
    assert torch.equal(estimate, expected)
    assert not torch.allclose(estimate, _phase_3d().reconstruct(stack, pixel_size=PIXEL_SIZE).estimate)


def test_calibrate_moves_a_free_aperture_toward_the_truth():
    aperture = Free(1.05, bounds=(1.0, 1.3))
    model = Fluorescence3D(
        _detection(0.52, aperture),
        index_of_refraction_media=1.33,
        object_prior=Tikhonov(1e-2),
        noise=Gaussian(0.05),
        search=[GridSearch(points=4), GradientSearch(iterations=5)],
    )
    data = _simulated(model, aperture=1.25)
    result = model.calibrate(data, pixel_size=PIXEL_SIZE)
    estimate = result.parameters[APERTURE].item()
    assert abs(estimate - 1.25) < abs(1.05 - 1.25) / 2
    assert model.imaging_model.get_parameter(APERTURE).item() == pytest.approx(estimate)


def test_parts_given_by_argument_replace_the_defaults():
    model = _fluorescence_3d(object_prior=TotalVariation(weight=1e-3))
    assert isinstance(model.object_prior, TotalVariation)
    assert isinstance(model.solver, FISTA)  # no closed form for total variation
    counts = _fluorescence_3d(object_prior=NoPrior(nonnegative=True), noise=Poisson())
    assert isinstance(counts.solver, RichardsonLucy)
    iterative = _fluorescence_3d(solver=FISTA())
    assert isinstance(iterative.solver, FISTA)
    stack = _stack().abs()
    closed = _fluorescence_3d().reconstruct(stack, pixel_size=PIXEL_SIZE)
    assert closed.status == "direct"
    assert iterative.reconstruct(stack, pixel_size=PIXEL_SIZE).status != "direct"
    heavy = _fluorescence_3d(object_prior=Tikhonov(weight=10.0)).reconstruct(stack, pixel_size=PIXEL_SIZE)
    assert heavy.estimate.norm() < closed.estimate.norm()


def test_a_swapped_argument_is_named():
    with pytest.raises(ParameterError, match="Phase3D solver must be") as error:
        _phase_3d(solver=Tikhonov())
    assert error.value.path == "solver"
    with pytest.raises(ParameterError, match=r"pass an instance, Wiener\(\)"):
        _phase_3d(solver=Wiener)
    with pytest.raises(ParameterError, match="Phase3D z_padding must be at least 0"):
        _phase_3d(z_padding=-1)


@pytest.mark.parametrize(
    "pixel_size",
    [
        "0.1", (0.25, 0.1, 0.1, 0.1), ("a", 0.1, 0.1), (0.25, 0.1, -0.1), (0.25, 0.1, True),
        {"z": 0.25, "yx": 0.1, "y": 0.1}, {"z": 0.25, "y": 0.1}, {"zyx": 0.1}, {"z": 0.25, "yx": -0.1},
    ],
)  # fmt: skip
def test_every_verb_names_a_pixel_size_that_is_not_positive_numbers_first(pixel_size: object):
    model: Any = _fluorescence_3d()  # untyped, as from a script: ty would reject these calls statically
    nothing = torch.zeros(2, 2)  # misshapen data, tile and plan: the voxel size is refused before they are read
    calls = {
        "check": lambda: model.check("8x32x32", pixel_size=pixel_size),
        "reconstruct": lambda: model.reconstruct(nothing, pixel_size=pixel_size),
        "calibrate": lambda: model.calibrate(nothing, pixel_size=pixel_size),
        "simulate": lambda: model.simulate(nothing, pixel_size=pixel_size),
        "tile_plan": lambda: model.tile_plan(SHAPE, pixel_size=pixel_size, tile=(8, 16)),
        "field_tile_plan": lambda: model.field_tile_plan(SHAPE, pixel_size=pixel_size),
        "calibrate_tiles": lambda: model.calibrate_tiles(nothing, pixel_size=pixel_size, plan=None),
        "reconstruct_tiles": lambda: model.reconstruct_tiles([], pixel_size=pixel_size, plan=None),
    }
    for verb, call in calls.items():
        message = f"Fluorescence3D.{verb} pixel_size must be positive numbers in um: {FORMS}"
        with pytest.raises(ParameterError, match=re.escape(message)) as error:
            call()
        assert error.value.path == "pixel_size", verb


def test_a_tensor_verb_without_a_voxel_size_anywhere_says_where_to_set_it():
    model = _fluorescence_3d()
    with pytest.raises(ParameterError, match=r"Fluorescence3D\.reconstruct has no voxel size: set data\.pixel_size"):
        model.reconstruct(_stack())
    model.pixel_size = 0.2  # isotropic
    assert model.pixel_size == (0.2, 0.2, 0.2)
    model.pixel_size = (0.4, 0.1)  # (z, yx)
    assert model.pixel_size == (0.4, 0.1, 0.1)
    with pytest.raises(ParameterError, match=r"got the keys \['yx', 'z'\]"):
        model.pixel_size = {"z": 0.4, "yx": 0.12}  # the {z, yx} form of version 3 configurations is gone
    model.pixel_size = {"x": 0.11, "z": 0.4, "y": 0.12}
    assert model.pixel_size == (0.4, 0.12, 0.11)
    assert _fluorescence_3d(pixel_size={"z": 0.25, "y": 0.1, "x": 0.1}).pixel_size == PIXEL_SIZE
    mapped = model.reconstruct(_stack(), pixel_size=dict(zip("zyx", PIXEL_SIZE, strict=True))).estimate
    given = _fluorescence_3d(pixel_size=PIXEL_SIZE)
    assert torch.equal(
        given.reconstruct(_stack()).estimate, model.reconstruct(_stack(), pixel_size=PIXEL_SIZE).estimate
    )
    assert torch.equal(mapped, given.reconstruct(_stack()).estimate)


def test_a_solver_the_parts_cannot_run_is_refused_when_it_is_given():
    model = _phase_3d()
    model.check(SHAPE, pixel_size=PIXEL_SIZE)
    with pytest.raises(IncompatibleModelError, match=r"Phase3D solver: Wiener.*object_prior=TotalVariation"):
        _phase_3d(object_prior=TotalVariation(weight=1e-3), solver=Wiener())
    loose: Any = model  # untyped, as from a script: ty would reject a shape of two entries statically
    with pytest.raises(ParameterError, match=r"Phase3D\.check shape"):
        loose.check((8, 32), pixel_size=PIXEL_SIZE)


@pytest.mark.parametrize(
    ("raw", "model_class"),
    [
        (_phase_raw(), Phase3D),
        (_phase_raw(2), Phase2D),
        (_fluorescence_raw(), Fluorescence3D),
        (_fluorescence_raw(pinhole={"diameter_au": 1.0}), Fluorescence3D),
        (_fluorescence_raw(2), Fluorescence2D),
    ],
)
def test_from_config_returns_the_class_of_the_contrast_and_dimension(raw: dict, model_class: type):
    config = V4Config(**raw)
    model = ReconstructionModel.from_config(config)
    assert type(model) is model_class
    assert model.config is config
    assert type(model_class.from_config(config)) is model_class
    other = Fluorescence2D if model_class is Phase3D else Phase3D
    with pytest.raises(Rejected) as rejected:
        other.from_config(config)
    (problem,) = rejected.value.problems
    assert problem.message.startswith("imaging_system.contrast: builds a ")
    assert f"which {other.__name__} does not wrap" in problem.message


def _reconstruction(model: ReconstructionModel) -> Tensor:
    return model.reconstruct(_stack(), pixel_size=PIXEL_SIZE).estimate


def test_a_yaml_file_builds_the_model_its_mapping_builds(tmp_path):
    raw = _phase_raw()
    path = tmp_path / "phase.yml"
    path.write_text(yaml.safe_dump(raw))
    loaded = ReconstructionModel.from_config(path)
    assert type(loaded) is Phase3D
    assert loaded.config == V4Config(**raw)
    assert torch.equal(_reconstruction(loaded), _reconstruction(ReconstructionModel.from_config(raw)))


@pytest.mark.parametrize("build", [_phase_3d, lambda: ReconstructionModel.from_config(_phase_raw())])
def test_a_saved_model_loads_and_reconstructs_the_same(tmp_path, build: Callable[[], ReconstructionModel]):
    model = build()
    folder = model.save(tmp_path / "model")
    assert (folder / "model.json").is_file()
    loaded = ReconstructionModel.load(folder)
    assert type(loaded) is Phase3D
    assert loaded.config is None  # a saved model is its parts, not a configuration
    assert _same_values(loaded, model)
    assert torch.equal(_reconstruction(loaded), _reconstruction(model))


def test_a_calibrated_config_model_writes_its_config_unchanged_with_the_calibrated_values(tmp_path):
    config = V4Config(**_free_aperture_raw())
    model = ReconstructionModel.from_config(config)
    assert isinstance(model, Fluorescence3D)
    assert model.to_config() == config  # nothing moved: an empty calibrated block
    result = model.calibrate(_simulated(model, aperture=1.25), pixel_size=PIXEL_SIZE)
    estimate = result.parameters[APERTURE].item()
    assert abs(estimate - 1.25) < abs(1.05 - 1.25) / 2
    written = model.to_config(channels=["DAPI"], pixel_size=(0.5, 0.2, 0.2), times=[0, 2])
    assert isinstance(written, V4Config)
    assert written.imaging_system == config.imaging_system  # the configured prior, not rewritten
    assert written.calibrated == {APERTURE: pytest.approx(estimate)}
    assert written.data.arguments() == {"channels": ["DAPI"], "pixel_size": (0.5, 0.2, 0.2), "times": [0, 2]}
    rebuilt = ReconstructionModel.from_config(written)
    assert type(rebuilt) is Fluorescence3D
    assert _same_values(rebuilt, model)
    assert type(rebuilt.score) is type(model.score)
    assert [type(stage) for stage in rebuilt.search or ()] == [type(stage) for stage in model.search or ()]
    saved = ReconstructionModel.load(model.save(tmp_path / "calibrated"))  # the calibrated values come back
    assert saved.imaging_model.get_parameter(APERTURE).item() == pytest.approx(estimate)
    assert torch.equal(_reconstruction(saved), _reconstruction(model))


def test_the_calibrated_block_holds_the_zernike_terms_and_the_pinhole_by_parameter_name():
    raw = _free_aperture_raw()
    detection = raw["imaging_system"]["detection"]
    detection["pupil_aberrations"] = {"zernike_up_to_order": {"order": 2, "tolerance": 0.05}}
    detection["pinhole"] = {"diameter_au": {"nominal": 1.0, "tolerance": 0.2, "bounds": [0.5, 2.0]}}
    raw["solver"]["calibration"]["search"] = [{"gradient": {"max_iter": 5}}]
    model = ReconstructionModel.from_config(raw)
    values = {APERTURE: 1.2, "system.detection.pupil_aberrations.defocus": 0.03, "system.pinhole_diameter": 0.7}
    load_parameters(model.imaging_model, {name: torch.tensor(value) for name, value in values.items()})
    config = model.to_config()
    assert isinstance(config, V4Config)
    assert config.imaging_system == V4Config(**raw).imaging_system  # the shorthand and the Airy units stay
    assert config.calibrated == pytest.approx(values)
    assert _same_values(ReconstructionModel.from_config(config), model)


def test_a_calibrated_value_is_written_as_the_shortest_decimal_that_gives_the_same_parameter(tmp_path):
    raw = _free_aperture_raw()
    raw["imaging_system"]["detection"]["numerical_aperture"] = {"nominal": 0.55, "tolerance": 0.1, "bounds": [0.4, 0.6]}
    model = ReconstructionModel.from_config(raw)
    parameter = model.imaging_model.get_parameter(APERTURE)
    assert parameter.item() != 0.55  # float32 holds 0.550000011920929
    untouched = model.to_config()
    assert isinstance(untouched, V4Config)
    assert untouched.calibrated == {}  # at its nominal, the value moved nowhere
    load_parameters(model.imaging_model, {APERTURE: torch.tensor(0.52)})
    assert parameter.item() != 0.52  # 0.5199999809265137
    config = model.to_config()
    assert isinstance(config, V4Config)
    assert config.calibrated == {APERTURE: 0.52}
    rebuilt = ReconstructionModel.from_config(config)
    assert torch.equal(rebuilt.imaging_model.get_parameter(APERTURE), parameter)
    assert _same_values(rebuilt, model)
    text = (model.save(tmp_path / "saved") / "model.json").read_text()
    assert "0.52" in text
    assert "0.5199999" not in text


def test_a_python_built_model_is_written_as_its_description():
    described = _described(_fluorescence_3d(object_prior=Tikhonov(weight=0.01)), channels="GFP")
    assert (described["schema"], described["class"]) == ("waveorder.model/1", "Fluorescence3D")
    parts = described["parts"]
    assert list(parts["system"]) == ["widefield_fluorescence_3d"]
    assert parts["object_prior"] == {"tikhonov": {"weight": 0.01, "nonnegative": False}}
    assert (parts["score"], parts["search"]) == (None, None)
    assert described["settings"] == {
        "z_padding": 0,
        "field_tiles": None,
        "pixel_size": None,
        "apodization_rolloff": 0.0,
    }
    assert described["data"] == {"input_channel_names": ["GFP"]}


def test_calibrating_a_model_that_frees_nothing_says_how_to_free_a_quantity():
    model = ReconstructionModel.from_config(V4Config(**_fluorescence_raw()))
    with pytest.raises(ConfigurationError, match=r"\{nominal: \.\.\., tolerance: \.\.\., bounds") as error:
        model.calibrate(torch.zeros(1, *SHAPE), pixel_size=PIXEL_SIZE)
    assert error.value.field == "imaging_system"
    with pytest.raises(ParameterError, match=r"Fluorescence3D\.calibrate: .*Free\(nominal, bounds=\(low, high\)\)"):
        _fluorescence_3d().calibrate(torch.zeros(1, *SHAPE), pixel_size=PIXEL_SIZE)
    plan = TilePlan(shape=SHAPE, tile=SHAPE)
    with pytest.raises(ParameterError, match=r"Fluorescence3D\.calibrate_tiles: .*Free\("):
        _fluorescence_3d().calibrate_tiles(torch.zeros(1, *SHAPE), pixel_size=PIXEL_SIZE, plan=plan)


def test_missing_field_tiles_are_named_by_argument_or_by_yaml_path():
    with pytest.raises(ConfigurationError) as error:
        _fluorescence_3d().field_tile_plan(SHAPE, pixel_size=PIXEL_SIZE)
    assert error.value.field == "field_tiles"
    configured = ReconstructionModel.from_config(V4Config(**_fluorescence_raw()))
    with pytest.raises(ConfigurationError) as error:
        configured.field_tile_plan(SHAPE, pixel_size=PIXEL_SIZE)
    assert error.value.field == "imaging_system.field_dependence"
    tiled = _fluorescence_3d(field_tiles=FieldTiles(size_yx=(16, 16), overlap_yx=(4, 4), blend="cosine"))
    plan = tiled.field_tile_plan(SHAPE, pixel_size=PIXEL_SIZE)
    assert (plan.tile, plan.overlap, plan.blend) == ((8, 16, 16), (0, 4, 4), "cosine")


class Gain(ShiftInvariantSystem):
    """A made-up system: the data are the object times an amplitude."""

    def __init__(self, amplitude: Quantity) -> None:
        super().__init__(fields=("density",), channel_count=1)
        self.register_quantity("amplitude", amplitude, positive=True)

    @override
    def compute_transfer(self, sampling: Sampling) -> Tensor:
        amplitude = self.quantity("amplitude")
        nz, ny, nx = sampling.shape
        return amplitude * torch.ones(1, 1, nz, ny, nx // 2 + 1, dtype=torch.complex64, device=amplitude.device)


class Loss(Gain):
    """A second made-up system, which no model class lists: an instance of a listed one, so its class wraps it."""


class GainModel(ReconstructionModel):
    """A user-defined modality: it wraps :class:`Gain` and records data with an offset, which ``setup`` sets."""

    system_types = (Gain,)
    prepare_scope = "voxel"  # an offset: each voxel by itself

    def __init__(self, *, amplitude: Quantity, **parts) -> None:
        super().__init__(Gain(amplitude), **parts)

    @override
    def setup(self) -> None:
        self.offset = 100.0

    @override
    def prepare(self, data: Tensor, *, mean: Tensor | None = None) -> Tensor:
        return data - self.offset


class Flat(ShiftInvariantSystem):
    """A made-up system no model class lists: the data are the object."""

    def __init__(self) -> None:
        super().__init__(fields=("density",), channel_count=1)
        self.register_quantity("level", 1.0)

    @override
    def compute_transfer(self, sampling: Sampling) -> Tensor:
        nz, ny, nx = sampling.shape
        return torch.ones(1, 1, nz, ny, nx // 2 + 1, dtype=torch.complex64)


def test_a_user_defined_model_reconstructs_and_leaves_from_config_alone():
    model = GainModel(amplitude=2.0, object_prior=Tikhonov(weight=1e-8))
    obj = torch.randn(1, *SHAPE, generator=torch.Generator().manual_seed(0))
    estimate = model.reconstruct(2.0 * obj + 100.0, pixel_size=PIXEL_SIZE).estimate
    assert torch.allclose(estimate, obj, atol=1e-4)
    assert model.output_channel_names(["anything"]) == ["density"]
    with pytest.raises(ParameterError, match="GainModel system must be a Gain"):
        ReconstructionModel.__init__(GainModel.__new__(GainModel), _fluorescence_3d().system)
    assert type(ReconstructionModel.from_config(V4Config(**_phase_raw()))) is Phase3D


def test_output_channel_names_are_the_3x_names():
    assert _phase_3d().output_channel_names(["BF"]) == ["Phase3D"]
    assert _phase_2d().output_channel_names(["BF"]) == ["Phase2D", "Absorption2D"]
    assert _fluorescence_3d().output_channel_names(["GFP"]) == ["GFP_Density3D"]
    assert _confocal_3d().output_channel_names(["GFP"]) == ["GFP_Density3D"]
    assert _fluorescence_2d().output_channel_names(["GFP"]) == ["GFP_Density2D"]


def test_to_returns_the_model_and_keeps_its_device_in_step():
    model = _phase_3d()
    assert model.to("cpu") is model
    assert model.to(torch.device("cpu")).device == torch.device("cpu")
    assert all(parameter.device == model.device for parameter in model.imaging_model.parameters())
    with pytest.raises(ConfigurationError) as error:
        model.to("nowhere")
    assert error.value.field == "device"


# --- one data convention: the tile methods take recorded data, as reconstruct and calibrate do ---------------------


def _free_illumination_phase(**parts) -> Phase3D:
    aperture = Free(0.45, bounds=(0.3, 0.7))
    return Phase3D(
        Illumination(wavelength=0.532, numerical_aperture=aperture),
        _detection(),
        index_of_refraction_media=1.33,
        object_prior=Tikhonov(1e-2),
        noise=Gaussian(0.01),
        search=[GridSearch(points=9)],
        **parts,
    )


def _recorded_brightfield(aperture: float = 0.6) -> Tensor:
    """A brightfield stack ``(Z, Y, X)`` in camera counts whose contrast a model of ``aperture`` produced."""
    model = _free_illumination_phase()
    truth = copy.deepcopy(model.imaging_model)
    load_parameters(truth, {"system.illumination.numerical_aperture": torch.tensor(aperture)})
    sampling = Sampling(shape=SHAPE, pixel_size=PIXEL_SIZE)
    generator = torch.Generator().manual_seed(0)
    obj = torch.randn(truth.system.object_shape(sampling), generator=generator)
    with torch.no_grad():
        contrast = truth.noise.sample(truth(obj, sampling), generator=generator)
    return 1e3 * (1 + contrast[0])


def test_reconstruct_tiles_on_a_recorded_stack_equals_reconstruct():
    model = _phase_3d()
    stack = _stack()
    whole = model.reconstruct(stack, pixel_size=PIXEL_SIZE).estimate
    one_tile = torch.zeros(1, *SHAPE)
    model.reconstruct_tiles([(stack[None], one_tile)], pixel_size=PIXEL_SIZE, plan=TilePlan(shape=SHAPE, tile=SHAPE))
    assert torch.allclose(one_tile, whole, rtol=1e-5, atol=1e-7)
    # several tiles share the mean of the whole stack: recorded data gives what its contrast gives
    plan = model.tile_plan(SHAPE, pixel_size=PIXEL_SIZE, tile=(8, 16, 16), overlap=(0, 4, 4))
    assert len(plan.regions) == 4
    contrast = brightfield_contrast(stack)
    from_recorded, from_contrast, unprepared = (torch.zeros(1, *SHAPE) for _ in range(3))
    model.reconstruct_tiles([(stack, from_recorded)], pixel_size=PIXEL_SIZE, plan=plan)  # (Z, Y, X): one channel
    model.reconstruct_tiles([(contrast[None], from_contrast)], pixel_size=PIXEL_SIZE, plan=plan, prepared=True)
    assert torch.allclose(from_recorded, from_contrast, rtol=1e-5, atol=1e-7)
    model.reconstruct_tiles([(stack[None], unprepared)], pixel_size=PIXEL_SIZE, plan=plan, prepared=True)
    assert unprepared.abs().max() > 100 * from_recorded.abs().max()  # counts solved as if they were contrast


def test_calibrate_tiles_on_a_recorded_stack_equals_calibrate():
    stack = _recorded_brightfield()
    name = "system.illumination.numerical_aperture"
    plan = TilePlan(shape=SHAPE, tile=SHAPE)
    model = _free_illumination_phase()
    tiled = model.calibrate_tiles(stack, pixel_size=PIXEL_SIZE, plan=plan).parameters[name]
    assert model.imaging_model.get_parameter(name).item() == pytest.approx(0.45)  # per tile: the model is kept
    contrast = brightfield_contrast(stack.double()).float()[None]
    prepared = model.calibrate_tiles(contrast, pixel_size=PIXEL_SIZE, plan=plan, prepared=True).parameters[name]
    whole = model.calibrate(stack, pixel_size=PIXEL_SIZE).parameters[name]
    assert tiled.shape == (1,)
    assert 0.3 < whole.item() < 0.7  # an interior estimate, so the comparison says something
    assert tiled.item() == pytest.approx(whole.item(), rel=1e-5)
    assert prepared.item() == pytest.approx(whole.item(), rel=1e-5)
    assert model.imaging_model.get_parameter(name).item() == pytest.approx(whole.item())  # the whole field: updated


class BlockModel(GainModel):
    """The same modality, declaring that the result of its ``prepare`` depends on the block it is given."""

    prepare_scope = "block"


def test_a_class_whose_prepare_depends_on_the_block_is_not_split_into_tiles():
    model = BlockModel.for_system(Gain(2.0), object_prior=Tikhonov(weight=1e-8))
    obj = torch.randn(1, *SHAPE, generator=torch.Generator().manual_seed(0))
    recorded, sink = 2.0 * obj + 100.0, torch.zeros(1, *SHAPE)
    model.reconstruct_tiles([(recorded, sink)], pixel_size=PIXEL_SIZE, plan=TilePlan(shape=SHAPE, tile=SHAPE))
    assert torch.allclose(sink, obj, atol=1e-4)
    plan = TilePlan(shape=SHAPE, tile=(8, 16, 16))
    with pytest.raises(ParameterError, match=r"prepare_scope is 'block'.*prepared=True") as error:
        model.reconstruct_tiles([(recorded, sink)], pixel_size=PIXEL_SIZE, plan=plan)
    assert error.value.path == "plan"
    model.reconstruct_tiles([(recorded - 100.0, sink)], pixel_size=PIXEL_SIZE, plan=plan, prepared=True)
    assert torch.allclose(sink, obj, atol=1e-4)


class Counted:
    """A ``(C, Z, Y, X)`` array that counts the reads of whole z extents (what a mean of the stack takes)."""

    def __init__(self, data: Tensor) -> None:
        self.data = data
        self.shape = tuple(data.shape)
        self.reads: list[tuple[int, ...]] = []

    def __getitem__(self, key: Any) -> Tensor:
        block = self.data[key]
        self.reads.append(tuple(block.shape))
        return block


def test_a_voxel_wise_prepare_is_tiled_without_reading_the_stack_for_a_mean():
    assert (Phase3D.prepare_scope, Phase2D.prepare_scope) == ("stack_mean", "block")
    assert ReconstructionModel.prepare_scope == "voxel"  # the base prepare returns the data as recorded
    model = GainModel(amplitude=2.0, object_prior=Tikhonov(weight=1e-8))
    obj = torch.randn(1, *SHAPE, generator=torch.Generator().manual_seed(0))
    source, sink = Counted(2.0 * obj + 100.0), torch.zeros(1, *SHAPE)
    plan = TilePlan(shape=SHAPE, tile=(8, 16, 16))
    model.reconstruct_tiles([(source, sink)], pixel_size=PIXEL_SIZE, plan=plan)
    assert torch.allclose(sink, obj, atol=1e-4)
    assert len(source.reads) == len(plan.regions)  # one read per tile, none for a mean
    stack = Counted(_stack()[None])
    _phase_3d().reconstruct_tiles([(stack, torch.zeros(1, *SHAPE))], pixel_size=PIXEL_SIZE, plan=plan)
    assert len(stack.reads) == len(plan.regions) + 1  # the tiles, and the stack once for its mean
    flat = Counted(_stack()[None])
    _fluorescence_3d().reconstruct_tiles([(flat, torch.zeros(1, *SHAPE))], pixel_size=PIXEL_SIZE, plan=plan)
    assert len(flat.reads) == len(plan.regions)


@pytest.mark.parametrize("verb", ["reconstruct", "calibrate"])
def test_a_model_with_field_tiles_refuses_the_methods_that_take_the_whole_field(verb: str):
    tiles = FieldTiles(size_yx=(16, 16), overlap_yx=(4, 4))
    model = Fluorescence3D(
        _detection(0.52, Free(1.05, bounds=(1.0, 1.3))), index_of_refraction_media=1.33, field_tiles=tiles
    )
    with pytest.raises(ConfigurationError) as error:
        getattr(model, verb)(_stack().abs(), pixel_size=PIXEL_SIZE)
    assert error.value.field == "field_tiles"
    for named in ("field_tile_plan", "calibrate_tiles", "reconstruct_tiles", "calibrate_position"):
        assert named in str(error.value)
    assert "reconstruct_positions" in str(error.value)
    raw = _free_aperture_raw()
    raw["imaging_system"]["field_dependence"] = {"tiled": {"size_yx": [16, 16], "overlap_yx": [4, 4]}}
    configured = ReconstructionModel.from_config(V4Config(**raw))
    with pytest.raises(ConfigurationError, match="calibrate_tiles and reconstruct_tiles") as error:
        getattr(configured, verb)(_stack().abs(), pixel_size=PIXEL_SIZE)
    assert error.value.field == "imaging_system.field_dependence"
    configured.field_tiles = None  # the whole field as one system
    assert configured.reconstruct(_stack().abs(), pixel_size=PIXEL_SIZE).status == "direct"


# --- one class per system, on every construction path --------------------------------------------------------------


def test_for_system_builds_the_class_that_wraps_a_system():
    thick = ThickPhase3D(_illumination(), _detection(), index_of_refraction_media=1.33, cache="auto")
    model = ReconstructionModel.for_system(thick, z_padding=2)
    assert type(model) is Phase3D
    assert model.system is thick
    assert model.z_padding == 2
    assert model.output_channel_names(["BF"]) == ["Phase3D"]
    stack = _stack()
    expected = _phase_3d(z_padding=2).reconstruct(stack, pixel_size=PIXEL_SIZE).estimate
    assert torch.equal(model.reconstruct(stack, pixel_size=PIXEL_SIZE).estimate, expected)
    assert type(Phase3D.for_system(thick)) is Phase3D
    assert type(ReconstructionModel.for_system(_confocal_3d().system)) is Fluorescence3D
    assert type(ReconstructionModel.for_system(Flat())) is ReconstructionModel  # a system no class lists
    assert type(ReconstructionModel.for_system(Loss(2.0))) is GainModel  # an instance of a listed system
    with pytest.raises(ParameterError, match=r"Fluorescence3D system must be .*for_system") as error:
        Fluorescence3D.for_system(thick)
    assert error.value.path == "system"
    with pytest.raises(ParameterError, match=r"ReconstructionModel\.for_system system must be"):
        ReconstructionModel.for_system("phase")  # ty: ignore[invalid-argument-type]


def test_the_base_class_refuses_a_system_another_class_wraps():
    thick = _phase_3d().system
    with pytest.raises(ConfigurationError, match=r"Phase3D wraps.*ReconstructionModel\.for_system\(system") as error:
        ReconstructionModel(thick)
    assert error.value.field == "system"
    with pytest.raises(ConfigurationError, match="GainModel wraps"):
        ReconstructionModel(Gain(2.0))
    assert type(ReconstructionModel(Flat())) is ReconstructionModel


def test_two_classes_cannot_list_one_system_type():
    with pytest.raises(ParameterError, match=r"Rival\.system_types lists Gain, which GainModel already wraps") as error:

        class Rival(ReconstructionModel):
            system_types = (Gain,)

    assert error.value.path == "system_types"
    assert type(ReconstructionModel.for_system(Gain(2.0))) is GainModel


class ScaledPhase(Phase3D):
    """A modality built on a built-in one, with state its ``prepare`` reads."""

    scale = 1.0  # the class-level default
    prepare_scope = "stack_mean"  # stated again beside the prepare it describes

    @override
    def setup(self) -> None:
        self.scale = 2.0 if self.system.invert_phase_contrast else 1.0

    @override
    def prepare(self, data: Tensor, *, mean: Tensor | None = None) -> Tensor:
        return self.scale * super().prepare(data, mean=mean)


def test_setup_runs_on_every_construction_path():
    inverted = ThickPhase3D(_illumination(), _detection(), index_of_refraction_media=1.33, invert_phase_contrast=True)
    built = [
        ScaledPhase(_illumination(), _detection(), index_of_refraction_media=1.33, invert_phase_contrast=True),
        ScaledPhase.for_system(inverted),
    ]
    raw = _phase_raw()
    raw["imaging_system"]["illumination"]["invert_phase_contrast"] = True
    built.append(ScaledPhase.from_config(V4Config(**raw)))
    stack = _stack()
    expected = 2.0 * ReconstructionModel.for_system(inverted).reconstruct(stack, pixel_size=PIXEL_SIZE).estimate
    for model in built:
        assert type(model) is ScaledPhase
        assert model.scale == 2.0
        assert torch.allclose(model.reconstruct(stack, pixel_size=PIXEL_SIZE).estimate, expected, rtol=1e-5)
    assert ScaledPhase.from_config(V4Config(**_phase_raw())).scale == 1.0
    assert GainModel(amplitude=2.0).offset == GainModel.for_system(Gain(2.0)).offset == 100.0


# --- one source of truth -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["device", "dtype", "config", "system", "defaulted", "imaging_model"])
def test_what_is_read_from_the_models_state_cannot_be_assigned(name: str):
    model = _phase_3d(z_padding=2)
    with pytest.raises(AttributeError):
        setattr(model, name, getattr(model, name))


def test_dtype_and_device_follow_the_modules():
    model = _phase_3d()
    assert model.to(dtype=torch.float64) is model
    assert model.device == torch.device("cpu")
    tensors = [*model.imaging_model.parameters(), *model.imaging_model.buffers(), *model.solver.buffers()]
    assert tensors
    assert all(tensor.dtype == torch.float64 for tensor in tensors if tensor.is_floating_point())
    aperture = model.imaging_model.get_buffer("system.detection.numerical_aperture")
    assert aperture.item() == 1.2  # filled again from the number given, not float32's 1.2000000476837158
    assert model.reconstruct(_stack(), pixel_size=PIXEL_SIZE).estimate.dtype == torch.float64
    with pytest.raises(ParameterError, match=r"Phase3D\.to device is the dtype torch\.float32; pass it as dtype="):
        model.to(torch.float32)  # ty: ignore[invalid-argument-type]
    for dtype in (torch.int32, torch.float16, torch.complex64):
        with pytest.raises(ParameterError, match=r"dtype must be torch\.float32 or torch\.float64") as error:
            model.to(dtype=dtype)
        assert error.value.path == "dtype"
    assert model.reconstruct(_stack(), pixel_size=PIXEL_SIZE).estimate.dtype == torch.float64  # nothing was moved


def test_parts_follow_the_placement_of_the_system():
    system = ThickPhase3D(_illumination(), _detection(), index_of_refraction_media=1.33).double()
    model = ReconstructionModel.for_system(system, solver=FISTA())
    parts = (model.object_prior, model.noise, model.solver)
    assert all(tensor.dtype == torch.float64 for part in parts for tensor in part.buffers())
    assert model.reconstruct(_stack(), pixel_size=PIXEL_SIZE).estimate.dtype == torch.float64


def _described(model: ReconstructionModel, **data: Any) -> dict[str, Any]:
    """Return the description ``to_config`` gives a model without a stored configuration."""
    described = model.to_config(**data)
    assert isinstance(described, dict)
    return described


def test_an_assigned_part_is_checked_and_placed_and_the_stored_config_is_dropped(tmp_path):
    model = ReconstructionModel.from_config(V4Config(**_free_aperture_raw())).to(dtype=torch.float64)
    assert model.config is not None
    stored = model.to_config()
    assert isinstance(stored, V4Config)
    assert stored.solver.reconstruction == "closed_form"
    with pytest.raises(ParameterError, match="Fluorescence3D solver must be"):
        model.solver = Tikhonov()  # ty: ignore[invalid-assignment]
    assert model.config is not None  # a refused assignment changes nothing
    model.solver = FISTA()
    assert all(buffer.dtype == torch.float64 for buffer in model.solver.buffers())
    assert model.config is None
    assert list(_described(model)["parts"]["solver"]) == ["fista"]
    rebuilt = ReconstructionModel.load(model.save(tmp_path / "model"))
    assert isinstance(rebuilt.solver, FISTA)
    assert rebuilt.config is None


@pytest.mark.parametrize(
    ("name", "value"),
    [("score", None), ("search", [GridSearch(points=3)]), ("field_tiles", None)],
)
def test_assigning_a_calibration_setting_drops_the_stored_config(name: str, value: object):
    model = ReconstructionModel.from_config(V4Config(**_free_aperture_raw()))
    setattr(model, name, value)
    assert model.config is None
    _described(model)  # the live parts have a plain-data form


def test_replacing_a_module_of_the_imaging_model_drops_the_stored_config():
    model = ReconstructionModel.from_config(V4Config(**_free_aperture_raw()))
    load_parameters(model.imaging_model, {APERTURE: torch.tensor(1.2)})
    assert model.config is not None  # a value that changes in place keeps it
    model.imaging_model.noise = Gaussian(0.2)
    assert model.config is None
    assert _described(model)["parts"]["noise"] == {"gaussian": {"sigma": 0.2}}


def test_a_bad_setting_is_refused_when_it_is_given():
    model = _phase_3d()
    with pytest.raises(ParameterError, match=r"Phase3D search\.0 must be"):
        model.search = ["grid"]  # ty: ignore[invalid-assignment]
    with pytest.raises(ParameterError, match="Phase3D field_tiles must be"):
        model.field_tiles = (16, 16)  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("build", "name"),
    [  # the wrong types are the point
        (lambda: Fluorescence3D("x", index_of_refraction_media=1.33), "detection"),  # ty: ignore[invalid-argument-type]
        (lambda: Fluorescence2D(_detection(), index_of_refraction_media="1.33"), "index_of_refraction_media"),  # ty: ignore[invalid-argument-type]
        (lambda: Phase3D(_illumination(), "x", index_of_refraction_media=1.33), "detection"),  # ty: ignore[invalid-argument-type]
        (lambda: Phase2D(_detection(), _detection(), index_of_refraction_media=1.33), "illumination"),  # ty: ignore[invalid-argument-type]
    ],
)  # fmt: skip
def test_the_optics_a_model_is_built_from_are_checked_at_its_constructor(build: Callable[[], object], name: str):
    with pytest.raises(ParameterError, match=f"{name} must be") as error:
        build()
    assert error.value.path == name


# --- the same defaults in Python and in a configuration ------------------------------------------------------------


@pytest.mark.parametrize("noise", [Poisson(), PoissonGaussian(sigma=2.0)])
def test_noise_that_counts_photons_defaults_to_richardson_lucy_without_a_prior(noise: Any):
    model = _fluorescence_3d(noise=noise)
    assert isinstance(model.object_prior, NoPrior)
    assert isinstance(model.solver, RichardsonLucy)
    model.check(SHAPE, pixel_size=PIXEL_SIZE)
    result = model.reconstruct(_stack().abs(), pixel_size=PIXEL_SIZE)
    assert result.finite
    assert not result.estimate.requires_grad


class SlowModel(ReconstructionModel):
    """A modality that chooses its own default solver."""

    @override
    def default_solver(self) -> FISTA:
        return FISTA()


def test_the_default_solver_is_a_hook_and_the_closed_form_only_where_it_runs():
    assert isinstance(SlowModel(Flat()).solver, FISTA)
    assert isinstance(SlowModel.for_system(Flat()).solver, FISTA)
    assert isinstance(SlowModel(Flat(), solver=Wiener()).solver, Wiener)
    assert isinstance(ReconstructionModel(Flat()).solver, Wiener)
    assert isinstance(ReconstructionModel(Flat(), object_prior=Tikhonov(nonnegative=True)).solver, FISTA)


def test_free_quantities_without_a_calibration_block_calibrate_as_in_python():
    raw = _free_aperture_raw()
    del raw["solver"]
    configured = ReconstructionModel.from_config(V4Config(**raw))
    assert configured.score is None
    assert configured.search is None
    built = Fluorescence3D(
        _detection(0.52, Free(1.05, tolerance=0.2, bounds=(1.0, 1.3))),
        index_of_refraction_media=1.33,
        object_prior=Tikhonov(1e-2),
        noise=Gaussian(0.05),
    )
    data = _simulated(built, aperture=1.25)
    config = configured.config
    assert config is not None
    from_config = configured.calibrate(data, pixel_size=config.data.pixel_size)  # the schema's PixelSize
    in_python = built.calibrate(data, pixel_size=list(PIXEL_SIZE))
    assert from_config.parameters[APERTURE].item() == pytest.approx(in_python.parameters[APERTURE].item(), rel=1e-5)
    assert abs(in_python.parameters[APERTURE].item() - 1.25) < 0.1


@pytest.mark.parametrize(
    "pixel_size", [list(PIXEL_SIZE), torch.tensor(PIXEL_SIZE, dtype=torch.float64), iter(PIXEL_SIZE)]
)
def test_a_voxel_size_is_any_three_numbers_or_the_schemas(pixel_size: Any):
    model = _fluorescence_3d()
    expected = model.reconstruct(_stack(), pixel_size=PIXEL_SIZE).estimate
    assert torch.equal(model.reconstruct(_stack(), pixel_size=pixel_size).estimate, expected)
    data = V4Config(**_fluorescence_raw()).data
    assert torch.equal(model.reconstruct(_stack(), pixel_size=data.pixel_size).estimate, expected)
    written = _described(model, channels=["GFP"], pixel_size=data.pixel_size, times=None)["data"]
    assert written == {"input_channel_names": ["GFP"], "pixel_size": dict(zip("zyx", PIXEL_SIZE, strict=True))}
    assert _described(model, times=[1, 3])["data"] == {"time_indices": [1, 3]}
    configured = ReconstructionModel.from_config(_fluorescence_raw()).to_config(pixel_size=list(PIXEL_SIZE), times=[1])
    assert isinstance(configured, V4Config)
    assert configured.data.arguments() == {"channels": ["GFP"], "pixel_size": PIXEL_SIZE, "times": [1]}


# --- a likelihood needs the data's units ---------------------------------------------------------------------------


def test_calibrating_a_likelihood_with_default_noise_or_prior_warns_once():
    def model(**parts) -> Fluorescence3D:
        aperture = Free(1.05, bounds=(1.0, 1.3))
        return Fluorescence3D(
            _detection(0.52, aperture), index_of_refraction_media=1.33, search=[GridSearch(points=3)], **parts
        )

    data = _stack().abs()
    both = model()
    assert both.defaulted == ("object_prior", "noise")
    with pytest.warns(CalibrationWarning) as caught:
        both.calibrate(data, pixel_size=PIXEL_SIZE)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert message.startswith("Fluorescence3D.calibrate: Evidence is a likelihood")
    for expected in ("object_prior (Tikhonov", "noise (Gaussian", "noise=Gaussian(sigma=...)", "Tikhonov(Free(...))"):
        assert expected in message
    assert caught[0].filename == __file__  # the warning points at the call
    prior_only = model(noise=Gaussian(30.0))
    assert prior_only.defaulted == ("object_prior",)
    with pytest.warns(CalibrationWarning, match=r"object_prior \(Tikhonov.* is the default") as caught:
        prior_only.calibrate_tiles(data, pixel_size=PIXEL_SIZE, plan=TilePlan(shape=SHAPE, tile=SHAPE))
    assert "noise (" not in str(caught[0].message)
    assert caught[0].filename == __file__
    given = model(noise=Gaussian(30.0), object_prior=Tikhonov(Free(1e-2, bounds=(1e-6, 1e2))))
    assert given.defaulted == ()
    given.calibrate(data, pixel_size=PIXEL_SIZE)  # warnings are errors: none is raised
    both.imaging_model.noise = Gaussian(30.0)
    both.imaging_model.object_prior = Tikhonov(1e-2)
    assert both.defaulted == ()  # a part replaced since is no default


def test_a_configuration_that_leaves_out_the_noise_and_the_weight_warns_like_python():
    raw = _free_aperture_raw()
    del raw["noise"], raw["object"]["weight"]
    model = ReconstructionModel.from_config(V4Config(**raw))
    assert model.defaulted == ("object_prior", "noise")
    with pytest.warns(CalibrationWarning, match="noise.gaussian.sigma and object.weight"):
        model.calibrate(_stack().abs(), pixel_size=PIXEL_SIZE)
    assert ReconstructionModel.from_config(V4Config(**_free_aperture_raw())).defaulted == ()


# --- signatures ----------------------------------------------------------------------------------------------------


def test_the_position_methods_take_their_output_by_keyword_and_the_model_estimates_no_resources():
    parameters = inspect.signature(ReconstructionModel.reconstruct_positions).parameters
    assert parameters["output"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["times"].default is None
    assert not hasattr(ReconstructionModel, "estimate_resources")
    assert not hasattr(ReconstructionModel, "create_output")
    for method in (ReconstructionModel.reconstruct_tiles, ReconstructionModel.calibrate_tiles):
        assert inspect.signature(method).parameters["prepared"].default is False
    for name in ("pixel_size", "times", "tile", "tile_overlap", "write_pixel_size", "overwrite"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    calibrate = inspect.signature(ReconstructionModel.calibrate_position).parameters
    assert "start" in calibrate
    assert "init" not in calibrate


# --- reconstruct returns values; the graph is asked for -------------------------------------------------------------


def _free_aperture_model(**parts) -> Fluorescence3D:
    aperture = Free(1.05, bounds=(1.0, 1.3))
    return Fluorescence3D(_detection(0.52, aperture), index_of_refraction_media=1.33, **parts)


def test_reconstruct_returns_values_without_a_graph():
    model = _free_aperture_model()
    stack = _stack().abs()
    plain = model.reconstruct(stack, pixel_size=PIXEL_SIZE)
    assert not plain.estimate.requires_grad
    assert plain.estimate.numpy().shape == (1, *SHAPE)
    assert "differentiable" not in inspect.signature(ReconstructionModel.reconstruct).parameters


# --- an assigned object prior or noise ------------------------------------------------------------------------------


def test_an_assigned_object_prior_or_noise_leaves_the_model_as_one_built_with_it():
    model = _fluorescence_3d().to(dtype=torch.float64)
    assert model.defaulted == ("object_prior", "noise")
    model.object_prior = TotalVariation(weight=1e-3)
    assert isinstance(model.object_prior, TotalVariation)
    assert model.defaulted == ("noise",)
    assert isinstance(model.solver, FISTA)  # the default solver, chosen again: no closed form for total variation
    model.object_prior = Tikhonov(1e-2)
    assert isinstance(model.solver, Wiener)
    model.noise = Gaussian(0.05)
    assert model.defaulted == ()
    placed = [*model.object_prior.buffers(), *model.noise.buffers()]
    assert placed
    assert all(tensor.dtype == torch.float64 for tensor in placed)  # moved to the model, not left in float32
    built = _fluorescence_3d(object_prior=Tikhonov(1e-2), noise=Gaussian(0.05)).to(dtype=torch.float64)
    stack = _stack().abs()
    expected = built.reconstruct(stack, pixel_size=PIXEL_SIZE).estimate
    assert torch.equal(model.reconstruct(stack, pixel_size=PIXEL_SIZE).estimate, expected)


def test_a_default_prior_follows_an_assigned_noise_and_a_given_solver_stays():
    model = _fluorescence_3d()
    model.noise = Poisson()
    assert isinstance(model.object_prior, NoPrior)  # as Fluorescence3D(noise=Poisson()) is built
    assert isinstance(model.solver, RichardsonLucy)
    assert model.defaulted == ("object_prior",)
    model.noise = Gaussian(0.05)
    assert isinstance(model.object_prior, Tikhonov)
    assert isinstance(model.solver, Wiener)
    kept = _fluorescence_3d(solver=FISTA())
    kept.object_prior = Tikhonov(1e-2)
    kept.noise = Gaussian(0.05)
    assert isinstance(kept.solver, FISTA)


def test_a_refused_assignment_of_a_part_changes_nothing():
    model = _fluorescence_3d(object_prior=Tikhonov(1e-2))
    prior, noise, solver = model.object_prior, model.noise, model.solver
    with pytest.raises(ParameterError, match="Fluorescence3D noise must be") as error:
        model.noise = Tikhonov()  # ty: ignore[invalid-assignment]
    assert error.value.path == "noise"
    with pytest.raises(ParameterError, match="Fluorescence3D object_prior must be"):
        model.object_prior = None  # ty: ignore[invalid-assignment]
    with pytest.raises(IncompatibleModelError, match=r"object_prior=Tikhonov and noise=Poisson.*NoPrior\(\)"):
        model.noise = Poisson()  # the prior was given, so it stays, and no default solver runs the pair
    assert (model.object_prior, model.noise, model.solver) == (prior, noise, solver)
    assert model.defaulted == ("noise",)


def test_assigning_a_prior_or_a_noise_drops_the_stored_config():
    model = ReconstructionModel.from_config(V4Config(**_free_aperture_raw()))
    model.noise = Gaussian(0.2)
    assert model.config is None
    assert _described(model)["parts"]["noise"] == {"gaussian": {"sigma": 0.2}}
    model = ReconstructionModel.from_config(V4Config(**_free_aperture_raw()))
    model.object_prior = Tikhonov(0.5)
    assert model.config is None
    assert _described(model)["parts"]["object_prior"] == {"tikhonov": {"weight": 0.5, "nonnegative": False}}


# --- parts no default solver runs are refused when the model is built ------------------------------------------------


class Hasty(ReconstructionModel):
    """A modality whose default solver does not take its default parts."""

    @override
    def default_solver(self) -> RichardsonLucy:
        return RichardsonLucy()


def test_parts_whose_default_solver_cannot_run_are_refused_at_construction():
    with_prior = r"Fluorescence3D: no solver runs on WidefieldFluorescence3D with object_prior={} and noise=Poisson"
    fix = r".*Pass object_prior=NoPrior\(\) with this noise.*noise=Gaussian\(sigma=\.\.\.\)"
    with pytest.raises(IncompatibleModelError, match=with_prior.format("TotalVariation") + fix):
        _fluorescence_3d(noise=Poisson(), object_prior=TotalVariation(weight=1e-3))
    with pytest.raises(IncompatibleModelError, match=with_prior.format("Tikhonov") + fix):
        _fluorescence_3d(noise=Poisson(), object_prior=Tikhonov())
    signed = r"ThickPhase3D does not declare preserves_nonnegativity\. Pass noise=Gaussian\(sigma=\.\.\.\)"
    with pytest.raises(IncompatibleModelError, match=signed):
        _phase_3d(noise=Poisson())
    with pytest.raises(IncompatibleModelError, match="set preserves_nonnegativity = True on its class"):
        ReconstructionModel(Flat(), noise=PoissonGaussian(sigma=2.0))
    with pytest.raises(IncompatibleModelError, match=r"the default solver RichardsonLucy cannot run on Flat.*solver="):
        Hasty(Flat())
    assert isinstance(Hasty(Flat(), solver=Wiener()).solver, Wiener)
    with pytest.raises(IncompatibleModelError, match=r"Fluorescence3D solver: FISTA.*noise=Poisson.*solver = None"):
        _fluorescence_3d(noise=Poisson(), object_prior=Tikhonov(), solver=FISTA())


# --- per-tile values the model holds --------------------------------------------------------------------------------


def test_reconstruct_tiles_solves_field_tiles_with_the_models_tile_parameters():
    model = _free_aperture_model(field_tiles=FieldTiles(size_yx=(16, 16), overlap_yx=(4, 4)))
    plan = model.field_tile_plan(SHAPE, pixel_size=PIXEL_SIZE)
    count = len(plan.regions)
    assert count == 4
    stack = _stack().abs()[None]
    nominal, explicit, held = (torch.zeros(1, *SHAPE) for _ in range(3))
    model.reconstruct_tiles([(stack, nominal)], pixel_size=PIXEL_SIZE, plan=plan)
    table = {APERTURE: torch.linspace(1.0, 1.3, count)}
    model.reconstruct_tiles([(stack, explicit)], pixel_size=PIXEL_SIZE, plan=plan, parameters=table)
    assert not torch.allclose(explicit, nominal)
    model.tile_parameters = {APERTURE: table[APERTURE].double()}
    assert model.tile_parameters is not None
    assert model.tile_parameters[APERTURE].dtype == torch.float32  # moved to the quantity's dtype
    model.reconstruct_tiles([(stack, held)], pixel_size=PIXEL_SIZE, plan=plan)
    assert torch.equal(held, explicit)
    whole = TilePlan(shape=SHAPE, tile=SHAPE)
    with pytest.raises(ParameterError, match=r"tile_parameters hold \[4\] tiles, but the plan has 1") as error:
        model.reconstruct_tiles([(stack, held)], pixel_size=PIXEL_SIZE, plan=whole)
    assert error.value.path == "plan"
    model.tile_parameters = None
    model.reconstruct_tiles([(stack, held)], pixel_size=PIXEL_SIZE, plan=plan)
    assert torch.equal(held, nominal)
    model.to(dtype=torch.float64).tile_parameters = table
    in_double = model.tile_parameters
    assert in_double is not None
    assert in_double[APERTURE].dtype == torch.float64
    in_single = model.to(dtype=torch.float32).tile_parameters  # moved with the model
    assert in_single is not None
    assert in_single[APERTURE].dtype == torch.float32


def test_a_saved_model_keeps_its_field_tile_calibration(tmp_path):
    tiles = FieldTiles(size_yx=(16, 16), overlap_yx=(4, 4))
    parts = {"noise": Gaussian(10.0), "object_prior": Tikhonov(1e-3), "search": [GridSearch(points=5)]}
    model = _free_aperture_model(field_tiles=tiles, **parts)
    plan = model.field_tile_plan(SHAPE, pixel_size=PIXEL_SIZE)
    stack = _stack().abs()[None]
    model.tile_parameters = dict(model.calibrate_tiles(stack, pixel_size=PIXEL_SIZE, plan=plan).parameters)
    loaded = ReconstructionModel.load(model.save(tmp_path / "tiled"))
    assert model.tile_parameters is not None
    assert loaded.tile_parameters is not None
    assert loaded.tile_parameters.keys() == model.tile_parameters.keys()
    assert torch.equal(loaded.tile_parameters[APERTURE], model.tile_parameters[APERTURE])
    ours, theirs = torch.zeros(1, *SHAPE), torch.zeros(1, *SHAPE)
    model.reconstruct_tiles([(stack, ours)], pixel_size=PIXEL_SIZE, plan=plan)
    loaded.reconstruct_tiles([(stack, theirs)], pixel_size=PIXEL_SIZE, plan=plan)
    assert torch.equal(ours, theirs)
    model.tile_parameters = None
    assert ReconstructionModel.load(model.save(tmp_path / "tiled", overwrite=True)).tile_parameters is None


@pytest.mark.parametrize(
    ("volumes", "path", "match"),
    [
        ((torch.zeros(1, *SHAPE), torch.zeros(1, *SHAPE)), "volumes.0", r"for one volume pass \[\(source, sink\)\]"),
        ([], "volumes", "non-empty list"),
        ((pair for pair in [()]), "volumes", "non-empty list of .* got generator"),
        ([(torch.zeros(1, *SHAPE),)], "volumes.0", r"volumes\.0 is a tuple, not a \(source, sink\) pair"),
    ],
)
def test_reconstruct_tiles_takes_a_list_of_source_sink_pairs(volumes: Any, path: str, match: str):
    with pytest.raises(ParameterError, match=match) as error:
        _fluorescence_3d().reconstruct_tiles(volumes, pixel_size=PIXEL_SIZE, plan=TilePlan(shape=SHAPE, tile=SHAPE))
    assert error.value.path == path


@pytest.mark.parametrize(
    ("table", "path"),
    [
        ({"nope": torch.zeros(4)}, "tile_parameters.nope"),
        ({APERTURE: 3}, f"tile_parameters.{APERTURE}"),
        ({APERTURE: torch.zeros(4, 2)}, f"tile_parameters.{APERTURE}"),
        ({APERTURE: torch.tensor(1.2)}, f"tile_parameters.{APERTURE}"),
        ([1.2], "tile_parameters"),
    ],
)
def test_tile_parameters_are_checked_when_they_are_assigned(table: Any, path: str):
    model = _free_aperture_model()
    with pytest.raises(ParameterError, match="tile_parameters") as error:
        model.tile_parameters = table
    assert error.value.path == path
    assert model.tile_parameters is None


# --- what a calibration would run is checked, and what biases it is said ----------------------------------------------


def test_check_judges_the_score_only_when_asked_about_a_calibration():
    model = _free_aperture_model(object_prior=TotalVariation(weight=1e-3))
    assert model.score is None
    model.check(SHAPE, pixel_size=PIXEL_SIZE)  # a reconstruction needs no score
    assert model.reconstruct(_stack().abs(), pixel_size=PIXEL_SIZE).finite
    with pytest.raises(IncompatibleModelError, match="Evidence"):  # what a calibration would build
        model.check(SHAPE, pixel_size=PIXEL_SIZE, calibration=True)
    model.score = GCV()
    model.check(SHAPE, pixel_size=PIXEL_SIZE)
    with pytest.raises(IncompatibleModelError, match="GCV"):
        model.check(SHAPE, pixel_size=PIXEL_SIZE, calibration=True)
    _free_aperture_model().check(SHAPE, pixel_size=PIXEL_SIZE, calibration=True)  # no warning: nothing calibrates
    fixed = _fluorescence_3d()
    fixed.check(SHAPE, pixel_size=PIXEL_SIZE)
    with pytest.raises(ParameterError, match=r"Fluorescence3D\.check: the model has no free quantity"):
        fixed.check(SHAPE, pixel_size=PIXEL_SIZE, calibration=True)
    with pytest.raises(ParameterError, match=r"Fluorescence3D\.check calibration must be") as error:
        fixed.check(SHAPE, pixel_size=PIXEL_SIZE, calibration="yes")  # ty: ignore[invalid-argument-type]
    assert error.value.path == "calibration"


def test_calibrating_a_likelihood_on_a_padded_stack_warns_that_the_estimate_is_biased():
    parts = {"object_prior": Tikhonov(1e-2), "noise": Gaussian(30.0), "search": [GridSearch(points=3)]}
    model = _free_aperture_model(z_padding=2, **parts)
    data = _stack().abs()
    biased = r"z_padding is 2.*the estimate is biased.*model\.z_padding = 0"
    with pytest.warns(CalibrationWarning, match=biased) as caught:
        model.calibrate(data, pixel_size=PIXEL_SIZE)
    assert len(caught) == 1
    assert caught[0].filename == __file__
    model.score = GCV()  # no likelihood: warnings are errors, and none is raised
    model.calibrate(data, pixel_size=PIXEL_SIZE)
    raw = _free_aperture_raw()
    raw["solver"]["padding"] = {"z": 2}
    configured = ReconstructionModel.from_config(V4Config(**raw))
    with pytest.warns(CalibrationWarning, match=r"solver\.padding\.z is 2"):
        configured.calibrate(data, pixel_size=PIXEL_SIZE)


def test_a_calibration_store_given_as_the_prior_centres_it_on_the_stored_values():
    data = _stack().abs()
    parts = {"object_prior": Tikhonov(1e-2), "noise": Gaussian(30.0), "search": [GridSearch(points=3)]}
    store = _free_aperture_model(**parts).to_tf_store()
    from_store = _free_aperture_model(**parts).calibrate(data, pixel_size=PIXEL_SIZE, prior=store)
    model = _free_aperture_model(**parts)
    built = model.calibrate(data, pixel_size=PIXEL_SIZE, prior=store.to_prior(free_specs(model.imaging_model)))
    assert from_store.parameters[APERTURE].item() == built.parameters[APERTURE].item()
    plan = TilePlan(shape=SHAPE, tile=SHAPE)
    tiled = _free_aperture_model(**parts).calibrate_tiles(data, pixel_size=PIXEL_SIZE, plan=plan, prior=store)
    assert tiled.parameters[APERTURE].shape == (1,)


# --- the data arguments of to_config --------------------------------------------------------------------------------


def test_to_config_takes_one_channel_name_as_a_list_of_one():
    assert _described(_phase_3d(), channels="BF")["data"] == {"input_channel_names": ["BF"]}
    configured = ReconstructionModel.from_config(_phase_raw()).to_config(channels="BF")
    assert isinstance(configured, V4Config)
    assert configured.data.input_channel_names == ["BF"]


# --- a class says what its prepare reads -----------------------------------------------------------------------------


def test_a_class_that_defines_prepare_must_state_its_scope_and_the_value_is_checked():
    with pytest.raises(
        ParameterError, match=r"Unstated must set prepare_scope beside prepare, to one of voxel"
    ) as error:

        class Unstated(ReconstructionModel):
            @override
            def prepare(self, data: Tensor, *, mean: Tensor | None = None) -> Tensor:
                return data - 1.0

    assert error.value.path == "prepare_scope"
    with pytest.raises(ParameterError, match=r"Inherited must set prepare_scope"):  # Phase3D's scope is not assumed

        class Inherited(Phase3D):
            @override
            def prepare(self, data: Tensor, *, mean: Tensor | None = None) -> Tensor:
                return data.double()

    with pytest.raises(ParameterError, match=r"Misspelt must set prepare_scope .*got 'stack-mean'") as error:

        class Misspelt(Phase3D):
            prepare_scope = "stack-mean"  # ty: ignore[invalid-assignment]

    assert error.value.path == "prepare_scope"

    class Plain(ReconstructionModel):  # no prepare of its own: the base value stands
        pass

    assert Plain.prepare_scope == "voxel"


# --- defaults follow the class on the configuration path too --------------------------------------------------------


class Sparse(Fluorescence3D):
    """A modality on a built-in system with its own default parts."""

    @override
    def default_object_prior(self) -> TotalVariation:
        return TotalVariation(weight=5e-3)

    @override
    def default_noise(self) -> Gaussian:
        return Gaussian(sigma=0.02)


def test_a_configuration_leaves_what_it_does_not_write_to_the_class():
    config = V4Config(**_fluorescence_raw())
    in_python = Sparse(_detection(0.52), index_of_refraction_media=1.33)
    configured = Sparse.from_config(config)
    for model in (in_python, configured, Sparse.for_system(_fluorescence_3d().system)):
        assert isinstance(model.object_prior, TotalVariation)
        assert model.noise.get_buffer("sigma").item() == pytest.approx(0.02)
        assert isinstance(model.solver, FISTA)  # the default solver follows the class's prior
        assert model.defaulted == ("object_prior", "noise")
    assert _same_values(configured, in_python)
    assert isinstance(ReconstructionModel.from_config(config).solver, Wiener)  # the built-in class is unchanged
    written = _fluorescence_raw()
    written["object"]["weight"] = 1e-2
    written["noise"] = {"gaussian": {"sigma": 0.05}}
    written["solver"] = {"reconstruction": "closed_form"}
    explicit = Sparse.from_config(V4Config(**written))
    assert isinstance(explicit.object_prior, Tikhonov)  # what the configuration writes is what is built
    assert explicit.noise.get_buffer("sigma").item() == pytest.approx(0.05)
    assert isinstance(explicit.solver, Wiener)
    assert explicit.defaulted == ()
    explicit.object_prior = Tikhonov(1e-3)
    assert isinstance(explicit.solver, Wiener)  # a solver the configuration wrote stays


def test_a_configuration_with_photon_noise_and_nothing_else_builds_what_python_builds():
    raw = _fluorescence_raw()
    raw["noise"] = {"poisson": {}}
    configured = ReconstructionModel.from_config(V4Config(**raw))
    in_python = _fluorescence_3d(noise=Poisson())
    for model in (configured, in_python):
        assert isinstance(model.object_prior, NoPrior)
        assert isinstance(model.solver, RichardsonLucy)


# --- a solver that is given is checked where it is given -------------------------------------------------------------


def test_an_assigned_solver_is_checked_against_the_parts_and_none_follows_them():
    model = _fluorescence_3d(object_prior=TotalVariation(weight=1e-3))
    solver = model.solver
    with pytest.raises(IncompatibleModelError, match=r"Fluorescence3D solver: Wiener.*object_prior=TotalVariation"):
        model.solver = Wiener()
    assert model.solver is solver  # a refused assignment changes nothing
    given = _fluorescence_3d(solver=Wiener())
    with pytest.raises(IncompatibleModelError, match=r"model\.solver is the Wiener.*TotalVariation") as error:
        given.object_prior = TotalVariation(weight=1e-3)
    assert "model.solver = None" in str(error.value)
    assert isinstance(given.object_prior, Tikhonov)
    assert isinstance(given.solver, Wiener)
    given.solver = None  # the default again, which follows the parts
    given.object_prior = TotalVariation(weight=1e-3)
    assert isinstance(given.solver, FISTA)
    with pytest.raises(IncompatibleModelError, match="no solver runs on"):
        given.noise = Poisson()  # the prior was given, and no default solver takes the pair
    assert isinstance(given.noise, Gaussian)


def test_parts_with_a_given_solver_change_together_through_the_default():
    model = _fluorescence_3d(solver=Wiener())
    with pytest.raises(IncompatibleModelError, match=r"model\.solver is the Wiener"):
        model.noise = Poisson()
    model.solver = None
    model.noise = Poisson()
    assert isinstance(model.object_prior, NoPrior)
    assert isinstance(model.solver, RichardsonLucy)
    model.solver = RichardsonLucy()
    with pytest.raises(IncompatibleModelError, match=r"model\.solver is the RichardsonLucy.*noise=Gaussian"):
        model.noise = Gaussian(0.05)
    assert isinstance(model.noise, Poisson)


# --- a sharpness score takes the model's solver and the system's cutoff ----------------------------------------------


def test_a_sharpness_score_built_without_solver_and_cutoff_is_filled_by_the_model():
    bare = MidbandPowerScore()
    assert bare.unfilled == ("solver", "cutoff")
    model = _free_aperture_model(object_prior=Tikhonov(1e-2), noise=Gaussian(30.0), score=bare)
    assert model.score is bare
    assert bare.unfilled == ()
    assert bare.solver is model.solver
    assert bare.cutoff == transverse_cutoff(model.system) == pytest.approx(2 * 1.05 / 0.52)
    model.solver = FISTA()
    assert bare.solver is model.solver  # the score follows the model's solver
    assigned = _free_aperture_model(object_prior=Tikhonov(1e-2), noise=Gaussian(30.0), search=[GridSearch(points=3)])
    assigned.score = MidbandPowerScore()
    written = _free_aperture_model(
        object_prior=Tikhonov(1e-2),
        noise=Gaussian(30.0),
        search=[GridSearch(points=3)],
        score=MidbandPowerScore(solver=Wiener(), cutoff=2 * 1.05 / 0.52),
    )
    data = _stack().abs()
    filled = assigned.calibrate(data, pixel_size=PIXEL_SIZE).parameters[APERTURE]
    by_hand = written.calibrate(data, pixel_size=PIXEL_SIZE).parameters[APERTURE]
    assert filled.item() == by_hand.item()
    kept = MidbandPowerScore(solver=FISTA(), cutoff=3.0)
    _free_aperture_model(score=kept)
    assert isinstance(kept.solver, FISTA)  # what the score was built with is kept
    assert kept.cutoff == 3.0
    with pytest.raises(ParameterError, match="cutoff") as error:
        ReconstructionModel(Flat(), score=MidbandPowerScore())  # a system without a detection has no cutoff
    assert error.value.path == "cutoff"


# --- a configuration as a mapping, and a configuration that is refused ----------------------------------------------


def test_from_config_takes_a_mapping_and_refuses_an_invalid_one_with_its_located_problems(tmp_path):
    model = ReconstructionModel.from_config(_phase_raw())
    assert type(model) is Phase3D
    assert model.config == V4Config(**_phase_raw())
    config = model.to_config(channels=["BF"], pixel_size=PIXEL_SIZE)
    assert isinstance(config, V4Config)
    recorded = config.model_dump(mode="json")
    assert _same_values(ReconstructionModel.from_config(recorded), model)  # what an output position records
    wrong = _phase_raw()
    wrong["imaging_system"]["detection"]["numerical_apperture"] = 1.2
    with pytest.raises(Rejected, match="numerical_apperture") as error:
        ReconstructionModel.from_config(wrong)
    assert any("numerical_apperture" in problem.message for problem in error.value.problems)
    path = tmp_path / "wrong.yml"
    path.write_text("version: 4\nimaging_system:\n  contrast: phase\n  numerical_apperture: 1.2\n")
    with pytest.raises(Rejected, match=r"wrong\.yml") as error:
        ReconstructionModel.from_config(path)
    assert error.value.problems
    with pytest.raises(ParameterError, match=r"ReconstructionModel\.from_config config must be"):
        ReconstructionModel.from_config(4)  # ty: ignore[invalid-argument-type]


# --- arguments that are refused by name ------------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["reconstruct", "calibrate"])
def test_prepared_must_be_a_bool(verb: str):
    model = _free_aperture_model(object_prior=Tikhonov(1e-2), noise=Gaussian(30.0))
    with pytest.raises(ParameterError, match=rf"Fluorescence3D\.{verb} prepared must be") as error:
        getattr(model, verb)(_stack().abs(), pixel_size=PIXEL_SIZE, prepared="no")
    assert error.value.path == "prepared"


def test_field_tiles_refuse_an_overlap_above_half_a_tile_by_its_own_name():
    with pytest.raises(ParameterError, match=r"FieldTiles overlap_yx\.y = 32 exceeds half of size_yx\.y = 16") as error:
        FieldTiles(size_yx=(16, 16), overlap_yx=(32, 32))
    assert error.value.path == "overlap_yx.y"
    assert "at most 8" in str(error.value)
    assert FieldTiles(size_yx=(16, 16)).overlap_yx == (8, 8)  # the default fits every tile size
    assert FieldTiles(size_yx=(128, 40)).overlap_yx == (32, 20)
    assert FieldTiles(size_yx=(64, 64)).overlap_yx == (32, 32)
    assert FieldTiles((64, 64), None, "cosine") == FieldTiles(size_yx=(64, 64), overlap_yx=(32, 32), blend="cosine")
    with pytest.raises(ParameterError) as error:
        FieldTiles(size_yx=(64, 16), overlap_yx=(32, 9))
    assert error.value.path == "overlap_yx.x"
    assert FieldTiles(size_yx=(64, 16), overlap_yx=(32, 8)).overlap_yx == (32, 8)


def test_a_model_lists_only_its_own_members_and_keeps_its_imaging_model():
    model = _phase_3d()
    listed = dir(model)
    for name in ("reconstruct", "calibrate", "to", "imaging_model", "solver", "z_padding"):
        assert name in listed
    for name in ("parameters", "state_dict", "cuda", "double", "eval", "register_buffer"):
        assert name not in listed
    misspelt = "solvr"
    with pytest.raises(AttributeError, match=r"'Phase3D' object has no attribute 'solvr'"):
        getattr(model, misspelt)
    tree = model.imaging_model
    model.noise = Gaussian(0.05)  # a part is replaced through the model, in the same tree
    assert model.imaging_model is tree
    assert copy.deepcopy(model).imaging_model is not tree


# --- the model and its grid are judged before a transfer function is computed ----------------------------------------


@pytest.mark.parametrize("method", ["reconstruct", "calibrate", "reconstruct_tiles", "calibrate_tiles", "tile_plan"])
def test_a_voxel_size_in_nanometres_is_refused_before_anything_is_computed(method: str):
    model = _free_aperture_model(object_prior=Tikhonov(1e-2), noise=Gaussian(30.0))
    stack, nanometres = _stack().abs(), (250.0, 100.0, 100.0)
    plan = TilePlan(shape=SHAPE, tile=SHAPE)
    calls = {
        "reconstruct": lambda: model.reconstruct(stack, pixel_size=nanometres),
        "calibrate": lambda: model.calibrate(stack, pixel_size=nanometres),
        "reconstruct_tiles": lambda: model.reconstruct_tiles(
            [(stack[None], torch.zeros(1, *SHAPE))], pixel_size=nanometres, plan=plan
        ),
        "calibrate_tiles": lambda: model.calibrate_tiles(stack, pixel_size=nanometres, plan=plan),
        "tile_plan": lambda: model.tile_plan(SHAPE, pixel_size=nanometres, tile=(8, 16, 16)),
    }
    with pytest.raises(IncompatibleModelError, match="pixel_size is in um"):
        calls[method]()


@pytest.mark.parametrize("method", ["reconstruct", "calibrate"])
def test_a_half_precision_model_is_refused_by_the_library_not_by_torch(method: str):
    model = _free_aperture_model(object_prior=Tikhonov(1e-2), noise=Gaussian(30.0))
    model.imaging_model.half()
    with pytest.raises(IncompatibleModelError, match=r"float32"):
        getattr(model, method)(_stack().abs(), pixel_size=PIXEL_SIZE)


# --- one model calibrates unpadded and reconstructs padded -----------------------------------------------------------


def test_z_padding_is_assignable_and_dtype_is_read_from_the_model():
    model = _phase_3d()
    stack = _stack()
    unpadded = model.reconstruct(stack, pixel_size=PIXEL_SIZE).estimate
    model.z_padding = 2
    assert model.z_padding == 2
    expected = _phase_3d(z_padding=2).reconstruct(stack, pixel_size=PIXEL_SIZE).estimate
    assert torch.equal(model.reconstruct(stack, pixel_size=PIXEL_SIZE).estimate, expected)
    model.z_padding = 0
    assert torch.equal(model.reconstruct(stack, pixel_size=PIXEL_SIZE).estimate, unpadded)
    with pytest.raises(ParameterError, match="Phase3D z_padding must be at least 0") as error:
        model.z_padding = -1
    assert error.value.path == "z_padding"
    assert model.z_padding == 0
    configured = ReconstructionModel.from_config(V4Config.model_validate(_phase_raw()))
    configured.z_padding = 3
    assert configured.config is None
    assert _described(configured)["settings"]["z_padding"] == 3
    assert model.dtype == torch.float32
    assert model.to("cpu", torch.float64).dtype == torch.float64  # the dtype second, as nn.Module.to takes it
    assert model.to(dtype=torch.float32).dtype == torch.float32


# --- a saved model is not written over -------------------------------------------------------------------------------


def test_save_refuses_an_existing_model_unless_told_to_overwrite(tmp_path):
    model = _phase_3d()
    folder = model.save(tmp_path / "phase")
    first = (folder / "model.json").read_text()
    model.z_padding = 2
    with pytest.raises(Rejected, match="exists; pass overwrite") as error:
        model.save(folder)
    assert error.value.problems[0].location == str(folder)
    assert (folder / "model.json").read_text() == first
    model.save(folder, overwrite=True)
    assert ReconstructionModel.load(folder).z_padding == 2


# --- a configuration that cannot be built is refused like one that is not valid --------------------------------------


def test_a_valid_configuration_that_cannot_be_built_here_is_rejected():
    raw = _phase_raw()
    raw["solver"] = {"device": "nowhere"}
    with pytest.raises(Rejected, match=r"solver\.device") as error:
        ReconstructionModel.from_config(raw)
    assert error.value.problems


# --- the score and the uncertainty are checked against each other where they are given -------------------------------


def test_free_quantities_of_the_solver_are_named_as_not_calibrated():
    solver = RichardsonLucy(background=Free(0.1, bounds=(0.0, 1.0)))
    model = _fluorescence_3d(solver=solver, noise=Poisson(), object_prior=NoPrior())
    message = r"no free quantity to calibrate.*\['background'\] of model\.solver are not calibrated"
    with pytest.raises(ParameterError, match=message):
        model.calibrate(_stack().abs(), pixel_size=PIXEL_SIZE)
    with pytest.raises(ParameterError, match=message):
        model.check(SHAPE, pixel_size=PIXEL_SIZE, calibration=True)


# --- a registered system of one's own and the class of its model ----------------------------------------------------


def test_a_registered_system_saves_and_loads_as_the_class_that_wraps_it(tmp_path):
    with temporarily("gain_listed", Gain, kind="system"), temporarily("flat", Flat, kind="system"):
        listed = GainModel(amplitude=2.0, object_prior=Tikhonov(weight=1e-8))
        loaded = ReconstructionModel.load(listed.save(tmp_path / "gain"))
        assert type(loaded) is GainModel
        assert loaded.offset == 100.0  # setup ran
        assert _described(loaded)["parts"]["system"] == {"gain_listed": {"amplitude": 2.0}}
        unlisted = ReconstructionModel.for_system(Flat())
        assert type(ReconstructionModel.load(unlisted.save(tmp_path / "flat"))) is ReconstructionModel
    with pytest.raises(ConfigurationError, match="Gain is not registered") as error:
        listed.save(tmp_path / "again")  # outside the block the name is gone
    assert error.value.field == "system"
    assert not (tmp_path / "again").exists()


@pytest.mark.parametrize("times", [-1, "all", [0, -2], [0.5], True])
def test_times_that_are_no_time_indices_are_refused_at_times(times: Any):
    for model in (_phase_3d(), ReconstructionModel.from_config(V4Config.model_validate(_phase_raw()))):
        with pytest.raises(ParameterError, match=r"Phase3D\.to_config times must be None") as error:
            model.to_config(channels=["BF"], pixel_size=PIXEL_SIZE, times=times)
        assert error.value.path == "times"
    assert _described(_phase_3d(), times=range(2))["data"] == {"time_indices": [0, 1]}


def test_an_output_that_is_no_path_is_refused_at_output(tmp_path):
    model = _phase_3d()
    with pytest.raises(ParameterError, match=r"Phase3D\.reconstruct_positions output must be") as error:
        model.reconstruct_positions(tmp_path / "in.zarr", output=None, channels=["BF"])  # ty: ignore[invalid-argument-type]
    assert error.value.path == "output"


def test_calibrate_position_has_a_dry_run():
    parameters = inspect.signature(ReconstructionModel.calibrate_position).parameters
    assert parameters["dry_run"].default is False
    assert parameters["dry_run"].kind is inspect.Parameter.KEYWORD_ONLY
