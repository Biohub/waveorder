"""``calibrate_position``, ``load_calibration`` and ``to_tf_store`` of a model, on tiny simulated plates."""

import dataclasses
import math
import re
import warnings
from pathlib import Path

import pytest
import torch

from tests.models.calibration_data import (
    FIELD_TILED,
    NA,
    PIXEL_SIZE,
    SHAPE,
    TILED_NA,
    TILED_PIXEL_SIZE,
    TILED_SHAPE,
    TRUE_NA,
    WEIGHT,
    intensities,
    model_of,
    raw_config,
    simulate_phase,
    simulate_tiled,
    store_of,
    write_position,
)
from waveorder.errors import (
    ParameterError,
    PositionWarning,
    Problems,
    ReadError,
    Rejected,
    StoreWarning,
    WaveorderError,
    argument_names,
)
from waveorder.io import TfStore
from waveorder.models import Fluorescence3D, Phase3D
from waveorder.models._stores import grid_problems, search_inputs, system_record
from waveorder.nn import (
    FISTA,
    Detection,
    Free,
    Gaussian,
    GradientSearch,
    GridSearch,
    Illumination,
    ParameterPrior,
    Sampling,
    Tikhonov,
    TotalVariation,
    free_specs,
    load_parameters,
)
from waveorder.nn.parameters import shortest_decimal
from waveorder.nn.priors import Beads

DATA_592 = Path(__file__).parents[1] / "io" / "data"


@pytest.fixture(scope="module")
def phase_volume() -> torch.Tensor:
    """Contrast of the simulated object (what the model measures)."""
    return simulate_phase()


@pytest.fixture
def position(tmp_path, phase_volume) -> Path:
    """A position holding brightfield intensities, as a camera records them (not contrast)."""
    return write_position(tmp_path / "in.zarr", intensities(phase_volume))


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory, phase_volume):
    """One calibration of the phase model: the position, the model after it and the store."""
    folder = tmp_path_factory.mktemp("calibrated")
    where = write_position(folder / "in.zarr", intensities(phase_volume))
    model = model_of()
    store = model.calibrate_position(where, channels=["BF"])
    return where, model, store


def _value(model, name: str = NA) -> float:
    return float(dict(model.imaging_model.named_parameters())[name].detach())


# --- calibrate_position ---


def test_calibration_recovers_the_aperture_and_updates_the_model(calibrated):
    where, model, store = calibrated
    value = store.row()[NA]
    assert value == pytest.approx(TRUE_NA, abs=0.04)
    assert _value(model) == pytest.approx(value, rel=1e-6)
    assert store.tile_grid["shape_zyx"] == list(SHAPE)
    assert store.tile_grid["size_yx"] is None
    assert store.transfers == ()  # a calibration store holds parameters only
    provenance = store.provenance
    assert provenance["kind"] == "calibrated"
    assert provenance["position"] == str(where.resolve())
    assert provenance["position_name"] == "A/1/0"
    assert provenance["time_index"] == 0
    assert provenance["device"] == "cpu"
    assert provenance["torch_version"] == torch.__version__
    assert provenance["at_bound"] == [[]]
    assert provenance["score"] == "Evidence"
    assert (provenance["prior"], provenance["start"]) == (None, None)
    host, system, config = provenance["host"], provenance["imaging_system"], provenance["config"]
    assert isinstance(host, dict)
    assert isinstance(system, dict)
    assert isinstance(config, dict)
    assert host["host"]
    assert system["thick_phase_3d"]["illumination"]["illumination"]["numerical_aperture"] == "free"
    assert config["data"]["input_channel_names"] == ["BF"]
    assert config["data"]["time_indices"] == [0]
    assert config["data"]["pixel_size"] == pytest.approx(dict(zip("zyx", PIXEL_SIZE, strict=True)))  # in um, as written
    assert config["calibrated"][NA] == pytest.approx(value)  # the calibrated block; the written nominal stays
    assert len(provenance["imaging_system_sha256"]) == 64
    assert {"waveorder_version", "iohub_version", "created"} <= set(provenance)
    assert not {"calibration_key", "calibration_inputs", "config_hash"} & set(provenance)


def test_a_saved_store_gives_a_second_model_the_same_values(calibrated, tmp_path):
    _, model, store = calibrated
    path = store.save(tmp_path / "tf.zarr")
    second = model_of()
    assert _value(second) == pytest.approx(0.45)
    second.load_calibration(path)
    assert _value(second) == pytest.approx(_value(model), rel=1e-6)
    third = model_of()
    third.load_calibration(store)  # the store in memory, without a file
    assert _value(third) == pytest.approx(_value(model), rel=1e-6)
    assert third.tile_parameters is None


def test_a_quantity_that_ends_on_a_bound_warns(position):
    raw = raw_config()
    raw["object"]["weight"] = 1e-3  # this weight drives the aperture onto its bound
    with pytest.warns(StoreWarning) as caught:
        store = model_of(raw).calibrate_position(position, channels=["BF"])
    assert store is not None
    messages = [str(item.message) for item in caught]
    assert any(f"{NA} ended on a bound" in message for message in messages), messages
    assert store.provenance["at_bound"] == [[NA]]
    assert store.transfers == ()


def test_channel_and_time_problems_are_reported_together_and_another_voxel_size_is_warned_of(position, tmp_path):
    model = model_of()
    with pytest.raises(Rejected) as error, pytest.warns(PositionWarning, match=r"\(0\.5, 0\.15, 0\.15\) um, but"):
        model.calibrate_position(
            position,
            channels=["GFP"],
            time=3,
            pixel_size=(0.5, 0.15, 0.15),
            prior=tmp_path / "absent.zarr",
        )
    messages = [problem.message for problem in error.value.problems]
    assert len(messages) == 3
    assert any("list GFP, which" in message for message in messages)
    assert any("time is 3, but" in message and "has 1 time points" in message for message in messages)
    assert any(message == "does not exist" for message in messages)
    locations = [problem.location for problem in error.value.problems]
    assert locations.count(str(position.resolve())) == 2
    assert f"prior {tmp_path / 'absent.zarr'}" in locations
    assert _value(model) == pytest.approx(0.45)  # nothing was computed


def test_a_given_voxel_size_that_differs_is_used_with_a_warning(position):
    raw = raw_config()
    raw["solver"]["calibration"]["search"] = [{"grid": {"points": 3}}]
    with pytest.warns(PositionWarning, match="reconstructing at pixel_size"):
        store = model_of(raw).calibrate_position(position, channels=["BF"], pixel_size=(0.5, 0.15, 0.15))
    assert store is not None
    config = store.provenance["config"]
    assert isinstance(config, dict)
    assert config["data"]["pixel_size"] == {"z": 0.5, "y": 0.15, "x": 0.15}


def test_what_is_not_a_position_is_refused(tmp_path):
    (tmp_path / "notes").mkdir()
    with pytest.raises(Rejected, match="is not an OME-Zarr position"):
        model_of().calibrate_position(tmp_path / "notes", channels=["BF"])


def test_a_model_with_nothing_free_is_refused_naming_what_to_free(position):
    raw = raw_config()
    raw["imaging_system"]["illumination"]["numerical_aperture"] = 0.5
    with pytest.raises(Rejected, match=r"nothing to calibrate.*waveorder\.nn\.Free\(nominal, tolerance=t"):
        model_of(raw).calibrate_position(position, channels=["BF"])
    python = Phase3D(
        Illumination(wavelength=0.532, numerical_aperture=0.5),
        Detection(wavelength=0.532, numerical_aperture=0.9),
        index_of_refraction_media=1.33,
    )
    with pytest.raises(Rejected, match="nothing to calibrate"):
        python.calibrate_position(position, channels=["BF"])


def test_free_quantities_without_solver_calibration_calibrate_with_the_defaults(position):
    raw = raw_config()
    del raw["solver"]["calibration"]
    model = model_of(raw)
    assert (model.score, model.search) == (None, None)  # as a model built in Python without them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StoreWarning)  # the default search is not tuned to this tiny volume
        store = model.calibrate_position(position, channels=["BF"])
    assert store is not None
    assert store.provenance["score"] == "Evidence"
    assert 0.3 <= store.row()[NA] <= 0.7
    assert _value(model) == pytest.approx(store.row()[NA], rel=1e-6)


def test_a_model_built_in_python_calibrates_and_saves(position, tmp_path):
    def build() -> Phase3D:
        return Phase3D(
            Illumination(wavelength=0.532, numerical_aperture=Free(0.45, bounds=(0.3, 0.7))),
            Detection(wavelength=0.532, numerical_aperture=0.9),
            index_of_refraction_media=1.33,
            object_prior=Tikhonov(weight=4e-4),
            noise=Gaussian(sigma=1e-3),
            search=[GridSearch(points=17)],
        )

    model = build()
    store = model.calibrate_position(position, channels=["BF"])
    assert store is not None
    assert store.row()[NA] == pytest.approx(TRUE_NA, abs=0.04)
    assert _value(model) == pytest.approx(store.row()[NA], rel=1e-6)
    assert store.provenance["score"] == "Evidence"  # the default of a model built without a score
    config = store.provenance["config"]
    assert isinstance(config, dict)
    assert (config["class"], list(config["parts"]["system"])) == ("Phase3D", ["thick_phase_3d"])  # the description
    second = build()
    second.load_calibration(store.save(tmp_path / "tf.zarr"))
    assert _value(second) == pytest.approx(_value(model), rel=1e-6)


def test_a_model_with_its_own_voxel_size_calibrates_a_position_that_records_none(tmp_path, phase_volume):
    where = write_position(tmp_path / "unscaled.zarr", intensities(phase_volume), pixel_size=(1.0, 1.0, 1.0))
    model = Phase3D(
        Illumination(wavelength=0.532, numerical_aperture=Free(0.45, bounds=(0.3, 0.7))),
        Detection(wavelength=0.532, numerical_aperture=0.9),
        index_of_refraction_media=1.33,
        object_prior=Tikhonov(weight=4e-4),
        noise=Gaussian(sigma=1e-3),
        search=[GridSearch(points=17)],
        pixel_size=PIXEL_SIZE,
    )
    with pytest.warns(PositionWarning, match="records \\(1, 1, 1\\) um"):  # the stated size is used
        store = model.calibrate_position(where, channels=["BF"])
    assert store is not None
    assert store.row()[NA] == pytest.approx(TRUE_NA, abs=0.04)
    config = store.provenance["config"]
    assert isinstance(config, dict)
    assert config["data"]["pixel_size"] == pytest.approx(dict(zip("zyx", PIXEL_SIZE, strict=True)))
    centers = store.tile_grid["centers_yx"]
    assert isinstance(centers, list)
    assert centers[0] == pytest.approx([31 / 2 * 0.15, 31 / 2 * 0.15])


def test_prior_and_start_stores_with_a_beads_prior_are_refused(tmp_path):
    where = write_position(tmp_path / "beads.zarr", torch.rand(*SHAPE), channel="GFP")
    model = Fluorescence3D(
        Detection(wavelength=0.52, numerical_aperture=Free(0.8, tolerance=0.05, bounds=(0.6, 1.0))),
        index_of_refraction_media=1.33,
        object_prior=Beads(diameter=1.0),
        noise=Gaussian(sigma=0.01),
    )
    prior = store_of((TILED_NA,), (0.8,))
    with pytest.raises(Rejected, match="prior and start stores with a Beads object prior are not supported yet"):
        model.calibrate_position(where, channels=["GFP"], prior=prior, dry_run=True)


def test_a_bead_calibration_of_a_position_records_and_takes_an_estimated_noise_level(tmp_path):
    voxel = (0.25, 0.1, 0.1)

    def model(aperture: float | Free) -> Fluorescence3D:
        detection = Detection(wavelength=0.532, numerical_aperture=aperture, edge="sigmoid")
        return Fluorescence3D(
            detection,
            index_of_refraction_media=1.3,
            object_prior=Beads(0.3),
            noise=Gaussian("estimate"),
            search=[GradientSearch(iterations=40, learning_rate=0.01)],
            pixel_size=voxel,
        )

    sampling = Sampling(shape=(16, 32, 32), pixel_size=voxel)
    positions = torch.tensor([(8, 8, 8), (8, 8, 24), (8, 24, 16)], dtype=torch.float32)
    clean, _ = model(1.2).simulate(Beads(0.3).render(positions, torch.full((3,), 3000.0), sampling)[None], noise=False)
    noise = 2 * torch.randn(clean.shape, generator=torch.Generator().manual_seed(3))
    where = write_position(tmp_path / "beads.zarr", (clean + 100 + noise)[0], channel="GFP", pixel_size=voxel)
    bead = model(Free(1.15, bounds=(1.0, 1.3)))
    store = bead.calibrate_position(where, channels=["GFP"])
    assert store is not None
    assert store.parameter_names == ("system.detection.numerical_aperture", "noise.sigma")
    sigma = store.row()["noise.sigma"]
    assert sigma == pytest.approx(2, rel=0.2)  # the bead fit's, not the start's
    assert float(bead.imaging_model.get_parameter("noise.sigma").detach()) == pytest.approx(sigma, rel=1e-6)


# --- prior and start ---

SPECS = {
    "a": Free(1.0, tolerance=0.1, bounds=(0.0, 2.0)),
    "b": Free(0.0, bounds=(-1.0, 1.0)),
    "c": Free(5.0, tolerance=1.0),
}


def test_a_prior_store_moves_the_centres_and_keeps_the_tolerances():
    store = TfStore(parameter_names=("a", "b"), parameters=torch.tensor([[1.2, 0.3]], dtype=torch.float64))
    prior = store.to_prior(SPECS)
    assert isinstance(prior, ParameterPrior)
    assert prior.covered == ("a", "c")  # b is flat in the model: only its start moves
    at_centre = {"a": torch.tensor(1.2), "b": torch.tensor(0.9), "c": torch.tensor(5.0)}
    assert float(prior.nll(at_centre)) == 0.0
    assert float(prior.nll(at_centre | {"a": torch.tensor(1.3)})) == pytest.approx(0.5, rel=1e-5)
    assert prior.clamp({"a": torch.tensor(3.0)})["a"].item() == 2.0  # the bounds stay the model's


def test_a_prior_object_or_another_value_for_prior_and_start_is_refused_naming_the_store(position):
    model = model_of()
    prior = ParameterPrior(free_specs(model.imaging_model))
    with pytest.raises(Rejected) as error:
        model.calibrate_position(position, channels=["BF"], prior=prior, start={NA: 0.5})  # ty: ignore[invalid-argument-type]
    by_location = {problem.location: problem.message for problem in error.value.problems}
    assert set(by_location) == {"prior", "start"}
    assert by_location["prior"].startswith("must be a TfStore or the path of a tf.zarr store, got ParameterPrior")
    assert "model.calibrate" in by_location["prior"]
    assert "TfStore.to_prior(specs)" in by_location["prior"]
    assert by_location["start"] == "must be a TfStore or the path of a tf.zarr store, got dict"
    assert _value(model) == pytest.approx(0.45)  # nothing was computed


@pytest.mark.filterwarnings("ignore:sets no prior on")
def test_a_stores_prior_calibrates_in_memory_as_the_store_does_on_a_position(position, phase_volume):
    store = store_of(values=(0.6,))
    on_position = model_of().calibrate_position(position, channels=["BF"], prior=store)
    assert on_position is not None
    model = model_of()
    load_parameters(model.imaging_model, {NA: 0.6})  # the start a prior store also gives
    prior = store.to_prior(free_specs(model.imaging_model))
    result = model.calibrate(intensities(phase_volume)[None], pixel_size=PIXEL_SIZE, prior=prior)
    assert float(result.parameters[NA]) == pytest.approx(on_position.row()[NA], rel=1e-5)


def _corrupt(position: Path) -> None:
    for chunk in (position / "0").rglob("*"):
        if chunk.is_file() and not chunk.name.startswith(("z", ".")):
            chunk.write_bytes(b"not a chunk")


def test_a_volume_that_cannot_be_read_is_a_waveorder_error_naming_the_position_and_time(position):
    _corrupt(position)
    model = model_of()
    with pytest.raises(ReadError, match=rf"could not read time 0 of {re.escape(str(position.resolve()))}: ") as error:
        model.calibrate_position(position, channels=["BF"])
    assert isinstance(error.value, WaveorderError)
    assert _value(model) == pytest.approx(0.45)  # the model is as it was


@pytest.mark.filterwarnings("ignore:sets no prior on")
def test_a_dry_run_runs_every_check_reads_no_data_and_returns_none(position, tmp_path):
    model = model_of()
    prior = store_of(values=(0.6,)).save(tmp_path / "prior.zarr")
    assert model.calibrate_position(position, channels=["BF"], prior=prior, dry_run=True) is None
    assert _value(model) == pytest.approx(0.45)  # nothing was calibrated
    with pytest.raises(Rejected) as error:
        model.calibrate_position(
            position,
            channels=["GFP"],
            pixel_size=(0.4, 0.15, 0.15),
            prior=tmp_path / "absent.zarr",
            start=store_of(values=(0.9,)),
            dry_run=True,
        )
    messages = "\n".join(problem.message for problem in error.value.problems)
    for expected in ("list GFP, which", "does not exist"):
        assert expected in messages
    assert "outside its bounds" in messages  # the stores are checked against the model, as in the real call
    _corrupt(position)
    assert model.calibrate_position(position, channels=["BF"], dry_run=True) is None  # no data is read


def test_the_calibration_score_is_checked_where_something_calibrates(position):
    def build() -> Phase3D:
        return Phase3D(
            Illumination(wavelength=0.532, numerical_aperture=Free(0.45, bounds=(0.3, 0.7))),
            Detection(wavelength=0.532, numerical_aperture=0.9),
            index_of_refraction_media=1.33,
            object_prior=TotalVariation(1e-3),  # the default score, the evidence, needs a Gaussian prior
            noise=Gaussian(sigma=1e-3),
            solver=FISTA(),
        )

    with pytest.raises(Rejected, match="Evidence"):
        build().calibrate_position(position, channels=["BF"])
    store = build().to_tf_store(shapes=[SHAPE], pixel_size=PIXEL_SIZE)  # nothing calibrates: no score is checked
    assert store.row()[NA] == pytest.approx(0.45)
    loaded = build()
    loaded.load_calibration(store_of(values=(0.5,), provenance=system_record(loaded)))
    assert _value(loaded) == pytest.approx(0.5)


def _search(model, **stores):
    problems = Problems()
    prior, start = stores.get("prior"), stores.get("start")
    found = search_inputs(problems, model, prior=prior, start=start, tiles=1, plan_grid=None)
    problems.raise_if_any()
    return found


@pytest.mark.filterwarnings("ignore:sets no prior on")
def test_prior_and_start_stores_reach_the_search(position, tmp_path):
    path = store_of(values=(0.6,)).save(tmp_path / "prior.zarr")
    found = _search(model_of(), prior=path)
    assert found.prior is not None
    assert isinstance(found.prior, TfStore)  # the store itself: the calibration turns it into the prior
    assert found.start is not None
    assert found.start.row() == {NA: 0.6}
    assert found.prior_path == str(path.absolute())
    store = model_of().calibrate_position(position, channels=["BF"], prior=path)
    assert store is not None
    assert store.provenance["prior"] == str(path.absolute())

    # start only moves the start: the prior stays the model's own.
    started = _search(model_of(), start=store_of(values=(0.6,)))
    assert started.prior is None
    assert started.start is not None
    assert started.start.row() == {NA: 0.6}
    assert started.start_path is None  # a store that was never saved has no path

    model = model_of()
    with pytest.raises(Rejected, match="outside its bounds") as error:
        model.calibrate_position(position, channels=["BF"], start=store_of(values=(0.9,)))
    assert [problem.location for problem in error.value.problems] == ["start"]
    assert _value(model) == pytest.approx(0.45)


def test_a_store_naming_no_free_quantity_is_refused_and_ignored_names_warn():
    with pytest.raises(Rejected, match=r"holds object_prior.weight, none of which is free") as error:
        _search(model_of(), prior=store_of((WEIGHT,), (1e-3,)))
    assert [problem.location for problem in error.value.problems] == ["prior"]
    with pytest.warns(StoreWarning, match=r"object_prior\.weight is not free in the model; its stored value is"):
        found = _search(model_of(), start=store_of((NA, WEIGHT), (0.5, 1e-3)))
    assert found.start is not None
    assert found.start.row() == {NA: 0.5}


@pytest.mark.filterwarnings("ignore:sets no prior on")
def test_a_prior_on_a_bound_or_one_that_constrains_nothing_warns(tmp_path):
    with pytest.warns(StoreWarning, match=r"the prior of system.illumination.numerical_aperture is centred on 0.7, at"):
        _search(model_of(), prior=store_of(values=(0.7,)))
    flat = store_of().save(tmp_path / "flat.zarr")  # the aperture has no tolerance
    with pytest.warns(StoreWarning, match="sets no prior on") as record:
        found = _search(model_of(), prior=flat)
    assert found.prior is not None
    assert isinstance(found.prior.to_prior(free_specs(model_of().imaging_model)), ParameterPrior)
    flat_warnings = [item.message for item in record if "sets no prior on" in str(item.message)]
    assert [getattr(message, "location", None) for message in flat_warnings] == [f"prior {flat}"]


def test_locations_use_the_names_the_caller_maps_arguments_to(tmp_path):
    token = argument_names.set({"prior": "--prior", "store": "-t"})
    try:
        with pytest.raises(Rejected) as error:
            _search(model_of(), prior=tmp_path / "absent.zarr")
        assert [problem.location for problem in error.value.problems] == [f"--prior {tmp_path / 'absent.zarr'}"]
        path = store_of().save(tmp_path / "tf.zarr")
        with pytest.raises(Rejected, match="records no imaging_system_sha256; recalibrate") as error:
            model_of().load_calibration(path)
        assert [problem.location for problem in error.value.problems] == [f"-t {path}"]
    finally:
        argument_names.reset(token)


# --- field tiles ---


@pytest.fixture(scope="module")
def tiled(tmp_path_factory):
    """One calibration of the field-tiled model: the position, the model after it and the saved store's path."""
    folder = tmp_path_factory.mktemp("tiled")
    where = write_position(folder / "in.zarr", simulate_tiled(), channel="GFP", pixel_size=TILED_PIXEL_SIZE)
    model = model_of(FIELD_TILED)
    store = model.calibrate_position(where, channels=["GFP"])
    assert store is not None
    return where, model, store, store.save(folder / "tf.zarr")


def test_field_tiles_write_one_row_per_tile_and_set_tile_parameters(tiled):
    _, model, store, _ = tiled
    assert store.tiles == 2
    assert store.parameters.shape == (2, 1)
    assert store.tile_grid["size_yx"] == [32, 32]
    assert store.tile_grid["shape_zyx"] == list(TILED_SHAPE)
    centers, at_bound = store.tile_grid["centers_yx"], store.provenance["at_bound"]
    assert isinstance(centers, list)
    assert len(centers) == 2
    assert isinstance(at_bound, list)
    assert len(at_bound) == 2
    assert store.transfers == ()  # a field-tiled model has no single transfer function
    assert model.tile_parameters is not None
    per_tile = model.tile_parameters[TILED_NA].double().cpu()
    torch.testing.assert_close(per_tile, store.parameters[:, 0], rtol=1e-6, atol=0)
    assert _value(model, TILED_NA) == pytest.approx(1.15)  # the model's own values are not modified


def test_a_tiled_store_loads_as_tile_parameters_and_comes_back_as_a_store(tiled):
    _, model, store, path = tiled
    second = model_of(FIELD_TILED)
    second.load_calibration(path)
    assert second.tile_parameters is not None
    assert second.tile_parameters[TILED_NA].shape == (2,)
    assert second.tile_parameters[TILED_NA].dtype == dict(second.imaging_model.named_parameters())[TILED_NA].dtype
    torch.testing.assert_close(second.tile_parameters[TILED_NA], model.tile_parameters[TILED_NA])
    again = second.to_tf_store(shapes=[TILED_SHAPE], pixel_size=TILED_PIXEL_SIZE)
    assert again.provenance["kind"] == "parameters"
    torch.testing.assert_close(again.parameters, store.parameters)
    assert again.tile_grid == store.tile_grid
    assert grid_problems(again, store.tile_grid) == []
    with pytest.raises(Rejected, match="a model with field tiles takes one volume shape"):
        second.to_tf_store()


def test_a_tiled_store_starts_another_tiled_calibration(tiled):
    where, _, store, path = tiled
    again = model_of(FIELD_TILED).calibrate_position(where, channels=["GFP"], start=path)
    assert again is not None
    assert again.tiles == 2
    assert again.provenance["start"] == str(path.absolute())
    three = dataclasses.replace(
        store, parameters=store.parameters[[0, 0, 1]], tile_grid=store.tile_grid | {"centers_yx": [[0.0, 0.0]] * 3}
    )
    with pytest.raises(Rejected, match="holds 3 tiles, but the model calibrates 2"):
        model_of(FIELD_TILED).calibrate_position(where, channels=["GFP"], start=three)


def test_a_store_of_another_tile_grid_is_refused(tiled):
    *_, path = tiled
    raw = raw_config(FIELD_TILED)
    raw["imaging_system"]["field_dependence"]["tiled"]["size_yx"] = [16, 16]
    with pytest.raises(Rejected, match=r"another tile grid: size_yx \[32,32\] \(model: \[16,16\]\)"):
        model_of(raw).load_calibration(path)
    raw = raw_config(FIELD_TILED)
    del raw["imaging_system"]["field_dependence"]
    with pytest.raises(Rejected, match="holds 2 field tiles, but the model is shift-invariant") as error:
        model_of(raw).load_calibration(path)
    assert {problem.location for problem in error.value.problems} == {f"store {path}"}


def test_the_whole_tile_grid_must_match():
    grid = {"size_yx": [32, 32], "overlap_yx": [0, 0], "centers_yx": [[1.0, 2.0], [1.0, 5.0]], "shape_zyx": [8, 32, 64]}
    store = TfStore(parameter_names=(NA,), parameters=torch.ones(2, 1, dtype=torch.float64), tile_grid=dict(grid))
    assert grid_problems(store, grid) == []
    assert grid_problems(store, grid | {"centers_yx": [[1.0, 5.0], [1.0, 2.0]]}) == [
        "was made on another tile grid: tile centres [[1.0,2.0],[1.0,5.0]] (model: [[1.0,5.0],[1.0,2.0]])"
    ]
    assert "overlap_yx [0,0] (model: [4,4])" in grid_problems(store, grid | {"overlap_yx": [4, 4]})[0]
    assert grid_problems(store, None) == ["holds 2 field tiles, but the model is shift-invariant"]


def test_a_592_tiled_store_without_its_volume_shape_is_refused_for_recalibration():
    read = TfStore.load(DATA_592 / "tf_592_tiled.zarr")
    store = dataclasses.replace(read, parameter_names=(TILED_NA,), parameters=read.parameters[:, :1])
    with pytest.raises(Rejected) as error:
        model_of(FIELD_TILED).load_calibration(store)
    (problem,) = error.value.problems  # no imaging-system problem: the flag would not help
    assert "recorded no volume shape" in problem.message
    assert "calibrate_position" in problem.message
    assert "allow_system_mismatch" not in problem.message


# --- load_calibration: the store against the model ---


def _matching(**provenance) -> dict[str, object]:
    return {**system_record(model_of()), **provenance}


def test_a_store_of_this_system_loads_and_a_fixed_weight_only_warns():
    model = model_of()
    model.load_calibration(store_of(provenance=_matching()))
    assert _value(model) == pytest.approx(0.5)
    with pytest.warns(StoreWarning, match=r"holds object_prior.weight, which the model fixes; the model's values"):
        model.load_calibration(store_of((NA, WEIGHT), (0.6, 1e-3), provenance=_matching()))
    assert _value(model) == pytest.approx(0.6)
    assert float(torch.as_tensor(model.object_prior.weight).detach()) == pytest.approx(4e-4)


def test_a_store_with_other_parameter_names_is_refused():
    model = model_of()
    with pytest.raises(Rejected, match=re.escape(f"has no value for the free quantities {NA}")) as error:
        model.load_calibration(store_of((WEIGHT,), (1e-3,), provenance=_matching()))
    (problem,) = error.value.problems
    assert problem.location == "store"
    store = store_of((NA, "system.detection.numerical_aperture"), (0.5, 0.9), provenance=_matching())
    with pytest.raises(Rejected, match="fixes in its imaging system"):
        model.load_calibration(store)
    assert _value(model) == pytest.approx(0.45)  # a refused store changes nothing


def test_another_imaging_system_is_refused_naming_the_field(calibrated):
    *_, store = calibrated
    raw = raw_config()
    raw["imaging_system"]["detection"]["numerical_aperture"] = 0.8
    other = model_of(raw)
    changed = (
        r"another imaging_system \(imaging_system.thick_phase_3d.detection.detection.numerical_aperture: 0.9 -> 0.8\); "
        "recalibrate"
    )
    with pytest.raises(Rejected, match=changed):
        other.load_calibration(store)
    assert _value(other) == pytest.approx(0.45)  # refused: unchanged
    with pytest.raises(Rejected, match="records no imaging_system_sha256; recalibrate"):
        model_of().load_calibration(store_of())


def test_a_mismatch_shows_a_free_quantity_as_free_not_its_parameter_name(calibrated):
    *_, store = calibrated
    raw = raw_config()
    raw["imaging_system"]["illumination"]["numerical_aperture"] = 0.45
    fixed = model_of(raw)
    made_fixed = store_of((WEIGHT,), (1e-3,), provenance=system_record(fixed))
    with pytest.raises(Rejected) as error:
        model_of().load_calibration(dataclasses.replace(store, provenance=made_fixed.provenance))
    (problem,) = error.value.problems
    assert "imaging_system.thick_phase_3d.illumination.illumination.numerical_aperture: 0.45 -> free" in problem.message
    assert f'"{NA}"' not in problem.message
    # A store that does not cover the model is not also blamed on the system.
    with pytest.raises(Rejected) as error:
        fixed.load_calibration(store_of())
    (problem,) = error.value.problems
    assert NA in problem.message
    assert "recalibrate" not in problem.message


def test_a_592_store_is_refused():
    read = TfStore.load(DATA_592 / "tf_592.zarr")
    store = dataclasses.replace(  # one #592 value under the name this model frees
        read, parameter_names=(NA,), parameters=torch.full_like(read.parameters[:, :1], 0.5)
    )
    with pytest.raises(Rejected, match="written by #592, which records no imaging_system_sha256; recalibrate"):
        model_of().load_calibration(store)


def test_models_that_differ_only_in_a_free_quantitys_prior_share_the_system_record():
    raw = raw_config()
    aperture = {"nominal": 0.5, "tolerance": 0.1, "bounds": [0.3, 0.7]}
    raw["imaging_system"]["illumination"]["numerical_aperture"] = aperture
    assert system_record(model_of(raw)) == system_record(model_of())
    raw["imaging_system"]["index_of_refraction_media"] = 1.4
    assert system_record(model_of(raw))["imaging_system_sha256"] != system_record(model_of())["imaging_system_sha256"]


# --- to_tf_store ---


def test_to_tf_store_without_shapes_holds_the_nominal_values():
    store = model_of().to_tf_store()
    assert store.row() == {NA: 0.45}
    assert store.transfers == ()
    assert store.tile_grid == {}
    assert store.provenance["kind"] == "nominal"
    assert store.provenance["shapes_zyx"] == []
    assert store.provenance["imaging_system_sha256"] == system_record(model_of())["imaging_system_sha256"]


def test_to_tf_store_with_shapes_holds_the_transfer_at_each(calibrated, tmp_path):
    _, model, calibration = calibrated
    store = model.to_tf_store(shapes=[(4, 16, 16), SHAPE, (4, 16, 16)], pixel_size=PIXEL_SIZE)
    assert [item.shape for item in store.transfers] == [(4, 16, 16), SHAPE]
    assert store.provenance["kind"] == "parameters"  # the model carries calibrated values
    assert store.row() == calibration.row()
    assert store.tile_grid["shape_zyx"] == [4, 16, 16]
    transfer = store.transfers[0].transfer
    assert transfer is not None
    assert transfer.shape[-3:] == (4, 16, 9)
    read = TfStore.load(store.save(tmp_path / "tf.zarr"))
    assert sorted(item.shape for item in read.transfers) == sorted([(4, 16, 16), SHAPE])
    second = model_of()
    second.load_calibration(read)
    assert _value(second) == pytest.approx(_value(model), rel=1e-6)


def test_to_tf_store_refusals():
    model = model_of()
    unsized = Phase3D(
        Illumination(wavelength=0.532, numerical_aperture=Free(0.45, bounds=(0.3, 0.7))),
        Detection(wavelength=0.532, numerical_aperture=0.9),
        index_of_refraction_media=1.33,
    )
    with pytest.raises(ParameterError, match="has no voxel size") as error:
        unsized.to_tf_store(shapes=[(4, 16, 16)])
    assert error.value.path == "pixel_size"
    assert model.to_tf_store(shapes=[(4, 16, 16)]).tile_grid["shape_zyx"] == [4, 16, 16]  # data.pixel_size
    with pytest.raises(ParameterError, match="must be positive numbers in um"):
        model.to_tf_store(shapes=[(4, 16, 16)], pixel_size=(0.0, 0.1, 0.1))
    for shapes in ((8, 32, 32), [(8, 32)], [(8, 32, 0)], "8,32,32"):
        with pytest.raises(ParameterError, match=r"for one shape pass \[\(z, y, x\)\]") as refused:
            model.to_tf_store(shapes=shapes, pixel_size=PIXEL_SIZE)  # ty: ignore[invalid-argument-type]
        assert refused.value.path == "shapes"


# --- numbers ---


def test_float32_values_are_stored_as_their_shortest_decimal():
    single = float(torch.tensor(0.52, dtype=torch.float32))
    assert single != 0.52
    assert shortest_decimal(single, torch.float32) == 0.52
    assert shortest_decimal(0.1 + 0.2, torch.float32) == 0.1 + 0.2  # a float64 value is kept as it is
    assert math.isnan(shortest_decimal(math.nan, torch.float32))


def test_simulated_data_differs_with_the_aperture():
    assert not torch.allclose(simulate_phase(aperture=0.4), simulate_phase(aperture=0.6))
