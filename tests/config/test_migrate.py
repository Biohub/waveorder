"""``waveorder.config.migrate``: the 3.x examples, every rule of the per-key table, and #592 version-4 files."""

import copy
from pathlib import Path

import pytest
import yaml

from waveorder.config import load
from waveorder.config.migrate import _BLOCK_KEYS, TABLE, migrate, render, table_help
from waveorder.config.schema import ReconstructionRichardsonLucy
from waveorder.models import ReconstructionModel

DATA = Path(__file__).parent / "data"
EXAMPLES = DATA / "3x"
MIGRATED = ["phase_3d", "phase_2d", "fluorescence_3d", "fluorescence_2d"]
REJECTED = {
    "birefringence_3d": ["birefringence"],
    "birefringence-and-phase_3d": ["birefringence"],
    "tile-stitch_phase_3d": ["tile"],
}


def example(name: str) -> dict:
    data = yaml.safe_load((EXAMPLES / f"{name}.yml").read_text())
    assert isinstance(data, dict)
    return data


def preview() -> dict:
    """Return the #592 fixture, a fresh copy per call."""
    return copy.deepcopy(example("../592/phase_3d_optimized"))


def phase(**transfer_function) -> dict:
    """A minimal 3.x phase file; everything left out takes its 3.x default."""
    return {"input_channel_names": ["BF"], "phase": {"transfer_function": transfer_function}}


def fluorescence(**apply_inverse) -> dict:
    return {"input_channel_names": ["GFP"], "fluorescence": {"apply_inverse": apply_inverse}}


def free_phase(**optimization) -> dict:
    raw = phase(numerical_aperture_illumination={"init": 0.8, "lr": 0.02})
    if optimization:
        raw["optimization"] = optimization
    return raw


def converted(raw: dict):
    """Migrate ``raw`` and build the model of the result, as ``wo config check`` does (on the CPU)."""
    result = migrate(raw)
    assert result.rejections == (), result.rejections
    assert result.config is not None
    checked = copy.deepcopy(result.config)
    checked["solver"]["device"] = None  # the device is checked where the run executes
    ReconstructionModel.from_config(load(checked))
    return result


def rejected_keys(raw: dict) -> list[str]:
    result = migrate(raw)
    assert result.config is None
    return [rejection.key for rejection in result.rejections]


# --- the 3.x examples --------------------------------------------------------------------------------------------


def test_fixtures_name_their_source_commit():
    for path in EXAMPLES.glob("*.yml"):
        assert "4ea8027:docs/examples/cli/configs/" in path.read_text().splitlines()[0]


@pytest.mark.parametrize("name", MIGRATED)
def test_examples_migrate_and_validate(name):
    result = converted(example(name))
    assert result.dialect == "3.x"
    config = load(result.config)
    assert config.imaging_system.contrast == name.split("_")[0]
    assert config.object.dimension == int(name[-2])
    assert config.imaging_system.detection.edge == "sigmoid"
    text = render(result, source=f"{name}.yml")
    assert text.startswith(f"# WaveOrder 4 configuration migrated by wo config migrate from waveorder 3.x: {name}.yml")
    assert "# - data.pixel_size is used as written, as in 3.x: wo rec warns where it differs" in text
    assert yaml.safe_load(text) == result.config
    load(yaml.safe_load(text))


@pytest.mark.parametrize(("name", "keys"), REJECTED.items())
def test_examples_rejected_with_their_keys(name, keys):
    assert rejected_keys(example(name)) == keys


def test_phase_3d_example_in_full():
    assert converted(example("phase_3d")).config == {
        "version": 4,
        "data": {
            "input_channel_names": ["Brightfield"],
            "time_indices": "all",
            "pixel_size": {"z": 0.25, "y": 0.1, "x": 0.1},
        },
        "imaging_system": {
            "contrast": "phase",
            "index_of_refraction_media": 1.3,
            "detection": {"numerical_aperture": 1.2, "z_focus_offset": 0, "edge": "sigmoid"},
            "illumination": {
                "wavelength": 0.532,
                "numerical_aperture": 0.9,
                "tilt": {"zenith": 0.0, "azimuth": 0.0},
                "invert_phase_contrast": False,
            },
        },
        "object": {"dimension": 3, "prior": "tikhonov", "weight": 0.001},
        "noise": {"gaussian": {"sigma": 1.0}},
        "solver": {
            "padding": {"z": 0},
            "reconstruction": "closed_form",
            "apodization_rolloff": 0.0,
            "device": None,
        },
    }


def test_rejection_messages():
    (biref,) = migrate(example("birefringence_3d")).rejections
    assert biref.message == "birefringence is not in waveorder 4; use waveorder 3.x"
    (tile,) = migrate(example("tile-stitch_phase_3d")).rejections
    assert tile.message.startswith("tile-stitch (tile, blend, recon) is not in waveorder 4")
    assert "wo rec --tile" in tile.message


def test_3d_tikhonov_auto_regularization_becomes_the_rule_based_weight():
    result = converted(example("phase_3d_auto_regularization"))
    rule = {"rule": "otsu_cnr", "search_min": -6.0, "search_max": 2.0, "num_samples": 25, "search_scale": "relative"}
    assert result.config["object"] == {"dimension": 3, "prior": "tikhonov", "weight": {"auto": rule}}
    raw = example("phase_3d_auto_regularization")
    raw["phase"]["apply_inverse"]["auto_regularization"] = {"rule": "l_curve"}
    raw["phase"]["apply_inverse"]["regularization_strength"] = 0.5
    result = converted(raw)
    assert result.config["object"]["weight"] == {"auto": {"rule": "l_curve"}}
    assert "phase.apply_inverse.regularization_strength dropped: auto_regularization sets it" in result.notes


@pytest.mark.parametrize("name", ["report_path", "plot_path"])
def test_a_sweep_report_or_plot_path_is_kept(name):
    raw = example("phase_3d_auto_regularization")
    raw["phase"]["apply_inverse"]["auto_regularization"][name] = "sweep.out"
    assert converted(raw).config["object"]["weight"]["auto"][name] == "sweep.out"
    preview_raw = preview()
    del preview_raw["solver"]["calibration"]
    preview_raw["imaging_system"]["illumination"]["numerical_aperture"] = 0.9
    preview_raw["object"]["weight"] = {"auto": {"rule": "otsu_cnr", name: "sweep.out"}}
    assert converted(preview_raw).config["object"]["weight"] == {"auto": {"rule": "otsu_cnr", name: "sweep.out"}}


# --- defaults and conversions ------------------------------------------------------------------------------------


def test_left_out_keys_take_their_3x_defaults():
    config = converted(phase()).config
    assert config["data"]["pixel_size"] == {"z": 0.25, "y": 0.1, "x": 0.1}
    assert config["imaging_system"]["illumination"]["wavelength"] == 0.532
    assert config["imaging_system"]["illumination"]["numerical_aperture"] == 0.9
    assert config["imaging_system"]["detection"]["numerical_aperture"] == 1.2
    assert config["object"] == {"dimension": 3, "prior": "tikhonov", "weight": 0.001}
    load(config)


def test_left_out_channels_take_the_3x_default_and_are_refused():
    raw = phase()
    del raw["input_channel_names"]
    assert rejected_keys(raw) == ["input_channel_names"]


def test_anisotropic_pixel_size():
    config = converted(phase(yx_pixel_size={"y": 0.2, "x": 0.1}, z_pixel_size=0.5)).config
    assert config["data"]["pixel_size"] == {"z": 0.5, "y": 0.2, "x": 0.1}


def test_padding_device_time_and_rolloff():
    raw = phase(z_padding=4)
    raw |= {"device": "cuda", "time_indices": [0, 2]}
    raw["phase"]["apply_inverse"] = {"apodization_rolloff": 0.25}
    config = converted(raw).config
    assert config["solver"]["padding"] == {"z": 4}
    assert config["solver"]["device"] == "cuda"
    assert config["solver"]["apodization_rolloff"] == 0.25
    assert config["data"]["time_indices"] == [0, 2]


def test_edge_is_pinned_with_a_note():
    result = converted(phase())
    assert result.config["imaging_system"]["detection"]["edge"] == "sigmoid"
    assert any("edge: sigmoid keeps the 3.x pupil" in note for note in result.notes)


def test_tv_becomes_admm_with_a_note():
    raw = phase()
    raw["phase"]["apply_inverse"] = {"reconstruction_algorithm": "TV", "TV_iterations": 3, "TV_rho_strength": 0.01}
    result = converted(raw)
    assert result.config["object"]["prior"] == "tv"
    assert result.config["solver"]["reconstruction"] == {"admm": {"iterations": 3, "rho": 0.01}}
    assert any("isotropic total variation" in note for note in result.notes)


def test_tv_settings_under_tikhonov_are_dropped_with_a_note():
    raw = phase()
    raw["phase"]["apply_inverse"] = {"TV_iterations": 5, "TV_rho_strength": 0.001}
    result = converted(raw)
    assert "phase.apply_inverse.TV_iterations dropped: read only by TV" in result.notes
    assert not any("TV_rho_strength" in note for note in result.notes)  # at its default


def test_confocal_pinhole():
    raw = fluorescence()
    raw["fluorescence"]["transfer_function"] = {"confocal_pinhole_diameter": 0.5}
    detection = converted(raw).config["imaging_system"]["detection"]
    assert detection["pinhole"] == {"diameter_um": 0.5}
    assert detection["wavelength"] == 0.532


def test_fluorescence_tilt_is_dropped_with_a_note():
    raw = fluorescence()
    raw["fluorescence"]["transfer_function"] = {"tilt_angle_zenith": 0.1}
    result = converted(raw)
    assert "illumination" not in result.config["imaging_system"]
    assert "fluorescence.transfer_function.tilt_angle_zenith dropped: 3.x fluorescence did not read it" in result.notes


def test_richardson_lucy_pins_the_3x_start_and_check_cadence():
    rl = {"iterations": 40, "background": 2.0, "stopping_tolerance": 1e-3, "back_projector": "matched"}
    result = converted(fluorescence(reconstruction_algorithm="RLGC", rl=rl, regularization_strength=0.5))
    config = result.config
    assert config["noise"] == {"poisson": {}}
    assert config["object"] == {"dimension": 3, "prior": "none"}
    assert config["solver"]["reconstruction"] == {
        "richardson_lucy": {
            "iterations": 40,
            "background": 2.0,
            "back_projector": "matched",
            "gradient_consensus": True,
            "tolerance": 1e-3,
            "check_every": 1,
            "start": "ones",
        }
    }
    assert any("start: ones keeps the 3.x first estimate" in note for note in result.notes)
    assert "fluorescence.apply_inverse.regularization_strength dropped: RLGC does not read it" in result.notes
    reconstruction = load(config).solver.reconstruction
    assert isinstance(reconstruction, ReconstructionRichardsonLucy)
    assert reconstruction.richardson_lucy.start == "ones"


def test_richardson_lucy_defaults_when_the_block_is_left_out():
    config = converted(fluorescence(reconstruction_algorithm="RL")).config
    assert config["solver"]["reconstruction"]["richardson_lucy"] == {
        "iterations": 25,
        "background": 0.0,
        "back_projector": "matched",
        "gradient_consensus": False,
        "start": "ones",
    }
    load(config)


def test_zero_stopping_tolerance_is_dropped_with_a_note():
    result = converted(fluorescence(reconstruction_algorithm="RL", rl={"stopping_tolerance": 0}))
    assert "tolerance" not in result.config["solver"]["reconstruction"]["richardson_lucy"]
    assert any(note.startswith("fluorescence.apply_inverse.rl.stopping_tolerance: 0 dropped") for note in result.notes)


def test_rl_block_under_tikhonov_is_dropped_with_a_note():
    result = converted(fluorescence(rl={"iterations": 3}))
    assert result.config["solver"]["reconstruction"] == "closed_form"
    assert "fluorescence.apply_inverse.rl dropped: 3.x ignored it for Tikhonov" in result.notes


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ({**phase(), "reconstruction_dimension": 2}, None),
        ({"input_channel_names": ["BF"], "phase": {"apply_inverse": {"reconstruction_algorithm": "RL"}}}, "phase"),
    ],
)
def test_richardson_lucy_is_3d_fluorescence_only(raw, key):
    if key is None:
        raw = fluorescence(reconstruction_algorithm="RL") | {"reconstruction_dimension": 2}
        key = "fluorescence"
    assert rejected_keys(raw) == [f"{key}.apply_inverse.reconstruction_algorithm"]


def test_auto_regularization_where_3x_ignored_it_is_dropped_with_a_note():
    thin = phase()
    thin["reconstruction_dimension"] = 2
    thin["phase"]["apply_inverse"] = {"auto_regularization": {"rule": "l_curve"}}
    result = converted(thin)
    assert "phase.apply_inverse.auto_regularization dropped: 3.x ignored it (the reconstruction is 2D)" in result.notes
    tv = phase()
    tv["phase"]["apply_inverse"] = {"reconstruction_algorithm": "TV", "auto_regularization": {}}
    assert any("(reconstruction_algorithm is TV)" in note for note in converted(tv).notes)


def test_unknown_keys_are_rejected():
    raw = phase(pixel_size=0.1)
    raw["colour"] = "blue"
    assert rejected_keys(raw) == ["colour", "phase.transfer_function.pixel_size"]


def test_phase_and_fluorescence_together_are_rejected():
    assert rejected_keys(phase() | {"fluorescence": {}}) == ["fluorescence"]


def test_no_contrast_block_is_rejected():
    assert rejected_keys({"input_channel_names": ["BF"]}) == ["phase"]


# --- optimization ------------------------------------------------------------------------------------------------


def test_learning_rate_frees_a_value():
    result = converted(free_phase())
    config = result.config
    assert config["imaging_system"]["illumination"]["numerical_aperture"] == {
        "nominal": 0.8,
        "tolerance": "free",
        "bounds": [0.01, 1.3],
    }
    assert any("gets bounds [0.01, 1.3]" in note for note in result.notes)
    assert config["solver"]["calibration"] == {
        "score": "sharpness",
        "search": [{"gradient": {"max_iter": 10, "optimizer": "adam"}}],
        "advanced": {"learning_rates": {"illumination.numerical_aperture": 0.02}},
    }
    assert any("_optimized.yml" in note for note in result.notes)
    load(config)


def free_thin_phase(**apertures: object) -> dict:
    """The 2D phase case of the review: air, detection NA 0.55, illumination NA 0.4, ``apertures`` replacing them."""
    transfer_function: dict[str, object] = {"index_of_refraction_media": 1.0, "wavelength_illumination": 0.45}
    transfer_function |= {"numerical_aperture_detection": 0.55, "numerical_aperture_illumination": 0.4, **apertures}
    raw = phase(**transfer_function) | {"reconstruction_dimension": 2}
    raw["optimization"] = {"method": "adam", "max_iterations": 20}
    return raw


@pytest.mark.parametrize(
    ("apertures", "bounds"),
    [
        ({"numerical_aperture_illumination": {"init": 0.4, "lr": 0.01}}, {"illumination": [0.01, 0.55]}),
        ({"numerical_aperture_detection": {"init": 0.5, "lr": 0.01}}, {"detection": [0.4, 1.0]}),
        (
            {
                "numerical_aperture_illumination": {"init": 0.4, "lr": 0.01},
                "numerical_aperture_detection": {"init": 0.55, "lr": 0.01},
            },
            {"detection": [0.475, 1.0], "illumination": [0.01, 0.475]},
        ),
    ],
    ids=["illumination", "detection", "both"],
)
def test_a_free_2d_phase_aperture_is_bounded_by_the_other(apertures, bounds):
    system = converted(free_thin_phase(**apertures)).config["imaging_system"]  # builds: the system takes the pair
    assert {arm: system[arm]["numerical_aperture"]["bounds"] for arm in bounds} == bounds


def test_preview_free_2d_phase_aperture_is_bounded_by_the_detection():
    raw = preview()
    raw["object"]["dimension"] = 2
    raw["imaging_system"]["detection"]["numerical_aperture"] = 0.95
    aperture = converted(raw).config["imaging_system"]["illumination"]["numerical_aperture"]
    assert aperture["bounds"] == [0.01, 0.95]


def test_zero_learning_rate_is_a_plain_number():
    config = converted(phase(numerical_aperture_detection={"initial_value": 1.1, "learning_rate": 0})).config
    assert config["imaging_system"]["detection"]["numerical_aperture"] == 1.1
    assert "calibration" not in config["solver"]


def test_optimization_without_free_values_is_dropped_with_a_note():
    result = converted(phase() | {"optimization": {"method": "lbfgs"}})
    assert "calibration" not in result.config["solver"]
    assert any(note.startswith("optimization dropped") for note in result.notes)


@pytest.mark.parametrize(
    ("optimization", "stage"),
    [
        ({"method": "nadam", "max_iterations": 30}, {"gradient": {"max_iter": 30, "optimizer": "nadam"}}),
        ({"method": "lbfgs", "max_iterations": 5}, {"lbfgs": {"iterations": 5}}),
        ({"method": "nelder_mead", "convergence_tol": 1e-3}, {"nelder_mead": {"iterations": 10, "tolerance": 1e-3}}),
        ({"method": "nelder_mead"}, {"nelder_mead": {"iterations": 10, "tolerance": 1e-4}}),
        ({"method": "grid_search", "grid_points": 5}, {"grid": {"points": 5}}),
    ],
)
def test_each_3x_method_becomes_a_stage(optimization, stage):
    config = converted(free_phase(**optimization)).config
    assert config["solver"]["calibration"]["search"] == [stage]
    load(config)


def test_grid_search_spans_the_3x_grid():
    config = converted(free_phase(method="grid_search", grid_points=5)).config
    parameter = config["imaging_system"]["illumination"]["numerical_aperture"]
    assert parameter["bounds"] == pytest.approx([0.8 - 2 * 0.02, 0.8 + 2 * 0.02])


def free_tilt(*, lr: float, max_iterations: int, method: str = "adam") -> dict:
    """The 3.x free tilt of the OPS phenotyping case: air, detection NA 0.55, so brightfield needs |zenith| < 0.582."""
    raw = phase(
        index_of_refraction_media=1.0,
        numerical_aperture_detection=0.55,
        numerical_aperture_illumination=0.4,
        tilt_angle_zenith={"init": 0.0, "lr": lr},
        tilt_angle_azimuth={"init": 0.0, "lr": lr},
    )
    raw["optimization"] = {"method": method, "max_iterations": max_iterations}
    return raw


@pytest.mark.parametrize(
    ("raw", "bounds"),
    [
        (free_tilt(lr=0.01, max_iterations=20), [-0.2, 0.2]),  # adam moves it at most 20 * 0.01
        (free_tilt(lr=0.05, max_iterations=50), [-0.582, 0.582]),  # beyond asin(0.55) the light leaves detection
        (free_tilt(lr=0.01, max_iterations=20, method="lbfgs"), [-0.582, 0.582]),
    ],
)
def test_free_tilt_gets_the_range_3x_could_reach_inside_brightfield(raw, bounds):
    result = converted(raw)
    tilt = result.config["imaging_system"]["illumination"]["tilt"]
    assert tilt["zenith"] == {"nominal": 0.0, "tolerance": "free", "bounds": bounds}
    assert "bounds" not in tilt["azimuth"]
    assert any(note.startswith("imaging_system.illumination.tilt.zenith is free") for note in result.notes)
    ReconstructionModel.from_config(load(result.config))


def test_preview_free_tilt_gets_bounds():
    raw = preview()
    raw["imaging_system"]["illumination"]["tilt"]["zenith"] = {"nominal": 0.1, "tolerance": "free"}
    raw["solver"]["calibration"]["advanced"]["learning_rates"]["illumination.tilt.zenith"] = 0.002
    config = converted(raw).config
    assert config["imaging_system"]["illumination"]["tilt"]["zenith"]["bounds"] == [0.08, 0.12]  # 10 adam steps
    ReconstructionModel.from_config(load(config))


OFFSET_NOTE = "imaging_system.detection.z_focus_offset: -1 written as 0: 3.x and wo-4.0 ignore the offset of a 3D"


@pytest.mark.parametrize(("contrast", "channel"), [("phase", "BF"), ("fluorescence", "GFP")])
def test_3d_focus_offset_is_written_as_0_with_a_note(contrast, channel):
    raw = {"input_channel_names": [channel], contrast: {"transfer_function": {"z_focus_offset": -1}}}
    result = converted(raw)
    assert result.config["imaging_system"]["detection"]["z_focus_offset"] == 0
    assert any(note.startswith(OFFSET_NOTE) for note in result.notes)
    load(result.config)


def test_free_3d_focus_offset_is_not_optimized():
    result = converted(phase(z_focus_offset={"init": -1, "lr": 0.05}) | {"optimization": {"method": "adam"}})
    assert result.config["imaging_system"]["detection"]["z_focus_offset"] == 0
    assert "calibration" not in result.config["solver"]


def test_2d_focus_offset_is_kept():
    raw = phase(z_focus_offset=-1)
    raw["reconstruction_dimension"] = 2
    result = converted(raw)
    assert result.config["imaging_system"]["detection"]["z_focus_offset"] == -1
    assert not any(note.startswith(OFFSET_NOTE) for note in result.notes)


def test_preview_3d_focus_offset_is_written_as_0_with_a_note():
    raw = preview()
    raw["imaging_system"]["detection"]["z_focus_offset"] = {"nominal": -1, "tolerance": "free"}
    raw["solver"]["calibration"]["advanced"]["learning_rates"]["detection.z_focus_offset"] = 0.05
    result = converted(raw)
    config = result.config
    assert config["imaging_system"]["detection"]["z_focus_offset"] == 0
    assert config["solver"]["calibration"]["advanced"]["learning_rates"] == {"illumination.numerical_aperture": 0.01}
    assert any(note.startswith("imaging_system.detection.z_focus_offset: {'nominal': -1") for note in result.notes)
    load(config)


def test_preview_calibration_of_only_the_3d_focus_offset_is_dropped():
    raw = preview()
    raw["imaging_system"]["illumination"]["numerical_aperture"] = 0.9
    raw["imaging_system"]["detection"]["z_focus_offset"] = {"nominal": -1, "tolerance": "free"}
    raw["solver"]["calibration"]["advanced"]["learning_rates"] = {"detection.z_focus_offset": 0.05}
    result = converted(raw)
    assert "calibration" not in result.config["solver"]
    assert any(note.startswith("solver.calibration dropped: nothing else is free") for note in result.notes)
    ReconstructionModel.from_config(load(result.config))


def test_loss_is_carried():
    loss = {"type": "midband_power", "midband_fractions": [0.1, 0.3]}
    config = converted(free_phase(loss=loss)).config
    assert config["solver"]["calibration"]["advanced"]["loss"] == loss
    load(config)


def test_settings_at_their_3x_default_are_dropped_with_a_note():
    result = converted(free_phase(convergence_patience=5, grid_points=7, use_gradients=True))
    assert any(
        note.startswith("optimization.convergence_patience dropped: 3.x stopped after 5") for note in result.notes
    )
    assert "optimization.grid_points dropped: read only by grid_search" in result.notes
    assert "optimization.use_gradients dropped: it matches what adam does" in result.notes


@pytest.mark.parametrize(
    ("optimization", "key"),
    [
        ({"convergence_patience": 3}, "optimization.convergence_patience"),
        ({"convergence_tol": 1e-3}, "optimization.convergence_tol"),
        ({"grid_points": 9}, "optimization.grid_points"),
        ({"use_gradients": False}, "optimization.use_gradients"),
        ({"log_dir": "logs"}, "optimization.log_dir"),
        ({"method": "sgd"}, "optimization.method"),
        ({"patience": 3}, "optimization.patience"),
    ],
)
def test_unexecuted_settings_are_rejected(optimization, key):
    assert rejected_keys(free_phase(**optimization)) == [key]


# --- version-4 files ---------------------------------------------------------------------------------------------


def test_a_current_file_is_returned_unchanged():
    raw = copy.deepcopy(converted(phase()).config)
    result = migrate(raw)
    assert result.dialect == "4"
    assert result.config == raw


def test_an_unknown_version_is_rejected():
    assert rejected_keys({"version": 5}) == ["version"]


def test_preview_file_is_upgraded():
    raw = preview()
    result = converted(raw)
    assert result.dialect == "592"
    config = result.config
    assert list(config)[:4] == ["version", "data", "imaging_system", "object"]
    assert config["data"]["pixel_size"] == {"z": 0.25, "y": 0.1, "x": 0.1}
    assert config["imaging_system"]["detection"]["edge"] == "sigmoid"
    calibration = config["solver"]["calibration"]
    assert calibration["search"] == [{"gradient": {"max_iter": 10, "optimizer": "adam"}}]
    assert calibration["advanced"] == {
        "learning_rates": {"illumination.numerical_aperture": 0.01},
        "loss": {"type": "midband_power", "midband_fractions": [0.125, 0.25]},
    }
    assert "solver.calibration.advanced.convergence_patience dropped" in " ".join(result.notes)
    assert render(result, source="x.yml").startswith(
        "# WaveOrder 4 configuration migrated by wo config migrate from a #592"
    )
    load(config)


def test_preview_rl_and_scalar_pixel_size():
    raw = {
        "version": 4,
        "data": {"input_channel_names": ["GFP"], "pixel_size": {"yx": 0.2, "z": 0.5}},
        "imaging_system": {"contrast": "fluorescence", "detection": {"wavelength": 0.5}},
        "object": {"prior": "none"},
        "noise": {"poisson": {}},
        "solver": {"reconstruction": {"richardson_lucy": {"iterations": 5, "stopping_tolerance": 1e-2}}},
    }
    config = converted(raw).config
    assert config["data"]["pixel_size"] == {"z": 0.5, "y": 0.2, "x": 0.2}
    assert config["solver"]["reconstruction"]["richardson_lucy"] == {
        "iterations": 5,
        "tolerance": 1e-2,
        "check_every": 1,
        "start": "ones",
    }
    load(config)


def test_preview_rule_based_weight_is_kept():
    raw = preview()
    del raw["solver"]["calibration"]
    raw["imaging_system"]["illumination"]["numerical_aperture"] = 0.9
    raw["object"]["weight"] = {"auto": {"rule": "otsu_cnr", "report_path": None, "plot_path": None}}
    assert converted(raw).config["object"]["weight"] == {"auto": {"rule": "otsu_cnr"}}


def test_preview_weight_of_a_prior_without_one_is_dropped_with_a_note():
    raw = preview()
    del raw["solver"]["calibration"]
    raw["imaging_system"] = {"contrast": "fluorescence", "detection": {"wavelength": 0.5, "numerical_aperture": 1.2}}
    raw["object"] = {"prior": "none", "weight": 0.001}
    raw["noise"] = {"poisson": {}}
    raw["solver"]["reconstruction"] = {"richardson_lucy": {"iterations": 7, "stopping_tolerance": 1e-3}}
    result = converted(raw)
    assert result.config["object"] == {"prior": "none"}
    assert any(
        note.startswith("object.weight: 0.001 dropped: object.prior: none has no weight") for note in result.notes
    )


def test_preview_unexecuted_settings_without_an_optimizer():
    raw = preview()
    del raw["solver"]["calibration"]["advanced"]["optimizer"]
    result = converted(raw)
    assert "grid_points" not in result.config["solver"]["calibration"]["advanced"]
    raw["solver"]["calibration"]["advanced"]["convergence_patience"] = 2
    assert rejected_keys(raw) == ["solver.calibration.advanced.convergence_patience"]


# --- the table ---------------------------------------------------------------------------------------------------


def test_the_table_names_the_3x_keys_of_each_block():
    top = {"input_channel_names", "time_indices", "reconstruction_dimension", "device", "birefringence"}
    assert _BLOCK_KEYS[""] == top | {"phase", "fluorescence", "optimization"}
    assert _BLOCK_KEYS["phase"] == _BLOCK_KEYS["fluorescence"] == {"transfer_function", "apply_inverse"}
    phase, fluorescence = _BLOCK_KEYS["phase.transfer_function"], _BLOCK_KEYS["fluorescence.transfer_function"]
    illumination = {"wavelength_illumination", "numerical_aperture_illumination", "invert_phase_contrast"}
    assert phase - fluorescence == illumination
    assert fluorescence - phase == {"wavelength_emission", "confocal_pinhole_diameter"}
    assert _BLOCK_KEYS["fluorescence.apply_inverse"] - _BLOCK_KEYS["phase.apply_inverse"] == {"rl"}
    assert _BLOCK_KEYS["fluorescence.apply_inverse.rl"] == {
        "iterations",
        "background",
        "stopping_tolerance",
        "back_projector",
        "bp_alpha",
        "bp_beta",
    }
    assert len(_BLOCK_KEYS["optimization"]) == 8


def test_help_is_the_table():
    text = table_help()
    assert len(text.splitlines()) == len(TABLE)
    assert "<contrast>.transfer_function.z_pixel_size (0.25): data.pixel_size.z" in text
    assert "[" not in text
