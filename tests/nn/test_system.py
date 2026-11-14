"""Transfer-function physics, static shapes under calibration, and the auto cache."""

import re
from typing import Any

import pytest
import torch
from torch import nn
from torch.func import vmap

from tests.nn.conftest import FREE_NA, INDEX, WAVELENGTH, make_model, make_system
from waveorder import nn as public
from waveorder.errors import IncompatibleModelError, ParameterError, ShapeError
from waveorder.nn import (
    ADMM,
    FISTA,
    ConfocalFluorescence3D,
    Detection,
    FourierDiagonal,
    Free,
    Gaussian,
    Illumination,
    IlluminationTilt,
    ImagingModel,
    Loop,
    MidbandPowerScore,
    Poisson,
    RichardsonLucy,
    Sampling,
    ShiftInvariantSystem,
    ThickPhase3D,
    ThinPhase2D,
    Tikhonov,
    TotalVariation,
    WidefieldFluorescence3D,
    Wiener,
    ZernikeAberration,
    evaluate,
    free_specs,
)
from waveorder.nn.functional import defocused_intensity, half_spectrum_frequencies, hermitian_half
from waveorder.nn.optics import PupilGrid
from waveorder.nn.parameters import repr_parts
from waveorder.nn.systems.base import CacheMode
from waveorder.testing import check_operator, check_system

NA = "detection.numerical_aperture"


def _transfer(system, sampling):
    return evaluate(lambda m: m.transfer(sampling), system, {})


def _half(sampling):
    nz, ny, nx = sampling.shape
    return (nz, ny, nx // 2 + 1)


def test_aberration_free_otf_is_real_normalized_and_band_limited(sampling):
    system = make_system(1.0)
    assert system.real_transfer  # no pupil factors and a fixed zero focus offset
    H = system.transfer(sampling)[0, 0]  # one channel, one field, stored without an imaginary part
    assert H.shape == _half(sampling)
    assert H[0, 0, 0] == pytest.approx(1.0)
    grid = PupilGrid.from_sampling(sampling, device=H.device, dtype=H.dtype)
    radius = grid.radius[:, : H.shape[-1]]  # the half spectrum keeps the nonnegative x frequencies
    outside = radius > 1.05 * 2 * 1.0 / WAVELENGTH  # incoherent cut-off is 2 NA / wavelength
    assert outside.any()
    assert H.abs()[:, outside].max() < 1e-4


def test_a_declared_real_otf_drops_only_a_zero_imaginary_part(sampling):
    # A free focus offset makes the same optics complex; at offset 0 its imaginary part is rounding only.
    general = make_system(1.0, z_focus_offset=Free(0.0, bounds=(-1.0, 1.0)))
    assert not general.real_transfer
    with torch.no_grad():
        complex_storage = general.transfer(sampling)
    assert complex_storage.shape == (1, 1, *_half(sampling), 2)
    assert complex_storage[..., 1].abs().max() < 1e-6
    torch.testing.assert_close(make_system(1.0).transfer(sampling), complex_storage[..., 0])


def test_even_crops_with_aberrations_stay_hermitian():
    # Coarse even-sized grid with coma and defocus: the plain crop of the oversampled OTF is not Hermitian on the
    # Nyquist frequencies. The build keeps the half of its Hermitian part: the transform of real(ifftn(crop)).
    coarse = Sampling(shape=(6, 8, 10), pixel_size=(0.4, 0.15, 0.15))
    system = make_system(1.2, z_focus_offset=0.7, pupil_aberrations=ZernikeAberration({"coma_x": 0.1}))
    nx = coarse.shape[-1]
    pupil, kz, z = system._pupil_planes(coarse)  # noqa: SLF001
    psf = defocused_intensity(pupil, kz, z)
    crop = torch.fft.fftn(psf / psf.sum())
    for dim, size in zip((-3, -2, -1), coarse.shape, strict=True):
        full = crop.shape[dim]
        crop = crop.index_select(dim, torch.cat([torch.arange((size + 1) // 2), torch.arange(full - size // 2, full)]))
    plain = torch.view_as_real(crop[..., : nx // 2 + 1].contiguous())
    assert not torch.allclose(plain, hermitian_half(plain, nx), atol=1e-6), "must exercise the Nyquist frequencies"
    cropped = system._cropped_otf(coarse)  # noqa: SLF001
    torch.testing.assert_close(cropped, torch.view_as_real(torch.fft.rfftn(torch.fft.ifftn(crop).real)))
    raw = system.compute_transfer(coarse)
    torch.testing.assert_close(raw, hermitian_half(raw, nx))
    H = system.transfer(coarse)
    torch.testing.assert_close(H, hermitian_half(H, nx))


def test_coarse_pixels_upsample_and_still_return_the_data_shape():
    coarse = Sampling(shape=(8, 32, 32), pixel_size=(1.0, 0.3, 0.3))
    system = make_system(1.2)
    assert all(k > 1 for k in system.upsampling(coarse))
    assert system.transfer(coarse).shape == (1, 1, *_half(coarse))


def test_reachable_na_above_index_rejected():
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=Free(1.2, bounds=(1.0, 1.5)))
    with pytest.raises(ParameterError, match="index_of_refraction_media") as caught:
        WidefieldFluorescence3D(detection, index_of_refraction_media=INDEX)
    assert caught.value.path == "detection.numerical_aperture"


@pytest.mark.parametrize(
    ("detection", "index", "path", "fragment"),
    [
        (None, float("inf"), "index_of_refraction_media", "positive and finite"),
        (None, 0.0, "index_of_refraction_media", "positive and finite"),
    ],
)
def test_widefield_fluorescence_refuses_wrong_arguments_at_their_path(detection, index, path, fragment):
    detection = detection or Detection(wavelength=WAVELENGTH, numerical_aperture=0.9)
    with pytest.raises(ParameterError, match=fragment) as caught:
        WidefieldFluorescence3D(detection, index_of_refraction_media=index)
    assert caught.value.path == path


INDEX_PATH = "index_of_refraction_media"


@pytest.mark.parametrize(
    ("build", "path"),
    [
        (lambda free: Illumination(wavelength=free, numerical_aperture=0.5), "illumination.wavelength"),
        (lambda free: Detection(wavelength=free, numerical_aperture=0.5), "wavelength"),
        (lambda free: WidefieldFluorescence3D(make_system().detection, index_of_refraction_media=free), INDEX_PATH),
        (
            lambda free: ThickPhase3D(
                Illumination(wavelength=WAVELENGTH, numerical_aperture=0.5),
                Detection(numerical_aperture=0.9),
                index_of_refraction_media=free,
            ),
            INDEX_PATH,
        ),
    ],
)
def test_a_free_wavelength_or_refractive_index_is_refused_at_its_path(build, path):
    with pytest.raises(ParameterError, match="cannot be free: it sets the grid") as caught:
        build(Free(1.0, bounds=(0.5, 1.5)))
    assert caught.value.path == path


def test_the_kit_names_the_fix_when_given_a_system_where_it_takes_a_builder_or_an_operator():
    system = make_system(1.0)
    with pytest.raises(AssertionError, match=r"check_system takes a builder, .*; pass `lambda: system`"):
        check_system(system, Sampling(shape=(6, 8, 10), pixel_size=(0.4, 0.15, 0.15)))
    bound = r"check_operator takes a bound operator, .*; pass `system\.bind\(sampling\)`"
    with pytest.raises(AssertionError, match=bound):
        check_operator(system)  # ty: ignore[invalid-argument-type]


def test_free_parameter_names_follow_the_module_tree():
    model = make_model(
        FREE_NA,
        pupil_aberrations=ZernikeAberration({"z11": Free(0.0, tolerance=0.02), "coma_x": 0.01}),
    )
    assert list(free_specs(model)) == [name for name, _ in model.named_parameters()]
    assert list(free_specs(model)) == [
        "system.detection.numerical_aperture",
        "system.detection.pupil_aberrations.spherical",
    ]


def test_free_na_and_focus_get_gradients(sampling):
    system = make_system(FREE_NA, z_focus_offset=Free(0.5, tolerance=1.0, bounds=(-4.0, 4.0)))
    # |H| ignores a focus shift (it only adds a phase ramp along kz), so the loss uses the real part.
    system.transfer(sampling)[..., 0].sum().backward()
    for parameter in (system.detection.numerical_aperture, system.detection.z_focus_offset):
        assert parameter.grad is not None
        assert parameter.grad.abs() > 0


def test_forward_rejects_extra_dims_and_complex_input(sampling):
    system = make_system(1.0)
    with pytest.raises(ShapeError, match=r"object has shape .* batch with torch.func.vmap"):
        system(torch.zeros(2, *system.object_shape(sampling)), sampling)
    with pytest.raises(ShapeError, match="must be real"):
        system(torch.zeros(system.object_shape(sampling), dtype=torch.complex64), sampling)


def test_a_bound_operator_rejects_a_transfer_for_another_sampling(sampling):
    system = make_system(1.0)
    wrong = system.transfer(Sampling(shape=(8, 32, 32), pixel_size=sampling.pixel_size))
    with pytest.raises(ShapeError, match="transfer has shape"):
        FourierDiagonal(wrong, sampling)


def test_double_precision_follows_the_module(sampling):
    system = make_system(FREE_NA).double()
    assert system.transfer(sampling).dtype == torch.float64
    assert system(torch.zeros(system.object_shape(sampling), dtype=torch.float64), sampling).dtype == torch.float64


class TestAutoCache:
    @staticmethod
    def cached_system(numerical_aperture=FREE_NA):
        system = make_system(numerical_aperture)
        system.cache = "auto"
        return system

    def test_reuses_h_while_nothing_changed(self, sampling):
        system = self.cached_system(1.0)
        assert system.transfer(sampling) is system.transfer(sampling)

    def test_recomputes_for_a_new_sampling(self, sampling):
        system = self.cached_system(1.0)
        system.transfer(sampling)
        other = Sampling(shape=(8, 32, 32), pixel_size=sampling.pixel_size)
        assert system.transfer(other).shape == (1, 1, *_half(other))

    def test_recomputes_after_an_in_place_update(self, sampling):
        system = self.cached_system()
        with torch.no_grad():
            first = system.transfer(sampling)
            system.detection.numerical_aperture.add_(-0.2)  # what an optimizer step does
            second = system.transfer(sampling)
        assert second is not first
        torch.testing.assert_close(second, system.transfer(sampling))

    def test_bypassed_when_h_needs_gradients(self, sampling):
        system = self.cached_system()
        assert system.transfer(sampling) is not system.transfer(sampling)

    def test_swapped_parameters_miss_the_cache(self, sampling):
        system = self.cached_system()
        with torch.no_grad():
            nominal = system.transfer(sampling)
            swapped = evaluate(lambda m: m.transfer(sampling), system, {NA: torch.tensor(0.9)})
        assert not torch.equal(swapped, nominal)

    def test_vmapped_results_never_enter_the_cache(self, sampling):
        system = self.cached_system()
        with torch.no_grad():
            vmap(lambda na: evaluate(lambda m: m.transfer(sampling), system, {NA: na}))(torch.tensor([0.9, 1.0]))
            after = system.transfer(sampling)
        expected = hermitian_half(system.compute_transfer(sampling), sampling.shape[-1], real=system.real_transfer)
        torch.testing.assert_close(after, expected)

    def test_modules_built_in_inference_mode_still_work(self, sampling):
        with torch.inference_mode():
            system = self.cached_system(1.0)
            first = system.transfer(sampling)
        assert system.transfer(sampling).shape == first.shape  # inference tensors have no version counter

    @pytest.mark.parametrize(
        "convert", [lambda system: system.double(), lambda system: system.to(torch.float64)], ids=["double", "to"]
    )
    def test_module_conversion_invalidates(self, convert, sampling):
        # Conversions swap a parameter's data in place: same object, same write version, new dtype and storage.
        system = _GainSystem(cache="auto")
        with torch.no_grad():
            first = system.transfer(sampling)
            convert(system)
            converted = system.transfer(sampling)
        assert first.dtype == torch.float32
        assert converted.dtype == torch.float64

    def test_assigning_parameter_data_invalidates(self, sampling):
        system = self.cached_system()
        numerical_aperture = system.detection.numerical_aperture
        with torch.no_grad():
            first = system.transfer(sampling)
            numerical_aperture.data = torch.tensor(0.9)  # same parameter object and version, new storage
            second = system.transfer(sampling)
        assert not torch.allclose(second, first)
        expected = hermitian_half(system.compute_transfer(sampling), sampling.shape[-1], real=system.real_transfer)
        torch.testing.assert_close(second, expected)


def test_a_seeded_transfer_is_returned_while_its_state_holds_whatever_the_cache_mode(sampling):
    system = _GainSystem(cache="off")
    with torch.no_grad():
        computed = system.transfer(sampling)
        handed = computed.clone()
        assert system.seed_transfer(sampling, handed)
        assert system.transfer(sampling) is handed  # cache="off" still takes what it is handed
        other = Sampling(shape=(8, 32, 32), pixel_size=sampling.pixel_size)
        assert system.transfer(other).shape == (1, 1, *_half(other), 2)  # another sampling is computed
        system.gain.add_(1.0)  # a new state
        assert system.transfer(sampling) is not handed
    assert not system.seed_transfer(sampling, handed)  # a transfer that carries gradients is never kept
    assert system.transfer(sampling) is not handed


class _GainSystem(ShiftInvariantSystem):
    """A flat transfer ``gain`` whose only state is one parameter, so conversions change no tensor's identity."""

    def __init__(self, *, cache):
        super().__init__(fields=("a",), channel_count=1, cache=cache)
        self.gain = nn.Parameter(torch.tensor(2.0))

    def compute_transfer(self, sampling):
        nz, ny, nx = sampling.shape
        return torch.complex(self.gain, torch.zeros_like(self.gain)).expand(1, 1, nz, ny, nx // 2 + 1)


@pytest.mark.parametrize("cache", ["none", "Off", None, True])
def test_a_cache_mode_outside_off_and_auto_is_rejected(cache):
    # transfer() tests only for "off", so a typo used to turn caching on.
    with pytest.raises(ParameterError, match=r"_GainSystem cache must be one of \('off', 'auto'\)") as caught:
        _GainSystem(cache=cache)
    assert caught.value.path == "cache"


def test_cache_mode_is_a_public_name():
    # The built-in model classes pass ``cache`` through, and the rule is to import from waveorder.nn.
    assert public.CacheMode is CacheMode
    assert "CacheMode" in public.__all__


def test_detection_names_its_pupil_aberrations_by_term():
    aberrations = ZernikeAberration({"spherical": Free(0.1, tolerance=0.05), "z15": Free(0.0, tolerance=0.01)})
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=1.0, pupil_aberrations=aberrations)
    assert detection.pupil_aberrations is aberrations
    names = [name for name, _ in detection.named_parameters() if name.startswith("pupil_aberrations.")]
    assert names == ["pupil_aberrations.spherical", "pupil_aberrations.z15"]
    refused: dict[str, Any] = {"aberrations": ZernikeAberration({"spherical": 0.1})}  # the keyword before the rename
    with pytest.raises(TypeError, match="aberrations"):
        Detection(wavelength=WAVELENGTH, numerical_aperture=1.0, **refused)


class _Recording(ShiftInvariantSystem):
    """A plugin system that notes the order of its own calls and refuses one-slice stacks."""

    def __init__(self) -> None:
        super().__init__(fields=("a",), channel_count=1)
        self.real_transfer = True
        self.calls: list[str] = []

    def check_sampling(self, sampling):
        self.calls.append("check_sampling")
        if sampling.shape[0] == 1:
            raise IncompatibleModelError("_Recording needs more than one slice")

    def compute_transfer(self, sampling):
        self.calls.append("compute_transfer")
        nz, ny, nx = sampling.shape
        return torch.ones(1, 1, nz, ny, nx // 2 + 1)


def test_bind_checks_the_sampling_before_anything_is_computed(sampling):
    """Every route to ``H`` is guarded, also a plugin system's and ``system(obj, sampling)``."""
    system = _Recording()
    system.bind(sampling)
    system(torch.zeros(system.object_shape(sampling)), sampling)
    system.bind(sampling).adjoint(torch.zeros(system.data_shape(sampling)))
    assert system.calls == ["check_sampling", "compute_transfer"] * 3
    system.calls.clear()
    flat = Sampling(shape=(1, 8, 8), pixel_size=sampling.pixel_size)
    for route in (
        lambda: system.bind(flat),
        lambda: system(torch.zeros(system.object_shape(flat)), flat),
        lambda: system.bind(flat).adjoint(torch.zeros(system.data_shape(flat))),
    ):
        with pytest.raises(IncompatibleModelError, match="needs more than one slice"):
            route()
    assert system.calls == ["check_sampling"] * 3  # refused before the transfer function was computed


def test_a_pixel_size_in_nm_is_refused_by_every_route_to_the_transfer_function():
    """``system(obj, sampling)`` and ``bind`` used to try the terabyte allocation; only the verbs checked first."""
    in_nm = Sampling(shape=(8, 32, 32), pixel_size=(250, 100, 100))
    system = make_system(1.0)
    for route in (
        lambda: system.bind(in_nm),
        lambda: system(torch.zeros(system.object_shape(in_nm)), in_nm),
        lambda: system.bind(in_nm).adjoint(torch.zeros(system.data_shape(in_nm))),
    ):
        with pytest.raises(IncompatibleModelError, match=r"pixel_size is in um, got 100 along y"):
            route()


class _Blur(ShiftInvariantSystem):
    """A Gaussian low-pass written as a plugin author would, with the mistakes ``returns`` selects."""

    def __init__(self, returns: str, *, real: bool) -> None:
        super().__init__(fields=("a",), channel_count=1)
        self.register_quantity("width", 0.2, positive=True)
        self.real_transfer = real
        self.returns = returns

    def compute_transfer(self, sampling):
        width = self.quantity("width")
        if self.returns == "full":
            sizes = zip(sampling.shape, sampling.pixel_size, strict=True)
            fz, fy, fx = (torch.fft.fftfreq(n, d=d, dtype=width.dtype) for n, d in sizes)
            fz, fy, fx = fz[:, None, None], fy[None, :, None], fx[None, None, :]
        else:
            fz, fy, fx = half_spectrum_frequencies(sampling, device=width.device, dtype=width.dtype)
        transfer = torch.exp(-2 * torch.pi**2 * width**2 * (fz**2 + fy**2 + fx**2))
        return transfer if self.returns == "no axes" else transfer[None, None]


def test_the_half_spectrum_helper_gives_a_transfer_of_the_right_shape(sampling):
    system = _Blur("half", real=True)
    assert system.transfer(sampling).shape == (1, 1, *_half(sampling))
    check_system(lambda: _Blur("half", real=True), Sampling(shape=(4, 8, 6), pixel_size=(0.3, 0.1, 0.1)))


@pytest.mark.parametrize(
    ("returns", "real", "fix"),
    [
        ("half", False, r"or set self\.real_transfer = True in __init__ to return a real one"),
        ("full", True, r"real_transfer is True, so return a real \(C, F, Z, Y, X // 2 \+ 1\) half spectrum"),
        ("no axes", True, r"real_transfer is True, so return a real \(C, F, Z, Y, X // 2 \+ 1\) half spectrum"),
    ],
    ids=["real-without-the-flag", "full-spectrum", "no-channel-and-field-axes"],
)
def test_a_transfer_of_the_wrong_shape_names_the_rule_and_the_helper(sampling, returns, real, fix):
    with pytest.raises(ShapeError, match=r"^_Blur\.compute_transfer returned a transfer of shape") as error:
        _Blur(returns, real=real).transfer(sampling)
    message = str(error.value)
    assert re.search(fix, message)
    assert "waveorder.nn.functional.half_spectrum_frequencies(sampling)" in message
    assert "vmap" not in message  # the generic shape hint has nothing to do with this mistake


# --- reprs show the quantities ----------------------------------------------------------------------------------------


def test_reprs_list_each_quantity_and_free_ones_as_their_spec():
    detection = Detection(wavelength=0.52, numerical_aperture=Free(1.2, tolerance=0.1, bounds=(0.8, 1.3)))
    assert repr(detection) == (
        "Detection(wavelength=0.52, numerical_aperture=Free(1.2, tolerance=0.1, bounds=(0.8, 1.3)), "
        "z_focus_offset=0, edge='ramp')"
    )
    with torch.no_grad():
        detection.numerical_aperture.fill_(1.2345678)
    assert "numerical_aperture=Free(1.2, tolerance=0.1, bounds=(0.8, 1.3)) at 1.23457," in repr(detection)
    assert repr(Tikhonov(1e-2)) == "Tikhonov(weight=0.01)"
    assert repr(Tikhonov(1e-2).double()) == "Tikhonov(weight=0.01)"  # the exact number, not its float32 rounding
    assert repr(Tikhonov(1e-2, nonnegative=True)) == "Tikhonov(weight=0.01, nonnegative=True)"
    assert repr(TotalVariation(Free(1e-3, bounds=(1e-6, 1.0)))) == (
        "TotalVariation(weight=Free(0.001, bounds=(1e-06, 1)), inner_iterations=20)"
    )
    assert repr(Gaussian(0.5)) == "Gaussian(sigma=0.5)"
    assert repr(Poisson(gain=2.0, offset=100.0)) == "Poisson(gain=2, offset=100)"
    aberration = ZernikeAberration({"spherical": Free(0.0, bounds=(-0.2, 0.2)), "coma_x": 0.05})
    assert repr(aberration).startswith("ZernikeAberration(")
    assert "Free(0, bounds=(-0.2, 0.2))" in repr(aberration)
    assert "=0.05" in repr(aberration)


def test_solver_reprs_show_their_loop():
    assert repr(Loop(50)) == "Loop(50)"
    assert repr(Loop(200, tolerance=1e-4)) == "Loop(200, tolerance=0.0001)"
    assert repr(FISTA(loop=Loop(20))) == "FISTA(loop=Loop(20))"
    assert repr(RichardsonLucy()) == "RichardsonLucy(loop=Loop(25), background=0)"
    assert repr(RichardsonLucy(loop=Loop(3), back_projector="gaussian", start="ones")) == (
        "RichardsonLucy(loop=Loop(3), background=0, back_projector='gaussian', start='ones')"
    )
    assert repr(ADMM(rho=0.01)) == "ADMM(loop=Loop(10), rho=0.01, damping=0.001)"
    assert repr(Wiener()) == "Wiener()"
    score = repr(MidbandPowerScore(solver=Wiener(), cutoff=3.4, fields=("phase",)))
    assert "cutoff=3.4, fractions=(0.125, 0.25), fields=('phase',)" in score
    assert "(solver): Wiener()" in score


def test_a_model_repr_shows_every_part_of_the_optics():
    tilt = IlluminationTilt(zenith=Free(0.1, bounds=(-0.3, 0.3)), azimuth=0.5)
    illumination = Illumination(wavelength=0.532, numerical_aperture=0.5, tilt=tilt)
    thick = ThickPhase3D(illumination, Detection(numerical_aperture=0.9), index_of_refraction_media=1.33)
    model = ImagingModel(thick, Tikhonov(1e-3), Gaussian(0.05))
    text = repr(model)
    for line in (
        "(system): ThickPhase3D(",
        "index_of_refraction_media=1.33, absorption_ratio=0",
        "(illumination): Illumination(",
        "wavelength=0.532, numerical_aperture=0.5",
        "(tilt): IlluminationTilt(zenith=Free(0.1, bounds=(-0.3, 0.3)), azimuth=0.5)",
        "(detection): Detection(wavelength=0.532, numerical_aperture=0.9, z_focus_offset=0, edge='ramp')",
        "(object_prior): Tikhonov(weight=0.001)",
        "(noise): Gaussian(sigma=0.05)",
    ):
        assert line in text
    independent = ThickPhase3D(
        Illumination(wavelength=0.532, numerical_aperture=0.5),
        Detection(numerical_aperture=0.9),
        index_of_refraction_media=1.33,
        invert_phase_contrast=True,
        absorption="independent",
    )
    assert 'index_of_refraction_media=1.33, invert_phase_contrast=True, absorption="independent"' in repr(independent)
    thin = ThinPhase2D(
        Illumination(wavelength=0.532, numerical_aperture=0.5),
        Detection(numerical_aperture=0.9),
        index_of_refraction_media=1.33,
        z_positions=(0.5, 0.0, -0.5),
    )
    assert "index_of_refraction_media=1.33, z_positions=(0.5, 0.0, -0.5)" in repr(thin)
    confocal = ConfocalFluorescence3D(
        Detection(wavelength=0.52, numerical_aperture=1.2), pinhole_diameter=0.4, index_of_refraction_media=1.33
    )
    assert "ConfocalFluorescence3D(\n  index_of_refraction_media=1.33, pinhole_diameter=0.4" in repr(confocal)


def test_a_repr_reads_no_value_where_values_cannot_be_read():
    # Inside vmap a quantity is a batch: the repr then shows the spec alone and does not raise.
    system = make_system(FREE_NA)
    seen: list[str] = []

    def describe(aperture):
        seen.append(evaluate(lambda module: repr(module.detection), system, {NA: aperture}))
        return aperture

    vmap(describe)(torch.tensor([0.9, 1.0]))
    assert "numerical_aperture=Free(1.1, tolerance=0.2, bounds=(0.8, 1.2))," in seen[0]
    assert repr_parts("", "a=1", "", "b=2") == "a=1, b=2"
