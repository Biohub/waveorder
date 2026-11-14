"""The calibration scores beyond the evidence: GCV and the sharpness scores."""

import pytest
import torch

from tests.nn import legacy_losses
from tests.nn.conftest import FREE_NA, WAVELENGTH, MatrixSystem, make_model, random_transfer, simulate
from waveorder.errors import IncompatibleModelError, ParameterError
from waveorder.nn import FISTA, Detection, Free, Gaussian, ImagingModel, Sampling, Tikhonov, Wiener, calibrate
from waveorder.nn.optics import Illumination
from waveorder.nn.scores import (
    GCV,
    LaplacianVarianceScore,
    MidbandPowerScore,
    NormalizedVarianceScore,
    SpectralFlatnessScore,
    TotalVariationScore,
    transverse_cutoff,
)
from waveorder.nn.systems.phase import ThinPhase2D
from waveorder.testing import check_score

GRID = Sampling(shape=(3, 32, 32), pixel_size=(0.25, 0.1, 0.1))
THIN = Sampling(shape=(1, 32, 32), pixel_size=(0.25, 0.1, 0.1))
TINY = Sampling(shape=(4, 8, 6), pixel_size=(0.4, 0.15, 0.15))
NA_DET = 1.0
CUTOFF = 2 * NA_DET / WAVELENGTH
FRACTIONS = (0.125, 0.25)
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICES = [torch.device("cpu"), pytest.param(torch.device("cuda"), marks=CUDA)]


def _sharpness_cases():
    """(score, the 3.x loss with the same settings) for each 3.x loss name."""
    return {
        "midband_power": (
            MidbandPowerScore(solver=Wiener(), cutoff=CUTOFF, fractions=FRACTIONS),
            legacy_losses.midband_power_loss(NA_DET, WAVELENGTH, 0.1, FRACTIONS),
        ),
        "spectral_flatness": (
            SpectralFlatnessScore(solver=Wiener(), cutoff=CUTOFF, fractions=FRACTIONS),
            legacy_losses.spectral_flatness_loss(NA_DET, WAVELENGTH, 0.1, FRACTIONS),
        ),
        "total_variation": (TotalVariationScore(solver=Wiener()), legacy_losses.total_variation_loss()),
        "laplacian_variance": (LaplacianVarianceScore(solver=Wiener()), legacy_losses.laplacian_variance_loss()),
        "normalized_variance": (NormalizedVarianceScore(solver=Wiener()), legacy_losses.normalized_variance_loss()),
    }


SHARPNESS = sorted(_sharpness_cases())


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("sampling", [GRID, THIN], ids=["3d", "2d"])
@pytest.mark.parametrize("name", SHARPNESS)
def test_sharpness_scores_match_the_3x_losses_on_the_reconstruction(name, sampling, dtype):
    score, legacy = _sharpness_cases()[name]
    model = make_model().to(dtype)
    _, data = simulate(model, sampling, scale=1.0, seed=2)
    data = data + 0.5  # a positive mean, as fluorescence has, keeps the normalized variance well conditioned
    reconstruction = Wiener()(data, model, sampling).estimate[0]
    expected = legacy(reconstruction[0] if sampling.shape[0] == 1 else reconstruction)  # 3.x took (Y, X) for 2D
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    torch.testing.assert_close(score(data, model, sampling), expected, rtol=tolerance, atol=0)


@pytest.mark.parametrize("name", SHARPNESS)
def test_sharpness_is_lower_for_the_in_focus_reconstruction(name):
    # Defocusing the data moves its content out of focus; every 3.x loss calls the focused reconstruction sharper.
    model = make_model().double()
    _, data = simulate(model, GRID, scale=1.0, seed=3)
    blurred = torch.fft.irfft2(torch.fft.rfft2(data) * _gaussian_blur(GRID, width=0.3), s=GRID.shape[-2:])
    score, _ = _sharpness_cases()[name]
    assert score(data + 0.5, model, GRID) < score(blurred + 0.5, model, GRID)


def _gaussian_blur(sampling: Sampling, *, width: float) -> torch.Tensor:
    """The rfft2 of a transverse Gaussian of standard deviation ``width`` (length units)."""
    _, ny, nx = sampling.shape
    fy = torch.fft.fftfreq(ny, d=sampling.pixel_size[1], dtype=torch.float64)
    fx = torch.fft.rfftfreq(nx, d=sampling.pixel_size[2], dtype=torch.float64)
    return torch.exp(-2 * (torch.pi * width) ** 2 * (fy[:, None] ** 2 + fx[None, :] ** 2))


def _matrix_model(channels: int, fields: int, *, weight: float = 0.3, sigma: float = 0.2) -> ImagingModel:
    transfer = random_transfer(channels, fields, TINY.shape, seed=channels + 10 * fields, dtype=torch.float64)
    system = MatrixSystem(transfer, fields=tuple(f"f{index}" for index in range(fields)))
    return ImagingModel(system, Tikhonov(weight), Gaussian(sigma)).double()


def _dense_hat(model: ImagingModel, data: torch.Tensor) -> torch.Tensor:
    """The hat matrix ``S = H M`` of the Wiener estimate, ``n x n``, from the operator and solver as linear maps."""
    operator = model.system.bind(TINY)
    fields = model.system.object_shape(TINY)
    count = data.numel()
    forward = torch.func.jacrev(lambda f: operator.apply(f))(torch.zeros(fields, dtype=data.dtype))
    estimate = torch.func.jacrev(lambda g: Wiener()(g, model, TINY).estimate)(data)
    return forward.reshape(count, -1) @ estimate.reshape(-1, count)


@pytest.mark.parametrize(("channels", "fields"), [(1, 1), (2, 1), (1, 2), (3, 2)])
def test_gcv_matches_the_dense_hat_matrix(channels, fields):
    model = _matrix_model(channels, fields)
    data = torch.randn(channels, *TINY.shape, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    hat = _dense_hat(model, data)
    flat = data.reshape(-1)
    residual = (flat - hat @ flat).square().sum()
    trace, count = hat.trace(), flat.numel()
    torch.testing.assert_close(GCV()(data, model, TINY), count * residual / (count - trace) ** 2, rtol=1e-10, atol=0)


def test_gcv_chooses_a_weight_near_the_risk_minimum():
    # The weight that minimizes each estimate is within a factor 3 of the one that minimizes the true prediction risk.
    generator = torch.Generator().manual_seed(6)
    weights = torch.logspace(-3, 1, 25, dtype=torch.float64)
    base = _matrix_model(2, 1, sigma=0.3)
    operator = base.system.bind(TINY)
    obj = torch.randn(base.system.object_shape(TINY), generator=generator, dtype=torch.float64)
    clean = operator.apply(obj)
    data = clean + 0.3 * torch.randn(clean.shape, generator=generator, dtype=torch.float64)
    gcv, risk = [], []
    for weight in weights.tolist():
        model = _matrix_model(2, 1, weight=weight, sigma=0.3)
        gcv.append(GCV()(data, model, TINY))
        risk.append((operator.apply(Wiener()(data, model, TINY).estimate) - clean).square().sum())
    best = weights[torch.stack(risk).argmin()]
    chosen = weights[torch.stack(gcv).argmin()]
    assert 1 / 3 < chosen / best < 3, f"gcv chose weight {chosen:.3g}, the risk minimum is at {best:.3g}"


def test_calibrating_a_free_weight_with_gcv_lands_near_the_risk_minimum():
    score = GCV()
    truth = make_model().double()
    obj, data = simulate(truth, GRID, scale=1.0, seed=9)
    operator = truth.system.bind(GRID)
    clean = operator.apply(obj)

    def model(weight):
        return ImagingModel(truth.system, Tikhonov(weight), Gaussian(0.05)).double()

    weights = torch.logspace(-5, 0, 41, dtype=torch.float64)
    risks = [(operator.apply(Wiener()(data, model(w), GRID).estimate) - clean).square().sum() for w in weights.tolist()]
    best = weights[torch.stack(risks).argmin()]
    result = calibrate(data, model(Free(1e-1, bounds=(1e-5, 10.0))), GRID, score=score)
    chosen = result.parameters["object_prior.weight"]
    assert 1 / 3 < chosen / best < 3, f"calibrated weight {chosen:.3g}, risk minimum {best:.3g}"


SCORES = {
    "gcv": GCV,
    **{name: (lambda name=name: _sharpness_cases()[name][0]) for name in SHARPNESS},
}


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("device", DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("name", sorted(SCORES))
def test_scores_pass_the_conformance_suite(name, device, dtype):
    model = make_model(FREE_NA).to(dtype=dtype)
    sampling = Sampling(shape=(2, 16, 16), pixel_size=(0.25, 0.1, 0.1))
    _, data = simulate(model, sampling, scale=1.0, seed=7)  # a CPU generator: simulate on the CPU, then move
    model, data = model.to(device=device), data.to(device)
    check_score(SCORES[name](), data=data + 0.5, model=model, sampling=sampling)


def test_gcv_needs_the_linear_gaussian_closed_form():
    score = GCV()
    model = ImagingModel(make_model().system, Tikhonov(1e-2, nonnegative=True), Gaussian(0.05))
    with pytest.raises(IncompatibleModelError, match=f"{type(score).__name__} has no closed form"):
        score(torch.zeros(1, *TINY.shape), model, TINY)


def test_an_empty_midband_annulus_gives_a_nan_score():
    # No sample of a 32-pixel grid at 0.1 lies strictly between these radii: the score is NaN, never a silent 0.
    model = make_model()
    _, data = simulate(model, GRID, scale=1.0, seed=8)
    for kind in (MidbandPowerScore, SpectralFlatnessScore):
        score = kind(solver=Wiener(), cutoff=1.0, fractions=(0.0, 0.1))
        assert torch.isnan(score(data, model, GRID))


@pytest.mark.parametrize(
    ("arguments", "path"),
    [
        ({"cutoff": 0.0}, "cutoff"),
        ({"cutoff": float("inf")}, "cutoff"),
        ({"cutoff": CUTOFF, "fractions": (0.25, 0.125)}, "fractions"),
        ({"cutoff": CUTOFF, "fractions": (-0.1, 0.2)}, "fractions"),
        ({"cutoff": CUTOFF, "fractions": (0.1,)}, "fractions"),
        ({"cutoff": CUTOFF, "fractions": "low"}, "fractions"),
    ],
)
def test_midband_settings_are_validated(arguments, path):
    for kind in (MidbandPowerScore, SpectralFlatnessScore):
        with pytest.raises(ParameterError, match=path) as raised:
            kind(solver=Wiener(), **arguments)
        assert raised.value.path == path


def _two_field_problem(device: torch.device | str = "cpu"):
    """A phase and an absorption field, as thin phase has: 3.x scored the phase alone."""
    transfer = random_transfer(1, 2, GRID.shape, seed=11, dtype=torch.float64)
    model = ImagingModel(MatrixSystem(transfer, fields=("phase", "absorption")), Tikhonov(0.3), Gaussian(0.2))
    data = torch.randn(1, *GRID.shape, generator=torch.Generator().manual_seed(12), dtype=torch.float64)
    return model.double().to(device), data.to(device)


@pytest.mark.parametrize("name", SHARPNESS)
def test_a_field_selection_scores_that_field_as_3x_did(name):
    model, data = _two_field_problem()
    score, legacy = _sharpness_cases()[name]
    arguments = {"cutoff": CUTOFF, "fractions": FRACTIONS} if name in ("midband_power", "spectral_flatness") else {}
    phase_only = type(score)(solver=Wiener(), fields=("phase",), **arguments)
    reconstruction = Wiener()(data, model, GRID).estimate
    torch.testing.assert_close(phase_only(data, model, GRID), legacy(reconstruction[0]), rtol=1e-12, atol=0)
    assert not torch.isclose(score(data, model, GRID), legacy(reconstruction[0]))  # every field: a different value


@pytest.mark.parametrize("device", DEVICES, ids=["cpu", "cuda"])
def test_a_field_selection_passes_the_conformance_suite(device):
    model, data = _two_field_problem(device)
    check_score(MidbandPowerScore(solver=Wiener(), cutoff=CUTOFF, fields=("absorption",)), data=data, model=model,
                sampling=GRID)  # fmt: skip


def test_a_field_the_system_lacks_is_incompatible():
    model, data = _two_field_problem()
    with pytest.raises(IncompatibleModelError, match=r"\['retardance'\].*\('phase', 'absorption'\)"):
        TotalVariationScore(solver=Wiener(), fields=("retardance",))(data, model, GRID)


@pytest.mark.parametrize("fields", [(), ("phase", "phase")])
def test_field_selections_are_validated(fields):
    with pytest.raises(ParameterError, match="fields") as raised:
        TotalVariationScore(solver=Wiener(), fields=fields)
    assert raised.value.path == "fields"


def _thin_phase(numerical_aperture: float | Free = 0.4) -> tuple[ImagingModel, Sampling]:
    illumination = Illumination(wavelength=WAVELENGTH, numerical_aperture=numerical_aperture)
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=0.9)
    system = ThinPhase2D(illumination, detection, index_of_refraction_media=1.33)
    sampling = Sampling(shape=(5, 32, 32), pixel_size=(0.6, 0.1, 0.1))
    return ImagingModel(system, Tikhonov(1e-2), Gaussian(0.05)).double(), sampling


def test_thin_phase_scores_the_phase_field_as_3x_phase_optimize_did():
    # 3.x phase.optimize scored isotropic_thin_3d.reconstruct(...)[1], the phase, with the illumination wavelength.
    model, sampling = _thin_phase()
    _, data = simulate(model, sampling, scale=0.3, seed=13)
    cutoff = 2 * 0.9 / WAVELENGTH
    phase_only = MidbandPowerScore(solver=Wiener(), cutoff=cutoff, fields=("phase",))
    phase = Wiener()(data, model, sampling).estimate[model.system.fields.index("phase"), 0]
    expected = legacy_losses.midband_power_loss(0.9, WAVELENGTH, 0.1, FRACTIONS)(phase)
    torch.testing.assert_close(phase_only(data, model, sampling), expected, rtol=1e-12, atol=0)
    every_field = MidbandPowerScore(solver=Wiener(), cutoff=cutoff)(data, model, sampling)
    assert abs(every_field / expected - 1) > 0.01  # the absorption field would move the 3.x objective


def test_a_phase_only_score_on_thin_phase_passes_the_conformance_suite():
    model, sampling = _thin_phase(Free(0.4, tolerance=0.1, bounds=(0.2, 0.6)))
    _, data = simulate(model, sampling, scale=0.3, seed=14)
    score = MidbandPowerScore(solver=Wiener(), cutoff=2 * 0.9 / WAVELENGTH, fields=("phase",))
    check_score(score, data=data, model=model, sampling=sampling)


# --- scores built without their solver and cutoff ---------------------------------------------------------------------


def test_a_sharpness_score_built_without_its_arguments_is_filled_and_scores_the_same():
    model = make_model()
    _, data = simulate(model, GRID, scale=1.0, seed=2)
    score = MidbandPowerScore()
    assert score.unfilled == ("solver", "cutoff")
    assert score.solver is None
    score.fill(model.system, solver=Wiener())
    assert score.unfilled == ()
    assert score.cutoff == CUTOFF == transverse_cutoff(model.system)  # the same Python floats: no float32 rounding
    explicit = MidbandPowerScore(solver=Wiener(), cutoff=CUTOFF)
    assert torch.equal(score(data, model, GRID), explicit(data, model, GRID))
    assert [name for name, _ in score.named_modules()] == ["", "solver"]  # the filled solver is a child module
    spatial = TotalVariationScore()
    assert spatial.unfilled == ("solver",)
    spatial.fill(model.system, solver=Wiener())
    assert torch.equal(spatial(data, model, GRID), TotalVariationScore(solver=Wiener())(data, model, GRID))


def test_filling_again_replaces_what_a_fill_set_and_never_what_was_given():
    model, first, second = make_model(), Wiener(), FISTA()
    score = SpectralFlatnessScore()
    score.fill(model.system, solver=first)
    score.fill(make_model(1.2).system, solver=second)  # the model's solver and system changed
    assert score.solver is second
    assert score.cutoff == 2 * 1.2 / WAVELENGTH
    given = MidbandPowerScore(solver=first, cutoff=3.0)
    given.fill(model.system, solver=second)
    assert given.solver is first
    assert given.cutoff == 3.0
    half = MidbandPowerScore(cutoff=3.0)
    assert half.unfilled == ("solver",)
    half.fill(model.system, solver=second)
    assert (half.solver, half.cutoff) == (second, 3.0)
    assigned = MidbandPowerScore()
    assigned.cutoff = 2.5  # an assigned cutoff is a given one
    assigned.fill(model.system, solver=first)
    assert assigned.cutoff == 2.5
    with pytest.raises(ParameterError, match="cutoff must be positive and finite") as raised:
        assigned.cutoff = 0.0
    assert raised.value.path == "cutoff"


def test_an_unfilled_score_used_with_the_nn_functions_names_what_to_pass():
    model, data = make_model(FREE_NA), torch.zeros(1, *GRID.shape)
    both = r"^MidbandPowerScore was built without solver= \(a Solver, e\.g\. Wiener\(\)\) and cutoff= \(2 NA / wavele"
    with pytest.raises(ParameterError, match=both) as raised:
        MidbandPowerScore()(data, model, GRID)
    assert raised.value.path == "solver"
    assert "waveorder.nn.scores.transverse_cutoff(system)" in str(raised.value)
    assert "score=MidbandPowerScore()" in str(raised.value)  # what a waveorder.models model takes
    with pytest.raises(ParameterError, match=r"^MidbandPowerScore was built without cutoff= ") as raised:
        MidbandPowerScore(solver=Wiener())(data, model, GRID)
    assert raised.value.path == "cutoff"
    with pytest.raises(ParameterError, match=r"^TotalVariationScore was built without solver= .*\. Pass it") as raised:
        calibrate(data, model, GRID, score=TotalVariationScore())
    assert raised.value.path == "solver"
    with pytest.raises(ParameterError, match=r"was built without cutoff= ") as raised:
        _ = MidbandPowerScore().cutoff
    assert raised.value.path == "cutoff"
    assert "cutoff=None" in repr(MidbandPowerScore())


def test_the_transverse_cutoff_is_two_apertures_over_the_detected_wavelength():
    free = make_model(Free(1.1, tolerance=0.2, bounds=(0.8, 1.2)))
    assert transverse_cutoff(free.system) == 2 * 1.1 / WAVELENGTH  # the nominal value of a free aperture
    with torch.no_grad():
        free.get_parameter("system.detection.numerical_aperture").fill_(0.9)
    assert transverse_cutoff(free.system) == 2 * 1.1 / WAVELENGTH  # fixed: it does not follow a calibration
    # Brightfield detects the illumination's light: a detection built without a wavelength took the illumination's.
    thin = ThinPhase2D(
        Illumination(wavelength=0.532, numerical_aperture=0.4),
        Detection(numerical_aperture=0.9),
        index_of_refraction_media=1.33,
    )
    assert transverse_cutoff(thin) == 2 * 0.9 / 0.532
    bare = MatrixSystem(random_transfer(1, 1, TINY.shape), fields=("a",))
    with pytest.raises(ParameterError, match=r"MatrixSystem has none, so pass cutoff= to the score") as raised:
        transverse_cutoff(bare)
    assert raised.value.path == "cutoff"
    with pytest.raises(ParameterError, match="so pass cutoff= to the score"):
        MidbandPowerScore().fill(bare, solver=Wiener())
    TotalVariationScore().fill(bare, solver=Wiener())  # a score without a cutoff needs no detection


def _evidence_model(sigma: float, weight: float) -> tuple[ImagingModel, torch.Tensor, Sampling]:
    from tests.nn.wo40_fixtures import GaussianTransfer, sweep_volume

    model = ImagingModel(GaussianTransfer(), Tikhonov(weight), Gaussian(sigma)).double()
    return model, sweep_volume().double()[None], Sampling(shape=(8, 32, 32), pixel_size=(1.0, 1.0, 1.0))


@pytest.mark.parametrize(
    ("sigma", "weight", "without_dc"),
    # wo-4.0 calibrate.scores.whittle_evidence(data_fft=fftn(sweep_volume(), norm="ortho"), otf=gaussian_otf(...),
    # noise_sigma, regularization_strength, exclude_dc=True).
    [(1.0, 1e-3, 95254.421875), (3.0, 0.05, 39378.4921875)],
)
def test_the_evidence_leaves_out_the_zero_frequency_as_the_wo40_evidence(sigma, weight, without_dc):
    from waveorder.nn import Evidence

    model, data, sampling = _evidence_model(sigma, weight)
    excluded = float(Evidence()(data, model, sampling))
    assert excluded == pytest.approx(without_dc, rel=1e-6)
    # A constant moves only the zero frequency, which the evidence leaves out.
    assert float(Evidence()(data + 100.0, model, sampling)) == pytest.approx(excluded, rel=1e-12)


def test_the_centre_slice_midband_is_the_wo40_recovery_loss():
    from waveorder.nn.scores import CentreSliceMidbandScore

    tile = torch.randn((1, 5, 26, 26), generator=torch.Generator().manual_seed(11))
    score = CentreSliceMidbandScore(solver=Wiener(), cutoff=2 * 1.2 / 0.532, fractions=(0.2, 0.4))
    sampling = Sampling(shape=(5, 26, 26), pixel_size=(0.25, 0.1, 0.1))
    loss = -score.sharpness(score.slices(tile), sampling)
    # wo-4.0 shift_variant_recovery._sharpness_loss_per_tile(tile, "midband", _midband_mask((26, 26), 0.1, 1.2, 0.532,
    # (0.2, 0.4))): centre slice, power, mean over every pixel.
    assert float(loss[0]) == pytest.approx(-59.05818176269531, rel=1e-6)
    other = tile.clone()
    other[:, [0, 1, 3, 4]] = 0
    assert torch.equal(score.slices(other), score.slices(tile))
