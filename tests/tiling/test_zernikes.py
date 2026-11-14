"""Per-tile Zernike recovery: the 4.x recipe against 3.x ``recover_zernikes`` on a simulated bead field."""

import pytest
import torch

from tests.tiling.legacy_zernikes import recover
from waveorder.errors import ParameterError, ShapeError
from waveorder.nn import (
    Detection,
    Free,
    Gaussian,
    GradientSearch,
    ImagingModel,
    Sampling,
    Tikhonov,
    WidefieldFluorescence3D,
    Wiener,
    ZernikeAberration,
    load_parameters,
)
from waveorder.nn.systems.tiled import LEARNING_RATES, BeadFit, zernike_terms
from waveorder.tiling import recover_zernikes
from waveorder.tiling.plan import TilePlan

WAVELENGTH, APERTURE, INDEX = 0.52, 0.8, 1.33
PIXEL = (0.2, 0.1, 0.1)
Z, SIDE = 8, 16
MODES = (5, 7, 11)
FIELD = (Z, 2 * SIDE, 2 * SIDE)
# Astigmatism, coma and spherical aberration in waves, one row per tile of a 2 x 2 field (raster order).
TRUTH = torch.tensor(
    [[0.06, -0.03, 0.02], [-0.04, 0.05, 0.0], [0.02, 0.0, -0.03], [0.0, -0.05, 0.04]], dtype=torch.float64
)
SAMPLING = Sampling(shape=(Z, SIDE, SIDE), pixel_size=PIXEL)


def _model(values: list[float] | None = None) -> ImagingModel:
    """The sigmoid-edged widefield model 3.x fitted; free bounded terms, or fixed at ``values``."""
    terms = {
        f"z{j}": Free(0.0, bounds=(-0.5, 0.5)) if values is None else v
        for j, v in zip(MODES, values or MODES, strict=True)
    }
    detection = Detection(
        wavelength=WAVELENGTH, numerical_aperture=APERTURE, edge="sigmoid", pupil_aberrations=ZernikeAberration(terms)
    )
    system = WidefieldFluorescence3D(detection, index_of_refraction_media=INDEX)
    return ImagingModel(system, Tikhonov(1e-3), Gaussian(1e-3)).double()


def _field() -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One bead per tile at its centre, each imaged with its row of :data:`TRUTH`, laid out as a 2 x 2 field."""
    bead = BeadFit.gaussian(SAMPLING, like=torch.zeros((), dtype=torch.float64)).obj
    field = torch.zeros(Z, 2 * SIDE, 2 * SIDE, dtype=torch.float64)
    tiles = []
    for index, (row, column) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        with torch.no_grad():
            tile = _model(TRUTH[index].tolist())(bead, SAMPLING)[0]
        field[:, row * SIDE : (row + 1) * SIDE, column * SIDE : (column + 1) * SIDE] = tile
        tiles.append(tile)
    return field, tiles


def test_the_recipe_recovers_each_tiles_coefficients_as_3x_does():
    field, tiles = _field()
    model = _model()
    plan = TilePlan(shape=FIELD, tile=(Z, SIDE, SIDE))
    terms = zernike_terms(model)
    search = GradientSearch(iterations=120, learning_rates={name: LEARNING_RATES[j] for name, j in terms.items()})
    result = recover_zernikes(field, model, plan, pixel_size=PIXEL, search=search)
    recovered = torch.stack(list(result.coefficients(model).values()), dim=1)
    assert list(result.coefficients(model)) == list(MODES)
    legacy = recover(
        torch.stack(tiles),
        yx_pixel_size=PIXEL[1],
        z_pixel_size=PIXEL[0],
        wavelength=WAVELENGTH,
        numerical_aperture=APERTURE,
        index=INDEX,
        noll_indices=MODES,
        lr_per_mode=LEARNING_RATES,
        n_iter=120,
    )
    # Both fits reach the truth to a few 1e-4 waves in 120 steps, and each other: the two PSF models agree.
    torch.testing.assert_close(recovered, TRUTH, rtol=0, atol=3e-4)
    torch.testing.assert_close(legacy, TRUTH, rtol=0, atol=3e-4)
    torch.testing.assert_close(recovered, legacy, rtol=0, atol=3e-4)
    # The blended deconvolution covers the field; each tile's own coefficients deconvolve it.
    assert result.reconstruction.shape == (1, *field.shape)
    assert bool(torch.isfinite(result.reconstruction).all())
    assert result.calibration.objective.shape == (4,)


def test_the_bead_fit_is_scale_free_and_checks_its_object():
    field, tiles = _field()
    fit = BeadFit.gaussian(SAMPLING, like=torch.zeros((), dtype=torch.float64))
    assert fit.obj[0, Z // 2, SIDE // 2, SIDE // 2].item() == 1.0
    model = _model(TRUTH[0].tolist())
    data = tiles[0][None]
    # At the truth the gain fits any brightness: the loss is zero for the data scaled by 7.
    assert fit(7 * data, model, SAMPLING).item() == pytest.approx(0.0, abs=1e-20)
    assert BeadFit(fit.obj, scale_fit=False)(7 * data, model, SAMPLING).item() > 1e-6
    with pytest.raises(ShapeError, match="BeadFit object has shape"):
        BeadFit(fit.obj[:, :-1])(data, model, SAMPLING)
    with pytest.raises(ParameterError) as error:
        BeadFit(torch.ones(1, Z, SIDE, SIDE, dtype=torch.int64))
    assert error.value.path == "obj"
    del field


def test_the_recipe_needs_free_zernike_terms():
    field, _ = _field()
    plan = TilePlan(shape=FIELD, tile=(Z, SIDE, SIDE))
    with pytest.raises(ParameterError, match="free ZernikeAberration terms") as error:
        recover_zernikes(field, _model(TRUTH[0].tolist()), plan, pixel_size=PIXEL)
    assert error.value.path == "model"


def test_a_float64_model_deconvolves_and_blends_in_float64():
    # One tile: the blended reconstruction is that tile's Wiener estimate with its recovered coefficients.
    _, tiles = _field()
    model = _model()
    plan = TilePlan(shape=(Z, SIDE, SIDE), tile=(Z, SIDE, SIDE))
    search = GradientSearch(iterations=5, learning_rates=dict.fromkeys(zernike_terms(model), 0.02))
    result = recover_zernikes(tiles[0], model, plan, pixel_size=PIXEL, search=search)
    assert result.reconstruction.dtype == torch.float64
    load_parameters(model, {name: values[0] for name, values in result.calibration.parameters.items()})
    with torch.no_grad():
        direct = Wiener()(tiles[0][None], model, SAMPLING).estimate
    torch.testing.assert_close(result.reconstruction, direct, rtol=1e-12, atol=1e-12 * direct.abs().max().item())
