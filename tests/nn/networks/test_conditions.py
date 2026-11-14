"""Conditions and normalizers: the values a student reads besides its data, and what happens around it."""

import math

import pytest
import torch

from tests.nn.conftest import make_model
from waveorder.errors import IncompatibleModelError, ParameterError, ShapeError
from waveorder.nn import Sampling, SolveContext
from waveorder.nn.networks import (
    Affine,
    Attributes,
    FieldPosition,
    FieldStandardize,
    Identity,
    Place,
    PlaneOffsets,
    Quantities,
)
from waveorder.registry import arguments, describe, resolve

SAMPLING = Sampling(shape=(4, 32, 48), pixel_size=(0.5, 0.1, 0.1))


def _place(**context: object) -> Place:
    return Place.of(SAMPLING, context=SolveContext(**context))  # ty: ignore[invalid-argument-type]


def test_data_without_a_context_is_a_whole_field_centred_at_one_half():
    place = Place.of(SAMPLING)
    assert place.field_shape == (32, 48)
    assert place.region == (slice(0, 32), slice(0, 48))
    assert FieldPosition().values(place, None).tolist() == [0.5, 0.5]


def test_the_field_position_of_a_tile_is_its_centre_in_the_field():
    place = _place(field_shape=(64, 96), region=(slice(32, 64), slice(0, 48)))
    assert FieldPosition().values(place, None).tolist() == [0.75, 0.25]


def test_the_field_position_maps_to_the_convention_a_network_was_trained_on():
    # x before y, in [-1, 1] at the centres of the outer patches of an 8 x 8 grid over a 2048 field.
    production = FieldPosition(axes="xy", offset=0.5, scale=0.4375)
    corner = Place.of(
        Sampling(shape=(4, 256, 256), pixel_size=(0.5, 0.1, 0.1)),
        context=SolveContext(field_shape=(2048, 2048), region=(slice(1792, 2048), slice(0, 256))),
    )
    assert production.values(corner, None).tolist() == [-1.0, 1.0]
    per_axis = FieldPosition(offset=(0.25, 0.5), scale=(0.5, 0.25), name="where")
    assert per_axis.values(_place(field_shape=(64, 96), region=(slice(32, 64), slice(0, 48))), None).tolist() == [
        1.0,
        -1.0,
    ]
    assert (per_axis.name, per_axis.size) == ("where", 2)
    with pytest.raises(ParameterError, match="axes must be 'yx' or 'xy'"):
        FieldPosition(axes="zx")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ParameterError, match="scale must not be 0"):
        FieldPosition(scale=(1.0, 0.0))
    with pytest.raises(ParameterError, match="one for each of the 2 values"):
        FieldPosition(offset=(0.1, 0.2, 0.3))


def test_a_context_that_does_not_describe_the_data_is_refused():
    with pytest.raises(ShapeError, match="names a region but no field"):
        _place(region=(slice(0, 32), slice(0, 48)))
    with pytest.raises(ShapeError, match="spans 16 pixels, but the block has 32"):
        _place(field_shape=(64, 96), region=(slice(0, 16), slice(0, 48)))
    with pytest.raises(ShapeError, match="outside the field"):
        _place(field_shape=(64, 96), region=(slice(48, 80), slice(0, 48)))
    with pytest.raises(ShapeError, match="does not describe data of Y, X = \\(32, 48\\)"):
        _place(field_shape=(64, 96))  # part of a field, without its region


def test_plane_offsets_are_the_defocus_from_the_focus_plane_in_axial_units_padded_with_nan():
    # unit = n * wavelength / NA ** 2 = 1.0 * 0.5 / 1.0 = 0.5 um, the plane spacing: one unit per plane.
    condition = PlaneOffsets(planes=6, wavelength=0.5, numerical_aperture=1.0)
    assert condition.unit == pytest.approx(0.5)
    values = condition.values(Place.of(SAMPLING), None)
    assert values[:4].tolist() == [2.0, 1.0, 0.0, -1.0]  # plane Z // 2 = 2 is in focus; z runs down the stack
    assert all(math.isnan(value) for value in values[4:].tolist())
    denser = PlaneOffsets(4, wavelength=0.5, numerical_aperture=1.0, index_of_refraction=2.0, focus_offset=0.5)
    assert denser.values(Place.of(SAMPLING), None).tolist() == [1.25, 0.75, 0.25, -0.25]
    by_attribute = PlaneOffsets(4, wavelength=0.5, numerical_aperture=1.0, focus_offset="focus")
    place = Place.of(SAMPLING, attributes={"focus": -1.0})
    assert by_attribute.values(place, None).tolist() == [1.0, 0.0, -1.0, -2.0]
    with pytest.raises(ParameterError, match=r"attribute 'focus', which this place does not have"):
        by_attribute.values(Place.of(SAMPLING), None)
    with pytest.raises(ShapeError, match="holds 3 planes, but the stack has 4"):
        PlaneOffsets(planes=3, wavelength=0.5, numerical_aperture=1.0).values(Place.of(SAMPLING), None)
    with pytest.raises(ParameterError, match="numerical_aperture must be a positive number"):
        PlaneOffsets(4, wavelength=0.5, numerical_aperture=0.0)


def test_attributes_are_read_by_name_and_scaled():
    condition = Attributes(["row", "column"], offset=(1.0, 2.0), scale=2.0, name="well")
    place = Place.of(SAMPLING, attributes={"column": 6.0, "row": 3.0, "other": 9.0})
    assert condition.values(place, None).tolist() == [1.0, 2.0]
    with pytest.raises(ParameterError, match=r"reads the attributes \['column'\], which this place does not have"):
        condition.values(Place.of(SAMPLING, attributes={"row": 1.0}), None)
    with pytest.raises(ParameterError, match="scale must not be 0"):
        Attributes(["row"], scale=0)
    with pytest.raises(ParameterError, match="distinct strings"):
        Attributes(["row", "row"])


def test_quantities_are_the_current_values_of_the_system():
    model = make_model(numerical_aperture=1.1)
    condition = Quantities(["detection.numerical_aperture", "detection.z_focus_offset"])
    torch.testing.assert_close(condition.values(Place.of(SAMPLING), model), torch.tensor([1.1, 0.0]))
    with pytest.raises(ParameterError, match="no model was given"):
        condition.values(Place.of(SAMPLING), None)
    with pytest.raises(ParameterError, match=r"'detection\.wavelength', which is not a scalar quantity of Widefield"):
        Quantities(["detection.wavelength"]).values(Place.of(SAMPLING), model)  # a setting, not a quantity


@pytest.mark.parametrize(
    "condition",
    [
        FieldPosition(),
        FieldPosition(axes="xy", offset=0.5, scale=0.4375, name="patch"),
        PlaneOffsets(7, wavelength=0.45, numerical_aperture=0.55, focus_offset="focus"),
        Attributes(["row"], offset=1.0, name="well"),
        Quantities(["a.b"], name="q"),
    ],
)
def test_a_condition_is_rebuilt_from_its_description_by_name(condition):
    rebuilt = resolve(describe(condition), "condition")
    assert type(rebuilt) is type(condition)
    assert (rebuilt.name, rebuilt.size, arguments(rebuilt)) == (condition.name, condition.size, arguments(condition))


def test_field_standardize_gives_each_field_zero_mean_and_unit_spread_and_has_no_inverse():
    values = 3 + 5 * torch.randn(2, 4, 16, 16, generator=torch.Generator().manual_seed(0))
    normalized = FieldStandardize()(values)
    torch.testing.assert_close(normalized.mean((-3, -2, -1)), torch.zeros(2), atol=1e-5, rtol=0)
    torch.testing.assert_close(normalized.std((-3, -2, -1), correction=0), torch.ones(2), atol=1e-5, rtol=0)
    assert (FieldStandardize.scope, FieldStandardize.invertible) == ("field", False)
    with pytest.raises(IncompatibleModelError, match="has no inverse"):
        FieldStandardize().inverse(normalized)


def test_a_robust_affine_fits_its_percentiles_and_keeps_them_through_a_rebuild():
    samples = [torch.linspace(0, 100, 101), torch.linspace(0, 100, 101).reshape(1, 101)]
    normalizer = Affine.robust(low=10, high=90)
    normalizer.fit(samples)
    assert (float(normalizer.offset), float(normalizer.scale)) == pytest.approx((10.0, 80.0))
    values = torch.tensor([10.0, 50.0, 90.0])
    torch.testing.assert_close(normalizer(values), torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(normalizer.inverse(normalizer(values)), values)
    rebuilt = resolve(describe(normalizer), "normalizer", Affine)
    rebuilt.load_state_dict(normalizer.state_dict())  # the fitted values are buffers
    torch.testing.assert_close(rebuilt(values), normalizer(values))
    with pytest.raises(ParameterError, match="give no scale"):
        Affine.robust().fit([torch.ones(10)])
    with pytest.raises(ParameterError, match="no finite value"):
        Affine.robust().fit([torch.full((4,), float("nan"))])


def test_an_affine_without_percentiles_is_what_it_was_given():
    normalizer = Affine(offset=1.0, scale=4.0)
    normalizer.fit([torch.randn(10)])  # nothing to fit
    torch.testing.assert_close(normalizer(torch.tensor([5.0])), torch.tensor([1.0]))
    assert Identity()(torch.tensor([5.0])).item() == 5.0
    with pytest.raises(ParameterError, match="scale must not be 0"):
        Affine(scale=0.0)
