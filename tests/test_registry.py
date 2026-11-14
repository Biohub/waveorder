"""One registry names every part class; a part's constructor arguments are its plain-data form."""

import copy
import subprocess
import sys
from typing import Any

import pytest

from tests.registration import temporarily
from waveorder import registry
from waveorder.errors import ParameterError, RegistrationError
from waveorder.nn import FISTA, Detection, Free, Tikhonov, Wiener, ZernikeAberration, load_parameters
from waveorder.nn.solvers import Loop


class Shrink:
    """A part of one's own: an object prior that only scales."""

    def __init__(self, factor: float = 0.5, *extra: object, offset: float = 0.0) -> None:
        self.factor = factor
        self.extra = extra
        self.offset = offset


class Shrinker(Shrink):
    """A subclass is another part, with a name of its own."""


@pytest.fixture
def shrink():
    with temporarily("shrink", Shrink, kind="object_prior"):
        yield


@pytest.mark.usefixtures("shrink")
def test_a_part_is_resolved_by_name_by_its_arguments_or_passed_through():
    bare = registry.resolve("shrink", "object_prior")
    assert (type(bare), bare.factor, bare.offset) == (Shrink, 0.5, 0.0)
    given = registry.resolve({"shrink": {"factor": 0.25, "offset": 1.0}}, "object_prior")
    assert (given.factor, given.offset) == (0.25, 1.0)
    assert registry.resolve({"shrink": None}, "object_prior").factor == 0.5
    assert registry.resolve(given, "object_prior") is given  # an object is already a part
    assert registry.describe(given) == {"shrink": {"factor": 0.25, "offset": 1.0}}
    assert "shrink" in registry.names("object_prior")


@pytest.mark.usefixtures("shrink")
@pytest.mark.parametrize(
    ("spec", "kind", "message"),
    [
        ("nope", "object_prior", "no object_prior is named 'nope'; choose one of"),
        ("shrink", "solver", "no solver is named 'shrink'"),
        ({"shrink": {"weight": 1.0}}, "object_prior", "the object_prior 'shrink' does not take these arguments"),
        ({"shrink": {}, "tikhonov": {}}, "object_prior", "is a name or one {name: arguments}, got 2 keys"),
    ],
    ids=["unknown", "other_kind", "bad_argument", "two_parts"],
)
def test_a_spec_that_names_no_part_of_the_kind_is_refused(spec, kind, message):
    with pytest.raises(ParameterError, match=message):
        registry.resolve(spec, kind)


@pytest.mark.usefixtures("shrink")
def test_a_name_is_taken_once():
    with pytest.raises(RegistrationError, match="is taken by Shrink"):
        registry.register("shrink", kind="object_prior")(Shrinker)


@pytest.mark.usefixtures("shrink")
def test_only_the_exact_registered_class_is_described():
    assert registry.describe(Shrink(0.25, "dropped", offset=2.0)) == {"shrink": {"factor": 0.25, "offset": 2.0}}
    assert registry.describe(Shrink()) == {"shrink": {"factor": 0.5, "offset": 0.0}}  # the captured defaults
    with pytest.raises(ParameterError, match="Shrinker is not registered, so it has no plain-data form"):
        registry.describe(Shrinker())

    class Unregistered(Shrink):
        pass

    with pytest.raises(ParameterError, match="Unregistered is not a registered part"):
        registry.arguments(Unregistered())  # a subclass instance keeps no arguments
    with pytest.raises(ParameterError, match="int is not a registered part"):
        registry.arguments(3)  # no __dict__ at all


def test_keyword_arguments_are_reported_by_their_own_names():
    aberrations = ZernikeAberration(terms={"defocus": 0.1})
    described = registry.describe(Detection(wavelength=0.52, numerical_aperture=1.2, pupil_aberrations=aberrations))
    assert described["detection"]["pupil_aberrations"] == registry.describe(aberrations)
    rebuilt = registry.resolve(described, "detection")
    assert isinstance(rebuilt, Detection)
    assert registry.describe(rebuilt) == described


def test_a_quantity_is_reported_at_its_current_value_and_a_copy_keeps_its_arguments():
    detection = Detection(wavelength=0.52, numerical_aperture=Free(1.2, bounds=(1.0, 1.3)))
    load_parameters(detection, {"numerical_aperture": 1.1})
    described = registry.describe(detection)
    quantity: Any = described["detection"]["numerical_aperture"]
    free = quantity["free"]
    assert (free["nominal"], free["tolerance"], free["bounds"]) == (pytest.approx(1.1), None, [1.0, 1.3])
    assert registry.describe(copy.deepcopy(detection)) == described
    rebuilt = registry.resolve(described, "detection")
    assert registry.describe(rebuilt) == described


def test_nested_specs_and_values_are_rebuilt_before_the_part():
    solver = registry.resolve({"fista": {"loop": {"loop": {"iterations": 3, "tolerance": 1e-3}}}}, "solver")
    assert isinstance(solver, FISTA)
    assert solver.loop == Loop(3, tolerance=1e-3)
    prior = registry.resolve(
        {"tikhonov": {"weight": {"free": {"nominal": 0.01, "bounds": [1e-4, 1.0]}}}}, "object_prior"
    )
    assert isinstance(prior, Tikhonov)
    assert registry.describe(prior)["tikhonov"]["weight"] == {
        "free": {"nominal": 0.01, "tolerance": None, "bounds": [1e-4, 1.0]}
    }
    assert registry.describe(Wiener()) == {"wiener": {}}


def test_the_registry_imports_no_torch():
    code = "import sys, waveorder.registry; assert 'torch' not in sys.modules, 'torch was imported'"
    subprocess.run([sys.executable, "-c", code], check=True)
