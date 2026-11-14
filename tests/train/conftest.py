"""The training tests use the fixtures of the data tests (the same tiny recorded plates and their indexes), and
``no_physics``, which makes any use of an imaging model's forward model fail the test.
"""

import pytest

from tests.data.conftest import index, phase_index, phase_model, phase_pair, plates
from waveorder.nn.systems.base import ImagingSystem

__all__ = ["index", "no_physics", "phase_index", "phase_model", "phase_pair", "plates"]


@pytest.fixture
def no_physics(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make binding any imaging system to a sampling raise, in this process, and return the systems that tried.

    Every forward model ``H f``, its adjoint and every objective go through :meth:`ImagingSystem.bind`; a training
    step reads recorded targets and must never get there. A test asserts the returned list is empty, in case a caller
    swallowed the error.
    """
    tried: list[str] = []

    def bind(self: ImagingSystem, *args: object, **kwargs: object) -> None:
        del args, kwargs
        tried.append(type(self).__name__)
        raise AssertionError(f"a training step ran the forward model of {type(self).__name__}")

    monkeypatch.setattr(ImagingSystem, "bind", bind)
    return tried
