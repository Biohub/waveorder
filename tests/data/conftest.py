"""Fixtures of the ``waveorder.data`` and ``waveorder.train`` tests: tiny recorded pairs of plates and their indexes."""

import pytest

from tests.data.plates import Plates, phase_plates, recorded_plates, thin_phase_model
from waveorder.data import FovIndex, GroupSplit, index_from_plates
from waveorder.models import Phase2D

SPLIT = GroupSplit(by="position", fractions={"train": 0.5, "val": 0.25, "test": 0.25})
"""Per well of four positions: two train, one val, one test."""


@pytest.fixture(scope="session")
def plates(tmp_path_factory: pytest.TempPathFactory) -> Plates:
    """A raw plate and its target plate with the same eight positions (wells ``A/1`` and ``A/2``, four each).

    Input ``(1, 2, 5, 64, 64)`` uint16, channels ``GFP`` and ``BF``; target ``(1, 2, 1, 64, 64)`` float32, channels
    ``Phase2D`` and ``Focus3D``; voxel size ``(2.0, 0.65, 0.65)`` um. Read-only: tests never write into it.
    """
    return recorded_plates(tmp_path_factory.mktemp("plates"))


@pytest.fixture(scope="session")
def index(plates: Plates) -> FovIndex:
    """The index of ``plates``: both channels and both target channels, split by position .5 / .25 / .25."""
    return index_from_plates(
        [plates.inputs],
        channels=plates.channels,
        targets=[plates.targets],
        target_channels=plates.target_channels,
        split=SPLIT,
    )


@pytest.fixture(scope="session")
def phase_model() -> Phase2D:
    """The tiny ``Phase2D`` that made ``phase_pair``'s reconstructed plate (from ``thin_phase_model``)."""
    return thin_phase_model()


@pytest.fixture(scope="session")
def phase_pair(tmp_path_factory: pytest.TempPathFactory) -> Plates:
    """Brightfield stacks and their ``Phase2D`` reconstruction by ``phase_model``: the recorded-pairs setting.

    Eight positions (wells ``A/1``, ``A/2``); input ``(1, 1, 5, 64, 64)`` uint16, channel ``BF``; target
    ``(1, 1, 1, 64, 64)`` float32, channel ``Phase2D`` (the phase field of the estimate); voxel size
    ``(0.4, 0.15, 0.15)`` um.
    """
    return phase_plates(tmp_path_factory.mktemp("phase"))


@pytest.fixture(scope="session")
def phase_index(phase_pair: Plates) -> FovIndex:
    """The index of ``phase_pair``, split by position .5 / .25 / .25 (4 train, 2 val, 2 test)."""
    return index_from_plates(
        [phase_pair.inputs],
        channels=phase_pair.channels,
        targets=[phase_pair.targets],
        target_channels=phase_pair.target_channels,
        split=SPLIT,
    )
