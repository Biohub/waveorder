"""wo-4.0's eight Zernike names map to their Noll indices; every other term is ``z<j>``."""

import pytest

from waveorder.zernike_names import (
    NAMED_TERMS,
    canonical_name,
    name_to_noll,
    noll_indices_up_to_order,
    noll_to_name,
)

NAMES = {
    4: "defocus",
    5: "astigmatism_45",
    6: "astigmatism_0",
    7: "coma_y",
    8: "coma_x",
    9: "trefoil_y",
    10: "trefoil_x",
    11: "spherical",
}


def test_named_terms_are_the_eight_wo40_names():
    assert {name: j for j, name in NAMES.items()} == NAMED_TERMS


@pytest.mark.parametrize(("j", "name"), NAMES.items())
def test_every_name_round_trips(j, name):
    assert noll_to_name(j) == name
    assert name_to_noll(name) == j
    assert canonical_name(name) == name


@pytest.mark.parametrize("j", [1, 2, 3, 12, 15, 22, 500])
def test_other_noll_indices_are_z_j(j):
    assert noll_to_name(j) == f"z{j}"
    assert name_to_noll(f"z{j}") == j
    assert canonical_name(f"z{j}") == f"z{j}"


def test_a_noll_alias_of_a_named_term_canonicalizes_to_its_name():
    assert canonical_name("z11") == "spherical"
    assert canonical_name("z4") == "defocus"


@pytest.mark.parametrize("bad", ["comatose", "piston", "secondary_astigmatism_0", "quadrafoil_0", "z0", "zx", "Z4", ""])
def test_unknown_names_rejected(bad):
    with pytest.raises(ValueError, match=r"Unknown Zernike term|Noll index must be >= 1"):
        canonical_name(bad)


def test_up_to_order():
    assert noll_indices_up_to_order(2) == [4, 5, 6]
    assert noll_indices_up_to_order(3) == [4, 5, 6, 7, 8, 9, 10]
    assert noll_indices_up_to_order(4) == list(range(4, 16))
    with pytest.raises(ValueError, match="order must be >= 2"):
        noll_indices_up_to_order(1)
