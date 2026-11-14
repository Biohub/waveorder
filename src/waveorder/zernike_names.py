"""Named Zernike terms and their Noll indices.

Indexing is Noll (``j = 1`` is piston) with the orthonormal normalization, so a coefficient is the RMS wavefront error
of its term. Coefficients are in waves of pupil phase, ``P0 * exp(i 2 pi sum_j c_j Z_j)``; a coefficient in radians is
``2 pi`` times one in waves, so convert at the boundary. The azimuth ``theta = atan2(f_y, f_x)`` is zero along +x.
Named terms: ``defocus`` (Noll 4), ``astigmatism_45`` (5, ``sin 2theta``), ``astigmatism_0`` (6, ``cos 2theta``),
``coma_y`` (7, ``sin theta``), ``coma_x`` (8, ``cos theta``), ``trefoil_y`` (9, ``sin 3theta``), ``trefoil_x`` (10,
``cos 3theta``), ``spherical`` (11). Any term may be written ``z<j>``; :func:`canonical_name` maps it to its name.
"""

import re


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert a Noll index ``j`` (1-based) to its ``(n, m)`` radial order and azimuthal frequency.

    Parameters
    ----------
    j : int
        Noll index, ``j >= 1``. ``j = 1`` is piston.

    Returns
    -------
    tuple[int, int]
        ``(n, m)`` with ``n >= 0``, ``-n <= m <= n`` and ``(n - |m|) % 2 == 0``. Even ``j`` gives ``m >= 0``
        (a ``cos(m theta)`` term), odd ``j`` gives ``m <= 0`` (a ``sin(|m| theta)`` term).
    """
    if j < 1:
        raise ValueError(f"Noll index j must be >= 1, got {j}")
    n = 0
    cumulative = 0
    while cumulative + (n + 1) < j:
        n += 1
        cumulative += n
    k = j - cumulative - 1
    if n % 2 == 0:
        ms = [0] + [v for pair in zip(range(2, n + 1, 2), range(-2, -n - 1, -2), strict=False) for v in pair]
    else:
        ms = [v for pair in zip(range(1, n + 1, 2), range(-1, -n - 1, -2), strict=False) for v in pair]
    m = ms[k]
    if j % 2 == 0 and m < 0:
        m = -m
    if j % 2 == 1 and m > 0:
        m = -m
    if j == 1:
        m = 0
    return n, m


NAMED_TERMS: dict[str, int] = {
    "defocus": 4,
    "astigmatism_45": 5,
    "astigmatism_0": 6,
    "coma_y": 7,
    "coma_x": 8,
    "trefoil_y": 9,
    "trefoil_x": 10,
    "spherical": 11,
}
"""Every named term and its Noll index."""

_NOLL_TO_NAME: dict[int, str] = {j: name for name, j in NAMED_TERMS.items()}

_ZN_PATTERN = re.compile(r"^z([0-9]+)$")


def name_to_noll(name: str) -> int:
    """Convert a Zernike term name to its Noll index.

    Parameters
    ----------
    name : str
        A named term (``defocus``, ``astigmatism_0``, ...) or ``z<j>`` by Noll index.

    Returns
    -------
    int
        Noll index (1-based).

    Raises
    ------
    ValueError
        If the name is neither a named term nor ``z<j>`` with ``j >= 1``.
    """
    if name in NAMED_TERMS:
        return NAMED_TERMS[name]
    match = _ZN_PATTERN.match(name)
    if match:
        j = int(match.group(1))
        if j < 1:
            raise ValueError(f"Noll index must be >= 1, got {name!r}")
        return j
    raise ValueError(f"Unknown Zernike term {name!r}. Use one of {sorted(NAMED_TERMS)} or 'z<j>' with a Noll index j.")


def noll_to_name(j: int) -> str:
    """Return the canonical name of a Noll index: the named term if it has one, else ``z<j>``."""
    return _NOLL_TO_NAME.get(j, f"z{j}")


def canonical_name(name: str) -> str:
    """Return the canonical spelling of a Zernike term name, so ``z11`` and ``spherical`` agree.

    Raises
    ------
    ValueError
        If the name is neither a named term nor ``z<j>``.
    """
    return noll_to_name(name_to_noll(name))


def noll_indices_up_to_order(order: int) -> list[int]:
    """Noll indices of every term through radial order ``order``, as ``zernike_up_to_order: {order: n}`` frees them.

    Piston, tip and tilt (Noll 1-3) are excluded: piston does not change an intensity PSF, and tip and tilt only
    translate the object.

    Parameters
    ----------
    order : int
        Highest radial order ``n`` to include, ``order >= 2``.

    Returns
    -------
    list[int]
        Sorted Noll indices with ``4 <= j`` and radial order ``<= order``.
    """
    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    indices = []
    j = 4
    while True:
        n, _ = noll_to_nm(j)
        if n > order:
            break
        indices.append(j)
        j += 1
    return indices
