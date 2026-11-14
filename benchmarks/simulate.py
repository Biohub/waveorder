"""Forward simulation of the synthetic cases through the waveorder 4 imaging systems.

The systems use ``edge="sigmoid"``, the 3.x pupil that ``wo config migrate`` writes into the case configurations, so
the simulated data are what the reconstruction's own model predicts. On the regression grids their transfer
functions equal wo-4.0's ``phase_thick_3d`` and ``isotropic_fluorescent_thick_3d`` ones to a relative 1e-5.
"""

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from benchmarks.phantoms import Phantom
from waveorder.nn import (
    Detection,
    Illumination,
    Quantity,
    Sampling,
    ThickPhase3D,
    WidefieldFluorescence3D,
    ZernikeAberration,
)


def _sampling(phantom: Phantom, shape: torch.Size) -> Sampling:
    return Sampling(shape=tuple(shape), pixel_size=tuple(float(p) for p in phantom.pixel_sizes))


def simulate_phase_3d(
    phantom: Phantom,
    *,
    wavelength_illumination: float = 0.532,
    index_of_refraction_media: float = 1.3,
    numerical_aperture_illumination: float = 0.9,
    numerical_aperture_detection: float = 1.2,
    brightness: float = 1e3,
) -> Tensor:
    """Simulate a brightfield stack of the phantom's ``phase`` (an index difference) with ``ThickPhase3D``.

    The index difference ``dn`` becomes the object ``dn * dz / (wavelength / n_media)``, the quantity the 3.x
    benchmark called the phase per voxel; ``ThickPhase3D``'s transfer function is the 3.x one, so the same number goes
    in as 3.x simulated. The stack is ``brightness * (1 + contrast)``.

    Parameters
    ----------
    phantom : Phantom
        Ground truth; its ``phase`` is the index difference.
    wavelength_illumination : float
        Illumination wavelength in um.
    index_of_refraction_media : float
        Refractive index of the medium.
    numerical_aperture_illumination : float
        Condenser numerical aperture.
    numerical_aperture_detection : float
        Objective numerical aperture.
    brightness : float
        Mean intensity of the stack.

    Returns
    -------
    Tensor
        Simulated stack, ``(Z, Y, X)``.
    """
    z_pixel_size = phantom.pixel_sizes[0]
    obj = phantom.phase * z_pixel_size / (wavelength_illumination / index_of_refraction_media)
    system = ThickPhase3D(
        Illumination(wavelength=wavelength_illumination, numerical_aperture=numerical_aperture_illumination),
        Detection(numerical_aperture=numerical_aperture_detection, edge="sigmoid"),
        index_of_refraction_media=index_of_refraction_media,
    )
    with torch.no_grad():
        contrast = system(obj[None], _sampling(phantom, obj.shape))[0]
    return contrast * brightness + brightness


def fluorescence_system(
    *,
    wavelength_emission: float,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
    aberrations: Mapping[str, Quantity] | None = None,
) -> WidefieldFluorescence3D:
    """Return the widefield system of the cases, with the 3.x pupil edge and optional Zernike terms (waves)."""
    factors = {} if aberrations is None else {"pupil_aberrations": ZernikeAberration(aberrations)}
    detection = Detection(
        wavelength=wavelength_emission, numerical_aperture=numerical_aperture_detection, edge="sigmoid", **factors
    )
    return WidefieldFluorescence3D(detection, index_of_refraction_media=index_of_refraction_media)


def simulate_fluorescence_3d(
    phantom: Phantom,
    *,
    wavelength_emission: float = 0.532,
    index_of_refraction_media: float = 1.3,
    numerical_aperture_detection: float = 1.2,
    background: float = 10,
) -> Tensor:
    """Simulate a widefield fluorescence stack of the phantom's ``fluorescence`` with ``WidefieldFluorescence3D``.

    Parameters
    ----------
    phantom : Phantom
        Ground truth; uses ``phantom.fluorescence``.
    wavelength_emission : float
        Emission wavelength in um.
    index_of_refraction_media : float
        Refractive index of the medium.
    numerical_aperture_detection : float
        Objective numerical aperture.
    background : float
        Constant added to the stack.

    Returns
    -------
    Tensor
        Simulated stack, ``(Z, Y, X)``.
    """
    obj = phantom.fluorescence
    system = fluorescence_system(
        wavelength_emission=wavelength_emission,
        index_of_refraction_media=index_of_refraction_media,
        numerical_aperture_detection=numerical_aperture_detection,
    )
    with torch.no_grad():
        return system(obj[None], _sampling(phantom, obj.shape))[0] + background


def field_coefficients(
    spatial_pupil_coefficients: dict[tuple[int, int, int], float], *, y_norm: float, x_norm: float
) -> dict[int, float]:
    """Evaluate ``c_j = sum_mn c_jmn x^m y^n`` at a field position normalized by the half field (3.x convention).

    Parameters
    ----------
    spatial_pupil_coefficients : dict
        ``{(j, m, n): c}``, Noll index ``j`` and powers of ``x`` (``m``) and ``y`` (``n``), waves RMS.
    y_norm, x_norm : float
        Field position over the half field, ``[-1, 1]`` inside the field.

    Returns
    -------
    dict
        Coefficient by Noll index, waves RMS.
    """
    out: dict[int, float] = {}
    for (j, m, n), c in spatial_pupil_coefficients.items():
        if c != 0:
            out[j] = out.get(j, 0.0) + c * (x_norm**m) * (y_norm**n)
    return out


def partition_of_unity_weights(
    yx_shape: tuple[int, int], yx_pixel_size: float, n_tiles_yx: tuple[int, int]
) -> tuple[Tensor, list[tuple[float, float]]]:
    """Return raised-cosine windows over an ``n_tiles_yx`` lattice that sum to 1 at every pixel (3.x simulator).

    Copied from wo-4.0 ``make_partition_of_unity_weights`` (overlap fraction 0.5): tile centres on a uniform lattice
    inside the field, each window ``cos^2`` with a full width of two tile spacings.

    Returns
    -------
    weights : Tensor
        ``(T, Y, X)``.
    centers_um : list of (float, float)
        ``(y, x)`` of each tile centre from the field centre, um.
    """
    ny_tiles, nx_tiles = n_tiles_yx
    ny, nx = yx_shape
    y_extent_um, x_extent_um = ny * yx_pixel_size, nx * yx_pixel_size
    y_centers_um = torch.linspace(-y_extent_um / 2, y_extent_um / 2, ny_tiles + 2)[1:-1]
    x_centers_um = torch.linspace(-x_extent_um / 2, x_extent_um / 2, nx_tiles + 2)[1:-1]
    width_y_um = y_extent_um / (ny_tiles + 1) / 0.5
    width_x_um = x_extent_um / (nx_tiles + 1) / 0.5
    y_um = (torch.arange(ny) - ny // 2 + 0.5) * yx_pixel_size
    x_um = (torch.arange(nx) - nx // 2 + 0.5) * yx_pixel_size
    weights, centers_um = [], []
    for yc in y_centers_um:
        for xc in x_centers_um:
            wy = torch.cos(math.pi * (y_um - yc) / width_y_um).clamp(min=0) ** 2
            wx = torch.cos(math.pi * (x_um - xc) / width_x_um).clamp(min=0) ** 2
            weights.append(wy[:, None] * wx[None, :])
            centers_um.append((float(yc), float(xc)))
    w = torch.stack(weights, dim=0)
    return w / w.sum(dim=0, keepdim=True).clamp(min=1e-12), centers_um


def simulate_shift_variant_fluorescence_3d(  # noqa: PLR0913 (the object, its field polynomial and the optics)
    phantom: Phantom,
    spatial_pupil_coefficients: dict[tuple[int, int, int], float] | dict[str, float],
    *,
    n_tiles_yx: tuple[int, int] = (8, 8),
    wavelength_emission: float = 0.532,
    index_of_refraction_media: float = 1.3,
    numerical_aperture_detection: float = 1.2,
    background: float = 0.0,
) -> Tensor:
    """Simulate a fluorescence stack whose pupil aberrations vary smoothly over the field.

    The field is split into ``n_tiles_yx`` raised-cosine windows (:func:`partition_of_unity_weights`); each windowed
    object is imaged by a ``WidefieldFluorescence3D`` whose ``ZernikeAberration`` holds the coefficients at the
    window's centre (:func:`field_coefficients`), and the images are summed.

    Parameters
    ----------
    phantom : Phantom
        Ground truth; uses ``phantom.fluorescence``.
    spatial_pupil_coefficients : dict
        ``{(j, m, n): c}`` or ``{"j_m_n": c}``, waves RMS; string keys come from YAML.
    n_tiles_yx : tuple[int, int]
        Windows along y and x.
    wavelength_emission : float
        Emission wavelength in um.
    index_of_refraction_media : float
        Refractive index of the medium.
    numerical_aperture_detection : float
        Objective numerical aperture.
    background : float
        Constant added to the stack.

    Returns
    -------
    Tensor
        Simulated stack, ``(Z, Y, X)``.
    """
    obj = phantom.fluorescence
    _, ny, nx = obj.shape
    yx_pixel_size = phantom.pixel_sizes[1]
    half_y, half_x = ny * yx_pixel_size / 2, nx * yx_pixel_size / 2
    coefs = normalize_pupil_coefficients(spatial_pupil_coefficients)
    weights, centers_um = partition_of_unity_weights((ny, nx), yx_pixel_size, tuple(n_tiles_yx))
    sampling = _sampling(phantom, obj.shape)
    out = torch.zeros_like(obj)
    with torch.no_grad():
        for weight, (yc, xc) in zip(weights, centers_um, strict=True):
            local = field_coefficients(coefs, y_norm=yc / half_y, x_norm=xc / half_x)
            system = fluorescence_system(
                wavelength_emission=wavelength_emission,
                index_of_refraction_media=index_of_refraction_media,
                numerical_aperture_detection=numerical_aperture_detection,
                aberrations={f"z{j}": c for j, c in local.items()},
            )
            out = out + system((obj * weight)[None], sampling)[0]
    return out + background


def normalize_pupil_coefficients(raw: dict) -> dict[tuple[int, int, int], float]:
    """Normalize ``{"j_m_n": c}`` or ``{(j, m, n): c}`` into tuple keys (YAML writes tuple keys as strings)."""
    out: dict[tuple[int, int, int], float] = {}
    for key, value in raw.items():
        if isinstance(key, tuple):
            j, m, n = key
        else:
            parts = str(key).split("_")
            if len(parts) != 3:
                raise ValueError(f"Pupil coefficient key {key!r} must be 'j_m_n'")
            j, m, n = (int(p) for p in parts)
        out[(int(j), int(m), int(n))] = float(value)
    return out
