"""Stateless wave-optics math on tensors; the modules in :mod:`waveorder.nn` hold the parameters and call these.

Frequencies are in cycles per um and full spectra are in FFT order (zero frequency at index 0). Imaging is circular
convolution over the sampled volume. A half spectrum is what ``torch.fft.rfftn`` returns, ``X' = X // 2 + 1`` columns.
Complex values are stored real with a trailing axis of size 2 (``torch.view_as_real``): spectra and vectors are
``(n, Z, Y, X', 2)``; per-frequency matrices are ``(n, m, Z, Y, X', 2)`` when complex or ``(n, m, Z, Y, X')`` when
real. Hermitian matrices of size ``n > 1`` are factored through their real embedding ``[[Re, -Im], [Im, Re]]``.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from waveorder.errors import ShapeError
from waveorder.nn.sampling import ZYX, Sampling
from waveorder.zernike_names import noll_to_nm

SPATIAL_DIMS = (-3, -2, -1)
"""The z, y and x axes of a volume ``(..., Z, Y, X)``."""
_COMPLEX_MATRIX_DIMS = 6
"""A per-frequency matrix ``(n, m, Z, Y, X')`` is real; with a trailing real/imaginary axis it is complex."""


def frequencies(sampling: Sampling, *, device: torch.device, dtype: torch.dtype) -> ZYX[Tensor]:
    """Return the 1D spatial frequencies along z, y and x of ``sampling``, in FFT order."""
    fy, fx = lateral_frequencies(sampling, device=device, dtype=dtype)
    fz = torch.fft.fftfreq(sampling.shape[0], d=sampling.pixel_size[0], device=device, dtype=dtype)
    return (fz, fy, fx)


def lateral_frequencies(sampling: Sampling, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Return the 1D spatial frequencies along y and x of ``sampling``, in FFT order."""
    (_, ny, nx), (_, dy, dx) = sampling.shape, sampling.pixel_size
    return (
        torch.fft.fftfreq(ny, d=dy, device=device, dtype=dtype),
        torch.fft.fftfreq(nx, d=dx, device=device, dtype=dtype),
    )


def half_spectrum_frequencies(
    sampling: Sampling, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None
) -> ZYX[Tensor]:
    """Return the spatial frequencies of the half spectrum of ``sampling``, shaped to broadcast against it.

    Parameters
    ----------
    sampling : Sampling
        The data's grid.
    device : torch.device or str, optional
        Device of the result; the CPU by default.
    dtype : torch.dtype, optional
        Real dtype of the result; torch's default dtype by default.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(fz, fy, fx)`` in cycles per um, shapes ``(Z, 1, 1)``, ``(1, Y, 1)`` and ``(1, 1, X // 2 + 1)``; FFT order
        along z and y, ``0 .. X // 2`` along x as ``torch.fft.rfftfreq`` gives them.
    """
    (nz, ny, nx), (dz, dy, dx) = sampling.shape, sampling.pixel_size
    return (
        torch.fft.fftfreq(nz, d=dz, device=device, dtype=dtype)[:, None, None],
        torch.fft.fftfreq(ny, d=dy, device=device, dtype=dtype)[None, :, None],
        torch.fft.rfftfreq(nx, d=dx, device=device, dtype=dtype)[None, None, :],
    )


def transverse_nyquist(wavelength: float, numerical_aperture_sum: float) -> float:
    """Return the largest lateral pixel size, in um, that samples a transfer function without aliasing.

    ``numerical_aperture_sum`` is the illumination plus detection NA, ``2 * NA`` for incoherent fluorescence.
    """
    return wavelength / (2 * numerical_aperture_sum)


def axial_nyquist(wavelength: float, numerical_aperture: float, index_of_refraction: float) -> float:
    """Return the largest axial pixel size, in um, that samples a widefield transfer function without aliasing."""
    n_over_lambda = index_of_refraction / wavelength
    cutoff = n_over_lambda - math.sqrt(n_over_lambda**2 - (numerical_aperture / wavelength) ** 2)
    return 1 / (2 * cutoff)


def zernike(j: int, rho: Tensor, theta: Tensor) -> Tensor:
    """Evaluate the orthonormal Noll Zernike polynomial ``Z_j`` at unit-disk polar coordinates.

    Orthonormal means the mean of ``Z_j ** 2`` over the unit disk is 1, so a coefficient is the RMS wavefront error
    of its term. ``theta = 0`` points along +x.
    """
    n, m = noll_to_nm(j)
    radial = _radial_polynomial(n, abs(m), rho)
    if m == 0:
        return math.sqrt(n + 1) * radial
    angular = torch.cos(m * theta) if m > 0 else torch.sin(-m * theta)
    return math.sqrt(2 * (n + 1)) * radial * angular


def _radial_polynomial(n: int, m: int, rho: Tensor) -> Tensor:
    """Evaluate the Zernike radial polynomial ``R_n^m(rho)`` for ``m >= 0``."""
    result = torch.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coefficient = (-1) ** k * math.comb(n - k, k) * math.comb(n - 2 * k, (n - m) // 2 - k)
        result = result + coefficient * rho ** (n - 2 * k)
    return result


def to_half_spectrum(volume: Tensor, *, norm: str | None = None) -> Tensor:
    """Return the ``rfftn`` of real volumes ``(..., Z, Y, X)`` over the last three axes, stored real.

    Parameters
    ----------
    volume : Tensor
        Real volumes, shape ``(..., Z, Y, X)``.
    norm : {None, "backward", "ortho", "forward"}
        As in :func:`torch.fft.rfftn`.

    Returns
    -------
    Tensor
        Real and imaginary parts, shape ``(..., Z, Y, X // 2 + 1, 2)``: a view of the complex FFT output.
    """
    return torch.view_as_real(torch.fft.rfftn(volume, dim=SPATIAL_DIMS, norm=norm))


def from_half_spectrum(spectrum: Tensor, shape: Sequence[int], *, norm: str | None = None) -> Tensor:
    """Return the real volumes ``(..., Z, Y, X)`` of half spectra ``(..., Z, Y, X // 2 + 1, 2)``: the ``irfftn``.

    ``shape`` is the volume shape ``(Z, Y, X)``, which fixes the parity of ``X``; ``norm`` is as in
    :func:`torch.fft.irfftn`.
    """
    return irfftn(_as_complex(spectrum), shape, norm=norm)


def irfftn(spectrum: Tensor, shape: Sequence[int], *, norm: str | None = None) -> Tensor:
    """Return :func:`torch.fft.irfftn` of complex half spectra over their last ``len(shape)`` axes, ``s=shape``.

    On the CPU it runs as one-dimensional transforms, which give the same values: the MKL build torch ships (2024.2)
    writes outside its buffers in a multi-dimensional ``irfftn`` on Intel CPUs with two or more threads.
    """
    count = len(shape)
    if spectrum.device.type != "cpu":
        return torch.fft.irfftn(spectrum, s=shape, dim=tuple(range(-count, 0)), norm=norm)
    for axis, size in zip(range(-count, -1), shape[:-1], strict=True):
        spectrum = torch.fft.ifft(spectrum, n=size, dim=axis, norm=norm)
    return torch.fft.irfft(spectrum, n=shape[-1], norm=norm)


def _as_complex(values: Tensor) -> Tensor:
    # view_as_complex needs a unit-stride last axis and even other strides, which a contiguous tensor has;
    # contiguous() returns the tensor itself when it already is.
    return torch.view_as_complex(values.contiguous())


def complex_multiply(first: Tensor, second: Tensor) -> Tensor:
    """Return the elementwise complex product of broadcastable tensors stored as ``(..., 2)``."""
    return torch.view_as_real(_as_complex(first) * _as_complex(second))


def abs2(values: Tensor) -> Tensor:
    """Return ``|z|^2 = re^2 + im^2`` of complex values stored as ``(..., 2)``: shape ``(...)``, no square root."""
    return values.square().sum(dim=-1)


def forward_differences(obj: Tensor) -> Tensor:
    """Return the forward differences with periodic boundaries along z, y, x: shape ``(3, *obj.shape)``."""
    return torch.stack([torch.roll(obj, -1, dims=dim) - obj for dim in SPATIAL_DIMS])


def forward_differences_adjoint(field: Tensor) -> Tensor:
    """Return the adjoint of :func:`forward_differences`: ``sum_a roll(p_a, +1, a) - p_a``."""
    return torch.stack([torch.roll(field[i], 1, dims=dim) - field[i] for i, dim in enumerate(SPATIAL_DIMS)]).sum(0)


_SLAB_ELEMENTS = 2**27
"""Elements of the oversampled lateral grid per slab of planes in the transfer-function builds.

cuFFT refuses one batched 2D transform of 2**31 elements or more; at 2**27 a complex64 slab is 1 GiB.
"""


def plane_slabs(values: Tensor, plane_elements: int) -> tuple[Tensor, ...]:
    """Split ``values`` along its first axis into slabs of planes, at most ``_SLAB_ELEMENTS`` elements per slab.

    ``plane_elements`` is the size of one plane of the oversampled lateral grid; a slab holds one plane at least.
    """
    return values.split(max(1, _SLAB_ELEMENTS // plane_elements))


def crop_full_lateral(spectrum: Tensor, shape: ZYX[int]) -> Tensor:
    """Keep the lateral frequencies of a ``shape`` grid, both signs of x, from finer full spectra ``(..., Y', X', 2)``.

    Along y and x it selects the lowest frequencies in FFT order, as :func:`crop_half_spectrum` does along z and y.

    Raises
    ------
    ShapeError
        If ``shape`` is finer than the spectrum's grid.
    """
    return _keep_lowest(spectrum, ((-3, shape[1]), (-2, shape[2])))


def axial_spectrum(slabs: Sequence[Tensor], shape: ZYX[int]) -> Tensor:
    """Return the FFT along z of a stack given as slabs of planes, cropped to the z frequencies of ``shape``.

    Parameters
    ----------
    slabs : sequence of Tensor
        Consecutive slabs ``(..., P, Y, X, 2)`` of a stack of full lateral spectra, its planes in FFT order, already
        cropped to the lateral frequencies of ``shape`` (:func:`crop_full_lateral`).
    shape : tuple[int, int, int]
        The coarser grid ``(Z, Y, X)``.

    Returns
    -------
    Tensor
        Full spectra stored real, shape ``(..., Z, Y, X, 2)``.
    """
    # z goes last before the transform: a transform over a leading axis would copy the complex stack.
    stack = torch.cat([slab.movedim(-4, -2) for slab in slabs], dim=-2)
    spectrum = torch.view_as_real(torch.fft.fft(torch.view_as_complex(stack), dim=-1))
    del stack  # the transform's backward does not keep its input
    return _keep_lowest(spectrum.movedim(-2, -4), ((-4, shape[0]),))


def half_of_hermitian_part(spectrum: Tensor, *, axes: int = 3) -> Tensor:
    """Return the half spectrum of ``0.5 (S(k) + conj S(-k))``, from full spectra ``S`` stored ``(..., X, 2)``.

    On real ``f``, ``real(ifftn(S fftn(f)))`` (the operator 3.x applied with a cropped, non-Hermitian ``S``) equals
    ``irfftn`` of this half times ``rfftn(f)``. ``-k`` is taken on the grid of ``S``, where the Nyquist frequency of
    an even axis is its own negative.

    Parameters
    ----------
    spectrum : Tensor
        Full spectra over the last ``axes`` axes of the complex view, FFT order, e.g. ``(..., Z, Y, X, 2)``.
    axes : int
        The number of transformed axes, ``x`` last: 3 for volumes, 2 for planes.

    Returns
    -------
    Tensor
        Half spectra stored real, ``(..., X // 2 + 1, 2)``, contiguous: already the Hermitian projection of
        :func:`hermitian_half`.
    """
    nx = spectrum.shape[-2]
    others = tuple(range(-1 - axes, -2))  # the transformed axes before x, in the stored view
    columns = torch.arange(nx // 2 + 1, device=spectrum.device).neg().remainder(nx)  # -x on the grid
    mirrored = spectrum.index_select(-2, columns)
    if others:
        mirrored = torch.roll(torch.flip(mirrored, dims=others), shifts=(1,) * len(others), dims=others)
    real, imaginary = mirrored.unbind(-1)
    return (0.5 * (spectrum.narrow(-2, 0, nx // 2 + 1) + torch.stack((real, -imaginary), dim=-1))).contiguous()


def defocused_intensity(pupil: Tensor, kz: Tensor, z: Tensor) -> Tensor:
    """Return ``|ifft2(pupil exp(i 2 pi z k_z))|^2`` for the planes ``z``, shape ``(Z, Y, X)``."""
    phase = 2 * math.pi * kz * z[:, None, None]
    field = complex_multiply(pupil, torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1))
    return abs2(torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(field))))


def intensity_otf(
    planes: Callable[[Tensor], Tensor], z: Tensor, *, lateral: tuple[int, int], shape: ZYX[int]
) -> Tensor:
    """Return the transfer function of a real point-spread function built plane by plane, on a coarser grid.

    The planes are built, transformed laterally and cropped one slab at a time (:func:`plane_slabs`) to bound memory.
    The result is normalized to 1 at zero frequency. The crop ``H_c`` of an oversampled grid need not be Hermitian on
    the Nyquist frequencies of an even axis (an aberrated or shifted point-spread function); the half spectrum kept is
    that of its Hermitian part (:func:`half_of_hermitian_part`), whose inverse transform is ``real(ifftn(H_c))``.

    Parameters
    ----------
    planes : callable
        Maps plane positions ``(P,)`` to the real point-spread function on those planes, ``(P, Y', X')``.
    z : Tensor
        Position of each plane, shape ``(Z',)``, FFT order (index 0 at the origin).
    lateral : tuple[int, int]
        The fine lateral grid ``(Y', X')``.
    shape : tuple[int, int, int]
        The coarser grid ``(Z, Y, X)`` whose frequencies are kept.

    Returns
    -------
    Tensor
        The half of the Hermitian part of the cropped ``fftn`` of the point-spread function over its sum, shape
        ``(Z, Y, X // 2 + 1, 2)``.
    """
    spectra, totals = [], []
    for part in plane_slabs(z, lateral[0] * lateral[1]):
        values = planes(part)
        totals.append(values.sum())
        spectra.append(crop_full_lateral(torch.view_as_real(torch.fft.fft2(values)), shape))
    # The zero-frequency term of fftn(I) is sum(I).
    return half_of_hermitian_part(axial_spectrum(spectra, shape)) / torch.stack(totals).sum()


def widefield_otf(pupil: Tensor, kz: Tensor, z: Tensor, *, shape: ZYX[int]) -> Tensor:
    """Return the 3D incoherent transfer function of a pupil on a coarser grid, normalized to 1 at zero frequency.

    Parameters
    ----------
    pupil : Tensor
        Complex pupil stored as real and imaginary parts, shape ``(Y', X', 2)``, FFT order.
    kz : Tensor
        Axial spatial frequency of each pupil point, cycles per um, shape ``(Y', X')``.
    z : Tensor
        Defocus of each plane in um, shape ``(Z',)``, FFT order (focus at index 0).
    shape : tuple[int, int, int]
        The coarser grid ``(Z, Y, X)`` whose frequencies are kept.

    Returns
    -------
    Tensor
        The transfer function of the real intensity point-spread function on the frequencies of ``shape``, as
        :func:`intensity_otf` returns it, shape ``(Z, Y, X // 2 + 1, 2)``.
    """
    lateral = (kz.shape[-2], kz.shape[-1])
    return intensity_otf(lambda part: defocused_intensity(pupil, kz, part), z, lateral=lateral, shape=shape)


def _low_frequencies(full: int, size: int, *, device: torch.device) -> Tensor:
    """FFT-order indices of the ``size`` lowest of ``full`` frequencies: ``0 .. (size-1)//2`` then the negatives."""
    return torch.cat(
        [torch.arange((size + 1) // 2, device=device), torch.arange(full - size // 2, full, device=device)]
    )


def _keep_lowest(spectrum: Tensor, sizes: Sequence[tuple[int, int]]) -> Tensor:
    """Select the ``size`` lowest frequencies along each ``(dim, size)``: a copy where the axis shrinks."""
    for dim, size in sizes:
        full = spectrum.shape[dim]
        if size > full:
            raise ShapeError(f"cannot crop {full} frequencies to {size}")
        if size != full:
            spectrum = spectrum.index_select(dim, _low_frequencies(full, size, device=spectrum.device))
    return spectrum


def crop_half_spectrum(spectrum: Tensor, shape: ZYX[int]) -> Tensor:
    """Keep the frequencies of a ``shape`` grid from a finer half spectrum.

    Along z and y it selects the lowest frequencies in FFT order (a copy); along x it keeps ``0 .. X // 2`` (a view).
    With the grid unchanged it returns a view of ``spectrum`` without copying.

    Parameters
    ----------
    spectrum : Tensor
        Complex values stored as ``(..., Z', Y', X' // 2 + 1, 2)``.
    shape : tuple[int, int, int]
        The coarser grid ``(Z, Y, X)``.

    Raises
    ------
    ShapeError
        If ``shape`` is finer than the spectrum's grid.
    """
    nz, ny, nx = shape
    spectrum = _keep_lowest(spectrum, ((-4, nz), (-3, ny)))
    columns = spectrum.shape[-2]
    if nx // 2 + 1 > columns:
        raise ShapeError(f"cannot crop a half spectrum of {columns} columns to {nx // 2 + 1}")
    return spectrum.narrow(-2, 0, nx // 2 + 1)


def hermitian_half(spectrum: Tensor, nx: int, *, real: bool = False) -> Tensor:
    """Project a half spectrum onto the transforms of real volumes: make its self-conjugate columns Hermitian.

    Column ``x = 0`` (and ``x = X / 2`` for even ``X``) must satisfy ``S(-z, -y) = conj(S(z, y))``. Cropping an
    even-sized z or y axis breaks this; ``irfftn`` silently applies the projection, so the closed-form verbs apply it
    too to describe the same operator. With ``real=True`` the projection makes those columns even. Returns a copy.

    Parameters
    ----------
    spectrum : Tensor
        Complex values stored as ``(..., Z, Y, X // 2 + 1, 2)``, or real values ``(..., Z, Y, X // 2 + 1)`` with
        ``real=True``.
    nx : int
        Length of the volume's x axis, which fixes whether a Nyquist column exists.
    real : bool
        Whether ``spectrum`` has no trailing real/imaginary axis.
    """
    offset = 0 if real else 1
    x_dim, zy_dims = -1 - offset, (-2 - offset, -1 - offset)

    def project(column: Tensor) -> Tensor:
        mirrored = torch.roll(torch.flip(column, dims=zy_dims), shifts=(1, 1), dims=zy_dims)
        if not real:
            re, im = mirrored.unbind(-1)
            mirrored = torch.stack((re, -im), dim=-1)
        return (0.5 * (column + mirrored)).unsqueeze(x_dim)

    first = project(spectrum.select(x_dim, 0))
    if nx % 2 == 0 and nx > 1:
        last = nx // 2
        middle = spectrum.narrow(x_dim, 1, last - 1)
        return torch.cat([first, middle, project(spectrum.select(x_dim, last))], dim=x_dim)
    return torch.cat([first, spectrum.narrow(x_dim, 1, spectrum.shape[x_dim] - 1)], dim=x_dim)


def half_spectrum_weights(nx: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return how many full-spectrum frequencies each half-spectrum column stands for, shape ``(nx // 2 + 1,)``.

    A sum over the full spectrum of a Hermitian-symmetric quantity equals the weighted sum over its half.
    """
    weights = torch.full((nx // 2 + 1,), 2.0, device=device, dtype=dtype)
    weights[0] = 1.0
    if nx % 2 == 0:
        weights[-1] = 1.0
    return weights


def is_complex_matrix(matrices: Tensor) -> bool:
    """Whether per-frequency matrices carry a trailing real/imaginary axis: ``(n, m, Z, Y, X', 2)``."""
    return matrices.dim() == _COMPLEX_MATRIX_DIMS


def _sum_or_select(values: Tensor, dim: int) -> Tensor:
    # Summing over an axis of size 1 would copy; selecting is a view.
    return values.select(dim, 0) if values.shape[dim] == 1 else values.sum(dim=dim)


def matrix_vector(matrices: Tensor, vectors: Tensor, *, adjoint: bool = False) -> Tensor:
    """Return ``M_k v_k`` (or ``M_k^H v_k`` with ``adjoint=True``) per frequency.

    Parameters
    ----------
    matrices : Tensor
        Per-frequency matrices ``(n, m, Z, Y, X')`` real, or ``(n, m, Z, Y, X', 2)`` complex.
    vectors : Tensor
        Complex vectors ``(m, Z, Y, X', 2)``, or ``(n, Z, Y, X', 2)`` with ``adjoint=True``.

    Returns
    -------
    Tensor
        Complex vectors ``(n, Z, Y, X', 2)``, or ``(m, Z, Y, X', 2)`` with ``adjoint=True``.
    """
    contracted = 0 if adjoint else 1
    paired = vectors[:, None] if adjoint else vectors[None]
    if not is_complex_matrix(matrices):
        return _sum_or_select(matrices[..., None] * paired, contracted)
    complex_matrices, complex_vectors = _as_complex(matrices), _as_complex(vectors)
    if matrices.shape[:2] == (1, 1):
        entry = complex_matrices[0, 0]
        return _as_real(((entry.conj() if adjoint else entry) * complex_vectors[0]).unsqueeze(0))
    if adjoint:
        return _as_real(torch.einsum("nm...,n...->m...", complex_matrices.conj(), complex_vectors))
    return _as_real(torch.einsum("nm...,m...->n...", complex_matrices, complex_vectors))


def _as_real(values: Tensor) -> Tensor:
    # view_as_real refuses a lazily conjugated tensor; resolving is free when no conjugation is pending.
    return torch.view_as_real(values.resolve_conj())


def apply_transfer(obj: Tensor, transfer: Tensor) -> Tensor:
    """Image real object fields: ``g_c = IRFFT(sum_f H_cf RFFT f_f)``.

    Parameters
    ----------
    obj : Tensor
        Real fields, shape ``(F, Z, Y, X)``.
    transfer : Tensor
        Half-spectrum transfer, ``(C, F, Z, Y, X // 2 + 1, 2)``, or ``(C, F, Z, Y, X // 2 + 1)`` where ``H`` is real
        (an even PSF, or the Gram ``H^H H`` of one field).

    Returns
    -------
    Tensor
        Real channels, shape ``(C, Z, Y, X)``.
    """
    imaged = matrix_vector(transfer, to_half_spectrum(obj))
    return from_half_spectrum(imaged, obj.shape[-3:])


def apply_adjoint(image: Tensor, transfer: Tensor) -> Tensor:
    """Apply the adjoint of :func:`apply_transfer`, ``f_f = IRFFT(sum_c conj(H_cf) RFFT g_c)``: ``(F, Z, Y, X)``."""
    adjoint = matrix_vector(transfer, to_half_spectrum(image), adjoint=True)
    return from_half_spectrum(adjoint, image.shape[-3:])


def gram(transfer: Tensor) -> Tensor:
    """Return ``H^H H`` per frequency, shape ``(F, F, ...)``; real for one field, where it is ``sum_c |H_c|^2``."""
    complex_valued = is_complex_matrix(transfer)
    if transfer.shape[1] == 1:  # one reduction over the channels (and the real/imaginary axis)
        if complex_valued:
            return transfer.square().sum(dim=(0, -1))[None]
        return _sum_or_select(transfer.square(), 0)[None]
    pattern = "cf...,cg...->fg..."
    if not complex_valued:
        return torch.einsum(pattern, transfer, transfer)
    complex_transfer = _as_complex(transfer)
    return _as_real(torch.einsum(pattern, complex_transfer.conj(), complex_transfer))


def _identity(n: int, like: Tensor) -> Tensor:
    """Return the identity broadcastable against real per-frequency matrices ``(n, n, Z, Y, X')``."""
    return torch.eye(n, dtype=like.dtype, device=like.device).reshape(n, n, 1, 1, 1)


def add_diagonal(matrices: Tensor, diagonal: Tensor) -> Tensor:
    """Return ``M_k + diag(d_k)`` per frequency, adding to the real part of a complex matrix.

    ``diagonal`` is real and broadcastable to ``(n, Z, Y, X')``, for example a scalar variance.
    """
    if not is_complex_matrix(matrices):
        return matrices + _identity(matrices.shape[0], matrices) * diagonal
    re, im = matrices.unbind(-1)
    return torch.stack((re + _identity(re.shape[0], re) * diagonal, im), dim=-1)


def channel_covariance(transfer: Tensor, spectrum: Tensor, noise_variance: Tensor) -> Tensor:
    """Return ``V = H S H^H + sigma^2 I`` per frequency, shape ``(C, C, ...)``.

    ``transfer`` is ``(C, F, ...)`` and ``spectrum`` the object variance ``S`` per field and frequency,
    broadcastable to ``(F, ...)``. For one channel ``V`` is real, ``sum_f |H_f|^2 S_f + sigma^2``.
    """
    complex_valued = is_complex_matrix(transfer)
    if transfer.shape[0] == 1:
        power = abs2(transfer[0]) if complex_valued else transfer[0].square()
        return (_sum_or_select(power * spectrum, 0) + noise_variance)[None, None]
    pattern = "cf...,df...->cd..."
    if not complex_valued:
        return add_diagonal(torch.einsum(pattern, transfer * spectrum, transfer), noise_variance)
    complex_transfer = _as_complex(transfer)
    covariance = torch.einsum(pattern, complex_transfer * spectrum, complex_transfer.conj())
    return add_diagonal(_as_real(covariance), noise_variance)


def _batched(matrices: Tensor) -> Tensor:
    return matrices.movedim((0, 1), (-2, -1))


def _unbatched(matrices: Tensor) -> Tensor:
    return matrices.movedim((-2, -1), (0, 1))


def _embed(matrices: Tensor) -> Tensor:
    """Return the real ``(2n, 2m, ...)`` embedding ``[[Re, -Im], [Im, Re]]`` of complex ``(n, m, ..., 2)`` matrices."""
    re, im = matrices.unbind(-1)
    return torch.cat((torch.cat((re, -im), dim=1), torch.cat((im, re), dim=1)), dim=0)


def _stack_parts(vectors: Tensor, *, dim: int) -> Tensor:
    """``(..., n, ..., 2)`` complex vectors to ``(..., 2n, ...)`` real ones: real parts, then imaginary parts."""
    return torch.cat(vectors.unbind(-1), dim=dim)


def _split_parts(vectors: Tensor, *, dim: int) -> Tensor:
    """Inverse of :func:`_stack_parts`."""
    return torch.stack(vectors.chunk(2, dim=dim), dim=-1)


@dataclass(frozen=True, slots=True)
class HermitianFactor:
    """The Cholesky factor of Hermitian positive definite per-frequency matrices, from :func:`cholesky_hermitian`.

    Attributes
    ----------
    lower : Tensor
        Real lower-triangular factors: ``(n, n, Z, Y, X')`` with ``L L^T = M`` for real ``M``, or
        ``(2n, 2n, Z, Y, X')`` for the real embedding of a complex ``M`` when ``embedded``.
    embedded : bool
        Whether ``lower`` factors the ``2n x 2n`` real embedding of a complex matrix.
    """

    lower: Tensor
    embedded: bool

    @property
    def size(self) -> int:
        """The order ``n`` of the factored matrices."""
        return self.lower.shape[0] // 2 if self.embedded else self.lower.shape[0]


def cholesky_hermitian(matrices: Tensor) -> HermitianFactor:
    """Factor Hermitian positive definite per-frequency matrices ``(n, n, ...)``, real or complex.

    A ``1 x 1`` matrix factors as its square root; complex matrices factor through their real embedding. There is no
    error check, which would synchronize the device: a failed factor shows up as NaN in the result.
    """
    complex_valued = is_complex_matrix(matrices)
    if matrices.shape[0] == 1:
        real = matrices[..., 0] if complex_valued else matrices
        return HermitianFactor(torch.sqrt(real.clamp_min(0.0)), embedded=False)
    real = _embed(matrices) if complex_valued else matrices
    return HermitianFactor(_unbatched(torch.linalg.cholesky_ex(_batched(real)).L), embedded=complex_valued)


def cholesky_solve(factor: HermitianFactor, rhs: Tensor) -> Tensor:
    """Solve ``M x = rhs`` per frequency for complex vectors ``rhs`` ``(n, Z, Y, X', 2)``; returns the same shape."""
    lower = factor.lower
    if factor.embedded:
        stacked = _stack_parts(rhs, dim=0).movedim(0, -1).unsqueeze(-1)
        solution = torch.cholesky_solve(stacked, _batched(lower)).squeeze(-1).movedim(-1, 0)
        return _split_parts(solution, dim=0)
    if lower.shape[0] == 1:
        return rhs / lower[0].square()[..., None]
    # The real and imaginary parts are two right-hand sides of one real system.
    return torch.cholesky_solve(rhs.movedim(0, -2), _batched(lower)).movedim(-2, 0)


def cholesky_logdet(factor: HermitianFactor) -> Tensor:
    """Return ``log det M`` per frequency, shape ``(Z, Y, X')``, from its factor."""
    lower = factor.lower
    if lower.shape[0] == 1:
        return 2 * torch.log(lower[0, 0])
    total = torch.log(torch.diagonal(lower, dim1=0, dim2=1)).sum(dim=-1)
    # det of the embedding is det(M)^2 for Hermitian M
    return total if factor.embedded else 2 * total


def hermitian_inverse(matrices: Tensor) -> Tensor:
    """Return ``M_k^-1`` per frequency for invertible Hermitian ``(n, n, ...)`` matrices, in the same layout."""
    complex_valued = is_complex_matrix(matrices)
    n = matrices.shape[0]
    if n == 1:
        return 1 / (matrices[..., 0] if complex_valued else matrices)
    if not complex_valued:
        return _unbatched(torch.linalg.inv(_batched(matrices)))
    inverse = _unbatched(torch.linalg.inv(_batched(_embed(matrices))))  # the embedding of M^-1
    return torch.stack((inverse[:n, :n], inverse[n:, :n]), dim=-1)


def largest_singular_value(transfer: Tensor) -> Tensor:
    """Return ``max_k ||H_k||_2``, the operator norm of per-frequency matrices ``(C, F, ...)``, real or complex.

    The real embedding of a complex matrix has the same singular values, each twice.
    """
    complex_valued = is_complex_matrix(transfer)
    if transfer.shape[:2] == (1, 1):
        return abs2(transfer).amax().sqrt() if complex_valued else transfer.abs().amax()
    real = _embed(transfer) if complex_valued else transfer
    return torch.linalg.matrix_norm(_batched(real), ord=2).amax()
