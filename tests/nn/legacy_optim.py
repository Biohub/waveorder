"""WaveOrder 3.x calibration optimizers, as a test-only reference for :mod:`waveorder.nn.optim`.

Copied from WaveOrder 3.x ``waveorder/optim/optimize.py`` at tag 3.0.5 (BSD-3-Clause, Chan Zuckerberg Biohub) and,
for ``method="nelder_mead"``, from the function 3.x called, SciPy's ``scipy.optimize._optimize._minimize_neldermead``
(BSD-3-Clause, the SciPy developers), because SciPy is no dependency of 4.x. Changes from the originals, none of
which alters a result:

- numpy arrays replaced by lists of Python floats (the same IEEE operations, in the same order);
- only the paths 3.x used: unbatched, no bounds, no logging, no adaptive coefficients, no ``initial_simplex``;
- the 3.x loop returns every evaluated point with its loss, so a test can compare against 4.x's best point.
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor


class _MaxFuncCallError(RuntimeError):
    pass


def minimize_neldermead(  # noqa: C901, PLR0912, PLR0915 (a verbatim port of SciPy's one function)
    func: Callable[[list[float]], float], x0: list[float], *, maxiter: int, maxfev: int
) -> tuple[list[float], float, int]:
    """SciPy ``_minimize_neldermead`` with its defaults ``xatol = fatol = 1e-4``: return ``(x, fun, nfev)``."""
    xatol = fatol = 1e-4
    rho, chi, psi, sigma = 1, 2, 0.5, 0.5
    nonzdelt, zdelt = 0.05, 0.00025
    ncalls = [0]

    def function(x: list[float]) -> float:
        if ncalls[0] >= maxfev:
            raise _MaxFuncCallError("Too many function calls")
        ncalls[0] += 1
        return func(list(x))

    N = len(x0)
    sim = [list(x0)]
    for k in range(N):
        y = list(x0)
        y[k] = (1 + nonzdelt) * y[k] if y[k] != 0 else zdelt
        sim.append(y)
    one2np1 = list(range(1, N + 1))
    fsim = [math.inf] * (N + 1)
    try:
        for k in range(N + 1):
            fsim[k] = function(sim[k])
    except _MaxFuncCallError:
        pass
    sim, fsim = _take_sorted(sim, fsim)
    iterations = 1
    while ncalls[0] < maxfev and iterations < maxiter:
        try:
            if (
                max(abs(sim[i][j] - sim[0][j]) for i in range(1, N + 1) for j in range(N)) <= xatol
                and max(abs(fsim[0] - fsim[i]) for i in range(1, N + 1)) <= fatol
            ):
                break
            xbar = [0.0] * N
            for row in sim[:-1]:
                xbar = [a + b for a, b in zip(xbar, row, strict=True)]
            xbar = [a / N for a in xbar]
            xr = [(1 + rho) * a - rho * b for a, b in zip(xbar, sim[-1], strict=True)]
            fxr = function(xr)
            doshrink = 0
            if fxr < fsim[0]:
                xe = [(1 + rho * chi) * a - rho * chi * b for a, b in zip(xbar, sim[-1], strict=True)]
                fxe = function(xe)
                if fxe < fxr:
                    sim[-1], fsim[-1] = xe, fxe
                else:
                    sim[-1], fsim[-1] = xr, fxr
            elif fxr < fsim[-2]:
                sim[-1], fsim[-1] = xr, fxr
            else:
                if fxr < fsim[-1]:
                    xc = [(1 + psi * rho) * a - psi * rho * b for a, b in zip(xbar, sim[-1], strict=True)]
                    fxc = function(xc)
                    if fxc <= fxr:
                        sim[-1], fsim[-1] = xc, fxc
                    else:
                        doshrink = 1
                else:
                    xcc = [(1 - psi) * a + psi * b for a, b in zip(xbar, sim[-1], strict=True)]
                    fxcc = function(xcc)
                    if fxcc < fsim[-1]:
                        sim[-1], fsim[-1] = xcc, fxcc
                    else:
                        doshrink = 1
                if doshrink:
                    for j in one2np1:
                        sim[j] = [a + sigma * (b - a) for a, b in zip(sim[0], sim[j], strict=True)]
                        fsim[j] = function(sim[j])
        except _MaxFuncCallError:
            pass
        finally:
            sim, fsim = _take_sorted(sim, fsim)
        iterations += 1
    return sim[0], min(fsim), ncalls[0]


def _take_sorted(sim: list[list[float]], fsim: list[float]) -> tuple[list[list[float]], list[float]]:
    ind = sorted(range(len(fsim)), key=fsim.__getitem__)  # np.argsort: insertion sort, stable, for N + 1 <= 16
    return [sim[i] for i in ind], [fsim[i] for i in ind]


def optimize_gradient(
    loss_fn: Callable[[dict[str, Tensor]], Tensor],
    optimizable_params: dict[str, tuple[float, float]],
    *,
    method: str,
    max_iterations: int,
    dtype: torch.dtype,
) -> list[tuple[dict[str, float], float]]:
    """3.x ``_optimize_gradient`` for ``method`` in ``adam``/``nadam``, plus 3.x ``recover_zernikes``'s ``adamw``.

    Returns every evaluated ``(params, loss)``: the loss before each step, then the final parameters' loss.
    """
    param_tensors: dict[str, Tensor] = {}
    param_groups = []
    for name, (init_val, lr) in optimizable_params.items():
        t = torch.tensor(init_val, dtype=dtype, requires_grad=True)
        param_tensors[name] = t
        param_groups.append({"params": [t], "lr": lr})
    if method == "nadam":
        optimizer = torch.optim.NAdam(param_groups)
    elif method == "adamw":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-3)
    else:
        optimizer = torch.optim.Adam(param_groups)
    visited = []
    for _ in range(max_iterations):
        optimizer.zero_grad()
        loss = loss_fn(param_tensors)
        visited.append(({name: t.item() for name, t in param_tensors.items()}, loss.item()))
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        visited.append(({name: t.item() for name, t in param_tensors.items()}, loss_fn(param_tensors).item()))
    return visited
