# Contributing guide

This guide is for adding to `waveorder`: a new imaging system, prior, noise model, solver, score or search stage, or
a change to the models, the configuration or the command line. The [README](./README.md) shows the package from the
user's side.

Report bugs and propose features in the [issues](https://github.com/Biohub/waveorder/issues). Search the existing
ones first, closed ones too.

Participation in waveorder is governed by the
[Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md).

Do not report security vulnerabilities in public issues. Follow our
[security policy](./SECURITY.md) to report them privately.

## Setup

Install [Git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone the
repository (or [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) it) and create the environment:

```sh
git clone https://github.com/Biohub/waveorder.git
cd waveorder
uv sync    # the dev group: test (pytest, coverage, napari, pytest-qt) and lint (ruff, ty), pinned to CI's versions
```

## The checks

Every pull request passes these four, which CI runs as written:

```sh
uv run ruff format --check
uv run ruff check
uv run ty check --error-on-warning
uv run pytest
```

`pytest` turns every warning into an error (`filterwarnings = ["error"]` in `pyproject.toml`), so a new warning
either gets fixed or gets an explicit `pytest.mark.filterwarnings` with the reason. On a shared machine, set
`OMP_NUM_THREADS=4`. Tests that need a GPU skip without one. The rules in `pyproject.toml` encode design decisions,
so fix the code instead of adding `noqa`.

## Layers

The package is split into layers, and imports go down only: at module level and inside functions alike.

```
cli ──> models ──> config ──> io, nn
 │        └──> tiling ──> io, nn
 └──> plugin ──> config
```

| Package | Holds |
|---|---|
| `waveorder.cli` | the `wo` command line; each command builds a model with `from_config` and calls its methods |
| `waveorder.plugin` | the napari plugin and the window of `wo interactive`; it runs `wo` commands as subprocesses |
| `waveorder.models` | the user API: `ReconstructionModel`, `Phase3D`, `Phase2D`, `Fluorescence3D`, `Fluorescence2D` and their verbs |
| `waveorder.config` | the version-4 schema (`schema.py`), its loader, `build.py` (configuration to parts, through the registry), `migrate.py` (3.x to version 4) |
| `waveorder.tiling` | tile plans, the tile loop and the blending of tile estimates |
| `waveorder.io` | OME-Zarr plates and positions, the transfer-function store `TfStore`, staging buffers |
| `waveorder.nn` | the torch modules of `g = H f + n`: systems, optics, priors, noise, solvers, scores, search stages |

The layer `cli` may also import `tiling`, `io` and `nn`; `testing` imports `nn`. The front doors
`waveorder.calibrate` and `waveorder.simulate` only re-export names from the layers below them. Any layer may import
the utility modules `errors`, `zernike_names` and `registry`, which import no layer. `waveorder.nn`
reads and writes no files: it imports neither `zarr` nor `iohub`.

`tests/test_layering.py` holds the table (`LAYERS`) and fails for an import that goes up, naming the file and the
line. A new package or top-level module needs a row there. `tests/cli/test_torch_free.py` checks that `wo --help`
and `wo config new` never import torch; commands import the layers below inside their bodies.

## Conventions

- Python 3.12 or newer. Write generics and aliases with PEP 695 syntax (`def f[T](x: T)`, `type Spec = ...`) and mark
  overrides with `typing.override`.
- Computation in `waveorder.nn` stays in torch, so autograd, `vmap` and GPUs work end to end. numpy is fine at file
  boundaries: the arrays zarr reads and writes (`io/`), napari's arrays (`plugin/`), the benchmarks' references.
- No type hint written as a string: import the type, and move a type both sides of a cycle need to a lower module.
  Within one module, a forward reference is a PEP 695 alias (`type _Fourier = FourierDiagonal | None`), which is
  evaluated lazily. `tests/test_annotations.py` fails on a string annotation.
- No `from __future__ import annotations`: pydantic and torch read annotations at runtime. ruff bans it too.
- `napari` and the other heavy or optional libraries are imported inside the function that needs them
  (ruff's `banned-module-level-imports` lists them).
- Complex values are stored as real pairs: a spectrum or per-frequency matrix is its `torch.view_as_real` form, with
  a trailing axis of size 2, and a real transfer (`real_transfer = True`) drops that axis. `waveorder.nn.functional`
  does the arithmetic between the FFTs as one complex kernel on `view_as_complex` views. New per-frequency code calls
  its helpers (`to_half_spectrum`, `from_half_spectrum`, `complex_multiply`, `matrix_vector`, `gram`, ...).
- Docstrings follow the NumPy convention (ruff `D`). Each argument is documented once: constructor arguments on the
  class, the others on the function that takes them. A function that passes arguments on (`**parts`, `**options`)
  names the one that documents them.
- Arguments are validated at two places: when a configuration is loaded (the pydantic schema in
  `config/schema.py`), and when a model method is entered (`models/_arguments.py`, `models/_checks.py`). Below them,
  code trusts its callers and ty checks the types. Every exception and warning lives in `waveorder.errors`; a
  `ParameterError` or `ConfigurationError` names the argument or the config path.
- Lines are at most 120 characters; boolean flags are keyword-only; a function takes at most 6 arguments, 3 of them
  positional (ruff `FBT`, `PLR0913`, `PLR0917`).

## Adding a part

Every part is a `torch.nn.Module` (a search stage is a plain class) in `waveorder.nn`, registered under a name that is
unique across kinds:

```python
from waveorder.registry import register


@register("my_prior", kind="object_prior")
class MyPrior(ObjectPrior): ...
```

Registration wraps `__init__` so that each instance keeps the arguments it was built with. That is its plain-data
form, `{"my_prior": {"weight": 0.01}}`, which `registry.resolve` builds and `registry.describe` writes; `model.save`,
`model.load` and the reconstruction configuration use it. A frozen dataclass reports its fields, and a class with an
`arguments()` method reports what it returns. A quantity that a calibration changes is reported at its current
value. In tests, register for the length of a `with` block with `tests.registration.temporarily(name, cls, kind=...)`.

Use `register_quantity(name, value, positive=...)` (from `CalibratableModule`) for every number that may be
calibrated: a fixed value becomes a buffer, a `Free(...)` value an `nn.Parameter` named by its dotted path.

Each kind has a base class, its required methods (at most three; every other method has a default, and
`tests/test_contracts_size.py` fails a base class that requires more) and a contract check in `waveorder.testing`.
The base classes are importable from `waveorder.nn`, except `SharpnessScore` (`waveorder.nn.scores`). Add the new
part to `tests/nn/test_contracts.py`, which runs the checks on the parts in its tables `SYSTEMS`, `PRIORS`, `SOLVERS`
and `STAGES` and in its parametrized noise and score tests.

| Kind (`register(..., kind=)`) | Subclass | Implement | Check |
|---|---|---|---|
| imaging system (`"system"`) | `ShiftInvariantSystem` | `compute_transfer(sampling)`: the `rfftn` half spectrum `(C, F, Z, Y, X // 2 + 1)` | `check_system(build, sampling)` |
| system bound on another grid (`"system"`) | `ImagingSystem` | `compute_operator(sampling)` returning a `ReshapedFourierDiagonal` (as the thin systems do); set `operator_type` | `check_system(build, sampling)` |
| object prior (`"object_prior"`) | `ObjectPrior` | `penalty(obj, *, scale)`, `prox(obj, step, *, scale)` | `check_prior(prior, obj)` |
| Gaussian object prior | `GaussianObjectPrior` | also `spectrum(sampling, *, scale)`, used by the closed-form solver and the evidence | `check_prior(prior, obj)` |
| noise model (`"noise"`) | `Noise` | `nll`, `sample`, `variance`; `gradient` defaults to autograd of `nll`, `scale` to `white_gaussian_variance()` or 1 | `check_noise(noise, data, mean)` |
| solver (`"solver"`) | `Solver` | `run(problem)` returning a `SolveResult`; `check(model)` when it cannot run on every model | `check_solver(solver, data=, model=, sampling=)` |
| iterative solver | `IterativeSolver` | `start(problem)`, `step(state, problem)`, `estimate(state)`; `check(model)` as for `Solver`; built with `iterations`, `tolerance` and `check_every` | `check_solver(...)` |
| score (`"score"`) | `Score` | `run(problem)` returning a 0-d tensor; `check(model)` as for `Solver`; `is_likelihood` | `check_score(score, data=, model=, sampling=)` |
| sharpness score | `SharpnessScore` | `sharpness(slices, sampling)` | `check_score(...)` |
| search stage (`"search"`) | the `SearchStage` protocol | `run(objective, *, start)` returning an `Estimate`; class attributes `reads_back`, `quantity_keyed` | `check_search_stage(stage, objective, start)` |

Notes per kind:

- **Systems.** The class attributes state what solvers may rely on: `real_transfer` (the transfer is real, stored
  without the pair axis), `preserves_nonnegativity` (needed by Richardson-Lucy, checked by
  `assert_preserves_nonnegativity`) and `operator_type` (`ShiftInvariantSystem` sets `FourierDiagonal`, which the
  closed-form solver needs). `waveorder.nn.functional.half_spectrum_frequencies(sampling)` gives the frequency grid.
  Systems live in `nn/systems/`, one file per contrast. Shift-variant optics are field tiles: each tile is a
  shift-invariant system with its own quantities (a model's `field_tiles`, `waveorder.nn.calibrate_tiles`), so there
  is no shift-variant operator to write.
- **Priors and noise.** Capabilities a solver reads are class flags with a matching method: `has_curvature_bound`
  with `curvature_bound()` (FISTA needs it), `has_photon_counts` with `photon_counts()` (Richardson-Lucy needs it),
  and `is_flat` on a prior. A verb's `check` reads them and raises `IncompatibleModelError` before anything is
  computed.
- **Solvers and scores** are `Verb`s: `forward` is final and runs the model's check, builds an `InverseProblem` and
  calls `run`, where `problem.operator` is `H` bound on first use. A solver that never reads it overrides
  `needs_operator` to return `False`. One file per solver in `nn/solvers/`; scores in `nn/scores.py`.
- **Search stages** keep their points inside `objective.prior`'s bounds and their tensors on the device, so stages
  chain without a host sync. Files in `nn/search/`.

To make a part selectable from a reconstruction configuration, add its settings to `config/schema.py` and resolve it
by its registered name in `config/build.py`; the tests are in `tests/config/`.

### A model class

A new contrast also gets a model class in `waveorder.models`: subclass `ReconstructionModel`, list the system in
`system_types` (one class per system type; `for_system`, `from_config` and `load` return the class that lists it)
and build the system in `__init__`, passing `**parts: Unpack[Parts]` on. Override where the defaults do not fit:

- `prepare(data, *, mean=None)`, which turns recorded data into what the system measures. A class that defines it
  sets `prepare_scope` beside it: `"voxel"` (reads only the voxel), `"stack_mean"` (also the stack's mean, passed as
  `mean` to tiles) or `"block"` (the whole block, so a volume is never split into tiles).
- `default_object_prior`, `default_noise`, `default_solver` and `output_channel_names`.
- `setup()`, for state beyond the parts: `for_system`, `from_config` and `load` do not call `__init__`.

`models/contrasts.py` holds the built-in model classes, the smallest complete cases. Model tests are in
`tests/models/`.

### A command or an option

Commands live in `src/waveorder/cli/` (shared flags in `options.py`) and call model methods; most options are one
argument of the method a command calls. Keep `wo --help` free of torch: what a command needs from the models runs
in an underscore module it imports inside its body (`_configured.py` builds the model from the configuration,
`_reconstruct.py`, `_simulate.py`, `_progress.py`), and add the command to the tests in `tests/cli/`.

## The regression suite

`benchmarks/` holds the regression suite, which compares reconstructions with those of the `wo-4.0` branch. Run it
from a source checkout; the runner calls `wo config migrate` and `wo rec` as subprocesses:

```sh
wo bm run --scope synthetic -o runs       # the synthetic cases; --scope all adds the cases on Biohub HPC data
wo bm latest -o runs                      # the summary of the newest run
wo bm history -o runs                     # recent runs
wo bm compare -o runs [RUN_A] [RUN_B]     # midband power, mse and ssim of two runs; by default the two newest
```

`-o` defaults to `$WAVEORDER_BENCH_OUTPUT`, else the current directory. `-e` runs another experiment file, and
`--save-all` keeps every output (by default transfer-function stores, and reconstructions and simulations above
25 MB, are deleted after scoring).

- `benchmarks/configs/` holds one 3.x configuration per case. Each run migrates it with `wo config migrate`,
  reconstructs with `wo rec` and scores the output.
- `benchmarks/experiments/regression.yml` lists the cases: synthetic ones (a phantom, simulated, then reconstructed)
  and `hpc` ones (an OME-Zarr input on the Biohub HPC). Each has a `reference` bound, so one run flags a collapsed or
  missing reconstruction.
- `benchmarks/run_regression.sbatch` runs `wo bm run --scope all` as a Slurm job (preempted partition, 1 GPU,
  16 CPUs, 64 GB, 45 minutes). Set `WAVEORDER_BENCH_OUTPUT` or pass `-o`:
  `sbatch benchmarks/run_regression.sbatch -o /path/to/runs`.
- To compare with `wo-4.0`, run the same suite from a `wo-4.0` checkout and from your branch into one output
  directory, then `wo bm compare` the two runs.
- `tests/benchmarks/` tests the runner itself; `benchmarks/README.md` describes the output layout and how to add a
  case.
