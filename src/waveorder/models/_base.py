"""The base of every user-facing model: an imaging model with the solver that inverts it.

:class:`ReconstructionModel` wraps the torch modules of :mod:`waveorder.nn` (the :class:`~waveorder.nn.ImagingModel`
of ``g = H f + n``, a solver, and the score and search that calibrate the free quantities) in one object with the
verbs a user calls. This file holds the class and its verbs. The helpers of the other underscore files take the model
first and import this class, so they are imported after it: at the end of this file, or, for the position methods
(whose helpers import iohub), in the methods when they run.
"""

import copy
import dataclasses
import operator
import os
import tempfile
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Literal, Self, Unpack, cast

import torch
from torch import Tensor, nn

from waveorder.config.schema import GaussianNoise, ObjectSettings, PoissonNoise, SourceGaussianSpectrum, V4Config
from waveorder.errors import (
    ConfigurationError,
    IncompatibleModelError,
    ParameterError,
    PositionWarning,
    Problem,
    Problems,
    ReconstructionReport,
    Rejected,
    ShapeError,
)
from waveorder.io.bundle import load_bundle
from waveorder.io.tf_store import TfStore
from waveorder.models import _arguments
from waveorder.models._arguments import Inputs, PixelSize, expect_instance, zyx
from waveorder.models._parts import PREPARE_SCOPES, FieldTiles, Parts, PrepareScope, ZPaddingSetting, auto_z_padding
from waveorder.nn import (
    FISTA,
    CalibrationResult,
    Distilled,
    Gaussian,
    GradientSearch,
    ImagingModel,
    ImagingSystem,
    Noise,
    NoPrior,
    ObjectPrior,
    ParameterPrior,
    RichardsonLucy,
    Sampling,
    Score,
    SearchStage,
    Solver,
    SolveResult,
    Tikhonov,
    TileCalibration,
    Wiener,
    calibrate,
    free_specs,
    load_parameters,
)
from waveorder.nn import calibrate_tiles as calibrate_tile_stack
from waveorder.nn.networks import RestorationNetwork
from waveorder.nn.parameters import placement, resolve_device
from waveorder.nn.preprocess import ZPadding, apodize, most_structured_crop
from waveorder.nn.priors import Beads
from waveorder.nn.sampling import ZYX
from waveorder.nn.scores import SharpnessScore
from waveorder.nn.search.beads import BeadCalibration, calibrate_beads
from waveorder.nn.search.sweep import WeightSweep
from waveorder.nn.systems.fluorescence import WidefieldFluorescence3D
from waveorder.nn.systems.tiled import TileParameters
from waveorder.tiling import TileOptions, TilePlan, TileStats, halo_for, reconstruct_series
from waveorder.tiling.merge import ArrayLike, WritableArray
from waveorder.tiling.run import expect_volumes, stack_tiles

_BEAD_ITERATIONS = 200
"""Gradient steps of a bead calibration whose search holds no :class:`~waveorder.nn.GradientSearch`."""

_HALO_ENERGY = 0.99
"""The fraction of the solver's filter (L1 mass per axis) a tile's halo keeps (:func:`~waveorder.tiling.halo_for`)."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TrainResult[M]:
    """What :meth:`ReconstructionModel.train` leaves.

    Parameters
    ----------
    bundle : Path
        The bundle to use: ``best/`` of the run folder, or ``last/`` when no validation improved the monitored metric.
    metrics : dict[str, float]
        The last value of every logged metric (``train/loss``, ``val/ssim``, ...).
    checkpoint : Path
        The run's ``checkpoints/last.ckpt``, which ``wo distill fit --ckpt_path`` resumes from.
    directory : Path
        The run folder, with its ``config.yaml``: ``wo distill fit --config RUN/config.yaml`` runs it again.
    model : ReconstructionModel
        A copy of the model whose solver is the trained network.
    """

    bundle: Path
    metrics: dict[str, float]
    checkpoint: Path
    directory: Path
    model: M


class ReconstructionModel:
    """An imaging model with the solver that inverts it and the settings that calibrate it.

    A plain Python class; the torch module tree is :attr:`imaging_model` (its parameters and state are the
    model's). The verbs: :meth:`reconstruct` estimates the object, :meth:`calibrate` the free quantities (those
    given as :class:`~waveorder.nn.Free`), :meth:`simulate` makes data, :meth:`train` distils the solver into a
    network, :meth:`save`, :meth:`load` and :meth:`export` keep the model, :meth:`to` moves it. The tile methods do
    the same on arrays too large for one solve, the position methods on OME-Zarr positions.

    A calibration of the whole field writes its estimate into the free quantities; a calibration per tile never
    does (:meth:`calibrate_tiles` returns its table, which :meth:`calibrate_position` keeps as
    :attr:`tile_parameters` on a model with field tiles).

    A modality subclasses it: it lists the systems it wraps in :attr:`system_types`, builds one in ``__init__``
    and passes ``**parts: Unpack[Parts]`` on; it overrides :meth:`prepare` (with :attr:`prepare_scope`), the
    ``default_*`` hooks and :meth:`output_channel_names` where the defaults do not fit. :meth:`for_system` and
    :meth:`from_config` build the class around a system without calling its ``__init__``, so state beyond the
    parts is set in :meth:`setup`.

    Parameters
    ----------
    system : ImagingSystem
        ``H`` and its free quantities. The model holds it, not a copy.
    object_prior : ObjectPrior, optional
        ``p(f)``; by default :meth:`default_object_prior`, or :class:`~waveorder.nn.NoPrior` under a noise that
        counts photons.
    noise : Noise, optional
        ``p(n)``; by default :meth:`default_noise`.
    solver : Solver, optional
        The reconstruction; by default :meth:`default_solver`, which follows the parts as they are assigned. A
        given solver is checked against the parts (a :class:`~waveorder.nn.Distilled` one also against the system
        its network was trained for).
    score : Score, optional
        The calibration objective; :class:`~waveorder.nn.Evidence` by default. A sharpness score given without its
        solver and cutoff is filled with the model's.
    search : Sequence[SearchStage], optional
        The calibration stages, in order; by default those of :func:`~waveorder.nn.calibrate`.
    z_padding : int or "auto"
        Mirrored z slices added around each solve; the estimate keeps the measured slices. ``"auto"`` pads each
        volume by at least the axial extent of the point-spread function (:meth:`z_padding_for`).
    field_tiles : FieldTiles, optional
        The tiles of a system that varies over the field of view; such a model is calibrated and reconstructed per
        tile (the tile and position methods), and :meth:`reconstruct` and :meth:`calibrate` refuse it.
    pixel_size : float, (z, yx), (z, y, x), {"z", "y", "x"} or "auto", optional
        Voxel size of the data in um (:data:`~waveorder.models._arguments.PixelSize`; the mappings as the
        configuration's ``data.pixel_size``). Every method that takes data uses it unless given another for that call;
        ``None`` or ``"auto"`` leaves it to the data: the position methods read each position's scale, the methods
        on tensors refuse.
    apodization_rolloff : float
        Fraction of the transverse band over which a Wiener estimate is rolled off to Nyquist
        (:func:`~waveorder.nn.preprocess.apodize`); 0 for none.
    device : str or torch.device, optional
        Where the model lives: ``None`` is where ``system`` is, ``"auto"`` the first of CUDA, MPS and CPU.

    Attributes
    ----------
    system_types : tuple[type[ImagingSystem], ...]
        The systems this class wraps (class attribute); :meth:`for_system` and :meth:`from_config` return the class
        that lists a system's type. No two classes list the same type.
    prepare_scope : {"voxel", "stack_mean", "block"}
        What :meth:`prepare` reads to prepare one voxel (class attribute): only it, also the stack's mean (given as
        ``mean`` to tiled work), or the whole block (a volume is then never split into tiles). A class that
        defines ``prepare`` defines it beside it.

    Notes
    -----
    The parts, :attr:`z_padding`, :attr:`field_tiles`, :attr:`tile_parameters` and :attr:`pixel_size` may be
    assigned: the value is checked and moved to the model, as if the model had been built with it.

    Raises
    ------
    ParameterError
        If an argument is not of its type or ``system`` is not one of :attr:`system_types`.
    ConfigurationError
        At ``device`` if no such device is available; at ``system`` for the base class given a system another
        class wraps.
    IncompatibleModelError
        If the solver cannot run on the parts.
    """

    system_types: ClassVar[tuple[type[ImagingSystem], ...]] = ()
    prepare_scope: ClassVar[PrepareScope] = "voxel"

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        body = vars(cls)
        if cls.prepare_scope not in PREPARE_SCOPES or ("prepare" in body and "prepare_scope" not in body):
            raise ParameterError(
                f"{cls.__name__} must set prepare_scope beside prepare, to one of {', '.join(PREPARE_SCOPES)}; got "
                f"{cls.prepare_scope!r}",
                path="prepare_scope",
            )
        if "system_types" not in body:
            return
        types = cls.system_types
        if not isinstance(types, tuple) or not all(isinstance(t, type) and issubclass(t, ImagingSystem) for t in types):
            raise ParameterError(
                f"{cls.__name__}.system_types must be a tuple of ImagingSystem classes", path="system_types"
            )
        same = (cls.__module__, cls.__qualname__)  # a class defined again (a notebook cell run twice) replaces itself
        others = [other for other in _MODEL_CLASSES if (other.__module__, other.__qualname__) != same]
        for other in others:
            shared = [kind.__name__ for kind in types if kind in other.system_types]
            if shared:
                raise ParameterError(
                    f"{cls.__name__}.system_types lists {', '.join(shared)}, which {other.__name__} already wraps; "
                    f"subclass {other.__name__} without system_types, or wrap a system class of your own",
                    path="system_types",
                )
        _MODEL_CLASSES[:] = [*others, cls]

    def __init__(  # noqa: PLR0913 (the system, and one keyword per part of the model)
        self,
        system: ImagingSystem,
        *,
        object_prior: ObjectPrior | None = None,
        noise: Noise | None = None,
        solver: Solver | None = None,
        score: Score | None = None,
        search: Sequence[SearchStage] | None = None,
        z_padding: ZPaddingSetting = 0,
        field_tiles: FieldTiles | None = None,
        pixel_size: PixelSize | Literal["auto"] | None = None,
        apodization_rolloff: float = 0.0,
        device: str | torch.device | None = None,
    ) -> None:
        where = type(self).__name__
        self._expect_system(system)
        expect_instance(object_prior, (ObjectPrior, type(None)), where=where, name="object_prior")
        expect_instance(noise, (Noise, type(None)), where=where, name="noise")
        expect_instance(device, (str, torch.device, type(None)), where=where, name="device")
        self._config: V4Config | None = None
        self._built: tuple[nn.Module, ...] = ()
        self._tile_parameters: dict[str, Tensor] | None = None
        self._stored_transfers: _stores.StoredTransfers | None = None  # the transfer functions of a loaded store
        self._score: Score | None = None
        self.z_padding = z_padding
        self.pixel_size = pixel_size
        self.apodization_rolloff = apodization_rolloff
        dtype, target = placement(system)
        if device is not None:
            target = resolve_device(str(device), field="device")
        chosen_noise = self.default_noise() if noise is None else noise
        chosen_prior = self._default_prior(chosen_noise) if object_prior is None else object_prior
        self._imaging_model = placed(ImagingModel(system, chosen_prior, chosen_noise), device=target, dtype=dtype)
        self._defaults: dict[str, nn.Module] = {}
        if object_prior is None:
            self._defaults["object_prior"] = chosen_prior
        if noise is None:
            self._defaults["noise"] = chosen_noise
        self.field_tiles = field_tiles  # before the solver: the system hash a distilled solver is checked by reads it
        self.solver = solver
        self.score = score
        self.search = search
        self._config = None
        self.setup()

    @classmethod
    def for_system(cls, system: ImagingSystem, **parts: Unpack[Parts]) -> Self:
        """Build the model of a system you already have, as an instance of the class that wraps it.

        The class is the one whose :attr:`system_types` list the system's type (else one it is an instance of,
        else the class this is called on). Its ``__init__`` is not called; :meth:`setup` is. The system is taken,
        not copied: two models of one system share its values and its placement.
        """
        expect_instance(system, ImagingSystem, where=f"{cls.__name__}.for_system", name="system")
        found = _model_class(system)
        model = object.__new__(found if issubclass(found, cls) else cls)
        ReconstructionModel.__init__(model, system, **parts)
        return model

    def _expect_system(self, system: object) -> None:
        where = type(self).__name__
        expect_instance(system, ImagingSystem, where=where, name="system")
        if self.system_types:
            if not isinstance(system, self.system_types):
                wrapped = " or ".join(kind.__name__ for kind in self.system_types)
                raise ParameterError(
                    f"{where} system must be a {wrapped}, got {type(system).__name__}; "
                    "ReconstructionModel.for_system(system) builds the class that wraps it",
                    path="system",
                )
            return
        listed = _model_class(cast(ImagingSystem, system))
        if type(self) is ReconstructionModel and listed is not ReconstructionModel:
            raise ConfigurationError(
                "system",
                f"is a {type(system).__name__}, which {listed.__name__} wraps; call "
                f"ReconstructionModel.for_system(system, ...), which returns a {listed.__name__}",
            )

    def __repr__(self) -> str:
        parts = {"system": self.system, "object_prior": self.object_prior, "noise": self.noise, "solver": self.solver}
        named = ", ".join(f"{name}={type(part).__name__}" for name, part in parts.items())
        return f"{type(self).__name__}({named}, device={str(self.device)!r})"

    # --- the parts -------------------------------------------------------------------------------------------------

    @property
    def imaging_model(self) -> ImagingModel:
        """The module tree: ``system``, ``object_prior`` and ``noise``. Replace a part through the model."""
        return self._imaging_model

    @property
    def system(self) -> ImagingSystem:
        """The imaging system ``H``."""
        return self.imaging_model.system

    @property
    def object_prior(self) -> ObjectPrior:
        """The prior on the object; a default solver is chosen again when it is assigned."""
        return self.imaging_model.object_prior

    @object_prior.setter
    def object_prior(self, object_prior: ObjectPrior) -> None:
        expect_instance(object_prior, ObjectPrior, where=type(self).__name__, name="object_prior")
        self._replace({"object_prior": self._placed(object_prior)}, defaults={})

    @property
    def noise(self) -> Noise:
        """The noise model; a default prior and a default solver follow it when it is assigned."""
        return self.imaging_model.noise

    @noise.setter
    def noise(self, noise: Noise) -> None:
        expect_instance(noise, Noise, where=type(self).__name__, name="noise")
        defaults: dict[str, nn.Module] = {}
        if "object_prior" in self.defaulted:
            defaults["object_prior"] = self._placed(self._default_prior(noise))
        self._replace({"noise": self._placed(noise)} | defaults, defaults=defaults)

    def _replace(self, parts: Mapping[str, nn.Module], *, defaults: Mapping[str, nn.Module]) -> None:
        """Take ``parts`` into the imaging model; put them back and raise if the solver cannot run on them."""
        before = {name: getattr(self.imaging_model, name) for name in parts}
        for name, part in parts.items():
            setattr(self.imaging_model, name, part)
        try:
            self._settle_solver(parts)
        except IncompatibleModelError:
            for name, part in before.items():
                setattr(self.imaging_model, name, part)
            raise
        self._defaults = {name: part for name, part in self._defaults.items() if name not in parts} | dict(defaults)
        self._config = None

    def _default_prior(self, noise: Noise) -> ObjectPrior:
        """Return the default prior; under noise that counts photons, :class:`~waveorder.nn.NoPrior`."""
        return NoPrior() if noise.has_photon_counts else self.default_object_prior()

    def _settle_solver(self, changed: Iterable[str]) -> None:
        if self._solver_is_default:
            self._choose_solver()
            return
        try:
            self._solver.check(self.imaging_model)
            self._expect_reconstructs(f"{type(self).__name__} solver")
        except IncompatibleModelError as error:
            new = " and ".join(f"{name}={type(getattr(self, name)).__name__}" for name in changed)
            raise IncompatibleModelError(
                f"{type(self).__name__}: model.solver is the {type(self._solver).__name__} the model was given, "
                f"which cannot run with {new}: {error}. Assign model.solver = None first, so that the solver follows "
                "the parts, or a solver that takes them"
            ) from error

    @property
    def solver(self) -> Solver:
        """The reconstruction. Assigning ``None`` takes :meth:`default_solver` again."""
        return self._solver

    @solver.setter
    def solver(self, solver: Solver | None) -> None:
        where = type(self).__name__
        expect_instance(solver, (Solver, type(None)), where=where, name="solver")
        if solver is None:
            self._choose_solver()
        else:
            try:
                solver.check(self.imaging_model)
            except IncompatibleModelError as error:
                prior, noise = type(self.object_prior).__name__, type(self.noise).__name__
                raise IncompatibleModelError(
                    f"{where} solver: {error}. The model has object_prior={prior} and noise={noise}; assign "
                    "model.solver = None for the solver that follows them"
                ) from error
            self._expect_reconstructs(f"{where} solver")
            self._expect_trained_system(solver, where=where)
            self._solver = self._placed(solver)
            self._solver_is_default = False
            self._fill_score()
        self._config = None

    def _expect_reconstructs(self, where: str) -> None:
        """Refuse to reconstruct with a calibration-only object prior (:class:`~waveorder.nn.priors.Beads`)."""
        if isinstance(self.object_prior, Beads):
            raise IncompatibleModelError(
                f"{where}: a Beads object prior calibrates the system (calibrate, calibrate_position, "
                "wo calibrate-tf); it does not reconstruct. Build the model without a solver to calibrate, and "
                "reconstruct with an object prior that has a penalty, such as Tikhonov"
            )

    def _expect_trained_system(self, solver: Solver, *, where: str) -> None:
        """Refuse a distilled ``solver`` whose contract records the hash of another imaging system."""
        trained = solver.contract.system.get("sha256") if isinstance(solver, Distilled) else None
        own = None if trained is None else _stores.system_sha256(self)
        if own is None or own == trained:
            return
        raise IncompatibleModelError(
            f"{where} solver: the student was trained for another imaging system: its contract records the imaging "
            f"system sha256 {trained}, and this model's imaging system hashes to {own}"
        )

    def _choose_solver(self) -> None:
        solver = self.default_solver()
        expect_instance(solver, Solver, where=f"{type(self).__name__}.default_solver", name="return value")
        try:
            solver.check(self.imaging_model)
        except IncompatibleModelError as error:
            raise self._no_default_solver(solver, error) from error
        self._solver = self._placed(solver)
        self._solver_is_default = True
        self._fill_score()

    def _no_default_solver(self, solver: Solver, error: IncompatibleModelError) -> IncompatibleModelError:
        system, prior, noise = (type(part).__name__ for part in (self.system, self.object_prior, self.noise))
        parts = f"{system} with object_prior={prior} and noise={noise}"
        if not self.noise.has_photon_counts:
            return IncompatibleModelError(
                f"{type(self).__name__}: the default solver {type(solver).__name__} cannot run on {parts}: {error}. "
                "Pass a solver that takes these parts as solver=..., or change the part the message names"
            )
        rule = (
            f"{type(self).__name__}: no solver runs on {parts} by default. Noise that counts photons is solved by "
            "Richardson-Lucy, which takes no prior and needs a system that maps a nonnegative object to "
            "nonnegative data"
        )
        if not self.system.preserves_nonnegativity:
            return IncompatibleModelError(
                f"{rule}; {system} does not declare preserves_nonnegativity. Pass noise=Gaussian(sigma=...); or, for "
                "a system whose point-spread function is nonnegative, set preserves_nonnegativity = True on its class"
            )
        return IncompatibleModelError(
            f"{rule}. Pass object_prior=NoPrior() with this noise, or keep the prior under noise=Gaussian(sigma=...)"
        )

    @property
    def score(self) -> Score | None:
        """The calibration objective; ``None`` is :class:`~waveorder.nn.Evidence`, made when a calibration runs."""
        return self._score

    @score.setter
    def score(self, score: Score | None) -> None:
        expect_instance(score, (Score, type(None)), where=type(self).__name__, name="score")
        if isinstance(score, SharpnessScore):
            score.fill(self.system, solver=self._solver)  # before it is kept: a system without a cutoff raises
        self._score = None if score is None else self._placed(score)
        self._config = None

    def _fill_score(self) -> None:
        if isinstance(self._score, SharpnessScore):
            self._score.fill(self.system, solver=self._solver)

    @property
    def search(self) -> tuple[SearchStage, ...] | None:
        """The calibration stages, in order; ``None`` runs the default stages of :func:`~waveorder.nn.calibrate`."""
        return self._search

    @search.setter
    def search(self, search: Sequence[SearchStage] | None) -> None:
        where = type(self).__name__
        expect_instance(search, (list, tuple, type(None)), where=where, name="search")
        for index, stage in enumerate(search or ()):
            expect_instance(stage, SearchStage, where=where, name=f"search.{index}")
        self._search = None if search is None else tuple(search)
        self._config = None

    @property
    def field_tiles(self) -> FieldTiles | None:
        """The field tiles of a system that varies over the field of view, if any."""
        return self._field_tiles

    @field_tiles.setter
    def field_tiles(self, field_tiles: FieldTiles | None) -> None:
        expect_instance(field_tiles, (FieldTiles, type(None)), where=type(self).__name__, name="field_tiles")
        self._field_tiles = field_tiles
        self._config = None

    @property
    def tile_parameters(self) -> dict[str, Tensor] | None:
        """Per-field-tile values of the free quantities, ``{name: (tiles, ...)}``, which tiled solves use.

        Set by :meth:`load_calibration`, by :meth:`calibrate_position` on a model with field tiles, or assigned
        (``model.calibrate_tiles(...).parameters``).
        """
        return self._tile_parameters

    @tile_parameters.setter
    def tile_parameters(self, table: Mapping[str, Tensor] | None) -> None:
        self._tile_parameters = (
            None if table is None else _tile_table(table, self.imaging_model, where=type(self).__name__)
        )

    @property
    def z_padding(self) -> ZPaddingSetting:
        """Mirrored z slices added around each solve, or ``"auto"`` (``model.z_padding = 0`` to calibrate unpadded).

        :meth:`z_padding_for` gives the slices of a volume.
        """
        return self._z_padding

    @z_padding.setter
    def z_padding(self, z_padding: ZPaddingSetting) -> None:
        where = type(self).__name__
        expect_instance(z_padding, (int, str), where=where, name="z_padding")
        if z_padding != "auto" and (isinstance(z_padding, str) or z_padding < 0):
            raise ParameterError(
                f"{where} z_padding must be at least 0, or 'auto', got {z_padding!r}", path="z_padding"
            )
        self._z_padding = z_padding
        self._config = None

    def z_padding_for(self, shape: ZYX[int], *, pixel_size: PixelSize | None = None) -> int:
        """Return the z slices mirrored at each end of a volume of ``shape`` (z, y, x) before a solve.

        An integer :attr:`z_padding` is that integer. ``"auto"`` is at least the axial extent of the point-spread
        function, ``n * wavelength / NA**2`` (the medium's index, the detection's wavelength and nominal numerical
        aperture), in whole z voxels rounded up, then the fewest more slices that make the padded stack a fast FFT
        length (no prime factor above 7). ``pixel_size`` is as :meth:`reconstruct` takes it.

        Raises
        ------
        ParameterError
            For ``"auto"`` without a voxel size, or on a system without a detection and a medium's index.
        ShapeError
            For ``"auto"`` on a stack with no more slices than the axial extent.
        """
        if isinstance(self._z_padding, int):
            return self._z_padding
        where = f"{type(self).__name__}.z_padding_for"
        size = self._size(pixel_size, where)
        sampling = Sampling(shape=zyx(shape, where=where, name="shape", minimum=1), pixel_size=size)
        return self._padding_for(sampling).slices

    def _padding_for(self, sampling: Sampling) -> ZPadding:
        """Return the z padding of a solve of a volume on ``sampling``, the measured grid."""
        if isinstance(self._z_padding, int):
            return ZPadding(self._z_padding)
        return ZPadding(auto_z_padding(self.system, sampling))

    @property
    def pixel_size(self) -> ZYX[float] | None:
        """The voxel size ``(z, y, x)`` in um the methods use; ``None`` when the data gives it (``"auto"``)."""
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, pixel_size: PixelSize | Literal["auto"] | None) -> None:
        auto = pixel_size is None or (isinstance(pixel_size, str) and pixel_size == "auto")
        self._pixel_size = None if auto else _arguments.pixel_size(pixel_size, where=type(self).__name__)

    @property
    def apodization_rolloff(self) -> float:
        """The roll-off of a Wiener estimate to Nyquist, a fraction of the band in [0, 1]."""
        return self._rolloff

    @apodization_rolloff.setter
    def apodization_rolloff(self, rolloff: float) -> None:
        where = type(self).__name__
        expect_instance(rolloff, (float, int), where=where, name="apodization_rolloff")
        if not 0 <= rolloff <= 1:
            raise ParameterError(
                f"{where} apodization_rolloff must be in [0, 1], got {rolloff}", path="apodization_rolloff"
            )
        self._rolloff = float(rolloff)

    @property
    def device(self) -> torch.device:
        """Where the model computes (:meth:`to` moves it)."""
        return placement(self.imaging_model)[1]

    @property
    def dtype(self) -> torch.dtype:
        """The real dtype the model computes in (:meth:`to` sets it)."""
        return placement(self.imaging_model)[0]

    @property
    def config(self) -> V4Config | None:
        """The configuration :meth:`from_config` built the model from, while its parts are still those it built."""
        current = (self.system, self.object_prior, self.noise)
        built = len(self._built) == len(current) and all(map(operator.is_, current, self._built))
        return self._config if built else None

    @property
    def defaulted(self) -> tuple[str, ...]:
        """Which of ``"object_prior"`` and ``"noise"`` are still the defaults of arguments that were not given."""
        return tuple(name for name, part in self._defaults.items() if getattr(self, name) is part)

    def _placed[M: nn.Module](self, module: M) -> M:
        dtype, device = placement(self.imaging_model)
        return placed(module, device=device, dtype=dtype)

    def _remember(self, config: V4Config, left_at_default: Sequence[str]) -> None:
        """Keep ``config`` as the configuration this model was built from, with the parts it left at their defaults."""
        self._config = config
        self._built = (self.system, self.object_prior, self.noise)
        self._defaults |= {name: getattr(self, name) for name in left_at_default}

    # --- hooks a modality overrides --------------------------------------------------------------------------------

    def setup(self) -> None:
        """Set the state the methods need beyond the parts; runs at the end of every construction."""

    def default_object_prior(self) -> ObjectPrior:
        """Return the object prior of a model built without one: Tikhonov at the configuration's default weight."""
        weight = ObjectSettings().weight
        return Tikhonov(weight=float(weight) if isinstance(weight, int | float) else 1e-3)

    def default_noise(self) -> Noise:
        """Return the noise of a model built without one: Gaussian at the configuration's default sigma."""
        sigma = GaussianNoise().sigma
        return Gaussian(sigma=float(sigma) if isinstance(sigma, int | float) else 1.0)

    def default_solver(self) -> Solver:
        """Return the solver of a model built without one, from its parts.

        :class:`~waveorder.nn.RichardsonLucy` for photon-counting noise without a prior; the closed-form
        :class:`~waveorder.nn.Wiener` wherever it runs; else :class:`~waveorder.nn.FISTA`.
        """
        if self.noise.has_photon_counts and isinstance(self.object_prior, NoPrior):
            return RichardsonLucy()
        closed_form = Wiener()
        try:
            closed_form.check(self.imaging_model)
        except IncompatibleModelError:
            return FISTA()
        return closed_form

    def prepare(self, data: Tensor, *, mean: Tensor | None = None) -> Tensor:
        """Turn recorded data ``(C, Z, Y, X)`` into what the system measures; the base class returns it as recorded.

        ``mean`` is the per-channel mean of the whole stack, given by tiled work to a class whose
        :attr:`prepare_scope` is ``"stack_mean"``.
        """
        del mean
        return data

    def output_channel_names(self, channels: Sequence[str]) -> list[str]:
        """Return the names of the object fields a reconstruction writes, given the input channel names."""
        del channels
        return list(self.system.fields)

    # --- verbs on tensors ------------------------------------------------------------------------------------------

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> Self:
        """Move the model (its modules and :attr:`tile_parameters`) to ``device`` and ``dtype``; return it.

        ``dtype`` is float32 or float64; fixed quantities are filled again from the numbers they were given as.
        """
        where = f"{type(self).__name__}.to"
        if isinstance(device, torch.dtype):
            raise ParameterError(f"{where} device is the dtype {device}; pass it as dtype={device}", path="device")
        expect_instance(device, (str, torch.device, type(None)), where=where, name="device")
        expect_instance(dtype, (torch.dtype, type(None)), where=where, name="dtype")
        kind, target = placement(self.imaging_model)
        if device is not None:
            target = resolve_device(str(device), field="device")
        if dtype is not None:
            if dtype not in (torch.float32, torch.float64):
                raise ParameterError(f"{where} dtype must be torch.float32 or torch.float64, got {dtype}", path="dtype")
            kind = dtype
        for module in (self.imaging_model, self._solver, self._score):
            if module is not None:
                placed(module, device=target, dtype=kind)
        self.tile_parameters = self._tile_parameters  # the setter moves the table with its quantities
        return self

    def check(self, shape: ZYX[int], *, pixel_size: PixelSize | None = None, calibration: bool = False) -> None:
        """Raise if the model cannot run on a volume of ``shape`` (z, y, x), before data is read.

        ``calibration=True`` checks what :meth:`calibrate` needs: a free quantity and a score that can judge the
        model; otherwise what :meth:`reconstruct` needs, which a :class:`~waveorder.nn.priors.Beads` object prior (a
        calibration prior) does not have. ``pixel_size`` is as :meth:`reconstruct` takes it.
        """
        where = f"{type(self).__name__}.check"
        size = self._size(pixel_size, where)
        sampling = Sampling(shape=zyx(shape, where=where, name="shape", minimum=1), pixel_size=size)
        expect_instance(calibration, bool, where=where, name="calibration")
        if not calibration:
            self._expect_reconstructs(where)
        with running(self, "check", reported=False):
            padded = self._padding_for(sampling).sampling(sampling)
            self.imaging_model.check(self.solver, padded)
            self._expect_trained_system(self.solver, where=where)
            if calibration and isinstance(self.object_prior, Beads):
                self._expect_beads(None, where=where)  # a bead fit scores by its own likelihood
            elif calibration:
                self.imaging_model.check(calibration_score(self, where=where, warn=False), padded)
            if calibration or self.solver.needs_operator:
                with torch.no_grad():  # a transfer without a graph is cached, so the solve that follows reuses it
                    bind(self, padded, stored=not calibration)

    def reconstruct(self, data: object, *, pixel_size: PixelSize | None = None, prepared: bool = False) -> SolveResult:
        """Estimate the object from recorded data with the model's solver and current values.

        The data is moved to the model's device, z-padded, prepared (:meth:`prepare`), cast to the model's dtype
        and solved without an autograd graph; a Wiener estimate is rolled off by :attr:`apodization_rolloff`.

        Parameters
        ----------
        data : array-like
            The recorded data ``(C, Z, Y, X)``, or ``(Z, Y, X)`` for one channel, in any real dtype.
        pixel_size : float, (z, yx) or (z, y, x), optional
            Voxel size in um for this call; by default the model's :attr:`pixel_size`.
        prepared : bool
            The data is already what the system measures (such as :meth:`simulate` returns), so :meth:`prepare`
            is skipped.

        Returns
        -------
        SolveResult
            ``estimate`` ``(F, Z, Y, X)`` on the model's device, with ``status`` and ``finite``.

        Raises
        ------
        ParameterError
            If ``data`` is no array, or no voxel size is known: set ``data.pixel_size`` in the configuration or
            pass ``pixel_size`` to the constructor.
        ShapeError
            If ``data`` is complex or not shaped as the system measures.
        ConfigurationError
            For a model with :attr:`field_tiles`, or an apodization roll-off with a solver other than Wiener.
        IncompatibleModelError
            For a :class:`~waveorder.nn.priors.Beads` object prior, which calibrates and does not reconstruct.
        """
        where = f"{type(self).__name__}.reconstruct"
        recorded, sampling = self._recorded(where, data, pixel_size)
        expect_instance(prepared, bool, where=where, name="prepared")
        self._expect_whole_field(where)
        self._expect_reconstructs(where)
        rolloff = self._rolloff
        if rolloff and not isinstance(self.solver, Wiener):
            raise self._refused(
                "solver.apodization_rolloff",
                "apodization_rolloff",
                f"applies to a Wiener estimate, not to {type(self.solver).__name__}'s",
            )
        with running(self, "reconstruct"), torch.no_grad():
            padding = self._padding_for(sampling)
            padded = padding.sampling(sampling)
            self.imaging_model.check(self.solver, padded)  # the model and its grid, before a transfer is computed
            measured = self._measured(recorded, padding, prepared=prepared)
            operator = bind(self, padded, stored=True) if self.solver.needs_operator else None
            result = padding.crop(self.solver(measured, self.imaging_model, padded, operator=operator))
            if rolloff:
                result = dataclasses.replace(result, estimate=apodize(result.estimate, rolloff))
            return result

    def calibrate(
        self,
        data: object,
        *,
        pixel_size: PixelSize | None = None,
        prior: ParameterPrior | TfStore | None = None,
        prepared: bool = False,
        history: bool = False,
    ) -> CalibrationResult | BeadCalibration:
        """Estimate the free quantities from recorded data with the model's score and search; updates the model.

        The search starts at the model's current values; ``data``, ``pixel_size`` and ``prepared`` are as
        :meth:`reconstruct` takes them. Parts that start from the data (``Gaussian(sigma="estimate")``) set their
        start and bounds before the prior is built. Before a :class:`~waveorder.nn.search.sweep.WeightSweep` stage
        the recorded data is cropped to its most structured 256-pixel square (:func:`~waveorder.nn.preprocess.
        most_structured_crop`), then padded and prepared alone, as wo-4.0 normalized its crop by the crop's own mean.
        A :class:`~waveorder.nn.priors.Beads` object prior is calibrated by
        :func:`~waveorder.nn.search.beads.calibrate_beads`, which returns the beads too.

        A likelihood score (the evidence) compares the data with the noise and the object prior, so both must be in
        the units of the data after :meth:`prepare`; defaults and z padding bias it, and are warned about.

        Parameters
        ----------
        prior : ParameterPrior or TfStore, optional
            Prior over the free quantities; a calibration store's values become the prior centres. By default the
            one the quantities' tolerances declare.
        history : bool
            Also return what the stages saw: the objective at each gradient step, the last grid over every free
            quantity, and a weight sweep's result with the crop it scored (:class:`~waveorder.nn.CalibrationResult`).
            A bead calibration always returns its trace.

        Raises
        ------
        ParameterError
            As :meth:`reconstruct`; if the model has no free quantity.
        ConfigurationError
            For a model with :attr:`field_tiles`; a bead calibration of a model that is no fluorescence model, has
            no Gaussian noise, or searches by other stages than one gradient search.

        Warns
        -----
        CalibrationWarning
            If a likelihood score meets a default noise or object prior (:attr:`defaulted`), or z padding.
        """
        where = f"{type(self).__name__}.calibrate"
        recorded, sampling = self._recorded(where, data, pixel_size)
        expect_instance(prepared, bool, where=where, name="prepared")
        expect_instance(history, bool, where=where, name="history")
        self._stored_transfers = None  # the values it finds are not those of the store's transfer functions
        if isinstance(self.object_prior, Beads):
            self._expect_beads(prior, where=where)
            stages = [stage for stage in self.search or () if isinstance(stage, GradientSearch)]
            iterations, rate = (stages[0].iterations, stages[0].learning_rate) if stages else (_BEAD_ITERATIONS, 0.01)
            with running(self, "calibrate"):
                padding = self._padding_for(sampling)
                measured = self._measured(recorded, padding, prepared=prepared)
                padded = padding.sampling(sampling)
                self._start_from(measured)
                return calibrate_beads(measured, self.imaging_model, padded, iterations=iterations, learning_rate=rate)
        self._expect_whole_field(where)
        score = calibration_score(self, where=where)
        with running(self, "calibrate"):
            padding = self._padding_for(sampling)
            padded = padding.sampling(sampling)
            self.imaging_model.check(score, padded)  # the model and its grid, before a transfer is computed
            region = None
            search = self.search
            if any(isinstance(stage, WeightSweep) for stage in search or ()):
                region = most_structured_crop(recorded, size=256)
                y, x, side = region
                recorded = recorded[..., y : y + side, x : x + side]
                padded = dataclasses.replace(padded, shape=(padded.shape[0], side, side))
                if self._z_padding == "auto":  # the sweep leaves out the slices auto chose for this stack
                    search = tuple(
                        dataclasses.replace(stage, z_padding=padding.slices)
                        if isinstance(stage, WeightSweep)
                        else stage
                        for stage in search or ()
                    )
            measured = self._measured(recorded, padding, prepared=prepared)
            bind(self, padded)
            self._start_from(measured)
            if prior is not None and not isinstance(prior, ParameterPrior):
                prior = prior.to_prior(free_specs(self.imaging_model))
            result = calibrate(
                measured, self.imaging_model, padded, score=score, search=search, prior=prior, history=history
            )
            if result.sweep is not None:
                result = dataclasses.replace(result, sweep=dataclasses.replace(result.sweep, region=region))
            return result

    def _start_from(self, data: Tensor) -> None:
        """Let the parts that start from the data set their start and bounds from ``data``."""
        for part in self.imaging_model.modules():
            if isinstance(part, Gaussian):
                part.start_from(data)

    def _expect_beads(self, prior: object, *, where: str) -> None:
        """Refuse a bead calibration the bead fit cannot run, with wo-4.0's reasons."""
        if not isinstance(self.system, WidefieldFluorescence3D):  # confocal included: it subclasses widefield
            raise self._refused(
                "imaging_system.contrast",
                "system",
                "bead calibration supports fluorescence contrast in 3D (widefield or confocal): it fits a 3D bead "
                f"volume, which a {type(self.system).__name__} does not image",
            )
        if self.field_tiles is not None:
            raise self._refused(
                "imaging_system.field_dependence", "field_tiles", "bead calibration is shift-invariant for now"
            )
        if not isinstance(self.noise, Gaussian):
            raise self._refused(
                "noise", "noise", f"bead calibration supports gaussian noise for now, got {type(self.noise).__name__}"
            )
        if any(not isinstance(stage, GradientSearch) for stage in self.search or ()) or len(self.search or ()) > 1:
            raise self._refused(
                "solver.calibration.search",
                "search",
                "bead calibration runs one gradient search; remove the other stages",
            )
        if prior is not None:
            raise ParameterError(f"{where}: a prior with prior: beads is not supported yet", path="prior")

    def simulate(
        self,
        obj: Tensor | None = None,
        *,
        shape: ZYX[int] | None = None,
        noise: bool = True,
        seed: int | torch.Generator | None = None,
        pixel_size: PixelSize | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return data ``H f + n`` that the model's imaging model gives for an object, and the object.

        Without ``obj`` the object is a zero-mean, unit-variance Gaussian random field of power spectrum
        ``(|k|^2 + k0^2)^(-exponent/2)`` per field (``k`` in cycles per voxel, ``k0 = 1 / max(shape)``; the exponent
        of the configuration's ``object.source.gaussian_spectrum``, else 2). The noise is the model's, or, from a
        configuration with a ``simulate`` block, Gaussian at ``noise.gaussian.snr_db`` below the clean data's
        standard deviation, or Poisson of ``noise.poisson.photons`` mean counts of the clean data shifted to 0. The
        draws come from one CPU generator: object, then noise.

        Parameters
        ----------
        obj : Tensor, optional
            The object ``(F, Z, Y, X)`` in the units of the system's object.
        shape : tuple[int, int, int], optional
            The data grid ``(Z, Y, X)`` of a drawn object; by default the configuration's ``simulate.shape``.
        noise : bool
            Draw the noise; without it the data is ``H f``.
        seed : int or torch.Generator, optional
            Seed of the draws, or the generator to draw from; by default the configuration's ``simulate.seed``,
            else a fresh seed.
        pixel_size : float, (z, yx) or (z, y, x), optional
            As :meth:`reconstruct` takes it.

        Returns
        -------
        tuple[Tensor, Tensor]
            The data ``(C, Z, Y, X)`` as the system measures it (no :meth:`prepare` inverse) and the object
            ``(F, Z', Y, X)``, both on the model's device in its dtype.
        """
        where = f"{type(self).__name__}.simulate"
        config = self.config
        simulated = None if config is None else config.simulate
        expect_instance(noise, bool, where=where, name="noise")
        if isinstance(seed, torch.Generator):
            generator = seed
        else:
            expect_instance(seed, (int, type(None)), where=where, name="seed")
            generator = torch.Generator()
            seed = simulated.seed if seed is None and simulated is not None else seed
            generator.manual_seed(seed) if seed is not None else generator.seed()
        size = self._size(pixel_size, where)
        if obj is None:
            shape = simulated.shape if shape is None and simulated is not None else shape
            if shape is None:
                raise ParameterError(
                    f"{where} needs an object or the shape (Z, Y, X) of the data to draw one", path="shape"
                )
            sampling = Sampling(shape=zyx(tuple(shape), where=where, name="shape", minimum=1), pixel_size=size)
            source = None if config is None else config.object.source
            exponent = source.gaussian_spectrum.exponent if isinstance(source, SourceGaussianSpectrum) else 2.0
            fields, *grid = self.system.object_shape(sampling)
            obj = torch.stack([_power_law((grid[0], grid[1], grid[2]), exponent, generator) for _ in range(fields)])
        else:
            expect_instance(obj, Tensor, where=where, name="obj")
            if obj.ndim != 4:
                raise ShapeError(f"{where} obj must be (F, Z, Y, X), got shape {tuple(obj.shape)}")
            planes = obj.shape[1] if shape is None else zyx(tuple(shape), where=where, name="shape", minimum=1)[0]
            sampling = Sampling(shape=(planes, obj.shape[2], obj.shape[3]), pixel_size=size)
        dtype, device = placement(self.imaging_model)
        obj = obj.to(device=device, dtype=dtype)
        config = None if simulated is None else config
        with torch.no_grad(), running(self, "simulate"):  # a free noise parameter must not reach the data either
            clean = self.imaging_model(obj, sampling).cpu()
            data = _noisy(clean, self.noise, config=config, generator=generator) if noise else clean
        return data.to(device=device, dtype=dtype), obj

    def train(  # noqa: PLR0913 (the index, and one keyword per setting a configuration holds)
        self,
        index: str | Path,
        *,
        student: RestorationNetwork | None = None,
        out: str | Path | None = None,
        patch: int = 64,
        batch_size: int = 4,
        workers: int = 0,
        epochs: int = 60,
        seed: int = 0,
        module: Mapping[str, object] | None = None,
        data: Mapping[str, object] | None = None,
        trainer: Mapping[str, object] | None = None,
    ) -> TrainResult[Self]:
        """Distil the model's solver into a network on recorded pairs; return the run and a model that runs it.

        A thin wrapper over ``wo distill fit``: the model is written to ``out/model``, the configuration a file
        would hold is built from the arguments, and :class:`~waveorder.models.DistillCLI` builds the same
        :class:`~waveorder.models.DistillationModule`, :class:`~waveorder.train.lightning.PatchDataModule` and
        trainer the command does. The run folder is the command's: ``config.yaml``, ``run.json``, ``best/``,
        ``last/`` and ``checkpoints/``. The model is not changed; ``result.model`` is a copy whose solver is the
        trained :class:`~waveorder.nn.Distilled` network.

        Parameters
        ----------
        index : str or Path
            The index directory of the training fields (``wo distill index``), whose targets this model made.
        student : RestorationNetwork, optional
            The network; by default a strided U-Net over the whole stack, one plane per target channel.
        out : str or Path, optional
            The run folder; a new temporary one by default.
        patch : int
            Side of the square training patches, in pixels (``data.patch_size``).
        batch_size, workers : int
            Patches per batch; loader processes, 0 reading in this process (``data.batch_size``, ``data.workers``).
        epochs : int
            ``trainer.max_epochs``.
        seed : int
            ``seed_everything``.
        module, data, trainer : Mapping, optional
            Any other argument of the ``model``, ``data`` and ``trainer`` blocks of a configuration, by name; a part
            may be given as an object (``{"losses": {"l1": L1(), "ssim": SSIM(window=7)}}``).

        Raises
        ------
        ConfigurationError
            At the configuration key (``data.batch_size``, ``model.student.head``) of an argument the configuration
            refuses, as ``wo distill fit`` names it.
        """
        from waveorder.data import FovIndex
        from waveorder.models._lightning import DistillCLI, configured
        from waveorder.nn.networks import ResNetEncoder, StridedVolume, SubPixelHead, UNet
        from waveorder.train.lightning import metrics_of, quiet

        run = Path(tempfile.mkdtemp(prefix="waveorder-train-")) if out is None else Path(out)
        if student is None:
            found = FovIndex.load(index)
            (planes,) = found.planes[:1] or (1,)
            head = SubPixelHead(len(found.target_channels))
            student = UNet(StridedVolume(planes), ResNetEncoder(), head, decoder_channels=(256, 128, 64))
        logger = {"save_dir": str(run), "name": "", "version": ""}  # the run folder is out itself
        config = {
            "seed_everything": seed,
            "trainer": {
                "max_epochs": epochs,
                "logger": {"class_path": "lightning.pytorch.loggers.CSVLogger", "init_args": logger},
                **cast(dict, configured(dict(trainer or {}))),
            },
            "model": {
                "reconstruction": str(self.save(run / "model", overwrite=True)),
                "student": configured(student),
                **cast(dict, configured(dict(module or {}))),
            },
            "data": {
                "index": str(index),
                "patch_size": [patch, patch],
                "batch_size": batch_size,
                "workers": workers,
                **cast(dict, configured(dict(data or {}))),
            },
        }
        with quiet():
            cli = DistillCLI(config, run=False)
            cli.trainer.fit(cli.model, datamodule=cli.datamodule)
        directory = Path(str(cli.trainer.log_dir))
        bundle = directory / ("best" if (directory / "best").is_dir() else "last")
        trained = copy.deepcopy(self)
        trained.solver = Distilled(load_bundle(bundle))
        return TrainResult(
            bundle=bundle,
            metrics=metrics_of(cli.trainer),
            checkpoint=directory / "checkpoints" / "last.ckpt",
            directory=directory,
            model=trained,
        )

    def export(self, path: str | Path) -> Path:
        """Write the network of a model whose solver is a network as an ONNX file (:func:`waveorder.train.export_onnx`).

        Raises
        ------
        ConfigurationError
            At ``solver``, if the solver is no :class:`~waveorder.nn.Distilled` network loaded from a bundle.
        """
        from waveorder.models._distill import export

        solver = self.solver
        if not isinstance(solver, Distilled) or solver.bundle.path is None:
            raise ConfigurationError(
                "solver",
                f"is {type(solver).__name__}; export writes a network, so give the model one (model.train(...).model, "
                "or solver=Distilled(waveorder.io.load_bundle(bundle)))",
            )
        return export(solver.bundle, path)

    # --- tile methods ----------------------------------------------------------------------------------------------

    def tile_plan(
        self, shape: ZYX[int], *, tile: ZYX[int], overlap: ZYX[int] = (0, 0, 0), pixel_size: PixelSize | None = None
    ) -> TilePlan:
        """Plan tiles of a ``(Z, Y, X)`` volume, with the halo that keeps 99 % of the solver's filter per axis.

        ``tile`` is the voxel count of a tile core and ``overlap`` the voxels each core is widened by and cross-faded
        with its neighbours; ``pixel_size`` is as :meth:`reconstruct` takes it. The halo is computed at the sampling of
        one tile's read region.
        """
        where = f"{type(self).__name__}.tile_plan"
        size = _arguments.pixel_size(pixel_size, where=where, default=self.pixel_size)
        volume = zyx(shape, where=where, name="shape", minimum=1)
        core = zyx(tile, where=where, name="tile", minimum=1)
        widened = zyx(overlap, where=where, name="overlap", minimum=0)

        def read(halo: ZYX[int]) -> Sampling:
            z, y, x = (min(n, t + 2 * (o + h)) for n, t, o, h in zip(volume, core, widened, halo, strict=True))
            return Sampling(shape=(z, y, x), pixel_size=size)

        self.imaging_model.check(self.solver, read((0, 0, 0)))  # the grid is judged before a filter is computed on it
        halo = halo_for(self.imaging_model, read((0, 0, 0)), energy=_HALO_ENERGY, solver=self.solver)
        halo = halo_for(self.imaging_model, read(halo), energy=_HALO_ENERGY, solver=self.solver)
        return TilePlan(shape=volume, tile=core, overlap=widened, halo=halo)

    def field_tile_plan(self, shape: ZYX[int], *, pixel_size: PixelSize | None = None) -> TilePlan:
        """Plan the :attr:`field_tiles` over a ``(Z, Y, X)`` volume, as :meth:`tile_plan` plans tiles; each spans z."""
        where = f"{type(self).__name__}.field_tile_plan"
        size = _arguments.pixel_size(pixel_size, where=where, default=self.pixel_size)
        tiling = self.field_tiles
        if tiling is None:
            raise self._refused(
                "imaging_system.field_dependence", "field_tiles", "is not set, so there are no field tiles to plan"
            )
        z, _, _ = zyx(shape, where=where, name="shape", minimum=1)
        (ty, tx), (oy, ox) = tiling.size_yx, tiling.overlap_yx
        plan = self.tile_plan(shape, pixel_size=size, tile=(z, ty, tx), overlap=(0, oy, ox))
        return dataclasses.replace(plan, blend=tiling.blend)

    def calibrate_tiles(  # noqa: PLR0913 (the source, and one keyword per argument of calibrate and of a tiling)
        self,
        source: ArrayLike,
        *,
        plan: TilePlan,
        start: Mapping[str, Tensor] | None = None,
        chunk_size: int | None = None,
        prior: ParameterPrior | TfStore | None = None,
        prepared: bool = False,
        pixel_size: PixelSize | None = None,
    ) -> TileCalibration:
        """Calibrate the free quantities on every tile of ``plan`` independently; the model does not take the estimates.

        ``source`` is any array with ``shape`` and numpy slicing, read tile by tile through :meth:`prepare`.
        ``start`` holds per-tile start values ``{name: (tiles,)}``, ``chunk_size`` the tiles calibrated at once;
        ``prior`` (one row, or one per tile), ``prepared`` and ``pixel_size`` are as :meth:`calibrate` takes them. As in
        :meth:`calibrate`, parts that start from the data (``Gaussian(sigma="estimate")``) set their start and bounds
        from all the tiles before the prior is built. Pass the result's ``parameters`` to :meth:`reconstruct_tiles`.
        """
        where = f"{type(self).__name__}.calibrate_tiles"
        pixel_size = _arguments.pixel_size(pixel_size, where=where, default=self.pixel_size)
        expect_instance(plan, TilePlan, where=where, name="plan")
        expect_instance(prepared, bool, where=where, name="prepared")
        if self.z_padding:
            raise self._refused(
                "solver.padding.z", "z_padding", f"is {self.z_padding}, but tiles are calibrated unpadded; set it to 0"
            )
        score = calibration_score(self, where=where)
        if not prepared:
            source = _tiles.as_measured(self, source, plan=plan, where=where)
        with running(self, "calibrate_tiles"):
            # The model and the grid of a tile are judged before a tile is read or a transfer function computed.
            self.imaging_model.check(score, Sampling(shape=plan.read_shape, pixel_size=pixel_size))
            stack, sampling = stack_tiles(source, plan, pixel_size=pixel_size, device=self.device, dtype=self.dtype)
            bind(self, sampling)
            self._start_from(stack)
            if prior is not None and not isinstance(prior, ParameterPrior):
                prior = prior.to_prior(free_specs(self.imaging_model))
            return calibrate_tile_stack(
                stack,
                self.imaging_model,
                sampling,
                score=score,
                search=self.search,
                start=start,
                chunk_size=chunk_size,
                prior=prior,
            )

    def reconstruct_tiles(
        self,
        volumes: Sequence[tuple[ArrayLike, WritableArray]],
        *,
        plan: TilePlan,
        options: TileOptions | None = None,
        parameters: TileParameters | None = None,
        prepared: bool = False,
        pixel_size: PixelSize | None = None,
    ) -> TileStats:
        """Reconstruct volumes too large for one solve, tile by tile; return the time spent on each step.

        ``volumes`` are ``(source, sink)`` pairs, one per volume: the recorded ``(C, Z, Y, X)`` array and the array
        that receives the ``(F, Z, Y, X)`` estimate. ``options`` are the loader's settings, ``parameters`` per-tile
        values (by default the model's :attr:`tile_parameters` on a model with field tiles). A z-padded model reads
        each source with its mirrored slices and crops each estimate back, as a whole stack is padded.
        """
        where = f"{type(self).__name__}.reconstruct_tiles"
        pixel_size = _arguments.pixel_size(pixel_size, where=where, default=self.pixel_size)
        expect_instance(plan, TilePlan, where=where, name="plan")
        expect_instance(options, (TileOptions, type(None)), where=where, name="options")
        expect_instance(prepared, bool, where=where, name="prepared")
        self._expect_reconstructs(where)
        if self.apodization_rolloff:
            raise self._refused(
                "solver.apodization_rolloff",
                "apodization_rolloff",
                "applies to whole-volume solves; set it to 0 for tiles",
            )
        _tiles.expect_pairs(volumes, where=where)
        pairs = expect_volumes(volumes, where=where)
        if parameters is None:
            parameters = _tiles.field_tile_parameters(self, plan=plan, where=where)
        plan, pairs = _tiles.z_padded(self, plan, pairs, pixel_size=pixel_size)
        sampling = Sampling(shape=plan.read_shape, pixel_size=pixel_size)
        if not prepared:
            pairs = tuple((_tiles.as_measured(self, source, plan=plan, where=where), sink) for source, sink in pairs)
        with running(self, "reconstruct_tiles"):
            self.imaging_model.check(self.solver, sampling)  # before the bind computes a transfer function
            if self.solver.needs_operator:
                with torch.no_grad():  # cached without a graph: the tile driver binds the same sampling again
                    bind(self, sampling)
            return reconstruct_series(
                pairs,
                model=self.imaging_model,
                solver=self.solver,
                pixel_size=sampling.pixel_size,
                plan=plan,
                device=self.device,
                options=options,
                parameters=parameters,
            )

    # --- position methods (OME-Zarr); their helpers import iohub, so each method imports them when it runs ---------

    def reconstruct_positions(  # noqa: PLR0913 (one keyword per setting of a run; each is a flag of ``wo reconstruct``)
        self,
        positions: Inputs,
        *,
        output: str | Path,
        channels: Sequence[str],
        pixel_size: PixelSize | Literal["auto"] | None = None,
        times: Sequence[int] | int | None = None,
        devices: Literal["auto"] | Sequence[str | torch.device] = "auto",
        workers: Literal["auto"] | int = "auto",
        tile: ZYX[int] | None = None,
        tile_overlap: ZYX[int] | None = None,
        keep_going: bool = False,
        dry_run: bool = False,
        overwrite: bool = False,
        resume: bool = False,
        write_pixel_size: bool = False,
    ) -> ReconstructionReport:
        """Reconstruct the selected time points of OME-Zarr positions into an output plate, over the devices.

        The unit of work is one (position, time). Every check runs first and all problems are raised together
        (``Rejected``); then the plate is created and the units reconstructed with the model's current values (per
        field tile with :attr:`tile_parameters`). A unit that fails raises ``ReconstructionFailed`` with the report
        so far, unless ``keep_going``. iohub records each finished unit under a fingerprint of the reconstruction,
        which ``resume`` skips.

        Parameters
        ----------
        positions : str, Path, iohub Position or Plate, or a sequence of them
            One OME-Zarr position, several, quoted globs, a plate, or open iohub nodes (a ``Plate`` stands for its
            positions); each is opened read-only through iohub once, and the nodes given are left open.
        output : str or Path
            The output plate; never an input position or inside one.
        channels : Sequence[str]
            The input channel names, one per channel the system measures.
        pixel_size : float, (z, yx), (z, y, x) or "auto", optional
            For this call; by default the model's :attr:`pixel_size`, else each position's recorded scale. A size
            that differs from the recorded one by more than 5 % is used with a ``PositionWarning``.
        times : Sequence[int] or int, optional
            Absolute time indices; all by default.
        devices : "auto" or sequence of devices
            ``"auto"`` is every visible CUDA device for a model on CUDA, else the model's device.
        workers : "auto" or int
            Worker processes, round-robin over the devices; ``"auto"`` is one per device. One worker on the model's
            own device runs in this process.
        tile, tile_overlap : tuple[int, int, int], optional
            Reconstruct each volume tile by tile with these cores and overlaps (:meth:`tile_plan`).
        keep_going : bool
            Continue after a failed unit; the report's status is then ``"failed"``.
        dry_run : bool
            Check only; return the report with status ``"planned"``.
        overwrite : bool
            Delete an existing output plate first (never one that holds an input).
        resume : bool
            Skip the units an interrupted run of the same reconstruction finished (iohub's progress record).
        write_pixel_size : bool
            Record the stated ``pixel_size`` as the output's scale instead of each input's.

        Raises
        ------
        Rejected
            With every problem the checks found; nothing was computed or written.
        ReconstructionFailed
            If a unit fails (or a worker dies) and ``keep_going`` is false; it carries the report so far.
        """
        from waveorder.models import _pool, _positions
        from waveorder.models._checks import Arguments, check_positions

        expect_instance(output, (str, os.PathLike), where=f"{type(self).__name__}.reconstruct_positions", name="output")
        problems = Problems()
        arguments = Arguments(
            positions=positions,
            output=output,
            channels=channels,
            pixel_size=pixel_size,
            times=times,
            overwrite=overwrite,
            resume=resume,
            write_pixel_size=write_pixel_size,
            tile=tile,
            tile_overlap=tile_overlap,
        )
        with _positions.transfer_kept(self):
            plan = check_positions(self, arguments, problems=problems)
            units = [_positions.Unit(meta.name, t) for meta in plan.positions for t in plan.times.get(meta.name, ())]
            assigned = None
            payload = None
            with problems.collect():
                assigned = _pool.plan_workers(self, devices, workers, units=len(units))
                if not assigned.in_process:
                    payload = _pool.pickled_model(self)
            problems.raise_if_any()
            assert assigned is not None
            names = tuple(unit.name for unit in units)
            report = ReconstructionReport(output=str(plan.output), status="planned", total=len(units), pending=names)
            if dry_run:
                return report
            if resume and plan.version == "0.4":
                warnings.warn(PositionWarning(_positions.RESUME_UNTRACKED, location=str(plan.output)), stacklevel=2)
            records = _positions.output_records(self, plan, times=times)
            indices = _positions.write_plate(plan, records)
            work = _positions.unit_work(self, plan, indices, records=records, resume=resume)
            report.status = "finished"
            try:
                _pool.run_units(
                    self, units, work=work, workers=assigned, payload=payload, keep_going=keep_going, report=report
                )
            finally:
                ended = {result.unit for result in report.results}
                report.pending = tuple(name for name in names if name not in ended)
        if report.count("failed"):
            report.status = "failed"
        return report

    def calibrate_position(  # noqa: PLR0913 (the position, and one keyword per flag of ``wo calibrate-tf``)
        self,
        position: Inputs,
        *,
        channels: Sequence[str],
        pixel_size: PixelSize | Literal["auto"] | None = None,
        time: int | None = None,
        prior: TfStore | str | Path | None = None,
        start: TfStore | str | Path | None = None,
        dry_run: bool = False,
    ) -> TfStore | None:
        """Calibrate on one time point (``time``, the first by default) of a position; return the store, unsaved.

        ``position`` takes what :meth:`reconstruct_positions` takes; the first position it names is calibrated, and
        nodes the call opens are closed when it returns. ``channels`` and ``pixel_size`` are as
        :meth:`reconstruct_positions` takes them. The model takes the estimates, or with field tiles the per-tile
        table as :attr:`tile_parameters`. ``prior`` is a previous calibration (a store or its path) whose values
        become the prior centres, ``start`` one whose values start the search; ``dry_run`` only checks and returns
        ``None``. The store holds parameters only, no transfer function. Warns
        (:class:`~waveorder.errors.StoreWarning`) of each quantity that ended on a bound.

        Raises
        ------
        Rejected
            With every problem found, before anything is computed.
        ReadError
            If the volume cannot be read; the message names the position and the time.
        """
        from waveorder.io.positions import open_positions, read_volume
        from waveorder.models import _calibration

        expect_instance(dry_run, bool, where=f"{type(self).__name__}.calibrate_position", name="dry_run")
        problems = Problems()
        with open_positions(position, problems=problems) as opened:
            chosen = opened[0] if opened else None
            plan = _calibration.checked_plan(
                self, chosen, problems=problems, channels=channels, pixel_size=pixel_size, time=time, prior=prior,
                start=start,
            )  # fmt: skip
            if dry_run or chosen is None:
                return None
            volume = read_volume(chosen, time=plan.time, channels=plan.channels).double()
        return _calibration.calibrated_store(self, volume, plan)

    def load_calibration(self, store: TfStore | str | Path) -> None:
        """Load a calibration store made for this model's imaging system.

        Its row sets the free quantities; its rows per field tile, :attr:`tile_parameters`. A stored quantity the
        model fixes outside the imaging system keeps the model's value, with a warning. Its transfer function on a
        grid is then used instead of computed where it applies (an INFO line says where, and why not elsewhere).

        Raises
        ------
        Rejected
            At ``store``, with every problem found and the model unchanged.
        """
        read = _stores.read_store(store, "store")
        location = _stores.where("store", read.path)
        free = tuple(free_specs(self.imaging_model))
        problems = _stores.store_problems(read, self, free=free)
        if problems:
            raise Rejected([Problem(message, location) for message in problems])
        fixed = [name for name in read.parameter_names if name not in free]
        if fixed:
            message = f"holds {', '.join(fixed)}, which the model fixes; the model's values are used"
            _stores.warn(message, location=location)
        columns = {name: read.parameters[:, read.parameter_names.index(name)] for name in free}
        self._stored_transfers = _stores.stored_transfers(read, self)
        if self.field_tiles is None:
            load_parameters(self.imaging_model, {name: float(values[0]) for name, values in columns.items()})
            self.tile_parameters = None
            return
        parameters = dict(self.imaging_model.named_parameters())
        self.tile_parameters = {
            name: values.to(dtype=parameters[name].dtype, device=parameters[name].device)
            for name, values in columns.items()
        }

    def to_tf_store(self, *, shapes: Sequence[ZYX[int]] = (), pixel_size: PixelSize | None = None) -> TfStore:
        """Return the current values of the free quantities as a store, with the transfer function on each shape.

        Kind ``"nominal"`` when every free quantity is at its nominal value, else ``"parameters"``; one row, or one
        per field tile. The transfer functions are on each shape after z padding (none for field tiles), for
        :meth:`load_calibration`. A model with field tiles takes one shape, which places its tiles.

        Raises
        ------
        ParameterError
            At ``shapes``, if it is not a sequence of ``(z, y, x)`` shapes.
        Rejected
            If a model with field tiles is given other than one shape, the model cannot run on a shape, or
            :attr:`tile_parameters` holds another number of tiles than the shape has.
        """
        size = None if not shapes else self._size(pixel_size, f"{type(self).__name__}.to_tf_store")
        return _stores.current_store(self, shapes=shapes, pixel_size=size)

    # --- configuration and files -----------------------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: V4Config | Mapping[str, object] | str | Path) -> Self:
        """Build the model a configuration (or its mapping, or its YAML path) describes, as the class of its system.

        ``data.pixel_size`` becomes the model's :attr:`pixel_size` and ``solver.apodization_rolloff`` its roll-off;
        the ``calibrated`` block is loaded into the free quantities. The model keeps the configuration
        (:attr:`config`). Problems are raised together, at their config paths (``Rejected``).
        """
        return _config.from_config(cls, config)

    def to_config(
        self,
        *,
        channels: Sequence[str] | None = None,
        pixel_size: PixelSize | None = None,
        times: Sequence[int] | int | None = None,
    ) -> V4Config | dict[str, object]:
        """Return the configuration the model was built from with its ``calibrated`` block.

        A model built in Python returns its description (each part as ``{name: arguments}``). The arguments replace
        the ``data`` block.
        """
        return _config.to_config(self, channels=channels, pixel_size=pixel_size, times=times)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write the model as a folder that :meth:`load` reads (``model.json``); return the folder."""
        return _config.save(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, source: str | Path | Mapping[str, object]) -> Self:
        """Return the model :meth:`save` wrote to a folder, or the one a description holds, on the CPU in float32."""
        return _config.load(cls, source)

    # --- shared machinery ------------------------------------------------------------------------------------------

    def _size(self, pixel_size: object, where: str) -> ZYX[float]:
        """Return the voxel size of a call: ``pixel_size``, else the model's; refuse when neither is known."""
        size = self._pixel_size if pixel_size is None else pixel_size
        if size is None:
            raise ParameterError(
                f"{where} has no voxel size: set data.pixel_size in the configuration, pass pixel_size= to the "
                "model's constructor, or pass pixel_size= to this call",
                path="pixel_size",
            )
        return _arguments.pixel_size(size, where=where)

    def _recorded(self, where: str, data: object, pixel_size: object) -> tuple[Tensor, Sampling]:
        """Return ``data`` as a real ``(C, Z, Y, X)`` tensor and the sampling of its grid."""
        size = self._size(pixel_size, where)
        try:
            tensor = torch.as_tensor(data)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ParameterError(
                f"{where} data must be a tensor or an array torch.as_tensor accepts, got {type(data).__name__}: "
                f"{error}",
                path="data",
            ) from error
        if tensor.is_complex():
            raise ShapeError(f"{where} data must be real, got {tensor.dtype}")
        channels = self.system.channel_count
        if tensor.ndim == 3 and channels == 1:
            tensor = tensor[None]
        if tensor.ndim != 4 or tensor.shape[0] != channels:
            also = " or (Z, Y, X)" if channels == 1 else ""
            raise ShapeError(
                f"{where} data has shape {tuple(tensor.shape)}, but {type(self.system).__name__} measures "
                f"(C, Z, Y, X) with C = {channels}{also}"
            )
        _, z, y, x = tensor.shape
        return tensor, Sampling(shape=(z, y, x), pixel_size=size)

    def _measured(self, recorded: Tensor, padding: ZPadding, *, prepared: bool) -> Tensor:
        """Return recorded data on the device, z-padded, prepared unless ``prepared``, in the model's dtype."""
        data = padding.pad(recorded.to(self.device))
        return (data if prepared else self.prepare(data)).to(self.dtype)

    def _refused(self, config_path: str, argument: str, why: str) -> ConfigurationError:
        """Return the refusal at the config path of a model built from a configuration, else at the argument."""
        return ConfigurationError(config_path if self.config is not None else argument, why)

    def _expect_whole_field(self, where: str) -> None:
        """Refuse a model with field tiles, which is calibrated and reconstructed per tile."""
        if self.field_tiles is not None:
            raise self._refused(
                "imaging_system.field_dependence",
                "field_tiles",
                f"is set, so {where} does not take the whole field as one system; on arrays, plan the tiles with "
                "field_tile_plan and call calibrate_tiles and reconstruct_tiles, on positions calibrate_position and "
                "reconstruct_positions",
            )


_MODEL_CLASSES: list[type[ReconstructionModel]] = []
"""The model classes that declare ``system_types``, in the order they were defined; no two list one system type."""


def _model_class(system: ImagingSystem) -> type[ReconstructionModel]:
    """Return the class whose ``system_types`` list ``system``'s type, else one it is an instance of, else the base."""
    for model_class in _MODEL_CLASSES:
        if type(system) in model_class.system_types:
            return model_class
    for model_class in _MODEL_CLASSES:
        if isinstance(system, model_class.system_types):
            return model_class
    return ReconstructionModel


def _tile_table(table: object, model: ImagingModel, *, where: str) -> dict[str, Tensor]:
    """Return ``table`` as per-tile values ``(tiles, *shape)`` of ``model``'s free quantities, on their placement."""
    expect_instance(table, Mapping, where=where, name="tile_parameters")
    free = dict(model.named_parameters())
    checked: dict[str, Tensor] = {}
    for name, values in cast(Mapping[object, object], table).items():
        path = f"tile_parameters.{name}"
        if not isinstance(name, str) or name not in free:
            raise ParameterError(
                f"{where} {path}: {name!r} is not a free quantity of the model; it has {sorted(free)}", path=path
            )
        expect_instance(values, Tensor, where=where, name=path)
        parameter, rows = free[name], cast(Tensor, values)
        if rows.ndim < 1 or tuple(rows.shape[1:]) != tuple(parameter.shape):
            raise ParameterError(
                f"{where} {path} has shape {tuple(rows.shape)}, expected (tiles, *{tuple(parameter.shape)})", path=path
            )
        checked[name] = rows.detach().to(dtype=parameter.dtype, device=parameter.device)
    if len({values.shape[0] for values in checked.values()}) > 1:
        raise ParameterError(f"{where} tile_parameters hold different numbers of tiles", path="tile_parameters")
    return checked


def _power_law(shape: ZYX[int], exponent: float, generator: torch.Generator) -> Tensor:
    """Return a zero-mean, unit-std Gaussian field of power spectrum ``(|k|^2 + k0^2)^(-exponent/2)``, float32."""
    kz, ky, kx = torch.meshgrid(*(torch.fft.fftfreq(n) for n in shape), indexing="ij")
    amplitude = (kz**2 + ky**2 + kx**2 + (1.0 / max(shape)) ** 2) ** (-exponent / 4.0)
    white = torch.randn(shape, generator=generator)
    field = torch.real(torch.fft.ifftn(torch.fft.fftn(white) * amplitude))
    return (field - field.mean()) / field.std().clamp(min=1e-12)


def _noisy(clean: Tensor, noise: Noise, *, config: V4Config | None, generator: torch.Generator) -> Tensor:
    """Draw data around ``clean`` (on the CPU): a simulation's ``snr_db`` or ``photons``, else the model's noise."""
    settings = None if config is None else config.noise
    gaussian = None if settings is None else settings.gaussian
    poisson: PoissonNoise | None = None if settings is None else settings.poisson
    if gaussian is not None and gaussian.snr_db is not None:
        sigma = float(clean.std()) / 10.0 ** (gaussian.snr_db / 20.0)
        return clean + sigma * torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    if poisson is not None and poisson.photons is not None:
        shifted = clean - clean.min()
        scale = poisson.photons / float(shifted.mean().clamp(min=1e-12))
        return torch.poisson(shifted * scale, generator=generator)
    return copy.deepcopy(noise).to("cpu").sample(clean, generator=generator)


# The helpers take a ReconstructionModel first, so they are imported once the class exists.
from waveorder.models import _config, _stores, _tiles  # noqa: E402
from waveorder.models._run import bind, calibration_score, placed, running  # noqa: E402
