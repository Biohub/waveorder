"""The version-4 configuration schema.

Five blocks (``data``, ``imaging_system``, ``object``, ``noise``, ``solver``) plus ``outputs``, ``simulate``,
``calibrated`` and a required ``version``.

Conventions
-----------
* A calibratable number under ``imaging_system`` may be written ``{nominal, tolerance, bounds}`` (a
  :class:`FreeParameter`): it is then a free quantity, estimated from data. ``tolerance`` is the standard deviation of
  a Gaussian prior in the quantity's units, or ``"free"`` for a flat prior; ``bounds`` truncate it. A plain number is
  fixed.
* A field with several options takes a bare name, or a mapping of one name to its parameters:
  ``reconstruction: closed_form`` or ``reconstruction: {fista: {iterations: 50}}``.
* Zernike terms under ``pupil_aberrations`` are named (``defocus``, ``spherical``, ``z<n>`` by Noll index, see
  :mod:`waveorder.zernike_names`), in waves of RMS wavefront error.

The schema checks the document. Whether its parts can run together is checked by the parts themselves when
:meth:`waveorder.models.ReconstructionModel.from_config` builds them.
"""

import re
import warnings
from collections.abc import Callable, Mapping
from typing import Annotated, Any, Literal, LiteralString, Self, cast

from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    ValidationError,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

from waveorder.errors import ConfigurationWarning
from waveorder.zernike_names import canonical_name, noll_indices_up_to_order, noll_to_name


def _not_a_bool(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("must be a number, not a bool (YAML true/false)")
    return value


def _positive(value: float) -> float:
    if not value > 0:
        raise ValueError(f"must be greater than 0, got {value}")
    return value


def _non_negative(value: float) -> float:
    if not value >= 0:
        raise ValueError(f"must be at least 0, got {value}")
    return value


def _zero_is_none(value: object) -> object:
    """Read a stopping tolerance of 0 as none: wo-4.0 accepted 0, which never stops early."""
    return None if value == 0 and not isinstance(value, bool) else value


type FiniteFloat = Annotated[float, BeforeValidator(_not_a_bool), Field(allow_inf_nan=False)]
type PositiveFiniteFloat = Annotated[
    float, BeforeValidator(_not_a_bool), Field(allow_inf_nan=False), AfterValidator(_positive)
]
type NonNegativeFiniteFloat = Annotated[
    float, BeforeValidator(_not_a_bool), Field(allow_inf_nan=False), AfterValidator(_non_negative)
]
type Integer = Annotated[int, BeforeValidator(_not_a_bool)]
type Count = Annotated[int, BeforeValidator(_not_a_bool), Field(gt=0)]
type NonNegativeCount = Annotated[int, BeforeValidator(_not_a_bool), Field(ge=0)]
type GridPoints = Annotated[int, BeforeValidator(_not_a_bool), Field(ge=2)]


def _fixed(value: object) -> object:
    if isinstance(value, Mapping) and "nominal" in value:
        raise ValueError("a free value here is not available in this version of WaveOrder; give a number")
    return value


type FixedFloat = Annotated[PositiveFiniteFloat, BeforeValidator(_fixed)]
"""A positive optical number that cannot be free: a wavelength or the refractive index."""


class V4BaseModel(BaseModel):
    """Base of every block: an unknown key is an error."""

    model_config = ConfigDict(extra="forbid")


def field_error(field: str, message: str, value: object = None) -> ValidationError:
    """Return a validation error at the dotted path ``field`` (relative to the model that raises it)."""
    detail = InitErrorDetails(
        type=PydanticCustomError("config_rule", "{message}", {"message": message}),
        loc=tuple(field.split(".")),
        input=value,
    )
    return ValidationError.from_exception_data("V4Config", [detail])


def _relocated(
    error: ValidationError, move: Callable[[tuple[int | str, ...]], tuple[int | str, ...]]
) -> ValidationError:
    details = [
        InitErrorDetails(
            type=PydanticCustomError(cast(LiteralString, line["type"]), "{message}", {"message": str(line["msg"])}),
            loc=move(tuple(line["loc"])),
            input=line["input"],
        )
        for line in error.errors()
    ]
    return ValidationError.from_exception_data(error.title, details)


class _Option:
    """The validators of an option field: a bare name (tag ``name``) or a one-key mapping of one of ``names``.

    ``bare_names`` are the options written only as a name. A bare name of an option with parameters
    (``parameterized``) takes their defaults, and an error is reported at the path the document writes
    (``solver.reconstruction.fista.iterations``), without pydantic's union tags; a name that is no option lists them
    all.
    """

    def __init__(
        self, *names: str, bare_names: tuple[str, ...] = (), parameterized: tuple[str, ...] = (), bare: bool = True
    ) -> None:
        self.names = frozenset(names)
        self.choices = (*bare_names, *names)
        self.parameterized = frozenset(parameterized)
        self.bare = bare

    def tag(self, value: object) -> str:
        if isinstance(value, BaseModel):
            return next(iter(type(value).model_fields))
        if isinstance(value, dict) and len(value) == 1 and next(iter(value)) in self.names:
            return next(iter(value))
        return "name" if self.bare or not isinstance(value, str) else value

    def normalize(self, value: object) -> object:
        return {value: {}} if isinstance(value, str) and value in self.parameterized else value

    def strip(self, value: object, handler: ValidatorFunctionWrapHandler) -> object:
        chosen = next(iter(value)) if isinstance(value, dict) and len(value) == 1 else value
        if isinstance(chosen, str) and chosen not in self.choices:
            message = f"{chosen!r} is not an option; choose one of {', '.join(self.choices)}"
            raise PydanticCustomError("option", "{message}", {"message": message})
        try:
            return handler(value)
        except ValidationError as error:
            tags = {"name", *self.names}
            raise _relocated(error, lambda loc: loc[1:] if loc and loc[0] in tags else loc) from None


def _available(*values: object) -> AfterValidator:
    """Refuse every value of a setting but ``values``: the others are not available in this version."""
    shown = "remove it" if values == (None,) else f"use {' or '.join(repr(value) for value in values)}"

    def check(value: object) -> object:
        if value not in values:
            raise ValueError(f"{value!r} is not available in this version of WaveOrder; {shown}")
        return value

    return AfterValidator(check)


def no_effect(field: str, value: object, reason: str) -> None:
    """Warn that the setting at ``field``, which wo-4.0 accepted, is accepted here and changes nothing, and why."""
    warnings.warn(ConfigurationWarning(f"{value!r} has no effect: {reason}", field=field), stacklevel=3)


# --- free parameters -------------------------------------------------------------------------------------------------


class FreeParameter(V4BaseModel):
    """A calibratable imaging-system number: ``{nominal, tolerance, bounds}``."""

    nominal: FiniteFloat
    tolerance: PositiveFiniteFloat | Literal["free"] = "free"
    bounds: tuple[FiniteFloat, FiniteFloat] | None = None

    @model_validator(mode="after")
    def _nominal_inside_bounds(self) -> Self:
        if self.bounds is not None:
            low, high = self.bounds
            if low >= high:
                raise ValueError(f"bounds must satisfy min < max, got {self.bounds}")
            if not low <= self.nominal <= high:
                raise ValueError(f"nominal = {self.nominal} lies outside bounds = {self.bounds}")
        return self


type CalibratableFloat = FiniteFloat | FreeParameter


def nominal_of(value: CalibratableFloat | None) -> float | None:
    """Return the nominal value of a number or a :class:`FreeParameter`."""
    if value is None:
        return None
    return value.nominal if isinstance(value, FreeParameter) else float(value)


# --- data ------------------------------------------------------------------------------------------------------------


class PixelSize(V4BaseModel):
    """Voxel size in um along z, y and x."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    z: PositiveFiniteFloat
    y: PositiveFiniteFloat
    x: PositiveFiniteFloat


def _pixel_size(value: object) -> object:
    """Read the forms of ``data.pixel_size``: one number (isotropic), ``{z, yx}`` or ``{z, y, x}``.

    wo-4.0's ``yx`` is one number or a ``{y, x}`` mapping.
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        return {"z": value, "y": value, "x": value}
    if isinstance(value, dict) and "yx" in value:
        rest = {key: item for key, item in value.items() if key != "yx"}
        yx = value["yx"]
        if isinstance(yx, dict):
            y, x = (yx.get("y"), yx.get("x")) if yx else (0.1, 0.1)  # wo-4.0 read {} as its default, 0.1 um
        else:
            y, x = yx, yx
        return {**rest, "y": y, "x": x}
    return value


class DataSettings(V4BaseModel):
    """Which data is reconstructed, and its sampling."""

    input_channel_names: list[str] = Field(default_factory=lambda: ["simulated"])
    time_indices: NonNegativeCount | list[NonNegativeCount] | Literal["all"] = "all"
    pixel_size: Annotated[PixelSize | Literal["auto"], BeforeValidator(_pixel_size)] = Field(
        default="auto", description="voxel size in um: auto (from the OME-Zarr store), a number, {z, yx} or {z, y, x}"
    )

    def arguments(self) -> dict[str, Any]:
        """Return the block as the ``channels``, ``pixel_size`` and ``times`` arguments of the position methods.

        ``pixel_size`` is ``(z, y, x)``, or ``None`` for ``auto``; ``times`` is a list, or ``None`` for all.
        """
        times = self.time_indices
        size = self.pixel_size
        return {
            "channels": list(self.input_channel_names),
            "pixel_size": None if size == "auto" else (size.z, size.y, size.x),
            "times": None if times == "all" else [times] if isinstance(times, int) else list(times),
        }


# --- imaging_system --------------------------------------------------------------------------------------------------


class ZernikeUpToOrder(V4BaseModel):
    """Free every Zernike term through radial ``order`` (piston, tip and tilt excluded), nominal 0."""

    order: Count = Field(ge=2)
    tolerance: PositiveFiniteFloat | Literal["free"] = "free"
    bounds: tuple[FiniteFloat, FiniteFloat] | None = None


_CALIBRATABLE = TypeAdapter(CalibratableFloat)


class PupilAberrations(BaseModel):
    """Zernike pupil aberrations by name, plus the ``zernike_up_to_order`` shorthand; unlisted terms are 0."""

    model_config = ConfigDict(extra="allow")

    zernike_up_to_order: ZernikeUpToOrder | None = None

    @model_validator(mode="after")
    def _canonical_terms(self) -> Self:
        extra = self.__pydantic_extra__ or {}
        canonical: dict[str, CalibratableFloat] = {}
        for name, value in extra.items():
            key = canonical_name(name)
            if key in canonical:
                raise ValueError(f"pupil_aberrations lists Zernike term {key!r} twice (as {name!r} and another name)")
            canonical[key] = _CALIBRATABLE.validate_python(value)
        extra.clear()
        extra.update(canonical)
        return self

    @property
    def terms(self) -> dict[str, CalibratableFloat]:
        """The named terms, without the shorthand."""
        return dict(self.__pydantic_extra__ or {})


class Pinhole(V4BaseModel):
    """Confocal pinhole: exactly one of ``diameter_au`` (Airy units) or ``diameter_um`` (object plane)."""

    diameter_au: CalibratableFloat | None = None
    diameter_um: CalibratableFloat | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.diameter_au is None) == (self.diameter_um is None):
            raise ValueError("pinhole requires exactly one of 'diameter_au' or 'diameter_um'")
        return self


class DetectionSettings(V4BaseModel):
    """The detection arm."""

    wavelength: FixedFloat | None = Field(default=None, description="emission wavelength in um")
    numerical_aperture: CalibratableFloat = 1.2
    pinhole: Literal["none", "point"] | Pinhole = "none"
    pupil_aberrations: PupilAberrations | None = None
    z_focus_offset: CalibratableFloat = Field(default=0.0, description="focal plane offset in slices")
    edge: Literal["ramp", "sigmoid"] = Field(default="ramp", description="pupil edge; sigmoid is the 3.x pupil")


class Tilt(V4BaseModel):
    """Illumination tilt in radians."""

    zenith: CalibratableFloat = 0.0
    azimuth: CalibratableFloat = 0.0


class IlluminationSettings(V4BaseModel):
    """The illumination arm (brightfield phase)."""

    wavelength: FixedFloat | None = Field(default=None, description="illumination wavelength in um")
    numerical_aperture: CalibratableFloat | None = None
    tilt: Tilt = Field(default_factory=Tilt)
    invert_phase_contrast: bool = False


class TiledFieldDependence(V4BaseModel):
    """One set of values per field tile, cross-faded in the overlaps."""

    size_yx: tuple[Count, Count]
    overlap_yx: tuple[NonNegativeCount, NonNegativeCount] | None = Field(
        default=None, description="cross-fade in pixels per side; default min(32, size // 2)"
    )
    tile_blend: Literal["linear", "cosine"] = "linear"


class FieldDependenceTiled(V4BaseModel):
    """``field_dependence: {tiled: ...}``."""

    tiled: TiledFieldDependence


class ImagingSystemSettings(V4BaseModel):
    """The imaging system ``H`` and the priors of its free quantities."""

    contrast: Literal["phase", "fluorescence"]
    field_dependence: Literal["shift_invariant"] | FieldDependenceTiled = "shift_invariant"
    index_of_refraction_media: FixedFloat = 1.3
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    illumination: IlluminationSettings = Field(default_factory=IlluminationSettings)


# --- object ----------------------------------------------------------------------------------------------------------


class BeadsPrior(V4BaseModel):
    """Sparse beads of known diameter at unknown positions, to calibrate ``H``."""

    diameter: PositiveFiniteFloat = Field(description="bead diameter in um")
    snr_threshold: PositiveFiniteFloat = Field(default=6.0, description="detection threshold in robust noise units")
    max_beads: Count = 50
    min_separation: Count = Field(default=5, description="non-maximum suppression window in voxels")


class PriorBeads(V4BaseModel):
    """``prior: {beads: ...}``."""

    beads: BeadsPrior


class DenoiserPrior(V4BaseModel):
    """wo-4.0's learned denoiser prior, which neither wo-4.0 nor this version runs."""

    checkpoint: str


class PriorDenoiser(V4BaseModel):
    """``prior: {denoiser: ...}``: accepted as wo-4.0 accepted it, and reconstructed with no prior."""

    denoiser: DenoiserPrior


_PRIOR = _Option("beads", "denoiser", bare_names=("tikhonov", "tv", "none"))
type Prior = Annotated[
    Annotated[Literal["tikhonov", "tv", "none"], Tag("name")]
    | Annotated[PriorBeads, Tag("beads")]
    | Annotated[PriorDenoiser, Tag("denoiser")],
    Discriminator(_PRIOR.tag),
    BeforeValidator(_PRIOR.normalize),
    WrapValidator(_PRIOR.strip),
]


class LegacyAutoWeight(V4BaseModel):
    """The 3.x rule-based sweep of the weight (``otsu_cnr`` or ``l_curve``)."""

    rule: Literal["otsu_cnr", "l_curve"] = "otsu_cnr"
    search_min: FiniteFloat = -6.0
    search_max: FiniteFloat = 2.0
    num_samples: Count = 25
    search_scale: Literal["relative", "absolute"] = "relative"
    report_path: str | None = Field(default=None, description="JSON of the sweep: weights, scores, the pick")
    plot_path: str | None = Field(default=None, description="PNG montage of the sweep, the pick framed")


class WeightAuto(V4BaseModel):
    """``weight: {auto: ...}``."""

    auto: LegacyAutoWeight


class GaussianSpectrumSource(V4BaseModel):
    """A Gaussian random field with a power-law spectrum (simulation)."""

    exponent: FiniteFloat = 2.0


class SourceGaussianSpectrum(V4BaseModel):
    """``source: {gaussian_spectrum: ...}``."""

    gaussian_spectrum: GaussianSpectrumSource


class PhantomSource(V4BaseModel):
    """wo-4.0's named phantom, which neither wo-4.0 nor this version simulates."""

    kind: str


class SourcePhantom(V4BaseModel):
    """``source: {phantom: ...}``: accepted as wo-4.0 accepted it; a simulation draws the default object."""

    phantom: PhantomSource


class FromReconSource(V4BaseModel):
    """wo-4.0's object read from a reconstruction, which neither wo-4.0 nor this version simulates."""

    path: str


class SourceFromRecon(V4BaseModel):
    """``source: {from_recon: ...}``: accepted as wo-4.0 accepted it; a simulation draws the default object."""

    from_recon: FromReconSource


class ObjectSettings(V4BaseModel):
    """The object prior (or the true object of a simulation)."""

    dimension: Literal[2, 3] = Field(default=3, description="3, or 2 for a thin object")
    prior: Prior = "tikhonov"
    weight: NonNegativeFiniteFloat | Literal["auto"] | WeightAuto = Field(
        default=1e-3, description="regularization weight, or auto"
    )
    constraints: list[Literal["nonneg"]] = Field(default_factory=list)
    source: SourceGaussianSpectrum | SourcePhantom | SourceFromRecon | None = None


# --- noise -----------------------------------------------------------------------------------------------------------


class DarkFrames(V4BaseModel):
    """wo-4.0's noise level from dark frames, which neither wo-4.0 nor this version reads."""

    dark_frames: str


type Sigma = PositiveFiniteFloat | Literal["estimate"] | DarkFrames


class GaussianNoise(V4BaseModel):
    """Additive Gaussian noise of standard deviation ``sigma``, in the units of the data."""

    sigma: Sigma = 1.0
    snr_db: FiniteFloat | None = Field(default=None, description="simulation only: sigma from the signal power")


class PoissonNoise(V4BaseModel):
    """Shot noise: counts are Poisson in ``(signal - offset) / gain``."""

    gain: PositiveFiniteFloat = 1.0
    offset: FiniteFloat = 0.0
    photons: PositiveFiniteFloat | None = Field(default=None, description="simulation only: mean photons per voxel")


class PoissonGaussianNoise(V4BaseModel):
    """Shot noise plus Gaussian read noise of standard deviation ``sigma``."""

    gain: PositiveFiniteFloat = 1.0
    offset: FiniteFloat = 0.0
    sigma: Sigma = 1.0


class NoiseModel(V4BaseModel):
    """Exactly one of ``gaussian``, ``poisson``, ``poisson_gaussian``; ``gaussian`` when none is written."""

    gaussian: GaussianNoise | None = None
    poisson: PoissonNoise | None = None
    poisson_gaussian: PoissonGaussianNoise | None = None

    @model_validator(mode="before")
    @classmethod
    def _a_mapping(cls, data: object) -> object:
        if isinstance(data, str):
            raise ValueError(f"is a mapping, not a name; write noise: {{{data}: {{}}}}")
        return data

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        chosen = [name for name in type(self).model_fields if getattr(self, name) is not None]
        if not chosen:
            self.gaussian = GaussianNoise()
        elif len(chosen) > 1:
            raise ValueError(f"noise accepts exactly one option, got {chosen}")
        return self

    @property
    def kind(self) -> str:
        """The chosen option."""
        return next(name for name in type(self).model_fields if getattr(self, name) is not None)


# --- solver ----------------------------------------------------------------------------------------------------------


class Padding(V4BaseModel):
    """Mirrored padding around each solve."""

    z: NonNegativeCount | Literal["auto"] = Field(
        default=0,
        description="z slices on each side; auto: at least n * wavelength / NA_detection^2 in slices, then more for "
        "a fast FFT length of the padded stack",
    )
    yx: NonNegativeCount | tuple[NonNegativeCount, NonNegativeCount] = Field(
        default=0, description="wo-4.0's; yx padding is not in this version, so any value pads nothing"
    )


class LoopOptions(V4BaseModel):
    """The iteration settings every iterative solver takes."""

    iterations: Count = 50
    tolerance: PositiveFiniteFloat | None = Field(default=None, description="stop at this relative change")
    check_every: Count = 10


class FistaReconstruction(LoopOptions):
    """FISTA."""


class AdmmReconstruction(V4BaseModel):
    """ADMM with total variation."""

    iterations: Count = 1
    rho: PositiveFiniteFloat = 1e-3


class RichardsonLucyReconstruction(LoopOptions):
    """Richardson-Lucy with the 3.x back projectors."""

    iterations: Count = 25
    background: NonNegativeFiniteFloat = 0.0
    back_projector: Literal["matched", "gaussian", "butterworth", "wiener", "wiener_butterworth"] = "matched"
    bp_alpha: PositiveFiniteFloat | None = None
    bp_beta: PositiveFiniteFloat | None = None
    gradient_consensus: bool = Field(default=False, description="the 3.x RLGC variant")
    start: Literal["mean", "ones"] = Field(default="mean", description="first estimate; ones is the 3.x start")
    tolerance: Annotated[PositiveFiniteFloat | None, BeforeValidator(_zero_is_none)] = Field(
        default=None,
        validation_alias=AliasChoices("tolerance", "stopping_tolerance"),
        description="stop at this relative change; stopping_tolerance is wo-4.0's name, and 0 never stops early",
    )


class DistilledReconstruction(V4BaseModel):
    """A distilled network solver."""

    bundle: str = Field(description="the bundle directory wo distill fit wrote")
    precision: Literal["float32", "bf16", "fp16"] = "float32"


class ReconstructionFista(V4BaseModel):
    """``reconstruction: {fista: ...}``."""

    fista: FistaReconstruction


class ReconstructionAdmm(V4BaseModel):
    """``reconstruction: {admm: ...}``."""

    admm: AdmmReconstruction


class ReconstructionRichardsonLucy(V4BaseModel):
    """``reconstruction: {richardson_lucy: ...}``."""

    richardson_lucy: RichardsonLucyReconstruction


class ReconstructionDistilled(V4BaseModel):
    """``reconstruction: {distilled: ...}``."""

    distilled: DistilledReconstruction


_RECONSTRUCTION = _Option(
    "fista",
    "admm",
    "richardson_lucy",
    "distilled",
    bare_names=("closed_form",),
    parameterized=("fista", "admm", "richardson_lucy", "distilled"),
)
type Reconstruction = Annotated[
    Annotated[Literal["closed_form"], Tag("name")]
    | Annotated[ReconstructionFista, Tag("fista")]
    | Annotated[ReconstructionAdmm, Tag("admm")]
    | Annotated[ReconstructionRichardsonLucy, Tag("richardson_lucy")]
    | Annotated[ReconstructionDistilled, Tag("distilled")],
    Discriminator(_RECONSTRUCTION.tag),
    BeforeValidator(_RECONSTRUCTION.normalize),
    WrapValidator(_RECONSTRUCTION.strip),
]


class GridStage(V4BaseModel):
    """A grid over the free quantities: points per quantity, or per quantity path."""

    points: GridPoints | dict[str, GridPoints] = 7


class RandomStage(V4BaseModel):
    """Random draws from the prior."""

    samples: Count = 16
    seed: Integer = 0


class GradientStage(V4BaseModel):
    """Gradient descent from the best point so far."""

    max_iter: Count = 200
    method: Literal["split", "alternating", "joint"] = Field(
        default="split", description="wo-4.0's; only split runs, and the others have no effect"
    )
    optimizer: Literal["adam", "nadam", "adamw", "sgd"] = "adam"


class LBFGSStage(V4BaseModel):
    """L-BFGS with a strong-Wolfe line search."""

    iterations: Count = 20
    history: Count = 10


class NelderMeadStage(V4BaseModel):
    """The Nelder-Mead simplex (3.x ``nelder_mead``)."""

    iterations: Count = 200
    tolerance: PositiveFiniteFloat = 1e-4


class SearchGrid(V4BaseModel):
    """``{grid: ...}``."""

    grid: GridStage


class SearchRandom(V4BaseModel):
    """``{random: ...}``."""

    random: RandomStage


class SearchGradient(V4BaseModel):
    """``{gradient: ...}``."""

    gradient: GradientStage


class SearchLBFGS(V4BaseModel):
    """``{lbfgs: ...}``."""

    lbfgs: LBFGSStage


class SearchNelderMead(V4BaseModel):
    """``{nelder_mead: ...}``."""

    nelder_mead: NelderMeadStage


_STAGES = ("grid", "random", "gradient", "lbfgs", "nelder_mead")
_STAGE = _Option(*_STAGES, parameterized=_STAGES, bare=False)
type SearchStageSettings = Annotated[
    Annotated[SearchGrid, Tag("grid")]
    | Annotated[SearchRandom, Tag("random")]
    | Annotated[SearchGradient, Tag("gradient")]
    | Annotated[SearchLBFGS, Tag("lbfgs")]
    | Annotated[SearchNelderMead, Tag("nelder_mead")],
    Discriminator(_STAGE.tag),
    BeforeValidator(_STAGE.normalize),
    WrapValidator(_STAGE.strip),
]


class CalibrationAdvanced(V4BaseModel):
    """Step sizes and the 3.x optimizer settings."""

    optimizer: Literal["adam", "nadam", "lbfgs", "nelder_mead", "grid_search"] | None = Field(
        default=None, description="3.x optimizer; it drives a sharpness calibration, as in wo-4.0"
    )
    learning_rates: dict[str, PositiveFiniteFloat] = Field(
        default_factory=dict, description="gradient step size per quantity path under imaging_system"
    )
    convergence_tol: FiniteFloat | None = Field(default=None, description="3.x nelder_mead tolerance")
    convergence_patience: Count | None = Field(default=None, description="wo-4.0's; accepted and not executed")
    use_gradients: bool | None = Field(default=None, description="wo-4.0's; accepted and not executed")
    grid_points: Count | None = Field(default=None, description="3.x grid_search points")
    loss: dict[str, Any] | None = Field(default=None, description="3.x sharpness loss: type and parameters")
    log_dir: str | None = Field(default=None, description="wo-4.0's TensorBoard folder; accepted and not written")

    @model_validator(mode="after")
    def _warn_ignored(self) -> Self:
        for name in ("convergence_patience", "use_gradients", "log_dir"):
            if getattr(self, name) is not None:
                no_effect(
                    f"solver.calibration.advanced.{name}",
                    getattr(self, name),
                    "this version of WaveOrder does not execute it; remove it",
                )
        return self


def _default_search() -> list[SearchStageSettings]:
    return [SearchGradient(gradient=GradientStage())]


class Calibration(V4BaseModel):
    """How the free quantities are estimated."""

    score: Literal["evidence", "gcv", "sharpness"] = "evidence"
    search: list[SearchStageSettings] = Field(default_factory=_default_search, min_length=1)
    final_estimate: Literal["best_H", "average_near_best_H"] = Field(
        default="best_H", description="wo-4.0's; only best_H runs, and average_near_best_H has no effect"
    )
    advanced: CalibrationAdvanced | None = None

    @property
    def optimizer_drives(self) -> bool:
        """Whether ``advanced.optimizer`` is the search: under score sharpness, as in wo-4.0 (evidence, gcv: search)."""
        return self.advanced is not None and self.advanced.optimizer is not None and self.score == "sharpness"

    @property
    def budget(self) -> int | None:
        """The iterations of the 3.x optimizer: the ``max_iter`` of a lone gradient stage, as wo-4.0 read it."""
        match self.search:
            case [SearchGradient(gradient=gradient)]:
                return gradient.max_iter
        return None


class SolverSettings(V4BaseModel):
    """How the object and the free quantities are solved for, and where."""

    padding: Padding = Field(default_factory=Padding)
    reconstruction: Reconstruction = "closed_form"
    calibration: Calibration | None = None
    apodization_rolloff: FiniteFloat = Field(default=0.0, ge=0.0, le=1.0, description="inverse-filter roll-off")
    device: str | None = Field(default=None, description="null (cpu), auto, cuda:0, mps, ...")


# --- top level -------------------------------------------------------------------------------------------------------

_OUTPUTS = (
    "reconstruction",
    "transfer_function",
    "score_trace",
    "score_grid",
    "parameters",
    "psf",
    "parameters_covariance",
    "weight_trace",
    "residual",
    "tile_map",
)
_SCORES = _OUTPUTS[2:4]
"""The score arrays a calibration store may hold."""
_UNWRITTEN = frozenset(_OUTPUTS[5:])
"""The outputs wo-4.0 accepted and wrote none of: accepted here, with no effect."""

type OutputName = Annotated[str, _available(*_OUTPUTS)]
"""An output of ``outputs``. The reconstruction is always written; the score arrays go into the calibration store,
which always holds the parameters."""


def _default_outputs() -> list[OutputName]:
    return ["reconstruction", "transfer_function"]


class SimulateSettings(V4BaseModel):
    """The ``simulate`` block of a simulation."""

    seed: Integer = 0
    shape: tuple[Count, Count, Count]


_GRID_POINT_WARNING = 10_000
_LABEL = re.compile(r"[a-z]+(?:-[a-z]+)*\[.*\]|[A-Z][A-Za-z0-9]*|float|int|str|bool|none")


def _readable(error: ValidationError, data: object) -> ValidationError:
    """Return ``error`` at the paths the document writes.

    Pydantic's union labels are dropped, and so are the type errors of union members another error explains.
    """

    def path(loc: tuple[int | str, ...]) -> tuple[int | str, ...]:
        kept: list[int | str] = []
        node = data
        for part in loc:
            children: Mapping[object, object] = (
                node if isinstance(node, Mapping) else dict(enumerate(node)) if isinstance(node, list) else {}
            )
            if part in children:
                node = children[part]
            elif isinstance(part, str) and _LABEL.fullmatch(part):
                continue
            else:
                node = None
            kept.append(part)
        return tuple(kept)

    lines = [{**line, "loc": path(tuple(line["loc"]))} for line in error.errors()]

    def mismatch(line: Mapping[str, Any]) -> bool:
        return line["type"].endswith("_type") or line["type"] in ("is_instance_of", "literal_error")

    def explained(line: Mapping[str, Any]) -> bool:
        loc = line["loc"]
        return any(other is not line and not mismatch(other) and other["loc"][: len(loc)] == loc for other in lines)

    details = [
        InitErrorDetails(
            type=PydanticCustomError(
                cast(LiteralString, line["type"]),
                "{message}",
                {"message": str(line["msg"]).removeprefix("Value error, ")},
            ),
            loc=line["loc"],
            input=line["input"],
        )
        for line in lines
        if not (mismatch(line) and explained(line))
    ]
    return ValidationError.from_exception_data(error.title, details)


class V4Config(V4BaseModel):
    """A complete version-4 configuration."""

    @model_validator(mode="wrap")
    @classmethod
    def _report_paths(cls, data: object, handler: ValidatorFunctionWrapHandler) -> Self:
        try:
            return handler(data)
        except ValidationError as error:
            raise _readable(error, data) from None

    version: Literal[4]
    data: DataSettings = Field(default_factory=DataSettings)
    imaging_system: ImagingSystemSettings
    object: ObjectSettings = Field(default_factory=ObjectSettings)
    noise: NoiseModel = Field(default_factory=NoiseModel)
    solver: SolverSettings = Field(default_factory=SolverSettings)
    outputs: Literal["all"] | list[OutputName] = Field(default_factory=_default_outputs)
    simulate: SimulateSettings | None = None
    calibrated: dict[str, FiniteFloat] = Field(
        default_factory=dict, description="calibrated values by parameter name, applied after building"
    )

    @model_validator(mode="after")
    def _document_rules(self) -> Self:
        free = free_parameter_specs(self.imaging_system)
        if isinstance(self.object.prior, PriorBeads) and not free:
            raise field_error("object.prior", "beads requires a free imaging-system quantity (beads calibrate H)")
        calibration = self.solver.calibration
        if calibration is not None and isinstance(self.solver.reconstruction, ReconstructionDistilled):
            raise field_error(
                "solver.calibration",
                "needs a physics solver: a distilled solve does not depend on the free quantities; remove it",
            )
        _warn_unexecuted(self)
        _check_scores(self)
        if calibration is None or not free:
            return self
        if calibration.score == "sharpness" and self.object.weight == "auto":
            raise field_error("object.weight", "auto needs a score that estimates the weight (gcv or evidence)")
        _check_searches(calibration)
        quantities = free_parameter_specs(self.imaging_system, expand_shorthand=True)  # a grid axis per quantity
        for index, stage in enumerate(calibration.search):
            _check_stage(stage, field=f"solver.calibration.search.{index}", free=quantities)
        return self


def _warn_unexecuted(config: V4Config) -> None:
    """Warn of the settings wo-4.0 accepted that change nothing here (nor ran in wo-4.0), each at its path."""
    gaussian, outputs = config.noise.gaussian, config.outputs
    if config.simulate is None and gaussian is not None and gaussian.snr_db is not None:
        no_effect("noise.gaussian.snr_db", gaussian.snr_db, "it sets the noise of a simulation; give sigma")
    sigma = getattr(getattr(config.noise, config.noise.kind), "sigma", None)
    if isinstance(sigma, DarkFrames):
        no_effect(
            f"noise.{config.noise.kind}.sigma.dark_frames",
            sigma.dark_frames,
            "this version reads no dark frames, and wo-4.0 ran none; sigma is 1, the default",
        )
    if isinstance(config.object.prior, PriorDenoiser):
        no_effect(
            "object.prior.denoiser",
            config.object.prior.denoiser.checkpoint,
            "this version has no denoiser prior, and wo-4.0 ran none; the object is reconstructed with prior none",
        )
    source = config.object.source
    if isinstance(source, SourcePhantom | SourceFromRecon):
        ((name, _),) = source.model_dump().items()
        no_effect(f"object.source.{name}", name, "a simulation draws the gaussian_spectrum object, as wo-4.0 did")
    if config.solver.padding.yx not in (0, (0, 0)):
        no_effect("solver.padding.yx", config.solver.padding.yx, "yx padding is not in this version; nothing is padded")
    if outputs == "all":
        return
    if "reconstruction" not in outputs:
        no_effect("outputs", outputs, "the reconstruction is always written")
    for index, name in enumerate(outputs):
        if name in _UNWRITTEN:
            no_effect(f"outputs.{index}", name, "this version does not write it, and neither did wo-4.0")


def _check_scores(config: V4Config) -> None:
    """Warn of score arrays a field-tiled calibration does not keep, and of a score grid with no grid to write."""
    outputs, calibration = config.outputs, config.solver.calibration
    scores = [] if outputs == "all" else [(index, name) for index, name in enumerate(outputs) if name in _SCORES]
    if isinstance(config.imaging_system.field_dependence, FieldDependenceTiled):
        for index, name in scores:
            no_effect(f"outputs.{index}", name, "a field-tiled calibration keeps no score arrays")
        return
    grids = calibration is not None and (
        calibration.advanced is not None and calibration.advanced.optimizer == "grid_search"
        if calibration.optimizer_drives
        else any(isinstance(stage, SearchGrid) for stage in calibration.search)
    )
    for index, name in scores:
        if name == "score_grid" and not grids:
            no_effect(
                f"outputs.{index}",
                name,
                "the calibration runs no grid (a grid stage in solver.calibration.search, or advanced.optimizer: "
                "grid_search under score sharpness), so there is none to write",
            )


def _check_searches(calibration: Calibration) -> None:
    """Warn of the search settings no stage reads: wo-4.0 wrote ``advanced.optimizer`` beside ``search``.

    Under score sharpness the optimizer is the search, and reads only the ``max_iter`` of a lone gradient stage; under
    evidence and gcv the search stages run and the optimizer is read by nothing, as in wo-4.0.
    """
    at = "solver.calibration"
    if calibration.final_estimate != "best_H":
        no_effect(f"{at}.final_estimate", calibration.final_estimate, "only best_H runs, as in wo-4.0")
    for index, stage in enumerate(calibration.search):
        if isinstance(stage, SearchGradient) and stage.gradient.method != "split":
            no_effect(f"{at}.search.{index}.gradient.method", stage.gradient.method, "only split runs, as in wo-4.0")
    optimizer = None if calibration.advanced is None else calibration.advanced.optimizer
    if optimizer is None:
        return
    gradients = [stage.gradient for stage in calibration.search if isinstance(stage, SearchGradient)]
    if not calibration.optimizer_drives:
        if not gradients or any(gradient.optimizer != optimizer for gradient in gradients):
            no_effect(
                f"{at}.advanced.optimizer",
                optimizer,
                f"with score {calibration.score} the stages of {at}.search run, as in wo-4.0; it drives only a "
                "sharpness calibration",
            )
        return
    if calibration.budget is None:
        no_effect(
            f"{at}.search",
            [next(iter(stage.model_dump())) for stage in calibration.search],
            "advanced.optimizer drives a sharpness calibration, as in wo-4.0, and reads only the max_iter of a "
            "lone gradient stage",
        )
        return
    (gradient,) = gradients
    if "optimizer" in gradient.model_fields_set and gradient.optimizer != optimizer:
        no_effect(
            f"{at}.search.0.gradient.optimizer",
            gradient.optimizer,
            f"advanced.optimizer ({optimizer}) drives a sharpness calibration, as in wo-4.0",
        )


def _check_stage(stage: BaseModel, *, field: str, free: Mapping[str, FreeParameter]) -> None:
    """Refuse a grid or random stage over a quantity with no range; warn about a grid that will run long."""
    if isinstance(stage, SearchGrid | SearchRandom):
        unbounded = [name for name, spec in free.items() if spec.tolerance == "free" and spec.bounds is None]
        if unbounded:
            raise field_error(field, f"a grid or random stage needs bounds on the free quantities {unbounded}")
    if isinstance(stage, SearchGrid):
        points = stage.grid.points
        count = 1
        for name in free:
            count *= points if isinstance(points, int) else points.get(name, 7)
        if count > _GRID_POINT_WARNING:
            warnings.warn(
                ConfigurationWarning(f"the grid evaluates {count} points; use fewer or a random stage", field=field),
                stacklevel=4,
            )


def free_parameter_specs(system: ImagingSystemSettings, *, expand_shorthand: bool = False) -> dict[str, FreeParameter]:
    """Return every :class:`FreeParameter` of ``system`` by its dotted path under ``imaging_system``.

    With ``expand_shorthand`` the ``zernike_up_to_order`` shorthand is replaced by the terms it frees (those no
    named term overrides), so each key is one quantity; otherwise it is listed under its own key.
    """
    found: dict[str, FreeParameter] = {}

    def visit(node: object, prefix: str) -> None:
        if isinstance(node, FreeParameter):
            found[prefix] = node
        elif isinstance(node, PupilAberrations):
            found.update(_shorthand(node, prefix, expand=expand_shorthand))
            for name, value in node.terms.items():
                visit(value, f"{prefix}.{name}")
        elif isinstance(node, BaseModel):
            for name in type(node).model_fields:
                visit(getattr(node, name), f"{prefix}.{name}" if prefix else name)

    visit(system, "")
    return found


def _shorthand(aberrations: PupilAberrations, prefix: str, *, expand: bool) -> dict[str, FreeParameter]:
    """Return the ``zernike_up_to_order`` shorthand under its key, or (``expand``) the terms it frees."""
    shorthand = aberrations.zernike_up_to_order
    if shorthand is None:
        return {}
    spec = FreeParameter(nominal=0.0, tolerance=shorthand.tolerance, bounds=shorthand.bounds)
    if not expand:
        return {f"{prefix}.zernike_up_to_order": spec}
    named = (noll_to_name(j) for j in noll_indices_up_to_order(shorthand.order))
    return {f"{prefix}.{name}": spec for name in named if name not in aberrations.terms}


def parameter_name(path: str) -> str:
    """Return the model's parameter name of the free quantity at ``path`` under ``imaging_system``.

    ``detection.numerical_aperture`` is ``system.detection.numerical_aperture``; either pinhole diameter is the
    confocal system's ``system.pinhole_diameter``.
    """
    if path.startswith("detection.pinhole."):
        return "system.pinhole_diameter"
    return f"system.{path}"
