"""The version-4 configuration: its schema, its loader and the migration of 3.x files.

A configuration groups a reconstruction into ``data``, ``imaging_system``, ``object``, ``noise`` and ``solver``, plus
``outputs``, ``simulate``, ``calibrated`` and a required ``version``. :func:`load` reads and validates one;
:meth:`waveorder.models.ReconstructionModel.from_config` builds its model (:func:`waveorder.config.build.build`).
"""

from waveorder.config.loader import CONFIG_ERRORS, dump, dump_yaml, expect_version, load, problems, read, suggest_key
from waveorder.config.schema import (
    Calibration,
    DataSettings,
    DetectionSettings,
    FreeParameter,
    IlluminationSettings,
    ImagingSystemSettings,
    NoiseModel,
    ObjectSettings,
    PixelSize,
    PupilAberrations,
    SolverSettings,
    V4Config,
    free_parameter_specs,
    nominal_of,
    parameter_name,
)

__all__ = [
    "CONFIG_ERRORS",
    "Calibration",
    "DataSettings",
    "DetectionSettings",
    "FreeParameter",
    "IlluminationSettings",
    "ImagingSystemSettings",
    "NoiseModel",
    "ObjectSettings",
    "PixelSize",
    "PupilAberrations",
    "SolverSettings",
    "V4Config",
    "dump",
    "dump_yaml",
    "expect_version",
    "free_parameter_specs",
    "load",
    "nominal_of",
    "parameter_name",
    "problems",
    "read",
    "suggest_key",
]
