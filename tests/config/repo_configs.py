"""Every reconstruction configuration YAML committed to the repository.

The design document (section 2) requires the regression tests to cover
every 3.x configuration in the repository and in the published examples.
This module is the single source of that list;
``test_migrate.test_every_repo_config_is_listed`` guards it against new
configuration directories appearing without coverage.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent

#: Directories holding reconstruction configuration files.
CONFIG_DIRS = (
    REPO_ROOT / "docs/examples/cli/configs",
    REPO_ROOT / "docs/examples/optimization",
    REPO_ROOT / "benchmarks/configs",
)

ALL_CONFIG_PATHS = sorted(path for config_dir in CONFIG_DIRS for path in config_dir.glob("*.yml"))

#: Birefringence stays 3.x this release (design document section 2).
BIREFRINGENCE_CONFIG_PATHS = [p for p in ALL_CONFIG_PATHS if "birefringence" in p.name]


def _is_tile_stitch_config(path: Path) -> bool:
    """Whether a configuration is a ``wo tile-stitch`` configuration.

    Tile-stitch configurations wrap a reconstruction block under a top-level
    ``recon`` key alongside ``tile`` / ``blend`` stitching parameters, rather
    than being a plain ``wo rec`` configuration. Detection is by structure, not
    filename, so new tile-stitch cases are classified correctly regardless of
    how they are named.

    Parameters
    ----------
    path : Path
        Path to a configuration YAML file.

    Returns
    -------
    bool
        ``True`` if the configuration has the tile-stitch structure.
    """
    try:
        doc = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(doc, dict) and "recon" in doc and ("tile" in doc or "blend" in doc)


#: Tile-stitch configurations wrap a reconstruction block in stitching
#: parameters; they are not plain ``wo rec`` configurations.
TILE_STITCH_CONFIG_PATHS = [p for p in ALL_CONFIG_PATHS if _is_tile_stitch_config(p)]

#: Every 3.x phase / fluorescence configuration in the repository.
MIGRATABLE_CONFIG_PATHS = [
    p for p in ALL_CONFIG_PATHS if p not in BIREFRINGENCE_CONFIG_PATHS and p not in TILE_STITCH_CONFIG_PATHS
]
