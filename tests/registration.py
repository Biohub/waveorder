"""Register a part class for the length of a test."""

from collections.abc import Generator
from contextlib import contextmanager

from waveorder import registry


@contextmanager
def temporarily(name: str, cls: type, *, kind: str) -> Generator[None]:
    """Register ``cls`` under ``name`` for the ``with`` block, then remove it."""
    registry.register(name, kind=kind)(cls)
    try:
        yield
    finally:
        registry._classes.pop(name, None)  # noqa: SLF001 (the registry has no public removal; tests only)
        registry._names.pop(cls, None)  # noqa: SLF001
        registry._builders.pop(cls, None)  # noqa: SLF001
