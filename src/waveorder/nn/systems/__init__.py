"""The imaging systems ``H``: the abstract bases and one file per contrast.

:mod:`~waveorder.nn.systems.base` holds :class:`~waveorder.nn.ImagingSystem` and
:class:`~waveorder.nn.ShiftInvariantSystem`; :mod:`~waveorder.nn.systems.phase` holds brightfield phase,
:mod:`~waveorder.nn.systems.fluorescence` fluorescence, and :mod:`~waveorder.nn.systems.tiled` field-dependent optics
as independent tiles. A system is named for its contrast and object dimension: ``ThickPhase3D``, ``ThinPhase2D``,
``WidefieldFluorescence3D``, ``ConfocalFluorescence3D``, ``ThinFluorescence2D``.
"""
