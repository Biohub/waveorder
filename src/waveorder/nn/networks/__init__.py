"""Students: networks trained to give the estimate a physics solver gives, and what inference needs to run them.

A student (:class:`RestorationNetwork`) maps data ``(B, C, Z, Y, X)`` and named conditions ``(B, K)`` to an object
estimate ``(B, F, Zout, Y, X)``. :class:`UNet` composes a stem, an encoder, a head and a conditioning;
:class:`SlidingPlanes` runs a plane student over a volume. A :class:`Bundle` carries the student with its
:class:`Contract`, conditions and normalizers to :class:`~waveorder.nn.Distilled`. Everything here needs torch only.
Every class a bundle rebuilds is registered by name in :mod:`waveorder.registry`, e.g.
``@waveorder.registry.register("my_net", kind="network")``.
"""

from waveorder.nn.networks.base import PRECISIONS, Case, Contract, Precision, RestorationNetwork
from waveorder.nn.networks.bundle import (
    Affine,
    Bundle,
    Example,
    FieldStandardize,
    Identity,
    Normalizer,
    autocast,
    predict,
)
from waveorder.nn.networks.conditioning import (
    Attributes,
    Bias,
    ConcatPlanes,
    Condition,
    Conditioning,
    FieldPosition,
    Place,
    PlaneOffsets,
    Quantities,
    SlidingPlanes,
)
from waveorder.nn.networks.encoders import Encoder, EncoderModule, ResNetEncoder, TimmEncoder
from waveorder.nn.networks.heads import Head, PlaneHead, SubPixelHead
from waveorder.nn.networks.stems import DefocusKernel, PlaneProjection, Stem, StemModule, StridedVolume
from waveorder.nn.networks.unet import UNet

__all__ = [
    "PRECISIONS",
    "Affine",
    "Attributes",
    "Bias",
    "Bundle",
    "Case",
    "ConcatPlanes",
    "Condition",
    "Conditioning",
    "Contract",
    "DefocusKernel",
    "Encoder",
    "EncoderModule",
    "Example",
    "FieldPosition",
    "FieldStandardize",
    "Head",
    "Identity",
    "Normalizer",
    "Place",
    "PlaneHead",
    "PlaneOffsets",
    "PlaneProjection",
    "Precision",
    "Quantities",
    "ResNetEncoder",
    "RestorationNetwork",
    "SlidingPlanes",
    "Stem",
    "StemModule",
    "StridedVolume",
    "SubPixelHead",
    "TimmEncoder",
    "UNet",
    "autocast",
    "predict",
]
