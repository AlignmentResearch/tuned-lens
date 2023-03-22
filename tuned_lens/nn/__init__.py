"""A set of PyTorch modules for transforming the residual streams of models."""
from .decoder import (
    Unembed,
    InversionOutput,
)
from .downstream_wrapper import DownstreamWrapper
from .lenses import TunedLens, LogitLens
