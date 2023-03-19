"""A set of PyTorch modules for transforming the residual streams of models."""
from .decoder import (
    Decoder,
    InversionOutput,
)
from .downstream_wrapper import DownstreamWrapper
from .probe_dict import ProbeDict
from .lenses import TunedLens, LogitLens
