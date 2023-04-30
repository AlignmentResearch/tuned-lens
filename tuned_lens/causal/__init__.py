"""Tools for finding and intervening on important subspaces of the residual stream."""
from .subspaces import (
    CausalBasis,
    ablate_subspace,
    extract_causal_bases,
    remove_subspace,
)
from .utils import derange, sample_derangement
