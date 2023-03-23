"""Provides implementation of spearman corelation for logit vectors."""
from typing import Optional
import torch as th


def spearmanr(
    x1: th.Tensor, x2: Optional[th.Tensor] = None, dim: int = -1
) -> th.Tensor:
    """Compute the Spearman rank correlation coefficient between two tensors.

    When `x2` is not provided, the correlation is computed between the ranks of `x1`
    and `th.arange(x1.shape[dim])`. This can be viewed as a measure of how well the
    elements of `x1` are ordered.
    """
    assert x1.shape[dim] > 1, f"Expected at least 2 elements along dim {dim}"
    assert x2 is None or x1.shape == x2.shape, "Shapes of input tensors must match"

    shape = [1] * x1.ndim
    shape[dim] = -1
    ranks = th.arange(x1.shape[dim], dtype=x1.dtype, device=x1.device).view(*shape)

    # Convert argsort indices into ranks
    rank1 = th.empty_like(x1).scatter_(
        dim, index=x1.argsort(dim), src=ranks.expand_as(x1)
    )
    if x2 is None:
        rank2 = ranks.expand_as(x1)
    else:
        rank2 = th.empty_like(x2).scatter_(
            dim, index=x2.argsort(dim), src=ranks.expand_as(x2)
        )

    # Pearson correlation between the ranks
    var1, mean1 = th.var_mean(rank1, dim)
    var2, mean2 = th.var_mean(rank2, dim)
    return (th.mean(rank1 * rank2, dim=dim) - mean1 * mean2) / th.sqrt(var1 * var2)
