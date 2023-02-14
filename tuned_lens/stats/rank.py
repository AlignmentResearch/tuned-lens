from typing import Optional, NamedTuple
import math
import torch as th


class NearestNeighbors(NamedTuple):
    """Return type of `nearest_neighbors`."""

    kl_divergences: th.Tensor
    indices: th.Tensor


def nearest_neighbors(x: th.Tensor) -> NearestNeighbors:
    """Find index of the KL nearest neighbor of each logit vector in a batch.

    Args:
        x: Tensor of shape [..., N, C] where C is the number of classes and N is the
            number of logit vectors from which neighbors are selected.

    Returns:
        A namedtuple of tensors with shape [..., N]. The first tensor contains the KL
        divergence of each logit vector to its nearest neighbor. The second tensor
        contains the indices of the neighbors along the penultimate dimension of `x`.
    """
    # Normalize logits to log probabilities
    log_p = x.log_softmax(-1)

    # Matrix of pairwise cross-entropies
    H = -log_p.exp() @ log_p.mT

    # The second smallest value in each row is the nearest neighbor
    H_p_q, indices = H.kthvalue(2)
    return NearestNeighbors(
        # KL = H(P, Q) - H(P)
        H_p_q - th.linalg.diagonal(H),
        indices,
    )


def sample_neighbors(x, tau: float = 1.0, *, generator: Optional[th.Generator] = None):
    """Sample neighbors q of logit vectors p, inversely proportional to KL(p, q)."""
    # If temperature is infinite, sample uniformly
    if not math.isfinite(tau):
        return th.randperm(x.shape[0], device=x.device, generator=generator)

    # Matrix of pairwise cross-entropies
    log_p = x.log_softmax(-1)
    H = -log_p.exp() @ log_p.mT

    # If temperature is zero, sample the nearest neighbor
    if tau == 0.0:
        # The second smallest value in each row is the nearest neighbor
        return H.kthvalue(2).indices

    kl = H - th.linalg.diagonal(H).unsqueeze(-1)
    th.linalg.diagonal(kl).fill_(th.inf)
    dist = kl.neg().div(tau).softmax(-1)

    samples = dist.flatten(0, -2).multinomial(1, generator=generator)
    return samples.view(*dist.shape[:-1])


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
