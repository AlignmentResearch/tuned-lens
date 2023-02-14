import torch as th


def effective_rank(A: th.Tensor, num_rogue_dims: int = 0) -> th.Tensor:
    """Return the perplexity (exponentiated entropy) of the singular values of A.

    Intuitively, the effective rank of a matrix measures how uniformly distributed its
    singular values are. For any nonzero matrix A, 1 <= effective_rank(A) <= rank(A),
    with equality iff its singular values are all equal. For square matrices, it
    follows that erank(A) = rank(A) iff A is a scalar multiple of a unitary matrix.
    In general, the effective rank is invariant to isotropic scaling, rotation, and
    transposition.

    See "The effective rank: A measure of effective dimensionality" for more details.
    Link: https://infoscience.epfl.ch/record/110188/files/RoyV07.pdf.
    """
    *_, m, n = A.shape
    max_rank = min(m, n)

    sigma = th.linalg.svdvals(A)
    if num_rogue_dims:
        sigma = sigma[num_rogue_dims:]

    probs = sigma / sigma.sum(dim=-1, keepdim=True)
    entropy = -th.sum(probs * th.log(probs), dim=-1)

    # Sometimes due to numerical issues the exponentiated entropy can come out
    # slightly higher than the matrix rank. Clamp the output to fix this.
    return entropy.exp().clamp(0, max_rank)
