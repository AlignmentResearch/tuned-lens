from typing import Optional

import torch as th


def derange(batch: th.Tensor, generator: Optional[th.Generator] = None) -> th.Tensor:
    """Shuffle a tensor along axis 0, making sure there are no fixed points."""
    # Things get more complicated if there are multiple ranks. We perform the
    # derangement *hierarchically*, first generating a shared permutation of the ranks
    indices = sample_derangement(
        batch.shape[0], device=batch.device, generator=generator
    )
    return batch[indices]


def sample_derangement(
    n: int,
    device: th.device = th.device("cpu"),
    generator: Optional[th.Generator] = None,
) -> th.Tensor:
    """Uniformly sample a random permutation with no fixed points."""
    if n < 2:
        raise ValueError("Derangements only exist for n > 1")

    indices = th.arange(n, device=device)
    permutation = th.randperm(n, device=device, generator=generator)

    # Reject any permutations with fixed points. This seems inefficient,
    # but the expected number of th.randperm calls is actually O(1); it
    # asymptotically approaches e ≈ 2.7.
    # See https://www.cs.upc.edu/~conrado/research/talks/analco08.pdf.
    while th.any(permutation == indices):
        permutation = th.randperm(n, device=device, generator=generator)

    return permutation
