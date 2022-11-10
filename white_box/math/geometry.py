import torch as th


def geodesic_distance(
    logit_p: th.Tensor, logit_q: th.Tensor, dim: int = -1
) -> th.Tensor:
    """
    Compute the length of the Fisher-Rao geodesic connecting two logit vectors.

    Guaranteed to be in the range [0, pi]. See https://arxiv.org/abs/2106.05367,
    Equation 11 and Appendix B for derivation.
    """
    log_p = logit_p.log_softmax(dim)
    log_q = logit_q.log_softmax(dim)

    # Usually written sqrt(p * q) but this is more numerically stable
    affinity = th.exp(0.5 * (log_p + log_q)).sum(dim)
    return 2 * affinity.clip(-1.0, 1.0).acos()
