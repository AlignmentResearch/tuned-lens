from typing import cast
import math
import torch as th
import torch.distributions as D


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


def js_divergence(logit_p: th.Tensor, logit_q: th.Tensor, dim: int = -1) -> th.Tensor:
    """
    Compute the Jensen-Shannon divergence between two sets of logits.

    Conceptually, the JSD is the info value of learning which of two distributions,
    P or Q, that a random variable is drawn from, starting from a uniform prior over
    P and Q. Since the entropy of a Bernoulli variable is at most ln(2), the JSD is
    guaranteed to be in the range [0, ln(2)]. It is also symmetric and finite even
    for distributions with disjoint supports.

    Mathematically, the JSD is simply [KL(P || M) + KL(Q || M)] / 2, where M
    is the mean of P and Q.
    """
    log_p = logit_p.log_softmax(dim)
    log_q = logit_q.log_softmax(dim)

    # Mean of P and Q
    log_m = th.stack([log_p, log_q]).sub(math.log(2)).logsumexp(0)

    kl_p = th.sum(log_p.exp() * (log_p - log_m), dim)
    kl_q = th.sum(log_q.exp() * (log_q - log_m), dim)
    return 0.5 * (kl_p + kl_q)


def js_distance(logit_p: th.Tensor, logit_q: th.Tensor, dim: int = -1) -> th.Tensor:
    """Compute the square root of the Jensen-Shannon divergence of two logit vectors."""
    return js_divergence(logit_p, logit_q, dim).sqrt()


def kl_divergence(logit_p: th.Tensor, logit_q: th.Tensor, dim: int = -1) -> th.Tensor:
    """Compute the KL divergence between two sets of logits."""
    log_p = logit_p.log_softmax(dim)
    log_q = logit_q.log_softmax(dim)
    return th.sum(log_p.exp() * (log_p - log_q), dim)


def sqrtmh(x: th.Tensor) -> th.Tensor:
    """Unique PSD square root of a Hermitian positive semi-definite matrix"""
    dtype = x.dtype

    # This is actually precision-sensitive
    L, Q = th.linalg.eigh(x.double())
    res = Q * L.clamp(0.0).sqrt() @ Q.mH
    return res.to(dtype)


def gaussian_wasserstein_l2(p: D.MultivariateNormal, q: D.MultivariateNormal):
    """Analytically compute the 2-Wasserstein distance between two Gaussians"""
    cov_p = cast(th.Tensor, p.covariance_matrix).double()
    cov_q = cast(th.Tensor, q.covariance_matrix).double()
    cov_q_sqrt = sqrtmh(cov_q)

    loc_dist = th.square(p.loc.double() - q.loc.double()).sum(-1)
    traces = th.trace(cov_p) + th.trace(cov_q)
    interaction = sqrtmh(cov_q_sqrt @ cov_p @ cov_q_sqrt)
    dist_sq = loc_dist + traces - 2 * th.trace(interaction)
    return dist_sq.clamp(0.0).sqrt()


def gaussian_wasserstein_l2_origin(p: D.MultivariateNormal):
    """Compute the 2-Wasserstein distance between a Gaussian and the origin"""
    cov_p = cast(th.Tensor, p.covariance_matrix)
    return th.sqrt(th.square(p.loc).sum(-1) + th.trace(cov_p))
