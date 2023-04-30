"""Various distance metrics for probability distributions."""
import math

import torch as th


def js_divergence(logit_p: th.Tensor, logit_q: th.Tensor, dim: int = -1) -> th.Tensor:
    """Compute the Jensen-Shannon divergence between two sets of logits.

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
    """Unique PSD square root of a Hermitian positive semi-definite matrix."""
    dtype = x.dtype

    # This is actually precision-sensitive
    L, Q = th.linalg.eigh(x.double())
    res = Q * L.clamp(0.0).sqrt() @ Q.mH
    return res.to(dtype)
