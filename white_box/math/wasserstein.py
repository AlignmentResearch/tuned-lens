from typing import cast
import torch as th
import torch.distributions as D


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
    """Analytically compute the 2-Wasserstein distance between a Gaussian and the origin"""
    cov_p = cast(th.Tensor, p.covariance_matrix)
    return th.sqrt(th.square(p.loc).sum(-1) + th.trace(cov_p))
