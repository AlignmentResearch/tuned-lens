from torch.distributions import Dirichlet, kl_divergence
from tuned_lens.stats import LogitStats, ResidualStats
from tuned_lens.residual_stream import ResidualStream
import pytest
import random
import torch as th


def test_logit_stats_correctness():
    """Test that `LogitStats` recovers the true Dirichlet within a small error."""
    th.manual_seed(42)

    x = Dirichlet(th.tensor([1.0, 1.0, 1.0]))
    logits1 = x.sample(th.Size([10000])).log() + random.uniform(-0.1, 0.1)
    logits2 = x.sample(th.Size([10000])).log() + random.uniform(-0.1, 0.1)

    stats = LogitStats()
    stats.update(logits1)
    stats.update(logits2)
    x2 = stats.mle()

    assert kl_divergence(x, x2) < 1e-3


CONFIGS: list[tuple[str, th.dtype]] = [("cpu", th.float32)]
if th.cuda.is_available():
    # Don't require CUDA for tests
    CONFIGS.append(("cuda", th.float16))
    CONFIGS.append(("cuda", th.float32))


@pytest.mark.parametrize("config", CONFIGS)
def test_residual_stats_correctness(config):
    """Test that the `ResidualStats` mean and covariance are correct."""
    device, dtype = config

    th.manual_seed(42)
    N = 100  # Number of batches
    B = 2  # Batch size
    D = 256  # Hidden dimension
    L = 3  # Number of layers
    S = 128  # Sequence length

    # Generate correlated random data from a multivariate Gaussian
    A = th.randn(D, D, device=device) / D**0.5
    master_cov = A @ A.T  # Ensure positive definite
    dist = th.distributions.MultivariateNormal(
        loc=th.randn(D, device=device), covariance_matrix=master_cov
    )
    batch_dims = th.Size([B, S])

    stats = ResidualStats()
    streams = []
    for _ in range(N):
        stream = ResidualStream(
            embeddings=dist.sample(batch_dims).to(dtype),
            attentions=[dist.sample(batch_dims).to(dtype)] * L,
            layers=[dist.sample(batch_dims).to(dtype)] * L,
        )
        streams.append(stream)
        stats.update(stream)

    # Stack all the streams together so we can compute the non-streaming stats
    master = ResidualStream.stack(streams).map(lambda h: h.view(N * B * S, D))

    for h, mu, cov, norm in zip(
        master, stats.mean(), stats.covariance(), stats.mean_norm()
    ):
        th.testing.assert_close(mu.to(h.dtype), h.mean(0))
        th.testing.assert_close(cov.to(h.dtype), h.T.cov(), atol=0.005, rtol=1e-5)
        th.testing.assert_close(norm.to(h.dtype), h.norm(dim=-1).mean())

        # We should at least get close-ish to the true covariance
        assert th.dist(cov, master_cov) / th.norm(master_cov) < 0.1
