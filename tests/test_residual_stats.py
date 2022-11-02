from white_box import ResidualStats, ResidualStream
import torch as th


def test_correctness():
    """Test that the mean and covariance are correct."""

    th.manual_seed(42)
    N = 100  # Number of batches
    B = 2  # Batch size
    D = 256  # Hidden dimension
    L = 3  # Number of layers
    S = 128  # Sequence length

    # Generate correlated random data from a multivariate Gaussian
    A = th.randn(D, D) / D**0.5
    master_cov = A @ A.T  # Ensure positive definite
    dist = th.distributions.MultivariateNormal(th.randn(D), master_cov)
    batch_dims = th.Size([B, S])

    stats = ResidualStats()
    streams = []
    for _ in range(N):
        stream = ResidualStream(
            embeddings=dist.sample(batch_dims),
            attentions=[dist.sample(batch_dims)] * L,
            layers=[dist.sample(batch_dims)] * L,
        )
        streams.append(stream)
        stats.update(stream)

    # Stack all the streams together so we can compute the non-streaming stats
    master = ResidualStream.stack(streams).map(lambda h: h.view(N * B * S, D))

    for h, mu, cov, norm in zip(master, stats.mean, stats.covariance, stats.mean_norm):
        norm_true = h.norm(dim=-1).mean()

        th.testing.assert_close(mu, h.mean(0))
        th.testing.assert_close(cov, h.T.cov(), atol=0.005, rtol=1e-5)
        th.testing.assert_close(norm_true, norm)

        # We should at least get close-ish to the true covariance
        assert th.dist(cov, master_cov) / th.norm(master_cov) < 0.1
