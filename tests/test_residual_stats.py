from white_box import ResidualStats, ResidualStream
import pytest
import torch as th


@pytest.mark.parametrize("pool", [True, False])
def test_correctness(pool: bool):
    """Test that the stats are correct."""
    B = 2
    D = 768
    L = 12
    S = 2048

    stats = ResidualStats(pool=pool)
    stream = ResidualStream(
        embeddings=th.randn(B, S, D),
        attentions=[th.randn(B, S, D)] * L,
        layers=[th.randn(B, S, D)] * L,
    )
    stats.update(stream)
    assert stats.mean and stats.M2 and stats.autocorr

    for h, mu, var, autocorr in zip(stream, stats.mean, stats.variance, stats.autocorr):
        autocorr_tgt = th.mean(h[:, 1:] * h[:, :-1], dim=(0, 1) if pool else 0)
        th.testing.assert_allclose(autocorr_tgt, autocorr)
        if pool:
            h = h.reshape(-1, D)

        th.testing.assert_allclose(h.mean(0), mu)
        th.testing.assert_allclose(h.var(0), var)
