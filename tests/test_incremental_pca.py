from white_box.math import IncrementalPCA
import torch as th


def test_correctness():
    """Test that the algorithm is correct."""
    D = 768
    S = 2048

    pca = IncrementalPCA()
    X = th.randn(4, 2048, 768)
    for batch in X:
        pca.update(batch)

    # Ground truth
    X = X.reshape(-1, D)
    mu = X.mean(0)
    U, S, _ = th.linalg.svd(th.t(X - mu))

    assert pca.U is not None and pca.S is not None
    th.testing.assert_close(pca.mean, mu)
    th.testing.assert_close(pca.U.abs(), U.abs(), atol=2e-3, rtol=1e-4)
    th.testing.assert_close(pca.S, S, atol=1e-3, rtol=1e-4)
