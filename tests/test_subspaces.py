from tuned_lens.causal import remove_subspace
import pytest
import torch as th


@pytest.mark.parametrize("d", list(range(1, 1000, 100)))
def test_remove_subspace(d: int):
    a = th.randn(10, d, dtype=th.float64)

    for k in range(1, d, 10):
        b = th.randn(d, k, dtype=th.float64)
        inner = a @ b

        a_ = remove_subspace(a, b, mode="zero")
        inner_ = a_ @ b
        th.testing.assert_close(inner_, th.zeros_like(inner_))

        a_ = remove_subspace(a, b, mode="mean")
        inner_ = a_ @ b
        th.testing.assert_close(inner_, inner.mean(0, keepdim=True).expand_as(inner_))
