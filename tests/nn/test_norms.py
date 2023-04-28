import torch as th
from torch import nn

from tuned_lens.nn.norms import LayerNorm, RMSNorm


def test_layer_norm():
    x = th.randn(5, 5)
    normalized_shape = [5]
    eps = 1e-6

    ln = LayerNorm(normalized_shape, eps=eps)
    ln_expected = nn.LayerNorm(normalized_shape, eps=eps)

    ln_out = ln(x)
    ln_expected_out = ln_expected(x)

    assert th.allclose(ln_out, ln_expected_out)


def test_llama_rms_norm():
    x = th.randn(5, 5)
    hidden_size = 5
    eps = 1e-6

    rms = RMSNorm(hidden_size, eps=eps)

    rms_out = rms(x)
    rms_expected_out = x * th.rsqrt((x**2).mean(-1, keepdim=True) + eps)

    assert th.allclose(rms_out, rms_expected_out)
