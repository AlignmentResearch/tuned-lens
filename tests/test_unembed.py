from tuned_lens.nn import Unembed
from tuned_lens.model_surgery import get_final_norm
import transformers as tr
import torch as th


def back_translate(unembed: Unembed, h: th.Tensor, tol: float = 1e-4) -> th.Tensor:
    """Project hidden states into logits and then back into hidden states."""
    scale = h.norm(dim=-1, keepdim=True) / h.shape[-1] ** 0.5
    logits = unembed(h)
    return unembed.invert(logits, h0=th.randn_like(h), tol=tol).preimage * scale


def test_correctness(random_small_model: tr.PreTrainedModel):
    # One problem: we want to check that we handle GPT-J's unembedding bias
    # correctly, but it's zero-initialized. Give it a random Gaussian bias.
    U = random_small_model.get_output_embeddings()
    if U.bias is not None:
        U.bias.data.normal_()

    unembed = Unembed(random_small_model)
    ln_f = get_final_norm(random_small_model)

    x = th.randn(1, 1, random_small_model.config.hidden_size)
    y = U(ln_f(x)).log_softmax(-1)  # type: ignore[attr-defined]

    th.testing.assert_close(y, unembed(x).log_softmax(-1))

    x_hat = back_translate(unembed, x, tol=1e-5)
    th.testing.assert_close(y.exp(), unembed(x_hat).softmax(-1), atol=5e-4, rtol=0.01)
