from tuned_lens.model_surgery import get_final_norm
from tuned_lens.nn import Decoder
import transformers as tr
import torch as th


def correctness(random_small_model: tr.PreTrainedModel):
    # One problem: we want to check that we handle GPT-J's unembedding bias
    # correctly, but it's zero-initialized. Give it a random Gaussian bias.
    U = random_small_model.get_output_embeddings()
    if U.bias is not None:
        U.bias.data.normal_()

    decoder = Decoder(random_small_model)
    ln_f = get_final_norm(random_small_model)

    x = th.randn(1, 1, random_small_model.config.hidden_size)
    y = U(ln_f(x)).log_softmax(-1)  # type: ignore[attr-defined]

    th.testing.assert_close(y, decoder(x).log_softmax(-1))

    x_hat = decoder.back_translate(x, tol=1e-5)
    th.testing.assert_close(y.exp(), decoder(x_hat).softmax(-1), atol=5e-4, rtol=0.01)
