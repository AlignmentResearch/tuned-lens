from white_box.nn import Decoder
from transformers import AutoModelForCausalLM
import torch as th


def test_correctness():
    th.manual_seed(42)

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    decoder = Decoder(model)

    U = model.get_output_embeddings()
    x = th.randn(1, 1, 768)
    y = U(model.base_model.ln_f(x)).log_softmax(-1)  # type: ignore[attr-defined]

    th.testing.assert_close(y, decoder(x).log_softmax(-1))


def test_back_translation():
    th.manual_seed(42)

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    decoder = Decoder(model)

    U = model.get_output_embeddings()
    x = th.randn(1, 1, 768)
    y = U(model.base_model.ln_f(x)).softmax(-1)  # type: ignore[attr-defined]

    x_hat = decoder.back_translate(x, tol=1e-5)
    th.testing.assert_close(y, decoder(x_hat).softmax(-1), atol=1e-4, rtol=0.01)
