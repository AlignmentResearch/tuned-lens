from tuned_lens.model_surgery import get_final_norm
from tuned_lens.nn import Decoder
import transformers as tr
import pytest
import torch as th


def correctness(model_str: str):
    th.manual_seed(42)

    # We use a random model with the correct config instead of downloading the
    # whole pretrained checkpoint.
    config = tr.AutoConfig.from_pretrained(model_str)
    model = tr.AutoModelForCausalLM.from_config(config)

    # One problem: we want to check that we handle GPT-J's unembedding bias
    # correctly, but it's zero-initialized. Give it a random Gaussian bias.
    U = model.get_output_embeddings()
    if U.bias is not None:
        U.bias.data.normal_()

    decoder = Decoder(model)
    ln_f = get_final_norm(model)

    x = th.randn(1, 1, config.hidden_size)
    y = U(ln_f(x)).log_softmax(-1)  # type: ignore[attr-defined]

    th.testing.assert_close(y, decoder(x).log_softmax(-1))

    x_hat = decoder.back_translate(x, tol=1e-5)
    th.testing.assert_close(y.exp(), decoder(x_hat).softmax(-1), atol=5e-4, rtol=0.01)


@pytest.mark.slow
def test_correctness_slow():
    correctness("EleutherAI/gpt-j-6B")


@pytest.mark.parametrize(
    "model_str",
    [
        "EleutherAI/pythia-125m",
        "bigscience/bloom-560m",
        "EleutherAI/gpt-neo-125M",
        "facebook/opt-125m",
        "gpt2",
    ],
)
def test_correctness(model_str: str):
    correctness(model_str)
