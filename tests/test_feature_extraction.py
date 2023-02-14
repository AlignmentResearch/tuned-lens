from tuned_lens.residual_stream import record_residual_stream
from transformers import GPT2LMHeadModel, GPT2Config
import torch as th


def test_record_residual_stream():
    """
    Test that every other residual stream state is identical to the corresponding
    hidden state reported by HuggingFace.
    """
    model = GPT2LMHeadModel(GPT2Config())

    with record_residual_stream(model, sublayers=True) as stream:
        outputs = model(th.ones(1, 1, dtype=th.long), output_hidden_states=True)

    # Exclude the final hidden state because HF applies ln_f while we don't
    their_hiddens = outputs.hidden_states[1:-1]
    for their_hidden, our_hidden in zip(their_hiddens, stream.layers):
        th.testing.assert_close(their_hidden, our_hidden)
