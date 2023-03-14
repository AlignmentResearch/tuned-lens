from tuned_lens.nn.lenses import TunedLens, LogitLens
import transformers as trf

import tempfile
import torch as th
import pytest
import mock


@pytest.fixture
def logit_lens():
    model = mock.MagicMock(trf.PreTrainedModel)
    model.config = mock.MagicMock(trf.PretrainedConfig)
    model.config.hidden_size = 128
    model.config.num_layers = 3
    model.config.vocab_size = 100
    model.get_output_embeddings = mock.MagicMock(return_value=th.nn.Linear(128, 100))

    with mock.patch("tuned_lens.model_surgery.get_final_layer_norm") as get_final_ln:
        get_final_ln.return_value = th.nn.LayerNorm(128)

    return LogitLens(model)


@pytest.fixture
def tuned_lens():
    return TunedLens(d_model=128, num_layers=3, vocab_size=100)


def test_logit_lens_smoke(logit_lens):
    randn = th.randn(1, 10, 128)
    logit_lens(randn, 0)


def test_tuned_lens_smoke(tuned_lens: TunedLens):
    randn = th.randn(1, 10, 128)
    logits_forward = tuned_lens(randn, 0)
    logits = tuned_lens.unembedding(
        tuned_lens.layer_norm(randn + tuned_lens[0](tuned_lens.layer_norm(randn)))
    )
    assert th.allclose(logits_forward, logits)


def test_tuned_lens_save_and_load(tuned_lens):
    randn = th.randn(1, 10, 128)

    logits_before = tuned_lens(randn, 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        tuned_lens.save(tmpdir)
        tuned_lens = TunedLens.load(tmpdir)
        logits_after = tuned_lens(randn, 1)
        assert th.allclose(logits_before, logits_after)
