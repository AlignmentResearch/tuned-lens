import tempfile

import mock
import pytest
import torch as th
import transformers as trf

from tuned_lens.nn.lenses import LogitLens, TunedLens, TunedLensConfig
from tuned_lens.nn.unembed import Unembed


@pytest.fixture
def model_config():
    config = mock.MagicMock(trf.PretrainedConfig)
    config.hidden_size = 128
    config.vocab_size = 100
    config.num_hidden_layers = 3
    return config


@pytest.fixture
def model(model_config):
    model = mock.MagicMock(trf.PreTrainedModel)
    model.config = model_config
    model.get_output_embeddings = mock.MagicMock(return_value=th.nn.Linear(128, 100))
    return model


@pytest.fixture
def unembed():
    mock_unembed = mock.MagicMock(Unembed)
    W = th.randn(100, 128)
    mock_unembed.forward = lambda x: th.matmul(x, W.T)
    mock_unembed.unembedding_hash.return_value = 42
    return mock_unembed


@pytest.fixture
def logit_lens(unembed):
    logit_lens = LogitLens(unembed)
    return logit_lens


@pytest.fixture
def tuned_lens_config():
    return TunedLensConfig(
        base_model_name_or_path="test-model",
        d_model=128,
        num_hidden_layers=3,
        bias=True,
    )


@pytest.fixture
def tuned_lens(tuned_lens_config, unembed):
    tuned_lens = TunedLens(
        unembed,
        tuned_lens_config,
    )
    return tuned_lens


def test_logit_lens_smoke(logit_lens):
    randn = th.randn(1, 10, 128)
    logit_lens(randn, 0)


def test_tuned_lens_from_model(random_small_model: trf.PreTrainedModel):
    tuned_lens = TunedLens.from_model(random_small_model)
    assert tuned_lens.config.d_model == random_small_model.config.hidden_size


def test_tuned_lens_forward(tuned_lens: TunedLens):
    randn = th.randn(1, 10, 128)
    logits_forward = tuned_lens.forward(randn, 0)
    logits = tuned_lens.unembed.forward(randn + tuned_lens[0](randn))
    assert th.allclose(logits_forward, logits)


def test_tuned_lens_save_and_load(unembed: Unembed, tuned_lens: TunedLens):
    randn = th.randn(1, 10, 128)

    logits_before = tuned_lens(randn, 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        tuned_lens.save(tmpdir)
        tuned_lens = TunedLens.from_unembed_and_pretrained(
            lens_resource_id=tmpdir, unembed=unembed
        )
        logits_after = tuned_lens(randn, 1)
        assert th.allclose(logits_before, logits_after)


def test_tuned_lens_from_unemebd_and_pretrained_raises(unembed: Unembed):
    with pytest.raises(ValueError, match="Unrecognized keyword argument"):
        TunedLens.from_unembed_and_pretrained(
            unembed=unembed,
            lens_resource_id="will-never-reach",
            banana="non-existent",
            apple=1,
        )
