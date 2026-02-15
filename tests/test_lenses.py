from pathlib import Path

import mock
import pytest
import torch as th
import transformers as trf

from tuned_lens.load_artifacts import load_lens_artifacts
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
    mock_unembed.unembedding = th.nn.Linear(128, 100)
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
def random_tuned_lens(tuned_lens_config, unembed):
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


def test_tuned_lens_forward(random_tuned_lens: TunedLens):
    randn = th.randn(1, 10, 128)
    logits_forward = random_tuned_lens.forward(randn, 0)
    logits = random_tuned_lens.unembed.forward(randn + random_tuned_lens[0](randn))
    assert th.allclose(logits_forward, logits)


def test_tuned_lens_save_and_load(
    unembed: Unembed, random_tuned_lens: TunedLens, tmp_path: Path
):
    randn = th.randn(1, 10, 128)

    logits_before = random_tuned_lens(randn, 1)
    random_tuned_lens.save(tmp_path)
    reloaded_tuned_lens = TunedLens.from_unembed_and_pretrained(
        lens_resource_id=tmp_path, unembed=unembed
    )
    logits_after = reloaded_tuned_lens(randn, 1)
    assert th.allclose(logits_before, logits_after)


def test_from_model_and_pretrained_propogates_kwargs(
    random_tuned_lens: TunedLens, unembed: Unembed, tmp_path: Path
):
    random_tuned_lens.save(tmp_path)

    with mock.patch(
        "tuned_lens.load_artifacts.load_lens_artifacts",
        mock.MagicMock(
            load_lens_artifacts,
            return_value=(tmp_path / "config.json", tmp_path / "params.pt"),
        ),
    ) as mock_load_lens_artifacts:
        mock_load_lens_artifacts.__code__.co_varnames = (
            "resource_id",
            "unembed",
            "revision",
        )
        TunedLens.from_unembed_and_pretrained(
            lens_resource_id="does not use", unembed=unembed, revision="foo"
        )
        assert mock_load_lens_artifacts.call_args.kwargs["revision"] == "foo"

        with pytest.raises(TypeError):
            # Should not just be able to pass any kwarg
            TunedLens.from_unembed_and_pretrained(
                lens_resource_id="does not use",
                unembed=unembed,
                revision="foo",
                bad_kwarg="bar",
            )

        with pytest.raises(TypeError):
            # Should not be able to specify both resource_id and and lens_resource_id
            TunedLens.from_unembed_and_pretrained(
                lens_resource_id="does not use", unembed=unembed, resource_id="bar"
            )


def test_tuned_lens_generate_smoke(random_small_model: trf.PreTrainedModel):
    tuned_lens = TunedLens.from_model(random_small_model)
    bos_token_id = random_small_model.config.bos_token_id
    input_ids = th.tensor([bos_token_id])
    tokens = tuned_lens.generate(
        model=random_small_model,
        layer=2,
        do_sample=True,
        input_ids=input_ids,
        max_new_tokens=10,
    )
    assert tokens.shape[-1] <= 11
    assert tokens.shape[-1] > 1
    assert input_ids == tokens[:, :1]
    assert input_ids == th.tensor([bos_token_id]), "Don't mutate input_ids!"

    tokens = tuned_lens.generate(
        model=random_small_model,
        layer=2,
        input_ids=input_ids,
        do_sample=False,
        max_new_tokens=10,
    )
    assert tokens.shape[-1] <= 11
    assert tokens.shape[-1] > 1


# --- Tests for negative indexing ---


def test_tuned_lens_negative_index(random_tuned_lens: TunedLens):
    """Negative index -1 should return the last translator."""
    last = random_tuned_lens[-1]
    explicit_last = random_tuned_lens[len(random_tuned_lens) - 1]
    assert last is explicit_last


def test_tuned_lens_negative_index_minus_n(random_tuned_lens: TunedLens):
    """Negative index -N should return the first translator."""
    n = len(random_tuned_lens)
    first = random_tuned_lens[-n]
    explicit_first = random_tuned_lens[0]
    assert first is explicit_first


def test_tuned_lens_index_out_of_range(random_tuned_lens: TunedLens):
    """Out-of-range indices should raise IndexError."""
    n = len(random_tuned_lens)
    with pytest.raises(IndexError):
        random_tuned_lens[n]
    with pytest.raises(IndexError):
        random_tuned_lens[-(n + 1)]


def test_tuned_lens_forward_negative_idx(random_tuned_lens: TunedLens):
    """forward() should accept negative layer indices."""
    randn = th.randn(1, 10, 128)
    logits_neg = random_tuned_lens.forward(randn, -1)
    logits_pos = random_tuned_lens.forward(randn, len(random_tuned_lens) - 1)
    assert th.allclose(logits_neg, logits_pos)


def test_tuned_lens_transform_hidden_negative_idx(random_tuned_lens: TunedLens):
    """transform_hidden() should accept negative layer indices."""
    randn = th.randn(1, 10, 128)
    h_neg = random_tuned_lens.transform_hidden(randn, -1)
    h_pos = random_tuned_lens.transform_hidden(randn, len(random_tuned_lens) - 1)
    assert th.allclose(h_neg, h_pos)


# --- Tests for forward_all ---


def test_logit_lens_forward_all(logit_lens):
    """forward_all() should return one logit tensor per layer."""
    hidden_states = [th.randn(1, 10, 128) for _ in range(3)]
    results = logit_lens.forward_all(hidden_states)
    assert len(results) == 3
    for r in results:
        assert r.shape == (1, 10, 100)


def test_tuned_lens_forward_all(random_tuned_lens: TunedLens):
    """forward_all() should return one logit tensor per layer."""
    hidden_states = [th.randn(1, 10, 128) for _ in range(3)]
    results = random_tuned_lens.forward_all(hidden_states)
    assert len(results) == 3


def test_forward_all_matches_sequential(random_tuned_lens: TunedLens):
    """forward_all() results should match sequential forward() calls."""
    hidden_states = [th.randn(1, 10, 128) for _ in range(3)]
    batch_results = random_tuned_lens.forward_all(hidden_states)
    for i, h in enumerate(hidden_states):
        single = random_tuned_lens.forward(h, i)
        assert th.allclose(batch_results[i], single)
