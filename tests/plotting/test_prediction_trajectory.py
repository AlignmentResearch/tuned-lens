import numpy as np
import pytest
import torch as th
import transformer_lens as tl
from transformers import AutoModelForCausalLM, AutoTokenizer

from tuned_lens.nn.lenses import LogitLens, TunedLens, Unembed
from tuned_lens.plotting import PredictionTrajectory
from tuned_lens.plotting.prediction_trajectory import _select_values_along_seq_axis


@pytest.fixture(params=[(), (2, 2), (1,)])
def prediction_trajectory(request):
    batch_shape = request.param
    layers = 3
    num_tokens = 10
    vocab_size = 12
    return PredictionTrajectory(
        log_probs=np.log(
            np.ones(batch_shape + (layers, num_tokens, vocab_size), dtype=np.float32)
            / vocab_size
        ),
        input_ids=np.ones(batch_shape + (num_tokens,), dtype=np.int64),
        targets=np.ones(batch_shape + (num_tokens,), dtype=np.int64),
    )


@pytest.fixture
def prediction_trajectory_with_tok(prediction_trajectory: PredictionTrajectory):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    prediction_trajectory.tokenizer = tokenizer
    return prediction_trajectory


@pytest.fixture
def hooked_transformer():
    return tl.HookedTransformer.from_pretrained(
        "EleutherAI/pythia-70m-deduped", device="cpu"
    )


@pytest.fixture
def model_and_tokenizer():
    model_name = "EleutherAI/pythia-70m-deduped"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


@pytest.fixture
def lens(model_and_tokenizer):
    model, _ = model_and_tokenizer
    return LogitLens.from_model(model)


def test_select_values():
    log_probs = np.array(
        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]]
    )

    targets = np.array([[1, 2]])

    result = _select_values_along_seq_axis(log_probs, targets)

    expected_result = np.array([[[0.2, 0.6], [0.8, 1.2]]])

    np.testing.assert_almost_equal(result, expected_result)

    assert result.shape == (1, 2, 2)


def test_prediction_trajectory_from_lens_and_model_smoke(model_and_tokenizer, lens):
    model, tokenizer = model_and_tokenizer
    input_ids = tokenizer.encode("Hello world!")
    traj = PredictionTrajectory.from_lens_and_model(
        lens, model, input_ids, tokenizer=tokenizer
    )
    assert traj.num_layers == model.config.num_hidden_layers
    assert traj.num_tokens == len(input_ids)
    assert traj.vocab_size == model.config.vocab_size


def test_prediction_trajectory_from_cache_no_batch(hooked_transformer):
    lens = TunedLens.from_unembed_and_pretrained(
        unembed=Unembed(hooked_transformer),
        lens_resource_id="EleutherAI/pythia-70m-deduped",
        map_location="cpu",
    )
    input_ids = th.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    targets = input_ids.clone()
    with th.inference_mode():
        logits, cache = hooked_transformer.run_with_cache(
            input=input_ids, return_type="logits"
        )
        assert isinstance(logits, th.Tensor)
        PredictionTrajectory.from_lens_and_cache(
            lens=lens,
            cache=cache,
            model_logits=logits,
            input_ids=input_ids,
            targets=targets,
        )


def test_get_sequence_labels_smoke(
    prediction_trajectory_with_tok: PredictionTrajectory,
):
    labels = prediction_trajectory_with_tok._get_sequence_labels()
    assert labels.shape == (10,)


def test_largest_prob_labels_smoke(
    prediction_trajectory_with_tok: PredictionTrajectory,
):
    labels = prediction_trajectory_with_tok._largest_prob_labels(min_prob=0.1, topk=5)
    assert labels is not None
    assert labels.label_strings.shape == (3, 10)
    assert labels.hover_over_entries is not None
    assert labels.hover_over_entries.shape[:-1] == (3, 10, 5)


def test_largest_delta_in_prob_labels_smoke(
    prediction_trajectory_with_tok: PredictionTrajectory,
):
    other = prediction_trajectory_with_tok
    labels = prediction_trajectory_with_tok._largest_delta_in_prob_labels(
        other, min_prob_delta=0.1, topk=5
    )
    assert labels is not None
    assert labels.label_strings.shape == (3, 10)
    assert labels.hover_over_entries is not None
    assert labels.hover_over_entries.shape[:-1] == (3, 10, 5)


def test_cross_entropy_smoke(prediction_trajectory: PredictionTrajectory):
    traj = prediction_trajectory
    ce_stat = traj.cross_entropy()

    assert ce_stat.name == "Cross Entropy"
    assert ce_stat.units == "nats"
    assert ce_stat.trajectory_labels is None
    assert ce_stat.sequence_labels is None
    assert ce_stat.stats.shape == (3, 10)


def test_rank_smoke(prediction_trajectory: PredictionTrajectory):
    traj = prediction_trajectory
    rank_stat = traj.rank(show_ranks=True)

    assert rank_stat.name == "Rank"
    assert rank_stat.units == ""
    assert rank_stat.trajectory_labels is None
    assert rank_stat.sequence_labels is None
    assert rank_stat.stats.shape == (3, 10)


def test_rank_correctness():
    # Test that the rank is correct.
    log_probs = np.array(
        [[[[0.1, 0.2, 0.3], [0.6, 0.5, 0.4]], [[0.85, 0.8, 0.9], [1.0, 1.1, 1.2]]]]
    )
    assert log_probs.shape == (1, 2, 2, 3)  # (batch, layer, seq, vocab)

    traj = PredictionTrajectory(
        log_probs=log_probs,
        input_ids=np.ones((1, 2), dtype=np.int64),
        targets=np.array([[0, 1]], dtype=np.int64),
    )

    rank_stat = traj.rank(show_ranks=True)

    assert rank_stat.stats.shape == (2, 2)
    assert rank_stat.stats[0, 0] == 2
    assert rank_stat.stats[0, 1] == 1
    assert rank_stat.stats[1, 0] == 1
    assert rank_stat.stats[1, 1] == 1


def test_entropy_smoke(prediction_trajectory: PredictionTrajectory):
    traj = prediction_trajectory
    entropy_stat = traj.entropy()

    assert entropy_stat.name == "Entropy"
    assert entropy_stat.units == "nats"
    assert entropy_stat.trajectory_labels is None
    assert entropy_stat.sequence_labels is None
    assert entropy_stat.stats.shape == (3, 10)


def test_forward_kl_smoke(prediction_trajectory: PredictionTrajectory):
    traj = prediction_trajectory
    forward_kl_stat = traj.forward_kl()

    assert forward_kl_stat.name == "Forward KL"
    assert forward_kl_stat.units == "nats"
    assert forward_kl_stat.sequence_labels is None
    assert forward_kl_stat.trajectory_labels is None
    assert forward_kl_stat.stats.shape == (3, 10)


def test_max_probability_smoke(prediction_trajectory: PredictionTrajectory):
    traj = prediction_trajectory
    max_probability_stat = traj.max_probability()

    assert max_probability_stat.name == "Max Probability"
    assert max_probability_stat.units == "probs"
    assert max_probability_stat.sequence_labels is None
    assert max_probability_stat.trajectory_labels is None
    assert max_probability_stat.stats.shape == (3, 10)


def test_kl_divergence_smoke(
    prediction_trajectory: PredictionTrajectory,
):
    traj = prediction_trajectory
    other = prediction_trajectory

    kl_stat = traj.kl_divergence(other)
    assert kl_stat.name == "KL(Self | Other)"
    assert kl_stat.units == "nats"
    assert kl_stat.sequence_labels is None
    assert kl_stat.trajectory_labels is None
    assert kl_stat.stats.shape == (3, 10)

    assert np.isclose(kl_stat.stats, 0.0).all()


def test_js_divergence_smoke(
    prediction_trajectory: PredictionTrajectory,
):
    traj = prediction_trajectory
    other = prediction_trajectory

    js_stat = traj.js_divergence(other)
    assert js_stat.name == "JS(Self | Other)"
    assert js_stat.units == "nats"
    assert js_stat.sequence_labels is None
    assert js_stat.trajectory_labels is None
    assert js_stat.stats.shape == (3, 10)

    assert np.isclose(js_stat.stats, 0.0).all()


def test_total_variation_smoke(
    prediction_trajectory: PredictionTrajectory,
):
    traj = prediction_trajectory
    other = prediction_trajectory

    js_stat = traj.total_variation(other)
    assert js_stat.name == "TV(Self | Other)"
    assert js_stat.units == "probs"
    assert js_stat.sequence_labels is None
    assert js_stat.trajectory_labels is None
    assert js_stat.stats.shape == (3, 10)

    assert np.isclose(js_stat.stats, 0.0).all()
