import pytest

from tuned_lens.nn.lenses import LogitLens
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.plotting import PredictionTrajectory
import numpy as np


@pytest.fixture
def prediction_trajectory_no_tokenizer():
    layers = 3
    num_tokens = 10
    vocab_size = 12
    return PredictionTrajectory(
        log_probs=np.zeros((layers, num_tokens, vocab_size), dtype=np.float32),
        input_ids=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        targets=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    )


@pytest.fixture
def prediction_other__trajectory_no_tokenizer():
    layers = 3
    num_tokens = 10
    vocab_size = 12
    return PredictionTrajectory(
        log_probs=np.zeros((layers, num_tokens, vocab_size), dtype=np.float32),
        input_ids=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        targets=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    )


@pytest.fixture
def prediction_trajectory_with_tokenizer():
    layers = 3
    num_tokens = 10
    vocab_size = 12
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return PredictionTrajectory(
        log_probs=np.zeros((layers, num_tokens, vocab_size), dtype=np.float32),
        input_ids=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        targets=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        tokenizer=tokenizer,
    )


@pytest.fixture
def model_and_tokenizer():
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


@pytest.fixture
def lens(model_and_tokenizer):
    model, _ = model_and_tokenizer
    return LogitLens(model)


def test_prediction_trajectory_from_lens_and_model_smoke(model_and_tokenizer, lens):
    model, tokenizer = model_and_tokenizer
    input_ids = tokenizer.encode("Hello world!")
    traj = PredictionTrajectory.from_lens_and_model(
        lens, model, input_ids, tokenizer=tokenizer
    )
    assert traj.num_layers == model.config.n_layer
    assert traj.num_tokens == len(input_ids)
    assert traj.vocab_size == model.config.vocab_size


def test_largest_prob_labels_smoke(prediction_trajectory_with_tokenizer):
    labels = prediction_trajectory_with_tokenizer.largest_prob_labels(
        min_prob=0.1, topk=5
    )
    assert labels.label_strings.shape == (3, 10)
    assert labels.sequence_labels.shape == (10,)
    assert labels.hover_over_entries.shape == (3, 10, 5)


def test_largest_delta_in_prob_labels_smoke(prediction_trajectory_with_tokenizer):
    layers = 3
    num_tokens = 10
    vocab_size = 12
    other = PredictionTrajectory(
        log_probs=np.zeros((layers, num_tokens, vocab_size), dtype=np.float32),
        input_ids=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        targets=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        tokenizer=prediction_trajectory_with_tokenizer.tokenizer,
    )
    labels = prediction_trajectory_with_tokenizer.largest_delta_in_prob_labels(
        other, min_prob_delta=0.1, topk=5
    )
    assert labels.label_strings.shape == (3, 10)
    assert labels.sequence_labels.shape == (10,)
    assert labels.hover_over_entries.shape == (3, 10, 5)


def test_cross_entropy_smoke(prediction_trajectory_no_tokenizer):
    traj = prediction_trajectory_no_tokenizer
    ce_stat = traj.cross_entropy()

    assert ce_stat.name == "Cross Entropy"
    assert ce_stat.units == "nats"
    assert ce_stat.labels is None
    assert ce_stat.stats.shape == (3, 10)


def test_entropy_smoke(prediction_trajectory_no_tokenizer):
    traj = prediction_trajectory_no_tokenizer
    entropy_stat = traj.entropy()

    assert entropy_stat.name == "Entropy"
    assert entropy_stat.units == "nats"
    assert entropy_stat.labels is None
    assert entropy_stat.stats.shape == (3, 10)


def test_forward_kl_smoke(prediction_trajectory_no_tokenizer):
    traj = prediction_trajectory_no_tokenizer
    forward_kl_stat = traj.forward_kl()

    assert forward_kl_stat.name == "Forward KL"
    assert forward_kl_stat.units == "nats"
    assert forward_kl_stat.labels is None
    assert forward_kl_stat.stats.shape == (3, 10)


def test_max_probability_smoke(prediction_trajectory_no_tokenizer):
    traj = prediction_trajectory_no_tokenizer
    max_probability_stat = traj.max_probability()

    assert max_probability_stat.name == "Max Probability"
    assert max_probability_stat.units == "probs"
    assert max_probability_stat.labels is None
    assert max_probability_stat.stats.shape == (3, 10)


def test_kl_divergence_smoke(
    prediction_trajectory_no_tokenizer, prediction_other__trajectory_no_tokenizer
):
    traj = prediction_trajectory_no_tokenizer
    other = prediction_other__trajectory_no_tokenizer

    kl_stat = traj.kl_divergence(other)
    assert kl_stat.name == "KL(Self | Other)"
    assert kl_stat.units == "nats"
    assert kl_stat.labels is None
    assert kl_stat.stats.shape == (3, 10)

    assert np.isclose(kl_stat.stats, 0.0).all()


def test_js_divergence_smoke(
    prediction_trajectory_no_tokenizer, prediction_other__trajectory_no_tokenizer
):
    traj = prediction_trajectory_no_tokenizer
    other = prediction_other__trajectory_no_tokenizer

    js_stat = traj.js_divergence(other)
    assert js_stat.name == "JS(Self | Other)"
    assert js_stat.units == "nats"
    assert js_stat.labels is None
    assert js_stat.stats.shape == (3, 10)

    assert np.isclose(js_stat.stats, 0.0).all()


def test_total_variation_smoke(
    prediction_trajectory_no_tokenizer, prediction_other__trajectory_no_tokenizer
):
    traj = prediction_trajectory_no_tokenizer
    other = prediction_other__trajectory_no_tokenizer

    js_stat = traj.total_variation(other)
    assert js_stat.name == "TV(Self | Other)"
    assert js_stat.units == "probs"
    assert js_stat.labels is None
    assert js_stat.stats.shape == (3, 10)

    assert np.isclose(js_stat.stats, 0.0).all()


def test_calc_first_order_diff_smoke(prediction_trajectory_with_tokenizer):
    # WIP
    traj = prediction_trajectory_with_tokenizer
    traj.log_probs[1][0][0] = 0.1
    traj.log_probs[2][0][0] = 0.5
    token_deltas = traj.get_first_order_diff(3)
    assert token_deltas[0][0][0][1] == 0.1
    assert token_deltas[0][1][0][1] == 0.4
