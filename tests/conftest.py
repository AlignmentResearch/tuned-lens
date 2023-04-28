import os
import pytest
from datasets import Dataset
import torch as th
import transformers as tr
from transformers.testing_utils import get_tests_dir

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@pytest.fixture(scope="module")
def text_dataset() -> Dataset:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset = Dataset.from_json(os.path.join(dir_path, "test_data/pile_text.jsonl"))
    assert isinstance(dataset, Dataset)
    return dataset


@pytest.fixture(
    scope="module",
    params=[
        "EleutherAI/pythia-70m-deduped",
        "bigscience/bloom-560m",
        "EleutherAI/gpt-neo-125M",
        "facebook/opt-125m",
        "mockmodel/llamma-tiny",
        "gpt2",
    ],
)
def random_small_model(request: str) -> tr.PreTrainedModel:
    small_model_name = request.param
    th.manual_seed(42)

    # We use a random model with the correct config instead of downloading the
    # whole pretrained checkpoint.
    if small_model_name == "mockmodel/llamma-tiny":
        config = tr.LlamaConfig(
            vocab_size=32_000,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
        )
    else:
        config = tr.AutoConfig.from_pretrained(small_model_name)

    model = tr.AutoModelForCausalLM.from_config(config)
    model.eval()

    return model


@pytest.fixture(
    scope="module",
    params=[
        "EleutherAI/pythia-70m-deduped",
        "bigscience/bloom-560m",
        "EleutherAI/gpt-neo-125M",
        "facebook/opt-125m",
        "gpt2",
    ],
)
def small_model_tokenizer(request: str) -> tr.PreTrainedTokenizerBase:
    return tr.AutoTokenizer.from_pretrained(request.param, use_fast=True)


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    return tr.AutoTokenizer.from_pretrained("gpt2", use_fast=True)


@pytest.fixture(scope="module")
def opt_random_model() -> tr.PreTrainedModel:
    config = tr.AutoConfig.from_pretrained("facebook/opt-125m")
    model = tr.AutoModelForCausalLM.from_config(config)
    model.eval()
    return model
