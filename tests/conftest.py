import os
import pytest
from datasets import Dataset
import torch as th
import transformers as tr


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
        "gpt2",
    ],
)
def small_model_name(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def random_small_model(small_model_name: str):
    th.manual_seed(42)

    # We use a random model with the correct config instead of downloading the
    # whole pretrained checkpoint.
    config = tr.AutoConfig.from_pretrained(small_model_name)
    model = tr.AutoModelForCausalLM.from_config(config)
    return model


@pytest.fixture(scope="module")
def small_model_tokenizer(small_model_name: str) -> tr.PreTrainedTokenizerBase:
    return tr.AutoTokenizer.from_pretrained(small_model_name, use_fast=True)


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    return tr.AutoTokenizer.from_pretrained("gpt2", use_fast=True)
