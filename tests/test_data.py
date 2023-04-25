from tuned_lens import data
from datasets import Dataset
import transformers as tr
import math


def test_chunk_and_tokenize(
    text_dataset: Dataset, small_model_tokenizer: tr.PreTrainedTokenizerBase
):
    max_length = 128
    chunked, _ = data.chunk_and_tokenize(
        text_dataset,
        small_model_tokenizer,
        load_from_cache_file=False,
        max_length=max_length,
    )

    length = min(small_model_tokenizer.model_max_length, max_length)
    for i in range(len(chunked)):
        assert len(chunked[i]["input_ids"]) == length


def test_compute_nats_to_bpb_ratio(
    text_dataset: Dataset, gpt2_tokenizer: tr.PreTrainedTokenizerBase
):
    max_length = 128
    _, ratio = data.chunk_and_tokenize(
        text_dataset, gpt2_tokenizer, load_from_cache_file=True, max_length=max_length
    )
    # We expect the ratio to be around 0.29, see https://arxiv.org/pdf/2101.00027.pdf,
    # section 3.1
    assert 0.2 / math.log(2) < ratio < 0.4 / math.log(2)
