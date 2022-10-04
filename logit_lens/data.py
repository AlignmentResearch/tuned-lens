from datasets import Dataset, DatasetDict
from multiprocessing import cpu_count
from transformers import PreTrainedTokenizerBase
from typing import TypeVar, Union
import logging


T = TypeVar("T", bound=Union[Dataset, DatasetDict])


def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    text_key: str = "text",
) -> T:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_length` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        text_key: The key in the dataset to use as the text to tokenize.

    Returns:
        The chunked and tokenized dataset.
    """
    return data.map(
        lambda x: {
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            k: v[:-1]
            for k, v in tokenizer(
                # Concatenate all the samples together, separated by the EOS token.
                tokenizer.eos_token.join(x[text_key]),
                # The tokenizer will split this long text into 2048 token chunks.
                return_overflowing_tokens=True,
                truncation=True,
            ).items()
        },
        batched=True,
        num_proc=cpu_count() // 2,
        remove_columns=get_columns_all_equal(data),
    ).with_format(
        format,
        # Remove the "overflow_to_sample_mapping" column so we can directly pass
        # elements of the dataset to a model
        columns=["input_ids", "attention_mask"],
    )


def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> list[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names


def silence_datasets_messages():
    """Silence the very annoying wall of 'Loading cached processed dataset' messages."""
    handler = logging.StreamHandler()
    handler.addFilter(
        lambda log_record: not log_record.getMessage().startswith("Loading cached")
    )
    logging.getLogger("datasets").addHandler(handler)
