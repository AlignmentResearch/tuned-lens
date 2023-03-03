import json

import zstandard as zstd
import datasets

"""The Pile silver dataset. Based on the pile's validation and test sets.
See original dataset here: https://huggingface.co/datasets/the_pile
"""

_HOST_URL = "https://the-eye.eu"


class PileSliver(datasets.GeneratorBasedBuilder):
    """The Pile sliver dataset. Based on the pile's validation and test sets."""

    def _info(self):
        return datasets.DatasetInfo(
            description=("The Pile silver dataset."
                         " Based on the pile's validation and test sets."),
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "meta": {"pile_set_name": datasets.Value("string")},
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager) -> list[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'files': dl_manager.download_and_extract(
                    f"{_HOST_URL}/public/AI/pile/val.jsonl.zst"
                )},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={'files': dl_manager.download_and_extract(
                    f"{_HOST_URL}/public/AI/pile/test.jsonl.zst"
                )},
            )
        ]

    def _generate_examples(self, files):
        """Yield examples as (key, example) tuples."""
        key = 0
        if isinstance(files, str):
            files = [files]

        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    yield key, data
                    key += 1
