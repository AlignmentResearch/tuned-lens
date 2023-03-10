import os
from huggingface_hub import login
from tuned_lens.load_artifacts import load_lens_artifact


def test_load_lens_artifact_smoke():
    assert os.environ["HUGGINGFACE_TOKEN"], \
        "Please set the HUGGINGFACE_TOKEN environment variable."
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    load_lens_artifact("gpt2", "AlignmentResearch/tuned-lens")
