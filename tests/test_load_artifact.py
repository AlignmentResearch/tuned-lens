import pytest

from tuned_lens.load_artifacts import available_lens_artifacts, load_lens_artifacts


def test_load_lens_artifact_smoke():
    load_lens_artifacts("gpt2", "AlignmentResearch/tuned-lens")


def test_load_lens_artifact_raises_smoke():
    with pytest.raises(ValueError, match="Could not find lens at the specified"):
        load_lens_artifacts("sia23s3asdr", "AlignmentResearch/tuned-lens")


def test_list_available_lens_artifacts_smoke():
    artifacts = available_lens_artifacts("AlignmentResearch/tuned-lens", "space")
    assert len(artifacts) > 0
    artifacts = available_lens_artifacts(
        "AlignmentResearch/tuned-lens",
        "space",
        "revision_does_not_exist",
    )
    assert len(artifacts) == 0
