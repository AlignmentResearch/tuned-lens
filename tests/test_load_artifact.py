from tuned_lens.load_artifacts import load_lens_artifacts


def test_load_lens_artifact_smoke():
    load_lens_artifacts("gpt2", "AlignmentResearch/tuned-lens")
