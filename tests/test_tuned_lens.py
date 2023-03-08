from tuned_lens.nn.tuned_lens import TunedLens
import tempfile
import torch as th
import pytest

@pytest.fixture
def tuned_lens():
    return TunedLens(d_model=128, num_layers=3, vocab_size=100)

def test_tuned_lens_smoke(tuned_lens):
    randn = th.randn(1, 10, 128)
    logits_0 = tuned_lens(randn, 0)

def test_tuned_lens_save_and_load(tuned_lens):
    randn = th.randn(1, 10, 128)

    logits_before = tuned_lens(randn, 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        tuned_lens.save(tmpdir)
        tuned_lens = TunedLens.load(tmpdir)
        logits_after = tuned_lens(randn, 1)
        assert th.allclose(logits_before, logits_after)
