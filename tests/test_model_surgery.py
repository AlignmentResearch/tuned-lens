import pytest
import torch as th
from transformers import AutoModelForCausalLM
from tuned_lens import model_surgery


@pytest.fixture(
    params=[
        "EleutherAI/pythia-70m-deduped",
        "bigscience/bloom-560m",
        "EleutherAI/gpt-neo-125M",
        "facebook/opt-125m",
        "gpt2",
    ]
)
def small_model(request):
    return AutoModelForCausalLM.from_pretrained(request.param)


@pytest.fixture
def opt():
    return AutoModelForCausalLM.from_pretrained("facebook/opt-125m")


def test_get_final_layer_norm(small_model):
    ln = model_surgery.get_final_layer_norm(small_model)
    assert isinstance(ln, th.nn.LayerNorm)


def test_get_final_layer_norm_raises(opt):
    opt.base_model.decoder.final_layer_norm = None
    with pytest.raises(ValueError):
        assert model_surgery.get_final_layer_norm(opt)


def test_get_layers_from_model(small_model):
    path, model = model_surgery.get_transformer_layers(small_model)
    assert isinstance(model, th.nn.ModuleList)
    assert isinstance(path, str)
    assert len(model) == small_model.config.num_hidden_layers
