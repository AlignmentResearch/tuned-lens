import pytest
import torch as th
from tuned_lens import model_surgery
from transformers import PreTrainedModel, models


def test_get_final_layer_norm_raises(opt_random_model: PreTrainedModel):
    opt_random_model.base_model.decoder.final_layer_norm = None
    with pytest.raises(ValueError):
        assert model_surgery.get_final_norm(opt_random_model)


def test_get_final_layer_norm(random_small_model: PreTrainedModel):
    ln = model_surgery.get_final_norm(random_small_model)
    assert isinstance(ln, (th.nn.LayerNorm, models.llama.modeling_llama.LlamaRMSNorm))


def test_get_layers_from_model(random_small_model: PreTrainedModel):
    path, layers = model_surgery.get_transformer_layers(random_small_model)
    assert isinstance(layers, th.nn.ModuleList)
    assert isinstance(path, str)
    assert len(layers) == random_small_model.config.num_hidden_layers
