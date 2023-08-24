import pytest
import torch as th
from transformers import PreTrainedModel, models

from tuned_lens import model_surgery


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


def test_add_steering_vector(random_small_model: PreTrainedModel):
    # First run the model on some data without the steering vector.
    model = random_small_model
    input_ids = th.ones((1, 4), dtype=th.long)
    generator = th.default_generator.manual_seed(0)
    steering_vector = th.randn(
        (model.config.hidden_size), dtype=th.float, generator=generator
    )
    y1 = random_small_model(input_ids).logits

    # Now add the steering vector.
    with model_surgery.add_steering_vector(model, [1, 2], steering_vector):
        y2 = model(input_ids=input_ids).logits

    # Check that the output changed.
    assert not th.allclose(y1, y2)
