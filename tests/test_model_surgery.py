import pytest
import torch as th
from transformers import PreTrainedModel

from tuned_lens import model_surgery


def test_get_final_layer_norm_raises(opt_random_model: PreTrainedModel):
    opt_random_model.base_model.decoder.final_layer_norm = None
    with pytest.raises(ValueError):
        assert model_surgery.get_final_norm(opt_random_model)


def test_get_final_layer_norm(random_small_model: PreTrainedModel):
    ln = model_surgery.get_final_norm(random_small_model)
    assert any(isinstance(ln, Norm) for Norm in model_surgery.Norm.__args__)


def test_get_layers_from_model(random_small_model: PreTrainedModel):
    path, layers = model_surgery.get_transformer_layers(random_small_model)
    assert isinstance(layers, th.nn.ModuleList)
    assert isinstance(path, str)
    assert len(layers) == random_small_model.config.num_hidden_layers


# --- Tests for attribute-based detection ---


def test_attribute_norm_detection(random_small_model: PreTrainedModel):
    """Attribute-based probing should find the same norm as isinstance checks."""
    norm = model_surgery._try_attribute_norm(random_small_model.base_model)
    assert norm is not None
    assert isinstance(norm, th.nn.Module)


def test_attribute_layer_detection(random_small_model: PreTrainedModel):
    """Attribute-based probing should find the same layers as isinstance checks."""
    result = model_surgery._try_attribute_layers(random_small_model.base_model)
    assert result is not None
    path_components, layers = result
    assert isinstance(layers, th.nn.ModuleList)
    assert len(layers) == random_small_model.config.num_hidden_layers


def test_get_base_model_missing():
    """A model without base_model should raise a helpful error."""
    fake_model = th.nn.Linear(10, 10)
    with pytest.raises(ValueError, match="does not have a `base_model`"):
        model_surgery._get_base_model(fake_model)


def test_error_message_includes_attributes():
    """Error messages should list available model attributes."""
    fake_model = th.nn.Linear(10, 10)
    with pytest.raises(ValueError, match="Available attributes"):
        model_surgery._get_base_model(fake_model)
