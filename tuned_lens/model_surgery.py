"""Tools for finding and modifying components in a transformer model."""

from contextlib import contextmanager
from typing import Any, Generator, TypeVar, Union

try:
    import transformer_lens as tl

    _transformer_lens_available = True
except ImportError:
    _transformer_lens_available = False

import torch as th
import transformers as tr
from torch import nn
from transformers import models


def get_value_for_key(obj: Any, key: str) -> Any:
    """Get a value using `__getitem__` if `key` is numeric and `getattr` otherwise."""
    return obj[int(key)] if key.isdigit() else getattr(obj, key)


def set_value_for_key_(obj: Any, key: str, value: Any) -> None:
    """Set value in-place if `key` is numeric and `getattr` otherwise."""
    if key.isdigit():
        obj[int(key)] = value
    else:
        setattr(obj, key, value)


def get_key_path(model: th.nn.Module, key_path: str) -> Any:
    """Get a value by key path, e.g. `layers.0.attention.query.weight`."""
    for key in key_path.split("."):
        model = get_value_for_key(model, key)

    return model


def set_key_path_(
    model: th.nn.Module, key_path: str, value: Union[th.nn.Module, th.Tensor]
) -> None:
    """Set a value by key path in-place, e.g. `layers.0.attention.query.weight`."""
    keys = key_path.split(".")
    for key in keys[:-1]:
        model = get_value_for_key(model, key)

    setattr(model, keys[-1], value)


T = TypeVar("T", bound=th.nn.Module)


@contextmanager
def assign_key_path(model: T, key_path: str, value: Any) -> Generator[T, None, None]:
    """Temporarily set a value by key path while in the context."""
    old_value = get_key_path(model, key_path)
    set_key_path_(model, key_path, value)
    try:
        yield model
    finally:
        set_key_path_(model, key_path, old_value)


Model = Union[tr.PreTrainedModel, "tl.HookedTransformer"]
Norm = Union[th.nn.LayerNorm, models.llama.modeling_llama.LlamaRMSNorm, nn.Module]


def get_unembedding_matrix(model: Model) -> nn.Linear:
    """The final linear tranformation from the model hidden state to the output."""
    if isinstance(model, tr.PreTrainedModel):
        unembed = model.get_output_embeddings()
        if not isinstance(unembed, nn.Linear):
            raise ValueError("We currently only support linear unemebdings")
        return unembed
    elif _transformer_lens_available and isinstance(model, tl.HookedTransformer):
        linear = nn.Linear(
            in_features=model.cfg.d_model,
            out_features=model.cfg.d_vocab_out,
        )
        linear.bias.data = model.unembed.b_U
        linear.weight.data = model.unembed.W_U.transpose(0, 1)
        return linear
    else:
        raise ValueError(f"Model class {type(model)} not recognized!")


def get_final_norm(model: Model) -> Norm:
    """Get the final norm from a model.

    This isn't standardized across models, so this will need to be updated as
    we add new models.
    """
    if _transformer_lens_available and isinstance(model, tl.HookedTransformer):
        return model.ln_final

    if not hasattr(model, "base_model"):
        raise ValueError("Model does not have a `base_model` attribute.")

    base_model = model.base_model
    if isinstance(base_model, models.opt.modeling_opt.OPTModel):
        final_layer_norm = base_model.decoder.final_layer_norm
    elif isinstance(base_model, models.gpt_neox.modeling_gpt_neox.GPTNeoXModel):
        final_layer_norm = base_model.final_layer_norm
    elif isinstance(
        base_model,
        (
            models.bloom.modeling_bloom.BloomModel,
            models.gpt2.modeling_gpt2.GPT2Model,
            models.gpt_neo.modeling_gpt_neo.GPTNeoModel,
            models.gptj.modeling_gptj.GPTJModel,
        ),
    ):
        final_layer_norm = base_model.ln_f
    elif isinstance(base_model, models.llama.modeling_llama.LlamaModel):
        final_layer_norm = base_model.norm
    else:
        raise NotImplementedError(f"Unknown model type {type(base_model)}")

    if final_layer_norm is None:
        raise ValueError("Model does not have a final layer norm.")

    assert isinstance(final_layer_norm, Norm.__args__)  # type: ignore

    return final_layer_norm


def get_transformer_layers(model: Model) -> tuple[str, th.nn.ModuleList]:
    """Get the decoder layers from a model.

    Args:
        model: The model to search.

    Returns:
        A tuple containing the key path to the layer list and the list itself.

    Raises:
        ValueError: If no such list exists.
    """
    # TODO implement this so that we can do hooked transformer training.
    if not hasattr(model, "base_model"):
        raise ValueError("Model does not have a `base_model` attribute.")

    path_to_layers = ["base_model"]
    base_model = model.base_model
    if isinstance(base_model, models.opt.modeling_opt.OPTModel):
        path_to_layers += ["decoder", "layers"]
    elif isinstance(base_model, models.gpt_neox.modeling_gpt_neox.GPTNeoXModel):
        path_to_layers += ["layers"]
    elif isinstance(
        base_model,
        (
            models.bloom.modeling_bloom.BloomModel,
            models.gpt2.modeling_gpt2.GPT2Model,
            models.gpt_neo.modeling_gpt_neo.GPTNeoModel,
            models.gptj.modeling_gptj.GPTJModel,
        ),
    ):
        path_to_layers += ["h"]
    elif isinstance(base_model, models.llama.modeling_llama.LlamaModel):
        path_to_layers += ["layers"]
    else:
        raise NotImplementedError(f"Unknown model type {type(base_model)}")

    path_to_layers = ".".join(path_to_layers)
    return path_to_layers, get_key_path(model, path_to_layers)


@contextmanager
def delete_layers(model: T, indices: list[int]) -> Generator[T, None, None]:
    """Temporarily delete the layers at `indices` from `model` while in the context."""
    list_path, layer_list = get_transformer_layers(model)
    modified_list = th.nn.ModuleList(layer_list)
    for i in sorted(indices, reverse=True):
        del modified_list[i]

    set_key_path_(model, list_path, modified_list)
    try:
        yield model
    finally:
        set_key_path_(model, list_path, layer_list)


@contextmanager
def permute_layers(model: T, indices: list[int]) -> Generator[T, None, None]:
    """Temporarily permute the layers of `model` by `indices` while in the context.

    The number of indices provided may be not be equal to the number of
    layers in the model. Layers will be dropped or duplicated accordingly.
    """
    list_path, layer_list = get_transformer_layers(model)
    permuted_list = th.nn.ModuleList([layer_list[i] for i in indices])
    set_key_path_(model, list_path, permuted_list)

    try:
        yield model
    finally:
        set_key_path_(model, list_path, layer_list)


def permute_layers_(model: th.nn.Module, indices: list[int]):
    """Permute the layers of `model` by `indices` in-place.

    The number of indices provided may be not be equal to the number of
    layers in the model. Layers will be dropped or duplicated accordingly.
    """
    list_path, layer_list = get_transformer_layers(model)
    permuted_list = th.nn.ModuleList([layer_list[i] for i in indices])
    set_key_path_(model, list_path, permuted_list)


@contextmanager
def replace_layers(
    model: T, indices: list[int], replacements: list[th.nn.Module]
) -> Generator[T, None, None]:
    """Replace the layers at `indices` with `replacements` while in the context."""
    list_path, layer_list = get_transformer_layers(model)
    modified_list = th.nn.ModuleList(layer_list)
    for i, replacement in zip(indices, replacements):
        modified_list[i] = replacement

    set_key_path_(model, list_path, modified_list)
    try:
        yield model
    finally:
        set_key_path_(model, list_path, layer_list)
