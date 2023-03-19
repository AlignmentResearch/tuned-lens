"""Tools for finding and modifying components in a transformer model."""

from contextlib import contextmanager
from transformers import PreTrainedModel
from typing import Any, Generator, Optional, Type, TypeVar, Union
import torch as th


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


def get_final_layer_norm(
    model: PreTrainedModel, norm_class: Type[th.nn.Module] = th.nn.LayerNorm
) -> Optional[th.nn.Module]:
    """Use heuristics to find the final layer norm in a model, if it exists."""
    base = model.base_model
    if decoder := getattr(base, "decoder", None):
        base = decoder

    top_level_lns = [
        module for module in base.children() if isinstance(module, norm_class)
    ]
    return top_level_lns[-1] if top_level_lns else None


def get_transformer_layers(model: th.nn.Module) -> tuple[str, th.nn.ModuleList]:
    """Get "the" list of transformer layers from a model.

    This is operationalized as the unique `nn.ModuleList` that contains
    more than half of all the parameters in the model, if it exists.

    Args:
        model: The model to search.

    Returns:
        A tuple containing the key path to the layer list and the list itself.

    Raises:
        ValueError: If no such list exists.
    """
    total_params = sum(p.numel() for p in model.parameters())
    for name, module in model.named_modules():
        if isinstance(module, th.nn.ModuleList):
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > total_params / 2:
                return name, module

    raise ValueError(
        "Could not find suitable `ModuleList`; is this an encoder-decoder model?"
    )


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
