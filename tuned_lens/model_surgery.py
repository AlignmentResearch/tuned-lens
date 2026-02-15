"""Tools for finding and modifying components in a transformer model."""

from contextlib import contextmanager
from typing import Any, Generator, Optional, TypeVar, Union

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
Norm = Union[
    th.nn.LayerNorm,
    models.llama.modeling_llama.LlamaRMSNorm,
    models.gemma.modeling_gemma.GemmaRMSNorm,
    nn.Module,
]

# Ordered list of (attribute_name, model_classes) for final norm detection.
# Attribute-based lookup is tried first; isinstance is the fallback.
_NORM_PATHS = [
    ("norm", (
        models.llama.modeling_llama.LlamaModel,
        models.mistral.modeling_mistral.MistralModel,
        models.gemma.modeling_gemma.GemmaModel,
    )),
    ("ln_f", (
        models.bloom.modeling_bloom.BloomModel,
        models.gpt2.modeling_gpt2.GPT2Model,
        models.gpt_neo.modeling_gpt_neo.GPTNeoModel,
        models.gptj.modeling_gptj.GPTJModel,
    )),
    ("final_layer_norm", (
        models.gpt_neox.modeling_gpt_neox.GPTNeoXModel,
    )),
]

# Ordered list of (attribute_name, model_classes) for layer detection.
_LAYER_PATHS = [
    ("layers", (
        models.llama.modeling_llama.LlamaModel,
        models.mistral.modeling_mistral.MistralModel,
        models.gemma.modeling_gemma.GemmaModel,
        models.gpt_neox.modeling_gpt_neox.GPTNeoXModel,
    )),
    ("h", (
        models.bloom.modeling_bloom.BloomModel,
        models.gpt2.modeling_gpt2.GPT2Model,
        models.gpt_neo.modeling_gpt_neo.GPTNeoModel,
        models.gptj.modeling_gptj.GPTJModel,
    )),
]


def _get_base_model(model: Model) -> th.nn.Module:
    """Get the base model, raising a helpful error if not found.

    Args:
        model: A pretrained model or HookedTransformer.

    Returns:
        The base model module.

    Raises:
        ValueError: If the model has no ``base_model`` attribute.
    """
    if not hasattr(model, "base_model"):
        available = [a for a in dir(model) if not a.startswith("_")]
        raise ValueError(
            f"Model {type(model).__name__} does not have a `base_model` attribute. "
            f"Available attributes: {available[:15]}. "
            f"If this is a custom model, please open an issue at: "
            f"https://github.com/AlignmentResearch/tuned-lens/issues"
        )
    return model.base_model


def _try_attribute_norm(base_model: th.nn.Module) -> Optional[nn.Module]:
    """Try to find the final norm via attribute-based probing.

    Checks common attribute names on the base model and its ``decoder``
    sub-module (for OPT-style architectures). Returns the norm module if
    found and it is an instance of ``nn.Module``, otherwise ``None``.

    Args:
        base_model: The unwrapped base model to probe.

    Returns:
        The final norm module, or ``None`` if not found.
    """
    # Direct attributes on base_model (covers Llama, Mistral, Gemma, GPT-2, etc.)
    for attr in ("norm", "ln_f", "final_layer_norm"):
        norm = getattr(base_model, attr, None)
        if norm is not None and isinstance(norm, nn.Module):
            return norm

    # OPT-style: base_model.decoder.final_layer_norm
    decoder = getattr(base_model, "decoder", None)
    if decoder is not None:
        norm = getattr(decoder, "final_layer_norm", None)
        if norm is not None and isinstance(norm, nn.Module):
            return norm

    return None


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

    Uses attribute-based probing to detect the final normalization layer,
    which makes this function forward-compatible with new architectures that
    follow standard naming conventions. Falls back to ``isinstance`` checks
    for known architectures.

    Args:
        model: A pretrained model or HookedTransformer.

    Returns:
        The final normalization module.

    Raises:
        ValueError: If the model has no ``base_model`` or the norm is ``None``.
        NotImplementedError: If the architecture is not recognized.
    """
    if _transformer_lens_available and isinstance(model, tl.HookedTransformer):
        return model.ln_final

    base_model = _get_base_model(model)

    # Try attribute-based detection first (handles new architectures automatically)
    final_layer_norm = _try_attribute_norm(base_model)

    # Fall back to isinstance checks for known architectures
    if final_layer_norm is None:
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
        elif isinstance(base_model, models.mistral.modeling_mistral.MistralModel):
            final_layer_norm = base_model.norm
        elif isinstance(base_model, models.gemma.modeling_gemma.GemmaModel):
            final_layer_norm = base_model.norm
        else:
            available = [a for a in dir(base_model) if not a.startswith("_")]
            raise NotImplementedError(
                f"Unsupported model architecture: {type(base_model).__name__}. "
                f"Could not auto-detect a final layer norm via attribute probing. "
                f"Available attributes: {available[:15]}. "
                f"Please open an issue at: "
                f"https://github.com/AlignmentResearch/tuned-lens/issues"
            )

    if final_layer_norm is None:
        raise ValueError("Model does not have a final layer norm.")

    assert isinstance(final_layer_norm, Norm.__args__)  # type: ignore

    return final_layer_norm


def _try_attribute_layers(
    base_model: th.nn.Module,
) -> Optional[tuple[list[str], th.nn.ModuleList]]:
    """Try to find transformer layers via attribute-based probing.

    Checks common attribute names on the base model and its ``decoder``
    sub-module (for OPT-style architectures). Returns the path components
    and the ``ModuleList`` if found.

    Args:
        base_model: The unwrapped base model to probe.

    Returns:
        A tuple of ``(path_components, module_list)`` or ``None``.
    """
    # Direct attributes on base_model (covers Llama, Mistral, Gemma, GPT-2, etc.)
    for attr in ("layers", "h"):
        layers = getattr(base_model, attr, None)
        if isinstance(layers, th.nn.ModuleList):
            return [attr], layers

    # OPT-style: base_model.decoder.layers
    decoder = getattr(base_model, "decoder", None)
    if decoder is not None:
        layers = getattr(decoder, "layers", None)
        if isinstance(layers, th.nn.ModuleList):
            return ["decoder", "layers"], layers

    return None


def get_transformer_layers(model: Model) -> tuple[str, th.nn.ModuleList]:
    """Get the decoder layers from a model.

    Uses attribute-based probing to detect transformer layers, which makes
    this function forward-compatible with new architectures. Falls back to
    ``isinstance`` checks for known architectures.

    Args:
        model: The model to search.

    Returns:
        A tuple containing the key path to the layer list and the list itself.

    Raises:
        ValueError: If the model has no ``base_model`` attribute.
        NotImplementedError: If the architecture is not recognized.
    """
    # TODO implement this so that we can do hooked transformer training.
    base_model = _get_base_model(model)

    # Try attribute-based detection first
    result = _try_attribute_layers(base_model)
    if result is not None:
        path_components, layers = result
        path = ".".join(["base_model"] + path_components)
        return path, layers

    # Fall back to isinstance checks for known architectures
    path_to_layers = ["base_model"]
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
    elif isinstance(base_model, models.mistral.modeling_mistral.MistralModel):
        path_to_layers += ["layers"]
    elif isinstance(base_model, models.gemma.modeling_gemma.GemmaModel):
        path_to_layers += ["layers"]
    else:
        available = [a for a in dir(base_model) if not a.startswith("_")]
        raise NotImplementedError(
            f"Unsupported model architecture: {type(base_model).__name__}. "
            f"Could not auto-detect transformer layers via attribute probing. "
            f"Available attributes: {available[:15]}. "
            f"Please open an issue at: "
            f"https://github.com/AlignmentResearch/tuned-lens/issues"
        )

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
