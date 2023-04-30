"""Provides tools for ablating layers of a transformer model."""
from contextlib import contextmanager
from typing import Literal

import torch as th

from ..model_surgery import get_transformer_layers
from .utils import derange


@contextmanager
def ablate_layer(
    model: th.nn.Module,
    layer_index: int,
    method: Literal["resample", "mean", "zero"],
    *,
    mode: Literal["batch", "token"] = "batch",
):
    """Replace residual outputs of the specified layer with dummy values.

    If the method is "resample", the residuals are replaced with corresponding
    residuals from a randomly sampled sequence in the batch. If the method is "mean",
    the residuals are replaced with their minibatch means. If the method is "zero",
    all residuals are replaced with the zero vector.

    Args:
        model: The model to modify.
        layer_index: The index of the layer to modify.
        method: How to ablate the layer see above.
        mode: Whether to compute the mean only over the batch dimension or over the
            batch and token dimensions.
    """
    assert layer_index >= 0

    def ablate_hook(_, inputs, outputs):
        x, *_ = inputs
        y, *extras = outputs
        if method == "zero":
            return x, *extras

        residuals = y - x
        original_shape = x.shape
        if mode == "token":
            x = x.flatten(0, 1)
            residuals = residuals.flatten(0, 1)

        batch_size = x.shape[0]
        if batch_size < 2:
            raise ValueError("Mean ablation requires a batch size >= 2")

        if method == "resample":
            ablated = x + derange(residuals)
        elif method == "mean":
            ablated = x + residuals.mean(0, keepdim=True)
        else:
            raise ValueError(f"Unknown ablation method: {method}")

        return ablated.reshape(original_shape), *extras

    _, layers = get_transformer_layers(model)
    handle = layers[layer_index].register_forward_hook(ablate_hook)  # type: ignore

    try:
        yield model
    finally:
        handle.remove()
