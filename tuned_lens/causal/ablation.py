from ..model_surgery import get_transformer_layers
from ..utils import revcumsum
from .utils import derange
from contextlib import contextmanager
from typing import Callable, Literal, Optional, Sequence
import torch as th
import torch.nn.functional as F


@contextmanager
def ablate_layer(
    model: th.nn.Module,
    layer_index: int,
    method: Literal["resample", "mean", "zero"],
    *,
    mode: Literal["batch", "token"] = "batch",
    target_sample: Optional[int] = None,
):
    """Replace residual outputs of the specified layer with dummy values.

    If the method is "resample", the residuals are replaced with corresponding
    residuals from a randomly sampled sequence in the batch. If the method is "mean",
    the residuals are replaced with their minibatch means. If the method is "zero",
    all residuals are replaced with the zero vector.

    Args:
        model: The model to modify.
        layer_index: The index of the layer to modify.
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

        nonlocal target_sample
        if target_sample is not None:
            x = x[None, target_sample]
            target_sample = None

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


@th.jit.script
def _loo_geom_mixture_ce(logits: th.Tensor, labels: th.Tensor) -> th.Tensor:
    """Cross entropy loss of leave-one-out geometric mixtures of a batch of logits."""
    log_probs = logits.log_softmax(-1)
    log_probs.diagonal().fill_(0)  # Mask out the original sample

    return F.cross_entropy(
        # Skip dividing by the number of samples, since cross_entropy will normalize
        # the distribution anyway.
        log_probs.mean(0).flatten(0, -2),
        labels.flatten(),
    )


@th.jit.script
def _loo_arith_mixture_ce(logits: th.Tensor, labels: th.Tensor) -> th.Tensor:
    """Cross entropy loss of leave-one-out mixtures of a batch of logits."""
    log_probs = logits.log_softmax(-1)
    log_probs.diagonal().fill_(-th.inf)  # Mask out the original sample

    return F.cross_entropy(
        # Skip dividing by the number of samples, since cross_entropy will normalize
        # the distribution anyway.
        log_probs.logsumexp(0).flatten(0, -2),
        labels.flatten(),
    )


@th.autocast("cuda", enabled=th.cuda.is_available())
@th.no_grad()
def resampling_probe_loss(
    decoder: Callable[[th.Tensor], th.Tensor],
    stream: Sequence[th.Tensor],
    labels: th.Tensor,
    low_memory: bool = False,
    mean: Literal["arith", "geom"] = "geom",
) -> list[th.Tensor]:
    """Compute cross-entropy loss of hidden states decoded with resampling."""
    if len(labels) < 2:
        raise ValueError("Resampling requires at least two samples")
    if len(stream) < 2:
        raise ValueError("Resampling requires at least two layers")

    biases = revcumsum([h_ - h for h, h_ in zip(stream[:-1], stream[1:])])
    losses = []

    mixture_fn = _loo_arith_mixture_ce if mean == "arith" else _loo_geom_mixture_ce
    for hidden, bias in zip(stream, biases):
        # Sequentially compute the loss for each token in the sequence
        if low_memory:
            token_losses = []
            for b, h, y in zip(bias.unbind(-2), hidden.unbind(-2), labels[:, 1:].T):
                logits = decoder(h + b[:, None])
                token_losses.append(mixture_fn(logits, y))

            losses.append(th.stack(token_losses).mean())

        # Compute the loss for the entire sequence at once in parallel
        else:
            logits = decoder(hidden + bias[:, None])
            losses.append(mixture_fn(logits[..., :-1, :], labels[:, 1:]))

    return losses
