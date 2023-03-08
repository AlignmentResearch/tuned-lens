from ..model_surgery import get_final_layer_norm, get_transformer_layers
from ..nn.lenses import Lens
from ..residual_stream import ResidualStream, record_residual_stream
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typing import Any, cast, Literal, Optional, Sequence, Union
import math
import numpy as np
import plotly.graph_objects as go
import torch as th


Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


@th.autocast("cuda", enabled=th.cuda.is_available())
@th.no_grad()
def plot_lens(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    lens: Lens,
    *,
    text: Optional[str] = None,
    input_ids: Optional[th.Tensor] = None,
    mask_input: bool = False,
    start_pos: int = 0,
    end_pos: Optional[int] = None,
    layer_stride: int = 1,
    metric: Literal["ce", "entropy", "kl", "log_prob", "prob"] = "entropy",
    min_prob: float = 0.0,
    newline_replacement: str = "\\n",
    newline_token: str = "Ċ",
    whitespace_token: str = "Ġ",
    whitespace_replacement: str = "_",
    topk_diff: bool = False,
    topk: int = 10,
) -> go.Figure:
    """Plot a logit lens table for the given text.

    Args:
        model: The model to be examined.
        tokenizer: The tokenizer to use for encoding the text.
        lens: The lens use for intermediate predictions.

    KWArgs:
        text: The text to use for evaluated. If not provided, the input_ids will be
            used.
        input_ids: The input IDs to use for evaluated. If not provided, the text will
            be encoded.
        mask_input: Forbid the lens from predicting the input tokens.
        start_pos: The first token id to visualize.
        end_pos: The token id to stop visualizing before.
        extra_decoder_layers: The number of extra decoder layers to apply after before
            the unembeding.
        layer_stride: The number of layers to skip between each layer displayed.
        metric: The metric to use for the lens table.
        min_prob: At least one token must have a probability greater than this for the
            lens prediction to be displayed.
        newline_replacement: The string to replace newline tokens with.
        newline_token: The token to replace with newline_replacement.
        whitespace_replacement: The string to replace whitespace tokens with.
        topk_diff: If true, only show the topk most different tokens.

    Returns:
        A plotly figure containing the lens table.
    """
    if topk < 1:
        raise ValueError("topk must be greater than 0")

    """Plot a logit lens table for the given text."""
    if text is not None:
        input_ids = cast(th.Tensor, tokenizer.encode(text, return_tensors="pt"))
    elif input_ids is None:
        raise ValueError("Either text or input_ids must be provided")

    with record_residual_stream(model) as stream:
        outputs = model(input_ids.to(model.device))

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

    outputs.logits = outputs.logits[..., start_pos:end_pos, :]
    stream = stream.map(lambda x: x[..., start_pos:end_pos, :])
    tokens = tokens[start_pos:end_pos]

    def decode_tl(h, i):
        logits = lens.forward(h, i)
        if mask_input:
            logits[..., input_ids] = -th.finfo(h.dtype).max

        return logits.log_softmax(dim=-1)

    hidden_lps = stream.zip_map(
        decode_tl,
        range(len(stream) - 1),
    )
    hidden_lps.layers.append(outputs.logits.log_softmax(dim=-1))

    # Replace whitespace and newline tokens with their respective replacements
    format_fn = np.vectorize(
        lambda x: x.replace(whitespace_token, whitespace_replacement).replace(
            newline_token, newline_replacement
        )
        if isinstance(x, str)
        else "<unk>"
    )

    top_strings = (
        hidden_lps.map(lambda x: x.argmax(-1).squeeze().cpu().tolist())
        .map(tokenizer.convert_ids_to_tokens)  # type: ignore[arg-type]
        .map(format_fn)
    )

    max_color = math.log(model.config.vocab_size)

    if min_prob:
        top_strings = top_strings.zip_map(
            lambda strings, log_probs: [
                s if lp.max() > np.log(min_prob) else "" for s, lp in zip(strings, log_probs)
            ],
            hidden_lps,
        )

    if metric == "ce":
        raise NotImplementedError
    elif metric == "entropy":
        stats = hidden_lps.map(lambda x: -th.sum(x.exp() * x, dim=-1))
    elif metric == "kl":
        log_probs = outputs.logits.log_softmax(-1)
        stats = hidden_lps.map(
            lambda x: th.sum(log_probs.exp() * (log_probs - x), dim=-1)
        )
    elif metric == "prob":
        max_color = 1.0
        stats = hidden_lps.map(lambda x: x.max(-1).values.exp())
    else:
        raise ValueError(f"Unknown metric: {metric}")

    topk_strings_and_probs = _get_topk_probs(
        hidden_lps=hidden_lps, tokenizer=tokenizer, k=topk, topk_diff=topk_diff
    )

    color_scale = "blues"

    if metric == "prob":
        color_label = "Probability"
    else:
        color_label = f"{metric.capitalize()} (nats)"
        color_scale = "rdbu_r"

    return _plot_stream(
        color_stream=stats.map(lambda x: x.squeeze().cpu()),
        colorbar_label=color_label,
        layer_stride=layer_stride,
        top_1_strings=top_strings,
        top_k_strings_and_probs=format_fn(topk_strings_and_probs),
        x_labels=format_fn(tokens),
        vmax=max_color,
        k=topk,
        title=lens.__class__.__name__
        + (
            f" ({model.name_or_path})"
        ),
        colorscale=color_scale,
    )


def _get_topk_probs(
    hidden_lps: ResidualStream,
    tokenizer: Tokenizer,
    k: int,
    topk_diff: bool,
):
    probs = hidden_lps.map(lambda x: x.exp() * 100)
    if topk_diff:
        probs = probs.pairwise_map(lambda x, y: y - x)
    probs = th.stack(list(probs)).squeeze(1)

    if topk_diff:
        topk = probs.abs().topk(k, dim=-1)
        topk_values = probs.gather(
            -1, topk.indices
        )  # get the topk values but include negative values
    else:
        # Get the top-k tokens & probabilities for each
        topk = probs.topk(k, dim=-1)
        topk_values = topk.values

    # reshape topk_ind from (layers, seq, k) to (layers*seq*k), convert_ids_to_tokens,
    # then reshape back to (layers, seq, k)
    topk_ind = tokenizer.convert_ids_to_tokens(topk.indices.reshape(-1).tolist())
    topk_ind = np.array(topk_ind).reshape(topk.indices.shape)

    topk_strings_and_probs = np.stack((topk_ind, topk_values.cpu()), axis=-1)
    if topk_diff:
        # add a new bottom row of "N/A" for topk_strings_and_probs because we don't
        # have a "previous" layer to compare to
        topk_strings_and_probs = np.concatenate(
            (
                np.full((1, topk_strings_and_probs.shape[1], k, 2), "N/A"),
                topk_strings_and_probs,
            ),
            axis=0,
        )

    return topk_strings_and_probs


def _plot_stream(
    color_stream: ResidualStream,
    top_k_strings_and_probs,
    top_1_strings: ResidualStream,
    x_labels: Sequence[str] = (),
    colorbar_label: str = "",
    layer_stride: int = 1,
    vmax: Optional[float] = None,
    k: int = 10,
    title: str = "",
    colorscale: str = "rdbu_r",
) -> go.Figure:

    # Hack to ensure that Plotly doesn't de-duplicate the x-axis labels
    x_labels = [x + "\u200c" * i for i, x in enumerate(x_labels)]

    labels = ["input", *range(1, len(color_stream) - 1), "output"]

    color_matrix = np.stack(list(color_stream))
    top_1_strings = np.stack(list(top_1_strings))

    def stride_keep_last(x: Sequence[Any], stride: int):
        if stride == 1:
            return x
        elif len(x) % stride != 1:
            return np.concatenate([x[::stride], [x[-1]]])
        else:
            return x[::stride]

    color_matrix = stride_keep_last(color_matrix, layer_stride)
    labels = stride_keep_last(labels, layer_stride)
    top_1_strings = stride_keep_last(top_1_strings, layer_stride)
    top_k_strings_and_probs = stride_keep_last(
        top_k_strings_and_probs, layer_stride
    )

    heatmap = go.Heatmap(
        colorscale=colorscale,
        customdata=top_k_strings_and_probs,
        text=top_1_strings,
        texttemplate="<b>%{text}</b>",
        x=x_labels,
        y=labels,
        z=color_matrix,
        hoverlabel=dict(bgcolor="rgb(42, 42, 50)"),
        hovertemplate="<br>".join(
            f" %{{customdata[{i}][0]}} %{{customdata[{i}][1]:.1f}}% " for i in range(k)
        )
        + "<extra></extra>",
        colorbar=dict(
            title=colorbar_label,
            titleside="right",
        ),
        zmax=vmax,
        zmin=0,
    )

    # TODO Height needs to equal some function of Max(num_layers, topk).
    # Ignore for now. Works until k=18
    fig = go.Figure(heatmap).update_layout(
        title_text=title,
        title_x=0.5,
        width=200 + 80 * len(x_labels),
        xaxis_title="Input",
        yaxis_title="Layer",
    )
    return fig
