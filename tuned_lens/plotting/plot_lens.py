"""Plot a lens table for some given text and model."""

from ..nn.lenses import Lens
from ..residual_stream import ResidualStream, record_residual_stream
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typing import Any, NamedTuple, cast, Literal, Optional, Sequence, Union
import math
import numpy as np
import plotly.graph_objects as go
import torch as th
import torch.nn.functional as F


Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
Statistic = Literal["ce", "entropy", "forward_kl", "max_prob"]


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
    statistic: Statistic = "entropy",
    min_prob: float = 0.0,
    max_string_len: Optional[int] = 7,
    ellipsis: str = "…",
    newline_replacement: str = "\\n",
    newline_token: str = "Ċ",
    whitespace_token: str = "Ġ",
    whitespace_replacement: str = "_",
    topk: int = 10,
    topk_diff: bool = False,
) -> go.Figure:
    """Plot a lens table for the given text.

    Args:
        model: The model to be examined.
        tokenizer: The tokenizer to use for encoding the text.
        lens: The lens use for intermediate predictions.
        text: The text to use for evaluated. If not provided, the input_ids will be
            used.
        input_ids: The input IDs to use for evaluated. If not provided, the text will
            be encoded.
        mask_input: Forbid the lens from predicting the input tokens.
        start_pos: The first token id to visualize.
        end_pos: The token id to stop visualizing before.
        extra_decoder_layers: The number of extra decoder layers to apply after
            before the unembeding.
        layer_stride: The number of layers to skip between each layer displayed.
        statistic: The statistic to use for the lens table.
            * ce: The cross entropy between the labels and the lens predictions.
            * entropy: The entropy of the lens prediction.
            * forward_kl: The KL divergence between the model and the lens.
            * max_prob: The probability of the most likely token.
        min_prob: At least one token must have a probability greater than this for the
            lens prediction to be displayed.
        max_string_len: If not None, clip the string representation of the tokens to
            this length and add an ellipsis.
        ellipsis: The string to use for the ellipsis.
        newline_replacement: The string to replace newline tokens with.
        newline_token: The substring to replace with newline_replacement.
        whitespace_replacement: The string to replace whitespace tokens with.
        whitespace_token: The substring to replace with whitespace_replacement.
        topk: The number of tokens to visualize when hovering over a cell.
        topk_diff: If true show the top k tokens where the metric has changed the from
            the previous layer.

    Returns:
        A plotly figure containing the lens table.
    """
    if topk < 1:
        raise ValueError("topk must be greater than 0")

    if text is not None:
        input_ids = cast(th.Tensor, tokenizer.encode(text, return_tensors="pt"))
    elif input_ids is None:
        raise ValueError("Either text or input_ids must be provided")

    if input_ids.nelement() < 1:
        raise ValueError("Input must be at least 1 token long.")

    with record_residual_stream(model) as stream:
        outputs = model(input_ids.to(model.device))

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    model_logits = outputs.logits[..., start_pos:end_pos, :]
    stream = stream.map(lambda x: x[..., start_pos:end_pos, :])
    targets = th.cat(
        (input_ids, th.full_like(input_ids[..., -1:], tokenizer.eos_token_id)),
        dim=-1,
    )
    t_start_pos = start_pos if start_pos < 0 else start_pos + 1
    if end_pos is None:
        t_end_pos = None
    elif end_pos < 0:
        t_end_pos = end_pos
    else:
        t_end_pos = end_pos + 1

    targets = targets[..., t_start_pos:t_end_pos]
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
    # Add model predictions
    hidden_lps.layers.append(
        outputs.logits.log_softmax(dim=-1)[..., start_pos:end_pos, :]
    )

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
                s if lp.max() > np.log(min_prob) else ""
                for s, lp in zip(strings, log_probs)
            ],
            hidden_lps,
        )

    p_stat = compute_statistics(
        statistic, hidden_lps, model_logits=model_logits, targets=targets
    )

    topk_strings_and_probs = _get_topk_probs(
        hidden_lps=hidden_lps, tokenizer=tokenizer, k=topk, topk_diff=topk_diff
    )

    color_scale = "blues"
    color_scale = "rdbu_r"

    return _plot_stream(
        color_stream=p_stat.stats.map(lambda x: x.squeeze().cpu()),
        colorbar_label=(p_stat.name + (f" ({p_stat.units})" if p_stat.units else "")),
        layer_stride=layer_stride,
        top_1_strings=top_strings,
        top_k_strings_and_probs=format_fn(topk_strings_and_probs),
        x_labels=format_fn(tokens),
        vmax=max_color,
        k=topk,
        title=lens.__class__.__name__ + (f" ({model.name_or_path})"),
        colorscale=color_scale,
        ellipsis=ellipsis,
        max_string_len=max_string_len,
    )


class PlotableStatistic(NamedTuple):
    """A plotable statistic."""

    stats: ResidualStream
    name: str
    units: Optional[str] = None
    max: Optional[float] = None
    min: Optional[float] = None


def compute_statistics(
    statistic: Statistic,
    hidden_lps: ResidualStream,
    model_logits: th.Tensor,
    targets: th.Tensor,
) -> PlotableStatistic:
    """Compute a statistic for each layer in the stream.

    Args:
        statistic: The statistic to compute. One of "ce", "entropy", "kl", "kl_div".
        hidden_lps: The stream of hidden layer log probabilities produced by a lens.
        model_logits: The logits produced by the model.
        targets: The target ids for the sequence.

    Returns:
        A named tuple containing the statistics value at each layer and position
        and its name and units.
    """
    if statistic == "ce":
        assert targets.shape == hidden_lps[-1].shape[:-1], (
            "Batch and sequence lengths of targets and log probs must match."
            f"Got {targets.shape} and {hidden_lps[-1].shape[:-1]} respectively."
        )
        num_tokens = targets.nelement()
        targets = targets.reshape(num_tokens)
        hidden_lps = hidden_lps.map(lambda x: x.reshape(num_tokens, -1))
        return PlotableStatistic(
            name="Cross Entropy",
            units="nats",
            stats=hidden_lps.map(
                lambda hlp: F.cross_entropy(hlp, targets, reduction="none")
            ),
        )
    elif statistic == "entropy":
        return PlotableStatistic(
            name="Entropy",
            units="nats",
            stats=hidden_lps.map(lambda hlp: -th.sum(hlp.exp() * hlp, dim=-1)),
        )
    elif statistic == "forward_kl":
        log_probs = model_logits.log_softmax(-1)
        return PlotableStatistic(
            name="Forward KL",
            units="nats",
            stats=hidden_lps.map(
                lambda hlp: th.sum(log_probs.exp() * (log_probs - hlp), dim=-1)
            ),
        )
    elif statistic == "max_prob":
        return PlotableStatistic(
            name="Max Probability",
            stats=hidden_lps.map(lambda x: x.max(-1).values.exp()),
        )
    else:
        raise ValueError(f"Unknown statistic: {statistic}")


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
    max_string_len: Optional[int] = None,
    ellipsis: str = "…",
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

    labels = ["input", *map(str, range(1, len(color_stream) - 1)), "output"]

    if max_string_len is not None:
        # Clip top 1 strings to a maximum length and add an ellipsis
        top_1_strings = top_1_strings.map(
            lambda x: [
                s if len(s) <= max_string_len else s[:max_string_len] + ellipsis
                for s in x
            ]
        )

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
    top_k_strings_and_probs = stride_keep_last(top_k_strings_and_probs, layer_stride)

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
