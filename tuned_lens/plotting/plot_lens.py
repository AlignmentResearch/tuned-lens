"""Plot a lens table for some given text and model."""

from ..nn.lenses import Lens
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import (
    Iterable,
    cast,
    Literal,
    Optional,
    Sequence,
    Union,
)
import numpy as np
import plotly.graph_objects as go
import torch as th
import torch.nn.functional as F


@dataclass
class TokenFormatter:
    """Format tokens for display in a plot."""

    ellipsis: str = "…"
    newline_replacement: str = "\\n"
    newline_token: str = "Ċ"
    whitespace_token: str = "Ġ"
    whitespace_replacement: str = "_"
    max_string_len: Optional[int] = 7

    def format(self, token: str) -> str:
        """Format a token for display in a plot."""
        if self.max_string_len is not None and len(token) > self.max_string_len:
            token = token[: self.max_string_len - len(self.ellipsis)] + self.ellipsis
        token = token.replace(self.newline_token, self.newline_replacement)
        token = token.replace(self.whitespace_token, self.whitespace_replacement)
        return token


@dataclass
class StreamLabels:
    """Contains sets of labels for each layer and position in the residual stream."""

    # (n_layers x sequence_length) label for each layer and position in the stream.
    label_strings: NDArray[np.str_]
    # (sequence_length) labels for the sequence dimension typically the input tokens.
    sequence_labels: NDArray[np.str_]
    # (n_layers x sequence_length x k) k entries to display when hovering over a cell.
    # For example, the top k prediction from the lens at each layer.
    hover_over_entries: Optional[NDArray[np.str_]] = None


@dataclass
class PloatableStreamStatistic:
    """This class represents a stream statistic that can be visualized.

    For example, the entropy of the lens predictions at each layer. This class is
    meant to serve as an interface for the plotting code.
    """

    # The name of the statistic.
    name: str
    # (n_layers x sequence_length) value of the statistic at each layer and position.
    stats: NDArray[np.float32] = np.zeros((0, 0), dtype=float)
    # labels for each layer and position in the stream. For example, the top 1
    # prediction from the lens at each layer.
    stream_labels: Optional[StreamLabels] = None

    units: Optional[str] = None
    max: Optional[float] = None
    min: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate class invariants."""
        assert len(self.stats.shape) == 2
        assert self.stream_labels is None or (
            self.stream_labels.label_strings.shape == self.stats.shape
            and self.stream_labels.sequence_labels.shape[0] == self.stats.shape[1]
        )

    @property
    def num_layers(self) -> int:
        """Return the number of layers in the stream."""
        return self.stats.shape[0]


Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
Statistic = Literal["ce", "entropy", "forward_kl", "max_prob"]
Divergence = Literal["kl", "js"]


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
    token_formatter: Optional[TokenFormatter] = None,
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
        token_formatter: A TokenFormatter to use for formatting the tokens to be
            used as labels.
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

    if token_formatter is None:
        token_formatter = TokenFormatter()

    # Replace whitespace and newline tokens with their respective replacements
    format_fn = np.vectorize(
        lambda x: token_formatter.format(x) if isinstance(x, str) else "<unk>"
    )

    stream_lps, model_logits, targets, tokens = _get_stream_lps_from_lens(
        lens=lens,
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        start_pos=start_pos,
        end_pos=end_pos,
        mask_input=mask_input,
    )

    top_strings = []
    for lps in stream_lps:
        ids = lps.argmax(-1).squeeze().cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids)
        top_strings.append(format_fn(tokens))

    if min_prob:
        top_strings = [
            [
                s if lp.max() > np.log(min_prob) else ""
                for s, lp in zip(strings, log_probs)
            ]
            for strings, log_probs in zip(top_strings, stream_lps)
        ]

    plotable_stream = compute_statistics(
        statistic, list(stream_lps), model_logits=model_logits, targets=targets
    )

    plotable_stream.stream_labels = StreamLabels(
        label_strings=np.vstack(list(top_strings)),
        sequence_labels=np.array(format_fn(tokens)),
        hover_over_entries=_get_topk_probs(
            stream_lps=stream_lps,
            tokenizer=tokenizer,
            formatter=token_formatter,
            k=topk,
            topk_diff=topk_diff,
        ),
    )

    color_scale = "blues"
    color_scale = "rdbu_r"

    return _plot_stream(
        plotable_stream=plotable_stream,
        layer_stride=layer_stride,
        title=lens.__class__.__name__ + (f" ({model.name_or_path})"),
        colorscale=color_scale,
    )


@th.autocast("cuda", enabled=th.cuda.is_available())
@th.no_grad()
def compare_models(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    tokenizer: Tokenizer,
    lens: Lens,
    *,
    text: Optional[str] = None,
    input_ids: Optional[th.Tensor] = None,
    mask_input: bool = False,
    start_pos: int = 0,
    end_pos: Optional[int] = None,
    layer_stride: int = 1,
    token_formatter: Optional[TokenFormatter] = None,
    divergence: Divergence = "kl",
    min_prob: float = 0.0,
    topk: int = 10,
    topk_diff: bool = False,
) -> go.Figure:
    """Compare the predictions of two models using a lens."""
    if topk < 1:
        raise ValueError("topk must be greater than 0")

    if text is not None:
        input_ids = cast(th.Tensor, tokenizer.encode(text, return_tensors="pt"))
    elif input_ids is None:
        raise ValueError("Either text or input_ids must be provided")

    if input_ids.nelement() < 1:
        raise ValueError("Input must be at least 1 token long.")

    if token_formatter is None:
        token_formatter = TokenFormatter()

    locals()
    # Plot these divergences


def _get_stream_lps_from_lens(
    *,
    lens: Lens,
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    input_ids: th.IntTensor,
    start_pos: int,
    end_pos: Optional[int],
    mask_input: bool,
):
    outputs = model(input_ids.to(model.device), output_hidden_states=True)

    # Slice arrays the specified range
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    model_logits = outputs.logits[..., start_pos:end_pos, :]
    stream = [h[..., start_pos:end_pos, :] for h in outputs.hidden_states]
    targets = th.cat(
        (input_ids, th.full_like(input_ids[..., -1:], tokenizer.eos_token_id)),
        dim=-1,
    )

    # Adjust start and end positions for the targets accounting for negative indices
    t_start_pos = start_pos if start_pos < 0 else start_pos + 1
    if end_pos is None:
        t_end_pos = None
    elif end_pos < 0:
        t_end_pos = end_pos
    else:
        t_end_pos = end_pos + 1

    targets = targets[..., t_start_pos:t_end_pos]
    tokens = tokens[start_pos:end_pos]

    # Create the stream of log probabilities from the lens
    stream_lps = []
    for i, h in enumerate(stream[:-1]):
        logits = lens.forward(h, i)

        if mask_input:
            logits[..., input_ids] = -th.finfo(h.dtype).max

        stream_lps.append(logits.log_softmax(dim=-1))

    # Add model predictions
    stream_lps.append(model_logits.log_softmax(dim=-1)[..., start_pos:end_pos, :])
    return stream_lps, model_logits, targets, tokens


def compute_statistics(
    statistic: Statistic,
    stream_lps: Sequence[th.Tensor],
    model_logits: th.Tensor,
    targets: th.Tensor,
) -> PloatableStreamStatistic:
    """Compute a statistic for each layer in the stream.

    Args:
        statistic: The statistic to compute. One of "ce", "entropy", "kl", "kl_div".
        stream_lps: (n_layers x vocab) The stream of hidden layer
        log probabilities produced by a lens.
        model_logits: The logits produced by the model.
        targets: The target ids for the sequence.

    Returns:
        A named tuple containing the statistics value at each layer and position
        and its name and units.
    """

    def collect(stream: Iterable[th.Tensor]) -> NDArray[np.float32]:
        return np.vstack(list(map(lambda x: x.cpu().numpy(), stream)))

    if statistic == "ce":
        assert targets.shape == stream_lps[-1].shape[:-1], (
            "Batch and sequence lengths of targets and log probs must match."
            f"Got {targets.shape} and {stream_lps[-1].shape[:-1]} respectively."
        )
        num_tokens = targets.nelement()
        targets = targets.reshape(num_tokens)
        stream_lps = list(map(lambda x: x.reshape(num_tokens, -1), stream_lps))
        return PloatableStreamStatistic(
            name="Cross Entropy",
            units="nats",
            stats=collect(
                map(
                    lambda hlp: F.cross_entropy(hlp, targets, reduction="none"),
                    stream_lps,
                )
            ),
        )
    elif statistic == "entropy":
        return PloatableStreamStatistic(
            name="Entropy",
            units="nats",
            stats=collect(
                map(lambda hlp: -th.sum(hlp.exp() * hlp, dim=-1), stream_lps)
            ),
        )
    elif statistic == "forward_kl":
        log_probs = model_logits.log_softmax(-1)
        return PloatableStreamStatistic(
            name="Forward KL",
            units="nats",
            stats=collect(
                map(
                    lambda hlp: th.sum(log_probs.exp() * (log_probs - hlp), dim=-1),
                    stream_lps,
                )
            ),
        )
    elif statistic == "max_prob":
        return PloatableStreamStatistic(
            name="Max Probability",
            units="Probability",
            stats=collect(map(lambda x: x.max(-1).values.exp(), stream_lps)),
        )
    else:
        raise ValueError(f"Unknown statistic: {statistic}")


def _get_topk_probs(
    stream_lps: Sequence[th.FloatTensor],
    tokenizer: Tokenizer,
    formatter: TokenFormatter,
    k: int,
    topk_diff: bool,
):
    precents = [x.exp() * 100 for x in stream_lps]

    if topk_diff:
        raise NotImplementedError("topk_diff not implemented")

    precents = th.stack(list(precents)).squeeze(1)

    # Get the top-k tokens & probabilities for each
    topk = precents.topk(k, dim=-1)

    topk_values = topk.values
    # reshape topk_ind from (layers, seq, k) to (layers*seq*k), convert_ids_to_tokens,
    # then reshape back to (layers, seq, k)
    topk_tokens = tokenizer.convert_ids_to_tokens(topk.indices.reshape(-1).tolist())
    topk_tokens = np.array(topk_tokens).reshape(topk.indices.shape)

    def format_fn(token: str, percent: float):
        return f"{formatter.format(token)} %{percent:.2f}"

    format_fn = np.vectorize(format_fn)

    topk_strings_and_probs = format_fn(topk_tokens, topk_values.cpu().numpy())

    return topk_strings_and_probs


def _plot_stream(
    plotable_stream: PloatableStreamStatistic,
    layer_stride: int = 1,
    title: str = "",
    colorscale: str = "rdbu_r",
) -> go.Figure:
    labels = np.array(
        ["input", *map(str, range(1, plotable_stream.num_layers - 1)), "output"]
    )

    color_matrix = plotable_stream.stats

    if plotable_stream.stream_labels is not None:
        label_strings = plotable_stream.stream_labels.label_strings
        hover_over_entries = plotable_stream.stream_labels.hover_over_entries
        label_strings = _stride_keep_last(label_strings, layer_stride)
        # Hack to ensure that Plotly doesn't de-duplicate the x-axis labels
        x_labels = [
            x + "\u200c" * i
            for i, x in enumerate(plotable_stream.stream_labels.sequence_labels)
        ]

        if hover_over_entries is not None:
            hover_over_entries = _stride_keep_last(hover_over_entries, layer_stride)
            print(hover_over_entries)
    else:
        label_strings = None
        hover_over_entries = None
        x_labels = None

    color_matrix = _stride_keep_last(color_matrix, layer_stride)
    labels = _stride_keep_last(labels, layer_stride)

    heatmap = go.Heatmap(
        colorscale=colorscale,
        customdata=hover_over_entries,
        text=label_strings,
        texttemplate="<b>%{text}</b>",
        x=x_labels,
        y=labels,
        z=color_matrix,
        hoverlabel=dict(bgcolor="rgb(42, 42, 50)"),
        hovertemplate="<br>".join(
            f" %{{customdata[{i}]}}" for i in range(hover_over_entries.shape[2])
        )
        + "<extra></extra>",
        colorbar=dict(
            title=plotable_stream.units,
            titleside="right",
        ),
        zmax=plotable_stream.max,
        zmin=plotable_stream.min,
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


def _stride_keep_last(x: NDArray, stride: int):
    if stride == 1:
        return x
    elif len(x) % stride != 1:
        return np.concatenate([x[::stride], [x[-1]]])
    else:
        return x[::stride]
