from ..model_surgery import get_final_layer_norm, get_transformer_layers
from ..nn.tuned_lens import TunedLens
from ..residual_stream import ResidualStream, record_residual_stream
from ..stats import js_divergence
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typing import cast, Literal, Optional, Sequence, Union
import math
import numpy as np
import plotly.graph_objects as go
import torch as th


Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


@th.autocast("cuda", enabled=th.cuda.is_available())
@th.no_grad()
def plot_logit_lens(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    *,
    input_ids: Optional[th.Tensor] = None,
    end_pos: Optional[int] = None,
    extra_decoder_layers: int = 0,
    layer_stride: int = 1,
    metric: Literal["ce", "entropy", "js", "kl"] = "entropy",
    newline_replacement: str = "\\n",
    newline_token: str = "Ċ",
    rank: int = 0,
    start_pos: int = 0,
    sublayers: bool = False,
    text: Optional[str] = None,
    topk_diff: bool = False,
    topk: int = 10,
    tuned_lens: Optional[TunedLens] = None,
    whitespace_replacement: str = "_",
    whitespace_token: str = "Ġ",
) -> go.Figure:
    if topk < 1:
        raise ValueError("topk must be greater than 0")

    """Plot a logit lens table for the given text."""
    tokens, outputs, stream = _run_inference(
        model, tokenizer, input_ids, text, sublayers, start_pos, end_pos
    )

    if tuned_lens is not None:
        hidden_lps = stream.zip_map(
            lambda h, i: tuned_lens(h, i).log_softmax(dim=-1),
            range(len(stream) - 1),
        )
        hidden_lps.layers.append(outputs.logits.log_softmax(dim=-1))
    else:
        E = model.get_output_embeddings()
        ln_f = get_final_layer_norm(model.base_model)
        assert isinstance(ln_f, th.nn.LayerNorm)

        _, layers = get_transformer_layers(model.base_model)
        L = len(layers)

        def decode_fn(x):
            # Apply extra decoder layers if needed
            for i in range(L - extra_decoder_layers, L):
                x, *_ = layers[i](x)

            return E(ln_f(x)).log_softmax(dim=-1)

        hidden_lps = stream.map(decode_fn)

    # Replace whitespace and newline tokens with their respective replacements
    format_fn = np.vectorize(
        lambda x: x.replace(whitespace_token, whitespace_replacement).replace(
            newline_token, newline_replacement
        )
    )

    # If rank, get the rank of next token predicted
    # Set this to the stats & top_strings
    if rank > 0:
        # remove the last rank sequences from the stream
        token_offset = rank
        hidden_lps = hidden_lps.map(lambda x: x[:, :-token_offset, :])
        ranks = _get_rank(hidden_lps, tokens, token_offset)
        top_strings = ranks
        # stats = ranks
        stats = ranks
        stats = stats.map(lambda x: th.log10(x + 1))
        max_color = int(np.log10(10000))  # out of 50k tokens
        tokens = tokens[:-token_offset]
    else:
        top_strings = (
            hidden_lps.map(lambda x: x.argmax(-1).squeeze().cpu().tolist())
            .map(tokenizer.convert_ids_to_tokens)  # type: ignore[arg-type]
            .map(format_fn)
        )
        max_color = math.log(model.config.vocab_size)

        if metric == "ce":
            raise NotImplementedError
        elif metric == "entropy":
            stats = hidden_lps.map(lambda x: -th.sum(x.exp() * x, dim=-1))
        elif metric == "js":
            max_color = None
            stats = hidden_lps.pairwise_map(js_divergence)
            top_strings.embeddings = None
        elif metric == "kl":
            log_probs = outputs.logits.log_softmax(-1)
            stats = hidden_lps.map(
                lambda x: th.sum(log_probs.exp() * (log_probs - x), dim=-1)
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

    topk_strings_and_probs = _get_topk_probs(
        hidden_lps=hidden_lps, tokenizer=tokenizer, k=topk, topk_diff=topk_diff
    )

    return _plot_stream(
        color_stream=stats.map(lambda x: x.squeeze().cpu()),
        colorbar_label=f"{metric} (nats)" if not rank else "Rank",
        layer_stride=layer_stride,
        top_1_strings=top_strings,
        top_k_strings_and_probs=format_fn(topk_strings_and_probs),
        x_labels=format_fn(tokens),
        vmax=max_color,
        k=topk,
        title=("Tuned Lens" if tuned_lens is not None else "Logit Lens")
        + (
            f" ({model.name_or_path})"
            if rank < 1
            else f": Rank {rank} ({model.name_or_path})"
        ),
        colorscale="rdbu_r" if rank < 1 else "blues",
        rank=rank,
    )


def _get_rank(
    stream: ResidualStream,
    input_ids: th.Tensor,
    token_offset: int = 1,
):
    # Skip the first token because not predicted
    next_token_ids = input_ids[:, token_offset:]
    # Get the sorted indices of the logits for each layer of hidden_lps
    stream = stream.map(lambda x: x.argsort(dim=-1, descending=True))
    # Get the rank of the ground-truth next token for each layer
    return stream.map(lambda x: (next_token_ids.unsqueeze(-1) == x).nonzero()[:, -1])


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
    top_k_strings_and_probs=None,
    x_labels: Sequence[str] = (),
    top_1_strings=None,
    colorbar_label: str = "",
    layer_stride: int = 1,
    vmax: Optional[int] = None,
    k: int = 10,
    title: str = "",
    colorscale: str = "rdbu_r",
    rank: int = 0,
) -> go.Figure:

    color_matrix = np.stack(list(color_stream))[::layer_stride]
    top_1_strings = np.stack(list(top_1_strings))[::layer_stride]

    # Hack to ensure that Plotly doesn't de-duplicate the x-axis labels
    x_labels = [x + "\u200c" * i for i, x in enumerate(x_labels)]

    colorbar = dict(
        title=colorbar_label,
        titleside="right",
        tickfont=dict(size=20),
    )
    if rank > 0:  # make log-scale for rank plots
        colorbar["tickvals"] = [0, 1, 2, 3, 4]
        colorbar["ticktext"] = ["1", "10", "100", "1000", "10000"]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            colorscale=colorscale,
            customdata=top_k_strings_and_probs[::layer_stride],
            text=top_1_strings,
            texttemplate="%{text}",
            x=x_labels,
            y=color_stream.labels()[::layer_stride],
            z=color_matrix,
            hoverlabel=dict(bgcolor="rgb(42, 42, 50)"),
            hovertemplate="<br>".join(
                f" %{{customdata[{i}][0]}} %{{customdata[{i}][1]:.1f}}% "
                for i in range(k)
            )
            + "<extra></extra>",
            colorbar=dict(
                title=colorbar_label,
                titleside="right",
            ),
            zmax=vmax,
            zmin=0,
        )
    )

    # TODO Height needs to equal some function of Max(num_layers, topk).
    # Ignore for now. Works until k=18
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        width=200 + 80 * len(x_labels),
        xaxis_title="Input",
        yaxis_title="Layer",
    )
    return fig


# Allow all naming conventions
plot_focused_lens = plot_logit_lens
plot_tuned_lens = plot_logit_lens


def _run_inference(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    input_ids: Optional[th.Tensor],
    text: Optional[str],
    sublayers: bool,
    start_pos: int,
    end_pos: Optional[int],
) -> tuple:
    if text is not None:
        input_ids = cast(th.Tensor, tokenizer.encode(text, return_tensors="pt"))
    elif input_ids is None:
        raise ValueError("Either text or input_ids must be provided")

    with record_residual_stream(model, sublayers=sublayers) as stream:
        outputs = model(input_ids.to(model.device))

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    if start_pos > 0:
        outputs.logits = outputs.logits[..., start_pos:end_pos, :]
        stream = stream.map(lambda x: x[..., start_pos:end_pos, :])
        tokens = tokens[start_pos:end_pos]

    return tokens, outputs, stream
