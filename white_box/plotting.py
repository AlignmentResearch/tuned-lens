from .stats import geodesic_distance, js_divergence, js_distance
from .model_surgery import get_final_layer_norm, get_transformer_layers
from .residual_stream import ResidualStream, record_residual_stream
from .nn.tuned_lens import TunedLens
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from typing import cast, Literal, Optional, Sequence, Union
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import torch as th
import torch.nn.functional as F


@th.no_grad()
def plot_logit_lens(
    model_or_name: Union[PreTrainedModel, str],
    *,
    input_ids: Optional[th.Tensor] = None,
    extra_decoder_layers: int = 0,
    layer_stride: int = 1,
    metric: Literal["ce", "geodesic", "entropy", "js", "kl"] = "entropy",
    residual_means: Optional[ResidualStream] = None,
    sublayers: bool = False,
    start_pos: int = 0,
    end_pos: Optional[int] = None,
    text: Optional[str] = None,
    tuned_lens: Optional[TunedLens] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    rank: int = 0,
    topk: int = 5,
    topk_diff: bool = False,
    whitespace_token: str = "Ġ",
    whitespace_replacement: str = "_",
    newline_token: str = "Ċ",
    newline_replacement: str = "\\n",
    add_last_tuned_lens_layer: bool = False,
    test_text: str = None,
):
    if(topk<1):
        raise ValueError("topk must be greater than 0")

    """Plot a logit lens table for the given text."""
    model, tokens, outputs, stream = _run_inference(
        model_or_name, input_ids, text, tokenizer, sublayers, start_pos, end_pos
    )

    if residual_means is not None:
        acc = th.zeros_like(residual_means.layers[0])
        for state, mean in zip(reversed(stream), reversed(residual_means)):
            state += acc
            acc += mean

    if tuned_lens is not None:
        hidden_lps = tuned_lens.transform(stream).map(lambda x: x.log_softmax(dim=-1))
        if(add_last_tuned_lens_layer):
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

    #If rank, get the rank of next token predicted
    #Set this to the stats & top_strings
    if(rank > 0):
        #remove the last rank sequences from the stream
        token_offset = rank
        hidden_lps = hidden_lps.map(lambda x: x[:, :-token_offset, :])
        ranks = _get_rank(hidden_lps, token_offset, input_ids)
        top_strings = ranks
        # stats = ranks
        stats = ranks
        stats = stats.map(lambda x: th.log10(x + 1))
        metric = "rank"
        max_color = int(np.log10(10000)) #out of 50k tokens
        tokens = tokens[:-token_offset]
        colorbar_label = "Rank"
        colorscale = "blues"
    else:
        top_tokens = hidden_lps.map(lambda x: x.argmax(-1).squeeze().cpu().tolist())
        top_strings = top_tokens.map(
            tokenizer.convert_ids_to_tokens  # type: ignore[arg-type]
        )
        #Replace whitespace and newline tokens with their respective replacements
        top_strings = top_strings.map(
            lambda x: [t.replace(whitespace_token, whitespace_replacement) if whitespace_token in t else t.replace(newline_token, newline_replacement) if newline_token in t else t for t in x]
        )
        max_color = math.log(model.config.vocab_size)
        if metric == "ce":
            raise NotImplementedError
        elif metric == "entropy":
            stats = hidden_lps.map(lambda x: -th.sum(x.exp() * x, dim=-1))
        elif metric == "geodesic":
            max_color = None
            stats = hidden_lps.pairwise_map(geodesic_distance)
            top_strings.embeddings = None
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
        colorbar_label = f"{metric} (nats)"
        colorscale = "rdbu_r"

    x_labels =  [t.replace(whitespace_token, whitespace_replacement) if whitespace_token in t else t.replace(newline_token, newline_replacement) if newline_token in t else t for t in tokens]
    topk_strings_and_probs = _get_topk_probs(hidden_lps=hidden_lps, tokenizer=tokenizer, k=topk, topk_diff=topk_diff)
    topk_strings_and_probs = np.array([[[[t.replace(whitespace_token, whitespace_replacement) if whitespace_token in t else t.replace(newline_token, "\\n") if newline_token in t else t for t in x] for x in y] for y in z] for z in topk_strings_and_probs])

    _plot_stream( #TODO, do we have to return?
        color_stream = stats.map(lambda x: x.squeeze().cpu()),
        top_k_strings_and_probs = topk_strings_and_probs,
        x_labels = x_labels,
        top_1_strings = top_strings,
        colorbar_label = colorbar_label,
        layer_stride = 1,
        vmax = max_color,
        k = topk,
        title= ("Tuned Lens" if tuned_lens is not None else "Logit Lens") + (": Top Token" if rank < 1 else f": Rank {rank}") + ("[Test: " + test_text + "]" if test_text else ""),
        colorscale = "rdbu_r" if rank < 1 else "blues",
        rank = rank,
    )

def _get_rank(
    stream: ResidualStream,
    token_offset: int=1,
    input_ids: th.Tensor=None,
):
    #Skip the first token because not predicted
    next_token_ids = input_ids[:, token_offset:] 
    #Get the sorted indices of the logits for each layer of hidden_lps
    stream = stream.map(lambda x: x.argsort(dim=-1, descending=True))
    #Get the rank of the ground-truth next token for each layer
    return stream.map(lambda x: (next_token_ids.unsqueeze(-1) == x).nonzero()[:, -1])


def _get_topk_probs(
    hidden_lps: ResidualStream,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    k=5,
    topk_diff=False,
):  
    probs = hidden_lps.map(lambda x: x.exp()*100)
    if(topk_diff):
        probs = probs.pairwise_map(lambda x, y: y - x)
    probs = th.stack(list(probs)).squeeze(1)

    if(topk_diff):
        topk = probs.abs().topk(k, dim=-1)
        topk_values = probs.gather(-1, topk.indices) #get the topk values but include negative values
    else:
        #Get the top-k tokens & probabilities for each
        topk = probs.topk(k, dim=-1)
        topk_values = topk.values
    #reshape topk_ind from (layers, seq, k) to (layers*seq*k), convert_ids_to_tokens, then reshape back to (layers, seq, k)
    topk_ind = tokenizer.convert_ids_to_tokens(topk.indices.reshape(-1).tolist())
    topk_ind = np.array(topk_ind).reshape(topk.indices.shape)

    topk_strings_and_probs = np.stack((topk_ind, topk_values.numpy()), axis=-1)
    if topk_diff:
        #add a new bottom row of "N/A" for topk_strings_and_probs because we don't have a "previous" layer to compare to
        topk_strings_and_probs = np.concatenate((np.full((1, topk_strings_and_probs.shape[1], k, 2), "N/A"), topk_strings_and_probs), axis=0)

    return topk_strings_and_probs

def _plot_stream(
    color_stream: ResidualStream,
    top_k_strings_and_probs = None, #TODO np array?,
    x_labels: Sequence[str] = (),
    top_1_strings = None, # TODO np array?
    colorbar_label: str = "",
    layer_stride: int = 1,
    vmax: int = None,
    k: int =5,
    title: str="",
    colorscale: str="rdbu_r",
    rank: int = 0,
):

    color_matrix = np.stack(list(color_stream))[::layer_stride]
    top_1_strings = np.stack(list(top_1_strings))
    y_labels = color_stream.labels()
    x_labels = [x + "\u200c"*i for i, x in enumerate(x_labels)]

    colorbar=dict(
        title=colorbar_label, 
        titleside="right", 
        tickfont=dict(size=20),
    )
    if rank>0: #make log-scale for rank plots
        colorbar["tickvals"] = [0, 1, 2, 3, 4]
        colorbar["ticktext"] = ['1', '10', '100', '1000', '10000']

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=color_matrix,
        x = x_labels,
        text=top_1_strings,
        texttemplate="%{text}",
        textfont={"size":12},
        customdata = top_k_strings_and_probs,
        y=y_labels,
        colorscale=colorscale, #or "rdbu"
        hoverlabel=dict(bgcolor="white", font_size=20, bordercolor="black"),
        hovertemplate = '<br>'.join(f' %{{customdata[{i}][0]}}: %{{customdata[{i}][1]:.1f}}% ' for i in range(k))+ "<extra></extra>",
        colorbar=dict(
            title=colorbar_label, 
            titleside="right", 
            tickfont=dict(size=20),
            ),
        zmax=vmax,
        zmin=0,
    ))


    #TODO Height needs to equal some function of Max(num_layers, topk). Ignore for now. Works until k=18
    fig.update_layout(
        title_text=title, 
        title_x=0.5, 
        title_font_size=30, 
        width=200+80*len(x_labels), 
        xaxis_title = "Token Sequence Input",
        yaxis_title = "Layer",
        xaxis_title_font_size=24, yaxis_title_font_size=24,
    )
    
    fig.show()



@th.no_grad()
def plot_residuals(
    model_or_name: Union[PreTrainedModel, str],
    *,
    input_ids: Optional[th.Tensor] = None,
    start_pos: int = 0,
    sublayers: bool = False,
    text: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    """Plot the residuals."""
    model, tokens, _, stream = _run_inference(
        model_or_name, input_ids, text, tokenizer, sublayers, start_pos
    )

    E = model.get_output_embeddings()
    ln = get_final_layer_norm(model.base_model)
    assert isinstance(ln, th.nn.LayerNorm)

    prob_diffs = stream.map(lambda x: E(ln(x)).softmax(-1)).residuals()
    changed_ids = prob_diffs.map(lambda x: x.abs().argmax(-1))
    changed_tokens = changed_ids.map(
        tokenizer.convert_ids_to_tokens  # type: ignore[arg-type]
    )
    biggest_diffs = prob_diffs.zip_map(lambda x, y: x.gather(-1, y), changed_ids)

    _plot_stream(biggest_diffs, changed_tokens, tokens)
    plt.title("Residuals")


@th.no_grad()
def plot_residual_norms(
    model_or_name: Union[PreTrainedModel, str],
    *,
    input_ids: Optional[th.Tensor] = None,
    start_pos: int = 0,
    sublayers: bool = False,
    text: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    """Plot the residual L2 norms."""
    _, tokens, _, stream = _run_inference(
        model_or_name, input_ids, text, tokenizer, sublayers, start_pos
    )
    residual_norms = stream.residuals().map(lambda x: x.norm(dim=-1).squeeze().cpu())

    _plot_stream(residual_norms, residual_norms, tokens)
    plt.title("Residual L2 norms")


@th.no_grad()
def plot_similarity(
    model_or_name: Union[PreTrainedModel, str],
    *,
    input_ids: Optional[th.Tensor] = None,
    start_pos: int = 0,
    sublayers: bool = False,
    text: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    """Plot the cosine similarities of hidden states with the final state."""
    _, tokens, _, stream = _run_inference(
        model_or_name, input_ids, text, tokenizer, sublayers, start_pos
    )
    similarities = stream.map(
        lambda x: F.cosine_similarity(x, stream.layers[-1], dim=-1).squeeze().cpu()
    )
    _plot_stream(similarities, similarities, tokens)
    plt.title("Cosine similarity")


def _run_inference(
    model_or_name: Union[PreTrainedModel, str],
    input_ids: Optional[th.Tensor] = None,
    text: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    sublayers: bool = False,
    start_pos: int = 0,
    end_pos: Optional[int] = None,
) -> tuple:
    if isinstance(model_or_name, PreTrainedModel):
        model = model_or_name
    elif isinstance(model_or_name, str):
        model = AutoModelForCausalLM.from_pretrained(model_or_name)
    else:
        raise ValueError("model_or_name must be a model or a model name")

    # We always need a tokenizer, even if we're provided with input_ids,
    # because we need to decode the IDs to get labels for the heatmap
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)

    if text is not None:
        input_ids = cast(th.Tensor, tokenizer.encode(text, return_tensors="pt"))
    elif input_ids is None:
        raise ValueError("Either text or input_ids must be provided")

    model_device = next(model.parameters()).device
    with record_residual_stream(model, sublayers=sublayers) as stream:
        outputs = model(input_ids.to(model_device))

    tokens = tokenizer.convert_ids_to_tokens(  # type: ignore[arg-type]
        input_ids.squeeze().tolist()
    )

    if start_pos > 0:
        outputs.logits = outputs.logits[..., start_pos:end_pos, :]
        stream = stream.map(lambda x: x[..., start_pos:end_pos, :])
        tokens = tokens[start_pos:end_pos]

    return model, tokens, outputs, stream

