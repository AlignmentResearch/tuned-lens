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
    top_k: int = 0,
    top_k_diff: bool = False,
    whitespace_token: str = "Ġ",
    whitespace_replacement: str = " ",
    add_last_tuned_lens_layer: bool = False,
):
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

    reversed_colors = False #Only true for probs
    if rank > 0:
        #token_offset is the number of tokens predicted ahead. 1 means the next token, 2 means the token after that, etc.
        token_offset = rank
        #Skip the first token because not predicted
        next_token_ids = input_ids[:, token_offset:] 
        #Remove the last token seq from hidden_lps because it is not predicted
        hidden_lps = hidden_lps.map(lambda x: x[:, :-token_offset, :])
        #Get the sorted indices of the logits for each layer of hidden_lps
        sorted = hidden_lps.map(lambda x: x.argsort(dim=-1, descending=True))
        #Get the rank of the ground-truth next token for each layer
        ranks = sorted.map(lambda x: (next_token_ids.unsqueeze(-1) == x).nonzero()[:, -1])
        #Convert hidden_lps to probabilities
        hidden_lps = hidden_lps.map(lambda x: x.exp()) 
        #Get the prob of the ground-truth next token for each layer
        stats = hidden_lps.map(lambda x: x.gather(-1, next_token_ids.unsqueeze(-1))) 
        
        top_strings = ranks #TODO change to "display" variable instead of top_strings
        max_color = 1.0
        metric = "probs"
        reversed_colors = True
    else:

        top_tokens = hidden_lps.map(lambda x: x.argmax(-1).squeeze().cpu().tolist())
        top_strings = top_tokens.map(
            tokenizer.convert_ids_to_tokens  # type: ignore[arg-type]
        )
        top_strings = top_strings.map(
            lambda x: [t.replace(whitespace_token, whitespace_replacement) for t in x]
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

    if(top_k > 0): #TODO make title tuned_lens
        probs = hidden_lps.map(lambda x: x.exp()*100)
        return plot_topk_probs(
            stats.map(lambda x: x.squeeze().cpu()),
            probs,
            [t.replace(whitespace_token, whitespace_replacement) for t in tokens],
            tokenizer=tokenizer,
            colorbar_label=f"{metric} (nats)",
            fmt="",
            layer_stride=layer_stride,
            vmax=max_color,
            k=top_k,
            top_k_diff=top_k_diff,
            whitespace_token = whitespace_token,
            whitespace_replacement=whitespace_replacement,
        )
    else:
        _plot_stream(
            stats.map(lambda x: x.squeeze().cpu()),
            top_strings,
            [t.replace(whitespace_token, whitespace_replacement) for t in tokens],
            colorbar_label=f"{metric} (nats)",
            fmt="",
            layer_stride=layer_stride,
            vmax=max_color,
            reversed_colors=reversed_colors,
        )

        name = "Logit" if tuned_lens is None else "Tuned"
        plt.title(f"{name} lens ({model.name_or_path})")

def plot_topk_probs(
    color_stream: ResidualStream,
    text_stream: ResidualStream,
    x_labels: Sequence[str] = (),
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    colorbar_label: str = "",
    fmt: str = "0.2f",
    layer_stride: int = 1,
    vmax=None,
    k=5,
    top_k_diff=False,
    whitespace_token: str = "Ġ",
    whitespace_replacement: str = " ",

):
    if(top_k_diff):
        top_tokens = text_stream.map(lambda x: x.argmax(-1).squeeze().cpu().tolist())
        top_1_strings = top_tokens.map(
            tokenizer.convert_ids_to_tokens  # type: ignore[arg-type]
        )
        top_1_strings = np.stack(list(top_1_strings))[::layer_stride]
        text_stream = text_stream.pairwise_map(lambda x, y: y - x)

    color_matrix = np.stack(list(color_stream))[::layer_stride]
    probs = th.stack(list(text_stream))[::layer_stride].squeeze(1)
    y_labels = color_stream.labels()
    x_labels = [x + "\u200c"*i for i, x in enumerate(x_labels)]

    if(top_k_diff):
        topk = probs.abs().topk(k, dim=-1)
        topk_values = probs.gather(-1, topk.indices) #get the topk values but include negative values
        #TODO custom hovertemplate for green text up and red text down
    else:
        #Get the top-k tokens & probabilities for each
        topk = probs.topk(k, dim=-1)
        topk_values = topk.values
    #reshape top_k_ind from (layers, seq, k) to (layers*seq*k), convert_ids_to_tokens, then reshape back to (layers, seq, k)
    top_k_ind = tokenizer.convert_ids_to_tokens(topk.indices.reshape(-1).tolist())
    top_k_ind = np.array(top_k_ind).reshape(topk.indices.shape)
    # top_k_ind = [t.replace(whitespace_token, whitespace_replacement) for t in top_k_ind]

#Grab the top 1 strings for displaying on Z axis
    if not top_k_diff:
        top_1_strings = top_k_ind[:,:,0]
    
    #replace text for top_k_ind & top_1_strings
    newline_token = "Ċ"
    top_k_ind = np.array([[[t.replace(whitespace_token, whitespace_replacement) if whitespace_token in t else t.replace(newline_token, "\\n") if newline_token in t else t for t in x] for x in y] for y in top_k_ind])
    top_1_strings = np.array([[t.replace(whitespace_token, whitespace_replacement) if whitespace_token in t else t.replace(newline_token, "\\n") if newline_token in t else t for t in x] for x in top_1_strings])
    
    top_k_strings_and_probs = np.stack((top_k_ind, topk_values.numpy()), axis=-1)
    if top_k_diff:
        #add a new bottom row of "N/A" for top_k_strings_and_probs
        top_k_strings_and_probs = np.concatenate((np.full((1, top_k_strings_and_probs.shape[1], k, 2), "N/A"), top_k_strings_and_probs), axis=0)
    #return all the variables used in the figure below
    # return color_matrix, top_k_strings_and_probs, top_1_strings, y_labels, x_labels, colorbar_label, fmt, layer_stride, vmax, k
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=color_matrix,
        x = x_labels,
        text=top_1_strings,
        texttemplate="%{text}",
        textfont={"size":12},
        customdata = top_k_strings_and_probs,
        y=y_labels,
        colorscale='rdbu_r', #or "rdbu"
        hoverlabel=dict(bgcolor="white", font_size=20, bordercolor="black"),
        hovertemplate = '<br>'.join(f' %{{customdata[{i}][0]}}: %{{customdata[{i}][1]:.1f}}% ' for i in range(k))+ "<extra></extra>",
        colorbar=dict(title=colorbar_label, titleside="right", tickfont=dict(size=20)),
        zmax=vmax,
        zmin=0,
    ))
    fig.update_layout(title_text=f'Top {k} Logit Probabilities for Each Token', title_x=0.5, title_font_size=30, width=200+80*len(x_labels))
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


def _plot_stream(
    color_stream: ResidualStream,
    text_stream: ResidualStream,
    x_labels: Sequence[str] = (),
    colorbar_label: str = "",
    fmt: str = "0.2f",
    layer_stride: int = 1,
    vmax=None,
    reversed_colors: bool = False,

):
    color_matrix = np.stack(list(color_stream))[::layer_stride]
    text_matrix = np.stack(list(text_stream))[::layer_stride]
    y_labels = color_stream.labels()

    plt.subplots(figsize=(2 * len(x_labels), len(color_stream) // (2 * layer_stride)))
    sns.heatmap(
        np.flipud(color_matrix),
        annot=np.flipud(text_matrix),
        cbar_kws={"label": colorbar_label} if colorbar_label else None,
        cmap=sns.color_palette("coolwarm_r" if reversed_colors else "coolwarm", as_cmap=True),
        fmt=fmt,
        robust=True,
        vmax=vmax,
        xticklabels=x_labels,  # type: ignore[arg-type]
        yticklabels=y_labels[::-layer_stride],  # type: ignore[arg-type]
    )
