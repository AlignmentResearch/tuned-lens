from .residual_stream import ResidualStream, record_residual_stream
from .tuned_lens import TunedLens
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
import torch as th
import torch.nn.functional as F


@th.no_grad()
def plot_logit_lens(
    model_or_name: Union[PreTrainedModel, str],
    *,
    input_ids: Optional[th.Tensor] = None,
    metric: Literal["ce", "entropy", "kl"] = "entropy",
    residual_means: Optional[ResidualStream] = None,
    text: Optional[str] = None,
    tuned_lens: Optional[TunedLens] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    """Plot the cosine similarities of hidden states with the final state."""
    model, tokens, outputs, stream = _run_inference(
        model_or_name, input_ids, text, tokenizer
    )

    if residual_means is not None:
        acc = th.zeros_like(residual_means.layers[0])
        for state, mean in zip(reversed(stream), reversed(residual_means)):
            state += acc
            acc += mean

    if tuned_lens is not None:
        logits = stream.new_from_list(list(tuned_lens.iter_logits(stream)))
        hidden_lps = logits.map(lambda x: x.log_softmax(dim=-1))
    else:
        E = model.get_output_embeddings()
        ln = model.base_model.ln_f
        assert isinstance(ln, th.nn.LayerNorm)
        hidden_lps = stream.map(lambda x: E(ln(x)).log_softmax(-1))

    top_tokens = hidden_lps.map(lambda x: x.argmax(-1).squeeze().cpu().tolist())
    top_strings = top_tokens.map(tokenizer.convert_ids_to_tokens)  # type: ignore[arg-type]
    if metric == "ce":
        raise NotImplementedError
    elif metric == "kl":
        log_probs = outputs.logits.log_softmax(-1)
        probs = log_probs.exp()
        stats = hidden_lps.map(
            lambda x: th.sum(probs * (log_probs - x), dim=-1).squeeze().cpu()
        )
    elif metric == "entropy":
        stats = hidden_lps.map(lambda x: -th.sum(x.exp() * x, dim=-1).squeeze().cpu())
    else:
        raise ValueError(f"Unknown metric: {metric}")

    uniform_loss = math.log(model.config.vocab_size)
    _plot_stream(stats, top_strings, tokens, vmax=uniform_loss, fmt="")
    plt.title(f"Logit lens ({model.name_or_path})")


@th.no_grad()
def plot_residuals(
    model_or_name: Union[PreTrainedModel, str],
    *,
    input_ids: Optional[th.Tensor] = None,
    text: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    """Plot the residuals."""
    model, tokens, _, stream = _run_inference(model_or_name, input_ids, text, tokenizer)

    E = model.get_output_embeddings()
    ln = model.base_model.ln_f
    assert isinstance(ln, th.nn.LayerNorm)

    prob_diffs = stream.map(lambda x: E(ln(x)).softmax(-1)).residuals()
    changed_ids = prob_diffs.map(lambda x: x.abs().argmax(-1))
    changed_tokens = changed_ids.map(tokenizer.convert_ids_to_tokens)  # type: ignore[arg-type]
    biggest_diffs = prob_diffs.zip_map(changed_ids, lambda x, y: x.gather(-1, y))

    _plot_stream(biggest_diffs, changed_tokens, tokens)
    plt.title("Residuals")


@th.no_grad()
def plot_residual_norms(
    model_or_name: Union[PreTrainedModel, str],
    *,
    input_ids: Optional[th.Tensor] = None,
    text: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    """Plot the residual L2 norms."""
    _, tokens, _, stream = _run_inference(model_or_name, input_ids, text, tokenizer)
    residual_norms = stream.residuals().map(lambda x: x.norm(dim=-1).squeeze().cpu())

    _plot_stream(residual_norms, residual_norms, tokens)
    plt.title("Residual L2 norms")


@th.no_grad()
def plot_similarity(
    model_or_name: Union[PreTrainedModel, str],
    *,
    input_ids: Optional[th.Tensor] = None,
    text: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    """Plot the cosine similarities of hidden states with the final state."""
    _, tokens, _, stream = _run_inference(model_or_name, input_ids, text, tokenizer)
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
        print(input_ids)
    elif input_ids is None:
        raise ValueError("Either text or input_ids must be provided")

    model_device = next(model.parameters()).device
    with record_residual_stream(model) as stream:
        outputs = model(input_ids.to(model_device))

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())  # type: ignore[arg-type]
    return model, tokens, outputs, stream


def _plot_stream(
    color_stream: ResidualStream,
    text_stream: ResidualStream,
    x_labels: Sequence[str] = (),
    fmt: str = "0.2f",
    vmax=None,
):
    color_matrix = np.stack(list(color_stream))
    text_matrix = np.stack(list(text_stream))
    y_labels = color_stream.labels()

    fig, ax = plt.subplots(figsize=(2 * len(x_labels), len(color_stream) // 2))
    ax = sns.heatmap(
        np.flipud(color_matrix),
        annot=np.flipud(text_matrix),
        cmap=sns.color_palette("coolwarm", as_cmap=True),
        fmt=fmt,
        robust=True,
        vmax=vmax,
        xticklabels=x_labels,  # type: ignore[arg-type]
        yticklabels=y_labels[::-1],  # type: ignore[arg-type]
    )
