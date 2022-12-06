from .model_surgery import get_transformer_layers
from .nn import Decoder, TunedLens
from .stats import aitchison_similarity, kl_divergence
from .utils import pytree_map
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from tqdm.auto import trange
from transformers import PreTrainedModel
from typing import Callable, Optional, Sequence, TYPE_CHECKING
import torch as th
import warnings

if TYPE_CHECKING:
    # Avoid importing Pandas and Plotly if we don't need to
    import pandas as pd
    import plotly.graph_objects as go


@dataclass
class InterventionResult:
    doses: th.Tensor
    effect_alignments: th.Tensor
    effect_sizes: th.Tensor
    surprisals: th.Tensor

    most_changed_ids: th.Tensor
    most_changed_diffs: th.Tensor

    def line(self) -> "go.Figure":
        import plotly.graph_objects as go

        _, S, L = self.doses.shape
        _, means = th.var_mean(self.effect_alignments[..., -1], dim=0)

        fig = go.Figure(
            [
                go.Scatter(
                    x=th.arange(L),
                    y=means[i].cpu() if i else means.mean(0).cpu(),
                    mode="lines+markers",
                    name=f"Token {i}" if i else "All tokens",
                    visible=i == 0,
                )
                for i in range(S)
            ]
        )
        fig.update_layout(
            sliders=[
                dict(
                    currentvalue=dict(prefix="Token index: "),
                    steps=[
                        dict(
                            args=[dict(visible=visible_mask)],
                            label=str(i) if i else "all",
                            method="restyle",
                        )
                        for i, visible_mask in enumerate(th.eye(S).bool())
                    ],
                )
            ],
            title="Stimulus-response alignment by layer",
        )
        fig.update_xaxes(title="Stimulus layer")
        fig.update_yaxes(
            range=[min(0, means.min().item()), 1], title="Aitchison similarity"
        )
        return fig

    def scatter(self) -> "go.Figure":
        import plotly.graph_objects as go

        *_, L = self.doses.shape

        cmax = self.effect_sizes.quantile(0.95).item()
        fig = go.Figure(
            [
                go.Scatter(
                    x=self.doses[..., layer].flatten().cpu(),
                    y=self.effect_alignments[..., layer, -1].flatten().cpu(),
                    marker=dict(
                        cmin=0,
                        cmax=cmax,
                        color=self.effect_sizes[..., layer, -1].flatten().cpu(),
                        colorbar=dict(
                            title_side="right",
                            title="Effect size (nats)",
                        ),
                        colorscale="Viridis",
                        opacity=0.125,
                    ),
                    mode="markers",
                    name=f"Layer {layer}",
                    visible=layer == 0,
                )
                for layer in range(L)
            ]
        )
        fig.update_layout(
            sliders=[
                dict(
                    currentvalue=dict(prefix="Stimulus layer: "),
                    steps=[
                        dict(
                            args=[dict(visible=visible_mask)],
                            label=str(i),
                            method="update",
                        )
                        for i, visible_mask in enumerate(th.eye(L).bool())
                    ],
                )
            ],
            title="Stimulus-response alignment vs. dose",
        )
        fig.update_xaxes(title="Dose")
        fig.update_yaxes(range=[-0.4, 1], title="Aitchison similarity")
        return fig

    def to_pandas(self) -> "pd.DataFrame":
        """Convert results into a Pandas dataframe suitable for Plotly."""
        import pandas as pd

        alignment_var, alignment_mean = th.var_mean(
            self.effect_alignments.mean(1), dim=0
        )
        effect_var, effect_mean = th.var_mean(self.effect_sizes.mean(1), dim=0)
        surprisal_var, surprisal_mean = th.var_mean(self.surprisals.mean(1), dim=0)

        B, *_, L = self.effect_sizes.shape
        records = []

        for i, j in product(range(L - 1), range(L)):
            if j < i:
                continue

            records.append(
                {
                    "stimulus_layer": i,
                    "response_layer": j,
                    "alignment": alignment_mean[i, j].item(),
                    "alignment_stderr": alignment_var[i, j].div(B).sqrt().item(),
                    "effect_size": effect_mean[i, j].item(),
                    "effect_size_stderr": effect_var[i, j].div(B).sqrt().item(),
                    "surprisal": surprisal_mean[i, j].item(),
                    "surprisal_stderr": surprisal_var[i, j].div(B).sqrt().item(),
                }
            )

        return pd.DataFrame.from_records(records)


@th.autocast("cuda", enabled=th.cuda.is_available())
@th.no_grad()
def apply_intervention(
    model: PreTrainedModel,
    token_ids: th.Tensor,
    intervention: Optional[Callable[[list[th.Tensor]], list[th.Tensor]]] = None,
    *,
    aitchison_weights: Optional[th.Tensor] = None,
    decoder: Optional[Decoder] = None,
    divergence: Callable[[th.Tensor, th.Tensor], th.Tensor] = kl_divergence,
    dose_range: tuple[float, float] = (1.0, 1.0),
    lens: Optional[TunedLens] = None,
    noise_distributions: Sequence[th.distributions.Distribution] = (),
    seed: int = 42,
):
    assert token_ids.ndim == 2
    (B, S), L = token_ids.shape, model.config.num_hidden_layers
    if S // B > 128:
        warnings.warn("We recommend a larger batch size for better performance.")

    # This is sort of a hack to get around the fact that torch.distributions doesn't
    # support custom Generator objects. We'll just use the global RNG for now.
    initial_seed = th.initial_seed()
    th.manual_seed(seed)

    device = model.device
    token_ids = token_ids.to(device)

    # First do a clean forward pass on all the data. We save the final layer logits,
    # as well as the keys and values to speed up inference when we do interventions.
    control = model(token_ids, output_hidden_states=True, use_cache=True)

    decoder = decoder or Decoder(model)
    control_hs = control["hidden_states"][1:-1]
    assert len(control_hs) == L - 1

    # Randomly sample doses for each (token, layer) combination. This means we can't
    # directly compare responses across tokens and layers for a particular sample,
    # but we mostly don't care about that anyway. With random doses, we can compute a
    # low-variance estimate of the effect size for *any* dose on the specified interval
    # by interpolating between nearby datapoints.
    dose_min, dose_max = dose_range
    assert 0 <= dose_min <= dose_max
    doses = th.empty(B, S, L - 1, device=device).uniform_(dose_min, dose_max)

    if intervention:
        treatments = intervention(control_hs)  # type: ignore
    elif noise_distributions:
        treatments = [
            h + d.sample(h.shape[:-1]) * dose[..., None]
            for h, dose, d in zip(control_hs, doses.unbind(-1), noise_distributions)
        ]
    else:
        raise ValueError("Must provide either an intervention or noise distributions.")

    if lens:
        assert len(lens) == L

        # Putting the lens in eval mode is important because it may have dropout
        lens = lens.to(device).eval()
        control_hs = [
            lens.transform_hidden(h, i) for i, h in enumerate(control_hs, start=1)
        ]

    effect_alignments = th.zeros(B, S, L - 1, L, device=device)
    effect_sizes = th.zeros(B, S, L - 1, L, device=device)
    surprisals = th.zeros(B, S, L - 1, L, device=device)

    most_changed_diffs = th.zeros(B, S, L - 1, device=device)
    most_changed_ids = th.zeros(B, S, L - 1, dtype=th.long, device=device)

    for token_idx in trange(1, S, desc="Applying", unit="token"):
        left_ctx = pytree_map(lambda x: x[..., :token_idx, :], control.past_key_values)
        new_tokens = token_ids[:, token_idx, None]

        # Sequentially intervene on each layer
        for i, treatment in enumerate(treatments):
            with layer_intervention(model, [i], lambda _: treatment[:, token_idx]):
                treated = model(
                    new_tokens, output_hidden_states=True, past_key_values=left_ctx
                )

            responses = treated.hidden_states[i + 1 : -1]
            if lens:
                responses = [
                    lens.transform_hidden(h, i)
                    for i, h in enumerate(responses, start=i + 1)
                ]

            # Record response from layer i
            control_logits = decoder(control_hs[i][:, token_idx])
            response_logits_i = decoder(responses[0].squeeze(1))
            effect_alignments[:, token_idx, i, i] = 1
            effect_sizes[:, token_idx, i, i] = divergence(
                control_logits, response_logits_i
            )
            # Aitchison difference between the response and control logits
            diff_i = th.log_softmax(response_logits_i - control_logits, dim=-1)

            # Record the response from every layer j > i
            for j, response in enumerate(responses[1:], start=i + 1):
                control_logits = decoder(control_hs[j][:, token_idx])
                response_logits = decoder(response.squeeze(1))

                # Aitchison difference
                diff_j = th.log_softmax(response_logits - control_logits, dim=-1)
                alignments = aitchison_similarity(
                    diff_i, diff_j, weight=aitchison_weights
                )
                effect_alignments[:, token_idx, i, j] = alignments
                effect_sizes[:, token_idx, i, j] = divergence(
                    control_logits, response_logits
                )
                surprisals[:, token_idx, i, j] = divergence(
                    response_logits, control_logits + diff_i
                )

            # Record the response from the final layer
            control_logits = control.logits[:, token_idx]
            response_logits = treated.logits.squeeze(1)
            effect_sizes[:, token_idx, i, -1] = divergence(
                control_logits, response_logits
            )
            surprisals[:, token_idx, i, -1] = divergence(
                response_logits, control_logits + diff_i
            )

            # Record the most changed token
            prob_diffs = response_logits.softmax(-1) - control_logits.softmax(-1)
            most_changed = prob_diffs.abs().argmax(-1)
            largest_diff = prob_diffs.gather(-1, most_changed[..., None])
            most_changed_ids[:, token_idx, i] = most_changed
            most_changed_diffs[:, token_idx, i] = largest_diff.squeeze(-1)

            # Aitchison difference
            diff_j = th.log_softmax(response_logits - control_logits, dim=-1)
            alignments = aitchison_similarity(diff_i, diff_j, weight=aitchison_weights)
            effect_alignments[:, token_idx, i, -1] = alignments

    th.manual_seed(initial_seed)
    return InterventionResult(
        doses,
        effect_alignments,
        effect_sizes,
        surprisals,
        most_changed_ids,
        most_changed_diffs,
    )


@contextmanager
def layer_intervention(
    model: th.nn.Module,
    layer_indices: list[int],
    intervention: Callable,
    *,
    token_idx: int = -1,
):
    """Modify the output of a transformer layer during the forward pass."""
    _, layers = get_transformer_layers(model)

    def wrapper(_, __, outputs):
        y, *extras = outputs
        y[..., token_idx, :] = intervention(y[..., token_idx, :])
        return y, *extras

    hooks = []
    for i in layer_indices:
        hooks.append(layers[i].register_forward_hook(wrapper))  # type: ignore[arg-type]

    try:
        yield model
    finally:
        for hook in hooks:
            hook.remove()


def swap_topk(logits: th.Tensor, k: int = 2):
    """Reverse the order of the top-k logits."""
    top_logits, top_indices = logits.topk(k)

    swapped = logits.clone()
    swapped[..., top_indices] = top_logits.flip([-1])
    return swapped
