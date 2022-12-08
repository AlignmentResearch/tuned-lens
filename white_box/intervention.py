from .model_surgery import get_transformer_layers
from .nn import Decoder, TunedLens
from .stats import aitchison_similarity, kl_divergence, sample_neighbors
from .utils import pytree_map
from contextlib import contextmanager
from dataclasses import dataclass
from tqdm.auto import trange
from transformers import PreTrainedModel
from typing import Callable, Optional, TYPE_CHECKING
import torch as th
import warnings

if TYPE_CHECKING:
    # Avoid importing Pandas and Plotly if we don't need to
    import pandas as pd
    import plotly.graph_objects as go


@dataclass
class InterventionResult:
    effect_alignments: th.Tensor
    effect_sizes: th.Tensor
    surprisals: th.Tensor

    loss_diffs: th.Tensor
    stimulus_norms: th.Tensor

    def line(self) -> "go.Figure":
        import plotly.graph_objects as go

        _, S, L = self.stimulus_norms.shape
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

        *_, L = self.stimulus_norms.shape

        cmax = self.effect_sizes.quantile(0.95).item()
        fig = go.Figure(
            [
                go.Scatter(
                    x=self.stimulus_norms[..., layer].flatten().cpu(),
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

    def to_pandas(self, agg: bool = False) -> "pd.DataFrame":
        """Convert results into a Pandas dataframe suitable for Plotly."""
        import pandas as pd

        B, S, L, _ = self.effect_sizes.shape
        if not agg:
            a, b, c = th.meshgrid(
                th.arange(B), th.arange(S), th.arange(L), indexing="ij"
            )
            return pd.DataFrame(
                {
                    "sample_index": a.flatten(),
                    "token_index": b.flatten(),
                    "stimulus_layer": c.flatten(),
                    "sr_alignment": self.effect_alignments[..., -1].flatten().cpu(),
                    "stimulus_norm": self.stimulus_norms.flatten().cpu(),
                    "stimulus_size": th.linalg.diagonal(self.effect_sizes)
                    .flatten()
                    .cpu(),
                    "effect_size": self.effect_sizes[..., -1].flatten().cpu(),
                    "loss_diff": self.loss_diffs.flatten().cpu(),
                    "surprisal": self.surprisals[..., -1].flatten().cpu(),
                }
            )

        alignment_var, alignment_mean = th.var_mean(
            self.effect_alignments[..., -1], dim=(0, 1)
        )
        effect_var, effect_mean = th.var_mean(self.effect_sizes[..., -1], dim=(0, 1))
        surprisal_var, surprisal_mean = th.var_mean(
            self.surprisals[..., -1], dim=(0, 1)
        )
        return pd.DataFrame(
            {
                "stimulus_layer": th.arange(L),
                "sr_alignment": alignment_mean.cpu(),
                "sr_alignment_stderr": alignment_var.div(B * S).sqrt().cpu(),
                "effect_size": effect_mean.cpu(),
                "effect_size_stderr": effect_var.div(B * S).sqrt().cpu(),
                "surprisal": surprisal_mean.cpu(),
                "surprisal_stderr": surprisal_var.div(B * S).sqrt().cpu(),
            }
        )


@th.autocast("cuda", enabled=th.cuda.is_available())
@th.no_grad()
def estimate_effects(
    model: PreTrainedModel,
    token_ids: th.Tensor,
    *,
    decoder: Optional[Decoder] = None,
    divergence: Callable[[th.Tensor, th.Tensor], th.Tensor] = kl_divergence,
    lens: Optional[TunedLens] = None,
    seed: int = 42,
    tau: float = th.inf,
):
    """Estimate the expected causal effect of random resampling on model outputs."""
    assert token_ids.ndim == 2
    (B, S), L = token_ids.shape, model.config.num_hidden_layers
    if S // B > 128:
        warnings.warn("We recommend a larger batch size for better performance.")

    device = model.device
    token_ids = token_ids.to(device)

    rng = th.Generator(device=device)
    rng.manual_seed(seed)

    # First do a clean forward pass on all the data. We save the final layer logits,
    # as well as the keys and values to speed up inference when we do interventions.
    control = model(token_ids, output_hidden_states=True, use_cache=True)
    c_inputs = control["hidden_states"][:-1]  # Drop final layer
    c_resids = [h_ - h for h, h_ in zip(c_inputs[:-1], c_inputs[1:])]

    decoder = decoder or Decoder(model)
    c_outputs = c_inputs[1:]  # Drop input embeddings
    assert len(c_outputs) == L - 1

    if lens:
        assert len(lens) == L

        # Putting the lens in eval mode is important because it may have dropout
        lens = lens.to(device).eval()
        c_outputs = [
            lens.transform_hidden(h, i) for i, h in enumerate(c_outputs, start=1)
        ]

    effect_alignments = th.zeros(B, S, L - 1, L, device=device)
    effect_sizes = th.zeros(B, S, L - 1, L, device=device)
    surprisals = th.zeros(B, S, L - 1, L, device=device)

    loss_diffs = th.zeros(B, S, L - 1, device=device)
    stimulus_norms = th.zeros(B, S, L - 1, device=device)

    for token_idx in trange(1, S, desc="Applying", unit="token"):
        left_ctx = pytree_map(lambda x: x[..., :token_idx, :], control.past_key_values)
        new_tokens = token_ids[:, token_idx, None]

        # Sequentially intervene on each layer
        for i, (c_in, c_res, c_out) in enumerate(zip(c_inputs, c_resids, c_outputs)):
            c_logits = decoder(c_out[:, token_idx])
            indices = sample_neighbors(c_logits, tau, generator=rng)
            treated_out = c_in[:, token_idx] + c_res[indices, token_idx]
            stimulus_norms[:, token_idx, i] = th.norm(
                treated_out - c_out[:, token_idx], dim=-1
            )

            with layer_intervention(model, [i], lambda _: treated_out):
                treated = model(
                    new_tokens, output_hidden_states=True, past_key_values=left_ctx
                )

            responses = treated.hidden_states[i + 1 : -1]
            if lens:
                responses = [
                    lens.transform_hidden(h, i)
                    for i, h in enumerate(responses, start=i + 1)
                ]

            weights = control.logits[:, token_idx].softmax(-1)

            # Record response from layer i
            response_logits_i = decoder(responses[0].squeeze(1))
            effect_alignments[:, token_idx, i, i] = 1
            effect_sizes[:, token_idx, i, i] = divergence(c_logits, response_logits_i)
            # Aitchison difference between the response and control logits
            diff_i = th.log_softmax(response_logits_i - c_logits, dim=-1)

            # Record the response from every layer j > i
            for j, response in enumerate(responses[1:], start=i + 1):
                c_logits = decoder(c_inputs[j][:, token_idx])
                response_logits = decoder(response.squeeze(1))

                # Aitchison difference
                diff_j = th.log_softmax(response_logits - c_logits, dim=-1)
                alignments = aitchison_similarity(diff_i, diff_j, weight=weights)
                effect_alignments[:, token_idx, i, j] = alignments
                effect_sizes[:, token_idx, i, j] = divergence(c_logits, response_logits)
                surprisals[:, token_idx, i, j] = divergence(
                    response_logits, c_logits + diff_i
                )

            # Record the response from the final layer
            c_logits = control.logits[:, token_idx]
            response_logits = treated.logits.squeeze(1)
            effect_sizes[:, token_idx, i, -1] = divergence(c_logits, response_logits)
            surprisals[:, token_idx, i, -1] = divergence(
                response_logits, response_logits_i
            )
            if token_idx < S - 1:
                treated_loss = th.nn.functional.cross_entropy(
                    response_logits, token_ids[:, token_idx + 1], reduction="none"
                )
                control_loss = th.nn.functional.cross_entropy(
                    c_logits, token_ids[:, token_idx + 1], reduction="none"
                )
                loss_diffs[:, token_idx, i] = treated_loss - control_loss

            # Aitchison difference
            diff_j = th.log_softmax(response_logits - c_logits, dim=-1)
            alignments = aitchison_similarity(diff_i, diff_j, weight=weights)
            effect_alignments[:, token_idx, i, -1] = alignments

    return InterventionResult(
        effect_alignments,
        effect_sizes,
        surprisals,
        loss_diffs,
        stimulus_norms,
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
