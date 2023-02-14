from ..model_surgery import get_transformer_layers
from ..nn import Decoder, TunedLens
from ..stats import aitchison_similarity, kl_divergence, sample_neighbors
from ..utils import pytree_map, revcumsum
from .utils import derange
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from tqdm.auto import trange
from transformers import PreTrainedModel
from typing import Callable, Literal, Optional, TYPE_CHECKING
import torch as th
import warnings

if TYPE_CHECKING:
    # Avoid importing Pandas and Plotly if we don't need to
    import pandas as pd
    import plotly.graph_objects as go


@dataclass
class InterventionResult:
    loss_diffs: th.Tensor
    response_sizes: th.Tensor
    stimulus_alignments: th.Tensor
    stimulus_angles: th.Tensor
    stimulus_sizes: th.Tensor
    stimulus_norms: th.Tensor
    surprisals: th.Tensor

    def line(self) -> "go.Figure":
        import plotly.graph_objects as go

        _, S, L = self.stimulus_norms.shape
        _, means = th.var_mean(self.stimulus_alignments, dim=0)

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

        fig = go.Figure(
            [
                go.Scatter(
                    x=self.stimulus_sizes[..., layer].flatten().cpu(),
                    y=self.response_sizes[..., layer].flatten().cpu(),
                    marker=dict(
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
            title="Response KL vs. stimulus KL",
        )
        fig.update_xaxes(title="Stimulus KL (nats)")
        fig.update_yaxes(title="Response KL (nats)")
        return fig

    def to_pandas(self, agg: bool = False) -> "pd.DataFrame":
        """Convert results into a Pandas dataframe suitable for Plotly."""
        import pandas as pd

        B, S, L = self.stimulus_sizes.shape
        if not agg:
            a, b, c = th.meshgrid(
                th.arange(B), th.arange(S), th.arange(L), indexing="ij"
            )
            return pd.DataFrame(
                {
                    "sample_index": a.flatten(),
                    "token_index": b.flatten(),
                    "stimulus_layer": c.flatten(),
                    **{k: v.flatten().cpu() for k, v in asdict(self).items()},
                }
            )

        alignment_var, alignment_mean = th.var_mean(
            self.stimulus_alignments, dim=(0, 1)
        )
        stimulus_var, stimulus_mean = th.var_mean(self.stimulus_sizes, dim=(0, 1))
        surprisal_var, surprisal_mean = th.var_mean(self.surprisals, dim=(0, 1))
        return pd.DataFrame(
            {
                "stimulus_layer": th.arange(L),
                "sr_alignment": alignment_mean.cpu(),
                "sr_alignment_stderr": alignment_var.div(B * S).sqrt().cpu(),
                "stimulus_size": stimulus_mean.cpu(),
                "stimulus_size_stderr": stimulus_var.div(B * S).sqrt().cpu(),
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
    mean_ablate: bool = False,
    mode: Literal["gaussian", "resample", "resample-logit"] = "resample",
    seed: int = 42,
    tau: float = th.inf,
):
    """Estimate the expected causal effect of resampling residuals on model outputs."""
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
    c_inputs = control["hidden_states"]
    c_resids = [h_ - h for h, h_ in zip(c_inputs[:-1], c_inputs[1:])]

    biases = []
    if mean_ablate:
        assert not lens, "Cannot use both mean ablation and a tuned lens."
        biases = revcumsum([r.mean(dim=0) for r in c_resids])

    def transform(x, i, j):
        if mean_ablate:
            return x + biases[i + 1][j]
        elif lens:
            return lens.transform_hidden(x, i + 1)
        else:
            return x

    decoder = decoder or Decoder(model)
    c_outputs = c_inputs[1:-1]  # Drop input embeddings and final layer
    assert len(c_outputs) == L - 1

    if lens:
        assert len(lens) == L
        lens = lens.to(device)

    target_logits = derange(control.logits, generator=rng)
    result = InterventionResult(
        loss_diffs=th.zeros(B, S, L - 1, device=device),
        response_sizes=th.zeros(B, S, L - 1, device=device),
        stimulus_alignments=th.zeros(B, S, L - 1, device=device),
        stimulus_sizes=th.zeros(B, S, L - 1, device=device),
        stimulus_angles=th.zeros(B, S, L - 1, device=device),
        stimulus_norms=th.zeros(B, S, L - 1, device=device),
        surprisals=th.zeros(B, S, L - 1, device=device),
    )

    for token_idx in trange(1, S, desc="Applying", unit="token"):
        left_ctx = pytree_map(lambda x: x[..., :token_idx, :], control.past_key_values)
        new_tokens = token_ids[:, token_idx, None]

        # Sequentially intervene on each layer
        for i, ctrl in enumerate(zip(c_inputs, c_resids, c_outputs)):
            c_in, c_res, c_out = map(lambda x: x[:, token_idx], ctrl)
            c_logits_i = decoder(transform(c_out, i, token_idx))

            if mode == "gaussian":
                treated_out = c_out + th.randn_like(c_out)
            elif mode == "resample":
                indices = sample_neighbors(c_logits_i, tau, generator=rng)
                treated_out = c_in + c_res[indices]
            elif mode == "resample-logit":
                with th.autocast("cuda", enabled=False):
                    treated_out = decoder.invert(
                        target_logits[:, token_idx].float(),
                        h0=c_out.float(),
                        max_iter=100,
                        transform=lambda x: transform(x, i, token_idx),
                    ).preimage
            else:
                raise ValueError(f"Unknown mode: {mode}")

            with layer_intervention(model, [i], lambda _: treated_out):
                treated = model(
                    new_tokens, output_hidden_states=True, past_key_values=left_ctx
                )

            # Aitchison difference between the treated and control logits
            treated_logits_i = decoder(transform(treated_out, i, token_idx))
            stimulus = th.log_softmax(treated_logits_i - c_logits_i, dim=-1)
            result.stimulus_sizes[:, token_idx, i] = divergence(
                c_logits_i, treated_logits_i
            )
            result.stimulus_angles[
                :, token_idx, i
            ] = th.nn.functional.cosine_similarity(treated_out, c_out, dim=-1).acos()
            result.stimulus_norms[:, token_idx, i] = th.norm(
                treated_out - c_out, dim=-1
            )

            # Record the response from the final layer
            c_logits_f = control.logits[:, token_idx]
            treated_logits_f = treated.logits.squeeze(1)
            result.surprisals[:, token_idx, i] = divergence(
                treated_logits_f, treated_logits_i
            )
            if token_idx < S - 1:
                treated_loss = th.nn.functional.cross_entropy(
                    treated_logits_f, token_ids[:, token_idx + 1], reduction="none"
                )
                control_loss = th.nn.functional.cross_entropy(
                    c_logits_f, token_ids[:, token_idx + 1], reduction="none"
                )
                result.loss_diffs[:, token_idx, i] = treated_loss - control_loss

            weights = 0.5 * (
                control.logits[:, token_idx].softmax(-1) + treated_logits_f.softmax(-1)
            )

            # Aitchison difference
            response = th.log_softmax(treated_logits_f - c_logits_f, dim=-1)
            result.response_sizes[:, token_idx, i] = divergence(
                c_logits_f, treated_logits_f
            )
            result.stimulus_alignments[:, token_idx, i] = aitchison_similarity(
                stimulus, response, weight=weights
            )

    return result


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
