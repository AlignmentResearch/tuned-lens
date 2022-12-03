from .model_surgery import get_transformer_layers
from .nn import Decoder, TunedLens
from .stats import kl_divergence
from .utils import pytree_map
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from tqdm.auto import trange
from transformers import PreTrainedModel
from typing import Callable, Optional, Sequence
import torch as th
import warnings


@dataclass
class InterventionResult:
    effect_sizes: th.Tensor
    surprisals: th.Tensor

    def to_pandas(self):
        """Convert results into a Pandas dataframe suitable for Plotly."""
        import pandas as pd

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
    decoder: Optional[Decoder] = None,
    divergence: Callable[[th.Tensor, th.Tensor], th.Tensor] = kl_divergence,
    dose: float = 1.0,
    lens: Optional[TunedLens] = None,
    noise_distributions: Sequence[th.distributions.Distribution] = (),
):
    assert token_ids.ndim == 2
    (B, S), L = token_ids.shape, model.config.num_hidden_layers
    if S // B > 128:
        warnings.warn("We recommend a larger batch size for better performance.")

    device = model.device
    token_ids = token_ids.to(device)

    # First do a clean forward pass on all the data. We save the final layer logits,
    # as well as the keys and values to speed up inference when we do interventions.
    control = model(token_ids, output_hidden_states=True, use_cache=True)

    decoder = decoder or Decoder(model)
    control_hs = control["hidden_states"][1:-1]
    assert len(control_hs) == L - 1

    if intervention:
        treatments = intervention(control_hs)  # type: ignore
    elif noise_distributions:
        treatments = [
            h + d.sample(h.shape[:-1]) * dose
            for h, d in zip(control_hs, noise_distributions)
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

    effect_sizes = th.zeros(B, S, L - 1, L, device=device)
    surprisals = th.zeros(B, S, L - 1, L, device=device)

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
            effect_sizes[:, token_idx, i, i] = divergence(
                control_logits, response_logits_i
            )
            diff = response_logits_i - control_logits

            # Record the response from every layer j > i
            for j, response in enumerate(responses[1:], start=i + 1):
                control_logits = decoder(control_hs[j][:, token_idx])
                response_logits = decoder(response.squeeze(1))

                effect_sizes[:, token_idx, i, j] = divergence(
                    control_logits, response_logits
                )
                surprisals[:, token_idx, i, j] = divergence(
                    response_logits, control_logits + diff
                )

            # Record the response from the final layer
            control_logits = control.logits[:, token_idx]
            response_logits = treated.logits.squeeze(1)
            effect_sizes[:, token_idx, i, -1] = divergence(
                control_logits, response_logits
            )
            surprisals[:, token_idx, i, -1] = divergence(
                response_logits, control_logits + diff
            )

    return InterventionResult(effect_sizes, surprisals)


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
