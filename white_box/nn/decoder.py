from copy import deepcopy
from transformers import PreTrainedModel
from typing import cast, Optional
from ..model_surgery import get_final_layer_norm
import torch as th


class Decoder(th.nn.Module):
    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        *,
        d_model: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()

        # Initializing from scratch without a model
        if not model:
            assert d_model and vocab_size
            self.layer_norm = th.nn.LayerNorm(d_model)
            self.unembedding = th.nn.Linear(d_model, vocab_size, bias=False)

        # Use HuggingFace methods to get decoder layers
        else:
            assert not d_model and not vocab_size
            d_model = model.config.hidden_size
            vocab_size = model.config.vocab_size
            assert isinstance(d_model, int) and isinstance(vocab_size, int)

            # Currently we convert the decoder to full precision
            self.unembedding = deepcopy(model.get_output_embeddings()).float()
            if ln := get_final_layer_norm(model):
                self.layer_norm = deepcopy(ln).float()
            else:
                self.layer_norm = th.nn.Identity()

        # In general we don't want to finetune the decoder
        self.requires_grad_(False)

    def forward(self, h: th.Tensor) -> th.Tensor:
        """Converts hidden states into logits."""
        h = self.layer_norm(h)
        return self.unembedding(h)

    def invert(
        self,
        logits: th.Tensor,
        *,
        h0: Optional[th.Tensor] = None,
        lens: Optional[th.nn.Module] = None,
        max_iter: int = 100,
        num_samples: int = 0,
        reverse: bool = False,
        tol: float = 1e-4,
    ) -> th.Tensor:
        """Find one or more hidden states that closely induce the given logits."""
        d_model = cast(int, self.unembedding.in_features)
        leading_dims = logits.shape[:-1]

        # Use Gaussian vector as the initial hidden state
        if h0 is None:
            if num_samples:
                leading_dims = (num_samples,) + leading_dims

            h0 = logits.new_empty(*leading_dims, d_model)
            h0.normal_()

        # Sanity check the shape of the initial hidden state. Can silently lead to
        # incorrect results due to broadcasting if we don't check this.
        elif h0.shape != (*leading_dims, d_model):
            raise ValueError(
                f"Initial hidden state has shape {h0.shape} but should have shape "
                f"{(*leading_dims, d_model)} given logits shape {logits.shape}."
            )

        h = th.nn.Parameter(h0)
        opt = th.optim.LBFGS(
            [h],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=tol,
        )

        log_p = logits.log_softmax(dim=-1)
        p = log_p.exp()

        def closure() -> th.Tensor:
            log_q = self(h + lens(h) if lens else h).log_softmax(-1)
            if reverse:
                H_p_q = -th.sum(log_q.exp() * log_p, dim=-1).mean()
            else:
                H_p_q = -th.sum(p * log_q, dim=-1).mean()

            H_p_q.backward()
            return H_p_q

        opt.step(closure)  # type: ignore
        return h.data
