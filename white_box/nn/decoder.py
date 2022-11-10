from transformers import PreTrainedModel
from typing import cast, Optional
from white_box.model_surgery import get_final_layer_norm
import torch as th


class Decoder(th.nn.Module):
    singular_values: th.Tensor

    def __init__(self, model: PreTrainedModel):
        super().__init__()

        # Use HuggingFace methods to get decoder layers
        raw_unembed = model.get_output_embeddings()
        if not hasattr(raw_unembed, "weight"):
            raise ValueError("Failed to extract unembedding matrix.")

        unembed = raw_unembed.weight.data
        assert isinstance(unembed, th.Tensor)
        vocab_size, d_model = unembed.shape

        self.unembedding = th.nn.Linear(d_model, vocab_size, device=unembed.device)
        U = unembed.float()

        raw_ln = get_final_layer_norm(model)
        assert raw_ln is not None

        self.layer_norm = th.nn.LayerNorm(
            d_model, elementwise_affine=False, eps=getattr(raw_ln, "eps", 1e-5)
        )

        # Roll the LN bias into our unembedding Linear
        gamma, beta = raw_ln.weight.data, raw_ln.bias.data
        if isinstance(beta, th.Tensor):
            bias = beta @ U.T
            bias -= bias.mean()  # Shift invariance of softmax
            self.unembedding.bias.data = bias

        # Roll the LN diagonal scaling factor into our unembedding matrix
        if isinstance(gamma, th.Tensor):
            U = U * gamma

        # Softmax is invariant to constant shifts, so we can canonicalize U
        # by centering its rows. We can also center the columns because the
        # input gets centered by LayerNorm.
        U = U - U.mean(dim=0)
        U -= U.mean(dim=1, keepdim=True)
        self.unembedding.weight.data = U

        # Use SVD to compute the pseudo-inverse of U.
        u, s, v_h = th.linalg.svd(U, full_matrices=False)

        # Invert the singular values greater than a certain threshold,
        # replacing the rest with zeros, to get the pseudoinverse.
        # See https://www.johndcook.com/blog/2018/05/05/svd/.
        min_sigma = th.finfo(s.dtype).eps * max(U.shape) * s.max()
        s_plus = s.reciprocal().where(s > min_sigma, s.new_zeros(()))
        self.register_buffer("U_pinv", v_h.T @ th.diag(s_plus) @ u.T)

        # Save the singular values for analysis
        self.register_buffer("singular_values", s)

        # In general we don't want to finetune the decoder
        self.requires_grad_(False)

    def forward(self, h: th.Tensor) -> th.Tensor:
        """Convert hidden states into logits."""
        return self.unembedding(self.layer_norm(h))

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

        if h0 is None:
            # Initialize with the Moore-Penrose pseudoinverse
            if not num_samples:
                h0 = logits @ self.U_pinv.mT

            # Use Gaussian vectors as the initial hidden state
            else:
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
            opt.zero_grad()

            log_q = self(h + lens(h) if lens else h).log_softmax(-1)
            if reverse:
                H_p_q = -th.sum(log_q.exp() * log_p, dim=-1).mean()
            else:
                H_p_q = -th.sum(p * log_q, dim=-1).mean()

            H_p_q.backward()
            return H_p_q

        opt.step(closure)  # type: ignore
        return th.nn.functional.layer_norm(h.data, (d_model,))
