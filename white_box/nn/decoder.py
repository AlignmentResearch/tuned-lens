from dataclasses import dataclass
from torch.autograd.functional import hessian
from torch.distributions import Distribution
from transformers import PreTrainedModel
from typing import cast, Callable, Optional, Sequence
from white_box.model_surgery import get_final_layer_norm
from white_box.stats import kl_divergence
import torch as th
import torch.nn.functional as F


@dataclass
class AuxiliaryLoss:
    """Auxiliary loss for `Decoder.invert`"""

    probe: th.nn.Module

    # Must set exactly one of these
    target_ids: Optional[th.Tensor] = None
    target_logits: Optional[th.Tensor] = None
    weight: float = 1.0


@dataclass
class InversionOutput:
    """Output of `Decoder.invert`"""

    preimage: th.Tensor
    kl: th.Tensor
    loss: th.Tensor
    grad_norm: th.Tensor
    hessian: Optional[th.Tensor]

    nfev: int


class Decoder(th.nn.Module):
    singular_values: th.Tensor
    singular_vectors: th.Tensor

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
        gamma, beta = raw_ln.weight.data.float(), raw_ln.bias.data.float()
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
        self.register_buffer("singular_vectors", u)

        # In general we don't want to finetune the decoder
        self.requires_grad_(False)

    def forward(self, h: th.Tensor, transform: Callable = lambda x: x) -> th.Tensor:
        """Convert hidden states into logits."""
        return self.unembedding(self.layer_norm(transform(h)))

    def metric_tensor(
        self, h: th.Tensor, transform: Callable = lambda x: x
    ) -> th.Tensor:
        """Evaluate the pullback of the Fisher information metric at the point `h`."""
        # The Fisher-Rao metric tensor is the Hessian of the KL divergence
        import functorch as fth

        def kl_fn(h_p: th.Tensor, h_q: th.Tensor) -> th.Tensor:
            p = self(h_p, transform=transform)
            q = self(h_q, transform=transform)
            return kl_divergence(p, q)

        hess_fn = fth.hessian(kl_fn, argnums=1)
        if len(h.shape) == 2:
            hess_fn = fth.vmap(hess_fn)

        hess = hess_fn(h, h)
        assert isinstance(hess, th.Tensor)
        return hess

    def back_translate(
        self, h: th.Tensor, transform: Callable = lambda x: x
    ) -> th.Tensor:
        """Project hidden states into logits and then back into hidden states."""
        scale = h.norm(dim=-1, keepdim=True) / h.shape[-1] ** 0.5
        logits = self(h, transform=transform)
        return self.invert(logits, h0=th.randn_like(h)).preimage * scale

    def invert(
        self,
        logits: th.Tensor,
        *,
        aux_losses: Sequence[AuxiliaryLoss] = (),
        compute_hessian: bool = False,
        h0: Optional[th.Tensor] = None,
        transform: Callable = lambda x: x,
        max_iter: int = 1000,
        prior_weight: float = 0.0,
        prior: Optional[Distribution] = None,
        tol: float = 1e-4,
        weight: Optional[th.Tensor] = None,
    ) -> InversionOutput:
        """Project logits onto the image of the decoder, returning the preimage."""
        d_model = cast(int, self.unembedding.in_features)
        leading_dims = logits.shape[:-1]

        if h0 is None:
            # Initialize with the Moore-Penrose pseudoinverse
            # TODO: Can we learn a better initialization?
            h0 = logits @ self.U_pinv.mT

        # Sanity check the shape of the initial hidden state. Can silently lead to
        # incorrect results due to broadcasting if we don't check this.
        elif h0.shape != (*leading_dims, d_model):
            raise ValueError(
                f"Initial hidden state has shape {h0.shape} but should have shape "
                f"{(*leading_dims, d_model)} given logits shape {logits.shape}."
            )

        h_star = th.nn.Parameter(h0)
        opt = th.optim.LBFGS(
            [h_star],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=tol,
        )
        log_p = logits.log_softmax(dim=-1)
        p = log_p.exp()
        if weight is not None:
            p *= weight

        def compute_loss(h: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
            log_q = self(transform(h)).log_softmax(-1)
            kl = th.sum(p * (log_p - log_q), dim=-1).nanmean()
            loss = kl.clone()

            for aux_loss in aux_losses:
                logits = self(h + aux_loss.probe(h))
                scale = aux_loss.weight

                # This is a cross-entropy objective
                if aux_loss.target_ids is not None:
                    loss += scale * F.cross_entropy(logits, aux_loss.target_ids)

                # This is a KL-divergence objective
                elif aux_loss.target_logits is not None:
                    loss += scale * kl_divergence(logits, aux_loss.target_logits)
                else:
                    raise ValueError("Auxiliary loss has no target.")

            if prior_weight and prior is not None:
                # We evaluate the prior density on the post-norm hidden state,
                # to prevent the pre-norm hidden from collapsing towards zero.
                h_ = self.layer_norm(h)
                loss += prior_weight * -prior.log_prob(h_).mean()

            return loss, kl

        # Number of function evals, like in scipy.optimize.minimize
        nfev = 0
        loss, kl = log_p.new_tensor(th.inf), log_p.new_tensor(th.inf)

        def closure():
            nonlocal nfev, loss, kl
            nfev += 1

            opt.zero_grad()
            loss, kl = compute_loss(h_star)
            if loss.isfinite():
                loss.backward()
                return loss
            else:
                return 0.0

        grad_norm = log_p.new_tensor(th.inf)
        opt.step(closure)  # type: ignore
        while nfev < max_iter:
            opt.step(closure)  # type: ignore

            final_grad = h_star.grad
            assert final_grad is not None

            grad_norm = final_grad.norm()
            if grad_norm < tol or loss < tol:
                break

        with th.no_grad():
            output = InversionOutput(
                preimage=self.layer_norm(h_star.data),
                kl=kl.detach(),
                loss=loss.detach(),
                grad_norm=grad_norm,
                hessian=None,
                nfev=nfev,
            )

            if compute_hessian:
                hess = hessian(
                    lambda x: compute_loss(x)[0], h_star.data, vectorize=True
                )
                assert isinstance(hess, th.Tensor)
                output.hessian = hess

        return output
