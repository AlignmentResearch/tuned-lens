"""Provides a class for mapping transformer hidden states to logits (and vice versa)."""
from dataclasses import dataclass
from torch.autograd.functional import hessian
from torch.distributions import Distribution
from transformers import PreTrainedModel
from typing import cast, Callable, Literal, Optional
from tuned_lens.model_surgery import get_final_layer_norm, get_transformer_layers
from tuned_lens.stats import kl_divergence
from tuned_lens.utils import maybe_unpack
import torch as th


@dataclass
class InversionOutput:
    """Output of `Decoder.invert`."""

    preimage: th.Tensor
    grad_norm: th.Tensor
    kl: th.Tensor
    loss: th.Tensor
    nfev: int
    hessian: Optional[th.Tensor]


class Unembed(th.nn.Module):
    """Module that maps transformer hidden states to logits (and vice versa)."""

    transformer_layers: th.nn.ModuleList
    layer_norm: th.nn.LayerNorm
    unembedding: th.nn.Linear

    _U_pinv: th.Tensor

    def __init__(
        self,
        model: PreTrainedModel,
        extra_layers: int = 0,
    ):
        """Initialize the decoder.

        Args:
            model: A HuggingFace model from which to extract the unembedding matrix.
            extra_layers: To leave at the end of the transformer.
        """
        super().__init__()
        self.transformer_layers = th.nn.ModuleList()
        # Initializing from scratch without a model

        raw_ln = get_final_layer_norm(model)
        assert raw_ln is not None

        raw_unembed = model.get_output_embeddings()
        if not isinstance(raw_unembed, th.nn.Linear):
            # With nn.Linear we know that the unembedding matrix is .weight;
            # we don't want to guess incorrectly for other module classes.
            raise ValueError("Currently we only support nn.Linear unembeddings.")

        U = raw_unembed.weight.data.float()
        assert isinstance(U, th.Tensor)

        vocab_size, d_model = U.shape
        self.layer_norm = th.nn.LayerNorm(
            d_model, elementwise_affine=False, eps=getattr(raw_ln, "eps", 1e-5)
        )
        self.unembedding = th.nn.Linear(d_model, vocab_size, device=U.device)

        # Roll the LN bias into our unembedding Linear
        gamma, beta = raw_ln.weight.data.float(), raw_ln.bias.data
        if isinstance(beta, th.Tensor):
            bias = beta.float() @ U.T

            # GPT-J has a bias in the unembedding layer, so we need to add it
            if raw_unembed.bias is not None:
                bias += raw_unembed.bias.data.float()

            bias -= bias.mean()  # Shift invariance of softmax
            self.unembedding.bias.data = bias.to(beta.dtype)

        # Roll the LN diagonal scaling factor into our unembedding matrix
        if isinstance(gamma, th.Tensor):
            U = U * gamma

        # Softmax is invariant to constant shifts, so we can canonicalize U
        # by centering its rows. We can also center the columns because the
        # input gets centered by LayerNorm.
        U = U - U.mean(dim=0)
        U -= U.mean(dim=1, keepdim=True)
        self.unembedding.weight.data = U.to(raw_unembed.weight.dtype)

        # We precompute the pseudo-inverse of the unembedding matrix
        self.register_buffer("_U_pinv", U.pinverse())

        if extra_layers > 0:
            _, layers = get_transformer_layers(model)
            self.transformer_layers.extend(
                layers[-self.config.extra_layers, :]  # type: ignore[arg-type]
            )

        # In general we don't want to finetune the decoder
        self.requires_grad_(False)

    def unembedding_hash(self) -> int:
        """Hash the unmbedding matrix to identify the model."""
        parameter = self.unembedding.weight.data
        return hash(str(parameter.detach().cpu().numpy()))

    def forward(self, h: th.Tensor, transform: Callable = lambda x: x) -> th.Tensor:
        """Convert hidden states into logits."""
        h = transform(h)
        for layer in self.transformer_layers:
            h = maybe_unpack(layer(h))

        return self.unembedding(self.layer_norm(h))

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
        self, h: th.Tensor, transform: Callable = lambda x: x, tol: float = 1e-4
    ) -> th.Tensor:
        """Project hidden states into logits and then back into hidden states."""
        scale = h.norm(dim=-1, keepdim=True) / h.shape[-1] ** 0.5
        logits = self(h, transform=transform)
        return self.invert(logits, h0=th.randn_like(h), tol=tol).preimage * scale

    def invert(
        self,
        logits: th.Tensor,
        *,
        compute_hessian: bool = False,
        h0: Optional[th.Tensor] = None,
        max_iter: int = 1000,
        optimizer: Literal["lbfgs", "sgd"] = "lbfgs",
        prior_weight: float = 0.0,
        prior: Optional[Distribution] = None,
        step_size: float = 1.0,
        tol: float = 1e-3,
        transform: Callable = lambda x: x,
        weight: Optional[th.Tensor] = None,
    ) -> InversionOutput:
        """Project logits onto the image of the decoder, returning the preimage.

        When the hidden state dimension is smaller than the vocabulary size, the
        decoder cannot perfectly represent arbitrary logits, since its image is
        restricted to a subspace; this phenomenon is known as the softmax bottleneck
        (cf. https://arxiv.org/abs/1711.03953). Because of this, the inverse can only
        be approximate in general. Here, we use gradient-based optimization to find a
        hidden state that minimizes the KL divergence from the target distribution p to
        the decoder output q: h* = argmin KL(p || q).

        Args:
            logits: Tensor of shape `[..., vocab_size]` containing logits to invert.
            compute_hessian: Whether to compute and return the Hessian of the inversion
                objective at the solution.
            h0: Initial guess for the hidden state. If `None`, the least-squares
                solution of the linear equation xU = logits is used, where U is the
                unembedding matrix.
            max_iter: Maximum number of iterations for the optimizer to take.
            optimizer: Optimization algorithm to use. Currently, only "lbfgs" and "sgd"
                are supported.
            prior_weight: The weight of the prior distribution is given in the loss.
            prior: Prior distribution over hidden states used to regularize
                the inversion.
            step_size: The step size for the optimizer.
            tol: Tolerance for the inversion objective.
            transform: Callable = lambda x: x,
            weight: Optional tensor of shape `[..., vocab_size]` containing weights
                for each vocabulary item. If `None`, all classes are weighted equally.
        """
        d_model = cast(int, self.unembedding.in_features)
        leading_dims = logits.shape[:-1]

        if h0 is None:
            # Initialize with the Moore-Penrose pseudoinverse
            h0 = logits @ self.U_pinv.mT

        # Sanity check the shape of the initial hidden state. Can silently lead to
        # incorrect results due to broadcasting if we don't check this.
        elif h0.shape != (*leading_dims, d_model):
            raise ValueError(
                f"Initial hidden state has shape {h0.shape} but should have shape "
                f"{(*leading_dims, d_model)} given logits shape {logits.shape}."
            )

        h_star = th.nn.Parameter(h0)
        if optimizer == "lbfgs":
            opt = th.optim.LBFGS(
                [h_star],
                line_search_fn="strong_wolfe",
                lr=step_size,
                max_iter=max_iter,
                tolerance_change=tol,
            )
        elif optimizer == "sgd":
            opt = th.optim.SGD([h_star], lr=step_size)
        else:
            raise ValueError(f"Unknown optimizer '{optimizer}'")

        log_p = logits.log_softmax(dim=-1)
        p = log_p.exp()
        if weight is not None:
            p *= weight

        def compute_loss(h: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
            log_q = self(transform(h)).log_softmax(-1)
            kl = th.sum(p * (log_p - log_q), dim=-1).nanmean()
            loss = kl.clone()

            if prior_weight and prior is not None:
                # We evaluate the prior density on the post-norm hidden state,
                # to prevent the pre-norm hidden from collapsing towards zero.
                h_ = self.layer_norm(h)
                loss += prior_weight * -prior.log_prob(h_).mean()

            return loss, kl

        nfev = 0  # Number of function evals, like in scipy.optimize.minimize
        loss, kl = log_p.new_tensor(th.inf), log_p.new_tensor(th.inf)

        def closure():
            nonlocal nfev, loss, kl
            nfev += 1

            opt.zero_grad(set_to_none=False)
            loss, kl = compute_loss(h_star)

            if not loss.isfinite():
                raise RuntimeError("Inversion objective is not finite.")

            loss.backward()
            return loss

        grad_norm = log_p.new_tensor(th.inf)
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
                grad_norm=grad_norm,
                hessian=None,
                kl=kl.detach(),
                loss=loss.detach(),
                nfev=nfev,
            )

            if compute_hessian:
                hess = hessian(
                    lambda x: compute_loss(x)[0], h_star.data, vectorize=True
                )
                assert isinstance(hess, th.Tensor)
                output.hessian = hess

        return output
