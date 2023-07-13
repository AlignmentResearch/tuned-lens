"""Online MLE for the Dirichlet distribution from which logits are sampled."""
from typing import Optional

import torch as th
from torch.distributions import Dirichlet

from ..utils import maybe_all_reduce


class LogitStats:
    """Online MLE for the Dirichlet distribution from which logits are sampled.

    Shape and device are lazily inferred from the first stream that is passed to
    `update()`. Only a running mean of the log-likelihoods for each class is stored,
    so memory use is negligible and constant in the number of samples. The maximum
    likelihood distribution is computed on request using L-BFGS.
    """

    n: Optional[th.Tensor]
    marginal_probs: Optional[th.Tensor]
    sufficient_stats: Optional[th.Tensor]

    def __init__(
        self,
        n: Optional[th.Tensor] = None,
        marginal_probs: Optional[th.Tensor] = None,
        sufficient_stats: Optional[th.Tensor] = None,
    ):
        """Create a LogitStats object."""
        self.n = None
        self.marginal_probs = marginal_probs
        self.sufficient_stats = sufficient_stats

    def all_reduce_(self):
        """All-reduce the stats across all processes."""
        if (
            self.sufficient_stats is not None
            and self.marginal_probs is not None
            and self.n is not None
        ):
            n_x_sufficient_stats = self.n * self.sufficient_stats
            n_x_marginal_probs = self.n * self.marginal_probs
            maybe_all_reduce(n_x_sufficient_stats, op="sum")
            maybe_all_reduce(n_x_marginal_probs, op="sum")
            maybe_all_reduce(self.n, op="sum")
            self.sufficient_stats = n_x_sufficient_stats / self.n
            self.marginal_probs = n_x_marginal_probs / self.n
        else:
            raise ValueError("Attempting to reduce an uninitialized LogitStats object")

    @th.no_grad()
    def update(self, logits: th.Tensor, assume_normalized: bool = False):
        """Update the sufficient statistics with a new batch of logits."""
        K = logits.shape[-1]
        logits = logits.reshape(-1, K).float()
        if not assume_normalized:
            logits = logits.log_softmax(dim=-1)

        N = logits.shape[0]
        if self.n is None:
            self.n = th.tensor(0, dtype=th.int64, device=logits.device)
        elif len(self.n.shape) > 0:
            raise ValueError(f"Expected n to be a scalar but got {self.n.shape=}")

        if self.marginal_probs is None:
            self.marginal_probs = logits.new_zeros(K)
        elif self.marginal_probs.shape[-1] != K:
            raise ValueError(f"Expected {self.marginal_probs.shape[-1]} but got {K}")

        if self.sufficient_stats is None:
            self.sufficient_stats = logits.new_zeros(K)

        # Online mean update for the marginal probabilities
        delta = logits.exp().mean(0) - self.marginal_probs
        self.n += N
        self.marginal_probs += delta * N / self.n

        # Online mean update for the sufficient statistics
        delta = logits.mean(0) - self.sufficient_stats
        self.sufficient_stats += delta * N / self.n

    def mle(self, max_iter: int = 100, tol: float = 1e-4) -> Dirichlet:
        """Compute the MLE for the Dirichlet generating the logits seen so far."""
        if self.sufficient_stats is None:
            raise ValueError("No sufficient statistics available")

        log_alpha = th.nn.Parameter(th.zeros_like(self.sufficient_stats))
        opt = th.optim.LBFGS(
            [log_alpha],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=tol,
        )

        def closure():
            opt.zero_grad(set_to_none=False)

            # See http://jonathan-huang.org/research/dirichlet/dirichlet.pdf,
            # page 5 for the formula
            alpha = log_alpha.exp()
            normalizer = alpha.sum().lgamma() - alpha.lgamma().sum()
            loss = -(normalizer + (alpha - 1) @ self.sufficient_stats)
            loss.backward()
            return loss

        opt.step(closure)  # type: ignore
        return Dirichlet(log_alpha.data.exp())
