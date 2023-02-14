from torch.distributions import Dirichlet
from typing import Optional
from ..utils import maybe_all_reduce
import torch as th


class LogitStats:
    """Online MLE for the Dirichlet distribution from which logits are sampled.

    Shape and device are lazily inferred from the first stream that is passed to
    `update()`. Only a running mean of the log-likelihoods for each class is stored,
    so memory use is negligible and constant in the number of samples. The maximum
    likelihood distribution is computed on request using L-BFGS.
    """

    n: int
    marginal_probs: Optional[th.Tensor]
    sufficient_stats: Optional[th.Tensor]

    def __init__(self):
        self.n = 0
        self.marginal_probs = None
        self.sufficient_stats = None

    def all_reduce_(self):
        """All-reduce the stats across all processes."""
        if self.sufficient_stats is not None:
            maybe_all_reduce(self.sufficient_stats)

    @th.no_grad()
    def update(self, logits: th.Tensor, assume_normalized: bool = False):
        K = logits.shape[-1]
        logits = logits.reshape(-1, K).float()
        if not assume_normalized:
            logits = logits.log_softmax(dim=-1)

        N = logits.shape[0]
        if self.marginal_probs is None:
            self.marginal_probs = logits.new_zeros(K)
        if self.sufficient_stats is None:
            self.sufficient_stats = logits.new_zeros(K)
        elif self.sufficient_stats.shape[-1] != K:
            raise ValueError(f"Expected {self.sufficient_stats.shape[-1]} but got {K}")

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
            opt.zero_grad()

            # See http://jonathan-huang.org/research/dirichlet/dirichlet.pdf,
            # page 5 for the formula
            alpha = log_alpha.exp()
            normalizer = alpha.sum().lgamma() - alpha.lgamma().sum()
            loss = -(normalizer + (alpha - 1) @ self.sufficient_stats)
            loss.backward()
            return loss

        opt.step(closure)  # type: ignore
        return Dirichlet(log_alpha.data.exp())
