from torch.distributions import Dirichlet
from typing import Optional
import torch as th


class LogitStats:
    """Online MLE for the Dirichlet distribution from which logits are sampled.

    Shape and device are lazily inferred from the first stream that is passed to
    `update()`. Only a running mean of the log-likelihoods for each class is stored,
    so memory use is negligible and constant in the number of samples. The maximum
    likelihood distribution is computed on request using L-BFGS.
    """

    n: int
    sufficient_stats: Optional[th.Tensor]

    def __init__(self):
        self.n = 0
        self.sufficient_stats = None

    @th.no_grad()
    def update(self, logits: th.Tensor):
        K = logits.shape[-1]
        logits = logits.reshape(-1, K).float()
        N = logits.shape[0]

        if self.sufficient_stats is None:
            self.sufficient_stats = logits.new_zeros(K)
        elif self.sufficient_stats.shape[-1] != K:
            raise ValueError(f"Expected {self.sufficient_stats.shape[-1]} but got {K}")

        # Online mean update for the sufficient statistics
        delta = logits.log_softmax(-1).mean(0) - self.sufficient_stats
        self.n += N
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
