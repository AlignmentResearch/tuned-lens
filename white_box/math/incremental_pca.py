from dataclasses import dataclass
from typing import Optional
import math
import torch as th


@dataclass
class IncrementalPCA:
    """Incrementally compute PCA over a dataset without storing it in memory."""

    mean: Optional[th.Tensor] = None
    U: Optional[th.Tensor] = None
    S: Optional[th.Tensor] = None

    ema_beta: float = 1.0
    n: int = 0

    def update(self, B_t: th.Tensor):
        """Update the PCA with a new batch of data.

        Algorithm from Figure 1 in "Incremental learning for robust visual tracking" (Ross et al. 2008).
        See http://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf.
        """
        # We compute the SVD of C = [A B], where A is the implicitly represented earlier data and
        # B is the new batch.
        n, m, d = self.n, *B_t.shape  # type: ignore

        # Weird PCA convention: the data is transposed
        B = B_t.T
        total = n + m

        # Base case
        batch_mean = B.mean(dim=1)
        if self.mean is None or self.U is None or self.S is None:
            self.mean = batch_mean
            self.U, self.S, _ = th.linalg.svd(B - batch_mean[:, None])
            self.n = m
            return

        extra_col = math.sqrt(n * m / total) * (batch_mean - self.mean)
        B_hat = th.hstack([(B - batch_mean[:, None]), extra_col[:, None]])
        B_proj = B_hat - self.U @ self.U.T @ B_hat
        B_tilde, _ = th.linalg.qr(B_proj)

        # Lemma 2 in the paper shows that multiplying the singular values by `f`
        # scales the influence of earlier observations by f^2 each update
        sigma = math.sqrt(self.ema_beta) * self.S
        R = th.vstack(
            [
                th.hstack([th.diag(sigma), self.U.T @ B_hat]),
                th.hstack([th.zeros_like(self.U), B_tilde @ B_proj]),
            ]
        )
        U_tilde, S_tilde, _ = th.linalg.svd(R)
        U_aug = th.hstack([self.U, B_tilde]) @ U_tilde

        self.U = U_aug[:, :d]
        self.S = S_tilde[:d]

        # Online mean update
        self.n += m
        if self.ema_beta < 1:
            self.mean = self.ema_beta * self.mean + (1 - self.ema_beta) * batch_mean
        else:
            self.mean = (n * self.mean + m * batch_mean) / self.n

    def transform(self, X: th.Tensor) -> th.Tensor:
        """Transform the data into the PCA space."""
        X = X - self.mean
        return X @ self.U

    def inverse(self, X: th.Tensor) -> th.Tensor:
        """Transform the data back to the original space."""
        assert self.U is not None
        return X @ self.U.T + self.mean

    __call__ = transform
