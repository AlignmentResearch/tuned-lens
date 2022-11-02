from .residual_stream import ResidualStream
from typing import Optional
import torch as th


class ResidualStats:
    """Online mean and covariance matrix computation for residual streams.

    Shape and device are lazily inferred from the first stream that is passed to
    `update()`. The mean and variance are computed using the Welford algorithm.
    Streams are automatically cast to full precision before updating the stats.
    """

    def __init__(self):
        self._mu: Optional[ResidualStream] = None
        self._M2: Optional[ResidualStream] = None
        self._mean_norm: Optional[ResidualStream] = None

        self.n: int = 0

    def all_reduce_(self):
        """All-reduce the stats across all processes."""
        if self._mu is not None:
            self._mu.all_reduce_()

        if self._M2 is not None:
            self._M2.all_reduce_()

        if self._mean_norm is not None:
            self._mean_norm.all_reduce_()

    @th.no_grad()
    def update(self, stream: ResidualStream):
        """Update the online stats in-place with a new stream."""
        # Compute stats in full precision. Flatten all but the last dimension.
        stream = stream.map(lambda x: x.reshape(-1, x.shape[-1]).float())

        N, D = stream.shape
        self.n += N

        if self._mu is None or self._M2 is None or self._mean_norm is None:
            self._mu = stream.map(lambda x: x.new_zeros(D))
            self._M2 = stream.map(lambda x: x.new_zeros(D, D))
            self._mean_norm = stream.map(lambda x: x.new_zeros(()))

        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        delta = stream.zip_map(lambda x, mu: x - mu, self._mu)
        self._mu = delta.zip_map(lambda d, mu: mu + d.sum(0) / self.n, self._mu)
        delta2 = stream.zip_map(lambda x, mu: x - mu, self._mu)

        self._M2 = self._M2.zip_map(lambda acc, d, d2: acc + d.mT @ d2, delta, delta2)
        self._mean_norm = stream.zip_map(
            lambda x, mu: mu + th.sum(x.norm(dim=-1) - mu) / self.n, self._mean_norm
        )

    @property
    def covariance(self) -> ResidualStream:
        """Return the covariance matrix."""
        if not self._M2 or self.n < 2:
            raise ValueError("Not enough data")

        return self._M2.map(lambda x: x / (self.n - 1))

    @property
    def mean(self) -> ResidualStream:
        """Return the mean, throwing an error if there's not enough data."""
        if not self._mu:
            raise ValueError("Not enough data")

        return self._mu

    @property
    def mean_norm(self) -> ResidualStream:
        """Return the mean L2 norm, throwing an error if there's not enough data."""
        if not self._mean_norm:
            raise ValueError("Not enough data")

        return self._mean_norm

    @property
    def variance(self) -> ResidualStream:
        """Return the current estimate of the variance."""
        if not self._M2 or self.n < 2:
            raise ValueError("Not enough data")

        return self._M2.map(lambda x: th.linalg.diagonal(x) / (self.n - 1))

    def __repr__(self) -> str:
        return f"ResidualStats(n={self.n})"
