"""Compute online mean and covariance for residual streams."""

from ..residual_stream import ResidualStream
from typing import Optional
import torch as th


class ResidualStats:
    """Online mean and covariance matrix computation for residual streams.

    Shape and device are lazily inferred from the first stream that is passed to
    `update()`. The mean and variance are computed using the Welford algorithm.
    Streams are automatically cast to full precision before updating the stats.
    """

    # By default we accumulate in double precision to minimize numerical error
    def __init__(self, cov: bool = True, dtype: th.dtype = th.float64):
        """Create a new ResidualStats object.

        Args:
            cov: Whether to compute the covariance matrix.
            dtype: The dtype to use for calculations.
        """
        self._mu: Optional[ResidualStream] = None
        self._M2: Optional[ResidualStream] = None
        self._mean_norm: Optional[ResidualStream] = None

        self.cov = cov
        self.dtype = dtype
        self.n: int = 0

    def all_reduce_(self):
        """All-reduce the stats across all processes."""
        if self._mu is not None:
            self._mu.all_reduce_()

        if self._M2 is not None:
            self._M2.all_reduce_()

        if self._mean_norm is not None:
            self._mean_norm.all_reduce_()

    @th.autocast("cuda", enabled=False)
    @th.no_grad()
    def update(self, stream: ResidualStream):
        """Update the online stats in-place with a new stream."""
        # Flatten all but the last dimension
        stream = stream.map(lambda x: x.reshape(-1, x.shape[-1]))

        N, D = stream.shape
        self.n += N

        if self._mu is None or self._mean_norm is None:
            self._mu = stream.map(lambda x: x.new_zeros(D, dtype=self.dtype))
            self._mean_norm = stream.map(lambda x: x.new_zeros((), dtype=self.dtype))

        # Update running mean
        delta = stream.zip_map(lambda x, mu: x.type_as(mu) - mu, self._mu)
        delta.zip_map(lambda d, mu: mu.add_(d.sum(0), alpha=1 / self.n), self._mu)

        # We allow the user to set cov = False because it's expensive and O(d^2)
        if self.cov:
            # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            delta2 = stream.zip_map(lambda x, mu: x.type_as(mu) - mu, self._mu)

            if self._M2 is None:
                # On the first iteration set the running covariance to the sample cov
                self._M2 = delta.zip_map(lambda d, d2: d.T @ d2, delta2)
            else:
                # This is the hottest part of the loop, where we compute the sample
                # covariance and add it to the running estimate. This can lead to OOMs.
                # We fuse the add and matmul in-place to save VRAM & memory bandwidth
                self._M2.zip_map(lambda m, d, d2: m.addmm_(d.T, d2), delta, delta2)

        stream.zip_map(
            lambda x, mu: mu.add_(
                th.sum(x.norm(dim=-1).type_as(mu) - mu), alpha=1 / self.n
            ),
            self._mean_norm,
        )

    def covariance(self, dtype: th.dtype = th.float32) -> ResidualStream:
        """Return the covariance matrix."""
        if not self._M2 or self.n < 2:
            raise ValueError("Not enough data")

        return self._M2.map(lambda x: x.div(self.n - 1).to(dtype))

    def mean(self, dtype: th.dtype = th.float32) -> ResidualStream:
        """Return the mean, throwing an error if there's not enough data."""
        if not self._mu:
            raise ValueError("Not enough data")

        return self._mu.map(lambda x: x.to(dtype))

    def mean_norm(self, dtype: th.dtype = th.float32) -> ResidualStream:
        """Return the mean L2 norm, throwing an error if there's not enough data."""
        if not self._mean_norm:
            raise ValueError("Not enough data")

        return self._mean_norm.map(lambda x: x.to(dtype))

    def variance(self, dtype: th.dtype = th.float32) -> ResidualStream:
        """Return the current estimate of the variance."""
        if not self._M2 or self.n < 2:
            raise ValueError("Not enough data")

        return self._M2.map(lambda x: th.linalg.diagonal(x).div(self.n - 1).to(dtype))

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"ResidualStats(n={self.n})"
