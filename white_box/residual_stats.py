from dataclasses import dataclass
from .residual_stream import ResidualStream
from typing import Optional
import math
import torch as th


@dataclass
class ResidualStats:
    """Online mean, variance, and auto-covariance for residual streams."""

    mean: Optional[ResidualStream] = None
    M2: Optional[ResidualStream] = None
    autocorr: Optional[ResidualStream] = None
    mean_norm: Optional[ResidualStream] = None

    n: int = 0
    pool: bool = True
    track_cov: bool = True

    def all_reduce_(self):
        """All-reduce the stats across all processes."""
        if self.mean is not None:
            self.mean.all_reduce_()

        if self.M2 is not None:
            self.M2.all_reduce_()

        if self.autocorr is not None:
            self.autocorr.all_reduce_()

        if self.mean_norm is not None:
            self.mean_norm.all_reduce_()

    @th.no_grad()
    def update(self, stream: ResidualStream):
        """Update the online stats in-place with a new stream."""
        # Compute stats in full precision
        stream = stream.map(lambda x: x.float())
        self.n += math.prod(stream.shape[:-1]) if self.pool else stream.shape[0]

        # Autocorrelation is E[x_t * x_{t-1}]. It gets computed first because it
        # needs to know about the sequence order of the tokens
        sample_autocorr = stream.map(lambda x: th.mean(x[:, :-1] * x[:, 1:], dim=0))
        if self.pool:  # Pool across the sequence length
            sample_autocorr = sample_autocorr.map(lambda x: x.mean(dim=0))

        if self.autocorr is None:
            self.autocorr = sample_autocorr
        else:
            # Incremental mean update
            self.autocorr = self.autocorr.zip_map(
                lambda mu, x: mu + (x - mu) / self.n, sample_autocorr
            )

        if self.mean is None or self.M2 is None or self.mean_norm is None:
            mu_shape = (stream.shape[-1],) if self.pool else stream.shape[1:]
            var_shape = (*mu_shape, mu_shape[-1]) if self.track_cov else mu_shape

            self.mean = stream.map(lambda x: x.new_zeros(mu_shape))
            self.mean_norm = stream.map(lambda x: x.new_zeros(()))
            self.M2 = stream.map(lambda x: x.new_zeros(var_shape))

        # Flatten all but the last dimension
        if self.pool:
            stream = stream.map(lambda x: x.reshape(-1, x.shape[-1]))  # type: ignore

        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        delta = stream.zip_map(lambda x, mu: x - mu, self.mean)
        self.mean = self.mean.zip_map(lambda acc, d: acc + d.sum(0) / self.n, delta)
        delta2 = stream.zip_map(lambda x, mu: x - mu, self.mean)

        self.mean_norm = self.mean_norm.mean_update(stream.map(th.norm), self.n)
        if self.track_cov:
            self.M2 = self.M2.zip_map(lambda acc, d, d2: acc + d.T @ d2, delta, delta2)
        else:
            self.M2 = self.M2.zip_map(
                lambda acc, d, d2: acc + th.sum(d * d2, dim=0), delta, delta2
            )

    @property
    def autocov(self) -> ResidualStream:
        """Return the autocovariance, the de-meaned version of autocorrelation."""
        if not self.autocorr or not self.mean:
            raise ValueError("Autocorrelation is not computed yet")
        if self.pool:
            # TODO: Implement the online autocovariance algorithm described
            # here https://www.npmjs.com/package/online-autocovariance
            raise NotImplementedError(
                "Autocovariance is not implemented for pooled streams"
            )

        return self.autocorr.zip_map(lambda ac, mu: ac - mu[:-1] * mu[1:], self.mean)

    @property
    def covariance(self) -> ResidualStream:
        """Return the covariance matrix."""
        if not self.track_cov:
            raise ValueError("Set track_cov=True to compute covariance")
        if not self.M2 or self.n < 2:
            raise ValueError("Not enough data")

        return self.M2.map(lambda x: x / (self.n - 1))

    @property
    def variance(self) -> ResidualStream:
        """Return the current estimate of the variance."""
        if not self.M2 or self.n < 2:
            raise ValueError("Not enough data")

        if self.track_cov:
            return self.M2.map(lambda x: x.diag() / (self.n - 1))
        else:
            return self.M2.map(lambda x: x / (self.n - 1))
