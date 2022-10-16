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

    n: int = 0
    pool: bool = True

    @th.no_grad()
    def update(self, stream: ResidualStream):
        """Update the online stats in-place with a new stream."""
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

        if self.mean is None or self.M2 is None:
            mean_shape = stream.shape[-1] if self.pool else stream.shape[1:]
            self.mean = stream.map(lambda x: x.new_zeros(mean_shape))
            self.M2 = stream.map(lambda x: x.new_zeros(mean_shape))

        # Flatten all but the last dimension
        if self.pool:
            stream = stream.map(lambda x: x.reshape(-1, x.shape[-1]))  # type: ignore

        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        delta = stream.zip_map(lambda x, mu: x - mu, self.mean)
        self.mean = self.mean.zip_map(lambda acc, d: acc + d.sum(0) / self.n, delta)
        delta2 = stream.zip_map(lambda x, mu: x - mu, self.mean)
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
    def variance(self) -> ResidualStream:
        """Return the current estimate of the variance."""
        if not self.M2 or self.n < 2:
            raise ValueError("Not enough data")

        return self.M2.map(lambda x: x / (self.n - 1))
