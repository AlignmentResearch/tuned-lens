from dataclasses import dataclass
from .residual_stream import ResidualStream
from typing import Optional
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
        self.n += stream.shape[0]
        if self.mean is None or self.M2 is None or self.autocorr is None:
            mean_shape = stream.shape[-1] if self.pool else stream.shape[1:]
            self.mean = stream.map(lambda x: x.new_zeros(mean_shape))
            self.M2 = stream.map(lambda x: x.new_zeros(mean_shape))
            self.autocorr = stream.map(lambda x: x.new_zeros(mean_shape))

        # Flatten all but the last dimension
        if self.pool:
            stream = stream.map(
                lambda x: x.reshape(-1, self.mean.shape[0])  # type: ignore
            )

        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        delta = stream.zip_map(lambda x, mu: x - mu, self.mean)
        self.mean = self.mean.zip_map(lambda acc, d: acc + d.sum(0) / self.n, delta)
        delta2 = stream.zip_map(lambda x, mu: x - mu, self.mean)
        self.M2 = self.M2.zip_map(
            lambda acc, d, d2: acc + th.sum(d * d2, dim=0), delta, delta2
        )

        # sample_autocorr = stream.map(lambda x: x[:, :-1] * x[:, 1:])
        # self.autocorr = self.autocorr.zip_map(
        #     lambda mu, x: mu + th.sum(x[:, :-1] * x[:, 1:] - mu, dim=0) / self.n,
        #     stream
        # )

    @property
    def variance(self) -> ResidualStream:
        """Return the current estimate of the variance."""
        if not self.M2 or self.n < 2:
            raise ValueError("Not enough data")

        return self.M2.map(lambda x: x / (self.n - 1))
