from ..utils import maybe_all_cat
from dataclasses import dataclass, field
from typing import Literal, NamedTuple
import torch as th
import warnings


class CalibrationEstimate(NamedTuple):
    ece: float
    num_bins: int


@dataclass
class CalibrationError:
    """Monotonic Sweep Calibration Error for the top label in multi-class problems.

    This method estimates the True Calibration Error (TCE) by searching for the largest
    number of bins into which the data can be split that preserves the monotonicity
    of the predicted confidence -> empirical accuracy mapping. We use equal mass bins
    (quantiles) instead of equal width bins. Roelofs et al. (2020) show that this
    estimator has especially low bias in simulations where the TCE is analytically
    computable, and is hyperparameter-free (except for the type of norm used).

    Paper: "Mitigating Bias in Calibration Error Estimation" by Roelofs et al. (2020)
    Link: https://arxiv.org/abs/2012.08668
    """

    confidences: list[th.Tensor] = field(default_factory=list)
    hits: list[th.Tensor] = field(default_factory=list)

    def all_gather_(self) -> None:
        self.confidences = [maybe_all_cat(x) for x in self.confidences]
        self.hits = [maybe_all_cat(x) for x in self.hits]

    def update(self, labels: th.Tensor, probs: th.Tensor) -> None:
        assert not th.is_floating_point(labels)
        assert th.is_floating_point(probs)
        assert labels.shape == probs.shape[:-1]
        assert probs.shape[-1] > 1

        labels = labels.detach().flatten()
        probs = probs.detach().flatten(end_dim=-2)

        # Find the top class and its confidence. We can throw away the rest of the
        # probabilities to save memory.
        confidences, predictions = probs.max(dim=-1)
        self.confidences.append(confidences)
        self.hits.append(predictions == labels)

    def compute(
        self, p: int = 2, strategy: Literal["quantile", "uniform"] = "quantile"
    ) -> CalibrationEstimate:
        """Compute the expected calibration error.

        Args:
            p: The norm to use for the calibration error. Defaults to 2 (Euclidean).
        """
        confidences = th.cat(self.confidences)
        hits = th.cat(self.hits)

        n = len(confidences)
        if n < 2:
            raise ValueError("Not enough data to compute calibration error.")

        # Sort the predictions and labels by confidence
        confidences, indices = confidences.sort()
        hits = hits[indices].float()

        # Search for the largest number of bins which preserves monotonicity.
        # Based on Algorithm 1 in Roelofs et al. (2020).
        # Using a single bin is guaranteed to be monotonic, so we start there.
        b_star, accs_star = 1, hits.mean().unsqueeze(0)
        for b in range(2, n + 1):
            if strategy == "quantile":
                # Split into (nearly) equal mass bins
                accs = th.stack([h.mean() for h in hits.tensor_split(b)])
            elif strategy == "uniform":
                # Split into equal width bins
                grid = th.linspace(0, 1, b + 1, device=hits.device)
                bin_sizes = th.searchsorted(confidences, grid, right=True).diff()
                if not th.all(bin_sizes):
                    break

                accs = th.segment_reduce(hits, "mean", lengths=bin_sizes)
            else:
                raise ValueError(f"Unknown strategy '{strategy}'.")

            # This binning is not strictly monotonic, let's break
            if not th.all(accs[1:] > accs[:-1]):
                break

            elif not th.all(accs * (1 - accs)):
                warnings.warn(
                    "Calibration error estimate may be unreliable due to insufficient"
                    " data in some bins."
                )
                break

            # Save the current binning, it's monotonic and may be the best one
            else:
                accs_star = accs
                b_star = b

        # Split into (nearly) equal mass bins. They won't be exactly equal, so we
        # still weight the bins by their size.
        conf_bins = confidences.tensor_split(b_star)
        w = th.tensor([len(c) / n for c in conf_bins])

        # See the definition of ECE_sweep in Equation 8 of Roelofs et al. (2020)
        mean_confs = th.stack([c.mean() for c in conf_bins])
        ece = th.sum(w * th.abs(accs_star - mean_confs) ** p) ** (1 / p)

        return CalibrationEstimate(float(ece), b_star)
