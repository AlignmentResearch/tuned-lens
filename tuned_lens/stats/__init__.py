from .anomaly import fit_anomaly_detector
from .calibration import CalibrationError
from .dimensionality import effective_rank
from .distance import (
    aitchison,
    aitchison_similarity,
    gaussian_wasserstein_l2,
    gaussian_wasserstein_l2_origin,
    geodesic_distance,
    js_divergence,
    js_distance,
    kl_divergence,
)
from .logit_stats import LogitStats
from .rank import nearest_neighbors, sample_neighbors, spearmanr
from .residual_stats import ResidualStats
