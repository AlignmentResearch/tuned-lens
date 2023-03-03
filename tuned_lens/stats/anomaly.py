from ..utils import assert_type
from dataclasses import dataclass
from numpy.typing import ArrayLike
from typing import Literal, Optional, TYPE_CHECKING
import numpy as np
import random

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.metrics import RocCurveDisplay


@dataclass
class AnomalyResult:
    """Result of an anomaly detection experiment."""

    model: "BaseEstimator"
    """The fitted anomaly detection model."""
    auroc: float
    """The AUROC on the held out mixed data."""
    bootstrapped_aurocs: list[float]
    """AUROCs computed on bootstrapped samples of the held out mixed data."""
    curve: Optional["RocCurveDisplay"]
    """The entire ROC curve on the held out mixed data."""


def bootstrap_auroc(
    labels: np.ndarray, scores: np.ndarray, num_samples: int = 1000, seed: int = 0
) -> list[float]:
    from sklearn.metrics import roc_auc_score

    rng = random.Random(seed)
    n = len(labels)
    aurocs = []

    for _ in range(num_samples):
        idx = rng.choices(range(n), k=n)
        aurocs.append(roc_auc_score(labels[idx], scores[idx]))

    return aurocs


def fit_anomaly_detector(
    normal: ArrayLike,
    anomalous: ArrayLike,
    *,
    bootstrap_iters: int = 1000,
    method: Literal["iforest", "lof", "svm"] = "lof",
    plot: bool = True,
    seed: int = 42,
    **kwargs,
) -> AnomalyResult:
    """Fit an unsupervised anomaly detector and test its AUROC on held out mixed data.

    The model only sees normal data during training, but is tested on a mix of normal
    and anomalous data. The AUROC is computed on the held out mixed data.

    Args:
        bootstrap_iters: The number of bootstrap iterations to use for computing the
            95% confidence interval of the AUROC.
        normal: Normal data to train on.
        anomalous: Anomalous data to test on.
        method: The anomaly detection method to use. "iforest" for `IsolationForest`,
            "lof" for `LocalOutlierFactor`, and "svm" for `OneClassSVM`.
        plot: Whether to return a `RocCurveDisplay` object instead of the AUROC.
        seed: The random seed to use for train/test split.
        **kwargs: Additional keyword arguments to pass to the scikit-learn constructor.

    Returns:
        The fitted model, the AUROC, the 95% confidence interval of the AUROC, and the
        entire ROC curve if `plot=True`, evaluated on the held out mixed data.
    """
    # Avoid importing sklearn at module level
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import RocCurveDisplay, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    normal = np.asarray(normal)
    anomalous = np.asarray(anomalous)
    assert len(normal.shape) == 2
    assert normal.shape[1] == anomalous.shape[1]

    # Train only on normal data, test on mixed data
    train_x, test_normal = train_test_split(normal, random_state=seed)
    test_x = np.concatenate([anomalous, test_normal])
    test_y = np.concatenate([np.zeros(len(anomalous)), np.ones(len(test_normal))])

    if method == "iforest":
        model = IsolationForest(**kwargs, random_state=seed).fit(train_x)
        test_preds = model.score_samples(test_x)
    elif method == "lof":
        model = LocalOutlierFactor(novelty=True, **kwargs).fit(train_x)
        test_preds = model.decision_function(test_x)
    elif method == "svm":
        model = OneClassSVM(**kwargs).fit(train_x)
        test_preds = model.decision_function(test_x)
    else:
        raise ValueError(f"Unknown anomaly detection method '{method}'")

    if plot:
        curve = RocCurveDisplay.from_predictions(test_y, test_preds)
        return AnomalyResult(
            model=model,
            auroc=assert_type(float, curve.roc_auc),
            bootstrapped_aurocs=bootstrap_auroc(test_y, test_preds, bootstrap_iters),
            curve=curve,
        )
    else:
        return AnomalyResult(
            model=model,
            auroc=float(roc_auc_score(test_y, test_preds)),
            bootstrapped_aurocs=bootstrap_auroc(test_y, test_preds, bootstrap_iters),
            curve=None,
        )
