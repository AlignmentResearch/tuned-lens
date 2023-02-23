from numpy.typing import ArrayLike
from typing import Literal, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


def fit_anomaly_detector(
    normal: ArrayLike,
    anomalous: ArrayLike,
    *,
    include_anomalies: bool = False,
    method: Literal["iforest", "lof", "svm"] = "lof",
    seed: int = 42,
) -> tuple["BaseEstimator", float]:
    """Fit an unsupervised anomaly detector and test its AUROC on held out mixed data.

    By default, the model only sees normal data during training, but anomalous data
    can be included in the training set by setting `include_anomalies=True`.

    Args:
        normal: Normal data to train on.
        anomalous: Anomalous data to test on.
        include_anomalies: Whether to include the anomalous data in the training
            set. If False, the the model is only trained on normal datapoints.
        method: The anomaly detection method to use.
        seed: The random seed to use for train/test split.

    Returns:
        The fitted model and the AUROC on the held out mixed data.
    """
    # Avoid importing sklearn at module level
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    normal = np.asarray(normal)
    anomalous = np.asarray(anomalous)
    assert len(normal.shape) == 2
    assert normal.shape[1] == anomalous.shape[1]

    # Include anomalous data in training set
    if include_anomalies:
        contamination = len(anomalous) / (len(normal) + len(anomalous))
        if contamination > 0.5:
            raise ValueError(
                "Cannot have more anomalies than normal data in training set"
            )

        labels = np.concatenate([np.zeros(len(anomalous)), np.ones(len(normal))])
        X = np.concatenate([anomalous, normal])

        train_x, test_x, _, test_y = train_test_split(
            X, labels, random_state=seed, stratify=labels
        )
    # Train only on normal data, test on mixed data
    else:
        contamination = "auto"
        train_x, test_normal = train_test_split(normal, random_state=seed)
        test_x = np.concatenate([anomalous, test_normal])
        test_y = np.concatenate([np.zeros(len(anomalous)), np.ones(len(test_normal))])

    if method == "iforest":
        model = IsolationForest(
            contamination=contamination  # type: ignore[arg-type]
        ).fit(train_x)
        test_preds = model.score_samples(test_x)
    elif method == "lof":
        model = LocalOutlierFactor(
            contamination=contamination, novelty=True  # type: ignore[arg-type]
        ).fit(train_x)
        test_preds = model.decision_function(test_x)
    elif method == "svm":
        model = OneClassSVM(contamination=contamination).fit(  # type: ignore[arg-type]
            train_x
        )
        test_preds = model.decision_function(test_x)
    else:
        raise ValueError(f"Unknown anomaly detection method '{method}'")

    auroc = roc_auc_score(test_y, test_preds)
    return model, float(auroc)
