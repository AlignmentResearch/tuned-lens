from numpy.typing import ArrayLike
from typing import Literal, TYPE_CHECKING, Union
import numpy as np

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.metrics import RocCurveDisplay


def fit_anomaly_detector(
    normal: ArrayLike,
    anomalous: ArrayLike,
    *,
    include_anomalies: bool = False,
    method: Literal["iforest", "lof", "svm"] = "lof",
    plot: bool = True,
    seed: int = 42,
    **kwargs,
) -> tuple["BaseEstimator", Union[float, "RocCurveDisplay"]]:
    """Fit an unsupervised anomaly detector and test its AUROC on held out mixed data.

    By default, the model only sees normal data during training, but anomalous data
    can be included in the training set by setting `include_anomalies=True`.

    Args:
        normal: Normal data to train on.
        anomalous: Anomalous data to test on.
        include_anomalies: Whether to include the anomalous data in the training
            set. If False, the the model is only trained on normal datapoints.
        method: The anomaly detection method to use. "iforest" for `IsolationForest`,
            "lof" for `LocalOutlierFactor`, and "svm" for `OneClassSVM`.
        plot: Whether to return a `RocCurveDisplay` object instead of the AUROC.
        seed: The random seed to use for train/test split.
        **kwargs: Additional keyword arguments to pass to the scikit-learn constructor.

    Returns:
        The fitted model and the AUROC (or entire ROC curve if `plot=True`) on the held
        out mixed data.
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

    # Include anomalous data in training set
    if include_anomalies:
        labels = np.concatenate([np.zeros(len(anomalous)), np.ones(len(normal))])
        X = np.concatenate([anomalous, normal])

        train_x, test_x, _, test_y = train_test_split(
            X, labels, random_state=seed, stratify=labels
        )
    # Train only on normal data, test on mixed data
    else:
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
        return model, curve
    else:
        return model, roc_auc_score(test_y, test_preds)
