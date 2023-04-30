import numpy as np
import pytest
from plotly import graph_objects as go

from tuned_lens.plotting.trajectory_plotting import (
    TrajectoryLabels,
    TrajectoryStatistic,
    _stride_keep_last,
)


def test_stride_keep_last():
    x = np.array([1, 2, 3, 4, 5])

    assert np.array_equal(_stride_keep_last(x, 1), x)
    assert np.array_equal(_stride_keep_last(x, 2), np.array([1, 3, 5]))
    assert np.array_equal(_stride_keep_last(x, 3), np.array([1, 4, 5]))
    assert np.array_equal(_stride_keep_last(x, 4), np.array([1, 5]))
    assert np.array_equal(_stride_keep_last(x, 5), np.array([1, 5]))


def test_trajectory_statistic_post_init():
    stats = np.zeros((2, 2), dtype=float)
    labels = TrajectoryLabels(
        label_strings=np.zeros((2, 2), dtype=np.str_),
        sequence_labels=np.zeros(2, dtype=np.str_),
    )

    with pytest.raises(AssertionError):
        TrajectoryStatistic("test", np.zeros((2, 3), dtype=float), labels)

    stats = np.zeros((3, 3), dtype=float)
    labels = TrajectoryLabels(
        label_strings=np.zeros((3, 3), dtype=np.str_),
        sequence_labels=np.zeros(3, dtype=np.str_),
    )

    ts = TrajectoryStatistic("test", stats, labels)
    assert ts is not None


def test_trajectory_statistic_num_layers():
    stats = np.zeros((2, 2), dtype=float)
    ts = TrajectoryStatistic("test", stats)
    assert ts.num_layers == 2

    stats = np.zeros((3, 3), dtype=float)
    ts = TrajectoryStatistic("test", stats)
    assert ts.num_layers == 3


def test_trajectory_statistic_heatmap():
    stats = np.zeros((2, 2), dtype=float)
    ts = TrajectoryStatistic("test", stats)
    heatmap = ts.heatmap()
    assert isinstance(heatmap, go.Heatmap)


def test_trajectory_statistic_figure():
    stats = np.zeros((2, 2), dtype=float)
    ts = TrajectoryStatistic("test", stats)
    figure = ts.figure()
    assert isinstance(figure, go.Figure)
