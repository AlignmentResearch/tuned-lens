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
    trajectory_labels = TrajectoryLabels(
        label_strings=np.zeros((2, 2), dtype=np.str_),
    )

    with pytest.raises(AssertionError):
        TrajectoryStatistic(
            name="test",
            stats=np.zeros((2, 3), dtype=float),
            trajectory_labels=trajectory_labels,
        )

    stats = np.zeros((3, 3), dtype=float)
    trajectory_labels = TrajectoryLabels(
        label_strings=np.zeros((3, 3), dtype=np.str_),
    )

    ts = TrajectoryStatistic(
        "test",
        stats,
        trajectory_labels=trajectory_labels,
    )
    assert ts is not None
    assert ts._layer_labels is not None
    assert np.array_equal(ts._layer_labels, np.array(["0", "1", "output"]))

    ts = TrajectoryStatistic(
        "test",
        stats,
        trajectory_labels=trajectory_labels,
        includes_output=False,
    )
    assert ts is not None
    assert ts._layer_labels is not None
    assert np.array_equal(ts._layer_labels, np.array(["0", "1", "2"]))


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


def test_stride_method():
    stats = np.zeros((3, 2), dtype=float)
    ts = TrajectoryStatistic("test", stats)
    stride = 2
    stride_ts = ts.stride(stride)
    assert stride_ts is not None
    assert stride_ts.name == ts.name
    assert stride_ts.units == ts.units
    assert stride_ts.max == ts.max
    assert stride_ts.min == ts.min
    assert stride_ts.includes_output == ts.includes_output
    assert stride_ts.stats is not None
    assert stride_ts.stats.shape == (2, 2)
    assert stride_ts._layer_labels is not None
    assert np.array_equal(stride_ts._layer_labels, np.array(["0", "output"]))
