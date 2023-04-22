"""Contains utility classes for creating heatmap visualizations."""
from dataclasses import dataclass
from typing import Any, Dict
from typing import Optional

from plotly import graph_objects as go
import numpy as np
from numpy.typing import NDArray


@dataclass
class TrajectoryLabels:
    """Contains sets of labels for each layer and position in the residual stream."""

    # (n_layers x sequence_length) label for each layer and position in the stream.
    label_strings: NDArray[np.str_]
    # (sequence_length) labels for the sequence dimension typically the input tokens.
    sequence_labels: NDArray[np.str_]
    # (n_layers x sequence_length x k) k entries to display when hovering over a cell.
    # For example, the top k prediction from the lens at each layer.
    hover_over_entries: Optional[NDArray[np.str_]] = None


@dataclass
class TrajectoryStatistic:
    """This class represents a trajectory statistic that can be visualized.

    For example, the entropy of the lens predictions at each layer.
    """

    # The name of the statistic.
    name: str
    # (n_layers x sequence_length) value of the statistic at each layer and position.
    stats: NDArray[np.float32]
    # labels for each layer and position in the stream. For example, the top 1
    # prediction from the lens at each layer.
    labels: Optional[TrajectoryLabels] = None
    # The units of the statistic.
    units: Optional[str] = None
    # The maximum value of the statistic.
    max: Optional[float] = None
    # The minimum value of the statistic.
    min: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate class invariants."""
        assert len(self.stats.shape) == 2
        assert self.labels is None or (
            self.labels.label_strings.shape == self.stats.shape
            and self.labels.sequence_labels.shape[0] == self.stats.shape[1]
        )

    @property
    def num_layers(self) -> int:
        """Return the number of layers in the stream."""
        return self.stats.shape[0]

    def heatmap(
        self,
        layer_stride: int = 1,
        colorscale: str = "rdbu_r",
        **kwargs,
    ) -> go.Heatmap:
        """Returns a Plotly Heatmap object for this statistic.

        Args:
            layer_stride : The number of layers between each layer plotted.
            colorscale : The colorscale to use for the heatmap.
            **kwargs : Additional keyword arguments to pass to the Heatmap constructor.

        Returns:
            A plotly Heatmap where the x-axis is the sequence dimension, the y-axis is
            the layer dimension, and the color of each cell is the value of
            the statistic.
        """
        labels = np.array(["input", *map(str, range(1, self.num_layers - 1)), "output"])

        color_matrix = self.stats

        color_matrix = _stride_keep_last(color_matrix, layer_stride)
        labels = _stride_keep_last(labels, layer_stride)

        heatmap_kwargs: Dict[str, Any] = dict(
            y=labels,
            z=color_matrix,
            colorbar=dict(
                title=f"{self.name} ({self.units})",
                titleside="right",
            ),
            zmax=self.max,
            zmin=self.min,
        )

        if self.labels is not None:
            label_strings = self.labels.label_strings
            label_strings = _stride_keep_last(label_strings, layer_stride)
            # Hack to ensure that Plotly doesn't de-duplicate the x-axis labels
            x_labels = [
                x + "\u200c" * i for i, x in enumerate(self.labels.sequence_labels)
            ]

            heatmap_kwargs.update(
                colorscale=colorscale,
                text=label_strings,
                texttemplate="<b>%{text}</b>",
                x=x_labels,
            )

            if self.labels.hover_over_entries is not None:
                hover_over_entries = _stride_keep_last(
                    self.labels.hover_over_entries, layer_stride
                )
                heatmap_kwargs.update(
                    customdata=hover_over_entries,
                    hoverlabel=dict(bgcolor="rgb(42, 42, 50)"),
                    hovertemplate="<br>".join(
                        f" %{{customdata[{i}]}}"
                        for i in range(hover_over_entries.shape[2])
                    )
                    + "<extra></extra>",
                )

        heatmap_kwargs.update(kwargs)
        return go.Heatmap(**heatmap_kwargs)

    def figure(
        self,
        title: str = "",
        layer_stride: int = 1,
        colorscale: str = "rdbu_r",
        token_width: int = 80,
    ) -> go.Figure:
        """Produce a heatmap plot of the statistic.

        Args:
            title : The title of the plot.
            layer_stride : The number of layers between each layer we plot.
            colorscale : The colorscale to use for the heatmap.
            token_width : The width of each token in the plot.

        Returns:
            The plotly heatmap figure.
        """
        heatmap = self.heatmap(layer_stride, colorscale)
        figure_width = 200 + token_width * self.stats.shape[1]

        fig = go.Figure(heatmap).update_layout(
            title_text=title,
            title_x=0.5,
            width=figure_width,
            xaxis_title="Input",
            yaxis_title="Layer",
        )
        return fig


def _stride_keep_last(x: NDArray, stride: int):
    return np.concatenate([x[:-1:stride], [x[-1]]])
