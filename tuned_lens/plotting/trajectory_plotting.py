"""Contains utility classes for creating heatmap visualizations."""
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from plotly import graph_objects as go


def trunc_string_left(string: str, new_len: int) -> str:
    """Truncate a string to the left."""
    return " " * (new_len - len(string)) + string[-new_len:]


@dataclass
class TrajectoryLabels:
    """Contains sets of labels for each layer and position in the residual stream."""

    label_strings: NDArray[np.str_]
    """(n_layers x sequence_length) label for each layer and position in the stream."""
    hover_over_entries: Optional[NDArray[np.str_]] = None
    """(n_layers x sequence_length x rows x cols) table of strings to display when
        hovering over a cell. For example, the top k prediction from the lens."""

    def stride(self, stride: int) -> "TrajectoryLabels":
        """Return a new TrajectoryLabels with the given stride.

        Args:
            stride : The number of layers between each layer we keep.

        Returns:
            A new TrajectoryLabels with the given stride.
        """
        assert stride > 0, f"stride must be positive, got {stride}"
        return replace(
            self,
            label_strings=_stride_keep_last(self.label_strings, stride),
            hover_over_entries=None
            if self.hover_over_entries is None
            else _stride_keep_last(self.hover_over_entries, stride),
        )

    def template_and_customdata(
        self, col_width_limit: int = 10
    ) -> Tuple[str, NDArray[np.str_]]:
        """Construct a template for use with Plotly's hovertemplate."""
        assert self.hover_over_entries is not None
        n_rows, n_cols = self.hover_over_entries.shape[-2:]

        vec_str_len = np.vectorize(len)
        lengths = vec_str_len(self.hover_over_entries)
        max_col_lens = np.max(lengths, axis=(0, 1, 2), keepdims=True)
        max_col_lens = np.minimum(max_col_lens + 1, col_width_limit)

        vec_truncate = np.vectorize(trunc_string_left)
        truncated_entries = vec_truncate(self.hover_over_entries, max_col_lens)

        html_table = ""
        for row in range(n_rows):
            for col in range(n_cols):
                html_table += f"%{{customdata[{row*n_cols + col}]}}"
            html_table += "<br>"
        html_table += "<extra></extra>"
        customdata = truncated_entries.reshape(
            self.hover_over_entries.shape[:2] + (-1,)
        )
        return html_table, customdata


@dataclass
class TrajectoryStatistic:
    """This class represents a trajectory statistic that can be visualized.

    For example, the entropy of the lens predictions at each layer.
    """

    name: str
    """The name of the statistic. For example, "entropy"."""
    stats: NDArray[np.float32]
    """(n_layers x sequence_length) value of the statistic across layer and position."""
    sequence_labels: Optional[NDArray[np.str_]] = None
    """(sequence_length) labels for the sequence dimension e.g. input tokens."""
    trajectory_labels: Optional[TrajectoryLabels] = None
    """Labels for each layer and position in the stream. For example, the top 1
    prediction from the lens at each layer."""
    units: Optional[str] = None
    """The units of the statistic."""
    max: Optional[float] = None
    """The maximum value of the statistic."""
    min: Optional[float] = None
    """The minimum value of the statistic."""
    includes_output: bool = True
    """Whether the statistic includes the final output layer."""

    _layer_labels: Optional[NDArray[np.str_]] = None

    def __post_init__(self) -> None:
        """Validate class invariants."""
        assert len(self.stats.shape) == 2, f"{self.stats.shape} != (n_layers, seq_len)"

        assert self.trajectory_labels is None or (
            self.trajectory_labels.label_strings.shape == self.stats.shape
        ), f"{self.trajectory_labels.label_strings.shape} != {self.stats.shape}"

        assert self.sequence_labels is None or (
            self.sequence_labels.shape[-1] == self.stats.shape[-1]
        ), f"{self.sequence_labels.shape[-1]} != {self.stats.shape[-1]}"

        if self._layer_labels is None:
            if self.includes_output:
                self._layer_labels = np.array(
                    [*map(str, range(self.stats.shape[0] - 1)), "output"]
                )
            else:
                self._layer_labels = np.array([*map(str, range(self.stats.shape[0]))])

    def clip(self, min: float, max: float) -> "TrajectoryStatistic":
        """Return a new TrajectoryStatistic with the given min and max.

        Args:
            min : The minimum value to clip to.
            max : The maximum value to clip to.

        Returns:
            A new TrajectoryStatistic with the given min and max.
        """
        assert min < max, f"min must be less than max, got {min} >= {max}"
        return replace(
            self,
            stats=np.clip(self.stats, min, max),
            max=max,
            min=min,
        )

    def stride(self, stride: int) -> "TrajectoryStatistic":
        """Return a new TrajectoryStatistic with the given stride.

        Args:
            stride : The number of layers between each layer we keep.

        Returns:
            A new TrajectoryStatistic with the given stride.
        """
        assert stride > 0, f"stride must be positive, got {stride}"
        assert self._layer_labels is not None
        return replace(
            self,
            stats=_stride_keep_last(self.stats, stride),
            trajectory_labels=None
            if self.trajectory_labels is None
            else self.trajectory_labels.stride(stride),
            _layer_labels=None
            if self._layer_labels is None
            else _stride_keep_last(self._layer_labels, stride),
        )

    def heatmap(
        self,
        colorscale: str = "rdbu_r",
        log_scale: bool = False,
        **kwargs,
    ) -> go.Heatmap:
        """Returns a Plotly Heatmap object for this statistic.

        Args:
            colorscale : The colorscale to use for the heatmap.
            log_scale : Whether to use a log scale for the colorbar.
            **kwargs : Additional keyword arguments to pass to the Heatmap constructor.

        Returns:
            A plotly Heatmap where the x-axis is the sequence dimension, the y-axis is
            the layer dimension, and the color of each cell is the value of
            the statistic.
        """
        max = self.max if self.max is not None else np.max(self.stats)
        min = self.min if self.min is not None else np.min(self.stats)
        heatmap_kwargs: Dict[str, Any] = dict(
            y=self._layer_labels,
            z=self.stats if not log_scale else np.log10(self.stats),
            colorbar=dict(
                title=f"{self.name} ({self.units})"
            ),
            colorscale=colorscale,
            zmax=max if not log_scale else np.log10(max),
            zmin=min if not log_scale else np.log10(min),
        )

        if log_scale:
            smallest_tick = np.ceil(np.log10(min))
            biggest_tick = np.floor(np.log10(max))
            tickvals = np.arange(smallest_tick, biggest_tick + 1)
            heatmap_kwargs["colorbar"] = dict(
                tickmode="array",
                tickvals=tickvals,
                ticktext=["10^{}".format(i) for i in tickvals],
            )

        if self.sequence_labels is not None:
            # Hack to ensure that Plotly doesn't de-duplicate the x-axis labels
            x_labels = [x + "\u200c" * i for i, x in enumerate(self.sequence_labels)]
            heatmap_kwargs.update(x=x_labels)

        if self.trajectory_labels is not None:
            heatmap_kwargs.update(
                text=self.trajectory_labels.label_strings,
                texttemplate="<b>%{text}</b>",
            )

            if self.trajectory_labels.hover_over_entries is not None:
                (
                    hovertemplate,
                    custom_data,
                ) = self.trajectory_labels.template_and_customdata()
                heatmap_kwargs.update(
                    hoverlabel=dict(bgcolor="rgb(42, 42, 50)", font_family="Monospace"),
                    customdata=custom_data,
                    hovertemplate=hovertemplate,
                )

        heatmap_kwargs.update(kwargs)
        return go.Heatmap(**heatmap_kwargs)

    def figure(
        self,
        title: str = "",
        colorscale: str = "rdbu_r",
        token_width: int = 80,
    ) -> go.Figure:
        """Produce a heatmap plot of the statistic.

        Args:
            title : The title of the plot.
            colorscale : The colorscale to use for the heatmap.
            token_width : The width of each token in the plot.

        Returns:
            The plotly heatmap figure.
        """
        heatmap = self.heatmap(colorscale)
        figure_width = 200 + token_width * self.stats.shape[1]

        fig = go.Figure(heatmap).update_layout(
            title_text=title,
            title_x=0.5,
            width=figure_width,
            xaxis_title="Input",
            yaxis_title="Layer",
        )

        return fig


def _stride_keep_last(x: NDArray, stride: int) -> NDArray:
    return np.concatenate([x[:-1:stride], [x[-1]]])
