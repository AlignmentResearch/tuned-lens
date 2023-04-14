"""Plot a lens table for some given text and model."""

from ..nn.lenses import Lens
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from numpy.typing import NDArray
import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Dict, Any
import numpy as np
import plotly.graph_objects as go
import torch as th

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


@dataclass
class TokenFormatter:
    """Format tokens for display in a plot."""

    ellipsis: str = "…"
    newline_replacement: str = "\\n"
    newline_token: str = "Ċ"
    whitespace_token: str = "Ġ"
    whitespace_replacement: str = "_"
    max_string_len: Optional[int] = 7

    def __post_init__(self) -> None:
        """Post init hook to vectorize the format function."""
        self.vectorized_format = np.vectorize(self.format)

    def format(self, token: str) -> str:
        """Format a token for display in a plot."""
        if not isinstance(token, str):
            return "<unk>"

        if self.max_string_len is not None and len(token) > self.max_string_len:
            token = token[: self.max_string_len - len(self.ellipsis)] + self.ellipsis
        token = token.replace(self.newline_token, self.newline_replacement)
        token = token.replace(self.whitespace_token, self.whitespace_replacement)
        return token


@dataclass
class StreamLabels:
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
    """This class represents a stream statistic that can be visualized.

    For example, the entropy of the lens predictions at each layer. This class is
    meant to serve as an interface for the plotting code.
    """

    # The name of the statistic.
    name: str
    # (n_layers x sequence_length) value of the statistic at each layer and position.
    stats: NDArray[np.float32] = np.zeros((0, 0), dtype=float)
    # labels for each layer and position in the stream. For example, the top 1
    # prediction from the lens at each layer.
    stream_labels: Optional[StreamLabels] = None
    # The units of the statistic.
    units: Optional[str] = None
    # The maximum value of the statistic.
    max: Optional[float] = None
    # The minimum value of the statistic.
    min: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate class invariants."""
        assert len(self.stats.shape) == 2
        assert self.stream_labels is None or (
            self.stream_labels.label_strings.shape == self.stats.shape
            and self.stream_labels.sequence_labels.shape[0] == self.stats.shape[1]
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
            layer_stride : The number of layers to between each layer plotted.
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

        if self.stream_labels is not None:
            label_strings = self.stream_labels.label_strings
            label_strings = _stride_keep_last(label_strings, layer_stride)
            # Hack to ensure that Plotly doesn't de-duplicate the x-axis labels
            x_labels = [
                x + "\u200c" * i
                for i, x in enumerate(self.stream_labels.sequence_labels)
            ]

            heatmap_kwargs.update(
                colorscale=colorscale,
                text=label_strings,
                texttemplate="<b>%{text}</b>",
                x=x_labels,
            )

            if self.stream_labels.hover_over_entries is not None:
                hover_over_entries = _stride_keep_last(
                    self.stream_labels.hover_over_entries, layer_stride
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

    def plot(
        self,
        title: str = "",
        layer_stride: int = 1,
        colorscale: str = "rdbu_r",
        token_width: int = 80,
    ) -> go.Figure:
        """Produce a heatmap plot of the statistic.

        Args:
            title : The title of the plot.
            layer_stride : The number of layers to between each layer we plot.
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


@dataclass
class PredictionTrajectory:
    """Contains the predictions for a sequence of tokens."""

    # The log probabilities of the predictions for each hidden layer + the models logits
    # Shape: (num_layers + 1, seq_len, vocab_size)
    log_probs: NDArray[np.float32]
    input_ids: NDArray[np.int64]
    targets: Optional[NDArray[np.int64]] = None
    tokenizer: Optional[Tokenizer] = None

    def __post_init__(self) -> None:
        """Validate class invariants."""
        assert len(self.log_probs.shape) == 3, "log_probs.shape: {}".format(
            self.log_probs.shape
        )
        assert (
            self.log_probs.shape[1] == self.input_ids.shape[0]
        ), "log_probs.shape: {}, input_ids.shape: {}".format(
            self.log_probs.shape, self.input_ids.shape
        )
        assert (
            self.targets is None or self.targets.shape[0] == self.input_ids.shape[0]
        ), "targets.shape: {}, input_ids.shape: {}".format(
            self.targets.shape, self.input_ids.shape
        )

    @property
    def num_layers(self) -> int:
        """Returns the number of layers in the stream."""
        return self.log_probs.shape[0] - 1

    @property
    def num_tokens(self) -> int:
        """Returns the number of tokens in this slice of the sequence."""
        return self.log_probs.shape[1]

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return self.log_probs.shape[2]

    @property
    def model_log_probs(self) -> NDArray[np.float32]:
        """Returns the log probs of the model."""
        return self.log_probs[-1, ...]

    @property
    def probs(self) -> NDArray[np.float32]:
        """Returns the probabilities of the predictions."""
        return np.exp(self.log_probs)

    @classmethod
    def from_lens_and_model(
        cls,
        lens: Lens,
        model: PreTrainedModel,
        input_ids: Sequence[int],
        tokenizer: Optional[Tokenizer] = None,
        targets: Optional[Sequence[int]] = None,
        start_pos: int = 0,
        end_pos: Optional[int] = None,
        mask_input: bool = False,
    ) -> "PredictionTrajectory":
        """Constructs a slice of the model's prediction trajectory.

        Args:
            lens : The lens to use for constructing the latent predictions.
            model : The model to get the predictions from.
            tokenizer : The tokenizer to use for decoding the predictions.
            input_ids : The input ids to pass to the model.
            targets : The targets for
            start_pos : The start position of the slice across the sequence dimension.
            end_pos : The end position of the slice accross the sequence dimension.
            mask_input : whether to forbid the lens from predicting the input tokens.

        Returns:
            A PredictionTrajectory object containing the requested slice.
        """
        with th.no_grad():
            input_ids_th = th.tensor(input_ids, dtype=th.int64, device=model.device)
            outputs = model(input_ids_th.unsqueeze(0), output_hidden_states=True)

        # Slice arrays the specified range
        model_log_probs = (
            outputs.logits[..., start_pos:end_pos, :]
            .log_softmax(-1)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        stream = [h[..., start_pos:end_pos, :] for h in outputs.hidden_states]

        input_ids_np = np.array(input_ids[start_pos:end_pos])
        targets_np = (
            np.array(targets[start_pos:end_pos]) if targets is not None else None
        )

        # Create the stream of log probabilities from the lens
        traj_log_probs = []
        for i, h in enumerate(stream[:-1]):
            logits = lens.forward(h, i)

            if mask_input:
                logits[..., input_ids_np] = -th.finfo(h.dtype).max

            traj_log_probs.append(
                logits.log_softmax(dim=-1).squeeze().detach().cpu().numpy()
            )

        # Add model predictions
        if traj_log_probs[-1].shape[-1] != model_log_probs.shape[-1]:
            logging.warning(
                "Lens vocab size does not match model vocab size."
                "Truncating model outputs to match lens vocab size."
            )
        # Handle the case where the model has more/less tokens than the lens
        min_logit = -np.finfo(model_log_probs.dtype).max
        trunc_model_log_probs = np.full_like(traj_log_probs[-1], min_logit)
        trunc_model_log_probs[..., : model_log_probs.shape[-1]] = model_log_probs

        traj_log_probs.append(trunc_model_log_probs)

        return cls(
            tokenizer=tokenizer,
            log_probs=np.array(traj_log_probs),
            targets=targets_np,
            input_ids=input_ids_np,
        )

    def _get_topk_tokens_and_values(
        self,
        k: int,
        sort_by: NDArray[np.float32],
        values: NDArray[np.float32],
    ) -> NDArray[np.str_]:

        # Get the top-k tokens & probabilities for each
        topk_inds = np.argpartition(sort_by, -k, axis=-1)[..., -k:]
        topk_sort_by = np.take_along_axis(sort_by, topk_inds, axis=-1)
        topk_values = np.take_along_axis(values, topk_inds, axis=-1)

        # Ensure that the top-k tokens are sorted by probability
        sorted_top_k_inds = np.argsort(-topk_sort_by, axis=-1)
        topk_inds = np.take_along_axis(topk_inds, sorted_top_k_inds, axis=-1)
        topk_values = np.take_along_axis(topk_values, sorted_top_k_inds, axis=-1)

        # reshape topk_ind from (layers, seq, k) to (layers*seq*k),
        # convert_ids_to_tokens, then reshape back to (layers, seq, k)
        topk_tokens = self.tokenizer.convert_ids_to_tokens(topk_inds.flatten().tolist())
        topk_tokens = np.array(topk_tokens).reshape(topk_inds.shape)

        return topk_tokens, topk_values

    def largest_prob_labels(
        self,
        formatter: Optional[TokenFormatter] = None,
        min_prob: np.float_ = np.finfo(np.float32).eps,
        topk: int = 10,
    ) -> StreamLabels:
        """Labels for the prediction trajectory based on the most probable tokens.

        Args:
            formatter : The formatter to use for formatting the tokens.
            min_prob : The minimum probability for a token to used as a label.
            topk : The number of top tokens to include in the hover over menu.

        Raises:
            ValueError: If the tokenizer is not set.

        Returns:
            a set of stream labels that can be applied to a trajectory statistic.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set to get labels.")

        if formatter is None:
            formatter = TokenFormatter()

        input_tokens = self.tokenizer.convert_ids_to_tokens(self.input_ids.tolist())

        entry_format_fn = np.vectorize(
            lambda token, percent: f"{formatter.format(token)} {percent:.2f}%"
        )

        topk_tokens, topk_probs = self._get_topk_tokens_and_values(
            k=topk, sort_by=self.log_probs, values=self.probs
        )

        top_tokens = topk_tokens[..., 0]
        top_probs = topk_probs[..., 0]

        label_strings = np.where(
            top_probs > min_prob, formatter.vectorized_format(top_tokens), ""
        )

        return StreamLabels(
            label_strings=label_strings,
            sequence_labels=formatter.vectorized_format(input_tokens),
            hover_over_entries=entry_format_fn(topk_tokens, topk_probs * 100),
        )

    def largest_delta_in_prob_labels(
        self,
        other: "PredictionTrajectory",
        formatter: Optional[TokenFormatter] = None,
        min_prob_delta: np.float_ = np.finfo(np.float32).eps,
        topk: int = 10,
    ) -> StreamLabels:
        """Labels for a trajectory statistic based on the largest change in probability.

        Args:
            other : The other prediction trajectory to compare to.
            formatter : A TokenFormatter to use for formatting the labels.
            min_prob_delta : The minimum change in probability to include a label.
            topk : The number of top tokens to include in the hover over menu.

        Raises:
            ValueError: If the tokenizer is not set.

        Returns:
            A set of stream labels that can be added to a trajectory statistic.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set to get labels.")

        if formatter is None:
            formatter = TokenFormatter()

        input_tokens = self.tokenizer.convert_ids_to_tokens(self.input_ids.tolist())

        entry_format_fn = np.vectorize(
            lambda token, percent: f"{formatter.format(token)} Δ{percent:.2f}%"
        )

        deltas = other.probs - self.probs

        topk_tokens, topk_deltas = self._get_topk_tokens_and_values(
            k=topk, sort_by=np.abs(deltas), values=deltas
        )

        top_tokens = topk_tokens[..., 0]
        top_deltas = topk_deltas[..., 0]

        label_strings = np.where(
            np.abs(top_deltas) > min_prob_delta,
            formatter.vectorized_format(top_tokens),
            "",
        )

        return StreamLabels(
            label_strings=label_strings,
            sequence_labels=formatter.vectorized_format(input_tokens),
            hover_over_entries=entry_format_fn(topk_tokens, 100 * topk_deltas),
        )

    def cross_entropy(self, **kwargs) -> TrajectoryStatistic:
        """The cross entropy of the predictions to the targets.

        **kwargs are passed to largest_prob_labels.
        """
        if self.targets is None:
            raise ValueError("Cannot compute cross entropy without targets.")

        assert self.targets.shape == self.log_probs[-1].shape[:-1], (
            "Batch and sequence lengths of targets and log probs must match."
            f"Got {self.targets.shape} and {self.log_probs[-1].shape[:-1]}."
        )

        return TrajectoryStatistic(
            name="Cross Entropy",
            units="nats",
            stream_labels=self.largest_prob_labels(**kwargs)
            if self.tokenizer
            else None,
            stats=-self.log_probs[:, np.arange(self.num_tokens), self.targets],
        )

    def entropy(self, **kwargs) -> TrajectoryStatistic:
        """The entropy of the predictions.

        **kwargs are passed to largest_prob_labels.
        """
        return TrajectoryStatistic(
            name="Entropy",
            units="nats",
            stream_labels=self.largest_prob_labels(**kwargs)
            if self.tokenizer
            else None,
            stats=-np.sum(np.exp(self.log_probs) * self.log_probs, axis=-1),
        )

    def forward_kl(self, **kwargs) -> TrajectoryStatistic:
        """KL divergence of the lens predictions to the model predictions.

        **kwargs are passed to largest_prob_labels.
        """
        model_log_probs = self.model_log_probs.reshape(
            1, self.num_tokens, self.vocab_size
        )
        return TrajectoryStatistic(
            name="Forward KL",
            units="nats",
            stream_labels=self.largest_prob_labels(**kwargs)
            if self.tokenizer
            else None,
            stats=np.sum(
                np.exp(model_log_probs) * (model_log_probs - self.log_probs), axis=-1
            ),
        )

    def max_probability(self, **kwargs) -> TrajectoryStatistic:
        """Max probability of the among the predictions.

        **kwargs are passed to largest_prob_labels.
        """
        return TrajectoryStatistic(
            name="Max Probability",
            units="prob",
            stream_labels=self.largest_prob_labels() if self.tokenizer else None,
            stats=np.exp(self.log_probs.max(-1)),
        )

    def kl_divergence(
        self, other: "PredictionTrajectory", **kwargs
    ) -> TrajectoryStatistic:
        """Compute the KL divergence between self and other prediction trajectory.

        Args:
            other : The other prediction trajectory to compare to.
            **kwargs: are passed to largest_delta_in_prob_labels.

        Returns:
            A TrajectoryStatistic with the KL divergence between self and other.
        """
        kl_div = np.sum(self.probs * (self.log_probs - other.log_probs), axis=-1)

        return TrajectoryStatistic(
            name="KL(Self | Other)",
            units="nats",
            stats=kl_div,
            stream_labels=self.largest_delta_in_prob_labels(other, **kwargs)
            if self.tokenizer
            else None,
            min=0,
            max=None,
        )

    def js_divergence(
        self, other: "PredictionTrajectory", **kwargs
    ) -> TrajectoryStatistic:
        """Compute the JS divergence between self and other prediction trajectory.

        Args:
            other : The other prediction trajectory to compare to.
            **kwargs: are passed to largest_delta_in_prob_labels.

        Returns:
            A TrajectoryStatistic with the JS divergence between self and other.
        """
        js_div = 0.5 * np.sum(
            self.probs * (self.log_probs - other.log_probs), axis=-1
        ) + 0.5 * np.sum(self.probs * (self.log_probs - self.log_probs), axis=-1)

        return TrajectoryStatistic(
            name="JS(Self | Other)",
            units="nats",
            stats=js_div,
            stream_labels=self.largest_delta_in_prob_labels(other, **kwargs)
            if self.tokenizer
            else None,
            min=0,
            max=None,
        )

    def total_variation(
        self, other: "PredictionTrajectory", **kwargs
    ) -> TrajectoryStatistic:
        """Total variation distance between self and other prediction trajectory.

        Args:
            other : The other prediction trajectory to compare to.
            **kwargs: are passed to largest_delta_in_prob_labels.

        Returns:
            A TrajectoryStatistic with the total variational distance between
            self and other.
        """
        t_var = np.abs(self.probs - other.probs).max(axis=-1)

        return TrajectoryStatistic(
            name="TV(Self | Other)",
            units="prob",
            stats=t_var,
            stream_labels=self.largest_delta_in_prob_labels(other, **kwargs)
            if self.tokenizer
            else None,
            min=0,
            max=1,
        )


def _stride_keep_last(x: NDArray, stride: int):
    if stride == 1:
        return x
    elif len(x) % stride != 1:
        return np.concatenate([x[::stride], [x[-1]]])
    else:
        return x[::stride]
