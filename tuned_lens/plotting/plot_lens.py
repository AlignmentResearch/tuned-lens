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

    def plot(
        self,
        title: str = "",
        layer_stride: int = 1,
        colorscale: str = "rdbu_r",
        figure_width: int = 500,
    ) -> go.Figure:
        """Produce a heatmap plot of the statistic.

        Args:
            title : The title of the plot.
            layer_stride : The number of layers to between each layer we plot.
            colorscale : The colorscale to use for the heatmap.
            figure_width : The width of the figure in pixels.

        Returns:
            The plotly heatmap figure.
        """
        labels = np.array(["input", *map(str, range(1, self.num_layers - 1)), "output"])

        color_matrix = self.stats

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
            hover_over_entries = self.stream_labels.hover_over_entries
            label_strings = _stride_keep_last(label_strings, layer_stride)
            # Hack to ensure that Plotly doesn't de-duplicate the x-axis labels
            x_labels = [
                x + "\u200c" * i
                for i, x in enumerate(self.stream_labels.sequence_labels)
            ]

            figure_width = 200 + 80 * len(x_labels)

            heatmap_kwargs.update(
                colorscale=colorscale,
                customdata=hover_over_entries,
                text=label_strings,
                texttemplate="<b>%{text}</b>",
                x=x_labels,
            )

            if hover_over_entries is not None:
                hover_over_entries = _stride_keep_last(hover_over_entries, layer_stride)
                heatmap_kwargs.update(
                    hoverlabel=dict(bgcolor="rgb(42, 42, 50)"),
                    hovertemplate="<br>".join(
                        f" %{{customdata[{i}]}}"
                        for i in range(hover_over_entries.shape[2])
                    )
                    + "<extra></extra>",
                )

        color_matrix = _stride_keep_last(color_matrix, layer_stride)
        labels = _stride_keep_last(labels, layer_stride)

        heatmap = go.Heatmap(**heatmap_kwargs)

        # TODO Height needs to equal some function of Max(num_layers, topk).
        # Ignore for now. Works until k=18
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
        targets: Optional[Sequence[int]] = None,
        start_pos: int = 0,
        end_pos: Optional[int] = None,
        mask_input: bool = False,
    ) -> "PredictionTrajectory":
        """Constructs a slice of the model's prediction trajectory.

        Args:
            lens : The lens to use for constructing the latent predictions.
            model : The model to get the predictions from.
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

        targets_np = (
            np.array(targets[start_pos:end_pos]) if targets is not None else None
        )

        # Create the stream of log probabilities from the lens
        traj_log_probs = []
        for i, h in enumerate(stream[:-1]):
            logits = lens.forward(h, i)

            if mask_input:
                logits[..., input_ids] = -th.finfo(h.dtype).max

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
            log_probs=np.array(traj_log_probs),
            targets=targets_np,
            input_ids=input_ids_th.detach().cpu().numpy(),
        )

    # def get_stream_labels() -> StreamLabels:
    #     top_strings = []
    #     for lps in pred_trajectory.log_probs:
    #         ids = lps.argmax(-1).squeeze().cpu().tolist()
    #         tokens = tokenizer.convert_ids_to_tokens(ids)
    #         top_strings.append(formatter.vectorized_format(tokens))

    #     if min_prob:
    #         top_strings = [
    #             [
    #                 s if lp.max() > np.log(min_prob) else ""
    #                 for s, lp in zip(strings, log_probs)
    #             ]
    #             for strings, log_probs in zip(top_strings, pred_trajectory.log_probs)
    #         ]

    #     plotable_stream.stream_labels = StreamLabels(
    #         label_strings=np.array(top_strings),
    #         sequence_labels=format_fn(input_tokens),
    #         hover_over_entries=_get_topk_probs(
    #             stream_lps=stream_lps,
    #             tokenizer=tokenizer,
    #             formatter=formatter,
    #             k=topk,
    #             topk_diff=topk_diff,
    #         ),
    #     )

    def cross_entropy(self) -> TrajectoryStatistic:
        """The cross entropy of the predictions to the targets."""
        if self.targets is None:
            raise ValueError("Cannot compute cross entropy without targets.")

        assert self.targets.shape == self.log_probs[-1].shape[:-1], (
            "Batch and sequence lengths of targets and log probs must match."
            f"Got {self.targets.shape} and {self.log_probs[-1].shape[:-1]}."
        )

        return TrajectoryStatistic(
            name="Cross Entropy",
            units="nats",
            stats=self.log_probs[
                :, np.arange(self.num_tokens), self.targets
            ],  # TODO not sure if this is correct
        )

    def entropy(self) -> TrajectoryStatistic:
        """The entropy of the predictions."""
        return TrajectoryStatistic(
            name="Entropy",
            units="nats",
            stats=-np.sum(np.exp(self.log_probs) * self.log_probs, axis=-1),
        )

    def forward_kl(self) -> TrajectoryStatistic:
        """KL divergence of the lens predictions to the model predictions."""
        model_log_probs = self.model_log_probs.reshape(
            1, self.num_tokens, self.vocab_size
        )
        return TrajectoryStatistic(
            name="Forward KL",
            units="nats",
            stats=np.sum(
                np.exp(model_log_probs) * (model_log_probs - self.log_probs), axis=-1
            ),
        )

    def max_probability(self) -> TrajectoryStatistic:
        """Max probability of the among the predictions."""
        return TrajectoryStatistic(
            name="Max Probability",
            units="Probability",
            stats=self.log_probs.max(-1).values.exp(),
        )

    def kl_divergence(self, other: "PredictionTrajectory") -> TrajectoryStatistic:
        """Compute the KL divergence between self and other prediction trajectory."""
        kl_div = np.sum(self.probs * (self.log_probs - other.log_probs), axis=-1)

        # top_strings = []
        # for lps_a, lps_b in zip(lps_a, lps_b):
        #     kls = lps_a.exp() * (lps_a - lps_b)
        #     ids = kls.argmax(-1).squeeze().cpu().tolist()
        #     tokens = tokenizer.convert_ids_to_tokens(ids)
        #     top_strings.append(formatter.vectorized_format(tokens))

        return TrajectoryStatistic(
            name="KL(Model A | Model B)",
            units="nats",
            stats=kl_div.cpu().numpy(),
            # stream_labels=StreamLabels(
            #     label_strings=np.array(top_strings),
            #     sequence_labels=formatter.vectorized_format(input_tokens),
            # ),
            min=0,
            max=None,
        )

    def js_divergence(self, other: "PredictionTrajectory") -> TrajectoryStatistic:
        """Compute the JS divergence between self and other prediction trajectory."""
        js_div = 0.5 * np.sum(
            self.probs * (self.log_probs - other.log_probs), axis=-1
        ) + 0.5 * np.sum(self.probs * (self.log_probs - self.log_probs), axis=-1)

        # top_strings = []
        # for lps_a, lps_b in zip(lps_a, lps_b):
        #     kls = 0.5*(lps_a.exp() * (lps_a - lps_b)) + 0.5*(lps_b.exp() * (lps_b -
        # lps_a))
        #     ids = kls.argmax(-1).squeeze().cpu().tolist()
        #     tokens = tokenizer.convert_ids_to_tokens(ids)
        #     top_strings.append(formatter.vectorized_format(tokens))

        return TrajectoryStatistic(
            name="JS(Model A | Model B)",
            units="nats",
            stats=js_div.cpu().numpy(),
            # stream_labels=StreamLabels(
            #     label_strings=np.array(top_strings),
            #     sequence_labels=formatter.vectorized_format(input_tokens),
            # ),
            min=0,
            max=None,
        )

    def total_variation(self, other: "PredictionTrajectory") -> TrajectoryStatistic:
        """Total variation distance between self and other prediction trajectory."""
        t_var = np.abs(self.probs - other.probs).max(axis=-1)
        t_var.squeeze_()

        # top_strings = []
        # for lps_a, lps_b in zip(lps_a, lps_b):
        #     diffs = (lps_a.exp() - lps_b.exp()).abs()
        #     max_diff_ids = diffs.argmax(-1).squeeze().cpu().tolist()
        #     tokens = tokenizer.convert_ids_to_tokens(max_diff_ids)
        #     top_strings.append(formatter.vectorized_format(tokens))

        return TrajectoryStatistic(
            name="TV(Model A | Model B)",
            units="nats",
            stats=t_var.cpu().numpy(),
            # stream_labels=StreamLabels(
            #     label_strings=np.array(top_strings),
            #     sequence_labels=formatter.vectorized_format(input_tokens),
            # ),
            min=0,
            max=1,
        )


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
        self.vectorized_format = np.vectorize(
            lambda x: self.format(x) if isinstance(x, str) else "<unk>"
        )

    def format(self, token: str) -> str:
        """Format a token for display in a plot."""
        if self.max_string_len is not None and len(token) > self.max_string_len:
            token = token[: self.max_string_len - len(self.ellipsis)] + self.ellipsis
        token = token.replace(self.newline_token, self.newline_replacement)
        token = token.replace(self.whitespace_token, self.whitespace_replacement)
        return token


def _get_topk_probs(
    stream_lps: Sequence[th.FloatTensor],
    tokenizer: Tokenizer,
    formatter: TokenFormatter,
    k: int,
    topk_diff: bool,
):
    precents = [x.exp() * 100 for x in stream_lps]

    if topk_diff:
        raise NotImplementedError("topk_diff not implemented")

    precents = th.stack(list(precents)).squeeze(1)

    # Get the top-k tokens & probabilities for each
    topk = precents.topk(k, dim=-1)

    topk_values = topk.values
    # reshape topk_ind from (layers, seq, k) to (layers*seq*k), convert_ids_to_tokens,
    # then reshape back to (layers, seq, k)
    topk_tokens = tokenizer.convert_ids_to_tokens(topk.indices.reshape(-1).tolist())
    topk_tokens = np.array(topk_tokens).reshape(topk.indices.shape)

    def format_fn(token: str, percent: float):
        return f"{formatter.format(token)} {percent:.2f}%"

    format_fn = np.vectorize(format_fn)

    topk_strings_and_probs = format_fn(topk_tokens, topk_values.cpu().numpy())

    return topk_strings_and_probs


def _stride_keep_last(x: NDArray, stride: int):
    if stride == 1:
        return x
    elif len(x) % stride != 1:
        return np.concatenate([x[::stride], [x[-1]]])
    else:
        return x[::stride]
