"""Plot a lens table for some given text and model."""

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

try:
    import transformer_lens as tl

    _transformer_lens_available = True
except ImportError:
    _transformer_lens_available = False

import numpy as np
import torch as th
from numpy.typing import NDArray
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from ..nn.lenses import Lens
from .token_formatter import TokenFormatter
from .trajectory_plotting import TrajectoryLabels, TrajectoryStatistic

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
ResidualComponent = Literal[
    "resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"
]


def _select_values_along_seq_axis(values: NDArray, targets: NDArray[np.int64]):
    """Select targe values along the the vocab dimension.

    Args:
        values: (..., n_layers, seq_len, vocab_size) the values to select from.
        targets: (..., seq_len) the indices to select.

    Returns:
        (..., n_layers, seq_len) the selected values.
    """
    return np.take_along_axis(
        values,
        targets[..., None, :, None],
        axis=-1,
    ).squeeze(-1)


def _ids_to_tokens(
    ids: NDArray[np.int64],
    tokenizer: Tokenizer,
) -> NDArray[np.str_]:
    """Convert a batch of ids to tokens.

    Args:
        ids: the input ids.
        tokenizer: The tokenizer to use for decoding the input ids.

    Returns:
        A batch of tokens.
    """
    tokens = tokenizer.convert_ids_to_tokens(ids.flatten().tolist())
    tokens = np.array(tokens).reshape(ids.shape)
    return tokens


def _consolidate_labels_from_batch(
    tokens: NDArray[np.str_],
    n_batch_axes: int,
    het_token_repr: str = "*",
) -> NDArray[np.str_]:
    """Get the input labels from a batch of input ids.

    Args:
        tokens: (*batch_axes, *axes_to_keep) the input ids.
        tokenizer: The tokenizer to use for decoding the input ids.
        n_batch_axes: The batch axes for in the input ids.
        het_token_repr: The string to use when the tokens are not the same across the
            batch i.e. they are heterogeneous.

    Returns:
        (*axes_to_keep) the input labels where all items in the batch are the same
        the token is used otherwise the token is replaced with `repeated_token_repr`.
    """
    first = tokens.reshape(-1, *tokens.shape[n_batch_axes:])[0, :]
    mask = np.all(
        tokens == first.reshape((1,) * n_batch_axes + tokens.shape[n_batch_axes:]),
        axis=tuple(range(n_batch_axes)),
    )
    return np.where(mask, first, het_token_repr)


@dataclass
class PredictionTrajectory:
    """Contains the trajectory predictions for a sequence of tokens.

    A prediction trajectory is the set of next token predictions produced by the
    conjunction of a lens and a model when evaluated on a specific sequence of tokens.
    This class include multiple methods for visualizing different
    aspects of the trajectory.
    """

    log_probs: NDArray[np.float32]
    """(..., n_layers, seq_len, vocab_size) The log probabilities of the predictions
    for each hidden layer + the models logits"""

    input_ids: NDArray[np.int64]
    """(..., seq_len)"""

    targets: Optional[NDArray[np.int64]] = None
    """(..., seq_len)"""

    anti_targets: Optional[NDArray[np.int64]] = None
    """(..., seq_len)"""

    tokenizer: Optional[Tokenizer] = None

    def __post_init__(self) -> None:
        """Validate class invariants."""
        assert (
            self.log_probs.shape[:-3] == self.input_ids.shape[:-1]
        ), "Batch shapes do not match log_probs.shape: {}, input_ids.shape: {}".format(
            self.log_probs.shape, self.input_ids.shape
        )

        assert (
            self.log_probs.shape[-2] == self.input_ids.shape[-1]
        ), "seq_len doesn't match log_probs.shape: {}, input_ids.shape: {}".format(
            self.log_probs.shape, self.input_ids.shape
        )

        assert (
            self.targets is None or self.targets.shape == self.input_ids.shape
        ), "Shapes don't match targets.shape: {}, input_ids.shape: {}".format(
            self.targets.shape, self.input_ids.shape
        )

        assert (
            self.anti_targets is None or self.anti_targets.shape == self.input_ids.shape
        ), "Shapes don't match anti_targets.shape: {}, input_ids.shape: {}".format(
            self.anti_targets.shape, self.input_ids.shape
        )

    @property
    def n_batch_axis(self) -> int:
        """Returns the number of batch dimensions."""
        return len(self.batch_axes)

    @property
    def batch_axes(self) -> Sequence[int]:
        """Returns the batch axes for the trajectory."""
        return tuple(range(len(self.log_probs.shape) - 3))

    @property
    def batch_shape(self) -> Sequence[int]:
        """Returns the batch shape of the trajectory."""
        return self.log_probs.shape[:-3]

    @property
    def num_layers(self) -> int:
        """Returns the number of layers in the stream not including the model output."""
        return self.log_probs.shape[-3] - 1

    @property
    def num_tokens(self) -> int:
        """Returns the number of tokens in this slice of the sequence."""
        return self.log_probs.shape[-2]

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return self.log_probs.shape[-1]

    @property
    def model_log_probs(self) -> NDArray[np.float32]:
        """Returns the log probs of the model (..., seq_len, vocab_size)."""
        return self.log_probs[..., -1, :, :]

    @property
    def probs(self) -> NDArray[np.float32]:
        """Returns the probabilities of the predictions."""
        return np.exp(self.log_probs)

    @classmethod
    def from_lens_and_cache(
        cls,
        lens: Lens,
        input_ids: th.Tensor,
        cache: "tl.ActivationCache",
        model_logits: th.Tensor,
        targets: Optional[th.Tensor] = None,
        anti_targets: Optional[th.Tensor] = None,
        residual_component: ResidualComponent = "resid_pre",
        mask_input: bool = False,
    ) -> "PredictionTrajectory":
        """Construct a prediction trajectory from a set of residual stream vectors.

        Args:
            lens: A lens to use to produce the predictions.
            cache: the activation cache produced by running the model.
            input_ids: (..., seq_len) Ids that where input into the model.
            model_logits: (..., seq_len x d_vocab) the models final output logits.
            targets: (..., seq_len) the targets the model is should predict. Used
                for :meth:`cross_entropy` and :meth:`log_prob_diff` visualization.
            anti_targets: (..., seq_len) the incorrect label the model should not
                predict. Used for :meth:`log_prob_diff` visualization.
            residual_component: Name of the stream vector being visualized.
            mask_input: Whether to mask the input ids when computing the log probs.

        Returns:
            PredictionTrajectory constructed from the residual stream vectors.
        """
        tokenizer = cache.model.tokenizer
        traj_log_probs = []
        for layer in range(cache.model.cfg.n_layers):
            hidden = cache[residual_component, layer]
            if input_ids.shape[-1] != hidden.shape[-2]:
                raise ValueError(
                    f"Length of input ids {input_ids.shape[-1]} does "
                    f"not match cache sequence length {hidden.shape[-2]}."
                )

            logits = lens.forward(hidden, layer)

            if mask_input:
                logits[..., input_ids] = -th.finfo(hidden.dtype).max

            traj_log_probs.append(logits.log_softmax(dim=-1).detach().cpu().numpy())

        model_log_probs = model_logits.log_softmax(-1).detach().cpu().numpy()

        traj_log_probs.append(model_log_probs)

        return cls(
            tokenizer=tokenizer,
            log_probs=np.stack(traj_log_probs, axis=-3),
            input_ids=input_ids.cpu().numpy(),
            targets=None if targets is None else targets.cpu().numpy(),
            anti_targets=None if anti_targets is None else anti_targets.cpu().numpy(),
        )

    @classmethod
    def from_lens_and_model(
        cls,
        lens: Lens,
        model: PreTrainedModel,
        input_ids: Sequence[int],
        tokenizer: Optional[Tokenizer] = None,
        targets: Optional[Sequence[int]] = None,
        anti_targets: Optional[Sequence[int]] = None,
        mask_input: bool = False,
    ) -> "PredictionTrajectory":
        """Construct a prediction trajectory from a set of residual stream vectors.

        Args:
            lens: A lens to use to produce the predictions. Note this should be
                compatible with the model.
            model: A Hugging Face causal language model to use to produce
                the predictions.
            tokenizer: The tokenizer to use for decoding the input ids.
            input_ids: (seq_len) Ids that where input into the model.
            targets: (seq_len) the targets the model is should predict. Used
                for :meth:`cross_entropy` and :meth:`log_prob_diff` visualization.
            anti_targets: (seq_len) the incorrect label the model should not
                predict. Used for :meth:`log_prob_diff` visualization.
            residual_component: Name of the stream vector being visualized.
            mask_input: Whether to mask the input ids when computing the log probs.

        Returns:
            PredictionTrajectory constructed from the residual stream vectors.
        """
        with th.no_grad():
            input_ids_th = th.tensor(input_ids, dtype=th.int64, device=model.device)
            outputs = model(input_ids_th.unsqueeze(0), output_hidden_states=True)

        # Slice arrays the specified range
        model_log_probs = (
            outputs.logits[..., :].log_softmax(-1).squeeze().detach().cpu().numpy()
        )
        stream = list(outputs.hidden_states)

        input_ids_np = np.array(input_ids)
        targets_np = np.array(targets) if targets is not None else None
        anti_targets_np = np.array(anti_targets) if anti_targets is not None else None

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
        traj_log_probs.append(model_log_probs)

        return cls(
            tokenizer=tokenizer,
            log_probs=np.array(traj_log_probs),
            targets=targets_np,
            input_ids=input_ids_np,
            anti_targets=anti_targets_np,
        )

    def _get_sequence_labels(
        self, token_formatter: Optional[TokenFormatter] = None
    ) -> Optional[NDArray[np.str_]]:
        """Get the input labels from a batch of input ids."""
        if self.tokenizer is None:
            return None

        if token_formatter is None:
            token_formatter = TokenFormatter()

        return _consolidate_labels_from_batch(
            tokens=token_formatter.vectorized_format(
                _ids_to_tokens(self.input_ids, self.tokenizer)
            ),
            n_batch_axes=self.n_batch_axis,
        )

    def _get_topk_tokens_and_values(
        self,
        k: int,
        sort_by: NDArray[np.float32],
        values: NDArray[np.float32],
    ) -> tuple[NDArray[np.str_], NDArray[np.float32]]:
        """Get the top-k tokens according to sort_by for each layer and position.

        Args:
            k: The number of top tokens to get.
            sort_by: (..., n_layers, seq_len) the values to sort by to get the top-k
            values: (..., n_layers, seq_len) the values to get the top-k tokens for.

        Returns:
            * (..., n_layers, seq_len, k) the top-k tokens for each layer and position.
            * (..., n_layers, seq_len, k) the top-k values for each layer and position.
        """
        assert self.tokenizer is not None

        # Get the top-k tokens & probabilities for each
        topk_inds = np.argpartition(sort_by, -k, axis=-1)[..., -k:]
        topk_sort_by = np.take_along_axis(sort_by, topk_inds, axis=-1)
        topk_values = np.take_along_axis(values, topk_inds, axis=-1)

        # Ensure that the top-k tokens are sorted by probability
        sorted_top_k_inds = np.argsort(-topk_sort_by, axis=-1)
        topk_inds = np.take_along_axis(topk_inds, sorted_top_k_inds, axis=-1)
        topk_values = np.take_along_axis(topk_values, sorted_top_k_inds, axis=-1)

        topk_tokens = _ids_to_tokens(topk_inds, self.tokenizer)

        return topk_tokens, topk_values

    def _hover_over_entries(
        self,
        topk_tokens: NDArray[np.str_],
        topk_values: NDArray[np.str_],
        max_entries_to_show: int = 3,
    ) -> NDArray[np.str_]:
        """Get the hover over entries for the stream.

        Args:
            topk_tokens: (..., n_layers, seq_len, k) the top-k tokens for each layer and
                position.
            topk_values: (..., n_layers, seq_len, k) the top-k values associated with
                each token.
            max_entries_to_show: The maximum number of entries in the batch to show in
                the hover over menu.

        Returns:
            (n_layers, seq_len, batch, 2*k) the table of entries to show when hovering
            over the stream. Here `batch` is the minimum of the batch size and the
            `max_entries_to_show`.
        """
        k = topk_tokens.shape[-1]
        topk_tokens = topk_tokens.reshape(-1, self.num_layers + 1, self.num_tokens, k)
        topk_values = topk_values.reshape(-1, self.num_layers + 1, self.num_tokens, k)
        topk_tokens = np.moveaxis(topk_tokens, 0, -1)
        topk_values = np.moveaxis(topk_values, 0, -1)
        hover_over_entries = np.empty(
            topk_tokens.shape[:-1] + (2 * topk_tokens.shape[-1],),
            dtype=topk_tokens.dtype,
        )
        hover_over_entries[..., 0::2] = topk_tokens
        hover_over_entries[..., 1::2] = topk_values
        return hover_over_entries[..., : 2 * max_entries_to_show]

    def _largest_prob_labels(
        self,
        formatter: Optional[TokenFormatter] = None,
        min_prob: float = 0,
        topk: int = 10,
        max_entries_to_show: int = 3,
    ) -> Optional[TrajectoryLabels]:
        """Labels for the prediction trajectory based on the most probable tokens.

        Args:
            formatter : The formatter to use for formatting the tokens.
            min_prob : The minimum probability for a token to used as a label.
            topk : The number of top tokens to include in the hover over menu.
            max_entries_to_show : The number of items in the batch to show in the
                hover over menu.
            show_values : Whether to show the probability values in the hover over

        Returns:
            A set of stream labels that can be applied to a trajectory statistic or
            None if the tokenizer is not set.
        """
        if self.tokenizer is None:
            return None

        if formatter is None:
            formatter = TokenFormatter()

        topk_tokens, topk_probs = self._get_topk_tokens_and_values(
            k=topk, sort_by=self.log_probs, values=self.probs
        )

        # Create the labels for the stream
        top_tokens = topk_tokens[..., 0]
        top_probs = topk_probs[..., 0]
        label_strings = _consolidate_labels_from_batch(
            tokens=formatter.vectorized_format(top_tokens),
            n_batch_axes=self.n_batch_axis,
        )
        label_strings = np.where((top_probs > min_prob).all(), label_strings, "")

        topk_probs_formatted = np.char.add(np.char.mod("%.2f", topk_probs * 100), "%")
        topk_tokens_formatted = formatter.vectorized_format(topk_tokens)

        topk_probs_formatted = np.char.add(np.char.mod("%.2f", topk_probs * 100), "%")
        topk_tokens_formatted = formatter.vectorized_format(topk_tokens)
        return TrajectoryLabels(
            label_strings=label_strings,
            hover_over_entries=self._hover_over_entries(
                topk_tokens=topk_tokens_formatted,
                topk_values=topk_probs_formatted,
                max_entries_to_show=max_entries_to_show,
            ),
        )

    def _largest_delta_in_prob_labels(
        self,
        other: "PredictionTrajectory",
        formatter: Optional[TokenFormatter] = None,
        min_prob_delta: float = 0,
        max_entries_to_show: int = 3,
        topk: int = 10,
    ) -> Optional[TrajectoryLabels]:
        """Labels for a trajectory statistic based on the largest change in probability.

        Args:
            other : The other prediction trajectory to compare to.
            formatter : A TokenFormatter to use for formatting the labels.
            min_prob_delta : The minimum change in probability to include a label.
            topk : The number of top tokens to include in the hover over menu.
            max_entries_to_show: The maximum number of entries in the batch to show in
                the hover over menu.

        Returns:
            A set of stream labels that can be added to a trajectory statistic.
        """
        if self.tokenizer is None:
            return None

        if formatter is None:
            formatter = TokenFormatter()

        deltas = other.probs - self.probs

        topk_tokens, topk_deltas = self._get_topk_tokens_and_values(
            k=topk, sort_by=np.abs(deltas), values=deltas
        )

        top_deltas = topk_deltas[..., 0]

        topk_tokens_formatted = formatter.vectorized_format(topk_tokens)
        top_tokens_formatted = topk_tokens_formatted[..., 0]
        topk_deltas_formatted = np.char.add(
            np.char.add("Δ", np.char.mod("%.2f", topk_deltas * 100)), "%"
        )

        label_strings = np.where(
            np.abs(top_deltas) > min_prob_delta,
            top_tokens_formatted,
            "",
        )

        label_strings = _consolidate_labels_from_batch(
            tokens=top_tokens_formatted,
            n_batch_axes=self.n_batch_axis,
        )
        return TrajectoryLabels(
            label_strings=label_strings,
            hover_over_entries=self._hover_over_entries(
                topk_tokens=topk_tokens_formatted,
                topk_values=topk_deltas_formatted,
                max_entries_to_show=max_entries_to_show,
            ),
        )

    def slice_sequence(self, slice: slice) -> "PredictionTrajectory":
        """Create a slice of the prediction trajectory along the sequence dimension."""
        return PredictionTrajectory(
            log_probs=self.log_probs[..., slice, :],
            input_ids=self.input_ids[..., slice],
            targets=self.targets[..., slice] if self.targets is not None else None,
            anti_targets=self.anti_targets[..., slice]
            if self.anti_targets is not None
            else None,
            tokenizer=self.tokenizer,
        )

    def cross_entropy(self, **kwargs) -> TrajectoryStatistic:
        """The cross entropy of the predictions to the targets.

        Args:
            **kwargs: are passed to largest_prob_labels.

        Returns:
            A TrajectoryStatistic with the cross entropy of the predictions to the
            targets.
        """
        if self.targets is None:
            raise ValueError("Cannot compute cross entropy without targets.")

        stats = -_select_values_along_seq_axis(self.log_probs, self.targets)

        if self.n_batch_axis:
            stats = stats.mean(axis=self.batch_axes)

        return TrajectoryStatistic(
            name="Cross Entropy",
            units="nats",
            trajectory_labels=self._largest_prob_labels(**kwargs),
            sequence_labels=self._get_sequence_labels(),
            stats=stats,
        )

    def rank(self, show_ranks=False, **kwargs) -> TrajectoryStatistic:
        """The rank of the targets among the predictions.

        That is, if the target is the most likely prediction, its rank is 1;
        the second most likely has rank 2, etc.

        Args:
            show_ranks: Whether to show the the rank of the target or the top token.
            **kwargs: are passed to largest_prob_labels.

        Returns:
            A TrajectoryStatistic with the rank of the targets among the predictions.
        """
        if self.targets is None:
            raise ValueError("Cannot compute rank without targets.")

        # With >, we disambiguate ties by taking the lowest rank for all the elements that are tied. E.g. if the top
        # predictions' logits are 0.3, 0.2, 0.2, 0.1; their ranks would be 1, 2, 2, 4.
        logprob_greater_than_target = self.log_probs > _select_values_along_seq_axis(self.log_probs, self.targets)[..., None]
        targets_rank = np.sum(logprob_greater_than_target, axis=-1) + 1

        if self.n_batch_axis:
            targets_rank = targets_rank.mean(axis=self.batch_axes)

        trajectory_labels = self._largest_prob_labels(**kwargs)

        if show_ranks and trajectory_labels is not None:
            trajectory_labels.label_strings = np.char.mod("%d", targets_rank)

        return TrajectoryStatistic(
            name="Rank",
            units="",
            trajectory_labels=trajectory_labels,
            sequence_labels=self._get_sequence_labels(),
            stats=targets_rank,
            min=1,
            max=None if self.tokenizer is None else self.tokenizer.vocab_size,
        )

    def entropy(self, **kwargs) -> TrajectoryStatistic:
        """The entropy of the predictions.

        Args:
            **kwargs: are passed to largest_prob_labels.

        Returns:
            A TrajectoryStatistic with the entropy of the predictions.
        """
        stats = -np.sum(self.probs * self.log_probs, axis=-1)

        if self.n_batch_axis:
            stats = stats.mean(axis=self.batch_axes)

        return TrajectoryStatistic(
            name="Entropy",
            units="nats",
            trajectory_labels=self._largest_prob_labels(**kwargs),
            sequence_labels=self._get_sequence_labels(),
            stats=stats,
        )

    def forward_kl(self, **kwargs) -> TrajectoryStatistic:
        """KL divergence of the lens predictions to the model predictions.

        Args:
            **kwargs: are passed to largest_prob_labels.

        Returns:
            A TrajectoryStatistic with the KL divergence of the lens predictions to the
            final output of the model.
        """
        model_log_probs = self.model_log_probs[..., np.newaxis, :, :]
        stats = np.sum(
            np.exp(model_log_probs) * (model_log_probs - self.log_probs), axis=-1
        )

        if self.n_batch_axis:
            stats = stats.mean(axis=self.batch_axes)

        return TrajectoryStatistic(
            name="Forward KL",
            units="nats",
            trajectory_labels=self._largest_prob_labels(**kwargs),
            sequence_labels=self._get_sequence_labels(),
            stats=stats,
        )

    def log_prob_diff(self, delta: bool = False) -> TrajectoryStatistic:
        """The difference in logits between two tokens.

        Returns:
            The difference between the log probabilities of the two tokens.
        """
        # TODO implement this as a way to compare two distributions
        if self.targets is None or self.anti_targets is None:
            raise ValueError(
                "Cannot compute log prob diff without targets" " and anti_targets."
            )

        targets_log_probs = _select_values_along_seq_axis(self.log_probs, self.targets)

        anti_targets_log_probs = _select_values_along_seq_axis(
            self.log_probs, self.anti_targets
        )

        stats = targets_log_probs - anti_targets_log_probs

        if delta:
            stats = stats[..., 1:, :] - stats[..., :-1, :]

        if self.n_batch_axis:
            stats = stats.mean(axis=self.batch_axes)

        return TrajectoryStatistic(
            name="Δ Log Prob Difference" if delta else "Log Prob Difference",
            units="nats",
            includes_output=not delta,
            sequence_labels=self._get_sequence_labels(),
            stats=stats,
        )

    def max_probability(self, **kwargs) -> TrajectoryStatistic:
        """Max probability of the among the predictions.

        Args:
            **kwargs: are passed to largest_prob_labels.

        Returns:
            A TrajectoryStatistic with the max probability of the among the predictions.
        """
        stats = np.exp(self.log_probs.max(-1))

        if self.n_batch_axis:
            stats = stats.mean(axis=self.batch_axes)

        return TrajectoryStatistic(
            name="Max Probability",
            units="probs",
            trajectory_labels=self._largest_prob_labels(**kwargs),
            sequence_labels=self._get_sequence_labels(),
            stats=stats,
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

        if self.n_batch_axis:
            kl_div = kl_div.mean(axis=self.batch_axes)

        return TrajectoryStatistic(
            name="KL(Self | Other)",
            units="nats",
            stats=kl_div,
            trajectory_labels=self._largest_delta_in_prob_labels(other, **kwargs),
            sequence_labels=self._get_sequence_labels(),
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
        ) + 0.5 * np.sum(other.probs * (other.log_probs - self.log_probs), axis=-1)

        if self.n_batch_axis:
            js_div = js_div.mean(axis=self.batch_axes)

        return TrajectoryStatistic(
            name="JS(Self | Other)",
            units="nats",
            stats=js_div,
            trajectory_labels=self._largest_delta_in_prob_labels(other, **kwargs),
            sequence_labels=self._get_sequence_labels(),
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

        if self.n_batch_axis:
            t_var = t_var.mean(axis=self.batch_axes)

        return TrajectoryStatistic(
            name="TV(Self | Other)",
            units="probs",
            stats=t_var,
            trajectory_labels=self._largest_delta_in_prob_labels(other, **kwargs),
            sequence_labels=self._get_sequence_labels(),
            min=0,
            max=1,
        )
