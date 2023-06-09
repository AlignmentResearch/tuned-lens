"""Plot a lens table for some given text and model."""

import logging
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

    tokenizer: Optional[Tokenizer] = None

    def __post_init__(self) -> None:
        """Validate class invariants."""
        assert len(self.log_probs.shape) in (3, 4), "log_probs.shape: {}".format(
            self.log_probs.shape
        )
        assert len(self.input_ids.shape) in (1, 2), "input_ids.shape: {}".format(
            self.input_ids.shape
        )
        assert (
            len(self.log_probs.shape) == len(self.input_ids.shape) + 2
        ), "log_probs.shape: {}, input_ids.shape: {}".format(
            self.log_probs.shape, self.input_ids.shape
        )
        if len(self.log_probs.shape) == 4:
            assert (
                self.log_probs.shape[0] == self.input_ids.shape[0]
            ), "log_probs.shape: {}, input_ids.shape: {}".format(
                self.log_probs.shape, self.input_ids.shape
            )

        assert (
            self.log_probs.shape[-2] == self.input_ids.shape[-1]
        ), "log_probs.shape: {}, input_ids.shape: {}".format(
            self.log_probs.shape, self.input_ids.shape
        )

        assert (
            self.targets is None or self.targets.shape == self.input_ids.shape
        ), "targets.shape: {}, input_ids.shape: {}".format(
            self.targets.shape, self.input_ids.shape
        )

    @property
    def has_batch(self) -> bool:
        """Returns true if the trajectory has a batch dimension."""
        return len(self.log_probs.shape) > 3

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
        residual_component: ResidualComponent = "resid_pre",
    ) -> "PredictionTrajectory":
        """Construct a prediction trajectory from a set of residual stream vectors.

        Args:
            lens: A lens to use to produce the predictions
            cache: the activation cache produced by running the model.
            input_ids: (batch x seq_len) Ids that where input into the model.
                The batch dimension is reduced by replacing any tokens that differ
                across the batch with the pad token.
            model_logits: (batch x seq_len x d_vocab) the models final output logits.
            targets: (batch x seq_len) the targets the model is trying to predict. Used
                for cross entropy visualization. The batch dimension is reduced in
                the same way as input_ids.
            residual_component: Name of the stream vector being visualized.

        Returns:
            PredictionTrajectory constructed from the residual stream vectors.
        """
        if targets is not None and len(input_ids) != len(targets):
            raise ValueError(
                f"Length of input_ids {len(input_ids)} does not"
                f" match targets {len(targets)}"
            )

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

            traj_log_probs.append(logits.log_softmax(dim=-1).detach().cpu().numpy())

        model_log_probs = model_logits.log_softmax(-1).detach().cpu().numpy()

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
            log_probs=np.stack(traj_log_probs, axis=-3),
            targets=targets.numpy() if targets is not None else None,
            input_ids=input_ids.numpy(),
        )

    @classmethod
    def from_lens_and_model(
        cls,
        lens: Lens,
        model: PreTrainedModel,
        input_ids: Sequence[int],
        tokenizer: Optional[Tokenizer] = None,
        targets: Optional[Sequence[int]] = None,
        slice: slice = slice(None),
        mask_input: bool = False,
    ) -> "PredictionTrajectory":
        """Constructs a slice of the model's prediction trajectory.

        Args:
            lens : The lens to use for constructing the latent predictions.
            model : The model to get the predictions from.
            tokenizer : The tokenizer to use for decoding the predictions.
            input_ids : The input ids to pass to the model.
            targets : The targets for the input sequence.
            slice: The slice of the position dimension to record.
            mask_input : whether to forbid the lens from predicting the input tokens.

        Returns:
            A PredictionTrajectory object containing the requested slice.
        """
        with th.no_grad():
            input_ids_th = th.tensor(input_ids, dtype=th.int64, device=model.device)
            outputs = model(input_ids_th.unsqueeze(0), output_hidden_states=True)

        # Slice arrays the specified range
        model_log_probs = (
            outputs.logits[..., slice, :]
            .log_softmax(-1)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        stream = [h[..., slice, :] for h in outputs.hidden_states]

        input_ids_np = np.array(input_ids[slice])
        targets_np = np.array(targets[slice]) if targets is not None else None

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
        n_items_in_batch_to_show: int = 3,
    ) -> TrajectoryLabels:
        """Labels for the prediction trajectory based on the most probable tokens.

        Args:
            formatter : The formatter to use for formatting the tokens.
            min_prob : The minimum probability for a token to used as a label.
            topk : The number of top tokens to include in the hover over menu.
            n_items_in_batch_to_show : The number of items in the batch to show in the
                hover over menu.

        Raises:
            ValueError: If the tokenizer is not set.

        Returns:
            a set of stream labels that can be applied to a trajectory statistic.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set to get labels.")

        if formatter is None:
            formatter = TokenFormatter()

        if self.has_batch:
            # We need to reduce allonge the batch dimension
            mask = np.all(self.input_ids == self.input_ids[0], axis=0)
            input_ids = self.input_ids[0, ...]
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
            input_tokens = [
                input_token if mask else "*"
                for input_token, mask in zip(input_tokens, mask)
            ]
            print(input_tokens)

            def format_row(tokens: NDArray, percents: NDArray) -> str:

                row = ""
                for i, (token, percent) in enumerate(zip(tokens, percents)):
                    # TODO pad all tokens to be the same length and figure out how to
                    # control the number of columns in the hover over menu.
                    formatted = formatter.pad_token_repr_to_max_len(
                        formatter.format(token)
                    )
                    assert len(formatted) == formatter.max_string_len
                    percent = f"{percent:.2f}"
                    # ensure percent is 4 characters long
                    percent = " " * (5 - len(percent)) + percent
                    entry = f" {formatted} {percent}%"
                    # Pad the entry to be the same length as the longest entry
                    row += entry
                    if i >= n_items_in_batch_to_show:
                        row += " …"
                        break

                return row.strip()

            entry_format_fn = np.vectorize(format_row, signature="(n),(n)->()")

            topk_tokens, topk_probs = self._get_topk_tokens_and_values(
                k=topk, sort_by=self.log_probs, values=self.probs
            )

            topk_tokens = np.moveaxis(topk_tokens, 0, -1)
            topk_probs = np.moveaxis(topk_probs, 0, -1)

            return TrajectoryLabels(
                label_strings=np.full(
                    (self.num_layers + 1, self.num_tokens), "", dtype=str
                ),
                sequence_labels=formatter.vectorized_format(input_tokens),
                hover_over_entries=entry_format_fn(topk_tokens, topk_probs * 100),
            )
        else:
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

            return TrajectoryLabels(
                label_strings=label_strings,
                sequence_labels=formatter.vectorized_format(input_tokens),
                hover_over_entries=entry_format_fn(topk_tokens, topk_probs * 100),
            )

    def slice_sequence(self, slice: slice) -> "PredictionTrajectory":
        """Create a slice of the prediction trajectory along the sequence dimension."""
        return PredictionTrajectory(
            log_probs=self.log_probs[..., slice, :],
            input_ids=self.input_ids[slice],
            targets=self.targets[slice] if self.targets is not None else None,
            tokenizer=self.tokenizer,
        )

    def largest_delta_in_prob_labels(
        self,
        other: "PredictionTrajectory",
        formatter: Optional[TokenFormatter] = None,
        min_prob_delta: np.float_ = np.finfo(np.float32).eps,
        topk: int = 10,
    ) -> TrajectoryLabels:
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

        return TrajectoryLabels(
            label_strings=label_strings,
            sequence_labels=formatter.vectorized_format(input_tokens),
            hover_over_entries=entry_format_fn(topk_tokens, 100 * topk_deltas),
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

        flat_log_probs = self.log_probs.reshape(
            self.num_layers + 1, -1, self.vocab_size
        )
        flat_targets = self.targets.reshape(-1)
        stats = -flat_log_probs[..., np.arange(flat_log_probs.shape[1]), flat_targets]
        stats = stats.reshape(self.log_probs.shape[:-1])

        if self.has_batch:
            stats = stats.mean(axis=self.batch_axes)

        return TrajectoryStatistic(
            name="Cross Entropy",
            units="nats",
            labels=self.largest_prob_labels(**kwargs) if self.tokenizer else None,
            stats=stats,
        )

    def entropy(self, **kwargs) -> TrajectoryStatistic:
        """The entropy of the predictions.

        Args:
            **kwargs: are passed to largest_prob_labels.

        Returns:
            A TrajectoryStatistic with the entropy of the predictions.
        """
        stats = -np.sum(self.probs * self.log_probs, axis=-1)

        if self.has_batch:
            stats = stats.mean(axis=self.batch_axes)

        return TrajectoryStatistic(
            name="Entropy",
            units="nats",
            labels=self.largest_prob_labels(**kwargs) if self.tokenizer else None,
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

        if self.has_batch:
            stats = stats.mean(axis=self.batch_axes)

        return TrajectoryStatistic(
            name="Forward KL",
            units="nats",
            labels=self.largest_prob_labels(**kwargs) if self.tokenizer else None,
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

        if self.has_batch:
            stats = stats.mean(axis=self.batch_axes)

        return TrajectoryStatistic(
            name="Max Probability",
            units="probs",
            labels=self.largest_prob_labels(**kwargs) if self.tokenizer else None,
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
        if self.has_batch:
            raise NotImplementedError(
                "Batch KL divergence not implemented. "
                "Please open an issue if you need this."
            )

        kl_div = np.sum(self.probs * (self.log_probs - other.log_probs), axis=-1)

        return TrajectoryStatistic(
            name="KL(Self | Other)",
            units="nats",
            stats=kl_div,
            labels=self.largest_delta_in_prob_labels(other, **kwargs)
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
        if self.has_batch:
            raise NotImplementedError(
                "Batch JS divergence not implemented. "
                "Please open an issue if you need this."
            )

        js_div = 0.5 * np.sum(
            self.probs * (self.log_probs - other.log_probs), axis=-1
        ) + 0.5 * np.sum(other.probs * (other.log_probs - self.log_probs), axis=-1)

        return TrajectoryStatistic(
            name="JS(Self | Other)",
            units="nats",
            stats=js_div,
            labels=self.largest_delta_in_prob_labels(other, **kwargs)
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
        if self.has_batch:
            raise NotImplementedError(
                "Batch total variation not implemented. "
                "Please open an issue if you need this."
            )

        t_var = np.abs(self.probs - other.probs).max(axis=-1)

        return TrajectoryStatistic(
            name="TV(Self | Other)",
            units="probs",
            stats=t_var,
            labels=self.largest_delta_in_prob_labels(other, **kwargs)
            if self.tokenizer
            else None,
            min=0,
            max=1,
        )
