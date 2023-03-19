"""Provides a wrapper for implementing downstream evaluation with a tuned lens."""
from .lenses import TunedLens
from ..utils import pytree_map
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Iterable, NamedTuple, Optional
import torch as th
import torch.distributed as dist


class DownstreamResult(NamedTuple):
    """Result of a single downstream evaluation."""

    results: list
    greedy_preds: list[list[int]]
    hiddens: list[th.Tensor]


class DownstreamWrapper(th.nn.Module):
    """Wrapper around a model, a tokenizer, and a tuned lens for downstream eval."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        tuned_lens: Optional[TunedLens] = None,
    ):
        """Construct a new DownstreamWrapper."""
        super().__init__()

        local_rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = th.device(local_rank)
        self.model = model
        self.tokenizer = tokenizer
        self.tuned_lens = tuned_lens
        if self.tuned_lens:
            self.tuned_lens.eval()

    @property
    def max_length(self) -> int:
        """Maximum length the context window."""
        return getattr(self.model.config, "max_position_embeddings", 2048)

    def tok_encode(self, string: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        return self.tokenizer.encode(string, add_special_tokens=False)

    def forward(self, request, prompt: str) -> DownstreamResult:
        """Process command line request and return a DownstreamResult."""
        _, target_raw = request.args

        # sanity check
        assert len(prompt) > 0
        assert len(target_raw) > 0
        assert len(target_raw) <= self.max_length

        ctx = self.tok_encode(prompt)
        target = self.tok_encode(target_raw)

        # when too long to fit in context, truncate from the left
        inputs = th.tensor(
            (ctx + target)[-(self.max_length + 1) :][:-1],
            dtype=th.long,
            device=self.device,
        )
        (input_len,) = inputs.shape

        layer_results = []
        preds = []
        outputs = self.model(inputs[None], output_hidden_states=True)
        hiddens = [
            h[:, input_len - len(target) : input_len] for h in outputs.hidden_states
        ]

        def iter_log_probs() -> Iterable:
            """Iterate over log probs from each layer, w/o putting them in VRAM."""
            if self.tuned_lens:
                for i, h in enumerate(outputs.hidden_states[:-1]):
                    yield self.tuned_lens(h, i).log_softmax(dim=-1)

            yield outputs.logits.log_softmax(dim=-1)

        for log_probs in iter_log_probs():
            # Slice to original seq length
            log_probs = log_probs[:, input_len - len(target) : input_len]

            # Check if per-token argmax is exactly equal to continuation
            greedy_tokens = log_probs.argmax(dim=-1)
            target_th = greedy_tokens.new_tensor(target).unsqueeze(0)  # [1, seq]
            max_equal = th.all(greedy_tokens == target_th)
            preds.append(greedy_tokens)

            # Obtain log-probs at the corresponding continuation token indices
            log_probs = th.gather(log_probs, 2, target_th.unsqueeze(-1)).squeeze(
                -1
            )  # [1, seq]

            # Answer: (log prob, is-exact-match)
            result = (log_probs.sum(), max_equal)
            layer_results.append(
                result if request.index is None else result[request.index]
            )

        # Force GPU sync at the end
        layer_results = pytree_map(lambda x: x.item(), layer_results)
        preds = pytree_map(lambda x: x.squeeze(0).tolist(), preds)
        return DownstreamResult(layer_results, preds, hiddens)
