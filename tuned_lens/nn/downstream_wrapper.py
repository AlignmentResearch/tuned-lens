from .tuned_lens import TunedLens
from ..utils import pytree_map
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Iterable, NamedTuple, Optional, Sequence
import torch as th
import torch.distributed as dist


class DownstreamResult(NamedTuple):
    """Result of a single downstream evaluation."""

    results: list
    greedy_preds: list[list[int]]
    is_exact_match: list[bool]


class DownstreamWrapper(th.nn.Module):
    """Wrapper around a model, a tokenizer, and a tuned lens for downstream eval."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        tuned_lens: Optional[TunedLens] = None,
    ):
        super().__init__()

        local_rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = th.device(local_rank)
        self.model = model
        self.tokenizer = tokenizer
        self.tuned_lens = tuned_lens
        if self.tuned_lens:
            self.tuned_lens.eval()

    @property
    def max_length(self):
        return getattr(self.model.config, "max_position_embeddings", 2048)

    def iter_log_probs(self, x: th.Tensor) -> Iterable:
        """Iterate over log probs from each layer, w/o putting all of them in VRAM."""
        outputs = self.model(x, output_hidden_states=self.tuned_lens is not None)

        if self.tuned_lens:
            for i, h in enumerate(outputs.hidden_states[:-1]):
                yield self.tuned_lens(h, i).log_softmax(dim=-1)

        yield outputs.logits.log_softmax(dim=-1)

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def forward(self, request, prompt: str) -> DownstreamResult:
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

        exact_matches = []
        layer_results = []
        preds = []

        for log_probs in self.iter_log_probs(inputs[None]):
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
            exact_matches.append(max_equal)
            layer_results.append(
                result if request.index is None else result[request.index]
            )
            # breakpoint()

        # Force GPU sync at the end
        layer_results = pytree_map(lambda x: x.item(), layer_results)
        preds = pytree_map(lambda x: x.squeeze(0).tolist(), preds)
        match_list = [bool(is_match) for is_match in exact_matches]
        return DownstreamResult(layer_results, preds, match_list)
