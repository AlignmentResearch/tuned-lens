from copy import deepcopy
from itertools import chain
from transformers import PreTrainedModel
from typing import Iterable, Optional, overload
import torch as th


class TunedLens(th.nn.Module):
    """Stores all parameters necessary to decode hidden states into logits."""

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        *,
        bias: bool = True,
        identity_init: bool = True,
        include_input: bool = True,
        orthogonal: bool = False,
        sublayers: bool = False,
        # Automatically set for HuggingFace models
        d_model: Optional[int] = None,
        num_layers: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()

        # Initializing from scratch without a model
        if not model:
            assert d_model and num_layers and vocab_size
            self.layer_norm = th.nn.LayerNorm(d_model)
            self.unembedding = th.nn.Linear(d_model, vocab_size, bias=False)

        # Use HuggingFace methods to get decoder layers
        else:
            assert not d_model and not num_layers and not vocab_size
            d_model = model.config.hidden_size
            num_layers = model.config.num_hidden_layers

            # For HuggingFace models, we use whatever they call the "output embeddings"
            self.unembedding = deepcopy(model.get_output_embeddings())

            # plus the final, top-level layer norm if it exists.
            top_level_lns = [
                m for m in model.base_model.children() if isinstance(m, th.nn.LayerNorm)
            ]
            self.layer_norm = top_level_lns[-1] if top_level_lns else th.nn.Identity()

        # Try to prevent finetuning the decoder
        assert d_model and num_layers
        self.unembedding.requires_grad_(False)

        lens = th.nn.Linear(d_model, d_model, bias=bias)
        if identity_init:
            lens.weight.data = th.eye(d_model)  # Initialize with identity matrix
            lens.bias.data.zero_()

        # Enforce orthogonality with matrix exponential parametrization
        if orthogonal:
            lens = th.nn.utils.parametrizations.orthogonal(lens)

        self.add_module("input_adapter", lens if include_input else None)
        self.layer_adapters = th.nn.ModuleList(
            [deepcopy(lens) for _ in range(num_layers)]
        )
        self.attn_adapters = th.nn.ModuleList(
            [deepcopy(lens) for _ in range(num_layers)] if sublayers else []
        )

    @overload
    def iter_logits(
        self, hiddens: Iterable[tuple[str, th.Tensor]], tuned: bool = True
    ) -> Iterable[tuple[str, th.Tensor]]:
        ...

    @overload
    def iter_logits(
        self, hiddens: Iterable[th.Tensor], tuned: bool = True
    ) -> Iterable[th.Tensor]:
        ...

    def iter_logits(self, hiddens: Iterable, tuned: bool = True) -> Iterable:
        """Yield the logits for each hidden state in an iterable."""
        # Sanity check to make sure we don't finetune the decoder
        if self.unembedding.weight.requires_grad:
            raise RuntimeError("Make sure to freeze the decoder")

        adapters = self.layer_adapters
        if self.attn_adapters:
            # Interleave attention adapters with layer adapters
            adapters = chain.from_iterable(zip(self.attn_adapters, self.layer_adapters))

        # Tack on the input adapter if it exists
        if isinstance(self.input_adapter, th.nn.Module):
            adapters = chain([self.input_adapter], adapters)

        for adapter, item in zip(adapters, hiddens):
            if isinstance(item, th.Tensor):
                h = self.layer_norm(item)
                yield self.unembedding(adapter(h) if tuned else h)

            elif isinstance(item, tuple):
                name, h = item
                h = self.layer_norm(h)
                yield name, self.unembedding(adapter(h) if tuned else h)
            else:
                raise TypeError(f"Unexpected type {type(item)}")

    def forward(self, hiddens: Iterable[th.Tensor]) -> list[th.Tensor]:
        """Decode hidden states into logits"""
        return [logits for _, logits in self.iter_logits(hiddens)]

    def __len__(self) -> int:
        N = len(self.attn_adapters) + len(self.layer_adapters)
        if self.input_adapter:
            N += 1

        return N
