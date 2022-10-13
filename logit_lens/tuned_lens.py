from copy import deepcopy
from itertools import chain
from pathlib import Path

from .nn.low_rank_linear import LowRankLinear
from .utils import pairwise
from transformers import PreTrainedModel
from typing import Generator, Iterable, Optional, Union, overload
import json
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
        rank: Optional[int] = None,
        sublayers: bool = True,
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
            vocab_size = model.config.vocab_size

            # For HuggingFace models, we use whatever they call the "output embeddings"
            self.unembedding = deepcopy(model.get_output_embeddings())
            self.layer_norm = (
                getattr(model.base_model, "ln_f", None) or th.nn.Identity()
            )

        # Save config for later
        self.config = {
            k: v
            for k, v in locals().items()
            if k not in ("self", "model") and not k.startswith("_")
        }

        # Try to prevent finetuning the decoder
        assert d_model and num_layers
        self.layer_norm.requires_grad_(False)
        self.unembedding.requires_grad_(False)

        if rank:
            lens = LowRankLinear(d_model, d_model, rank, bias=bias)
        else:
            lens = th.nn.Linear(d_model, d_model, bias=bias)
            if identity_init:
                lens.weight.data.zero_()
                lens.bias.data.zero_()

        # Enforce orthogonality with matrix exponential parametrization
        if orthogonal:
            assert not rank
            lens = th.nn.utils.parametrizations.orthogonal(lens)

        self.add_module("input_adapter", lens if include_input else None)
        self.layer_adapters = th.nn.ModuleList(
            [deepcopy(lens) for _ in range(num_layers)]
        )
        self.attn_adapters = th.nn.ModuleList(
            [deepcopy(lens) for _ in range(num_layers)] if sublayers else []
        )

    def __iter__(self) -> Generator[th.nn.Module, None, None]:
        if isinstance(self.input_adapter, th.nn.Module):
            yield self.input_adapter

        if self.attn_adapters:
            # Interleave attention adapters with layer adapters
            yield from chain.from_iterable(zip(self.attn_adapters, self.layer_adapters))
        else:
            yield from self.layer_adapters

    @classmethod
    def load(cls, path: Union[str, Path], ckpt: str = "params.pt") -> "TunedLens":
        """Load a TunedLens from a file."""
        path = Path(path)

        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        # Load parameters
        state = th.load(path / ckpt)

        model = cls(**config)
        model.load_state_dict(state, strict=False)
        return model

    def save(self, path: Union[Path, str], ckpt: str = "params.pt") -> None:
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        th.save(self.state_dict(), path / ckpt)

        with open(path / "config.json", "w") as f:
            json.dump(self.config, f)

    @overload
    def iter_hiddens(
        self, hiddens: Iterable[tuple[str, th.Tensor]], tuned: bool = True
    ) -> Iterable[tuple[str, th.Tensor]]:
        ...

    @overload
    def iter_hiddens(
        self, hiddens: Iterable[th.Tensor], tuned: bool = True
    ) -> Iterable[th.Tensor]:
        ...

    def iter_hiddens(self, hiddens: Iterable, tuned: bool = True) -> Iterable:
        """Yield the transformed hiddens for each hidden state in an iterable."""
        # Sanity check to make sure we don't finetune the decoder
        if any(p.requires_grad for p in self.parameters(recurse=False)):
            raise RuntimeError("Make sure to freeze the decoder")

        for adapter, item in zip(self, hiddens):
            if isinstance(item, th.Tensor):
                yield item + adapter(item) if tuned else item

            elif isinstance(item, tuple):
                name, h = item
                yield name, h + adapter(h) if tuned else h
            else:
                raise TypeError(f"Unexpected type {type(item)}")

    @overload
    def map(
        self, hiddens: Iterable[tuple[str, th.Tensor]], logits: bool = True
    ) -> Iterable[tuple[str, th.Tensor]]:
        ...

    @overload
    def map(
        self,
        hiddens: Iterable[th.Tensor],
        logits: bool = True,
    ) -> Iterable[th.Tensor]:
        ...

    def map(self, hiddens: Iterable, logits: bool = True) -> Iterable:
        """Yield the logits for each hidden state in an iterable."""
        # Sanity check to make sure we don't finetune the decoder
        if any(p.requires_grad for p in self.parameters(recurse=False)):
            raise RuntimeError("Make sure to freeze the decoder")

        for adapter, item in zip(self, hiddens):
            if isinstance(item, th.Tensor):
                h = self.layer_norm(item + adapter(item))
                yield self.unembedding(h) if logits else h

            elif isinstance(item, tuple):
                name, h = item
                h = self.layer_norm(h + adapter(h))
                yield name, self.unembedding(h) if logits else h
            else:
                raise TypeError(f"Unexpected type {type(item)}")

    def forward(self, hiddens: Iterable[th.Tensor]) -> list[th.Tensor]:
        """Decode hidden states into logits"""
        return [logits for _, logits in self.map(hiddens)]

    def __len__(self) -> int:
        N = len(self.attn_adapters) + len(self.layer_adapters)
        if self.input_adapter:
            N += 1

        return N
