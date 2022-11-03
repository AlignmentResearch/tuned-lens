from copy import deepcopy
from itertools import chain
from pathlib import Path

from ..model_surgery import get_final_layer_norm
from ..residual_stream import ResidualStream
from . import LowRankLinear
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
        include_final: bool = False,
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
            assert isinstance(d_model, int) and isinstance(vocab_size, int)

            self.unembedding = deepcopy(model.get_output_embeddings())
            if ln := get_final_layer_norm(model):
                self.layer_norm = deepcopy(ln)
            else:
                self.layer_norm = th.nn.Identity()

        # Save config for later
        self.config = {
            k: v
            for k, v in locals().items()
            if not k.startswith("_") and not isinstance(v, th.nn.Module)
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
        self.attn_adapters = th.nn.ModuleList(
            [deepcopy(lens) for _ in range(num_layers)] if sublayers else []
        )
        if not include_final:
            num_layers -= 1

        self.layer_adapters = th.nn.ModuleList(
            [deepcopy(lens) for _ in range(num_layers)]
        )

    def __getitem__(self, item: int) -> th.nn.Module:
        """Get the adapter module at the given index."""
        if isinstance(self.input_adapter, th.nn.Module):
            if item == 0:
                return self.input_adapter
            else:
                item -= 1

        if len(self.attn_adapters):
            idx, is_layer = divmod(item, 2)
            return self.layer_adapters[idx] if is_layer else self.attn_adapters[idx]
        else:
            return self.layer_adapters[item]

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
            config.setdefault("include_final", True)  # Backwards compatibility

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

    def normalize_(self):
        """
        Canonicalize the transforms by centering their weights and biases, then
        normalizing `weight + th.eye(n)` to have Frobenius norm `sqrt(n)`.
        """
        for linear in self:
            assert isinstance(linear, th.nn.Linear)

            A, b = linear.weight.data, linear.bias.data
            A -= A.mean(dim=0, keepdim=True)
            b -= b.mean()

            n, n = A.shape
            I = th.eye(n, device=A.device)
            norm = th.norm(A + I) / n**0.5
            A.copy_((A + I) / norm - I)

    def transform(self, stream: ResidualStream, logits: bool = True) -> ResidualStream:
        if len(stream) != len(self):
            raise ValueError(
                f"Expected {len(self)} layers, but got {len(stream)} layers."
            )

        return stream.new_from_list(list(self.map(stream, logits=logits)))

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

    def forward(self, h: th.Tensor, idx: int) -> th.Tensor:
        """Decode hidden states into logits"""
        # Sanity check to make sure we don't finetune the decoder
        if any(p.requires_grad for p in self.parameters(recurse=False)):
            raise RuntimeError("Make sure to freeze the decoder")

        h = self.layer_norm(h + self[idx](h))
        return self.unembedding(h)

    def __len__(self) -> int:
        N = len(self.attn_adapters) + len(self.layer_adapters)
        if self.input_adapter:
            N += 1

        return N
