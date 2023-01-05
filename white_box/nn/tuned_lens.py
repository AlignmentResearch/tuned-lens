from copy import deepcopy
from itertools import chain
from pathlib import Path

from ..model_surgery import get_final_layer_norm
from ..utils import pairwise
from . import LowRankLinear
from transformers import PreTrainedModel
from typing import Generator, Optional, Sequence, Union, cast
import inspect
import json
import torch as th
import torch.nn.functional as F


class TunedLens(th.nn.Module):
    """Stores all parameters necessary to decode hidden states into logits."""

    layer_norm: th.nn.LayerNorm
    unembedding: th.nn.Linear

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        *,
        bias: bool = True,
        dropout: float = 0.0,
        identity_init: bool = True,
        include_input: bool = True,
        layer_norm: bool = False,
        mlp_hidden_sizes: Sequence[int] = (),
        rank: Optional[int] = None,
        shared_mlp_hidden_sizes: Sequence[int] = (),
        share_weights: bool = False,
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

            # Currently we convert the decoder to full precision
            self.unembedding = deepcopy(model.get_output_embeddings()).float()
            if ln := get_final_layer_norm(model):
                self.layer_norm = deepcopy(ln).float()
            else:
                self.layer_norm = th.nn.Identity()

        # Save config for later
        config_keys = set(inspect.getfullargspec(TunedLens).kwonlyargs)
        self.config = {k: v for k, v in locals().items() if k in config_keys}
        self.dropout = th.nn.Dropout(dropout)

        # Try to prevent finetuning the decoder
        assert d_model and num_layers
        self.layer_norm.requires_grad_(False)
        self.unembedding.requires_grad_(False)

        def create_mlp(hidden_sizes: Sequence[int]) -> th.nn.Sequential:
            sizes = [d_model, *hidden_sizes, d_model]
            mlp = th.nn.Sequential()

            for i, j in pairwise(sizes):
                layer = th.nn.Linear(i, j, bias=bias)
                mlp.extend([layer, th.nn.GELU()])

            mlp.pop(-1)  # Remove the last GELU

            last = cast(th.nn.Linear, mlp[-1])
            last.bias.data.zero_()
            last.weight.data.zero_()

            assert len(mlp) == 2 * len(hidden_sizes) + 1
            return mlp

        if mlp_hidden_sizes:
            probe = create_mlp(mlp_hidden_sizes)
        elif rank:
            probe = LowRankLinear(d_model, d_model, rank, bias=bias)
        else:
            probe = th.nn.Linear(d_model, d_model, bias=bias)
            if identity_init:
                probe.weight.data.zero_()
                probe.bias.data.zero_()

        maybe_copy = deepcopy if not share_weights else lambda x: x
        self.add_module("input_probe", probe if include_input else None)
        self.attn_probes = th.nn.ModuleList(
            [maybe_copy(probe) for _ in range(num_layers)] if sublayers else []
        )
        # Don't include the final layer
        num_layers -= 1

        self.layer_probes = th.nn.ModuleList(
            [maybe_copy(probe) for _ in range(num_layers)]
        )
        self.add_module(
            "shared_mlp",
            create_mlp(shared_mlp_hidden_sizes) if shared_mlp_hidden_sizes else None,
        )

    def __getitem__(self, item: int) -> th.nn.Module:
        """Get the probe module at the given index."""
        if isinstance(self.input_probe, th.nn.Module):
            if item == 0:
                return self.input_probe
            else:
                item -= 1

        if len(self.attn_probes):
            idx, is_layer = divmod(item, 2)
            return self.layer_probes[idx] if is_layer else self.attn_probes[idx]
        else:
            return self.layer_probes[item]

    def __iter__(self) -> Generator[th.nn.Module, None, None]:
        if isinstance(self.input_probe, th.nn.Module):
            yield self.input_probe

        if self.attn_probes:
            # Interleave attention probes with layer probes
            yield from chain.from_iterable(zip(self.attn_probes, self.layer_probes))
        else:
            yield from self.layer_probes

    @classmethod
    def load(
        cls, path: Union[str, Path], ckpt: str = "params.pt", **kwargs
    ) -> "TunedLens":
        """Load a TunedLens from a file."""
        path = Path(path)

        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        # Load parameters
        state = th.load(path / ckpt, **kwargs)
        state.setdefault("layer_norm", False)  # Backwards compatibility

        # Drop unrecognized config keys
        unrecognized = set(config) - set(inspect.getfullargspec(cls).kwonlyargs)
        for key in unrecognized:
            print(f"TunedLens.load: ignoring config key '{key}'")
            del config[key]

        model = cls(**config)
        model.load_state_dict(state)
        return model

    def save(self, path: Union[Path, str], ckpt: str = "params.pt") -> None:
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        th.save(self.state_dict(), path / ckpt)

        with open(path / "config.json", "w") as f:
            json.dump(self.config, f)

    def normalize_(self):
        """Canonicalize the transforms by centering their weights and biases."""
        if self.config["mlp_hidden_sizes"]:
            return

        for linear in self:
            assert isinstance(linear, th.nn.Linear)

            A, b = linear.weight.data, linear.bias.data
            A -= A.mean(dim=0, keepdim=True)
            b -= b.mean()

    def transform_hidden(self, h: th.Tensor, idx: int) -> th.Tensor:
        """Transform hidden state from layer `idx`."""

        # Dropout encourages the probe to use all "copies" of redundant information
        # in the hidden state; see https://arxiv.org/abs/2204.09722.
        h = self.dropout(h)
        h_ = F.layer_norm(h, (h.shape[-1],)) if self.config["layer_norm"] else h
        h = h + self[idx](h_)

        if isinstance(self.shared_mlp, th.nn.Module):
            h = F.layer_norm(h, (h.shape[-1],))
            h = h + self.shared_mlp(h)

        return h

    def to_logits(self, h: th.Tensor) -> th.Tensor:
        """Decode a hidden state into logits."""
        return self.unembedding(self.layer_norm(h))

    def forward(self, h: th.Tensor, idx: int) -> th.Tensor:
        """Decode hidden states into logits"""
        # Sanity check to make sure we don't finetune the decoder
        if any(p.requires_grad for p in self.parameters(recurse=False)):
            raise RuntimeError("Make sure to freeze the decoder")

        h = self.transform_hidden(h, idx)
        return self.to_logits(h)

    def __len__(self) -> int:
        N = len(self.attn_probes) + len(self.layer_probes)
        if self.input_probe:
            N += 1

        return N
