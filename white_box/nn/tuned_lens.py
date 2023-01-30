from copy import deepcopy
from itertools import chain
from pathlib import Path

from ..model_surgery import get_final_layer_norm, get_transformer_layers
from ..utils import pairwise
from . import LowRankLinear
from transformers.models.bloom.modeling_bloom import BloomBlock
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
        extra_layers: int = 0,
        identity_init: bool = True,
        include_input: bool = True,
        layer_norm: bool = False,
        model_config: Optional[dict] = None,
        model_name: Optional[str] = None,
        mlp_hidden_sizes: Sequence[int] = (),
        pre_ln: bool = False,
        rank: Optional[int] = None,
        reuse_unembedding: bool = True,
        shared_mlp_hidden_sizes: Sequence[int] = (),
        sublayers: bool = True,
        # Automatically set for HuggingFace models
        d_model: Optional[int] = None,
        num_layers: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()

        self.extra_layers = th.nn.Sequential()

        # Initializing from scratch without a model
        if not model:
            assert d_model and num_layers and vocab_size
            self.layer_norm = th.nn.LayerNorm(d_model)
            self.model_name = model_name
            self.unembedding = th.nn.Linear(d_model, vocab_size, bias=False)

        # Use HuggingFace methods to get decoder layers
        else:
            assert not any([d_model, model_name, num_layers, vocab_size])
            d_model = model.config.hidden_size
            num_layers = model.config.num_hidden_layers
            vocab_size = model.config.vocab_size
            assert isinstance(d_model, int) and isinstance(vocab_size, int)

            model_config = model.config.to_dict()  # type: ignore[F841]
            model_name = model.name_or_path

            # Currently we convert the decoder to full precision
            self.unembedding = deepcopy(model.get_output_embeddings()).float()
            if ln := get_final_layer_norm(model):
                self.layer_norm = deepcopy(ln).float()
            else:
                self.layer_norm = th.nn.Identity()

            if extra_layers:
                _, layers = get_transformer_layers(model)
                self.extra_layers.extend(
                    [
                        _BloomBlockWrapper(layer)
                        if isinstance(layer, BloomBlock)
                        else layer
                        for layer in layers[-extra_layers:]
                    ]
                )

            # Annoying special case for OPT
            d_embed = getattr(model.config, "word_embed_proj_dim", None)
            if d_embed and d_embed != d_model:
                proj = model.base_model.decoder.project_out
                assert isinstance(proj, th.nn.Linear)

                U = self.unembedding.weight.data
                self.unembedding.weight.data = U @ proj.weight.data.float()

        # Save config for later
        config_keys = set(inspect.getfullargspec(TunedLens).kwonlyargs)
        self.config = {k: v for k, v in locals().items() if k in config_keys}
        self.dropout = th.nn.Dropout(dropout)
        del model_config

        # Try to prevent finetuning the decoder
        assert d_model and num_layers
        self.layer_norm.requires_grad_(False)
        self.unembedding.requires_grad_(False)

        out_features = d_model if reuse_unembedding else vocab_size

        def create_mlp(hidden_sizes: Sequence[int]) -> th.nn.Sequential:
            sizes = [d_model, *hidden_sizes, out_features]
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
            probe = LowRankLinear(d_model, out_features, rank, bias=bias)
        else:
            probe = th.nn.Linear(d_model, out_features, bias=bias)
            if not reuse_unembedding:
                probe.weight.data = self.unembedding.weight.data.clone()
                probe.bias.data.zero_()
            elif identity_init:
                probe.weight.data.zero_()
                probe.bias.data.zero_()

        self.add_module("input_probe", probe if include_input else None)
        self.attn_probes = th.nn.ModuleList(
            [deepcopy(probe) for _ in range(num_layers)] if sublayers else []
        )
        # Don't include the final layer
        num_layers -= 1

        self.layer_probes = th.nn.ModuleList(
            [deepcopy(probe) for _ in range(num_layers)]
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

        lens = cls(**config)

        if num_extras := config.get("extra_layers"):
            # This is sort of a hack but AutoConfig doesn't appear to have a from_dict
            # for some reason.
            from transformers.models.auto import CONFIG_MAPPING

            model_conf_dict = config.get("model_config")
            del model_conf_dict["torch_dtype"]
            assert model_conf_dict, "Need a 'model_config' entry to load extra layers"

            model_type = model_conf_dict["model_type"]
            config_cls = CONFIG_MAPPING[model_type]
            model_config = config_cls.from_dict(model_conf_dict)

            blocks = []
            for i in range(num_extras):
                layer_idx = model_config.num_hidden_layers - i - 1

                if model_type == "bloom":
                    from transformers.models.bloom.modeling_bloom import BloomBlock

                    block = _BloomBlockWrapper(
                        BloomBlock(model_config)  # type: ignore[arg-type]
                    )
                elif model_type == "gpt_neo":
                    from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock

                    block = GPTNeoBlock(model_config, layer_idx)
                elif model_type == "gpt_neox":
                    from transformers.models.gpt_neox.modeling_gpt_neox import (
                        GPTNeoXLayer,
                    )

                    block = GPTNeoXLayer(model_config)  # type: ignore[arg-type]
                elif model_type == "gpt2":
                    from transformers.models.gpt2.modeling_gpt2 import GPT2Block

                    block = GPT2Block(model_config, layer_idx)  # type: ignore[arg-type]
                elif model_type == "opt":
                    from transformers.models.opt.modeling_opt import OPTDecoderLayer

                    block = OPTDecoderLayer(model_config)  # type: ignore[arg-type]
                else:
                    raise ValueError(f"Unknown model type '{model_type}'")

                blocks.append(block)

            lens.extra_layers = th.nn.Sequential(*blocks)

        lens.load_state_dict(state)
        return lens

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
        if not self.config["reuse_unembedding"]:
            raise RuntimeError("TunedLens.transform_hidden requires reuse_unembedding")

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
        h = self.extra_layers(h)
        while isinstance(h, tuple):
            h, *_ = h

        return self.unembedding(self.layer_norm(h))

    def forward(self, h: th.Tensor, idx: int) -> th.Tensor:
        """Decode hidden states into logits"""
        # Sanity check to make sure we don't finetune the decoder
        # if any(p.requires_grad for p in self.parameters(recurse=False)):
        #     raise RuntimeError("Make sure to freeze the decoder")

        # We're learning a separate unembedding for each layer
        if not self.config["reuse_unembedding"]:
            h_ = self.layer_norm(h)
            return self[idx](h_)

        if self.config["pre_ln"]:
            h_ = self.layer_norm(h)
            return self.unembedding(self[idx](h_))

        h = self.transform_hidden(h, idx)
        return self.to_logits(h)

    def __len__(self) -> int:
        N = len(self.attn_probes) + len(self.layer_probes)
        if self.input_probe:
            N += 1

        return N


# Very annoying that we have to do this. See https://bit.ly/3XSQ7W6 for context on
# what we're doing here.
class _BloomBlockWrapper(th.nn.Module):
    def __init__(self, block: BloomBlock):
        super().__init__()
        self.block = block

    def forward(self, x: th.Tensor) -> th.Tensor:
        from transformers.models.bloom.modeling_bloom import (
            BloomModel,
            build_alibi_tensor,
        )

        batch_size, seq_len, _ = x.shape
        dummy_mask = x.new_ones([batch_size, seq_len])

        # Causal mask isn't created inside the block itself, so we have to do it here.
        # Weirdly _prepare_attn_mask doesn't depend on `self` at all but is still an
        # instance method for some reason, so we pass `None` as the first argument.
        causal_mask = BloomModel._prepare_attn_mask(
            None, dummy_mask, (batch_size, seq_len), 0  # type: ignore[arg-type]
        )
        alibi = build_alibi_tensor(dummy_mask, self.block.num_heads, x.dtype)
        h, *_ = self.block(x, alibi, causal_mask)
        return h
