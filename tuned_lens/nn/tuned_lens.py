from copy import deepcopy
from pathlib import Path

from ._model_specific import instantiate_layer, maybe_wrap
from ..model_surgery import get_final_layer_norm, get_transformer_layers
from transformers import PreTrainedModel
from typing import Generator, Optional, Union
import inspect
import json
import torch as th


class TunedLens(th.nn.Module):
    """Stores all parameters necessary to decode hidden states into logits."""

    layer_norm: th.nn.LayerNorm
    unembedding: th.nn.Linear

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        *,
        bias: bool = True,
        extra_layers: int = 0,
        include_input: bool = True,
        model_config: Optional[dict] = None,
        reuse_unembedding: bool = True,
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
            self.unembedding = th.nn.Linear(d_model, vocab_size, bias=False)

        # Use HuggingFace methods to get decoder layers
        else:
            assert not any([d_model, num_layers, vocab_size])
            d_model = model.config.hidden_size
            num_layers = model.config.num_hidden_layers
            vocab_size = model.config.vocab_size
            assert isinstance(d_model, int) and isinstance(vocab_size, int)

            model_config = model.config.to_dict()  # type: ignore[F841]

            # Currently we convert the decoder to full precision
            self.unembedding = deepcopy(model.get_output_embeddings()).float()
            if ln := get_final_layer_norm(model):
                self.layer_norm = deepcopy(ln).float()
            else:
                self.layer_norm = th.nn.Identity()

            if extra_layers:
                _, layers = get_transformer_layers(model)
                self.extra_layers.extend(
                    [maybe_wrap(layer) for layer in layers[-extra_layers:]]
                )

        # Save config for later
        config_keys = set(inspect.getfullargspec(TunedLens).kwonlyargs)
        self.config = {k: v for k, v in locals().items() if k in config_keys}
        del model_config

        # Try to prevent finetuning the decoder
        assert d_model and num_layers
        self.layer_norm.requires_grad_(False)
        self.unembedding.requires_grad_(False)

        out_features = d_model if reuse_unembedding else vocab_size
        adapter = th.nn.Linear(d_model, out_features, bias=bias)
        if not reuse_unembedding:
            adapter.weight.data = self.unembedding.weight.data.clone()
            adapter.bias.data.zero_()
        else:
            adapter.weight.data.zero_()
            adapter.bias.data.zero_()

        self.add_module("input_adapter", adapter if include_input else None)
        # Don't include the final layer
        num_layers -= 1

        self.layer_adapters = th.nn.ModuleList(
            [deepcopy(adapter) for _ in range(num_layers)]
        )

    def __getitem__(self, item: int) -> th.nn.Module:
        """Get the probe module at the given index."""
        if isinstance(self.input_adapter, th.nn.Module):
            if item == 0:
                return self.input_adapter
            else:
                item -= 1

        return self.layer_adapters[item]

    def __iter__(self) -> Generator[th.nn.Module, None, None]:
        if isinstance(self.input_adapter, th.nn.Module):
            yield self.input_adapter

        yield from self.layer_adapters

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

        # Backwards compatibility
        keys = list(state.keys())
        for key in keys:
            if "probe" in key:
                new_key = key.replace("probe", "adapter")
                state[new_key] = state.pop(key)

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

            lens.extra_layers = th.nn.Sequential(
                *[
                    instantiate_layer(
                        model_config, model_config.num_hidden_layers - i - 1, model_type
                    )
                    for i in range(num_extras)
                ]
            )

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

        for linear in self:
            assert isinstance(linear, th.nn.Linear)

            A, b = linear.weight.data, linear.bias.data
            A -= A.mean(dim=0, keepdim=True)
            b -= b.mean()

    def transform_hidden(self, h: th.Tensor, idx: int) -> th.Tensor:
        """Transform hidden state from layer `idx`."""
        if not self.config["reuse_unembedding"]:
            raise RuntimeError("TunedLens.transform_hidden requires reuse_unembedding")

        # Note that we add the adapter output residually, in contrast to the formula
        # in the paper. By parametrizing it this way we ensure that weight decay
        # regularizes the transform toward the identity, not the zero transformation.
        return h + self[idx](h)

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

        h = self.transform_hidden(h, idx)
        return self.to_logits(h)

    def __len__(self) -> int:
        N = len(self.layer_adapters)
        if self.input_adapter:
            N += 1

        return N
