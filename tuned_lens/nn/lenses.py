"""Provides lenses for decoding hidden states into logits."""
from copy import deepcopy
import inspect
from logging import warn
from pathlib import Path
import json
import abc

from ._model_specific import instantiate_layer, maybe_wrap
from ..model_surgery import get_final_layer_norm, get_transformer_layers
from ..load_artifacts import load_lens_artifacts
from transformers import PreTrainedModel
from typing import Optional, Generator, Union
import torch as th


class Lens(abc.ABC, th.nn.Module):
    """Abstract base class for all Lens."""

    @abc.abstractmethod
    def forward(self, h: th.Tensor, idx: int) -> th.Tensor:
        """Decode hidden states into logits."""


class LogitLens(Lens):
    """Decodes the residual stream into logits using the unembeding matrix."""

    layer_norm: th.nn.LayerNorm
    unembedding: th.nn.Linear
    extra_layers: th.nn.Sequential

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        *,
        extra_layers: int = 0,
    ):
        """Create a Logit Lens.

        Args:
            model: A pertained model from the transformers library you wish to inspect.
            extra_layers: The number of extra layers to apply to the residual stream
                before decoding into logits.
        """
        super().__init__()

        self.extra_layers = th.nn.Sequential()

        d_model = model.config.hidden_size
        vocab_size = model.config.vocab_size
        assert isinstance(d_model, int) and isinstance(vocab_size, int)

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

        # Try to prevent finetuning the decoder
        self.layer_norm.requires_grad_(False)
        self.unembedding.requires_grad_(False)

    def forward(self, h: th.Tensor, idx: int) -> th.Tensor:
        """Decode a hidden state into logits.

        Args:
            h: The hidden state to decode.
            idx: the layer of the transformer these hidden states come from.
        """
        h = self.extra_layers(h)
        while isinstance(h, tuple):
            h, *_ = h
        return self.unembedding(self.layer_norm(h))


class TunedLens(Lens):
    """A tuned lens for decoding hidden states into logits."""

    layer_norm: th.nn.LayerNorm
    unembedding: th.nn.Linear
    extra_layers: th.nn.Sequential
    layer_translators: th.nn.ModuleList

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        *,
        bias: bool = True,
        extra_layers: int = 0,
        include_input: bool = True,
        reuse_unembedding: bool = True,
        # Used when saving and loading the lens
        model_config: Optional[dict] = None,
        d_model: Optional[int] = None,
        num_layers: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ):
        """Create a TunedLens.

        Args:
            model : A pertained model from the transformers library you wish to inspect.
            bias : Whether to include a bias term in the translator layers.
            extra_layers : The number of extra layers to apply to the hidden states
                before decoding into logits.

            include_input : Whether to include a lens that decodes the word embeddings.
            reuse_unembedding : Weather to reuse the unembedding matrix from the model.
            model_config : The config of the model. Used for saving and loading.
            d_model : The models hidden size. Used for saving and loading.
            num_layers : The number of layers in the model. Used for saving and loading.
            vocab_size : The size of the vocabulary. Used for saving and loading.

        Raises:
            ValueError: if neither a model or d_model, num_layers, and vocab_size,
                are provided.
        """
        super().__init__()

        self.extra_layers = th.nn.Sequential()

        if (
            model
            is None
            == (d_model is None or num_layers is None or vocab_size is None)
        ):
            raise ValueError(
                "Must provide either a model or d_model, num_layers, and vocab_size"
            )

        # Initializing from scratch without a model
        if not model:
            assert d_model and num_layers and vocab_size
            self.layer_norm = th.nn.LayerNorm(d_model)
            self.unembedding = th.nn.Linear(d_model, vocab_size, bias=False)

        # Use HuggingFace methods to get decoder layers
        else:
            assert not (d_model or num_layers or vocab_size)
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
        translator = th.nn.Linear(d_model, out_features, bias=bias)
        if not reuse_unembedding:
            translator.weight.data = self.unembedding.weight.data.clone()
            translator.bias.data.zero_()
        else:
            translator.weight.data.zero_()
            translator.bias.data.zero_()

        self.add_module("input_translator", translator if include_input else None)
        # Don't include the final layer
        num_layers -= 1

        self.layer_translators = th.nn.ModuleList(
            [deepcopy(translator) for _ in range(num_layers)]
        )

    def __getitem__(self, item: int) -> th.nn.Module:
        """Get the probe module at the given index."""
        if isinstance(self.input_translator, th.nn.Module):
            if item == 0:
                return self.input_translator
            else:
                item -= 1

        return self.layer_translators[item]

    def __iter__(self) -> Generator[th.nn.Module, None, None]:
        """Get iterator over the translators within the lens."""
        if isinstance(self.input_translator, th.nn.Module):
            yield self.input_translator

        yield from self.layer_translators

    @classmethod
    def load(cls, resource_id: str, **kwargs) -> "TunedLens":
        """Load a tuned lens from a or hugging face hub.

        Args:
            resource_id : The path to the directory containing the config and checkpoint
                or the name of the model on the hugging face hub.
            **kwargs : Additional arguments to pass to torch.load.

        Returns:
            A TunedLens instance.
        """
        config_path, ckpt_path = load_lens_artifacts(resource_id)
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        # Load parameters
        state = th.load(ckpt_path, **kwargs)

        # Backwards compatibility we really need to stop renaming things
        keys = list(state.keys())
        for key in keys:
            for old_key in ["probe", "adapter"]:
                if old_key in key:
                    warn(
                        f"Loading a checkpoint with a '{old_key}' key. "
                        "This is deprecated and may be removed in a future version. "
                    )
                    new_key = key.replace(old_key, "translator")
                    state[new_key] = state.pop(key)

        # Drop unrecognized config keys
        unrecognized = set(config) - set(inspect.getfullargspec(cls).kwonlyargs)
        for key in unrecognized:
            warn(f"Ignoring config key '{key}'")
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

    def save(
        self,
        path: Union[Path, str],
        ckpt: str = "params.pt",
        config: str = "config.json",
    ) -> None:
        """Save the lens to a directory.

        Args:
            path : The path to the directory to save the lens to.
            ckpt : The name of the checkpoint file to save the parameters to.
            config : The name of the config file to save the config to.
        """
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        th.save(self.state_dict(), path / ckpt)

        with open(path / config, "w") as f:
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

        # Note that we add the translator output residually, in contrast to the formula
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
        """Transform and then decode the hidden states into logits."""
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
        """Return the number of layer translators in the lens."""
        N = len(self.layer_translators)
        if self.input_translator:
            N += 1

        return N
