"""Provides lenses for decoding hidden states into logits."""
from copy import deepcopy
import inspect
from logging import warn
from pathlib import Path
import json
import abc

from ..load_artifacts import load_lens_artifacts
from .unembed import Unembed, UnembedConfig
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Generator, Union
import torch as th


class Lens(abc.ABC, th.nn.Module):
    """Abstract base class for all Lens."""

    @abc.abstractmethod
    def forward(self, h: th.Tensor, idx: int) -> th.Tensor:
        """Decode hidden states into logits."""


class LogitLens(Lens):
    """Decodes the residual stream into logits using the unembeding matrix."""

    unembed: Unembed

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

        self.unembed = Unembed(model, extra_layers=extra_layers)

    def forward(self, h: th.Tensor, idx: int) -> th.Tensor:
        """Decode a hidden state into logits.

        Args:
            h: The hidden state to decode.
            idx: the layer of the transformer these hidden states come from.
        """
        del idx
        return self.unembed.forward(h)


class TunedLens(Lens):
    """A tuned lens for decoding hidden states into logits."""

    unembed: Unembed
    layer_translators: th.nn.ModuleList
    input_translator: Optional[th.nn.Linear]

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        *,
        bias: bool = True,
        include_input: bool = True,
        extra_layers: int = 0,
        # Used when saving and loading the lens
        model_config: Optional[PretrainedConfig] = None,
        unembed_config: Optional[UnembedConfig] = None,
    ):
        """Create a TunedLens.

        Args:
            model : A pertained model from the transformers library you wish to inspect.
            bias : Whether to include a bias term in the translator layers.
            include_input : Whether to include a lens that decodes the word embeddings.
            extra_layers : The number of extra layers to apply to the hidden states
                before decoding into logits.
            model_config : The config of the model. Used for saving and loading.
            unembed_config : The config of the unembeding matrix. Used for saving and

        Raises:
            ValueError: if neither a model or d_model, num_layers, and vocab_size,
                are provided.
        """
        super().__init__()

        self.extra_layers = th.nn.Sequential()

        # Initializing from scratch without a model
        if not (model is None or (model_config is None and unembed_config is None)):
            raise ValueError(
                "Must provide either a model or a model_config and unembed_config."
            )

        if model is not None:
            model_config = model.config
            self.unembed = Unembed(model=model, extra_layers=extra_layers)
        # Use HuggingFace methods to get decoder layers
        elif model_config is not None and unembed_config is not None:
            assert unembed_config is not None
            self.unembed = Unembed(config=unembed_config)
        else:
            raise ValueError(
                "Must provide either a model or a model_config and unembed_config."
            )

        translator = th.nn.Linear(model_config.d_model, model_config.d_model, bias=bias)
        translator.weight.data.zero_()
        translator.bias.data.zero_()

        self.input_translator = translator if include_input else None

        # Don't include the final layer
        num_layers = model_config.num_hidden_layers - 1

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
    def from_pretrained(
        cls,
        resource_id: str,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        **kwargs,
    ) -> "TunedLens":
        """Load a tuned lens from a folder or hugging face hub.

        Args:
            resource_id : The path to the directory containing the config and checkpoint
                or the name of the model on the hugging face hub.
            cache_dir : The directory to cache the artifacts in if downloaded. If None,
                will use the default huggingface cache directory.
            revision : The git revision to use if downloading from the hub.
            **kwargs : Additional arguments to pass to torch.load.

        Returns:
            A TunedLens instance.
        """
        config_path, ckpt_path = load_lens_artifacts(
            resource_id, cache_dir=cache_dir, revision=revision
        )
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        # Load parameters
        state = th.load(ckpt_path, **kwargs)

        # Drop unrecognized config keys
        unrecognized = set(config) - set(inspect.getfullargspec(cls).kwonlyargs)
        for key in unrecognized:
            warn(f"Ignoring config key '{key}'")
            del config[key]

        lens = cls(**config)

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
        # Note that we add the translator output residually, in contrast to the formula
        # in the paper. By parametrizing it this way we ensure that weight decay
        # regularizes the transform toward the identity, not the zero transformation.
        return h + self[idx](h)

    def forward(self, h: th.Tensor, idx: int) -> th.Tensor:
        """Transform and then decode the hidden states into logits."""
        # Sanity check to make sure we don't finetune the decoder
        # if any(p.requires_grad for p in self.parameters(recurse=False)):
        #     raise RuntimeError("Make sure to freeze the decoder")

        h = self.transform_hidden(h, idx)
        return self.unembed(h)

    def __len__(self) -> int:
        """Return the number of layer translators in the lens."""
        N = len(self.layer_translators)
        if self.input_translator:
            N += 1
        return N
