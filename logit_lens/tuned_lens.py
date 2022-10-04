from copy import deepcopy
from transformers import PreTrainedModel
from typing import Iterable, Optional
import torch as th


class TunedLens(th.nn.Module):
    """Stores all parameters necessary to decode hidden states into logits.

    There are three possible ways to initialize this class:
    1. From a HuggingFace model
    2. From a manually specified unembedding layer
    3. From scratch, with manually specified dimensions

    The third option is mainly supported for testing purposes, and to make
    serialization easier.
    """

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
        self.adapters = th.nn.ModuleDict()

        def create_adapter(name: str):
            lens = th.nn.Linear(d_model, d_model, bias=bias)
            if identity_init:
                lens.weight.data = th.eye(d_model)  # Initialize with identity matrix
                lens.bias.data.zero_()

            # Enforce orthogonality with matrix exponential parametrization
            if orthogonal:
                lens = th.nn.utils.parametrizations.orthogonal(lens)

            self.adapters[name] = lens

        if include_input:
            create_adapter("input")

        for i in range(num_layers):
            if sublayers:
                create_adapter(f"layer{i}_attn")

            create_adapter(f"layer{i}")

    def iter_logits(
        self, hiddens: Iterable[th.Tensor], tuned: bool = True
    ) -> Iterable[tuple[str, th.Tensor]]:
        """Yield the logits for each hidden state in an iterable."""
        # Sanity check to make sure we don't finetune the decoder
        if self.unembedding.weight.requires_grad:
            raise RuntimeError("Make sure to freeze the decoder")

        for (name, adapter), h in zip(self.adapters.items(), hiddens):
            h = self.layer_norm(h)
            yield name, self.unembedding(adapter(h) if tuned else h)

    def forward(self, hiddens: list[th.Tensor]) -> list[th.Tensor]:
        """Decode hidden states into logits"""
        return [logits for _, logits in self.iter_logits(hiddens)]
