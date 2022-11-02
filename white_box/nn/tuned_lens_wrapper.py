from ..residual_stream import record_residual_stream
from .tuned_lens import TunedLens
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
import torch as th


class TunedLensWrapper(th.nn.Module):
    """Combines a model and its tuned lens into a single module."""

    def __init__(self, model: PreTrainedModel, lens: TunedLens):
        super().__init__()

        model.requires_grad_(False)
        self.model = model
        self.lens = lens

    def forward(self, input_ids: th.Tensor, **kwargs) -> ModelOutput:
        """Forward pass through the model and extract hiddens."""
        use_sublayers = len(self.lens.attn_adapters) > 0
        if use_sublayers:
            with record_residual_stream(self.model, sublayers=True) as stream:
                output = self.model(input_ids, **kwargs)

            output.hidden_states = list(stream)
            return output

        return self.model(input_ids, output_hidden_states=True, **kwargs)
