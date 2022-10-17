from ..residual_stream import record_residual_stream, ResidualStream
from .tuned_lens import TunedLens
from transformers import PreTrainedModel
import torch as th


class TunedLensWrapper(th.nn.Module):
    """Combines a model and its tuned lens into a single module."""

    def __init__(self, model: PreTrainedModel, lens: TunedLens):
        super().__init__()
        self.model = model
        self.lens = lens

    def forward(
        self, input_ids: th.Tensor, **kwargs
    ) -> tuple[th.Tensor, ResidualStream]:
        """Forward pass through the model and tuned lens."""
        use_sublayers = len(self.lens.attn_adapters) > 0

        with record_residual_stream(self.model, sublayers=use_sublayers) as stream:
            output = self.model(input_ids, **kwargs)

        return output, self.lens.apply(stream)
