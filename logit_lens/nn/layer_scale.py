import torch as th


class LayerScale(th.nn.Module):
    """Scales the output of a layer by a trainable scalar."""

    def __init__(self, layer: th.nn.Module, init: float = 1.0):
        super().__init__()

        device = next(layer.parameters()).device
        self.layer = layer
        self.scale = th.nn.Parameter(th.tensor(init, device=device))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layer(x) * self.scale
