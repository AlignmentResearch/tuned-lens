from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import zip_longest
from .model_surgery import get_transformer_layers
from typing import Callable, Generator, Optional, overload, Type, Union
import torch as th


@dataclass
class ResidualStream:
    """Collection of transformer hidden states, or derived quantities."""

    embeddings: Optional[th.Tensor] = None
    attentions: list[th.Tensor] = field(default_factory=list)
    layers: list[th.Tensor] = field(default_factory=list)

    @classmethod
    def from_hf(cls, hiddens: list[th.Tensor]) -> "ResidualStream":
        """Create a `ResidualStream` from Huggingface hidden states."""
        embeddings, *layers = hiddens
        return cls(embeddings=embeddings, layers=layers)

    def clear(self) -> None:
        """Clear all residual states."""
        self.embeddings = None
        self.attentions.clear()
        self.layers.clear()

    @property
    def has_sublayers(self) -> bool:
        """Return whether the stream contains both attention and layer outputs."""
        return bool(self.attentions) and bool(self.layers)

    def items(self) -> Generator[tuple[str, th.Tensor], None, None]:
        """Iterate over residual states in topological order, yielding labels."""
        if self.embeddings is not None:
            yield "input", self.embeddings

        # Yield attention and layer outputs in alternating order
        for i, (attn, layer) in enumerate(zip_longest(self.attentions, self.layers)):
            if attn is not None:
                yield f"{i}.attn", attn
            if layer is not None:
                yield f"{i}.ffn", layer

    def labels(self) -> list[str]:
        """Return labels for the residual states suitable for e.g. plotting."""
        return [label for label, _ in self.items()]

    def map(self, fn: Callable) -> "ResidualStream":
        """Map a function over all states, returning a new `ResidualStream`."""
        return ResidualStream(
            embeddings=fn(self.embeddings) if self.embeddings is not None else None,
            attentions=[fn(t) for t in self.attentions],
            layers=[fn(t) for t in self.layers],
        )

    def pairwise_map(
        self, fn: Callable[[th.Tensor, th.Tensor], th.Tensor]
    ) -> "ResidualStream":
        """Map over adjacent pairs of states, returning a new `ResidualStream`."""
        if self.embeddings is None:
            raise ValueError("Can't map pairwise without input embeddings")

        states = list(self)
        results = [fn(s1, s2) for s1, s2 in zip(states[:-1], states[1:])]
        if self.has_sublayers:
            return ResidualStream(attentions=results[::2], layers=results[1::2])
        else:
            return ResidualStream(layers=results)

    def plot(self, tick_spacing: int = 2):
        """Plot the residual states."""
        import matplotlib.pyplot as plt

        plt.plot(self)
        if not self.has_sublayers:
            plt.xlabel("Layer")
        else:
            plt.xlabel("Sublayer")
            plt.xticks(
                labels=[
                    l for i, l in enumerate(self.labels()) if i % tick_spacing == 0
                ],
                ticks=range(0, len(self), tick_spacing),
                rotation=60,
            )

    def residuals(self) -> "ResidualStream":
        """Compute residual (hidden state diff) for each block."""
        return self.pairwise_map(lambda s1, s2: s2 - s1)

    def __contains__(self, item: th.Tensor) -> bool:
        """Check if the stream contains a reference to this exact tensor."""
        return any(item.is_set_to(state) for state in self)

    @overload
    def __getitem__(self, item: slice) -> list[th.Tensor]:
        ...

    @overload
    def __getitem__(self, item: int) -> th.Tensor:
        ...

    def __getitem__(self, item: Union[int, slice]):
        """Return the state at the given index, or with the given name."""
        return list(self)[item]

    def __iter__(self) -> Generator[th.Tensor, None, None]:
        """Iterate over residual states in topological order."""
        for _, state in self.items():
            yield state

    def __len__(self) -> int:
        """Return the number of residual states."""
        num_states = len(self.attentions) + len(self.layers)
        if self.embeddings is not None:
            num_states += 1

        return num_states


@contextmanager
def record_residual_stream(
    model: th.nn.Module,
    *,
    include_input: bool = True,
    norm_class: Type[th.nn.Module] = th.nn.LayerNorm,
    post_norm: bool = False,
    retain_grads: bool = False,
    sublayers: bool = True,
) -> Generator[ResidualStream, None, None]:
    """Record every state of the residual stream in a transformer forward pass.

    This is a context manager that adds forward hooks to each `nn.LayerNorm` module
    in the transformer, storing the input in a dictionary keyed by the layer norm's
    name. This dictionary is yielded by the context manager for later analysis.

    If you want to record multiple forward passes, you should either call `clear()`
    on the stream object or create a new `record_residual_stream` context each time.
    """
    hooks = []
    residual_stream = ResidualStream()

    def process(state: th.Tensor) -> th.Tensor:
        if retain_grads:
            state.requires_grad_(True)
            state.retain_grad()
        else:
            state = state.detach()

        return state

    def store_embeddings(module: th.nn.Module, inputs) -> None:
        residual_stream.embeddings = process(inputs[0])

    def store_attn(module: th.nn.Module, inputs) -> None:
        residual_stream.attentions.append(process(inputs[0]))

    def store_layer(module: th.nn.Module, inputs, output: Union[th.Tensor, tuple]):
        # HuggingFace layers usually return tuples
        if isinstance(output, tuple):
            output = output[0]
            if not isinstance(output, th.Tensor):
                idx = len(residual_stream.layers)
                raise RuntimeError(
                    f"Expected first element of layer {idx} output to be a Tensor"
                )

        residual_stream.layers.append(output)

    _, layers = get_transformer_layers(model)
    if include_input:
        hooks.append(layers[0].register_forward_pre_hook(store_embeddings))

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(store_layer))

        # For sublayers=True, we need to hook into one of the layer norms in this layer
        if sublayers:
            layer_norms = [m for m in layer.modules() if isinstance(m, norm_class)]
            if not layer_norms:
                raise ValueError(
                    f"No layer norms found in layer {i}; try specifying `norm_class`"
                )
            elif len(layer_norms) != 2:
                raise ValueError(
                    f"Expected 2 layer norms per layer when sublayers=True, "
                    f"but layer {i} has {len(layer_norms)}"
                )

            post_attn_ln = layer_norms[0 if post_norm else 1]
            hooks.append(post_attn_ln.register_forward_pre_hook(store_attn))

    try:
        yield residual_stream

    # Make sure we remove the hooks even if an exception is raised.
    finally:
        for hook in hooks:
            hook.remove()
