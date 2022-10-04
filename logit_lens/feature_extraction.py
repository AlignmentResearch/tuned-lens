from contextlib import contextmanager
from .model_surgery import get_transformer_layers
from typing import Generator, Type, Union
import torch as th


@contextmanager
def record_residual_stream(
    model: th.nn.Module,
    *,
    include_input: bool = True,
    norm_class: Type[th.nn.Module] = th.nn.LayerNorm,
    post_norm: bool = False,
    retain_grads: bool = False,
    sublayers: bool = False,
) -> Generator[dict[str, th.Tensor], None, None]:
    """Record every state of the residual stream in a transformer forward pass.

    This is a context manager that adds forward hooks to each `nn.LayerNorm` module
    in the transformer, storing the input in a dictionary keyed by the layer norm's
    name. This dictionary is yielded by the context manager for later analysis.

    If you want to record multiple forward passes, you should either call `clear()`
    on the dictionary or create a new `record_residual_stream` context each time.
    """
    hooks = {}
    module_to_name: dict[th.nn.Module, str] = {}
    residual_stream: dict[str, th.Tensor] = {}

    def record_hook(
        module: th.nn.Module, inputs: tuple[th.Tensor], output: Union[th.Tensor, tuple]
    ) -> None:
        module_name = module_to_name.get(module)
        assert module_name is not None, "Forward hook being called on unknown module"

        if module_name in residual_stream:
            raise RuntimeError(
                f"LayerNorm '{module_name}' was called more than once; please "
                f".clear() the stream dictionary"
            )

        # We need to support this output format for HuggingFace models
        if isinstance(output, tuple):
            output = output[0]
            if not isinstance(output, th.Tensor):
                raise RuntimeError(
                    f"Expected first element of {module_name} output to be a Tensor"
                )

        # We're supposed to record the input embeddings and we haven't yet
        if include_input and "input" not in residual_stream:
            record_state(inputs[0], "input")

        # When sublayers=True for pre-LN models, we record the *input* to the second LN
        if isinstance(module, norm_class) and not post_norm:
            assert len(inputs) == 1, "Expected single input to LayerNorm"
            record_state(inputs[0], module_name)
        else:
            record_state(output, module_name)

    def record_state(state: th.Tensor, name: str) -> None:
        if retain_grads:
            state.requires_grad_(True)
            state.retain_grad()
        else:
            state = state.detach()

        residual_stream[name] = state

    _, layers = get_transformer_layers(model)
    for i, layer in enumerate(layers):
        hooks[f"layer{i}"] = layer.register_forward_hook(record_hook)
        module_to_name[layer] = f"layer{i}"

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
            hooks[f"layer{i}_attn"] = post_attn_ln.register_forward_hook(record_hook)
            module_to_name[post_attn_ln] = f"layer{i}_attn"

    try:
        yield residual_stream

    # Make sure we remove the hooks even if an exception is raised.
    finally:
        for hook in hooks.values():
            hook.remove()
