from contextlib import contextmanager
from typing import Generator, Optional, Type
import torch as th


@contextmanager
def record_residual_stream(
    model: th.nn.Module,
    *,
    input_name: str = "input",
    norm_class: Type[th.nn.Module] = th.nn.LayerNorm,
    post_norm: bool = False,
    retain_grads: bool = False,
) -> Generator[dict[str, th.Tensor], None, None]:
    """Record every state of the residual stream in a transformer forward pass.

    This is a context manager that adds forward hooks to each `nn.LayerNorm` module
    in the transformer, storing the input in a dictionary keyed by the layer norm's
    name. This dictionary is yielded by the context manager for later analysis.

    If you want to record multiple forward passes, you should either call `clear()`
    on the dictionary or create a new `record_residual_stream` context each time.
    """
    hooks = {}
    last_module_name: Optional[str] = None
    module_to_name: dict[th.nn.Module, str] = {}
    residual_stream: dict[str, th.Tensor] = {}

    def record_hook(
        module: th.nn.Module, inputs: tuple[th.Tensor], output: th.Tensor
    ) -> None:
        module_name = module_to_name.get(module)

        assert module_name is not None, "Forward hook being called on unknown module"
        assert len(inputs) == 1, "Expected single input to LayerNorm"

        if module_name in residual_stream:
            raise RuntimeError(
                f"LayerNorm '{module_name}' was called more than once; please "
                f".clear() the stream dictionary"
            )

        # For post-norm architectures, we just record the output of the layer norm
        if post_norm:
            state_name = module_name
            stream_state = output
        else:
            (stream_state,) = inputs

            # If this is a pre-norm architecture, we need to record the *input* to
            # this LayerNorm under the name of the *previous* layer
            nonlocal last_module_name
            if last_module_name is None:
                # Special case for the first layer
                state_name = input_name
            else:
                state_name = last_module_name

            # Remember the name of this module for the next iteration
            last_module_name = module_name

        if retain_grads:
            stream_state.requires_grad_(True)
            stream_state.retain_grad()
        else:
            stream_state = stream_state.detach()

        residual_stream[state_name] = stream_state

    for name, module in model.named_modules():
        if not isinstance(module, norm_class):
            continue

        hooks[name] = module.register_forward_hook(record_hook)
        module_to_name[module] = name

    try:
        yield residual_stream

    # Make sure we remove the hooks even if an exception is raised.
    finally:
        for hook in hooks.values():
            hook.remove()
