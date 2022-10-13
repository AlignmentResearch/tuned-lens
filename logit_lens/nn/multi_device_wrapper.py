from copy import deepcopy
import torch as th


class MultiDeviceWrapper:
    """Lazily loads a module multiple times on different devices.

    This is intentionally NOT a subclass of `th.nn.Module`, so that it doesn't
    get saved in the state dict of its parent module.
    """

    def __init__(self, module: th.nn.Module):
        first_device = next(module.parameters()).device
        self._copies = {first_device: module}

    def __call__(self, x: th.Tensor, **kwargs) -> th.Tensor:
        # Check for a cached copy
        local_copy = self._copies.get(x.device)
        if local_copy is None:
            # Create a copy on the new device
            remote_copy = next(iter(self._copies.values()))
            local_copy = deepcopy(remote_copy).to(x.device)

            # Cache for later
            self._copies[x.device] = local_copy

        return local_copy(x, **kwargs)
