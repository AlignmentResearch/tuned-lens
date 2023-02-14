from contextlib import contextmanager
from typing import Generator
from tuned_lens.model_surgery import get_key_path
import torch as th


def mangle(x: str) -> str:
    return x.replace(".", "-")


def unmangle(x: str) -> str:
    return x.replace("-", ".")


class ProbeDict(th.nn.ModuleDict):
    """Collection of probes attached to a model, identified by key paths.

    This is a subclass of `nn.ModuleDict` that allows paths like "transformer.h.11"
    to be used as keys.
    """

    def forward(
        self, x: dict[str, th.Tensor], strict: bool = True
    ) -> dict[str, th.Tensor]:
        """Apply each probe to its corresponding activation in `x`.

        When `strict=True`, this method requires that the keys of `x` are equal to the
        keys in this `ProbeDict`. Otherwise, a `ValueError` will be raised.
        """
        try:
            outputs = {k: self[k](v) for k, v in x.items()}
        except KeyError as e:
            raise ValueError(f"No probe for module named '{e}'") from e

        # We didn't error earlier, so we know all keys in `x` are in `self`. If
        # `strict` is True, we also require that `x` isn't missing any keys in `self`.
        if strict and len(outputs) < len(self):
            missing = set(self.keys()) - set(x.keys())
            raise ValueError("Missing values for modules: " + ", ".join(missing))

        return outputs

    def maybe_map(self, x: dict[str, th.Tensor]) -> dict[str, th.Tensor]:
        """Alias for `forward` with `strict=False`."""
        return self(x, strict=False)

    @contextmanager
    def record(
        self, model: th.nn.Module
    ) -> Generator[dict[str, th.Tensor], None, None]:
        """Apply probes to the corresponding activations in the given model.

        Args:
            model: The model to record activations from.

        Yields:
            A dictionary mapping probe names to transformed activations.
        """
        outputs = {}
        handles = []

        for name, probe in self.items():

            def hook(module, inputs, outputs):
                x, *_ = outputs
                outputs[name] = probe(x)

            module = get_key_path(model, name)
            handles.append(module.register_forward_hook(hook))

        try:
            yield outputs
        finally:
            for handle in handles:
                handle.remove()

    def items(self) -> Generator[tuple[str, th.nn.Module], None, None]:
        for key, value in super().items():
            yield unmangle(key), value

    def __getitem__(self, key: str) -> th.nn.Module:
        return super().__getitem__(mangle(key))

    def __repr__(self):
        return unmangle(super().__repr__())

    def __setitem__(self, key: str, module: th.nn.Module) -> None:
        # We don't allow names of modules to contain hyphens, because we use them
        # as a replacement for periods in key paths.
        if "-" in key:
            raise ValueError("Module names cannot contain hyphens")

        return super().__setitem__(mangle(key), module)
