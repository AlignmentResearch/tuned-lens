from contextlib import ContextDecorator, redirect_stdout
from itertools import islice
from typing import Any, Callable, Iterable, TypeVar, Union
import torch as th


T = TypeVar("T")

# Backported from Python 3.10
def pairwise(iterable: Iterable[T]) -> Iterable[tuple[T, T]]:
    """Iterate over pairs of elements in an iterable."""
    yield from zip(iterable, islice(iterable, 1, None))


# Define pytree type recursively- this works for Pylance but unfortunately not MyPy
AnyTree = Union[th.Tensor, dict[Any, "AnyTree"], list["AnyTree"], tuple["AnyTree", ...]]
TreeType = TypeVar("TreeType", bound=AnyTree)


def pytree_map(
    func: Callable[[th.Tensor], Any], tree: TreeType, strict: bool = True
) -> TreeType:
    """
    Recursively apply a function to all tensors in a pytree, returning the results
    in a new pytree with the same structure. Non-tensor leaves are copied.
    """
    # Stopping condition
    if isinstance(tree, th.Tensor):
        return func(tree)

    # Recursive case
    if isinstance(tree, dict):
        return {k: pytree_map(func, v) for k, v in tree.items()}

    if isinstance(tree, list):
        return [pytree_map(func, v) for v in tree]

    if isinstance(tree, tuple):
        return tuple(pytree_map(func, v) for v in tree)

    if strict:
        raise TypeError(
            f"Found leaf '{tree}' of unsupported type '{type(tree).__name__}'- use `strict=False` to ignore"
        )
    else:
        return tree


def send_to_device(tree: TreeType, device: th.device) -> TreeType:
    """Recursively send all tensors in a pytree to a device."""
    return pytree_map(lambda t: t.to(device), tree)


def toggle_stdout(enable: bool):
    """Decorator to temporarily enable or disable stdout."""

    class _redirect_decorator(ContextDecorator, redirect_stdout):
        pass

    return _redirect_decorator(None) if not enable else lambda x: x
