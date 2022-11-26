from itertools import islice
from typing import Any, Callable, Iterable, TypeVar, Union
import torch as th
import torch.distributed as dist


def maybe_all_cat(x: th.Tensor) -> th.Tensor:
    if not dist.is_initialized():
        return x

    buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
    dist.all_gather_into_tensor(buffer, x)
    return buffer


def maybe_shift_labels(x: th.Tensor, shift: int):
    if shift > 0:
        return x[:, shift:]
    if shift < 0:
        return x[:, :shift]

    return x


def maybe_shift_preds(x: th.Tensor, shift: int):
    if shift > 0:
        return x[:, :-shift]
    if shift < 0:
        return x[:, -shift:]

    return x


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
            f"Found leaf '{tree}' of unsupported type '{type(tree).__name__}'- use "
            f"`strict=False` to ignore"
        )
    else:
        return tree


def send_to_device(tree: TreeType, device: th.device) -> TreeType:
    """Recursively send all tensors in a pytree to a device."""
    return pytree_map(lambda t: t.to(device), tree)
