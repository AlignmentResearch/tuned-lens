"""Utilities for distributed training and handling nested collections of tensors."""

import hashlib
from contextlib import contextmanager
from itertools import islice
from typing import Any, Callable, Iterable, Sequence, Type, TypeVar, Union, cast

import numpy as np
import torch as th
import torch.distributed as dist
from numpy.typing import NDArray

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def maybe_all_cat(x: th.Tensor) -> th.Tensor:
    """Concatenate a tensor across all processes."""
    if not dist.is_initialized():
        return x

    buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
    dist.all_gather_into_tensor(buffer, x)
    return buffer


def maybe_all_gather_lists(lst: list) -> list:
    """Gather a list of objects from all processes."""
    if not dist.is_initialized():
        return lst

    lists = [[] for _ in range(dist.get_world_size())]
    dist.all_gather_object(lists, lst)
    return sum(lists, [])


def maybe_all_reduce(x: th.Tensor, op: str = "mean") -> th.Tensor:
    """Reduce a tensor across all processes."""
    if not dist.is_initialized():
        return x

    if op == "sum":
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    elif op == "mean":
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    else:
        raise ValueError(f"Unknown reduction op '{op}'")

    return x


def maybe_unpack(x):
    """Unpack a tuple if it's a tuple, otherwise return the value."""
    if isinstance(x, tuple):
        x, *_ = x

    return x


def shift_labels(x: th.Tensor, shift: int):
    """Shift labels by a given amount.

    Args:
        x: (batch x seq_len) labels to shift.
        shift: Amount to shift by. Positive values take from the start, negative values
            negative values take from the end.

    Returns:
        (batch x (seq_len - shift)) labels shifted by the given amount.
    """
    if shift > 0:
        return x[:, shift:]
    if shift < 0:
        return x[:, :shift]

    return x


def shift_preds(x: th.Tensor, shift: int):
    """Shift predictions by a given amount.

    Args:
        x: (batch x seq_len) predictions to shift.
        shift: Amount to shift by. Positive values take from the end, negative values
            from the start.

    Returns:
        (batch x (seq_len - shift)) predictions shifted by the given amount.
    """
    if shift > 0:
        return x[:, :-shift]
    if shift < 0:
        return x[:, -shift:]

    return x


T = TypeVar("T")


# Backported from Python 3.10
def pairwise(it: Iterable[T]) -> Iterable[tuple[T, T]]:
    """Iterate over pairs of elements in an iterable."""
    yield from zip(it, islice(it, 1, None))


@contextmanager
def handle_name_conflicts():
    """Provide better error messages."""
    try:
        yield
    except OSError as e:
        raise RuntimeError(
            "HuggingFace is throwing an error during a `from_pretrained` call. Check "
            "your CWD to ensure there are no folders with names that may conflict "
            "with the model name you provided."
        ) from e


# Define pytree type recursively- this works for Pylance but unfortunately not MyPy
AnyTree = Union[th.Tensor, dict[Any, "AnyTree"], list["AnyTree"], tuple["AnyTree", ...]]
TreeType = TypeVar("TreeType", bound=AnyTree)


def pytree_flatten(tree: AnyTree) -> Iterable[th.Tensor]:
    """Recursively iterate over all tensors in a pytree, in topological order."""
    # Stopping condition
    if isinstance(tree, th.Tensor):
        yield tree

    # Recursive case
    elif isinstance(tree, dict):
        for elem in tree.values():
            yield from pytree_flatten(elem)

    elif isinstance(tree, Sequence):
        for elem in tree:
            yield from pytree_flatten(elem)


def pytree_map(
    func: Callable[[th.Tensor], Any], tree: TreeType, strict: bool = True
) -> TreeType:
    """Recursively apply a function to all tensors in a pytree.

    Args:
        func: Function to apply to each tensor.
        tree: Pytree to apply the function to.
        strict: If True, raise an error if a non-tensor leaf is encountered.

    Returns:
        A new pytree with the same structure. Non-tensor leaves are copied.
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


def pytree_cat(trees: Sequence[AnyTree], dim: int = 0) -> AnyTree:
    """Concatenate pytrees along a given dimension.

    All pytrees are expected to use the same collection; undefined behavior
    will occur if this is not the case.

    Args:
        trees: Sequence of pytrees containing tensors to concatenate.
        dim: Dimension to concatenate along.

    Returns:
        - A new pytree with the same structure.
    """
    transposed_iter = zip(*(pytree_flatten(tree) for tree in trees))
    leaf_iter = (th.cat(seq, dim) for seq in transposed_iter)
    try:
        return pytree_map(lambda _: next(leaf_iter), trees[0])  # type: ignore
    except (RuntimeError, StopIteration) as e:
        # Calling next() on an exhausted generator raises a RuntimeError, annoyingly
        if isinstance(e, StopIteration) or "StopIteration" in str(e):
            raise TypeError("All pytrees must have the same structure") from e
        else:
            raise


def pytree_stack(trees: Sequence, dim: int = 0) -> AnyTree:
    """Stack pytrees along a given dimension.

    All pytrees are expected to use the same collection; undefined behavior
    will occur if this is not the case.

    Args:
        trees: Sequence of pytrees containing tensors to stack.
        dim: Dimension to concatenate along.

    Returns:
        A new pytree with the same structure.
    """
    if not len(trees):
        raise ValueError("Cannot stack empty sequence of pytrees")

    transposed_iter = zip(*(pytree_flatten(tree) for tree in trees))
    leaf_iter = (th.stack(seq, dim) for seq in transposed_iter)
    try:
        return pytree_map(lambda _: next(leaf_iter), trees[0])  # type: ignore
    except (RuntimeError, StopIteration) as e:
        # Calling next() on an exhausted generator raises a RuntimeError, annoyingly
        if isinstance(e, StopIteration) or "StopIteration" in str(e):
            raise TypeError("All pytrees must have the same structure") from e
        else:
            raise


def revcumsum(x: Sequence[th.Tensor]) -> list[th.Tensor]:
    """Reverse cumulative sum of a sequence of tensors."""
    if not len(x):
        return []

    running_total = th.zeros_like(x[0])
    sums = [running_total.add_(r).clone() for r in reversed(x)]
    sums.reverse()
    return sums


def send_to_device(tree: TreeType, device: th.device) -> TreeType:
    """Recursively send all tensors in a pytree to a device."""
    return pytree_map(lambda t: t.to(device), tree)


def tensor_hash(tensor: NDArray) -> str:
    """Fast hash of a matrix that is robust to dtype and small perturbations.

    Note this relies on the ordering of the elements in the matrix, so it is
    if the matrix is in any way sorted this will not work well. In addition,
    this hash is intended for large tensors 64 + elements.
    """
    return hashlib.sha256(str.encode(np.array_str(tensor, precision=1))).hexdigest()
