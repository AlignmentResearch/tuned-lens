from itertools import islice
from typing import Iterable, TypeVar


# Backported from Python 3.10
T = TypeVar("T")


def pairwise(iterable: Iterable[T]) -> Iterable[tuple[T, T]]:
    """Iterate over pairs of elements in an iterable."""
    yield from zip(iterable, islice(iterable, 1, None))
