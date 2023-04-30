import numpy as np

from tuned_lens.utils import tensor_hash


def test_tensor_hash():
    random = np.random.default_rng(42)
    a = random.normal(size=(10, 1000)).astype(np.float32)
    b = random.normal(size=(10, 1000)).astype(np.float32)
    assert tensor_hash(a) != tensor_hash(b)
    assert tensor_hash(a) == tensor_hash(a)
    assert tensor_hash(a) == tensor_hash(a.astype(np.float16))
