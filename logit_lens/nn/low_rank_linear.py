from typing import Optional
import math
import torch as th


class LowRankLinear(th.nn.Module):
    """Linear layer with low-rank weight matrix."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        device: Optional[th.device] = None,
        dtype: Optional[th.dtype] = None,
    ):
        super().__init__()

        dtype = dtype or th.get_default_dtype()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if rank >= min(in_features, out_features):
            raise ValueError("rank must be less than both in_features and out_features")

        # First sample a full rank matrix using the default nn.Linear init
        full_rank = th.empty(out_features, in_features, device=device, dtype=dtype)
        th.nn.init.kaiming_uniform_(full_rank, a=math.sqrt(5))

        # Then use SVD to get the best low-rank approximation
        u, s, v_t = th.linalg.svd(full_rank)
        half_s, v = s[:rank].sqrt(), v_t.T

        self.u = th.nn.Parameter(u[:, :rank] * half_s)
        self.v = th.nn.Parameter(v[:, :rank] * half_s)

        if bias:
            self.bias = th.nn.Parameter(
                th.zeros(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Make sure to do the matmuls in the efficient order
        return th.nn.functional.linear(x @ self.v, self.u, self.bias)
