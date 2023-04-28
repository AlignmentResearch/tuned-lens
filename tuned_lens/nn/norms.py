"""Provides a shared interface for transformer norms."""
import abc
from typing import Optional

import torch as th
from torch import nn
import torch.nn.functional as F


class Norm(nn.Module, abc.ABC):
    """A common interface for transformer norms."""

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Apply the complete norm.

        Args:
            x: The input tensor to apply the norm to.

        Returns:
            The result of applying the norm to the input tensor.
        """
        return self.non_linear_part(x) * self.rescale + self.shift

    @abc.abstractproperty
    def rescale(self) -> th.Tensor:
        """The weight associated with the scaling step after the nonlinearity."""
        ...

    @abc.abstractproperty
    def shift(self) -> th.Tensor:
        """The bias term applied after the nonlinearity and th rescale."""
        ...

    @abc.abstractmethod
    def non_linear_part(self, x: th.Tensor) -> th.Tensor:
        """The non-linear part of the norm applied to the input tensor.

        Args:
            x: The input tensor to apply the non-linear part of the norm to.

        Returns:
            The result of applying the non-linear part of the norm to the input tensor.
        """
        ...


class LayerNorm(Norm):
    """A layer norm that can be applied to a tensor."""

    def __init__(
        self,
        normalized_shape: list[int],
        weight: Optional[th.Tensor] = None,
        bias: Optional[th.Tensor] = None,
        eps: float = 1.0e-6,
    ):
        """Create a layer norm.

        Args:
            normalized_shape: The shape of the final part of the tensor to
                be normalized.
            weight: The weight tensor associated with the rescale.
                If not provided, it will be initialized to ones.
            bias: The bias tensor associated with the shift.
                If not provided, it will be initialized to zeros.
            eps: The epsilon value used in the layer normalization.
                Defaults to 1.0e-6.
        """
        super().__init__()
        self.normalized_shape = normalized_shape

        if weight is None:
            self._weight = nn.Parameter(th.ones(normalized_shape))
        else:
            self._weight = nn.Parameter(weight.clone())

        if bias is None:
            self._bias = nn.Parameter(th.zeros(normalized_shape))
        else:
            self._bias = nn.Parameter(bias.clone())
        self.eps = eps

    @property
    def shift(self) -> th.Tensor:
        """Get the shift tensor."""
        return self._bias

    @property
    def rescale(self) -> th.Tensor:
        """Get the rescale tensor."""
        return self._weight

    def non_linear_part(self, x: th.Tensor) -> th.Tensor:
        """Apply the layer normalization to the input tensor."""
        return F.layer_norm(x, self.normalized_shape, None, None, self.eps)


class RMSNorm(Norm):
    """A root mean square normalization that can be applied to a tensor."""

    def __init__(self, hidden_size, weight: Optional[th.Tensor] = None, eps=1e-6):
        """Create a root mean square normalization.

        Args:
            hidden_size: The size of the hidden state tensor to be normalized.
            weight: The weight tensor associated with the rescale. If not provided, it
                will be initialized to ones.
            eps: The epsilon value used in the root mean square normalization.
                Defaults to 1.0e-6.
        """
        super().__init__()
        if weight is None:
            self._weight = nn.Parameter(th.ones(hidden_size))
        else:
            self._weight = nn.Parameter(weight.clone())

        self._bias = nn.Parameter(th.zeros(hidden_size), requires_grad=False)

        self.eps = eps

    @property
    def rescale(self) -> th.Tensor:
        """Get the rescale weight tensor."""
        return self._weight

    @property
    def shift(self) -> th.Tensor:
        """Get the shift bias tensor."""
        return self._bias

    def non_linear_part(self, hidden_states: th.Tensor) -> th.Tensor:
        """Apply the root mean square normalization to the input tensor.

        Args:
            hidden_states: The input tensor to apply the root mean square
                normalization to.

        Returns:
            The result of applying the root mean square normalization to the
                input tensor.
        """
        dtype = hidden_states.dtype

        variance = hidden_states.to(th.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * th.rsqrt(variance + self.eps)
        return hidden_states.to(dtype)
