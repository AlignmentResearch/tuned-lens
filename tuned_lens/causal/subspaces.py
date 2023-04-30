"""Provides tools for extracting causal bases from models and ablating subspaces."""
from contextlib import contextmanager
from typing import Iterable, Literal, NamedTuple, Optional, Sequence

import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import trange

from ..model_surgery import get_transformer_layers
from ..nn import Lens
from ..utils import maybe_all_reduce
from .utils import derange


@contextmanager
def ablate_subspace(
    model: th.nn.Module,
    A: th.Tensor,
    layer_index: int,
    mode: Literal["mean", "resample", "zero"] = "zero",
    orthonormal: bool = False,
):
    """Context manager that ablates a subspace of activations.

    Args:
        model: A hugging face transformer model.
        A: Either a 2D matrix whose column space is to be removed, or a 1D vector whose
            span is to be removed.
        layer_index: The index of the layer to ablate.
        mode: Which method to use for removing information along the subspace.
            Defaults to `"zero"`.
        orthonormal: if True, `A` is assumed to be orthonormal.
    """
    _, layers = get_transformer_layers(model)

    def wrapper(_, __, outputs):
        h, *extras = outputs
        h_ = remove_subspace(h, A, mode, orthonormal)

        return h_, *extras

    handle = layers[layer_index].register_forward_hook(wrapper)  # type: ignore
    try:
        yield model
    finally:
        handle.remove()


class CausalBasis(NamedTuple):
    """An ordered orthonormal basis for a subspace of activations.

    Attributes:
        energies: A vector of shape (k,) containing the energies of the
            basis vectors. Each energy is the expected KL divergence of
            the post-intervention logits wrt the control logits when the
            corresponding basis vector is ablated.
        vectors: A matrix of shape (d, k) where d is the ambient dimension
            and k is the dimension of the subspace. The columns of this
            matrix are basis vectors, ordered by decreasing energy.
    """

    energies: th.Tensor
    vectors: th.Tensor


def extract_causal_bases(
    lens: Lens,
    hiddens: Sequence[th.Tensor],
    k: int,
    *,
    labels: Optional[th.Tensor] = None,
    max_iter: int = 100,
    mode: Literal["mean", "resample", "zero"] = "mean",
) -> Iterable[CausalBasis]:
    """Extract causal bases for probes at each layer of a model.

    Args:
        lens: A lens to compute causal bases for.
        hiddens: A sequence of hidden states from the model.
        k: The number of basis vectors to compute for each layer.
        max_iter: The maximum number of iterations to run L-BFGS for each vector.
        mode: Which method to use for removing information along the subspace.
            Defaults to `"zero"`.
    """
    lens.requires_grad_(False)

    device = hiddens[0].device
    dtype = hiddens[0].dtype
    d = hiddens[0].shape[-1]

    hiddens = [h.detach() for h in hiddens]
    num_layers = len(hiddens) - 1

    assert k <= d
    if k < 1:
        k = d

    eye = th.eye(d, device=device, dtype=dtype)

    show_pbar = not dist.is_initialized() or dist.get_rank() == 0
    pbar = trange(num_layers * k) if show_pbar else None

    # Outer loop iterates over layers
    for i in range(num_layers):
        U = lens.unembed.unembedding.weight.data.T

        logits = lens(hiddens[i], i)
        log_p = logits.log_softmax(-1)
        U = lens.transform_hidden(U, i)  # TODO not sure if we need transposes here

        # Compute the baseline loss up front so that we can subtract it
        # from the post-ablation losses to get the loss increment
        if labels is not None:
            base_loss = F.cross_entropy(
                log_p[:, :-1].flatten(0, -2), labels[:, 1:].flatten()
            )
        else:
            base_loss = 0.0

        # Initialize basis vectors with left singular vectors of U
        u, *_ = th.linalg.svd(U, full_matrices=False)
        basis = CausalBasis(th.zeros(k, device=device), u[:, :k].float())

        # Inner loop iterates over directions
        p = log_p.exp()
        for j in range(k):
            if pbar:
                pbar.set_description(f"Layer {i + 1}/{num_layers}, vector {j + 1}/{k}")

            # Construct the operator for projecting away from the previously
            # identified basis vectors
            if j:
                A = basis.vectors[:, :j]
                proj = eye - A @ A.T
            else:
                proj = eye

            def project(x: th.Tensor) -> th.Tensor:
                # Project away from previously identified basis vectors
                x = proj @ x

                # Project to the unit sphere
                return x / (x.norm() + th.finfo(x.dtype).eps)

            basis.vectors[:, j] = project(basis.vectors[:, j])
            v = th.nn.Parameter(basis.vectors[:, j])

            nfev = 0
            energy_delta = th.tensor(0.0, device=device)
            last_energy = th.tensor(0.0, device=device)

            opt = th.optim.LBFGS(
                [v],
                line_search_fn="strong_wolfe",
                max_iter=max_iter,
            )

            def closure():
                nonlocal energy_delta, nfev, last_energy
                nfev += 1

                opt.zero_grad(set_to_none=False)
                v_ = project(v)
                h_ = remove_subspace(hiddens[i], v_, mode=mode, orthonormal=True)

                logits = lens(h_, i)

                if labels is not None:
                    loss = -F.cross_entropy(
                        logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten()
                    )
                else:
                    log_q = logits.log_softmax(-1)
                    loss = -th.sum(p * (log_p - log_q), dim=-1).mean()

                loss.backward()
                maybe_all_reduce(loss)
                maybe_all_reduce(v.grad)  # type: ignore[arg-type]

                assert v.grad is not None
                new_energy = -loss.detach() - base_loss
                energy_delta = new_energy - last_energy
                last_energy = new_energy

                if pbar:
                    pbar.set_postfix(energy=last_energy.item())

                if not loss.isfinite():
                    print("Loss is not finite")
                    loss = th.tensor(0.0, device=device)
                    opt.zero_grad(set_to_none=False)

                return loss

            while nfev < max_iter:
                opt.step(closure)  # type: ignore
                v.data = project(v.data)

                if abs(energy_delta / last_energy) < 1e-4:
                    break

            basis.vectors[:, j] = project(v.data)
            basis.energies[j] = last_energy

            if pbar:
                pbar.update()

        indices = basis.energies.argsort(descending=True)
        yield CausalBasis(basis.energies[indices], basis.vectors[:, indices])


def remove_subspace(
    u: th.Tensor,
    A: th.Tensor,
    mode: Literal["mean", "resample", "zero"] = "zero",
    orthonormal: bool = False,
) -> th.Tensor:
    """Remove all information in `u` along the column space of `A`.

    This can be done by zero, mean, or resample ablation. With zero ablation,
    `u` is projected onto the orthogonal complement of col(`A`), so the resulting
    vectors are orthogonal to every column in `A`. With mean ablation, `u` is projected
    onto the subspace s.t. the angles between the resulting vectors and the columns of
    `A` are equal to their mean values. With resample ablation, the variation in `u`
    is shuffled across vectors.

    Args:
        u: The vectors to be projected.
        A: Either a 2D matrix whose column space is to be removed, or a 1D vector whose
            span is to be removed.
        mode: Which method to use for removing information along the subspace.
            Defaults to `"zero"`.
        orthonormal: Whether to assume `A` is orthonormal. Defaults to `False`.

    Returns:
        th.Tensor: The transformed vectors.
    """
    if A.ndim == 1:
        A = A[..., None]

    d, _ = A.shape
    if u.shape[-1] != d:
        raise ValueError(f"Last dimension of u must be {d}, but is {u.shape[-1]}")

    # https://en.wikipedia.org/wiki/Projection_(linear_algebra)#Properties_and_special_cases
    if orthonormal:
        proj = A @ A.mT
    else:
        proj = A @ th.linalg.solve(A.mT @ A, A.mT)

    if mode == "zero":
        dummy = -u
    else:
        samples = u.flatten(0, -2)
        N = samples.shape[0]
        if N < 2:
            raise ValueError("Need at least 2 vectors for mean and resample ablation")

        if mode == "mean":
            dummy = samples.mean(0) - u
        elif mode == "resample":
            # Shuffle the rows of `samples` without fixed points.
            dummy = derange(samples).view_as(u) - u
        else:
            raise ValueError(f"Unknown mode {mode}")

    return u + th.einsum("ij,...j->...i", proj, dummy)
