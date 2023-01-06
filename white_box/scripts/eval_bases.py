from argparse import Namespace
from datasets import Dataset
from itertools import islice
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from tqdm.auto import tqdm
from white_box.causal import ablate_subspace, CausalBasis
from white_box.utils import maybe_all_reduce, send_to_device
import torch as th
import torch.distributed as dist


@th.autocast("cuda", enabled=th.cuda.is_available())
@th.no_grad()
def eval_bases(
    args: Namespace,
    model: PreTrainedModel,
    data: Dataset,
):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    basis_paths = sorted(
        args.bases.glob("layer_*.pt"),
        key=lambda x: int(x.stem.split("_")[1]),  # Sort numerically
    )
    L = len(basis_paths)
    K = len(th.load(basis_paths[0], map_location="cpu").energies)

    dl = DataLoader(
        data.shuffle(seed=args.seed),  # type: ignore[arg-type],
        batch_size=args.per_gpu_batch_size,
    )
    N = args.limit or len(dl)

    kls = th.zeros(N, L, K, device=th.device(local_rank))
    loss_increments = th.zeros(N, L, K, device=th.device(local_rank))

    pbar = tqdm(desc="Evaluating", position=local_rank, total=N * L * K)
    for batch_idx, batch in enumerate(islice(dl, N)):
        batch = send_to_device(batch, th.device(local_rank))

        for i, basis_path in enumerate(basis_paths):
            basis = th.load(basis_path, map_location=th.device(local_rank))
            assert isinstance(basis, CausalBasis)

            base_outputs = model(**batch, labels=batch["input_ids"])
            log_p = base_outputs.logits.log_softmax(dim=-1)
            p = log_p.exp()
            H = -th.sum(p * log_p, dim=-1).mean()

            for j in range(K):
                with ablate_subspace(model, basis.vectors[:, j], i, orthonormal=True):
                    outputs = model(**batch, labels=batch["input_ids"])
                    log_q = outputs.logits.log_softmax(dim=-1)
                    H_p_q = -th.sum(p * log_q, dim=-1).mean()

                    kls[batch_idx, i, j] = H_p_q - H
                    loss_increments[batch_idx, i, j] = outputs.loss - base_outputs.loss

                pbar.update()

    avg_kls, avg_incrs = kls.mean(0), loss_increments.mean(0)
    maybe_all_reduce(avg_kls, "mean")
    maybe_all_reduce(avg_incrs, "mean")

    if local_rank == 0:
        th.save(avg_kls, args.output / "kls.pt")
        th.save(avg_incrs, args.output / "loss_increments.pt")
