from argparse import Namespace
from datasets import Dataset
from itertools import islice
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from tqdm.auto import tqdm
from white_box.causal import ablate_subspace
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
    assert args.output is not None

    local_rank = dist.get_rank() if dist.is_initialized() else 0
    basis_paths = {int(p.stem.split("_")[1]): p for p in args.bases.glob("layer_*.pt")}
    L = len(basis_paths)
    K = args.k

    dl = DataLoader(
        data.shuffle(seed=args.seed),  # type: ignore[arg-type],
        batch_size=args.per_gpu_batch_size,
    )
    N = args.limit or len(dl)

    kls = th.zeros(N, L, K, device=th.device(local_rank))
    loss_increments = th.zeros(N, L, K, device=th.device(local_rank))

    bases = {
        layer_idx: th.load(path, map_location=th.device(local_rank))
        for layer_idx, path in basis_paths.items()
    }

    def save():
        avg_kls, avg_incrs = kls.mean(0), loss_increments.mean(0)
        maybe_all_reduce(avg_kls, "mean")
        maybe_all_reduce(avg_incrs, "mean")

        if local_rank == 0:
            th.save(avg_kls, args.output / "kls.pt")
            th.save(avg_incrs, args.output / "loss_increments.pt")

    pbar = tqdm(desc="Evaluating", position=local_rank, total=N * L * K)
    for batch_idx, batch in enumerate(islice(dl, N)):
        batch = send_to_device(batch, th.device(local_rank))

        base_outputs = model(**batch, labels=batch["input_ids"])
        log_p = base_outputs.logits.log_softmax(dim=-1)
        p = log_p.exp()
        H = -th.sum(p * log_p, dim=-1).mean()

        for k in range(K):
            for layer_idx, basis in bases.items():
                with ablate_subspace(
                    model, basis.vectors[:, k], layer_idx, orthonormal=True
                ):
                    outputs = model(**batch, labels=batch["input_ids"])
                    log_q = outputs.logits.log_softmax(dim=-1)
                    H_p_q = -th.sum(p * log_q, dim=-1).mean()

                    loss_incr = outputs.loss - base_outputs.loss
                    kls[batch_idx, layer_idx, k] = H_p_q - H
                    loss_increments[batch_idx, layer_idx, k] = loss_incr

                pbar.update()

            save()

    save()
