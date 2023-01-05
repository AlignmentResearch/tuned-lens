from argparse import Namespace
from datasets import Dataset
from white_box import TunedLens
from white_box.causal import extract_causal_bases
from white_box.utils import send_to_device
import torch as th
import torch.distributed as dist


def extract_bases(
    args: Namespace,
    model: th.nn.Module,
    data: Dataset,
    lens: TunedLens,
):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    args.output.mkdir(parents=True, exist_ok=True)

    data = data.shuffle(seed=args.seed)  # type: ignore[arg-type]
    batch = send_to_device(data[: args.per_gpu_batch_size], th.device(local_rank))

    with th.autocast("cuda"), th.no_grad():
        outputs = model(**batch, output_hidden_states=True)

    basis_iter = extract_causal_bases(
        # Unfortunately I can't get this to work in half precision
        lens.float(),
        [x.float() for x in outputs.hidden_states[:-1]],
        k=args.k,
    )
    for i, basis in enumerate(basis_iter):
        if local_rank == 0:
            th.save(basis, args.output / f"layer_{i}.pt")
