from argparse import Namespace
from datasets import Dataset
from collections import defaultdict
from itertools import islice
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from white_box import (
    record_residual_stream,
    LogitStats,
    ResidualStats,
    ResidualStream,
    TunedLens,
)
from white_box.utils import (
    maybe_all_reduce,
    maybe_shift_labels,
    maybe_shift_preds,
    pytree_map,
    pytree_stack,
    send_to_device,
)
import torch as th
import torch.distributed as dist


@th.autocast("cuda", enabled=th.cuda.is_available())
@th.no_grad()
def eval_loop(
    args: Namespace,
    model: th.nn.Module,
    data: Dataset,
    lens: TunedLens,
    nats_to_bpb_ratio: float,
):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    dl = DataLoader(
        data.shuffle(seed=args.seed),  # type: ignore[arg-type],
        batch_size=args.per_gpu_batch_size,
    )
    lens.eval()

    # Running mean & covariance of the hidden states
    first_token_stats = ResidualStats()
    delta_stats = ResidualStats()
    stream_stats = ResidualStats()

    if args.limit:
        dl = islice(dl, args.limit)
        total = args.limit
    else:
        total = len(dl)

    root_dir = args.output or args.lens / "eval"
    output_dir = root_dir / f"rank_{local_rank}"
    output_dir.mkdir(exist_ok=True, parents=True)

    L = len(lens)
    batches = []
    transfer_batches = []

    final_logit_stats = LogitStats()
    ll_logit_statistics = [LogitStats() for _ in range(L)]
    tl_logit_statistics = [LogitStats() for _ in range(L)]

    pbar = tqdm(dl, desc="Evaluating", position=local_rank, total=total)
    for batch in pbar:
        batch = send_to_device(batch, th.device(local_rank))
        output = model(**batch, output_hidden_states=True)
        stream = ResidualStream(
            embeddings=output.hidden_states[0], layers=output.hidden_states[1:-1]
        )
        shift = args.token_shift if args.token_shift is not None else 1

        final_lps = output.logits.log_softmax(dim=-1)
        final_probs = final_lps.exp()
        labels = maybe_shift_labels(batch["input_ids"], shift)

        batch_output = defaultdict(dict)
        transfer_ces = th.zeros(L, L, device=final_lps.device)
        transfer_kls = th.zeros(L, L, device=final_lps.device)

        for j, (name, h) in zip(range(L), stream.items()):
            baseline_lps = lens.to_logits(h).log_softmax(dim=-1)
            ll_logit_statistics[j].update(baseline_lps, assume_normalized=True)

            batch_output["baseline_ce"][name] = th.nn.functional.cross_entropy(
                maybe_shift_preds(baseline_lps, shift).flatten(0, 1),
                labels.flatten(),
                reduction="none",
            )
            batch_output["baseline_entropy"][name] = th.sum(
                -baseline_lps.exp() * baseline_lps, dim=-1
            )

            lens_lps = lens(h, idx=j).log_softmax(dim=-1)
            batch_output["lens_ce"][name] = th.nn.functional.cross_entropy(
                maybe_shift_preds(lens_lps, shift).flatten(0, 1),
                labels.flatten(),
                reduction="none",
            )
            batch_output["lens_entropy"][name] = th.sum(
                -lens_lps.exp() * lens_lps, dim=-1
            )
            batch_output["lens_kl"][name] = th.sum(
                final_probs * (final_lps - lens_lps), dim=-1
            )
            tl_logit_statistics[j].update(lens_lps, assume_normalized=True)

            if args.transfer:
                # Probs from the probe that was trained and tested on layer j.
                diag_probs = lens_lps.exp()

                # Each iteration of the loop processes a different *probe* layer i
                # for the test layer j.
                for i in range(L):
                    transfer_lps = lens(h, idx=i).log_softmax(dim=-1)
                    transfer_ces[i, j] = th.nn.functional.cross_entropy(
                        maybe_shift_preds(transfer_lps, shift).flatten(0, 1),
                        labels.flatten(),
                    )
                    transfer_kls[i, j] = th.sum(
                        diag_probs * (lens_lps - transfer_lps), dim=-1
                    ).mean()

        final_logit_stats.update(final_lps, assume_normalized=True)
        if args.residual_stats:
            first_tokens = stream.map(lambda x: x[:, 0])
            rest = stream.map(lambda x: x[:, 1:])

            first_token_stats.update(first_tokens)
            delta_stats.update(rest.residuals())
            stream_stats.update(rest)

        batch_output["baseline_ce"]["final"] = th.nn.functional.cross_entropy(
            maybe_shift_preds(final_lps, shift).flatten(0, 1),
            labels.flatten(),
            reduction="none",
        )
        batch_output["baseline_entropy"]["final"] = th.sum(
            -final_probs * final_lps, dim=-1
        )
        th.save(batch_output, output_dir / f"batch_{pbar.n}.pt")

        batches.append(pytree_map(th.mean, batch_output))  # type: ignore[arg-type]
        transfer_batches.append(
            {
                "transfer_ce": transfer_ces,
                "transfer_kl": transfer_kls,
            }
        )

    pbar.close()
    agg = pytree_map(lambda x: nats_to_bpb_ratio * x.mean(), pytree_stack(batches))
    agg = pytree_map(lambda x: maybe_all_reduce(x, "mean"), agg)
    if local_rank == 0:
        th.save(agg, root_dir / "aggregate_metrics.pt")

    if args.transfer:
        agg_transfer = pytree_map(
            lambda x: nats_to_bpb_ratio * x.mean(0), pytree_stack(transfer_batches)
        )
        agg_transfer = pytree_map(lambda x: maybe_all_reduce(x, "mean"), agg_transfer)
        if local_rank == 0:
            th.save(agg_transfer, root_dir / "aggregate_transfer_metrics.pt")

    first_token_stats.all_reduce_()
    delta_stats.all_reduce_()
    stream_stats.all_reduce_()
    for stats in ll_logit_statistics:
        stats.all_reduce_()
    for stats in tl_logit_statistics:
        stats.all_reduce_()

    if local_rank == 0:
        th.save(first_token_stats, root_dir / "first_token_stats.pt")
        th.save(delta_stats, root_dir / "delta_stats.pt")
        th.save(stream_stats, root_dir / "stream_stats.pt")

        th.save(final_logit_stats, root_dir / "final_logit_stats.pt")
        th.save(ll_logit_statistics, root_dir / "ll_logit_stats.pt")
        th.save(tl_logit_statistics, root_dir / "tl_logit_stats.pt")
