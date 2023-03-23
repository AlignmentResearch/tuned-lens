"""Evaluation loop for the tuned lens model."""
from argparse import Namespace
from datasets import Dataset
from collections import defaultdict
from itertools import islice
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel
from typing import Optional
from tuned_lens.residual_stream import record_residual_stream
from tuned_lens.stats import ResidualStats, LogitStats
from tuned_lens.nn import Decoder, TunedLens
from tuned_lens.utils import (
    maybe_all_cat,
    maybe_all_reduce,
    shift_labels,
    shift_preds,
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
    model: PreTrainedModel,
    data: Dataset,
    lens: Optional[TunedLens],
    nats_to_bpb_ratio: float,
):
    """Trains a TunedLens model against a transformer on a dataset.

    Args:
        args: The command-line arguments see __main__.py train subcommand.
        model: The transformer model to train.
        data: The dataset to train on.
        lens: The TunedLens model to train.
        nats_to_bpb_ratio: The ratio of nats to bits per byte for the dataset.
    """
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    dl = DataLoader(
        data.shuffle(seed=args.seed),  # type: ignore[arg-type],
        batch_size=args.per_gpu_batch_size,
    )
    if lens:
        lens.eval()

    # Running mean & covariance of the hidden states & residuals
    delta_stats = ResidualStats(cov=False)
    stream_stats = ResidualStats(dtype=th.float32)

    if args.limit:
        dl = islice(dl, args.limit)
        total = args.limit
    else:
        total = len(dl)

    root_dir = args.output or args.lens / "eval"
    output_dir = root_dir / f"rank_{local_rank}"
    output_dir.mkdir(exist_ok=True, parents=True)

    _to_logits = lens.to_logits if lens else Decoder(model)
    L = model.config.num_hidden_layers
    batches = []
    transfer_batches = []

    grad_alignments = [[] for _ in range(L)]

    final_logit_stats = LogitStats()
    ll_statistics = [LogitStats() for _ in range(L)]
    tl_statistics = [LogitStats() for _ in range(L)]

    pbar = tqdm(dl, desc="Evaluating", position=local_rank, total=total)
    for batch in pbar:
        batch = send_to_device(batch, th.device(local_rank))
        with record_residual_stream(model) as stream:
            output = model(**batch)

        shift = args.token_shift if args.token_shift is not None else 1
        final_lps = output.logits.log_softmax(dim=-1)
        final_probs = final_lps.exp()
        labels = shift_labels(batch["input_ids"], shift)

        batch_output = defaultdict(dict)
        transfer_ces = th.zeros(L, L, device=final_lps.device)
        transfer_kls = th.zeros(L, L, device=final_lps.device)

        # Compute logit lens eval and statistics
        for (j, d), (name, h) in zip(enumerate(stream.residuals()), stream.items()):
            if args.grad_alignment:
                h.requires_grad_(True)
                h.retain_grad()

            with th.set_grad_enabled(args.grad_alignment):
                baseline_lps = _to_logits(h).log_softmax(dim=-1)

                # Note that we don't reduce the loss here, since we want to look at
                # the full distribution of losses across tokens and samples
                losses = th.nn.functional.cross_entropy(
                    shift_preds(baseline_lps, shift).flatten(0, 1),
                    labels.flatten(),
                    reduction="none",
                )
                avg_loss = losses.mean()

            if args.grad_alignment:
                avg_loss.backward()

                assert h.grad is not None
                grad_alignments[j].append(
                    th.nn.functional.cosine_similarity(
                        h.grad.flatten(1), d.flatten(1), dim=-1
                    )
                )
                h.grad = None

            ll_statistics[j].update(baseline_lps, assume_normalized=True)

            batch_output["baseline_ce"][name] = losses
            batch_output["baseline_entropy"][name] = th.sum(
                -baseline_lps.exp() * baseline_lps, dim=-1
            )
            batch_output["baseline_kl"][name] = th.sum(
                final_probs * (final_lps - baseline_lps), dim=-1
            )

        # Compute tuned lens eval and statistics if applicable
        if lens:
            for j, (name, h) in zip(range(L), stream.items()):
                lens_lps = lens(h, idx=j).log_softmax(dim=-1)
                lens_probs = lens_lps.exp()

                batch_output["lens_ce"][name] = th.nn.functional.cross_entropy(
                    shift_preds(lens_lps, shift).flatten(0, 1),
                    labels.flatten(),
                    reduction="none",
                )
                batch_output["lens_entropy"][name] = th.sum(
                    -lens_probs * lens_lps, dim=-1
                )
                batch_output["lens_kl"][name] = th.sum(
                    final_probs * (final_lps - lens_lps), dim=-1
                )
                tl_statistics[j].update(lens_lps, assume_normalized=True)

                if args.transfer:
                    # Each iteration of the loop processes a different *probe* layer i
                    # for the test layer j.
                    for i in range(L):
                        transfer_lps = lens(h, idx=i).log_softmax(dim=-1)
                        transfer_ces[i, j] = th.nn.functional.cross_entropy(
                            shift_preds(transfer_lps, shift).flatten(0, 1),
                            labels.flatten(),
                        )
                        transfer_kls[i, j] = th.sum(
                            lens_probs * (lens_lps - transfer_lps), dim=-1
                        ).mean()

        final_logit_stats.update(final_lps, assume_normalized=True)
        if args.residual_stats:
            # Drop the first token because it's weird
            rest = stream.map(lambda x: x[:, 1:])

            delta_stats.update(rest.residuals())
            stream_stats.update(rest)

        batch_output["baseline_ce"]["final"] = th.nn.functional.cross_entropy(
            shift_preds(final_lps, shift).flatten(0, 1),
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
    agg = pytree_map(lambda x: maybe_all_reduce(x), agg)
    if local_rank == 0:
        th.save(agg, root_dir / "aggregate_metrics.pt")

    if args.transfer:
        agg_transfer = pytree_map(
            lambda x: nats_to_bpb_ratio * x.mean(0), pytree_stack(transfer_batches)
        )
        agg_transfer = pytree_map(lambda x: maybe_all_reduce(x), agg_transfer)
        if local_rank == 0:
            th.save(agg_transfer, root_dir / "aggregate_transfer_metrics.pt")

    # first_token_stats.all_reduce_()
    delta_stats.all_reduce_()
    stream_stats.all_reduce_()
    for stats in ll_statistics:
        stats.all_reduce_()

    if lens:
        for stats in tl_statistics:
            stats.all_reduce_()

    if args.grad_alignment:
        grad_alignments = [maybe_all_cat(th.cat(x, dim=0)) for x in grad_alignments]
        if local_rank == 0:
            th.save(grad_alignments, root_dir / "grad_alignments.pt")

    if local_rank == 0:
        th.save(delta_stats, root_dir / "delta_stats.pt")
        th.save(stream_stats, root_dir / "stream_stats.pt")

        th.save(final_logit_stats, root_dir / "final_logit_stats.pt")
        th.save(ll_statistics, root_dir / "ll_logit_stats.pt")
        if lens:
            th.save(tl_statistics, root_dir / "tl_logit_stats.pt")
