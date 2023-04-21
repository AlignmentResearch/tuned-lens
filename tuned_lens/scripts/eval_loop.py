"""Evaluation loop for the tuned lens model."""
import os
from pathlib import Path
from collections import defaultdict
from itertools import islice

from simple_parsing import field
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional
from tuned_lens.residual_stream import record_residual_stream
from tuned_lens.stats import ResidualStats, LogitStats
from tuned_lens.scripts.ingredients import (
    Model,
    Data,
    Distributed,
)

from tuned_lens.nn.lenses import TunedLens
from tuned_lens.utils import (
    shift_labels,
    shift_preds,
    pytree_map,
    pytree_stack,
)
import torch as th
from dataclasses import dataclass


@dataclass
class Eval:
    """Type hinting for CLI args."""

    lens: str

    data: Data

    model: Model

    dist: Distributed

    grad_alignment: Optional[bool] = field(action="store_true")
    """Evaluate gradient alignment."""

    limit: Optional[int] = None
    """Number of batches to evaluate on. If None, will use the entire dataset."""

    output: Optional[Path] = field(alias=["-o"], default=None)
    """JSON file to save the eval results to."""

    transfer: Optional[bool] = field(action="store_true")
    """Evaluate how well probes transfer to other layers."""

    token_shift: Optional[int] = None
    """How to shift the labels wrt the input tokens (1 = next token, 0 = current token,
    -1 = previous token, etc.)"""

    per_gpu_batch_size: int = 1
    """Number of samples to try to fit on a GPU at once."""

    residual_stats: bool = field(action="store_true")

    def load_lens(self) -> TunedLens:
        """Load the tuned lens model."""
        return TunedLens.load(self.lens)

    @th.autocast("cuda", enabled=th.cuda.is_available())
    @th.no_grad()
    def execute(self):
        """Trains a TunedLens model against a transformer on a dataset."""
        model, tokenizer = self.model.load()
        data, nats_to_bpb_ratio = self.data.load(tokenizer)
        lens = self.load_lens()

        model = self.dist.shard_model(model)
        data = self.dist.shard_dataset(data)
        lens = self.dist.shard_lens(lens)

        dl = DataLoader(
            data.shuffle(seed=self.seed),  # type: ignore[arg-type],
            batch_size=self.per_gpu_batch_size,
        )

        if lens:
            lens.eval()

        # Running mean & covariance of the hidden states & residuals
        delta_stats = ResidualStats(cov=False)
        stream_stats = ResidualStats(dtype=th.float32)

        if self.limit:
            dl = islice(dl, self.limit)
            total = self.limit
        else:
            total = len(dl)

        root_dir = self.output or os.path.join(self.lens / "eval")
        output_dir = root_dir / f"rank_{self.dist.local_rank}"
        output_dir.mkdir(exist_ok=True, parents=True)

        L = model.config.num_hidden_layers
        batches = []
        transfer_batches = []

        grad_alignments = [[] for _ in range(L)]

        final_logit_stats = LogitStats()
        ll_statistics = [LogitStats() for _ in range(L)]
        tl_statistics = [LogitStats() for _ in range(L)]

        pbar = tqdm(dl, desc="Evaluating", position=self.dist.local_rank, total=total)
        for batch in pbar:
            batch = self.dist.send_to_device(batch)
            with record_residual_stream(model) as stream:
                output = model(**batch)

            shift = self.token_shift if self.token_shift is not None else 1
            final_lps = output.logits.log_softmax(dim=-1)
            final_probs = final_lps.exp()
            labels = shift_labels(batch["input_ids"], shift)

            batch_output = defaultdict(dict)
            transfer_ces = th.zeros(L, L, device=final_lps.device)
            transfer_kls = th.zeros(L, L, device=final_lps.device)

            # Compute logit lens eval and statistics
            for (j, d), (name, h) in zip(enumerate(stream.residuals()), stream.items()):
                if self.grad_alignment:
                    h.requires_grad_(True)
                    h.retain_grad()

                with th.set_grad_enabled(self.grad_alignment):
                    baseline_lps = lens.to_logits(h).log_softmax(dim=-1)

                    # Note that we don't reduce the loss here, since we want to look at
                    # the full distribution of losses across tokens and samples
                    losses = th.nn.functional.cross_entropy(
                        shift_preds(baseline_lps, shift).flatten(0, 1),
                        labels.flatten(),
                        reduction="none",
                    )
                    avg_loss = losses.mean()

                if self.grad_alignment:
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

                    if self.transfer:
                        # Each iteration of the loop processes a different *probe*
                        # layer i for the test layer j.
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
            if self.residual_stats:
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
        agg = pytree_map(lambda x: self.dist.maybe_all_reduce(x), agg)
        if self.dist.primary:
            th.save(agg, root_dir / "aggregate_metrics.pt")

        if self.transfer:
            agg_transfer = pytree_map(
                lambda x: nats_to_bpb_ratio * x.mean(0), pytree_stack(transfer_batches)
            )
            agg_transfer = pytree_map(
                lambda x: self.dist.maybe_all_reduce(x), agg_transfer
            )
            if self.dist.primary:
                th.save(agg_transfer, root_dir / "aggregate_transfer_metrics.pt")

        # first_token_stats.all_reduce_()
        delta_stats.all_reduce_()
        stream_stats.all_reduce_()
        for stats in ll_statistics:
            stats.all_reduce_()

        if lens:
            for stats in tl_statistics:
                stats.all_reduce_()

        if self.grad_alignment:
            grad_alignments = [
                self.dist.maybe_all_cat(th.cat(x, dim=0)) for x in grad_alignments
            ]
            if self.dist.primary:
                th.save(grad_alignments, root_dir / "grad_alignments.pt")

        if self.dist.primary:
            th.save(delta_stats, root_dir / "delta_stats.pt")
            th.save(stream_stats, root_dir / "stream_stats.pt")

            th.save(final_logit_stats, root_dir / "final_logit_stats.pt")
            th.save(ll_statistics, root_dir / "ll_logit_stats.pt")
            if lens:
                th.save(tl_statistics, root_dir / "tl_logit_stats.pt")
