"""Evaluation loop for the tuned lens model."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Optional

import torch as th
from simple_parsing import field
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tuned_lens.nn.lenses import Lens, LogitLens, TunedLens
from tuned_lens.scripts.ingredients import (
    Data,
    Distributed,
    Model,
)
from tuned_lens.stats import LogitStats
from tuned_lens.utils import (
    maybe_all_reduce,
    pytree_map,
    pytree_stack,
    shift_labels,
    shift_preds,
)


@dataclass
class Eval:
    """Type hinting for CLI args."""

    data: Data

    model: Model

    dist: Distributed

    lens: Optional[str] = field(alias=["-l"], default=None)
    """Path to the tuned lens model."""

    seed: int = 42
    """Random seed used for data shuffling."""

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

    def load_lens(self, model) -> Lens:
        """Load the tuned lens model."""
        if self.lens is None:
            return LogitLens.from_model(model)
        else:
            return TunedLens.from_model_and_pretrained(model, self.lens)

    @th.autocast("cuda", enabled=th.cuda.is_available())
    @th.no_grad()
    def execute(self):
        """Trains a TunedLens model against a transformer on a dataset."""
        # Load model, tokenizer, data, and lens
        self.dist.init()
        model = tokenizer = data = lens = nats_to_bpb = model_name = None
        if self.dist.primary:
            # Let the primary processes populate the cache
            model, tokenizer = self.model.load()
            data, nats_to_bpb = self.data.load(tokenizer)
            lens = self.load_lens(model)

        self.dist.barrier()  # Wait for primary to finish filling the cache

        if not self.dist.primary:
            # Let the non-primary processes load from the cache
            model, tokenizer = self.model.load(must_use_cache=True)
            data, nats_to_bpb = self.data.load(tokenizer)
            lens = self.load_lens(model)

        assert model and tokenizer and data and lens and nats_to_bpb

        model = self.dist.shard_model(model)
        # Note since we are not training we can just move the lens to the device.
        # No need to use DDP
        lens = lens.to(self.dist.device)
        data = self.dist.shard_dataset(data)

        dl = DataLoader(
            data.shuffle(seed=self.seed),  # type: ignore[arg-type],
            batch_size=self.per_gpu_batch_size,
        )

        lens.eval()

        if self.limit:
            dl = islice(dl, self.limit)
            total = self.limit
        else:
            total = len(dl)

        *_, model_name = model.config.name_or_path.split("/")

        if self.lens:
            root_dir = Path(self.lens, "tuned-lens-eval")
        else:
            root_dir = Path(model_name, "logit_lens_eval")

        if self.output is not None:
            root_dir = self.output

        root_dir.mkdir(exist_ok=True, parents=True)

        L = model.config.num_hidden_layers
        batches = []
        transfer_batches = []

        final_logit_stats = LogitStats()
        lens_statistics = [LogitStats() for _ in range(L)]
        self.dist.barrier()
        print(f"All processes initialized. Running evaluation on {total} batches.")

        pbar = tqdm(dl, desc="Evaluating", position=self.dist.rank, total=total)
        for batch in pbar:
            batch = self.dist.send_to_device(batch)
            with th.no_grad():
                output = model(**batch, output_hidden_states=True)

            hidden_states = output.hidden_states[-1:]

            shift = self.token_shift if self.token_shift is not None else 1
            final_lps = output.logits.log_softmax(dim=-1)
            final_probs = final_lps.exp()
            labels = shift_labels(batch["input_ids"], shift)

            batch_output = defaultdict(dict)
            transfer_ces = th.zeros(L, L, device=final_lps.device)
            transfer_kls = th.zeros(L, L, device=final_lps.device)

            # Compute tuned lens eval and statistics if applicable
            for j, h in zip(range(L), hidden_states):
                name = f"layer_{j}"
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
                lens_statistics[j].update(lens_lps, assume_normalized=True)

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

            batch_output["baseline_ce"]["final"] = th.nn.functional.cross_entropy(
                shift_preds(final_lps, shift).flatten(0, 1),
                labels.flatten(),
                reduction="none",
            )
            batch_output["baseline_entropy"]["final"] = th.sum(
                -final_probs * final_lps, dim=-1
            )
            batches.append(pytree_map(th.mean, batch_output))  # type: ignore[arg-type]
            transfer_batches.append(
                {
                    "transfer_ce": transfer_ces,
                    "transfer_kl": transfer_kls,
                }
            )
            # Keep the processes synced
            self.dist.barrier()

        pbar.close()
        agg = pytree_map(lambda x: nats_to_bpb * x.mean(), pytree_stack(batches))
        agg = pytree_map(lambda x: maybe_all_reduce(x), agg)
        if self.dist.primary:
            th.save(agg, root_dir / "aggregate_metrics.pt")

        if self.transfer:
            agg_transfer = pytree_map(
                lambda x: nats_to_bpb * x.mean(0), pytree_stack(transfer_batches)
            )
            agg_transfer = pytree_map(lambda x: maybe_all_reduce(x), agg_transfer)
            if self.dist.primary:
                th.save(agg_transfer, root_dir / "aggregate_transfer_metrics.pt")

        for stats in lens_statistics:
            stats.all_reduce_()

        if self.dist.primary:
            th.save(lens_statistics, root_dir / "lens_logit_stats.pt")
