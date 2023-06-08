"""Evaluation loop for the tuned lens model."""
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal, Optional

import torch as th
from simple_parsing import field
from tqdm.auto import tqdm
from transformers import PreTrainedModel

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

LensType = Literal["logit", "tuned"]


def _nested_dict():
    return defaultdict(_nested_dict)


@dataclass
class Eval:
    """Type hinting for CLI args."""

    data: Data

    model: Model

    dist: Distributed

    output: Path = field(alias=["-o"])
    """Folder to save the eval results to."""

    lens_name: Optional[str] = field(alias=["-l"], default=None)
    """Path to the tuned lens model."""

    lens_types: list[LensType] = field(default_factory=lambda: ["logit"])
    """Types of lenses to evaluate can be a combination of (logit|tuned)."""

    seed: int = 42
    """Random seed used for data shuffling."""

    tokens: Optional[int] = None
    """Number of tokens to evaluate on. If None, will use the entire dataset."""

    token_shift: int = field(default=1)
    """How to shift the labels wrt the input tokens (1 = next token, 0 = current token,
    -1 = previous token, etc.)"""

    per_gpu_batch_size: int = 1
    """Number of samples to try to fit on a GPU at once."""

    layer_transfer: bool = field(action="store_true")
    """Evaluate the transfer of the lens to different layers of the transformer."""

    record_logit_stats: bool = field(action="store_true")
    """Record the statistics of the marginal token distribution at each layer."""

    def load_lens(self, model: PreTrainedModel) -> dict[str, Lens]:
        """Load the tuned lens model."""
        lenses = {}
        for lens_type in self.lens_types:
            if lens_type == "logit":
                lenses["logit"] = LogitLens.from_model(model)
            elif lens_type == "tuned":
                if self.lens_name is None:
                    raise ValueError(
                        "Must specify a lens name when evaluating a tuned lens."
                    )
                lenses["tuned"] = TunedLens.from_model_and_pretrained(
                    model, self.lens_name
                )
            else:
                raise ValueError(f"Unknown lens type: {lens_type}")
        return lenses

    def calculate_batch_limit(self, tokens_per_sample: int):
        """Calculate the total number of batches to evaluate on."""
        assert self.tokens is not None
        global_batch_size = self.dist.world_size * self.per_gpu_batch_size
        tokens_per_batch = global_batch_size * tokens_per_sample
        return self.tokens // tokens_per_batch

    def _initialize_logit_stats_recorders(
        self, lenses: dict[str, Lens], total_layers: int
    ):
        if self.record_logit_stats:
            self.logit_stats_recorders = {
                lens_type: {f"layer_{i}": LogitStats() for i in range(total_layers)}
                for lens_type in lenses.keys()
            }
            self.logit_stats_recorder_final = LogitStats()
        else:
            self.logit_stats_recorders = None
            self.logit_stats_recorder_final = None

    def _record_logit_stats(self, logp: th.Tensor, layer: int, lens_type: str):
        if self.logit_stats_recorders is not None:
            self.logit_stats_recorders[lens_type][f"layer_{layer}"].update(
                logp, assume_normalized=True
            )

    def _record_logit_stats_final(self, logp: th.Tensor):
        if self.logit_stats_recorder_final is not None:
            self.logit_stats_recorder_final.update(logp, assume_normalized=True)

    def _save_logit_stats(self) -> defaultdict:
        logit_stats = _nested_dict()
        if self.logit_stats_recorders is not None:
            for lens_type, recorders in self.logit_stats_recorders.items():
                for layer, recorder in recorders.items():
                    recorder.all_reduce_()
                    logit_stats[lens_type]["logit_stats"][layer] = (
                        recorder.marginal_probs.cpu().numpy().tolist()
                    )

        if self.logit_stats_recorder_final is not None:
            self.logit_stats_recorder_final.all_reduce_()
            logit_stats["baseline"]["logit_stats"]["final"] = (
                self.logit_stats_recorder_final.marginal_probs.cpu().numpy().tolist()
            )

        return logit_stats

    def _evaluate_lenses_on_hidden(
        self,
        lenses: dict[str, Lens],
        hidden: th.Tensor,
        layer: int,
        final_probs: th.Tensor,
        final_lps: th.Tensor,
        labels: th.Tensor,
        batch_output: defaultdict,
        total_layers: int,
    ):
        """Evaluate a lens at a given layer. Batch output is modified in place.

        Args:
            lenses: The dictionary of lenses to evaluate on this hidden state.
            hidden: (batch x seq x d_model) The hidden states of the transformer.
            layer: The layer this hidden state is from.
            final_probs: (batch x seq x vocab) The final probabilities of
                the transformer.
            final_lps: (batch x seq x vocab) The final log probabilities
                of the transformer.
            labels: (batch x seq) The labels for the transformer.
            batch_output: Where to store the logging results.
            total_layers: The total number of layers in the transformer.
            logp_stats: where to record the logging results.
        """
        for lens_type, lens in lenses.items():
            layer_name = f"layer_{layer}"
            lens_lps = lens(hidden, idx=layer).log_softmax(dim=-1)
            lens_probs = lens_lps.exp()

            self._record_logit_stats(lens_lps, layer, lens_type)

            batch_output[lens_type]["ce"][layer_name] = th.nn.functional.cross_entropy(
                shift_preds(lens_lps, self.token_shift).flatten(0, 1),
                labels.flatten(),
                reduction="none",
            )

            batch_output[lens_type]["entropy"][layer_name] = th.sum(
                -lens_probs * lens_lps, dim=-1
            )

            batch_output[lens_type]["kl"][layer_name] = th.sum(
                final_probs * (final_lps - lens_lps), dim=-1
            )

            if self.layer_transfer:
                for i in range(total_layers):
                    trans_name = f"layer_{i}"
                    transfer_lps = lens(hidden, idx=i).log_softmax(dim=-1)
                    batch_output[lens_type]["layer_transfer"]["ce"][trans_name][
                        layer_name
                    ] = th.nn.functional.cross_entropy(
                        shift_preds(transfer_lps, self.token_shift).flatten(0, 1),
                        labels.flatten(),
                    )
                    batch_output[lens_type]["layer_transfer"]["kl"][trans_name][
                        layer_name
                    ] = th.sum(lens_probs * (lens_lps - transfer_lps), dim=-1).mean()

    @th.autocast("cuda", enabled=th.cuda.is_available())
    @th.no_grad()
    def execute(self):
        """Evaluates a TunedLens model against a transformer on a dataset."""
        # Load model, tokenizer, data, and lens
        self.dist.init()
        model = tokenizer = data = lenses = nats_to_bpb = None

        # See comment in train_loop.py for why we do this
        load_device = self.dist.device if not self.dist.fsdp else None
        if self.dist.primary:
            # Let the primary processes populate the cache
            model, tokenizer = self.model.load(load_device)
            data, nats_to_bpb = self.data.load(tokenizer)
            lenses = self.load_lens(model)

        self.dist.barrier()  # Wait for primary to finish filling the cache

        if not self.dist.primary:
            # Let the non-primary processes load from the cache
            model, tokenizer = self.model.load(load_device, must_use_cache=True)
            data, nats_to_bpb = self.data.load(tokenizer)
            lenses = self.load_lens(model)

        assert model and tokenizer and data and lenses and nats_to_bpb

        model = self.dist.shard_model(model)
        # Note since we are not training we can just move the lens to the device.
        # No need to use DDP
        lenses = {name: lens.to(self.dist.device) for name, lens in lenses.items()}
        dl = self.dist.data_loader(data)
        dl.seed(self.seed)

        for lens in lenses.values():
            lens.eval()

        if self.tokens is not None:
            tokens_per_sample = len(data[0]["input_ids"])
            batch_limit = self.calculate_batch_limit(tokens_per_sample)
            assert batch_limit > 0, "Batch limit must be positive."
            dl = islice(dl, batch_limit)
            total = batch_limit
        else:
            total = len(data) // self.dist.world_size

        L = model.config.num_hidden_layers

        self._initialize_logit_stats_recorders(lenses, L)

        root_dir = self.output

        root_dir.mkdir(exist_ok=True, parents=True)

        batches = []

        self.dist.barrier()
        print(f"All processes initialized. Running evaluation on {total} batches.")

        pbar = tqdm(dl, desc="Evaluating", position=self.dist.rank, total=total)
        for batch in pbar:
            batch = self.dist.send_to_device(batch)
            output = model(**batch, output_hidden_states=True)

            hidden_states = output.hidden_states[:-1]

            final_lps = output.logits.log_softmax(dim=-1)

            final_probs = final_lps.exp()
            assert not th.isnan(output.logits).any(), "Logits are NaN"

            labels = shift_labels(batch["input_ids"], self.token_shift)

            batch_output = _nested_dict()

            # Compute tuned lens eval and statistics if applicable
            for j, h in zip(range(L), hidden_states):
                self._evaluate_lenses_on_hidden(
                    lenses=lenses,
                    hidden=h,
                    layer=j,
                    final_probs=final_probs,
                    final_lps=final_lps,
                    labels=labels,
                    batch_output=batch_output,
                    total_layers=L,
                )

            batch_output["baseline"]["ce"]["final"] = th.nn.functional.cross_entropy(
                shift_preds(final_lps, self.token_shift).flatten(0, 1),
                labels.flatten(),
                reduction="none",
            )
            batch_output["baseline"]["entropy"]["final"] = th.sum(
                -final_probs * final_lps, dim=-1
            )

            batches.append(pytree_map(th.mean, batch_output))  # type: ignore[arg-type]

            self._record_logit_stats_final(final_lps)

        pbar.close()
        agg = pytree_map(lambda x: nats_to_bpb * x.mean(), pytree_stack(batches))
        agg = pytree_map(lambda x: maybe_all_reduce(x), agg)
        agg = pytree_map(lambda x: x.cpu().numpy().item(), agg)

        assert isinstance(agg, dict)

        batches = pytree_map(lambda x: nats_to_bpb * x, batches)
        batches = pytree_map(lambda x: maybe_all_reduce(x), batches)
        batches = pytree_map(lambda x: x.cpu().item(), batches)
        assert isinstance(batches, list)

        logit_stats = self._save_logit_stats()

        if self.dist.primary:
            with (root_dir / "batches.jsonl").open("w") as f:
                json.dump(batches, f)

            with (root_dir / "aggregate_metrics.json").open("w") as f:
                json.dump(agg, f)

            if self.record_logit_stats:
                with (root_dir / "logit_stats.json").open("w") as f:
                    json.dump(logit_stats, f)
