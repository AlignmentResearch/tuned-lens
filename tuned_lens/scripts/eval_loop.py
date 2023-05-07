"""Evaluation loop for the tuned lens model."""
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Optional

import numpy as np
import torch as th
from flatten_dict import flatten
from simple_parsing import field
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel

from tuned_lens.nn.lenses import Lens, LogitLens, TunedLens
from tuned_lens.scripts.ingredients import (
    Data,
    Distributed,
    Model,
)
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

    lens_name: str = field(alias=["-l"])
    """Path to the tuned lens model."""

    output: Path = field(alias=["-o"])
    """Folder to save the eval results to."""

    lens_types: list[str] = field(default_factory=lambda: ["tuned", "logit"])
    """Types of lenses to evaluate."""

    seed: int = 42
    """Random seed used for data shuffling."""

    limit: Optional[int] = None
    """Number of batches to evaluate on. If None, will use the entire dataset."""

    token_shift: Optional[int] = None
    """How to shift the labels wrt the input tokens (1 = next token, 0 = current token,
    -1 = previous token, etc.)"""

    per_gpu_batch_size: int = 1
    """Number of samples to try to fit on a GPU at once."""

    def load_lens(self, model: PreTrainedModel) -> dict[str, Lens]:
        """Load the tuned lens model."""
        lenses = {}
        for lens_type in self.lens_types:
            if lens_type == "logit":
                lenses["logit"] = LogitLens.from_model(model)
            elif lens_type == "tuned":
                lenses[
                    f"tuned[{model.config.name_or_path}]"
                ] = TunedLens.from_model_and_pretrained(model, self.lens_name)
            elif match := re.match(r"tuned\[([a-zA-Z0-9/\.\-]+)\]", lens_type):
                model_name = match.group(1)
                new_model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                    model_name,
                    low_cpu_mem_usage=True,
                    torch_dtype="auto",
                )
                lenses[f"tuned[{model_name}]"] = TunedLens.from_model_and_pretrained(
                    new_model, self.lens_name
                )
            else:
                raise ValueError(f"Unknown lens type: {lens_type}")
        return lenses

    @th.autocast("cuda", enabled=th.cuda.is_available())
    @th.no_grad()
    def execute(self):
        """Trains a TunedLens model against a transformer on a dataset."""
        # Load model, tokenizer, data, and lens
        self.dist.init()
        model = tokenizer = data = lenses = nats_to_bpb = None
        if self.dist.primary:
            # Let the primary processes populate the cache
            model, tokenizer = self.model.load()
            data, nats_to_bpb = self.data.load(tokenizer)
            lenses = self.load_lens(model)

        self.dist.barrier()  # Wait for primary to finish filling the cache

        if not self.dist.primary:
            # Let the non-primary processes load from the cache
            model, tokenizer = self.model.load(must_use_cache=True)
            data, nats_to_bpb = self.data.load(tokenizer)
            lenses = self.load_lens(model)

        assert model and tokenizer and data and lenses and nats_to_bpb

        model = self.dist.shard_model(model)
        # Note since we are not training we can just move the lens to the device.
        # No need to use DDP
        lenses = {name: lens.to(self.dist.device) for name, lens in lenses.items()}
        data = self.dist.shard_dataset(data)

        dl = DataLoader(
            data.shuffle(seed=self.seed),  # type: ignore[arg-type],
            batch_size=self.per_gpu_batch_size,
        )

        for lens in lenses.values():
            lens.eval()

        if self.limit:
            dl = islice(dl, self.limit)
            total = self.limit
        else:
            total = len(dl)

        root_dir = self.output

        root_dir.mkdir(exist_ok=True, parents=True)

        L = model.config.num_hidden_layers
        batches = []

        self.dist.barrier()
        print(f"All processes initialized. Running evaluation on {total} batches.")

        pbar = tqdm(dl, desc="Evaluating", position=self.dist.rank, total=total)
        for batch in pbar:
            batch = self.dist.send_to_device(batch)
            with th.no_grad():
                output = model(**batch, output_hidden_states=True)

            hidden_states = output.hidden_states[:-1]

            shift = self.token_shift if self.token_shift is not None else 1
            final_lps = output.logits.log_softmax(dim=-1)
            final_probs = final_lps.exp()
            assert not th.isnan(output.logits).any(), "Logits are NaN"

            labels = shift_labels(batch["input_ids"], shift)

            def nested_dict():
                return defaultdict(nested_dict)

            batch_output = nested_dict()

            # Compute tuned lens eval and statistics if applicable
            for j, h in zip(range(L), hidden_states):
                name = f"layer_{j}"
                for lens_type, lens in lenses.items():
                    lens_lps = lens(h, idx=j).log_softmax(dim=-1)
                    lens_probs = lens_lps.exp()

                    batch_output[lens_type]["ce"][
                        name
                    ] = th.nn.functional.cross_entropy(
                        shift_preds(lens_lps, shift).flatten(0, 1),
                        labels.flatten(),
                        reduction="none",
                    )

                    batch_output[lens_type]["entropy"][name] = th.sum(
                        -lens_probs * lens_lps, dim=-1
                    )
                    batch_output[lens_type]["kl"][name] = th.sum(
                        final_probs * (final_lps - lens_lps), dim=-1
                    )

            batch_output["baseline_ce"]["final"] = th.nn.functional.cross_entropy(
                shift_preds(final_lps, shift).flatten(0, 1),
                labels.flatten(),
                reduction="none",
            )
            batch_output["baseline_entropy"]["final"] = th.sum(
                -final_probs * final_lps, dim=-1
            )
            batches.append(pytree_map(th.mean, batch_output))  # type: ignore[arg-type]

            # Keep the processes synced
            self.dist.barrier()

        pbar.close()
        agg = pytree_map(lambda x: nats_to_bpb * x.mean(), pytree_stack(batches))
        agg = pytree_map(lambda x: maybe_all_reduce(x), agg)
        agg = pytree_map(lambda x: x.cpu().numpy(), agg)
        assert isinstance(agg, dict)

        batches = pytree_map(lambda x: nats_to_bpb * x, pytree_stack(batches))
        batches = pytree_map(lambda x: maybe_all_reduce(x), batches)
        batches = pytree_map(lambda x: x.cpu().numpy(), batches)
        assert isinstance(batches, dict)

        if self.dist.primary:
            np.savez_compressed(
                root_dir / "batches.npz", **flatten(batches, reducer="dot")
            )
            np.savez(root_dir / "aggregate_metrics.npz", **flatten(agg, reducer="dot"))
