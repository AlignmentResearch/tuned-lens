"""Evaluation loop for the tuned lens model."""
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Optional

import torch as th
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

    output: Path = field(alias=["-o"])
    """Folder to save the eval results to."""

    lens_name: Optional[str] = field(alias=["-l"], default=None)
    """Path to the tuned lens model."""

    lens_types: list[str] = field(default_factory=lambda: ["logit"])
    """Types of lenses to evaluate."""

    seed: int = 42
    """Random seed used for data shuffling."""

    tokens: Optional[int] = None
    """Number of tokens to evaluate on. If None, will use the entire dataset."""

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
                if self.lens_name is None:
                    raise ValueError(
                        "Must specify a lens name when evaluating a tuned lens."
                    )
                lenses[
                    f"tuned[{model.config.name_or_path}]"
                ] = TunedLens.from_model_and_pretrained(model, self.lens_name)
            elif match := re.match(r"tuned\[([a-zA-Z0-9/\.\-]+)\]", lens_type):
                if self.lens_name is None:
                    raise ValueError(
                        "Must specify a lens name when evaluating a tuned lens."
                    )
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

    def calculate_batch_limit(self, tokens_per_sample: int):
        """Calculate the total number of batches to evaluate on."""
        assert self.tokens is not None
        global_batch_size = self.dist.world_size * self.per_gpu_batch_size
        tokens_per_batch = global_batch_size * tokens_per_sample
        return self.tokens // tokens_per_batch

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

        if self.tokens is not None:
            tokens_per_sample = len(data[0]["input_ids"])
            batch_limit = self.calculate_batch_limit(tokens_per_sample)
            assert batch_limit > 0, "Batch limit must be positive."
            assert batch_limit <= len(
                dl
            ), "Not enough data to evaluate on that many tokens."
            dl = islice(dl, batch_limit)
            total = batch_limit
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

                    # Handle the case where the model has more/less tokens than the lens
                    if final_lps.shape[-1] != lens_lps.shape[-1]:
                        logging.warning(
                            "Lens has different number of tokens than model."
                        )

                    common_vocab = min(final_lps.shape[-1], lens_lps.shape[-1])
                    trunc_final_lps = final_lps[..., :common_vocab]
                    trunc_lens_lps = lens_lps[..., :common_vocab]
                    trunc_final_probs = final_probs[..., :common_vocab]
                    trunc_lens_probs = lens_probs[..., :common_vocab]

                    batch_output[lens_type]["ce"][
                        name
                    ] = th.nn.functional.cross_entropy(
                        shift_preds(trunc_lens_lps, shift).flatten(0, 1),
                        labels.flatten(),
                        reduction="none",
                    )

                    batch_output[lens_type]["entropy"][name] = th.sum(
                        -trunc_lens_probs * trunc_lens_lps, dim=-1
                    )
                    batch_output[lens_type]["kl"][name] = th.sum(
                        trunc_final_probs * (trunc_final_lps - trunc_lens_lps), dim=-1
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
        agg = pytree_map(lambda x: x.cpu().numpy().item(), agg)
        assert isinstance(agg, dict)

        batches = pytree_map(lambda x: nats_to_bpb * x, batches)
        batches = pytree_map(lambda x: maybe_all_reduce(x), batches)
        batches = pytree_map(lambda x: x.cpu().item(), batches)
        assert isinstance(batches, list)

        if self.dist.primary:
            with (root_dir / "batches.jsonl").open("w") as f:
                json.dump(batches, f)

            with (root_dir / "aggregate_metrics.json").open("w") as f:
                json.dump(agg, f)
