"""Train or evaluate a tuned lens for a language model."""

from argparse import Namespace
from collections import defaultdict
from contextlib import nullcontext, redirect_stdout
from datasets import Dataset, DatasetDict, load_dataset
from functools import partial
from itertools import islice
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from white_box.scripts import get_lens_parser
from white_box import ResidualStats, ResidualStream, TunedLens
from white_box.data import (
    chunk_and_tokenize,
    compute_nats_to_bpb_ratio,
    silence_datasets_messages,
)
from white_box.model_surgery import get_transformer_layers
from white_box.utils import (
    maybe_all_cat,
    maybe_shift_labels,
    maybe_shift_preds,
    pytree_map,
    send_to_device,
)
from white_box.scripts import train_loop
import json
import os
import torch as th
import torch.distributed as dist


def main(args):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    silence_datasets_messages()

    print(f"Loading dataset '{' '.join(args.dataset)}'")
    if len(args.dataset) == 1 and args.dataset[0].endswith(".jsonl"):
        dataset = Dataset.from_json(args.dataset[0])
        assert isinstance(dataset, Dataset)
    else:
        dataset = load_dataset(*args.dataset, split=args.split)
        if not isinstance(dataset, (Dataset, DatasetDict)):
            raise ValueError("Only Dataset and DatasetDict instances are supported.")

    processed = chunk_and_tokenize(dataset, tokenizer, text_key=args.text_column)
    nats_to_bpb = compute_nats_to_bpb_ratio(dataset, processed)
    assert isinstance(processed, Dataset)
    if dist.is_initialized():
        processed = processed.shard(dist.get_world_size(), local_rank)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.revision,
        torch_dtype="auto",
    )
    model.eval()
    model.requires_grad_(False)
    assert isinstance(model, PreTrainedModel)

    th.cuda.set_device(local_rank)

    # Can be set either in eval or in training; in eval it's required
    if args.lens:
        lens = TunedLens.load(args.lens, map_location="cpu")
    else:
        lens = TunedLens(
            model,
            dropout=args.dropout,
            include_final=args.train_final_lens,
            mlp_hidden_sizes=args.mlp_hidden_sizes,
            orthogonal=args.orthogonal,
            rank=args.rank,
            shared_mlp_hidden_sizes=args.shared_mlp_hidden_sizes,
            sublayers=args.sublayers,
        ).to(
            dtype=th.float16 if args.lens_dtype == "float16" else th.float32,
        )

    lens = lens.to(device=th.device("cuda", local_rank))
    print(f"Using lens with config: {json.dumps(lens.config, indent=2)}")

    if args.fsdp:
        _, layers = get_transformer_layers(model)
        layer_cls = type(layers[0])
        print(f"Using '{layer_cls.__name__}' for transformer_auto_wrap_policy.")

        model = FSDP(
            model,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
            ),
            cpu_offload=CPUOffload(offload_params=True),
            device_id=local_rank,
            # This turns out to be important for training speed
            forward_prefetch=True,
            mixed_precision=MixedPrecision(
                param_dtype=th.float16,
                reduce_dtype=th.float16,
                buffer_dtype=th.float16,
            ),
        )
    else:
        model.to(local_rank)

    if args.command == "train":
        train_loop(args, model, processed, lens, float(nats_to_bpb))
    elif args.command == "eval":
        eval_loop(args, model, processed, lens, float(nats_to_bpb))
    else:
        raise ValueError(f"Unknown command: {args.command}")


@th.autocast("cuda")
@th.no_grad()
def eval_loop(
    args: Namespace,
    model: th.nn.Module,
    data: Dataset,
    lens: TunedLens,
    nats_to_bpb: float,
):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    dl = DataLoader(
        data.shuffle(seed=args.seed),  # type: ignore[arg-type],
        batch_size=args.per_gpu_batch_size,
    )
    lens.eval()

    # Running mean & covariance of the hidden states
    first_token_stats = ResidualStats()
    stream_stats = ResidualStats()

    # Keys are names of layers
    baseline_dict = defaultdict(list)
    ce_dict = defaultdict(list)
    kl_dict = defaultdict(list)
    final_losses = []

    if args.limit:
        dl = islice(dl, args.limit)
        total = args.limit
    else:
        total = len(dl)

    for batch in tqdm(dl, desc="Evaluating", position=local_rank, total=total):
        batch = send_to_device(batch, th.device(local_rank))
        output = model(**batch, labels=batch["input_ids"], output_hidden_states=True)

        final_log_probs = output.logits.log_softmax(dim=-1)
        final_probs = final_log_probs.exp()

        final_losses.append(output.loss)
        labels = maybe_shift_labels(batch["input_ids"], 1).flatten()

        stream = ResidualStream(
            embeddings=output.hidden_states[0], layers=output.hidden_states[1:-1]
        )

        # Do this sequentially to save VRAM
        for i, (name, h) in enumerate(stream.items()):
            logits = lens(h, idx=i)
            log_probs = logits.log_softmax(dim=-1)

            baseline_dict[name].append(
                th.nn.functional.cross_entropy(
                    maybe_shift_preds(lens.to_logits(h), 1).flatten(0, 1),
                    labels,
                    reduction="none",
                )
            )
            ce_dict[name].append(
                th.nn.functional.cross_entropy(
                    maybe_shift_preds(logits, 1).flatten(0, 1), labels, reduction="none"
                )
            )
            kl_dict[name].append(
                th.sum(final_probs * (final_log_probs - log_probs), dim=-1).flatten()
            )

        # Don't let the processes get too out of sync
        dist.barrier()
        if args.residual_stats:
            first_tokens = stream.map(lambda x: x[:, 0])
            rest = stream.map(lambda x: x[:, 1:])

            first_token_stats.update(first_tokens)
            stream_stats.update(rest)

    baselines = {
        k: maybe_all_cat(th.cat(v)) * nats_to_bpb for k, v in baseline_dict.items()
    }
    ces = {k: maybe_all_cat(th.cat(v)) * nats_to_bpb for k, v in ce_dict.items()}
    kls = {k: maybe_all_cat(th.cat(v)) * nats_to_bpb for k, v in kl_dict.items()}
    final_loss = maybe_all_cat(th.stack(final_losses)) * nats_to_bpb

    output_dir = args.output or args.lens
    if local_rank == 0:
        results = {
            "baseline": baselines,
            "ce": ces,
            "kl": kls,
            "final_loss": final_loss,
        }
        histograms = pytree_map(lambda x: x.sort().values.cpu(), results)

        hist_path = output_dir / "eval_histograms.pt"
        print(f"Saving histograms to '{hist_path}'")
        th.save(histograms, hist_path)

        json_path = output_dir / "eval.json"
        print(f"Saving aggregate results to '{json_path}'")

        with open(json_path, "w") as f:
            aggregated = pytree_map(lambda x: x.mean().item(), results)
            json.dump(aggregated, f, indent=2)

    if args.residual_stats:
        stream_stats.all_reduce_()
        if local_rank == 0:
            th.save(first_token_stats, output_dir / "first_token_stats.pt")
            th.save(stream_stats, output_dir / "stream_stats.pt")


if __name__ == "__main__":
    parser = get_lens_parser()

    # Support both distributed and non-distributed training
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        dist.init_process_group("nccl")

    # Only print on rank 0
    with nullcontext() if not local_rank else redirect_stdout(None):
        main(parser.parse_args())
