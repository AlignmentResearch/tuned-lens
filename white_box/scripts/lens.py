"""Train or evaluate a tuned lens for a language model."""

from contextlib import nullcontext, redirect_stdout
from datasets import Dataset, DatasetDict, load_dataset
from functools import partial
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from white_box.scripts import get_lens_parser
from white_box import TunedLens
from white_box.data import (
    chunk_and_tokenize,
    compute_nats_to_bpb_ratio,
    silence_datasets_messages,
)
from white_box.model_surgery import get_transformer_layers
from white_box.scripts import eval_loop, train_loop
import json
import os
import torch as th
import torch.distributed as dist


def main(args):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    print("Loading model...")

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
            mlp_hidden_sizes=args.mlp_hidden_sizes,
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

    # Load tokenizer & data
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
    print(f"Using nats per token to bits per byte ratio: {nats_to_bpb}")

    assert isinstance(processed, Dataset)
    if dist.is_initialized():
        processed = processed.shard(dist.get_world_size(), local_rank)

    if args.command == "train":
        train_loop(args, model, processed, lens, float(nats_to_bpb))
    elif args.command == "eval":
        eval_loop(args, model, processed, lens)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    parser = get_lens_parser()

    # Support both distributed and non-distributed training
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        dist.init_process_group("nccl")
        local_rank = int(local_rank)

    # Only print on rank 0
    with nullcontext() if not local_rank else redirect_stdout(None):
        main(parser.parse_args())
