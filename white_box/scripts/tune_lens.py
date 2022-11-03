"""Train a set of tuned lenses for a language model."""

from argparse import ArgumentParser
from collections import defaultdict
from datasets import Dataset, DatasetDict, load_dataset
from functools import partial
from itertools import islice
from pathlib import Path
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)
from white_box import ResidualStats, ResidualStream, TunedLens
from white_box.data import chunk_and_tokenize, silence_datasets_messages
from white_box.model_surgery import get_transformer_layers
from white_box.utils import send_to_device
import torch as th
import torch.distributed as dist


def main():
    parser = ArgumentParser(
        description="Train a set of tuned lenses for a language model."
    )
    parser.add_argument(
        "model_name", type=str, help="Name of model to use in the Huggingface Hub."
    )
    parser.add_argument(
        "--per-gpu-batch-size",
        type=int,
        default=12,
        help="Number of samples to try to fit on a GPU at once.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of dataset to use. Can either be a local .jsonl file or a name "
        "suitable to be passed to the HuggingFace load_dataset function.",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Use Fully Sharded Data Parallelism to train the model.",
    )
    parser.add_argument(
        "--grad-acc-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate.")
    parser.add_argument(
        "--loss",
        type=str,
        default="kl",
        choices=("ce", "kl"),
        help="Loss function to use for training.",
    )
    parser.add_argument(
        "--num-steps", type=int, default=100, help="Number of training steps."
    )
    parser.add_argument(
        "--orthogonal",
        action="store_true",
        help="Parametrize the tuned lenses as rotation matrices.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="File to save the lenses to. Defaults to the model name.",
    )
    parser.add_argument("--rank", type=int, help="Rank of the tuned lenses.")
    parser.add_argument(
        "--residual-stats",
        action="store_true",
        help="Save residual means and variances alongside the tuned lens.",
    )
    parser.add_argument("--resume", type=Path, help="File to resume training from.")
    parser.add_argument(
        "--sublayers",
        action="store_true",
        help="Train tuned lenses for attention blocks.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for training."
    )
    parser.add_argument(
        "--split", type=str, default="validation", help="Split of the dataset to use."
    )
    parser.add_argument(
        "--text-column", type=str, default="text", help="Column of the dataset to use."
    )
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        default=2**18,
        help="Number of tokens per step.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name of pretrained tokenizer to use from the Huggingface Hub. If None, "
        'will use AutoTokenizer.from_pretrained("<model name>").',
    )
    parser.add_argument(
        "--token-shift",
        type=int,
        default=None,
        help="How to shift the labels wrt the input tokens (1 = next token, "
        "0 = current token, -1 = previous token, etc.)",
    )
    parser.add_argument(
        "--train-final-lens",
        action="store_true",
        help="Train a lens for the final layer even though it's superfluous.",
    )
    parser.add_argument("--wandb", type=str, help="Name of run in Weights & Biases.")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    local_rank = dist.get_rank()

    if local_rank == 0:
        print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    silence_datasets_messages()

    if args.dataset.endswith(".jsonl"):
        dataset = Dataset.from_json(args.dataset)
        assert isinstance(dataset, Dataset)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
        if not isinstance(dataset, (Dataset, DatasetDict)):
            raise ValueError("Only Dataset and DatasetDict instances are supported.")

    processed = chunk_and_tokenize(dataset, tokenizer, text_key=args.text_column)
    assert isinstance(processed, Dataset)
    processed = processed.shard(dist.get_world_size(), local_rank)
    processed = processed.shuffle(seed=args.seed)

    # chunk_and_tokenize ensures the samples are all the same length
    tokens_per_sample = len(processed[0]["input_ids"])
    samples_per_step, rem = divmod(args.tokens_per_step, tokens_per_sample)
    if rem:
        raise ValueError(
            f"Number of tokens per step ({args.tokens_per_step:_}) must be divisible "
            f"by the number of tokens per sample ({tokens_per_sample})."
        )

    th.cuda.set_device(local_rank)
    if local_rank == 0:
        print(f"Using {args.tokens_per_step:_} tokens per training step.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
    )
    model.eval()
    model.requires_grad_(False)
    assert isinstance(model, PreTrainedModel)

    lens = (
        TunedLens(
            model,
            include_final=args.train_final_lens,
            orthogonal=args.orthogonal,
            rank=args.rank,
            sublayers=args.sublayers,
        )
        .cuda(local_rank)
        .float()
    )

    if args.wandb and local_rank == 0:
        import wandb

        wandb.init(
            name=args.wandb, project=args.model_name.split("/")[-1], config=vars(args)
        )
        wandb.watch(lens)

    # Running mean & variance of the residuals
    residual_stats = ResidualStats()
    stream_stats = ResidualStats()
    dl = DataLoader(processed, batch_size=args.per_gpu_batch_size)  # type: ignore

    if args.fsdp:
        _, layers = get_transformer_layers(model)
        layer_cls = type(layers[0])
        if local_rank == 0:
            print(f"Using '{layer_cls.__name__}' for transformer_auto_wrap_policy.")

        model = FSDP(
            model,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
            ),
            device_id=local_rank,
            mixed_precision=MixedPrecision(
                param_dtype=th.float16,
                reduce_dtype=th.float16,
                buffer_dtype=th.float16,
            ),
        )
    else:
        model = DDP(model, device_ids=[local_rank])

    ddp_lens = DDP(lens, device_ids=[local_rank], find_unused_parameters=True)

    # Skip the unembedding matrix and final layer norm
    params = [p for p in ddp_lens.parameters() if p.requires_grad]
    opt = th.optim.Adam(params, lr=args.lr)

    scheduler = get_linear_schedule_with_warmup(opt, 10, args.num_steps)
    if args.resume:
        # Get the most recently saved lens from the directory
        if args.resume.is_dir():
            ckpt = max(args.resume.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        else:
            ckpt = args.resume

        print(f"Loading checkpoint from {ckpt}")
        opt_path = ckpt / "optimizer.pt"
        ddp_lens.load_state_dict(th.load(ckpt))

        if opt_path.exists():
            print(f"Loading optimizer state from {opt_path}")
            opt.load_state_dict(th.load(opt_path))
        else:
            print("No optimizer state found. Starting Adam from scratch.")

    grad_acc_steps = samples_per_step // (
        args.per_gpu_batch_size * dist.get_world_size()
    )
    metrics = defaultdict(list)
    total_batches = args.num_steps * grad_acc_steps
    if local_rank == 0:
        print(f"Gradient accumulation steps: {grad_acc_steps}")

    pbar = tqdm(islice(dl, total_batches), desc="Training", total=total_batches)
    for step, batch in enumerate(pbar, start=1):
        assert isinstance(batch, dict)
        batch = send_to_device(batch, th.device(local_rank))
        output = model(**batch, output_hidden_states=True)

        final_logits = output.logits
        stream = ResidualStream(
            embeddings=output.hidden_states[0], layers=output.hidden_states[1:-1]
        )

        if args.residual_stats:
            sample_residuals = stream.residuals()
            residual_stats.update(sample_residuals)
            stream_stats.update(stream)

        shift = args.token_shift
        if args.loss == "ce":
            labels = batch["input_ids"]

            # Predict the *next* token by default w/ cross entropy
            if shift is None:
                shift = 1
        elif args.loss == "kl":
            labels = final_logits.log_softmax(dim=-1)

            # Match the *current* token distribution by default
            if shift is None:
                shift = 0
        else:
            raise NotImplementedError(f"Unknown loss {args.loss}")

        if shift > 0:
            labels = labels[:, shift:]
        elif shift < 0:
            labels = labels[:, :shift]

        # We do this sequentially to save VRAM
        for i, (name, h) in enumerate(stream.items()):
            # bfloat16 has larger dynamic range than float16 and seems to be better for
            # computing log softmax & KL loss
            with th.autocast("cuda", dtype=th.bfloat16):
                preds = ddp_lens(h, idx=i)

                preds = preds.to(labels.device)
                if shift > 0:
                    preds = preds[:, :-shift]
                elif shift < 0:
                    preds = preds[:, -shift:]

                if args.loss == "ce":
                    loss = th.nn.functional.cross_entropy(
                        preds.flatten(0, -2), labels.flatten()
                    )
                elif args.loss == "kl":
                    loss = th.sum(
                        labels.exp() * (labels - preds.log_softmax(-1)), dim=-1
                    ).mean()
                else:
                    raise NotImplementedError

                scaled_loss = loss / grad_acc_steps

            scaled_loss.backward()
            metrics[f"loss/{name}"].append(loss.detach())

        if step % grad_acc_steps == 0:
            th.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            scheduler.step()

            if local_rank == 0 and args.wandb:
                import wandb

                log_dict = {k: th.stack(v).mean().item() for k, v in metrics.items()}
                # Approximate the true gradient norm using Adam's moving average
                for i, l in enumerate(lens):
                    name = "input" if i == 0 else f"{i - 1}.ffn"

                    log_dict["grad_norm/" + name] = th.cat(
                        [opt.state[p]["exp_avg"].flatten() for p in l.parameters()]
                    ).norm()

                metrics.clear()
                wandb.log(log_dict)

        # Make the problem strictly convex with projected gradient descent,
        # centering the affine transform and normalizing the scale
        lens.normalize_()

    if local_rank == 0:
        print(f"Saving lens to {args.output}")
        lens.save(args.output)
        th.save(opt.state_dict(), args.output / "optimizer.pt")

    if args.residual_stats:
        stream_stats.all_reduce_()
        residual_stats.all_reduce_()

        if local_rank == 0:
            th.save(stream_stats, args.output / "stream_stats.pt")
            th.save(residual_stats, args.output / "residual_stats.pt")


if __name__ == "__main__":
    main()
