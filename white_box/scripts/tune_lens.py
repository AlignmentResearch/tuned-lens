"""Train a set of tuned lenses for a language model."""

from collections import defaultdict
from accelerate.utils import find_executable_batch_size, send_to_device
from argparse import ArgumentParser
from datasets import Dataset, DatasetDict, load_dataset
from itertools import islice
from white_box.data import chunk_and_tokenize, silence_datasets_messages
from logit_lens import ResidualStats, record_residual_stream, TunedLens
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from tqdm import tqdm
import torch as th


def main():
    parser = ArgumentParser(
        description="Train a set of tuned lenses for a language model."
    )
    parser.add_argument(
        "model_name", type=str, help="Name of model to use in the Huggingface Hub."
    )
    parser.add_argument(
        "--batch-size", type=int, default=6, help="Per-GPU batch size for training."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of dataset to use. Can either be a local .jsonl file or a name "
        "suitable to be passed to the HuggingFace load_dataset function.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training."
    )
    parser.add_argument(
        "--fp16",
        type=str,
        choices=("full", "mixed", "none"),
        default="none",
        help="Whether to use mixed precision training.",
    )
    parser.add_argument(
        "--grad-acc-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="kl",
        choices=("ce", "cosine", "kl", "mse"),
        help="Loss function to use for training.",
    )
    parser.add_argument(
        "--num-steps", type=int, default=1000, help="Number of training steps."
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
        "--save-every", type=int, default=50, help="Save the lenses every N steps."
    )
    parser.add_argument(
        "--no-sublayers",
        action="store_true",
        help="Omit tuned lenses for attention blocks.",
    )
    parser.add_argument(
        "--shard-model", action="store_true", help="Shard the model across GPUs."
    )
    parser.add_argument(
        "--slow-tokenizer", action="store_true", help="Use a Python tokenizer."
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
        "--token-shift",
        type=int,
        default=None,
        help="How to shift the labels wrt the input tokens (1 = next token, "
        "0 = current token, -1 = previous token, etc.)",
    )
    parser.add_argument("--wandb", type=str, help="Name of run in Weights & Biases.")
    args = parser.parse_args()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto" if args.shard_model else None,
        torch_dtype="auto",
    )
    if not args.shard_model:
        model = model.to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=not args.slow_tokenizer
    )

    # Just for type checking
    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    silence_datasets_messages()

    if args.dataset.endswith(".jsonl"):
        dataset = Dataset.from_json(args.dataset)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
        if not isinstance(dataset, (Dataset, DatasetDict)):
            raise ValueError("Only Dataset and DatasetDict instances are supported.")

    processed = chunk_and_tokenize(dataset, tokenizer, text_key=args.text_column)
    processed = processed.shuffle(seed=args.seed)

    lens = TunedLens(
        model,
        orthogonal=args.orthogonal,
        rank=args.rank,
        sublayers=not args.no_sublayers,
    ).float()

    if args.resume:
        # Get the most recently saved lens from the directory
        if args.resume.is_dir():
            ckpt = max(args.resume.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        else:
            ckpt = args.resume

        print(f"Resuming training from {ckpt}")
        lens.load_state_dict(th.load(ckpt))

    if args.wandb:
        import wandb

        wandb.init(
            name=args.wandb, project=args.model_name.split("/")[-1], config=vars(args)
        )
        wandb.watch(lens)

    # Running mean & variance of the residuals
    residual_stats = ResidualStats()
    stream_stats = ResidualStats()

    num_batches = args.grad_acc_steps * args.num_steps
    pbar = tqdm(desc="Tuning", total=num_batches)

    dl = DataLoader(processed, batch_size=args.batch_size)  # type: ignore[arg-type]
    opt = th.optim.AdamW(lens.parameters(), amsgrad=True)
    schedule = th.optim.lr_scheduler.CosineAnnealingLR(opt, args.num_steps)

    use_autocast = model.dtype == th.float16
    if use_autocast:
        print("Using fp16 inference for the model.")

    first_device = next(model.parameters()).device

    for step in range(args.num_steps):
        metrics = defaultdict(list)

        for batch in islice(dl, args.grad_acc_steps):
            batch = send_to_device(batch, first_device)
            with (
                th.autocast("cuda", enabled=use_autocast),
                record_residual_stream(
                    model, sublayers=not args.no_sublayers
                ) as stream,
                th.no_grad(),
            ):
                final_logits = model(**batch).logits.float()
            stream = stream.map(lambda t: t.float())
            if args.residual_stats:
                sample_residuals = stream.residuals()
                residual_stats.update(sample_residuals)
                stream_stats.update(stream)

            shift = args.token_shift
            use_logits = args.loss not in ("cosine", "mse")
            if not use_logits:
                labels = stream.layers[-1]

                # Predict the final hidden state for the *current* token by default
                if shift is None:
                    shift = 0
            else:
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
            for mod, (name, preds) in zip(
                lens, lens.map(stream.items(), logits=use_logits)
            ):
                preds = preds.to(labels.device)
                if shift > 0:
                    preds = preds[:, :-shift]
                elif shift < 0:
                    preds = preds[:, -shift:]

                if args.loss == "ce":
                    loss = th.nn.functional.cross_entropy(
                        preds.flatten(0, -2), labels.flatten()
                    )
                elif args.loss == "cosine":
                    loss = 1 - th.cosine_similarity(preds, labels, dim=-1).mean()
                elif args.loss == "kl":
                    loss = th.sum(
                        labels.exp() * (labels - preds.log_softmax(-1)), dim=-1
                    ).mean()
                elif args.loss == "mse":
                    loss = th.nn.functional.mse_loss(preds, labels)
                else:
                    raise NotImplementedError

                loss.div(args.grad_acc_steps).backward()
                metrics[f"grad_norm/{name}"].append(
                    th.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
                )
                metrics[f"loss/{name}"].append(loss.detach())

            # End of batch
            pbar.update()

        if args.wandb:
            import wandb

            wandb.log({k: th.stack(v).mean().item() for k, v in metrics.items()})

        opt.step()
        opt.zero_grad()
        schedule.step()

        if step % args.save_every == 0:
            lens.save(args.output, f"latest.pt")

    print(f"Saving lens to {args.output}")
    lens.save(args.output)
    if args.residual_means:
        th.save(residual_stats, args.output / "residual_stats.pt")


if __name__ == "__main__":
    main()
