"""Train a set of tuned lenses for a language model."""

from accelerate.utils import send_to_device
from argparse import ArgumentParser
from datasets import Dataset, DatasetDict, load_dataset
from itertools import islice
from logit_lens.data import chunk_and_tokenize, silence_datasets_messages
from logit_lens import record_residual_stream, TunedLens
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
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
        "--grad-acc-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
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
        type=str,
        default=None,
        help="File to save the lenses to. Defaults to the model name.",
    )
    parser.add_argument(
        "--slow-tokenizer", action="store_true", help="Use a Python tokenizer."
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
        default=1,
        help="How to shift the labels wrt the input tokens (1 = next token, "
        "0 = current token, -1 = previous token, etc.)",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
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
    lens = TunedLens(model, orthogonal=args.orthogonal).to(args.device)

    dl = DataLoader(processed, batch_size=args.batch_size)  # type: ignore[arg-type]
    opt = th.optim.Adam(lens.parameters())
    schedule = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=min(args.num_steps // 5, 1000),
        num_training_steps=args.num_steps,
    )

    if args.wandb:
        import wandb

        wandb.init(project=args.model_name.split("/")[-1], config=vars(args))
        wandb.watch(lens)

    # Exponential moving average of the loss
    ema = 0.0
    beta = 0.9

    num_batches = args.grad_acc_steps * args.num_steps
    pbar = tqdm(desc="Tuning", total=num_batches)
    for batch in islice(dl, num_batches):
        batch = send_to_device(batch, args.device)
        with record_residual_stream(model) as stream, th.no_grad():
            model(**batch)

        metrics = {}
        total_loss = 0.0

        # We do this sequentially to save VRAM
        for name, logits in lens.iter_logits(stream.items()):
            labels = batch["input_ids"]
            if args.token_shift > 0:
                labels = labels[:, args.token_shift :]
                logits = logits[:, : -args.token_shift]
            elif args.token_shift < 0:
                labels = labels[:, : args.token_shift]
                logits = logits[:, -args.token_shift :]

            loss = th.nn.functional.cross_entropy(
                logits.flatten(0, -2), labels.flatten()
            )
            total_loss += loss
            metrics[name] = loss

        if args.wandb:
            import wandb

            wandb.log(metrics)

        total_loss = total_loss / len(lens)
        scaled_loss = total_loss / args.grad_acc_steps
        scaled_loss.backward()

        if (pbar.n + 1) % args.grad_acc_steps == 0:
            opt.step()
            opt.zero_grad()
            schedule.step()

        # Update the exponential moving average of the loss
        ema = beta * ema + (1 - beta) * float(total_loss)

        # Bias correction
        pbar.set_postfix(loss=ema / (1 - beta ** (pbar.n + 1)))
        pbar.update()

    if args.output is None:
        model_name = args.model_name.split("/")[-1]
        args.output = f"{model_name}_lens.pt"

    print(f"Saving lens to {args.output}")
    th.save(lens.state_dict(), args.output)


if __name__ == "__main__":
    main()
