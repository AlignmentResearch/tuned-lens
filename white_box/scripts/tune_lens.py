"""Train a set of tuned lenses for a language model."""

from argparse import ArgumentParser
from collections import defaultdict
from datasets import Dataset, DatasetDict, load_dataset
from itertools import islice
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from white_box import ResidualStats, TunedLens, TunedLensWrapper
from white_box.data import chunk_and_tokenize, silence_datasets_messages
from white_box.utils import send_to_device
import deepspeed
import torch as th


def _get_momentum_norms(adam: th.optim.Adam) -> list:
    """Get the momentum norms for each tuned lens using hacky heuristics.

    Unfortunately this is needed because when Deepspeed wraps the optimizer,
    it apparently copies the parameter *objects* used to key the state dict,
    so we can't directly key into the state dict to get the momentum norms.
    """

    # Thank God for order preservation in Python dicts
    states = list(adam.state.values())
    step = states[0].get("step", 1)
    beta1, _ = adam.param_groups[0]["betas"]

    assert len(states) % 2 == 0
    return [
        th.cat(
            [
                weight.get("exp_avg", th.tensor([0.0])).flatten(),
                bias.get("exp_avg", th.tensor([0.0])),
            ]
        )
        .norm()
        .div(1 - beta1**step)
        for weight, bias in zip(*[iter(states)] * 2)  # type: ignore
    ]


@th.autocast("cuda")
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
        "--grad-acc-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="Local rank for deepspeed training."
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
        "--snapshot-every",
        type=int,
        default=10,
        help="Save a copy of the lenses in CPU RAM every N steps.",
    )
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
        default=2**19,
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
    if args.local_rank == 0:
        print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
    ).to(f"cuda:{args.local_rank}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_name)

    # Just for type checking
    assert isinstance(model, PreTrainedModel)
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
    processed = processed.shuffle(seed=args.seed)

    # chunk_and_tokenize ensures the samples are all the same length
    tokens_per_sample = len(processed[0]["input_ids"])
    samples_per_step, rem = divmod(args.tokens_per_step, tokens_per_sample)
    if rem:
        raise ValueError(
            f"Number of tokens per step ({args.tokens_per_step:_}) must be divisible "
            f"by the number of tokens per sample ({tokens_per_sample})."
        )

    if args.local_rank == 0:
        print(f"Using {args.tokens_per_step:_} tokens per training step.")

    lens = TunedLens(
        model,
        include_final=args.train_final_lens,
        orthogonal=args.orthogonal,
        rank=args.rank,
        sublayers=args.sublayers,
    )

    if args.wandb and args.local_rank == 0:
        import wandb

        wandb.init(
            name=args.wandb, project=args.model_name.split("/")[-1], config=vars(args)
        )
        wandb.watch(lens)

    # Running mean & variance of the residuals
    residual_stats = ResidualStats()
    stream_stats = ResidualStats()

    # Skip the unembedding matrix and final layer norm
    params = [p for p in lens.parameters() if p.requires_grad]
    opt = th.optim.Adam(params, lr=args.lr)

    if args.resume:
        # Get the most recently saved lens from the directory
        if args.resume.is_dir():
            ckpt = max(args.resume.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        else:
            ckpt = args.resume

        print(f"Loading checkpoint from {ckpt}")
        opt_path = ckpt / "optimizer.pt"
        lens.load_state_dict(th.load(ckpt))

        if opt_path.exists():
            print(f"Loading optimizer state from {opt_path}")
            opt.load_state_dict(th.load(opt_path))
        else:
            print("No optimizer state found. Starting Adam from scratch.")

    model, opt, dl, _ = deepspeed.initialize(
        args=args,
        config=dict(
            fp16=dict(
                auto_cast=True,
                enabled=True,
                # Don't waste the first few training steps reducing the scale
                # down from 2 ** 16
                initial_scale_power=10,
            ),
            gradient_clipping=1.0,
            train_batch_size=samples_per_step,
            train_micro_batch_size_per_gpu=args.per_gpu_batch_size,
        ),
        model_parameters=params,  # type: ignore[arg-type]
        model=TunedLensWrapper(model, lens),
        lr_scheduler=LambdaLR(opt, lambda t: 1 - t / args.num_steps),
        optimizer=opt,
        mpu=None,
        training_data=processed,
    )

    first_device = next(model.parameters()).device
    grad_acc_steps = model.gradient_accumulation_steps()
    metrics = defaultdict(list)
    total_batches = args.num_steps * grad_acc_steps

    for step, batch in enumerate(islice(dl, total_batches), start=1):
        assert isinstance(batch, dict)
        batch = send_to_device(batch, first_device)
        output, stream = model(**batch)
        final_logits = output.logits

        stream = stream.map(lambda t: t.float())
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
        for (name, preds), mom_norm in zip(
            lens.map(stream.items()), _get_momentum_norms(opt)
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
            elif args.loss == "kl":
                loss = th.sum(
                    labels.exp() * (labels - preds.log_softmax(-1)), dim=-1
                ).mean()
            else:
                raise NotImplementedError

            model.backward(loss)

            # Use Adam momentum buffers to get a lower variance estimate of the
            # true gradient (and therefore grad norm) at this step
            metrics[f"grad_norm/{name}"].append(mom_norm)
            metrics[f"loss/{name}"].append(loss.detach())

        # Deepspeed keeps track of grad accumulation
        model.step()
        lens.normalize_()

        if args.local_rank == 0 and args.wandb and step % grad_acc_steps == 0:
            import wandb

            wandb.log({k: th.stack(v).mean().item() for k, v in metrics.items()})
            metrics.clear()

    if args.local_rank == 0:
        print(f"Saving lens to {args.output}")
        lens.save(args.output)
        th.save(opt.state_dict(), args.output / "optimizer.pt")

    if args.residual_stats:
        stream_stats.all_reduce_()
        residual_stats.all_reduce_()

        if args.local_rank == 0:
            th.save(stream_stats, args.output / "stream_stats.pt")
            th.save(residual_stats, args.output / "residual_stats.pt")


if __name__ == "__main__":
    main()
