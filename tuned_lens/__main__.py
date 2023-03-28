"""Script to train or evaluate a set of tuned lenses for a language model."""

from .scripts.lens import main as lens_main
from argparse import ArgumentParser
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
import os
import torch.distributed as dist


def run():
    """Run the script."""
    parser = ArgumentParser(
        description="Train or evaluate a set of tuned lenses for a language model.",
    )
    # Arguments shared by train and eval; see https://stackoverflow.com/a/56595689.
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "model_name", type=str, help="Name of model to use in the Huggingface Hub."
    )
    parent_parser.add_argument(
        "dataset",
        type=str,
        default=("the_pile", "all"),
        nargs="*",
        help="Name of dataset to use. Can either be a local .jsonl file or a name "
        "suitable to be passed to the HuggingFace load_dataset function.",
    )
    parent_parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Use CPU offloading. Must be combined with --fsdp.",
    )
    parent_parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Run the model with Fully Sharded Data Parallelism.",
    )
    parent_parser.add_argument(
        "--loss",
        type=str,
        default="kl",
        choices=("ce", "kl"),
        help="Loss function to use.",
    )
    parent_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't permanently cache the model on disk.",
    )
    parent_parser.add_argument(
        "--per-gpu-batch-size",
        type=int,
        default=1,
        help="Number of samples to try to fit on a GPU at once.",
    )
    parent_parser.add_argument(
        "--random-model",
        action="store_true",
        help="Use a randomly initialized model instead of pretrained weights.",
    )
    parent_parser.add_argument(
        "--residual-stats",
        action="store_true",
        help="Save means and covariance matrices for states in the residual stream.",
    )
    parent_parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Git revision to use for pretrained models.",
    )
    parent_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for data shuffling."
    )
    parent_parser.add_argument(
        "--slow-tokenizer", action="store_true", help="Use a slow tokenizer."
    )
    parent_parser.add_argument(
        "--split", type=str, default="validation", help="Split of the dataset to use."
    )
    parent_parser.add_argument(
        "--sweep", type=str, help="Range of checkpoints to sweep over"
    )
    parent_parser.add_argument(
        "--task", type=str, nargs="+", help="lm-eval task to run the model on."
    )
    parent_parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column of the dataset containing text to run the model on.",
    )
    parent_parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name of pretrained tokenizer to use from the Huggingface Hub. If None, "
        'will use AutoTokenizer.from_pretrained("<model name>").',
    )
    parent_parser.add_argument(
        "--tokenizer-type",
        type=str,
        help="Name of tokenizer class to use. If None, will use AutoTokenizer.",
    )
    parent_parser.add_argument(
        "--token-shift",
        type=int,
        default=None,
        help="How to shift the labels wrt the input tokens (1 = next token, "
        "0 = current token, -1 = previous token, etc.)",
    )

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", parents=[parent_parser])
    downstream_parser = subparsers.add_parser("downstream", parents=[parent_parser])
    eval_parser = subparsers.add_parser("eval", parents=[parent_parser])

    # Training-only arguments
    train_parser.add_argument(
        "--constant", action="store_true", help="Train only the bias term."
    )
    train_parser.add_argument(
        "--extra-layers", type=int, default=0, help="Number of extra decoder layers."
    )
    train_parser.add_argument(
        "--lasso", type=float, default=0.0, help="LASSO (L1) regularization strength."
    )
    train_parser.add_argument(
        "--lens", type=Path, help="Directory containing a lens to warm-start training."
    )
    train_parser.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help="The default LR (1e-3 for Adam, 1.0 for SGD) is scaled by this factor.",
    )
    train_parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum coefficient for SGD, or beta1 for Adam.",
    )
    train_parser.add_argument(
        "--num-steps", type=int, default=250, help="Number of training steps."
    )
    train_parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=("adam", "sgd"),
        help="The type of optimizer to use.",
    )
    train_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="File to save the lenses to. Defaults to the model name.",
    )
    train_parser.add_argument(
        "--pre-ln",
        action="store_true",
        help="Apply layer norm before, and not after, each probe.",
    )
    train_parser.add_argument(
        "--resume", type=Path, help="File to resume training from."
    )
    train_parser.add_argument(
        "--separate-unembeddings",
        action="store_true",
        help="Learn a separate unembedding for each layer.",
    )
    train_parser.add_argument(
        "--tokens-per-step",
        type=int,
        default=2**18,
        help="Number of tokens per step.",
    )
    train_parser.add_argument(
        "--wandb", type=str, help="Name of run in Weights & Biases."
    )
    train_parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Number of warmup steps. Defaults to min(0.1 * num_steps, 1000) for Adam"
        " and 0 for SGD.",
    )
    train_parser.add_argument(
        "--weight-decay", type=float, default=1e-3, help="Weight decay coefficient."
    )
    train_parser.add_argument(
        "--zero", action="store_true", help="Use ZeroRedundancyOptimizer."
    )

    downstream_parser.add_argument(
        "--lens",
        type=Path,
        help="Directory containing the tuned lens to evaluate.",
        nargs="?",
    )
    downstream_parser.add_argument(
        "--injection", action="store_true", help="Simulate a prompt injection attack."
    )
    downstream_parser.add_argument(
        "--incorrect-fewshot", action="store_true", help="Permute the fewshot labels."
    )
    downstream_parser.add_argument(
        "--num-shots",
        type=int,
        default=0,
        help="Number of examples to use for few-shot evaluation.",
    )
    downstream_parser.add_argument(
        "--limit", type=int, default=500, help="Number of samples to evaluate on."
    )
    downstream_parser.add_argument(
        "-o", "--output", type=Path, help="Folder to save the results to."
    )

    # Evaluation-only arguments
    eval_parser.add_argument(
        "--lens",
        type=Path,
        help="Directory containing the tuned lens to evaluate.",
        nargs="?",
    )
    eval_parser.add_argument(
        "--grad-alignment", action="store_true", help="Evaluate gradient alignment."
    )
    eval_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of batches to evaluate on. If None, will use the entire dataset.",
    )
    eval_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="JSON file to save the eval results to.",
    )
    eval_parser.add_argument(
        "--transfer",
        action="store_true",
        help="Evaluate how well probes transfer to other layers.",
    )

    args = parser.parse_args()

    # Support both distributed and non-distributed training
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        dist.init_process_group("nccl")
        local_rank = int(local_rank)

    # Only print on rank 0
    with nullcontext() if not local_rank else redirect_stdout(None):
        args = parser.parse_args()

        if args.command is None:
            parser.print_help()
            exit(1)

        if args.sweep:
            ckpt_range = eval(f"range({args.sweep})")
            output_root = args.output
            assert output_root is not None
            print(f"Running sweep over {len(ckpt_range)} checkpoints.")

            for step in ckpt_range:
                step_output = output_root / f"step{step}"
                print(f"Running for step {step}, saving to '{step_output}'...")

                args.output = step_output
                args.revision = f"step{step}"
                lens_main(args)
        else:
            lens_main(args)


if __name__ == "__main__":
    run()
