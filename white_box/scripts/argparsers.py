from argparse import ArgumentParser
from pathlib import Path


def get_lens_parser() -> ArgumentParser:
    """Return the parser for the `lens` subcommand."""

    parser = ArgumentParser(
        description="Train or evaluate a set of tuned lenses for a language model."
    )
    # Arguments shared by train and eval; see https://stackoverflow.com/a/56595689.
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "model_name", type=str, help="Name of model to use in the Huggingface Hub."
    )
    parent_parser.add_argument(
        "--dataset",
        type=str,
        default=("wikitext", "wikitext-103-v1"),
        nargs="+",
        help="Name of dataset to use. Can either be a local .jsonl file or a name "
        "suitable to be passed to the HuggingFace load_dataset function.",
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
        "--per-gpu-batch-size",
        type=int,
        default=1,
        help="Number of samples to try to fit on a GPU at once.",
    )
    parent_parser.add_argument(
        "--split", type=str, default="validation", help="Split of the dataset to use."
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
        "--token-shift",
        type=int,
        default=None,
        help="How to shift the labels wrt the input tokens (1 = next token, "
        "0 = current token, -1 = previous token, etc.)",
    )

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", parents=[parent_parser])
    eval_parser = subparsers.add_parser("eval", parents=[parent_parser])

    # Training-only arguments
    train_parser.add_argument(
        "--grad-acc-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    train_parser.add_argument(
        "--lens", type=Path, help="Directory containing a lens to warm-start training."
    )
    train_parser.add_argument("--lr", type=float, default=0.002, help="Learning rate.")
    train_parser.add_argument(
        "--num-steps", type=int, default=100, help="Number of training steps."
    )
    train_parser.add_argument(
        "--orthogonal",
        action="store_true",
        help="Parametrize the tuned lenses as rotation matrices.",
    )
    train_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="File to save the lenses to. Defaults to the model name.",
    )
    train_parser.add_argument("--rank", type=int, help="Rank of the tuned lenses.")
    train_parser.add_argument(
        "--resume", type=Path, help="File to resume training from."
    )
    train_parser.add_argument(
        "--sublayers",
        action="store_true",
        help="Train tuned lenses for attention blocks.",
    )
    train_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for training."
    )
    train_parser.add_argument(
        "--tokens-per-step",
        type=int,
        default=2**18,
        help="Number of tokens per step.",
    )
    train_parser.add_argument(
        "--train-final-lens",
        action="store_true",
        help="Train a lens for the final layer even though it's superfluous.",
    )
    train_parser.add_argument(
        "--wandb", type=str, help="Name of run in Weights & Biases."
    )

    # Evaluation-only arguments
    eval_parser.add_argument(
        "lens", type=Path, help="Directory containing the tuned lens to evaluate."
    )
    eval_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="JSON file to save the eval results to.",
    )
    eval_parser.add_argument(
        "--residual-stats",
        action="store_true",
        help="Save means and covariance matrices for states in the residual stream.",
    )

    return parser
