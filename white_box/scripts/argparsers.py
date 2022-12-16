from argparse import ArgumentParser
from pathlib import Path


def get_lens_parser() -> ArgumentParser:
    """Return the parser for the `lens` subcommand."""

    parser = ArgumentParser(
        description="Train or evaluate a set of tuned lenses for a language model.",
        add_help=False,
    )
    # Arguments shared by train and eval; see https://stackoverflow.com/a/56595689.
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "model_name", type=str, help="Name of model to use in the Huggingface Hub."
    )
    parent_parser.add_argument(
        "--dataset",
        type=str,
        default=("wikitext", "wikitext-2-raw-v1"),
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
        "--constant", action="store_true", help="Train only the bias term."
    )
    train_parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout prob for lens inputs."
    )
    train_parser.add_argument(
        "--lasso", type=float, default=0.0, help="LASSO (L1) regularization strength."
    )
    train_parser.add_argument(
        "--lens", type=Path, help="Directory containing a lens to warm-start training."
    )
    train_parser.add_argument(
        "--lens-dtype",
        type=str,
        default="float32",
        choices=("float16", "float32"),
        help="dtype of lens weights.",
    )
    train_parser.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help="The default LR (1e-3 for Adam, 1.0 for SGD) is scaled by this factor.",
    )
    train_parser.add_argument(
        "--mlp-hidden-sizes",
        type=int,
        nargs="+",
        default=[],
        help="Hidden sizes o1f the MLPs used in the probes.",
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
    train_parser.add_argument("--rank", type=int, help="Rank of the tuned lenses.")
    train_parser.add_argument(
        "--resume", type=Path, help="File to resume training from."
    )
    train_parser.add_argument(
        "--shared-mlp-hidden-sizes",
        type=int,
        nargs="+",
        default=[],
        help="Hidden sizes of the MLP shared by all probes.",
    )
    train_parser.add_argument(
        "--sublayers",
        action="store_true",
        help="Train tuned lenses for attention blocks.",
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
        "--weight-decay", type=float, default=0.0, help="Weight decay coefficient."
    )
    train_parser.add_argument(
        "--zero", action="store_true", help="Use ZeroRedundancyOptimizer."
    )

    # Evaluation-only arguments
    eval_parser.add_argument(
        "lens", type=Path, help="Directory containing the tuned lens to evaluate."
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

    return parser
