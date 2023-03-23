"""Provides function for creating the argument parse for the CBE command."""
from argparse import ArgumentParser
from pathlib import Path


def get_cbe_parser() -> ArgumentParser:
    """Return the parser for the `bases` subcommand."""
    parser = ArgumentParser(
        description="Train or evaluate a set of causal bases for a language model.",
        add_help=False,
    )

    # Arguments shared by train and eval; see https://stackoverflow.com/a/56595689.
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--mode", type=str, choices=("mean", "resample", "zero"), default="mean"
    )
    parent_parser.add_argument("-o", "--output", type=Path)

    subparsers = parser.add_subparsers(dest="cbe_command", required=True)
    eval_parser = subparsers.add_parser("eval", parents=[parent_parser])
    extract_parser = subparsers.add_parser("extract", parents=[parent_parser])

    # Basis evaluation-only arguments
    eval_parser.add_argument(
        "bases", type=Path, help="Directory containing the causal bases to evaluate."
    )
    eval_parser.add_argument(
        "--k", type=int, default=1, help="Number of features to use.", required=True
    )
    eval_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of batches to evaluate on. If None, will use the entire dataset.",
    )

    # Basis extraction-only arguments
    extract_parser.add_argument(
        "lens", type=Path, help="Directory containing the tuned lens to use.", nargs="?"
    )
    extract_parser.add_argument(
        "--no-translator", action="store_true", help="Do not use learned probes."
    )
    extract_parser.add_argument(
        "--k", type=int, default=50, help="Number of basis vectors to extract."
    )
    return parser
