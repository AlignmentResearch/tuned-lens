"""Script to train or evaluate a set of tuned lenses for a language model."""

import enum
from typing import List, Optional
from .scripts.lens import main as lens_main
from contextlib import nullcontext, redirect_stdout
import os
import torch.distributed as dist
from .scripts.train_loop import CliArgs as TrainArgs
from .scripts.downstream import CliArgs as DownstreamArgs
from .scripts.eval_loop import CliArgs as EvalArgs
from simple_parsing import ArgumentParser, field
from dataclasses import dataclass


class LossChoice(enum.Enum):
    CE = "ce"
    KL = "kl"


@dataclass
class CliArgs:
    """Arguments shared by train and eval; see https://stackoverflow.com/a/56595689."""

    model_name: str = field(positional=True)
    """Name of model to use in the Huggingface Hub."""

    dataset: Optional[List[str]] = field(
        positional=True, default=["the_pile", "all"], nargs="*"
    )
    """Name of dataset to use. Can either be a local .jsonl file or a name
    suitable to be passed to the HuggingFace load_dataset function."""

    cpu_offload: Optional[bool] = field(action="store_true")
    """Use CPU offloading. Must be combined with --fsdp."""

    fsdp: Optional[bool] = field(action="store_true")
    """Run the model with Fully Sharded Data Parallelism."""

    loss: LossChoice = LossChoice.KL
    """Loss function to use."""

    no_cache: Optional[bool] = field(action="store_true")
    """Don't permanently cache the model on disk."""

    per_gpu_batch_size: int = 1
    """Number of samples to try to fit on a GPU at once."""

    random_model: Optional[bool] = field(action="store_true")
    """Use a randomly initialized model instead of pretrained weights."""

    residual_stats: Optional[bool] = field(action="store_true")
    """Save means and covariance matrices for states in the residual stream."""

    revision: Optional[str] = "main"
    """Git revision to use for pretrained models."""

    seed: int = 42
    """Random seed for data shuffling."""

    slow_tokenizer: Optional[bool] = field(action="store_true")
    """Use a slow tokenizer."""

    split: str = "validation"
    """Split of the dataset to use."""

    sweep: Optional[str] = None
    """Range of checkpoints to sweep over"""

    task: List[str] = field(nargs="+")
    """lm-eval task to run the model on."""

    tokenizer: Optional[str] = None
    """Name of pretrained tokenizer to use from the Huggingface Hub. If None, will use 
    AutoTokenizer.from_pretrained('<model name>')."""

    tokenizer_type: Optional[str] = None
    """Name of tokenizer class to use. If None, will use AutoTokenizer."""

    text_column: str = "text"
    """Column of the dataset containing text to run the model on."""

    token_shift: Optional[int] = None
    """How to shift the labels wrt the input tokens (1 = next token, 0 = current token, 
    -1 = previous token, etc.)"""


def run():
    """Run the script."""
    parser = ArgumentParser(
        description="Train or evaluate a set of tuned lenses for a language model.",
    )
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_arguments(CliArgs, dest="options")

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", parents=[parent_parser])
    downstream_parser = subparsers.add_parser("downstream", parents=[parent_parser])
    eval_parser = subparsers.add_parser("eval", parents=[parent_parser])

    train_parser.add_arguments(TrainArgs, dest="options")
    downstream_parser.add_arguments(DownstreamArgs, dest="options")
    eval_parser.add_arguments(EvalArgs, dest="options")

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
