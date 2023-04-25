"""Script to train or evaluate a set of tuned lenses for a language model."""
import os
from contextlib import nullcontext, redirect_stdout
from .scripts.train_loop import Train
from .scripts.eval_loop import Eval
from typing import Optional, Union
from dataclasses import dataclass
from torch import distributed as dist

from simple_parsing import ArgumentParser, ConflictResolution


@dataclass
class Main:
    """Routes to the subcommands."""

    command: Union[Train, Eval]

    def execute(self):
        """Run the script."""
        # Support both distributed and non-distributed training
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            dist.init_process_group("nccl")

        # Only print on rank 0
        with nullcontext() if not local_rank else redirect_stdout(None):
            self.command.execute()


def main(args: Optional[list[str]] = None):
    """Entry point for the CLI."""
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(Main, dest="prog")
    args = parser.parse_args(args=args)
    prog: Main = args.prog
    prog.execute()


if __name__ == "__main__":
    main()
