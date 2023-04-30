"""Script to train or evaluate a set of tuned lenses for a language model."""
from dataclasses import dataclass
from typing import Optional, Union

from simple_parsing import ArgumentParser, ConflictResolution
from torch.distributed.elastic.multiprocessing.errors import record

from .scripts.eval_loop import Eval
from .scripts.train_loop import Train


@dataclass
class Main:
    """Routes to the subcommands."""

    command: Union[Train, Eval]

    def execute(self):
        """Run the script."""
        self.command.execute()


@record
def main(args: Optional[list[str]] = None):
    """Entry point for the CLI."""
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(Main, dest="prog")
    args = parser.parse_args(args=args)
    prog: Main = args.prog
    prog.execute()


if __name__ == "__main__":
    main()
