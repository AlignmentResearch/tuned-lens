"""Script to train or evaluate a set of tuned lenses for a language model."""

import logging
import os
from dataclasses import dataclass
from typing import Literal, Optional, Union

from simple_parsing import ArgumentParser, ConflictResolution
from torch.distributed.elastic.multiprocessing.errors import record

from .scripts.eval_loop import Eval
from .scripts.train_loop import Train


@dataclass
class Main:
    """Routes to the subcommands."""

    command: Union[Train, Eval]

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    """The log level to use."""

    def execute(self):
        """Run the script."""
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            FORMAT = f"[%(levelname)s] rank={local_rank} %(message)s"
        else:
            FORMAT = "[%(levelname)s] %(message)s"

        logging.basicConfig(level=self.log_level, format=FORMAT)
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
