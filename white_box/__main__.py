from argparse import ArgumentParser
from .scripts.argparsers import get_lens_parser


if __name__ == "__main__":
    parser = ArgumentParser(description="Run the white-box lensing code.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    lens_parser = get_lens_parser()
    lens_parser.set_defaults(func="white_box.scripts.lens.main")
    subparsers.add_parser("lens", parents=[lens_parser])

    args = parser.parse_args()
    if args.command == "lens":
        from .scripts.lens import main

        main(args)
    else:
        raise ValueError(f"Unknown command '{args.command}'")
