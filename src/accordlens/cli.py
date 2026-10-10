"""Command-line experiment entry points."""

import argparse
from pathlib import Path

from .experiment import judge, train
from .inference import respond


def main():
    parser = argparse.ArgumentParser(
        description="Preference alignment and judge reliability experiments"
    )
    parser.add_argument("command", choices=["train", "judge", "respond"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--model", required=True, help="Local pinned Qwen model directory"
    )
    parser.add_argument("--adapter", default="artifacts/dpo")
    parser.add_argument("--prompt")
    args = parser.parse_args()
    if args.command == "respond":
        if not args.prompt:
            parser.error("--prompt is required for respond")
        print(respond(args.model, args.root / args.adapter, args.prompt))
    else:
        {"train": train, "judge": judge}[args.command](args.root.resolve(), args.model)


if __name__ == "__main__":
    main()
