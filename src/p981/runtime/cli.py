"""p981-runtime 실행용 CLI 엔트리포인트 (스켈레톤)."""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p981.runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "run":
        print("p981.runtime CLI is not implemented yet. Use p981-core for local execution.")


if __name__ == "__main__":
    main()
