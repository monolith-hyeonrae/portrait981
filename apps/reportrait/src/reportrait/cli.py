"""Command-line interface for reportrait."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional


def _cmd_generate(args: argparse.Namespace) -> int:
    """Execute the generate subcommand."""
    from reportrait.generator import PortraitGenerator
    from reportrait.types import GenerationConfig, GenerationRequest
    from reportrait.workflow import inject_prompt, inject_references, load_template

    # 1. Resolve reference image paths
    if args.ref:
        # Direct reference images — validate existence
        ref_paths: list[str] = []
        for p in args.ref:
            path = Path(p)
            if not path.exists():
                print(f"Error: Reference image not found: {p}", file=sys.stderr)
                return 1
            ref_paths.append(str(path.resolve()))
    else:
        # Lookup from momentbank
        if not args.member_id:
            print("Error: member_id or --ref required", file=sys.stderr)
            return 1

        from momentbank.ingest import lookup_frames

        frames = lookup_frames(
            args.member_id,
            pose=args.pose,
            category=args.category,
            top_k=args.top,
        )

        if not frames:
            msg = f"No frames found for member '{args.member_id}'"
            if args.pose:
                msg += f" pose={args.pose}"
            if args.category:
                msg += f" category={args.category}"
            print(f"Error: {msg}", file=sys.stderr)
            return 1

        ref_paths = [f["path"] for f in frames]

    print(f"Found {len(ref_paths)} reference image(s)", file=sys.stderr)

    # 2. Dry-run: inject workflow and print JSON
    node_ids = args.node if args.node else None
    if args.dry_run:
        workflow = load_template(args.workflow)
        workflow = inject_references(workflow, ref_paths, node_ids=node_ids)
        if args.prompt:
            workflow = inject_prompt(workflow, args.prompt)
        json.dump(workflow, sys.stdout, indent=2)
        print()  # trailing newline
        return 0

    # 3. Generate via ComfyUI
    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY")
    output_dir: Optional[Path] = Path(args.output) if args.output else None
    config = GenerationConfig(
        comfy_url=args.comfy_url,
        api_key=api_key,
        workflow_template=args.workflow,
        output_dir=output_dir,
    )
    generator = PortraitGenerator(config)

    request = GenerationRequest(
        person_id=0,
        ref_paths=ref_paths,
        workflow_template=args.workflow,
        style_prompt=args.prompt,
        node_ids=node_ids,
    )
    result = generator.generate(request)

    if result.success:
        print(f"Generated {len(result.output_paths)} image(s) in {result.elapsed_sec:.1f}s")
        for p in result.output_paths:
            file_url = Path(p).resolve().as_uri()
            print(f"  {file_url}")
        return 0
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reportrait - AI portrait generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  reportrait generate test_3 --pose left30 --dry-run
  reportrait generate test_3 --category warm_smile --prompt "portrait"
  reportrait generate --ref photo1.jpg photo2.jpg --prompt "portrait"
  reportrait generate --ref face.jpg --workflow i2i_workflow.json --dry-run
  reportrait generate --ref face.jpg --workflow /path/to/i2v.json
  reportrait generate --ref face.jpg --comfy-url https://xxx.proxy.runpod.net --api-key $RUNPOD_API_KEY
""",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- generate ---
    gen_parser = subparsers.add_parser(
        "generate", help="Generate portrait from reference frames"
    )
    gen_parser.add_argument(
        "member_id", nargs="?", default=None,
        help="Member identifier for frame lookup (not needed with --ref)",
    )
    gen_parser.add_argument(
        "--ref", nargs="+", metavar="IMAGE",
        help="Reference image path(s) to inject directly (skips lookup_frames)",
    )
    gen_parser.add_argument(
        "--node", nargs="+", metavar="ID",
        help="Target LoadImage node ID(s) to inject refs into (default: auto-detect by _meta.role)",
    )
    gen_parser.add_argument(
        "--pose", default=None, help="Filter by pose (e.g. left30, frontal)"
    )
    gen_parser.add_argument(
        "--category", default=None, help="Filter by category (e.g. warm_smile)"
    )
    gen_parser.add_argument(
        "--top", type=int, default=3, help="Max reference images (default: 3)"
    )
    gen_parser.add_argument(
        "--prompt", default="", help="Style prompt (default: empty)"
    )
    gen_parser.add_argument(
        "--workflow", default="default",
        help="Workflow template name or path to .json file (default: default)"
    )
    gen_parser.add_argument(
        "--comfy-url",
        default="http://127.0.0.1:8188",
        help="ComfyUI server URL (default: http://127.0.0.1:8188)",
    )
    gen_parser.add_argument(
        "--api-key", default=None,
        help="API key for remote ComfyUI (e.g. RunPod). Falls back to RUNPOD_API_KEY env",
    )
    gen_parser.add_argument(
        "--output", default=None, help="Output directory (default: auto)"
    )
    gen_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print injected workflow JSON without calling ComfyUI",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.command == "generate":
        return _cmd_generate(args)

    parser.print_help()
    return 0
