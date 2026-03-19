"""CLI entry point for annotator tools.

Subcommands:
    annotator label video.mp4 --fps 2 --output labels.html --max-frames 500
    annotator review data/datasets/portrait-v1 --output review.html
    annotator merge anchors_test2.zip anchors_test3.zip -o data/datasets/portrait-v1
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="annotator",
        description="Annotation tools for portrait dataset labeling, review, and merge",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- label ---
    p_label = sub.add_parser("label", help="Label video frames — server mode (--dataset) or static HTML")
    p_label.add_argument("video", help="mp4 video path")
    p_label.add_argument("--fps", type=int, default=2, help="frames per second to extract")
    p_label.add_argument("--dataset", "-d", default=None,
                         help="dataset directory for server mode (e.g. data/datasets/portrait-v1)")
    p_label.add_argument("--port", type=int, default=8766, help="server port (default: 8766)")
    p_label.add_argument("--output", "-o", default="labels.html", help="output HTML path (static mode)")
    p_label.add_argument("--max-frames", type=int, default=500, help="max frames to include")

    # --- review ---
    p_review = sub.add_parser("review", help="Review dataset — serve or generate HTML")
    p_review.add_argument("dataset", help="dataset directory (e.g. data/datasets/portrait-v1)")
    p_review.add_argument("--serve", action="store_true", help="Start local server (auto-save, no download needed)")
    p_review.add_argument("--port", type=int, default=8765, help="server port (default: 8765)")
    p_review.add_argument("--output", "-o", default="review.html", help="output HTML path (non-serve mode)")

    # --- merge ---
    p_merge = sub.add_parser("merge", help="Merge anchor ZIP files into dataset directory")
    p_merge.add_argument("zips", nargs="+", help="anchor ZIP files from label tool")
    p_merge.add_argument("--output", "-o", default="data/datasets/portrait-v1",
                         help="dataset directory (default: data/datasets/portrait-v1)")

    args = parser.parse_args(argv)

    if args.command == "label":
        if args.dataset:
            from annotator.label_server import start_label_server
            start_label_server(
                video_path=args.video,
                dataset_dir=args.dataset,
                fps=args.fps,
                max_frames=args.max_frames,
                port=args.port,
            )
        else:
            from annotator.label import generate_label_html
            generate_label_html(
                video_path=args.video,
                fps=args.fps,
                max_frames=args.max_frames,
                output_path=args.output,
            )

    elif args.command == "review":
        if args.serve:
            from annotator.serve import start_server
            start_server(dataset_dir=args.dataset, port=args.port)
        else:
            from annotator.review import generate_review_html
            generate_review_html(
                dataset_dir=args.dataset,
                output_path=args.output,
            )

    elif args.command == "merge":
        from annotator.merge import merge_zips
        merge_zips(
            zip_paths=args.zips,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
