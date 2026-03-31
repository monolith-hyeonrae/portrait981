"""PersonMemory CLI — person memory management.

Usage:
    personmemory list                              # 전체 member 목록
    personmemory show test_3                       # member 프로필
    personmemory rename test_3 member_042          # member_id 변경
    personmemory delete test_3                     # member 삭제

Ingest는 momentscan에서 수행:
    momentscan run video.mp4 --ingest test_3
"""

import argparse
import sys
import logging


def main():
    parser = argparse.ArgumentParser(
        description="PersonMemory — person memory management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  personmemory list
  personmemory show test_3
  personmemory rename test_3 member_042
  personmemory delete test_3

Ingest via momentscan:
  momentscan run video.mp4 --ingest test_3
""",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    # list
    subparsers.add_parser("list", help="List all members")

    # show
    show_parser = subparsers.add_parser("show", help="Show member profile")
    show_parser.add_argument("member_id")

    # rename
    rename_parser = subparsers.add_parser("rename", help="Rename member ID")
    rename_parser.add_argument("old_id")
    rename_parser.add_argument("new_id")

    # delete
    delete_parser = subparsers.add_parser("delete", help="Delete member memory")
    delete_parser.add_argument("member_id")
    delete_parser.add_argument("--yes", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.command == "list":
        _cmd_list()
    elif args.command == "show":
        _cmd_show(args.member_id)
    elif args.command == "rename":
        _cmd_rename(args.old_id, args.new_id)
    elif args.command == "delete":
        _cmd_delete(args.member_id, args.yes)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_list():
    from personmemory import PersonMemory

    members = PersonMemory.list_all()
    if not members:
        print("No members found.")
        return

    print(f"{'Member ID':<20s} {'Nodes':>5s} {'Visits':>6s} {'Frames':>7s}")
    print("-" * 42)
    for mid in members:
        mem = PersonMemory(mid)
        p = mem.profile()
        print(f"{mid:<20s} {p.n_nodes:>5d} {p.n_visits:>6d} {p.n_total_frames:>7d}")


def _cmd_show(member_id: str):
    from personmemory import PersonMemory

    mem = PersonMemory(member_id)
    if not mem.nodes:
        print(f"Member '{member_id}' not found or empty.")
        return

    p = mem.profile()

    print(f"\n{'='*50}")
    print(f"  Member: {p.member_id}")
    print(f"  Visits: {p.n_visits}  |  Frames: {p.n_total_frames}  |  Nodes: {p.n_nodes}")
    print(f"{'='*50}")

    print(f"\n  Expression distribution:")
    for expr, ratio in sorted(p.expression_dist.items(), key=lambda x: -x[1]):
        bar = "#" * int(ratio * 30)
        print(f"    {expr:<8s} {ratio:5.0%} {bar}")

    print(f"\n  Pose distribution:")
    for pose, ratio in sorted(p.pose_dist.items(), key=lambda x: -x[1]):
        bar = "#" * int(ratio * 30)
        print(f"    {pose:<8s} {ratio:5.0%} {bar}")

    print(f"\n  Memory nodes:")
    for n in sorted(mem.nodes, key=lambda n: -n.n_observed):
        img = "+" if n.best_frame_path else "-"
        print(f"    {n.expression:<8s} {n.pose:<6s}  n={n.n_observed:3d}  conf={n.best_confidence:.0%}  img={img}")

    if p.gaps:
        print(f"\n  Coverage gaps ({len(p.gaps)}):")
        print(f"    {', '.join(p.gaps[:8])}")
        if len(p.gaps) > 8:
            print(f"    ... +{len(p.gaps) - 8} more")

    if p.signal_std is not None and len(p.signal_std) > 1:
        from visualbind.signals import SIGNAL_FIELDS
        fields = list(SIGNAL_FIELDS)
        top_var = sorted(zip(fields, p.signal_std), key=lambda x: -x[1])[:5]
        print(f"\n  Top signal variance:")
        for name, std in top_var:
            print(f"    {name:<22s} σ={std:.4f}")

    print()


def _cmd_rename(old_id: str, new_id: str):
    from personmemory import PersonMemory
    PersonMemory.rename(old_id, new_id)
    print(f"Renamed: {old_id} → {new_id}")


def _cmd_delete(member_id: str, skip_confirm: bool):
    from personmemory import PersonMemory
    if not skip_confirm:
        answer = input(f"Delete member '{member_id}'? [y/N] ")
        if answer.lower() != "y":
            print("Cancelled.")
            return
    PersonMemory.delete(member_id)
    print(f"Deleted: {member_id}")


if __name__ == "__main__":
    main()
