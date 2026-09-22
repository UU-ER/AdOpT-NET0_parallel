"""
Files finished runs from the top level of ``results`` into one folder per stage.

A sweep leaves its runs at the top level, which is where ``already_done``
looks for them while a stage is going. Once a stage has finished they are
clutter: the top level of ``results`` had 125 folders from four stages mixed
together after the nine node campaign. This moves them into one folder per
stage, named by the date, the case study and the stage.

Nothing is lost by moving a run. ``collect_profiles`` globs
``**/profile_summary.csv`` so the dataset still holds every run, and
``run_folders`` walks the archive, so ``already_done`` still recognises an
archived run and a stage started after filing does not run it again.

**Never run this while a stage is running.** The archive is walked once per
process and cached, and a run always lands at the top level, so moving
folders under a running sweep makes it re-run what it has already done.

Run it from the benchmark folder::

    python archive_runs.py                  # print the plan, move nothing
    python archive_runs.py --apply          # move
    python archive_runs.py --results Z:/... --apply

The rules are matched in order and the first one that matches wins, so a
pack, which is also an nl_node run, is filed as a pack.
"""

import argparse
import re
import shutil
from pathlib import Path

BASE = Path(__file__).parent
RESULTS_PATH = BASE / "results"

# (regular expression against the case name, folder it is filed into). The
# date is the day the stage ran, so a second campaign of the same stage does
# not land on top of the first
RULES = [
    (r"_fill\d+x\d+_job\d+$", "2026-09-22_nl_node_stage15_workers"),
    (r"_pack\d+x\d+_job\d+$", "2026-09-22_nl_node_stage14_contention"),
    # Stages 12 and 13 share their cells at one and four threads: those ran
    # inside stage 12, so they are filed with it and stage 13's manifest
    # points into this folder. The manifest is what says which stage a run
    # belongs to, the folder only says where it lives
    (r"^nl_node_td\d+_.*_cut0_.*_tl4\.0$", "2026-09-22_nl_node_stage12_cuts"),
    (r"^nl_node_td\d+_.*_thr[14]_tl4\.0$", "2026-09-22_nl_node_stage12_cuts"),
    (r"^nl_node_td\d+_.*_thr\d+_tl[48]\.0$", "2026-09-22_nl_node_stage13_threads"),
    (r"^nl_node_", "2026-09-21_nl_node_pilots"),
]


def case_name_of(folder: Path):
    """
    The case name a result folder carries, without its timestamp and counter

    :param Path folder: result folder
    :return: case name as the run was given it
    """
    without_timestamp = re.sub(r"^\d{14}_", "", folder.name)
    return re.sub(r"-\d+$", "", without_timestamp)


def plan(results: Path):
    """
    Works out where every loose run folder should go.

    Only the top level is considered: a folder that is already filed stays
    where it is, so the script can be run twice without moving anything the
    second time.

    :param Path results: the results folder
    :return: list of (folder, target folder name) tuples, and the names that
        matched no rule
    """
    moves, unmatched = [], []

    for folder in sorted(results.iterdir()):
        if not folder.is_dir() or not re.match(r"^\d{14}_", folder.name):
            continue

        case_name = case_name_of(folder)
        for pattern, target in RULES:
            if re.search(pattern, case_name):
                moves.append((folder, target))
                break
        else:
            unmatched.append(folder)

    return moves, unmatched


def main():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="move the folders. Without it the plan is printed and nothing "
        "is touched",
    )
    args = parser.parse_args()

    moves, unmatched = plan(args.results)

    counts = {}
    for _, target in moves:
        counts[target] = counts.get(target, 0) + 1

    print(f"{len(moves)} folders to file, from {args.results}\n")
    for target, count in sorted(counts.items()):
        print(f"  {count:4d} -> {target}/")

    if unmatched:
        print(f"\n{len(unmatched)} folders match no rule and stay put:")
        for folder in unmatched[:10]:
            print(f"  {folder.name}")

    if not args.apply:
        print("\nNothing moved. Pass --apply to move them.")
        return

    for folder, target in moves:
        destination = args.results / target / folder.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(destination))

    print(f"\nMoved {len(moves)} folders.")
    print(
        "Run 'python run_benchmark.py collect' to rewrite the dataset, and "
        "the manifests of the stages you filed if you want them to point at "
        "the new paths."
    )


if __name__ == "__main__":
    main()
