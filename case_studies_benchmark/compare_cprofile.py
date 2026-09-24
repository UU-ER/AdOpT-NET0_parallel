"""
Compares two cProfile dumps of the same phase, function by function.

Meant for the reading of the input: one dump of a run alone on the machine and
one of the same run inside a full pack of workers. The table lists the
functions whose own time grew the most, with the time in both runs and the
ratio. A few functions carrying all of the growth point at one cause (a
library spawning threads, a file read from the share), growth spread evenly
over everything points at the cores being shared.

Usage:
    python compare_cprofile.py solo/cprofile_read_data.prof packed/cprofile_read_data.prof
    python compare_cprofile.py solo.prof packed.prof --top 40 --sort cumulative
"""

import argparse
import pstats


def load(path: str, key: str):
    """
    Reads a dump into {function: seconds}

    :param str path: cProfile dump
    :param str key: "tottime" for own time, "cumulative" for time with callees
    :return: dict keyed by "file:line(function)"
    """
    stats = pstats.Stats(path).stats
    index = 2 if key == "tottime" else 3
    return {
        f"{file}:{line}({function})": values[index]
        for (file, line, function), values in stats.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("solo", help="dump of the run alone")
    parser.add_argument("packed", help="dump of the run under load")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument(
        "--sort", choices=["tottime", "cumulative"], default="tottime"
    )
    args = parser.parse_args()

    solo = load(args.solo, args.sort)
    packed = load(args.packed, args.sort)

    total_solo = sum(load(args.solo, "tottime").values())
    total_packed = sum(load(args.packed, "tottime").values())
    print(
        f"total own time: solo {total_solo:.1f} s, packed {total_packed:.1f} s, "
        f"x{total_packed / max(total_solo, 1e-9):.2f}\n"
    )

    rows = []
    for function in set(solo) | set(packed):
        a, b = solo.get(function, 0.0), packed.get(function, 0.0)
        rows.append((b - a, a, b, function))
    rows.sort(reverse=True)

    print(f"{'extra s':>9} {'solo s':>9} {'packed s':>9} {'ratio':>7}  function")
    for extra, a, b, function in rows[: args.top]:
        ratio = f"x{b / a:.1f}" if a > 0 else "new"
        # The path up to site-packages or adopt_net0 is noise in the table
        for marker in ("site-packages", "adopt_net0"):
            if marker in function:
                function = function[function.index(marker) :]
                break
        print(f"{extra:9.2f} {a:9.2f} {b:9.2f} {ratio:>7}  {function}")


if __name__ == "__main__":
    main()
