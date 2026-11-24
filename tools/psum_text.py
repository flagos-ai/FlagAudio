#!/usr/bin/env python3
"""Generate a CSV summary from tools/run_tests.py summary.json."""

import argparse
import csv
from pathlib import Path

try:
    from .report import HEADERS, rows_from_summary
except ImportError:
    from report import HEADERS, rows_from_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="path to summary.json")
    parser.add_argument("--output", type=Path, help="default: summary.csv beside input")
    args = parser.parse_args(argv)
    output = args.output or args.summary.with_suffix(".csv")
    rows = rows_from_summary(args.summary)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
