#!/usr/bin/env python3
"""Generate an HTML summary from tools/run_tests.py summary.json."""

import argparse
from html import escape
from pathlib import Path

try:
    from .report import HEADERS, rows_from_summary
except ImportError:
    from report import HEADERS, rows_from_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="path to summary.json")
    parser.add_argument("--output", type=Path, help="default: summary.html beside input")
    args = parser.parse_args(argv)
    output = args.output or args.summary.with_suffix(".html")
    rows = rows_from_summary(args.summary)
    header = "".join(f"<th>{escape(name)}</th>" for name in HEADERS)
    body = "\n".join("<tr>" + "".join(
        f"<td>{escape(str(row.get(name, '') if row.get(name) is not None else ''))}</td>"
        for name in HEADERS
    ) + "</tr>" for row in rows)
    output.write_text(
        "<!doctype html>\n<html lang=\"en\"><meta charset=\"utf-8\">"
        "<title>FlagAudio test summary</title>"
        "<style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse}"
        "th,td{border:1px solid #bbb;padding:.4rem;text-align:left}"
        "th{background:#eee}tr:nth-child(even){background:#f7f7f7}</style>"
        "<h1>FlagAudio test summary</h1><table><thead><tr>" + header +
        "</tr></thead><tbody>" + body + "</tbody></table></html>\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
