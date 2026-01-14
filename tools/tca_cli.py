import argparse
import os
import sys
from typing import Optional

import pandas as pd

from tca_rollup import load_tca_jsonl, rollup_tca


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Roll up TCA JSONL logs to CSV summary")
    p.add_argument(
        "--input",
        "-i",
        default=os.path.join("logs", "tca.jsonl"),
        help="Path to input TCA JSONL file (default: logs/tca.jsonl)",
    )
    p.add_argument(
        "--output",
        "-o",
        default=os.path.join("logs", "tca_rollup.csv"),
        help="Path to output CSV file (use '-' for stdout; default: logs/tca_rollup.csv)",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    df = load_tca_jsonl(args.input)
    out = rollup_tca(df)

    if args.output == "-":
        # Write to stdout
        out.to_csv(sys.stdout, index=False)
        return 0

    _ensure_parent_dir(args.output)
    out.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
