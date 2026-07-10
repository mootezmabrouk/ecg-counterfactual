"""Run batch ECG segmentation quality checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quality_report import build_quality_report, save_quality_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="val")
    parser.add_argument("--records", nargs="+", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/quality_report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = (
        args.records
        if args.records is not None
        else list(range(args.start, args.start + args.limit))
    )
    detail, lead_summary, segment_summary = build_quality_report(
        records,
        data_dir=args.data_dir,
        split=args.split,
    )
    save_quality_report(
        detail,
        lead_summary,
        segment_summary,
        output_dir=args.output_dir,
    )

    print(f"processed {len(records)} records")
    print(f"wrote detail rows: {len(detail)}")
    print(segment_summary.to_string(index=False))


if __name__ == "__main__":
    main()
