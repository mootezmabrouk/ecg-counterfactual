"""Build a segment library CSV from EchoNext records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segment_library import build_segment_library_from_records, save_segment_library


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="val")
    parser.add_argument("--records", nargs="+", type=int, required=True)
    parser.add_argument("--output-csv", default="outputs/segment_library.csv")
    parser.add_argument("--waveform-npz", default=None)
    parser.add_argument("--include-waveforms", action="store_true")
    parser.add_argument("--drop-incomplete-beats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.include_waveforms and args.waveform_npz is None:
        raise SystemExit("--waveform-npz is required with --include-waveforms")

    library = build_segment_library_from_records(
        args.records,
        data_dir=args.data_dir,
        split=args.split,
        include_waveforms=args.include_waveforms,
        drop_incomplete_beats=args.drop_incomplete_beats,
    )
    save_segment_library(
        library,
        output_csv=args.output_csv,
        waveform_npz=args.waveform_npz,
    )
    print(f"saved {len(library)} rows to {args.output_csv}")
    if args.include_waveforms:
        print(f"saved waveform slices to {args.waveform_npz}")


if __name__ == "__main__":
    main()
