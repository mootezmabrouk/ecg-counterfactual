"""Generate ECG segment overlay plots for visual sanity checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import plot_segment_overlay, preprocess_echonext_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="val")
    parser.add_argument("--records", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--leads", nargs="+", default=["II"])
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for record in args.records:
        processed = preprocess_echonext_record(
            record,
            data_dir=args.data_dir,
            split=args.split,
        )
        for lead in args.leads:
            fig, ax = plt.subplots(figsize=(13, 4))
            plot_segment_overlay(processed, lead=lead, ax=ax)
            fig.tight_layout()
            path = output_dir / f"record_{record}_lead_{lead}_segments.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            print(path)


if __name__ == "__main__":
    main()
