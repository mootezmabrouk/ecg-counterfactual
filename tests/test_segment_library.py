import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import ProcessedECG, clean_waveform, segment_waveform
from segment_library import (
    SEGMENT_TYPES,
    build_segment_library,
    save_segment_library,
)
from test_preprocessing import synthetic_ecg


class SegmentLibraryTests(unittest.TestCase):
    def make_processed(self):
        waveform = synthetic_ecg()
        cleaned = clean_waveform(waveform)
        return ProcessedECG(
            raw=waveform,
            cleaned=cleaned,
            label=1,
            segments=segment_waveform(cleaned),
            fs=250,
            record_id=42,
        )

    def test_build_segment_library_returns_expected_columns(self):
        library = build_segment_library([self.make_processed()])

        self.assertFalse(library.empty)
        self.assertEqual(
            list(library.columns),
            [
                "record_id",
                "label",
                "lead",
                "lead_index",
                "beat_index",
                "r_peak_sample",
                "segment_type",
                "start_sample",
                "end_sample",
                "duration_samples",
                "start_time_s",
                "end_time_s",
                "fs",
            ],
        )
        self.assertTrue(set(library["segment_type"]).issubset(set(SEGMENT_TYPES)))
        self.assertIn("qrs_complex", set(library["segment_type"]))
        self.assertTrue((library["end_sample"] > library["start_sample"]).all())
        self.assertTrue((library["duration_samples"] > 0).all())

    def test_drop_incomplete_beats_keeps_full_segment_groups(self):
        library = build_segment_library(
            [self.make_processed()],
            drop_incomplete_beats=True,
        )

        counts = library.groupby(["record_id", "lead", "beat_index"])[
            "segment_type"
        ].nunique()
        self.assertTrue((counts == len(SEGMENT_TYPES)).all())

    def test_save_segment_library_splits_waveforms_from_csv(self):
        library = build_segment_library(
            [self.make_processed()],
            include_waveforms=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_csv = Path(tmp) / "segments.csv"
            waveform_npz = Path(tmp) / "segments.npz"
            save_segment_library(library, output_csv, waveform_npz)

            self.assertTrue(output_csv.exists())
            self.assertTrue(waveform_npz.exists())

            with np.load(waveform_npz) as loaded:
                self.assertGreater(len(loaded.files), 0)


if __name__ == "__main__":
    unittest.main()
