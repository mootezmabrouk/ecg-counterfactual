import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import ProcessedECG, clean_waveform, segment_waveform
from quality_report import summarize_processed_record
from test_preprocessing import synthetic_ecg


class QualityReportTests(unittest.TestCase):
    def test_summarize_processed_record_returns_per_lead_metrics(self):
        waveform = synthetic_ecg()
        cleaned = clean_waveform(waveform)
        processed = ProcessedECG(
            raw=waveform,
            cleaned=cleaned,
            label=1,
            segments=segment_waveform(cleaned),
            fs=250,
            record_id=7,
        )

        summary = summarize_processed_record(processed)

        self.assertEqual(len(summary), 12)
        self.assertIn("complete_beat_coverage", summary.columns)
        self.assertTrue((summary["r_peak_count"] > 0).all())
        self.assertTrue((summary["complete_beat_coverage"] >= 0).all())
        self.assertTrue((summary["complete_beat_coverage"] <= 1).all())


if __name__ == "__main__":
    unittest.main()
