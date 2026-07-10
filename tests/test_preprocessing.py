import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import clean_waveform, segment_waveform


def synthetic_ecg(fs=250, seconds=10):
    samples = fs * seconds
    time = np.arange(samples) / fs
    lead = np.zeros(samples, dtype=float)

    for beat_time in np.arange(0.8, seconds, 1.0):
        lead += 0.12 * np.exp(-0.5 * ((time - (beat_time - 0.16)) / 0.025) ** 2)
        lead += 1.20 * np.exp(-0.5 * ((time - beat_time) / 0.012) ** 2)
        lead += 0.35 * np.exp(-0.5 * ((time - (beat_time + 0.28)) / 0.060) ** 2)

    baseline = 0.05 * np.sin(2 * np.pi * 0.25 * time)
    noise = 0.01 * np.sin(2 * np.pi * 60.0 * time)
    waveform = lead + baseline + noise
    return np.vstack([waveform * (1.0 + i * 0.02) for i in range(12)])


class PreprocessingTests(unittest.TestCase):
    def test_clean_waveform_preserves_expected_shape(self):
        waveform = synthetic_ecg()
        cleaned = clean_waveform(waveform)
        self.assertEqual(cleaned.shape, (12, 2500))
        self.assertTrue(np.isfinite(cleaned).all())

    def test_segment_waveform_returns_ordered_intervals_per_lead(self):
        waveform = synthetic_ecg()
        cleaned = clean_waveform(waveform)
        segments = segment_waveform(cleaned)

        self.assertEqual(set(segments), {
            "I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6",
        })
        lead_ii = segments["II"]
        self.assertGreaterEqual(len(lead_ii.r_peaks), 8)
        self.assertEqual(lead_ii.p_wave.shape[1], 2)
        self.assertEqual(lead_ii.qrs_complex.shape[1], 2)
        self.assertEqual(lead_ii.st_segment.shape[1], 2)
        self.assertEqual(lead_ii.t_wave.shape[1], 2)

        count = min(
            len(lead_ii.p_wave),
            len(lead_ii.qrs_complex),
            len(lead_ii.st_segment),
            len(lead_ii.t_wave),
        )
        self.assertTrue(
            np.all(lead_ii.p_wave[:count, 1] <= lead_ii.qrs_complex[:count, 0])
        )
        self.assertTrue(
            np.all(lead_ii.qrs_complex[:count, 1] <= lead_ii.st_segment[:count, 0])
        )
        self.assertTrue(
            np.all(lead_ii.st_segment[:count, 1] <= lead_ii.t_wave[:count, 0])
        )


if __name__ == "__main__":
    unittest.main()
