"""
Phase 1 ECG preprocessing for EchoNext records.

The main entry point is preprocess_echonext_record(), which loads one record,
cleans the 12-lead waveform, and returns per-lead segment boundaries suitable
for classifier input and segment-library construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from loader import LABEL_COLUMN


DEFAULT_FS = 250
DEFAULT_SAMPLES = 2500
LEAD_NAMES = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


@dataclass(frozen=True)
class ECGRecord:
    """One raw ECG and its label."""

    waveform: np.ndarray
    label: Any | None
    record_id: str | int | None = None
    fs: int = DEFAULT_FS


@dataclass(frozen=True)
class ECGSegments:
    """
    Segment boundaries for one lead.

    Each array has shape (n_beats, 2), storing inclusive start and exclusive end
    sample indices for that waveform interval.
    """

    r_peaks: np.ndarray
    p_wave: np.ndarray
    qrs_complex: np.ndarray
    st_segment: np.ndarray
    t_wave: np.ndarray


@dataclass(frozen=True)
class ProcessedECG:
    """Cleaned ECG with segment boundaries."""

    raw: np.ndarray
    cleaned: np.ndarray
    label: Any | None
    segments: dict[str, ECGSegments]
    fs: int
    record_id: str | int | None = None


def load_echonext_record(
    record: int | str | Path,
    data_dir: str | Path = "data",
    split: str = "val",
    label_column: str = LABEL_COLUMN,
    fs: int = DEFAULT_FS,
) -> ECGRecord:
    """
    Load one EchoNext record as (12, 2500).

    For this workspace's released EchoNext arrays, pass an integer row index.
    For future WFDB access, pass a record path/name; wfdb must be installed.
    """
    if isinstance(record, int):
        data_dir = Path(data_dir)
        waveforms = np.load(
            data_dir / f"EchoNext_{split}_waveforms.npy",
            mmap_mode="r",
        )
        metadata = _load_split_metadata(data_dir, split)
        if record < 0 or record >= waveforms.shape[0]:
            raise IndexError(f"record index {record} out of range for split '{split}'")
        if len(metadata) != waveforms.shape[0]:
            raise ValueError(
                f"Row count mismatch for split='{split}': "
                f"metadata={len(metadata)}, waveforms={waveforms.shape[0]}"
            )

        waveform = _as_leads_by_samples(waveforms[record])
        label = metadata[label_column].to_numpy()[record]
        return ECGRecord(waveform=waveform, label=label, record_id=record, fs=fs)

    waveform, label = _load_wfdb_record(Path(record), label_column=label_column)
    return ECGRecord(
        waveform=_as_leads_by_samples(waveform),
        label=label,
        record_id=str(record),
        fs=fs,
    )


def clean_waveform(waveform: np.ndarray, fs: int = DEFAULT_FS) -> np.ndarray:
    """
    Clean a 12-lead ECG while preserving shape.

    Uses NeuroKit2 when available. If not, falls back to a deterministic FFT
    bandpass that is sufficient for tests and early visual QA.
    """
    waveform = _as_leads_by_samples(waveform)
    nk = _optional_import("neurokit2")

    if nk is not None:
        cleaned = [
            nk.ecg_clean(lead.astype(float), sampling_rate=fs, method="neurokit")
            for lead in waveform
        ]
        return np.asarray(cleaned, dtype=float)

    return np.asarray([_fft_bandpass(lead, fs=fs) for lead in waveform], dtype=float)


def segment_waveform(
    cleaned: np.ndarray,
    fs: int = DEFAULT_FS,
    lead_names: tuple[str, ...] = LEAD_NAMES,
) -> dict[str, ECGSegments]:
    """Detect R peaks and ECG morphology intervals for each lead."""
    cleaned = _as_leads_by_samples(cleaned)
    if len(lead_names) != cleaned.shape[0]:
        lead_names = tuple(f"lead_{i}" for i in range(cleaned.shape[0]))

    return {
        lead_name: _segment_one_lead(cleaned[i], fs=fs)
        for i, lead_name in enumerate(lead_names)
    }


def preprocess_echonext_record(
    record: int | str | Path,
    data_dir: str | Path = "data",
    split: str = "val",
    label_column: str = LABEL_COLUMN,
    fs: int = DEFAULT_FS,
) -> ProcessedECG:
    """
    Load, clean, and segment one EchoNext ECG record.

    Returns
    -------
    ProcessedECG
        raw and cleaned arrays are shape (12, 2500). segments is keyed by lead
        name and contains P-wave, QRS, ST, and T-wave intervals per beat.
    """
    record_data = load_echonext_record(
        record=record,
        data_dir=data_dir,
        split=split,
        label_column=label_column,
        fs=fs,
    )
    cleaned = clean_waveform(record_data.waveform, fs=record_data.fs)
    segments = segment_waveform(cleaned, fs=record_data.fs)

    return ProcessedECG(
        raw=record_data.waveform,
        cleaned=cleaned,
        label=record_data.label,
        segments=segments,
        fs=record_data.fs,
        record_id=record_data.record_id,
    )


def plot_segment_overlay(
    processed: ProcessedECG,
    lead: str = "II",
    ax: Any | None = None,
) -> Any:
    """Plot one cleaned lead with P/QRS/ST/T boundaries overlaid."""
    import matplotlib.pyplot as plt

    if lead not in processed.segments:
        raise KeyError(f"unknown lead '{lead}'. Available: {list(processed.segments)}")

    lead_index = list(processed.segments).index(lead)
    signal = processed.cleaned[lead_index]
    time = np.arange(signal.size) / processed.fs

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    ax.plot(time, signal, color="black", linewidth=1.0)
    spans = {
        "p_wave": ("#7aa6c2", 0.25),
        "qrs_complex": ("#d95f02", 0.30),
        "st_segment": ("#66a61e", 0.22),
        "t_wave": ("#7570b3", 0.25),
    }
    lead_segments = processed.segments[lead]
    for name, (color, alpha) in spans.items():
        intervals = getattr(lead_segments, name)
        for start, end in intervals:
            ax.axvspan(start / processed.fs, end / processed.fs, color=color, alpha=alpha)

    ax.scatter(
        lead_segments.r_peaks / processed.fs,
        signal[lead_segments.r_peaks],
        color="#b2182b",
        s=18,
        zorder=3,
        label="R peaks",
    )
    ax.set_title(f"Lead {lead} segment overlay")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    return ax


def _segment_one_lead(signal: np.ndarray, fs: int) -> ECGSegments:
    nk = _optional_import("neurokit2")
    if nk is not None:
        try:
            _, info = nk.ecg_peaks(signal, sampling_rate=fs)
            r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
            if r_peaks.size:
                _, waves = nk.ecg_delineate(
                    signal,
                    r_peaks,
                    sampling_rate=fs,
                    method="dwt",
                    show=False,
                    show_type="all",
                )
                return _segments_from_neurokit_waves(waves, r_peaks, signal.size, fs)
        except Exception:
            pass

    r_peaks = _detect_r_peaks(signal, fs=fs)
    return _heuristic_segments_from_r_peaks(r_peaks, signal.size, fs=fs)


def _segments_from_neurokit_waves(
    waves: dict[str, Any],
    r_peaks: np.ndarray,
    n_samples: int,
    fs: int,
) -> ECGSegments:
    heuristic = _heuristic_segments_from_r_peaks(r_peaks, n_samples, fs)
    qrs = _intervals_from_arrays(
        waves.get("ECG_R_Onsets"),
        waves.get("ECG_R_Offsets"),
        fallback=heuristic.qrs_complex,
        n_samples=n_samples,
    )
    t_wave = _intervals_from_arrays(
        waves.get("ECG_T_Onsets"),
        waves.get("ECG_T_Offsets"),
        fallback=heuristic.t_wave,
        n_samples=n_samples,
    )
    p_wave = _intervals_from_arrays(
        waves.get("ECG_P_Onsets"),
        waves.get("ECG_P_Offsets"),
        fallback=heuristic.p_wave,
        n_samples=n_samples,
    )
    p_wave, qrs, t_wave = _enforce_segment_order(p_wave, qrs, t_wave, heuristic)
    return ECGSegments(
        r_peaks=r_peaks,
        p_wave=p_wave,
        qrs_complex=qrs,
        st_segment=_make_st_segments(qrs, t_wave),
        t_wave=t_wave,
    )


def _heuristic_segments_from_r_peaks(
    r_peaks: np.ndarray,
    n_samples: int,
    fs: int,
) -> ECGSegments:
    p_wave = _offset_intervals(r_peaks, -0.22, -0.08, fs, n_samples)
    qrs_complex = _offset_intervals(r_peaks, -0.045, 0.055, fs, n_samples)
    t_wave = _offset_intervals(r_peaks, 0.12, 0.40, fs, n_samples)
    return ECGSegments(
        r_peaks=r_peaks,
        p_wave=p_wave,
        qrs_complex=qrs_complex,
        st_segment=_make_st_segments(qrs_complex, t_wave),
        t_wave=t_wave,
    )


def _enforce_segment_order(
    p_wave: np.ndarray,
    qrs_complex: np.ndarray,
    t_wave: np.ndarray,
    fallback: ECGSegments,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(
        len(p_wave),
        len(qrs_complex),
        len(t_wave),
        len(fallback.p_wave),
        len(fallback.qrs_complex),
        len(fallback.t_wave),
    )
    p_wave = p_wave[:count].copy()
    qrs_complex = qrs_complex[:count].copy()
    t_wave = t_wave[:count].copy()

    for beat_index in range(count):
        if p_wave[beat_index, 1] > qrs_complex[beat_index, 0]:
            p_wave[beat_index] = fallback.p_wave[beat_index]
        if qrs_complex[beat_index, 1] > t_wave[beat_index, 0]:
            qrs_complex[beat_index] = fallback.qrs_complex[beat_index]
            t_wave[beat_index] = fallback.t_wave[beat_index]

    return p_wave, qrs_complex, t_wave


def _detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    centered = signal - np.median(signal)
    if abs(np.min(centered)) > abs(np.max(centered)):
        centered = -centered
    scale = np.median(np.abs(centered)) or np.std(centered) or 1.0
    threshold = np.median(centered) + 3.5 * scale
    refractory = max(1, int(0.35 * fs))

    candidates = np.flatnonzero(
        (centered[1:-1] > centered[:-2])
        & (centered[1:-1] >= centered[2:])
        & (centered[1:-1] > threshold)
    ) + 1

    peaks: list[int] = []
    for idx in candidates:
        if not peaks or idx - peaks[-1] >= refractory:
            peaks.append(int(idx))
        elif centered[idx] > centered[peaks[-1]]:
            peaks[-1] = int(idx)
    return np.asarray(peaks, dtype=int)


def _fft_bandpass(
    signal: np.ndarray,
    fs: int,
    lowcut: float = 0.5,
    highcut: float = 40.0,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    spectrum = np.fft.rfft(signal - np.mean(signal))
    mask = (freqs >= lowcut) & (freqs <= highcut)
    filtered = np.fft.irfft(spectrum * mask, n=signal.size)
    return filtered.astype(float)


def _make_st_segments(qrs_complex: np.ndarray, t_wave: np.ndarray) -> np.ndarray:
    count = min(len(qrs_complex), len(t_wave))
    if count == 0:
        return np.empty((0, 2), dtype=int)

    intervals = np.column_stack((qrs_complex[:count, 1], t_wave[:count, 0]))
    return intervals[intervals[:, 1] > intervals[:, 0]].astype(int)


def _offset_intervals(
    r_peaks: np.ndarray,
    start_seconds: float,
    end_seconds: float,
    fs: int,
    n_samples: int,
) -> np.ndarray:
    if r_peaks.size == 0:
        return np.empty((0, 2), dtype=int)

    starts = r_peaks + int(round(start_seconds * fs))
    ends = r_peaks + int(round(end_seconds * fs))
    intervals = np.column_stack((starts, ends))
    intervals[:, 0] = np.clip(intervals[:, 0], 0, n_samples)
    intervals[:, 1] = np.clip(intervals[:, 1], 0, n_samples)
    return intervals[intervals[:, 1] > intervals[:, 0]].astype(int)


def _intervals_from_arrays(
    starts: Any,
    ends: Any,
    fallback: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    starts = np.asarray([] if starts is None else starts, dtype=float)
    ends = np.asarray([] if ends is None else ends, dtype=float)

    if starts.shape == ends.shape and len(starts) == len(fallback):
        intervals = fallback.copy()
        valid = np.isfinite(starts) & np.isfinite(ends)
        intervals[valid, 0] = starts[valid].astype(int)
        intervals[valid, 1] = ends[valid].astype(int)
        intervals[:, 0] = np.clip(intervals[:, 0], 0, n_samples)
        intervals[:, 1] = np.clip(intervals[:, 1], 0, n_samples)
        return intervals[intervals[:, 1] > intervals[:, 0]]

    valid = np.isfinite(starts) & np.isfinite(ends)
    intervals = np.column_stack((starts[valid], ends[valid])).astype(int)
    intervals[:, 0] = np.clip(intervals[:, 0], 0, n_samples)
    intervals[:, 1] = np.clip(intervals[:, 1], 0, n_samples)
    intervals = intervals[intervals[:, 1] > intervals[:, 0]]
    return intervals if intervals.size else fallback


def _as_leads_by_samples(waveform: np.ndarray) -> np.ndarray:
    arr = np.asarray(waveform, dtype=float)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(axis=0)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D ECG array, got shape {arr.shape}")
    if arr.shape == (DEFAULT_SAMPLES, 12):
        arr = arr.T
    if arr.shape[0] != 12:
        raise ValueError(f"expected 12 leads, got shape {arr.shape}")
    if arr.shape[1] != DEFAULT_SAMPLES:
        raise ValueError(f"expected 2500 samples, got shape {arr.shape}")
    return arr


def _load_wfdb_record(path: Path, label_column: str) -> tuple[np.ndarray, Any | None]:
    wfdb = _optional_import("wfdb")
    if wfdb is None:
        raise ImportError(
            "wfdb is required to load WFDB records. Install it with "
            "`pip install wfdb`, or pass an integer index for the local .npy split."
        )

    record = wfdb.rdrecord(str(path))
    label = None
    comments = getattr(record, "comments", []) or []
    for comment in comments:
        if comment.startswith(f"{label_column}:"):
            label = comment.split(":", 1)[1].strip()
            break
    return np.asarray(record.p_signal), label


def _load_split_metadata(data_dir: Path, split: str) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to read EchoNext metadata CSV files") from exc

    metadata = pd.read_csv(data_dir / "echonext_metadata_100k.csv")
    return metadata[metadata["split"] == split].reset_index(drop=True)


def _optional_import(module_name: str) -> Any | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return module
