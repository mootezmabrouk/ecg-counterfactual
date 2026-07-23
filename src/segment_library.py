"""Build tabular ECG segment libraries from preprocessed EchoNext records."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.preprocessing import LEAD_NAMES, ProcessedECG, preprocess_echonext_record


SEGMENT_TYPES = ("p_wave", "qrs_complex", "st_segment", "t_wave")


def build_segment_library(
    processed_records: Iterable[ProcessedECG],
    include_waveforms: bool = False,
    drop_incomplete_beats: bool = False,
) -> pd.DataFrame:
    """
    Convert processed ECG records into one row per ECG segment.

    Parameters
    ----------
    processed_records
        Iterable of outputs from preprocess_echonext_record().
    include_waveforms
        If True, add a ``waveform`` object column containing the cleaned segment
        slice. Leave False for CSV-friendly metadata.
    drop_incomplete_beats
        If True, keep only beat groups where P/QRS/ST/T are all present for the
        same record, lead, and nearest R peak.
    """
    rows: list[dict[str, object]] = []
    for processed in processed_records:
        rows.extend(_rows_for_record(processed, include_waveforms=include_waveforms))

    columns = [
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
    ]
    if include_waveforms:
        columns.append("waveform")

    library = pd.DataFrame(rows, columns=columns)
    if library.empty or not drop_incomplete_beats:
        return library

    beat_keys = ["record_id", "lead", "beat_index"]
    complete = (
        library.groupby(beat_keys)["segment_type"]
        .nunique()
        .rename("segment_type_count")
        .reset_index()
    )
    complete = complete[complete["segment_type_count"] == len(SEGMENT_TYPES)]
    return library.merge(complete[beat_keys], on=beat_keys, how="inner")


def build_segment_library_from_records(
    records: Iterable[int | str | Path],
    data_dir: str | Path = "data",
    split: str = "val",
    include_waveforms: bool = False,
    drop_incomplete_beats: bool = False,
) -> pd.DataFrame:
    """Preprocess records and return their segment library table."""
    processed = (
        preprocess_echonext_record(record, data_dir=data_dir, split=split)
        for record in records
    )
    return build_segment_library(
        processed,
        include_waveforms=include_waveforms,
        drop_incomplete_beats=drop_incomplete_beats,
    )


def save_segment_library(
    library: pd.DataFrame,
    output_csv: str | Path,
    waveform_npz: str | Path | None = None,
) -> None:
    """
    Save a segment library table.

    If the table contains waveform arrays, pass waveform_npz to store those
    arrays separately and write only waveform keys into the CSV.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if "waveform" not in library.columns:
        library.to_csv(output_csv, index=False)
        return

    if waveform_npz is None:
        raise ValueError("waveform_npz is required when library has a waveform column")

    waveform_npz = Path(waveform_npz)
    waveform_npz.parent.mkdir(parents=True, exist_ok=True)

    csv_library = library.copy()
    arrays: dict[str, np.ndarray] = {}
    keys: list[str] = []
    for row_index, waveform in enumerate(csv_library["waveform"]):
        key = f"segment_{row_index:08d}"
        arrays[key] = np.asarray(waveform, dtype=float)
        keys.append(key)

    csv_library["waveform_key"] = keys
    csv_library = csv_library.drop(columns=["waveform"])
    csv_library.to_csv(output_csv, index=False)
    np.savez_compressed(waveform_npz, **arrays)


def _rows_for_record(
    processed: ProcessedECG,
    include_waveforms: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lead_index, lead in enumerate(processed.segments):
        signal = processed.cleaned[lead_index]
        lead_segments = processed.segments[lead]
        for segment_type in SEGMENT_TYPES:
            intervals = getattr(lead_segments, segment_type)
            for start, end in intervals:
                beat_index = _nearest_r_peak_index(
                    start=start,
                    end=end,
                    segment_type=segment_type,
                    r_peaks=lead_segments.r_peaks,
                )
                r_peak_sample = (
                    int(lead_segments.r_peaks[beat_index])
                    if beat_index is not None
                    else None
                )
                row: dict[str, object] = {
                    "record_id": processed.record_id,
                    "label": processed.label,
                    "lead": lead,
                    "lead_index": lead_index,
                    "beat_index": beat_index,
                    "r_peak_sample": r_peak_sample,
                    "segment_type": segment_type,
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "duration_samples": int(end - start),
                    "start_time_s": float(start / processed.fs),
                    "end_time_s": float(end / processed.fs),
                    "fs": processed.fs,
                }
                if include_waveforms:
                    row["waveform"] = signal[int(start) : int(end)].copy()
                rows.append(row)
    return rows


def _nearest_r_peak_index(
    start: int,
    end: int,
    segment_type: str,
    r_peaks: np.ndarray,
) -> int | None:
    if len(r_peaks) == 0:
        return None

    if segment_type == "p_wave":
        candidates = np.flatnonzero(r_peaks >= end)
        if len(candidates):
            return int(candidates[0])
    elif segment_type in {"st_segment", "t_wave"}:
        candidates = np.flatnonzero(r_peaks <= start)
        if len(candidates):
            return int(candidates[-1])

    center = (start + end) / 2
    return int(np.argmin(np.abs(r_peaks - center)))
