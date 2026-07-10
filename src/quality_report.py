"""Quality summaries for ECG preprocessing and segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from preprocessing import ProcessedECG, preprocess_echonext_record
from segment_library import SEGMENT_TYPES, build_segment_library


def summarize_processed_record(processed: ProcessedECG) -> pd.DataFrame:
    """Return per-lead segmentation completeness metrics for one record."""
    rows: list[dict[str, object]] = []
    for lead in processed.segments:
        segments = processed.segments[lead]
        r_peak_count = len(segments.r_peaks)
        row: dict[str, object] = {
            "record_id": processed.record_id,
            "label": processed.label,
            "lead": lead,
            "r_peak_count": r_peak_count,
        }
        for segment_type in SEGMENT_TYPES:
            count = len(getattr(segments, segment_type))
            row[f"{segment_type}_count"] = count
            row[f"{segment_type}_coverage"] = (
                count / r_peak_count if r_peak_count else 0.0
            )

        row["complete_beat_count"] = _complete_beat_count(processed, lead)
        row["complete_beat_coverage"] = (
            row["complete_beat_count"] / r_peak_count if r_peak_count else 0.0
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_quality_report(
    records: Iterable[int | str | Path],
    data_dir: str | Path = "data",
    split: str = "val",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process records and return detail, per-lead, and per-segment summaries.

    detail has one row per record and lead. lead_summary aggregates coverage by
    lead. segment_summary aggregates coverage by segment type across all leads.
    """
    detail_frames = []
    for record in records:
        processed = preprocess_echonext_record(record, data_dir=data_dir, split=split)
        detail_frames.append(summarize_processed_record(processed))

    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    if detail.empty:
        return detail, pd.DataFrame(), pd.DataFrame()

    lead_summary = _lead_summary(detail)
    segment_summary = _segment_summary(detail)
    return detail, lead_summary, segment_summary


def save_quality_report(
    detail: pd.DataFrame,
    lead_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Write quality report CSVs into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "segmentation_quality_detail.csv", index=False)
    lead_summary.to_csv(output_dir / "segmentation_quality_by_lead.csv", index=False)
    segment_summary.to_csv(
        output_dir / "segmentation_quality_by_segment.csv",
        index=False,
    )


def _complete_beat_count(processed: ProcessedECG, lead: str) -> int:
    library = build_segment_library(
        [ProcessedECG(
            raw=processed.raw,
            cleaned=processed.cleaned,
            label=processed.label,
            segments={lead: processed.segments[lead]},
            fs=processed.fs,
            record_id=processed.record_id,
        )],
        drop_incomplete_beats=True,
    )
    if library.empty:
        return 0
    return library[["record_id", "lead", "beat_index"]].drop_duplicates().shape[0]


def _lead_summary(detail: pd.DataFrame) -> pd.DataFrame:
    coverage_columns = [
        f"{segment_type}_coverage" for segment_type in SEGMENT_TYPES
    ] + ["complete_beat_coverage"]
    count_columns = [
        "r_peak_count",
        *[f"{segment_type}_count" for segment_type in SEGMENT_TYPES],
        "complete_beat_count",
    ]

    summary = detail.groupby("lead", as_index=False).agg(
        record_count=("record_id", "count"),
        label_positive_rate=("label", "mean"),
        **{column: (column, "mean") for column in coverage_columns},
        **{column: (column, "sum") for column in count_columns},
    )
    return summary


def _segment_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_r_peaks = int(detail["r_peak_count"].sum())
    for segment_type in SEGMENT_TYPES:
        total_segments = int(detail[f"{segment_type}_count"].sum())
        rows.append(
            {
                "segment_type": segment_type,
                "total_segments": total_segments,
                "total_r_peaks": total_r_peaks,
                "coverage": total_segments / total_r_peaks if total_r_peaks else 0.0,
                "mean_lead_record_coverage": float(
                    detail[f"{segment_type}_coverage"].mean()
                ),
            }
        )

    complete_beats = int(detail["complete_beat_count"].sum())
    rows.append(
        {
            "segment_type": "complete_beat",
            "total_segments": complete_beats,
            "total_r_peaks": total_r_peaks,
            "coverage": complete_beats / total_r_peaks if total_r_peaks else 0.0,
            "mean_lead_record_coverage": float(detail["complete_beat_coverage"].mean()),
        }
    )
    return pd.DataFrame(rows)
