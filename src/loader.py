"""
loader.py — Step 1 of Phase 1: Load EchoNext waveforms, tabular features, and labels.

EchoNext data format (per PhysioNet documentation):
- Waveforms:  (N, 1, 2500, 12)  — already median-filtered, clipped, normalized
- Tabular:    (N, 7)            — [sex, ventricular_rate, atrial_rate, pr_interval,
                                    qrs_duration, qt_corrected, age_at_ecg], standardized
- Metadata:   echonext_metadata_100k.csv — one row per patient/ECG, includes a
              'split' column (train/val/test) and all disease flag labels.

IMPORTANT ASSUMPTION TO VERIFY ON REAL DATA:
We assume that filtering the metadata CSV by split == split_name, in the order
rows appear in the file, gives EXACTLY the same row order as the corresponding
.npy array. This is the standard PhysioNet convention but must be sanity-checked
(see verify_alignment()) before trusting anything downstream.
"""

import numpy as np
import pandas as pd
from pathlib import Path

TABULAR_COLUMNS = [
    "sex", "ventricular_rate", "atrial_rate",
    "pr_interval", "qrs_duration", "qt_corrected", "age_at_ecg",
]

LABEL_COLUMN = "shd_moderate_or_greater_flag"


def load_split(data_dir: str, split: str):
    """
    Load waveforms, tabular features, and matching metadata for one split.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing the EchoNext .npy and .csv files.
    split : str
        One of "train", "val", "test".

    Returns
    -------
    waveforms : np.ndarray, shape (N, 2500, 12)
        Squeezed to drop the redundant channel dim.
    tabular : np.ndarray, shape (N, 7)
    metadata : pd.DataFrame, shape (N, ...)
        Filtered to this split, index reset to match array row order (0..N-1).
    """
    data_dir = Path(data_dir)

    waveforms = np.load(data_dir / f"EchoNext_{split}_waveforms.npy")
    tabular = np.load(data_dir / f"EchoNext_{split}_tabular_features.npy")

    # drop the redundant channel axis: (N, 1, 2500, 12) -> (N, 2500, 12)
    if waveforms.ndim == 4 and waveforms.shape[1] == 1:
        waveforms = waveforms.squeeze(axis=1)

    metadata = pd.read_csv(data_dir / "echonext_metadata_100k.csv")
    metadata = metadata[metadata["split"] == split].reset_index(drop=True)

    assert len(metadata) == waveforms.shape[0] == tabular.shape[0], (
        f"Row count mismatch for split='{split}': "
        f"metadata={len(metadata)}, waveforms={waveforms.shape[0]}, tabular={tabular.shape[0]}. "
        f"This means the CSV filter order does NOT match the .npy row order — "
        f"do not proceed until this is fixed."
    )

    return waveforms, tabular, metadata


def verify_alignment(tabular: np.ndarray, metadata: pd.DataFrame, age_col_index: int = 6):
    """
    Sanity check: age_at_ecg in the tabular array (standardized) should be
    perfectly correlated with age_at_ecg in the metadata (raw), IF row order
    truly matches. Correlation close to 1.0 (or -1.0) confirms alignment.

    Note: values won't be numerically equal (tabular is z-scored), so we check
    correlation, not equality.
    """
    tabular_age = tabular[:, age_col_index]
    meta_age = metadata["age_at_ecg"].to_numpy()

    corr = np.corrcoef(tabular_age, meta_age)[0, 1]
    print(f"[verify_alignment] correlation between tabular age and metadata age: {corr:.4f}")
    if abs(corr) < 0.9:
        print("  WARNING: correlation is low — row order may NOT match. Investigate before proceeding.")
    else:
        print("  OK: strong correlation, row order appears aligned.")
    return corr


def get_labels(metadata: pd.DataFrame, label_column: str = LABEL_COLUMN) -> np.ndarray:
    """Extract the binary label array for classifier training."""
    return metadata[label_column].to_numpy()


if __name__ == "__main__":
    DATA_DIR = "data"
    SPLIT = "val"

    waveforms, tabular, metadata = load_split(DATA_DIR, SPLIT)

    print(f"waveforms shape : {waveforms.shape}")
    print(f"tabular shape   : {tabular.shape}")
    print(f"metadata shape  : {metadata.shape}")
    print(f"metadata columns: {list(metadata.columns)[:10]} ...")

    verify_alignment(tabular, metadata)

    labels = get_labels(metadata)
    print(f"label distribution ({LABEL_COLUMN}): "
          f"{np.bincount(labels.astype(int))} (0=normal, 1=SHD)")