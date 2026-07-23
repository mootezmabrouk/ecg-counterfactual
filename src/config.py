"""Configuration for Phase 3: Saliency Prior + Normal Segment Library."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class Phase3Config:
    """Configuration for Phase 3 pipeline."""

    # Data paths
    data_dir: Path = Path("data")
    metadata_path: Path = Path("data/echonext_metadata_100k.csv")
    val_waveforms_path: Path = Path("data/EchoNext_val_waveforms.npy")
    val_tabular_path: Path = Path("data/EchoNext_val_tabular_features.npy")

    # Output paths
    output_dir: Path = Path("outputs")
    segment_library_csv: Path = Path("outputs/segment_library.csv")
    segment_library_npz: Path = Path("outputs/segment_library_waveforms.npz")
    saliency_cache_path: Path = Path("outputs/saliency_scores.pkl")

    # ECG parameters (EchoNext: 250 Hz, 10 seconds)
    fs: float = 250.0
    n_samples: int = 2500
    n_leads: int = 12
    lead_names: Tuple[str, ...] = (
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    )

    # IG parameters
    ig_n_steps: int = 50
    ig_target_class: int = 11  # SHD is the 12th class (index 11)

    # Saliency aggregation: "mean", "max", or "l1"
    saliency_aggregation: str = "mean"

    # How many abnormal ECGs to compute saliency for (0 = all)
    n_saliency_samples: int = 100

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)