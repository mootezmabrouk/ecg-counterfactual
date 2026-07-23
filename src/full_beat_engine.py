"""Counterfactual Search Engine — Full-Beat Substitution Version.

Instead of swapping individual P/QRS/ST/T segments, we swap entire beats
(P-onset to T-offset). This is more impactful on global-pooling models.

Requires: existing segment library with p_wave, qrs_complex, st_segment, t_wave.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Phase3Config


@dataclass
class FullBeatConfig:
    """Config for full-beat counterfactual search."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    segment_library_csv: Path = Path("outputs/segment_library.csv")
    segment_library_npz: Path = Path("outputs/segment_library_waveforms.npz")
    saliency_cache_path: Path = Path("outputs/saliency_scores.pkl")

    max_beats_to_modify: int = 3
    max_candidates_per_beat: int = 20
    shd_threshold: float = 0.5

    # Beat composition: which segment types make a full beat
    beat_segments: Tuple[str, ...] = ("p_wave", "qrs_complex", "st_segment", "t_wave")

    results_path: Path = Path("outputs/counterfactual_fullbeat_results.pkl")

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


class FullBeatExtractor:
    """Extract full beats (P-onset to T-offset) from segment library."""

    def __init__(self, library_csv: Path, library_npz: Path):
        self.library_df = pd.read_csv(library_csv)

        npz = np.load(library_npz)
        self.waveforms = {k: npz[k] for k in npz.files}

    def extract_full_beats(self) -> pd.DataFrame:
        """
        Group segments by (record_id, lead, beat_index) and concatenate
        P + QRS + ST + T into full beats.

        Returns DataFrame with columns:
            record_id, lead, lead_index, beat_index, start_sample, end_sample,
            duration_samples, waveform_key
        """
        print("Extracting full beats from segment library...")

        # Group by record, lead, beat
        groups = self.library_df.groupby(["record_id", "lead", "beat_index"])

        beat_rows = []
        for (record_id, lead, beat_idx), group in groups:
            # Check if all 4 segment types are present
            seg_types = set(group["segment_type"].values)
            if seg_types < set(FullBeatConfig().beat_segments):
                continue  # Skip incomplete beats

            # Get segments in order: P, QRS, ST, T
            ordered = []
            for seg_type in FullBeatConfig().beat_segments:
                seg = group[group["segment_type"] == seg_type]
                if len(seg) == 0:
                    break
                ordered.append(seg.iloc[0])

            if len(ordered) != 4:
                continue

            # Concatenate waveforms
            wfs = []
            for seg in ordered:
                key = seg.get("waveform_key", f"segment_{seg.name:08d}")
                if key in self.waveforms:
                    wfs.append(self.waveforms[key])

            if len(wfs) != 4:
                continue

            full_beat = np.concatenate(wfs)

            # Save to a new key
            beat_key = f"beat_{record_id}_{lead}_{beat_idx}"

            beat_rows.append({
                "record_id": record_id,
                "lead": lead,
                "lead_index": ordered[0]["lead_index"],
                "beat_index": beat_idx,
                "start_sample": int(ordered[0]["start_sample"]),
                "end_sample": int(ordered[-1]["end_sample"]),
                "duration_samples": len(full_beat),
                "waveform_key": beat_key,
                "waveform": full_beat,  # Store temporarily
            })

        beats_df = pd.DataFrame(beat_rows)
        print(f"  Extracted {len(beats_df)} full beats")

        # Save waveforms to NPZ
        beat_waveforms = {row["waveform_key"]: row["waveform"] for _, row in beats_df.iterrows()}
        npz_path = self.output_dir / "full_beat_waveforms.npz"
        np.savez_compressed(npz_path, **beat_waveforms)
        print(f"  Saved beat waveforms to {npz_path}")

        # Drop waveform column, keep key
        beats_df = beats_df.drop(columns=["waveform"])

        csv_path = self.output_dir / "full_beat_library.csv"
        beats_df.to_csv(csv_path, index=False)
        print(f"  Saved beat metadata to {csv_path}")

        return beats_df


class FullBeatLibraryLoader:
    """Load full-beat library with precomputed features."""

    def __init__(self, config: FullBeatConfig):
        self.config = config
        self.beats_df: Optional[pd.DataFrame] = None
        self.waveforms: Optional[Dict[str, np.ndarray]] = None
        self.feature_index: Optional[pd.DataFrame] = None

    def load(self) -> "FullBeatLibraryLoader":
        print("Loading full-beat library...")

        csv_path = self.config.output_dir / "full_beat_library.csv"
        npz_path = self.config.output_dir / "full_beat_waveforms.npz"

        if not csv_path.exists():
            print("  Full-beat library not found. Building from segment library...")
            extractor = FullBeatExtractor(
                self.config.segment_library_csv,
                self.config.segment_library_npz,
            )
            extractor.output_dir = self.config.output_dir
            self.beats_df = extractor.extract_full_beats()
        else:
            self.beats_df = pd.read_csv(csv_path)
            npz = np.load(npz_path)
            self.waveforms = {k: npz[k] for k in npz.files}
            print(f"  Loaded {len(self.beats_df)} beats, {len(self.waveforms)} waveforms")

        # Precompute features
        print("Precomputing beat features...")
        features = []
        for _, row in self.beats_df.iterrows():
            key = row["waveform_key"]
            if key in self.waveforms:
                wf = self.waveforms[key]
                feats = {
                    "waveform_key": key,
                    "lead": row["lead"],
                    "lead_index": row["lead_index"],
                    "beat_index": row["beat_index"],
                    "mean": float(np.mean(wf)),
                    "std": float(np.std(wf)),
                    "length": float(len(wf)),
                    "min": float(np.min(wf)),
                    "max": float(np.max(wf)),
                }
                features.append(feats)

        self.feature_index = pd.DataFrame(features)
        print(f"  Feature index: {len(self.feature_index)} entries")

        return self

    def find_substitutes(
        self,
        target_lead: str,
        target_waveform: np.ndarray,
        n_candidates: int = 20,
    ) -> List[Tuple[str, float, np.ndarray]]:
        """Find normal full-beat substitutes for a lead."""
        mask = self.feature_index["lead"] == target_lead
        candidates = self.feature_index[mask].copy()

        if len(candidates) == 0:
            return []

        # Compute distance
        target_feats = {
            "mean": float(np.mean(target_waveform)),
            "std": float(np.std(target_waveform)),
            "length": float(len(target_waveform)),
            "min": float(np.min(target_waveform)),
            "max": float(np.max(target_waveform)),
        }

        distances = []
        for _, row in candidates.iterrows():
            cand_feats = {
                "mean": row["mean"], "std": row["std"], "length": row["length"],
                "min": row["min"], "max": row["max"],
            }
            dist = 0
            for key in ["mean", "std", "length", "min", "max"]:
                denom = abs(target_feats[key]) + abs(cand_feats[key]) + 1e-8
                dist += ((target_feats[key] - cand_feats[key]) / denom) ** 2
            distances.append(np.sqrt(dist))

        candidates["distance"] = distances
        # Use ANTI-similarity: most different normal beats
        candidates = candidates.sort_values("distance", ascending=False).head(n_candidates)

        results = []
        for _, row in candidates.iterrows():
            key = row["waveform_key"]
            wf = self.waveforms.get(key, np.array([]))
            results.append((key, row["distance"], wf))

        return results


class FullBeatSubstitutor:
    """Substitute full beats into ECG waveforms."""

    def substitute_beat(
        self,
        ecg_waveform: np.ndarray,
        lead_index: int,
        start_sample: int,
        end_sample: int,
        substitute_beat: np.ndarray,
    ) -> np.ndarray:
        """Swap a full beat in a specific lead."""
        modified = ecg_waveform.copy()

        if modified.ndim == 3:
            modified = modified.squeeze(0)

        beat_len = end_sample - start_sample
        sub_len = len(substitute_beat)

        if sub_len != beat_len:
            # Resample
            old_x = np.linspace(0, 1, sub_len)
            new_x = np.linspace(0, 1, beat_len)
            substitute_beat = np.interp(new_x, old_x, substitute_beat)

        modified[start_sample:end_sample, lead_index] = substitute_beat
        return modified


@dataclass
class FullBeatResult:
    ecg_idx: int
    original_prob: float
    counterfactual_prob: float
    flipped: bool
    n_beats_modified: int
    modified_beats: List[Dict]
    counterfactual_waveform: Optional[np.ndarray] = None


class FullBeatCounterfactualEngine:
    """Counterfactual engine using full-beat substitution."""

    def __init__(
        self,
        model: nn.Module,
        library: FullBeatLibraryLoader,
        config: FullBeatConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.library = library
        self.config = config
        self.substitutor = FullBeatSubstitutor()

    def _predict(self, waveform: torch.Tensor, tabular: torch.Tensor) -> float:
        with torch.no_grad():
            waveform = waveform.to(self.device)
            tabular = tabular.to(self.device)
            if waveform.ndim == 3:
                waveform = waveform.unsqueeze(1)
            logits = self.model((waveform, tabular))
            probs = torch.sigmoid(logits)
        return float(probs[0, 11].cpu())

    def search(
        self,
        ecg_idx: int,
        original_waveform: np.ndarray,
        tabular: torch.Tensor,
        ranked_beats: List[Dict],  # Now beats instead of segments
    ) -> FullBeatResult:
        """Search for counterfactual by modifying full beats."""

        if isinstance(original_waveform, np.ndarray):
            wf_tensor = torch.tensor(original_waveform, dtype=torch.float32)
        else:
            wf_tensor = original_waveform

        original_prob = self._predict(wf_tensor, tabular)
        print(f"  Original SHD prob: {original_prob:.4f}")

        if original_prob < self.config.shd_threshold:
            print(f"  Already normal — skipping.")
            return FullBeatResult(
                ecg_idx=ecg_idx,
                original_prob=original_prob,
                counterfactual_prob=original_prob,
                flipped=False,
                n_beats_modified=0,
                modified_beats=[],
            )

        best_result = FullBeatResult(
            ecg_idx=ecg_idx,
            original_prob=original_prob,
            counterfactual_prob=original_prob,
            flipped=False,
            n_beats_modified=0,
            modified_beats=[],
        )

        # Try 1, 2, 3 beats
        for n_mods in range(1, self.config.max_beats_to_modify + 1):
            print(f"  Trying {n_mods} beat modification(s)...")
            result = self._search_k_beats(
                ecg_idx, original_waveform, tabular, ranked_beats, n_mods, best_result
            )

            if result.flipped:
                print(f"  ✓ Flip found with {n_mods} beat(s)!")
                return result

            if result.counterfactual_prob < best_result.counterfactual_prob:
                best_result = result

        print(f"  ✗ No flip. Best prob: {best_result.counterfactual_prob:.4f}")
        return best_result

    def _search_k_beats(
        self,
        ecg_idx: int,
        original_waveform: np.ndarray,
        tabular: torch.Tensor,
        ranked_beats: List[Dict],
        k: int,
        best_so_far: FullBeatResult,
    ) -> FullBeatResult:

        best_result = best_so_far
        top_beats = ranked_beats[:min(15, len(ranked_beats))]

        if k == 1:
            for beat in top_beats:
                result = self._try_single_beat(ecg_idx, original_waveform, tabular, beat)
                if result.flipped:
                    return result
                if result.counterfactual_prob < best_result.counterfactual_prob:
                    best_result = result

        elif k == 2:
            for i, beat1 in enumerate(top_beats[:8]):
                for beat2 in top_beats[i+1:9]:
                    result = self._try_two_beats(ecg_idx, original_waveform, tabular, beat1, beat2)
                    if result.flipped:
                        return result
                    if result.counterfactual_prob < best_result.counterfactual_prob:
                        best_result = result

        elif k == 3:
            for i, beat1 in enumerate(top_beats[:5]):
                for j, beat2 in enumerate(top_beats[i+1:6]):
                    for beat3 in top_beats[j+1:7]:
                        result = self._try_three_beats(ecg_idx, original_waveform, tabular, beat1, beat2, beat3)
                        if result.flipped:
                            return result
                        if result.counterfactual_prob < best_result.counterfactual_prob:
                            best_result = result

        return best_result

    def _try_single_beat(
        self,
        ecg_idx: int,
        original_waveform: np.ndarray,
        tabular: torch.Tensor,
        beat_info: Dict,
    ) -> FullBeatResult:

        original_prob = self._predict(
            torch.tensor(original_waveform, dtype=torch.float32), tabular
        )

        lead_idx = beat_info["lead_index"]
        start = beat_info["start_sample"]
        end = beat_info["end_sample"]

        # Extract original beat
        ecg_2d = original_waveform.squeeze() if original_waveform.ndim == 3 else original_waveform
        if ecg_2d.ndim == 2 and ecg_2d.shape[0] == 12:
            ecg_2d = ecg_2d.T

        original_beat = ecg_2d[start:end, lead_idx]

        substitutes = self.library.find_substitutes(
            target_lead=beat_info["lead"],
            target_waveform=original_beat,
            n_candidates=self.config.max_candidates_per_beat,
        )

        if len(substitutes) == 0:
            return FullBeatResult(
                ecg_idx=ecg_idx,
                original_prob=original_prob,
                counterfactual_prob=original_prob,
                flipped=False,
                n_beats_modified=0,
                modified_beats=[],
            )

        best_prob = original_prob
        best_modified = None
        best_sub_info = None

        for sub_key, sub_dist, sub_wf in substitutes:
            modified = self.substitutor.substitute_beat(
                original_waveform, lead_idx, start, end, sub_wf
            )

            modified_tensor = torch.tensor(modified[np.newaxis, :, :], dtype=torch.float32)
            prob = self._predict(modified_tensor, tabular)

            if prob < best_prob:
                best_prob = prob
                best_modified = modified
                best_sub_info = {
                    "beat": beat_info,
                    "substitute_key": sub_key,
                    "distance": sub_dist,
                    "new_prob": prob,
                }

            if prob < self.config.shd_threshold:
                return FullBeatResult(
                    ecg_idx=ecg_idx,
                    original_prob=original_prob,
                    counterfactual_prob=prob,
                    flipped=True,
                    n_beats_modified=1,
                    modified_beats=[best_sub_info] if best_sub_info else [],
                    counterfactual_waveform=modified,
                )

        return FullBeatResult(
            ecg_idx=ecg_idx,
            original_prob=original_prob,
            counterfactual_prob=best_prob,
            flipped=False,
            n_beats_modified=1 if best_sub_info else 0,
            modified_beats=[best_sub_info] if best_sub_info else [],
            counterfactual_waveform=best_modified,
        )

    def _try_two_beats(self, ecg_idx, original_waveform, tabular, beat1, beat2):
        """Try two beat substitutions."""
        original_prob = self._predict(torch.tensor(original_waveform, dtype=torch.float32), tabular)

        ecg_2d = original_waveform.squeeze() if original_waveform.ndim == 3 else original_waveform
        if ecg_2d.ndim == 2 and ecg_2d.shape[0] == 12:
            ecg_2d = ecg_2d.T

        orig1 = ecg_2d[beat1["start_sample"]:beat1["end_sample"], beat1["lead_index"]]
        orig2 = ecg_2d[beat2["start_sample"]:beat2["end_sample"], beat2["lead_index"]]

        subs1 = self.library.find_substitutes(beat1["lead"], orig1, 5)
        subs2 = self.library.find_substitutes(beat2["lead"], orig2, 5)

        best_prob = original_prob
        best_modified = None
        best_info = None

        for sk1, sd1, sw1 in subs1:
            for sk2, sd2, sw2 in subs2:
                modified = self.substitutor.substitute_beat(
                    original_waveform, beat1["lead_index"], beat1["start_sample"], beat1["end_sample"], sw1
                )
                modified = self.substitutor.substitute_beat(
                    modified, beat2["lead_index"], beat2["start_sample"], beat2["end_sample"], sw2
                )

                prob = self._predict(torch.tensor(modified[np.newaxis, :, :], dtype=torch.float32), tabular)

                if prob < best_prob:
                    best_prob = prob
                    best_modified = modified
                    best_info = [
                        {"beat": beat1, "key": sk1, "dist": sd1},
                        {"beat": beat2, "key": sk2, "dist": sd2},
                    ]

                if prob < self.config.shd_threshold:
                    return FullBeatResult(
                        ecg_idx=ecg_idx,
                        original_prob=original_prob,
                        counterfactual_prob=prob,
                        flipped=True,
                        n_beats_modified=2,
                        modified_beats=best_info,
                        counterfactual_waveform=modified,
                    )

        return FullBeatResult(
            ecg_idx=ecg_idx,
            original_prob=original_prob,
            counterfactual_prob=best_prob,
            flipped=False,
            n_beats_modified=2 if best_info else 0,
            modified_beats=best_info or [],
            counterfactual_waveform=best_modified,
        )

    def _try_three_beats(self, ecg_idx, original_waveform, tabular, beat1, beat2, beat3):
        """Try three beat substitutions."""
        original_prob = self._predict(torch.tensor(original_waveform, dtype=torch.float32), tabular)

        ecg_2d = original_waveform.squeeze() if original_waveform.ndim == 3 else original_waveform
        if ecg_2d.ndim == 2 and ecg_2d.shape[0] == 12:
            ecg_2d = ecg_2d.T

        orig1 = ecg_2d[beat1["start_sample"]:beat1["end_sample"], beat1["lead_index"]]
        orig2 = ecg_2d[beat2["start_sample"]:beat2["end_sample"], beat2["lead_index"]]
        orig3 = ecg_2d[beat3["start_sample"]:beat3["end_sample"], beat3["lead_index"]]

        subs1 = self.library.find_substitutes(beat1["lead"], orig1, 3)
        subs2 = self.library.find_substitutes(beat2["lead"], orig2, 3)
        subs3 = self.library.find_substitutes(beat3["lead"], orig3, 3)

        best_prob = original_prob
        best_modified = None
        best_info = None

        for sk1, sd1, sw1 in subs1:
            for sk2, sd2, sw2 in subs2:
                for sk3, sd3, sw3 in subs3:
                    modified = original_waveform.copy()
                    for beat, sw in [(beat1, sw1), (beat2, sw2), (beat3, sw3)]:
                        modified = self.substitutor.substitute_beat(
                            modified, beat["lead_index"], beat["start_sample"], beat["end_sample"], sw
                        )

                    prob = self._predict(torch.tensor(modified[np.newaxis, :, :], dtype=torch.float32), tabular)

                    if prob < best_prob:
                        best_prob = prob
                        best_modified = modified
                        best_info = [
                            {"beat": beat1, "key": sk1, "dist": sd1},
                            {"beat": beat2, "key": sk2, "dist": sd2},
                            {"beat": beat3, "key": sk3, "dist": sd3},
                        ]

                    if prob < self.config.shd_threshold:
                        return FullBeatResult(
                            ecg_idx=ecg_idx,
                            original_prob=original_prob,
                            counterfactual_prob=prob,
                            flipped=True,
                            n_beats_modified=3,
                            modified_beats=best_info,
                            counterfactual_waveform=modified,
                        )

        return FullBeatResult(
            ecg_idx=ecg_idx,
            original_prob=original_prob,
            counterfactual_prob=best_prob,
            flipped=False,
            n_beats_modified=3 if best_info else 0,
            modified_beats=best_info or [],
            counterfactual_waveform=best_modified,
        )


def compute_beat_saliency(
    saliency_cache: Dict[int, List[Dict]],
) -> Dict[int, List[Dict]]:
    """
    Aggregate segment-level saliency to beat-level saliency.

    For each beat (identified by lead + beat_index), sum the importances
    of all its constituent segments (P, QRS, ST, T).
    """
    beat_saliency = {}

    for ecg_idx, segments in saliency_cache.items():
        # Group by (lead, beat_index)
        beat_importances = {}
        for seg in segments:
            key = (seg["lead"], seg["beat_index"])
            if key not in beat_importances:
                beat_importances[key] = {
                    "lead": seg["lead"],
                    "lead_index": seg["lead_index"],
                    "beat_index": seg["beat_index"],
                    "importance": 0.0,
                    "start_sample": seg["start_sample"],
                    "end_sample": seg["end_sample"],
                }
            beat_importances[key]["importance"] += seg["importance"]
            # Expand beat boundaries to cover all segments
            beat_importances[key]["start_sample"] = min(
                beat_importances[key]["start_sample"], seg["start_sample"]
            )
            beat_importances[key]["end_sample"] = max(
                beat_importances[key]["end_sample"], seg["end_sample"]
            )

        # Sort by total importance
        beats = sorted(beat_importances.values(), key=lambda x: x["importance"], reverse=True)
        beat_saliency[ecg_idx] = beats

    return beat_saliency


def run_full_beat_search(
    model: nn.Module,
    n_ecgs: int = 10,
    config: Optional[FullBeatConfig] = None,
) -> List[FullBeatResult]:
    config = config or FullBeatConfig()
    device = next(model.parameters()).device

    # Load saliency cache and convert to beat-level
    print("Loading saliency cache...")
    with open(config.saliency_cache_path, "rb") as f:
        saliency_cache = pickle.load(f)
    print(f"  Loaded {len(saliency_cache)} ECGs")

    print("Converting to beat-level saliency...")
    beat_saliency = compute_beat_saliency(saliency_cache)
    print("  Done")

    # Load full-beat library
    library = FullBeatLibraryLoader(config)
    library.load()

    # Initialize engine
    engine = FullBeatCounterfactualEngine(model, library, config, device=str(device))

    # Load data
    val_waveforms = np.load(config.data_dir / "EchoNext_val_waveforms.npy")
    val_tabular = np.load(config.data_dir / "EchoNext_val_tabular_features.npy")

    results = []
    ecg_indices = list(beat_saliency.keys())[:n_ecgs]

    print(f"Running full-beat search on {len(ecg_indices)} ECGs...")
    print("=" * 60)

    for i, ecg_idx in enumerate(ecg_indices):
        print(f"[{i+1}/{len(ecg_indices)}] ECG {ecg_idx}")

        wf = val_waveforms[ecg_idx]
        tab = torch.tensor(val_tabular[ecg_idx:ecg_idx+1], dtype=torch.float32)

        ranked_beats = beat_saliency[ecg_idx]
        result = engine.search(ecg_idx, wf, tab, ranked_beats)
        results.append(result)

    with open(config.results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {config.results_path}")

    n_flipped = sum(1 for r in results if r.flipped)
    n_abnormal = sum(1 for r in results if r.original_prob >= config.shd_threshold)
    print(f"Summary: {n_flipped}/{n_abnormal} abnormal ECGs flipped")

    return results


if __name__ == "__main__":
    print("Full-beat counterfactual engine loaded.")