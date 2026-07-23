"""Counterfactual Search Engine for ECG XAI — Phase 5 (Fixed).

Fixes applied:
  A. Diversify segment types (QRS + ST + T + P)
  B. Increase candidates per segment (50)
  C. Cross-lead combination strategy
  D. Verify ECG is actually abnormal before searching
"""

from __future__ import annotations

import pickle
import warnings
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
from src.preprocessing import preprocess_echonext_record


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CounterfactualConfig:
    """Configuration for the counterfactual search engine."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    segment_library_csv: Path = Path("outputs/segment_library.csv")
    segment_library_npz: Path = Path("outputs/segment_library_waveforms.npz")
    saliency_cache_path: Path = Path("outputs/saliency_scores.pkl")

    # Search parameters
    max_segments_to_modify: int = 3
    max_candidates_per_segment: int = 50   # FIX B: increased from 10
    shd_threshold: float = 0.5

    # FIX A: Diversify segment types
    top_qrs: int = 5
    top_st: int = 3
    top_t: int = 2
    top_p: int = 1

    # FIX C: Cross-lead strategy
    cross_lead_bonus: bool = True  # Prefer combining segments from different leads

    similarity_features: List[str] = field(
        default_factory=lambda: ["mean", "std", "length", "min", "max"]
    )

    results_path: Path = Path("outputs/counterfactual_results.pkl")

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SEGMENT FEATURE EXTRACTOR
# ============================================================================

class SegmentFeatureExtractor:
    """Extract features from a waveform segment for similarity matching."""

    FEATURES = ["mean", "std", "length", "min", "max"]

    @classmethod
    def extract(cls, waveform: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(waveform)),
            "std": float(np.std(waveform)),
            "length": float(len(waveform)),
            "min": float(np.min(waveform)),
            "max": float(np.max(waveform)),
        }

    @classmethod
    def distance(
        cls,
        features_a: Dict[str, float],
        features_b: Dict[str, float],
        features_to_use: Optional[List[str]] = None,
    ) -> float:
        keys = features_to_use or cls.FEATURES
        squared_diffs = []
        for key in keys:
            val_a = features_a[key]
            val_b = features_b[key]
            denom = abs(val_a) + abs(val_b) + 1e-8
            squared_diffs.append(((val_a - val_b) / denom) ** 2)
        return float(np.sqrt(np.sum(squared_diffs)))


# ============================================================================
# SEGMENT LIBRARY LOADER
# ============================================================================

class SegmentLibraryLoader:
    """Load segment library and precompute features for fast matching."""

    def __init__(self, config: CounterfactualConfig):
        self.config = config
        self.library_df: Optional[pd.DataFrame] = None
        self.waveforms: Optional[Dict[str, np.ndarray]] = None
        self.feature_index: Optional[pd.DataFrame] = None

    def load(self) -> "SegmentLibraryLoader":
        print("Loading segment library...")
        self.library_df = pd.read_csv(self.config.segment_library_csv)

        npz = np.load(self.config.segment_library_npz)
        self.waveforms = {k: npz[k] for k in npz.files}

        print(f"  Loaded {len(self.library_df)} segments")
        print(f"  Loaded {len(self.waveforms)} waveforms")

        print("Precomputing segment features...")
        features_list = []
        for idx, row in self.library_df.iterrows():
            key = row.get("waveform_key", f"segment_{idx:08d}")
            if key in self.waveforms:
                wf = self.waveforms[key]
                feats = SegmentFeatureExtractor.extract(wf)
                feats["waveform_key"] = key
                feats["lead"] = row["lead"]
                feats["segment_type"] = row["segment_type"]
                features_list.append(feats)

        self.feature_index = pd.DataFrame(features_list)
        print(f"  Feature index built: {len(self.feature_index)} entries")

        return self

    def find_substitutes(
        self,
        target_lead: str,
        target_segment_type: str,
        target_waveform: np.ndarray,
        n_candidates: int = 10,
    ) -> List[Tuple[str, float, np.ndarray]]:
        mask = (
            (self.feature_index["lead"] == target_lead) &
            (self.feature_index["segment_type"] == target_segment_type)
        )
        candidates = self.feature_index[mask].copy()

        if len(candidates) == 0:
            return []

        target_feats = SegmentFeatureExtractor.extract(target_waveform)

        distances = []
        for _, row in candidates.iterrows():
            cand_feats = {
                "mean": row["mean"], "std": row["std"], "length": row["length"],
                "min": row["min"], "max": row["max"],
            }
            dist = SegmentFeatureExtractor.distance(target_feats, cand_feats)
            distances.append(dist)

        candidates["distance"] = distances
        candidates = candidates.sort_values("distance", ascending=False).head(n_candidates)

        results = []
        for _, row in candidates.iterrows():
            key = row["waveform_key"]
            wf = self.waveforms.get(key, np.array([]))
            results.append((key, row["distance"], wf))

        return results


# ============================================================================
# SEGMENT SUBSTITUTOR
# ============================================================================

class SegmentSubstitutor:
    def __init__(self, fs: float = 250.0):
        self.fs = fs

    def substitute_lead_segment(
        self,
        ecg_waveform: np.ndarray,
        lead_index: int,
        start_sample: int,
        end_sample: int,
        substitute_segment: np.ndarray,
    ) -> np.ndarray:
        modified = ecg_waveform.copy()

        if modified.ndim == 3:
            modified = modified.squeeze(0)

        seg_len = end_sample - start_sample
        sub_len = len(substitute_segment)

        if sub_len != seg_len:
            substitute_segment = self._resample(substitute_segment, seg_len)

        modified[start_sample:end_sample, lead_index] = substitute_segment
        return modified

    @staticmethod
    def _resample(waveform: np.ndarray, target_length: int) -> np.ndarray:
        if len(waveform) == target_length:
            return waveform
        old_x = np.linspace(0, 1, len(waveform))
        new_x = np.linspace(0, 1, target_length)
        return np.interp(new_x, old_x, waveform)


# ============================================================================
# COUNTERFACTUAL ENGINE
# ============================================================================

@dataclass
class CounterfactualResult:
    ecg_idx: int
    original_prob: float
    counterfactual_prob: float
    flipped: bool
    n_segments_modified: int
    modified_segments: List[Dict]
    counterfactual_waveform: Optional[np.ndarray] = None
    search_path: List[Dict] = field(default_factory=list)


class CounterfactualEngine:
    """Greedy best-first counterfactual search with diversified segment types."""

    def __init__(
        self,
        model: nn.Module,
        library_loader: SegmentLibraryLoader,
        config: CounterfactualConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.library = library_loader
        self.config = config
        self.substitutor = SegmentSubstitutor()

    def _predict(self, waveform: torch.Tensor, tabular: torch.Tensor) -> float:
        with torch.no_grad():
            waveform = waveform.to(self.device)
            tabular = tabular.to(self.device)

            if waveform.ndim == 3:
                waveform = waveform.unsqueeze(1)

            logits = self.model((waveform, tabular))
            probs = torch.sigmoid(logits)
        return float(probs[0, 11].cpu())

    def _get_diversified_segments(self, ranked_segments: List[Dict]) -> List[Dict]:
        """FIX A: Get top segments from each type, not just overall top."""
        top_qrs = [s for s in ranked_segments if s["segment_type"] == "qrs_complex"][:self.config.top_qrs]
        top_st = [s for s in ranked_segments if s["segment_type"] == "st_segment"][:self.config.top_st]
        top_t = [s for s in ranked_segments if s["segment_type"] == "t_wave"][:self.config.top_t]
        top_p = [s for s in ranked_segments if s["segment_type"] == "p_wave"][:self.config.top_p]

        combined = top_qrs + top_st + top_t + top_p

        # Sort by importance within combined list
        combined = sorted(combined, key=lambda x: x["importance"], reverse=True)

        print(f"    Diversified search set: {len(top_qrs)} QRS + {len(top_st)} ST + {len(top_t)} T + {len(top_p)} P = {len(combined)} segments")

        return combined

    def search(
        self,
        ecg_idx: int,
        original_waveform: np.ndarray,
        tabular: torch.Tensor,
        ranked_segments: List[Dict],
    ) -> CounterfactualResult:

        if isinstance(original_waveform, np.ndarray):
            wf_tensor = torch.tensor(original_waveform, dtype=torch.float32)
        else:
            wf_tensor = original_waveform

        original_prob = self._predict(wf_tensor, tabular)

        print(f"  Original SHD prob: {original_prob:.4f}")

        # FIX D: Verify actually abnormal
        if original_prob < self.config.shd_threshold:
            print(f"  Already normal (prob {original_prob:.4f} < {self.config.shd_threshold}) — skipping.")
            return CounterfactualResult(
                ecg_idx=ecg_idx,
                original_prob=original_prob,
                counterfactual_prob=original_prob,
                flipped=False,
                n_segments_modified=0,
                modified_segments=[],
            )

        best_result = CounterfactualResult(
            ecg_idx=ecg_idx,
            original_prob=original_prob,
            counterfactual_prob=original_prob,
            flipped=False,
            n_segments_modified=0,
            modified_segments=[],
            search_path=[],
        )

        # Get diversified segment set
        search_segments = self._get_diversified_segments(ranked_segments)

        if len(search_segments) == 0:
            print("  No segments available for search.")
            return best_result

        # Try modifying 1, 2, 3 segments
        for n_mods in range(1, self.config.max_segments_to_modify + 1):
            print(f"  Trying {n_mods} segment modification(s)...")
            result = self._search_k_segments(
                ecg_idx=ecg_idx,
                original_waveform=original_waveform,
                tabular=tabular,
                search_segments=search_segments,
                k=n_mods,
                best_so_far=best_result,
            )

            if result.flipped:
                print(f"  ✓ Flip found with {n_mods} segment(s)!")
                return result

            if result.counterfactual_prob < best_result.counterfactual_prob:
                best_result = result

        print(f"  ✗ No flip found. Best prob achieved: {best_result.counterfactual_prob:.4f}")
        return best_result

    def _search_k_segments(
        self,
        ecg_idx: int,
        original_waveform: np.ndarray,
        tabular: torch.Tensor,
        search_segments: List[Dict],
        k: int,
        best_so_far: CounterfactualResult,
    ) -> CounterfactualResult:

        best_result = best_so_far

        if k == 1:
            for seg in search_segments:
                result = self._try_single_segment(ecg_idx, original_waveform, tabular, seg)
                if result.flipped:
                    return result
                if result.counterfactual_prob < best_result.counterfactual_prob:
                    best_result = result

        elif k == 2:
            # FIX C: Prioritize cross-lead combinations
            pairs = []
            for i, seg1 in enumerate(search_segments):
                for j, seg2 in enumerate(search_segments[i+1:], start=i+1):
                    # Bonus for different leads
                    cross_lead = seg1["lead_index"] != seg2["lead_index"]
                    # Sort by combined importance
                    combined_importance = seg1["importance"] + seg2["importance"]
                    if cross_lead and self.config.cross_lead_bonus:
                        combined_importance *= 1.1  # Slight boost
                    pairs.append((combined_importance, i, j, seg1, seg2))

            # Sort by combined importance (descending)
            pairs.sort(key=lambda x: x[0], reverse=True)

            # Try top pairs (limit to avoid explosion)
            for _, _, _, seg1, seg2 in pairs[:15]:
                result = self._try_two_segments(ecg_idx, original_waveform, tabular, seg1, seg2)
                if result.flipped:
                    return result
                if result.counterfactual_prob < best_result.counterfactual_prob:
                    best_result = result

        elif k == 3:
            # Try triplets with cross-lead diversity
            triplets = []
            for i, seg1 in enumerate(search_segments):
                for j, seg2 in enumerate(search_segments[i+1:], start=i+1):
                    for l, seg3 in enumerate(search_segments[j+1:], start=j+1):
                        n_unique_leads = len({seg1["lead_index"], seg2["lead_index"], seg3["lead_index"]})
                        combined_importance = seg1["importance"] + seg2["importance"] + seg3["importance"]
                        if n_unique_leads >= 2 and self.config.cross_lead_bonus:
                            combined_importance *= 1.15
                        triplets.append((combined_importance, seg1, seg2, seg3))

            triplets.sort(key=lambda x: x[0], reverse=True)

            for _, seg1, seg2, seg3 in triplets[:5]:
                result = self._try_three_segments(ecg_idx, original_waveform, tabular, seg1, seg2, seg3)
                if result.flipped:
                    return result
                if result.counterfactual_prob < best_result.counterfactual_prob:
                    best_result = result

        return best_result

    def _try_single_segment(
        self,
        ecg_idx: int,
        original_waveform: np.ndarray,
        tabular: torch.Tensor,
        segment_info: Dict,
    ) -> CounterfactualResult:

        original_prob = self._predict(
            torch.tensor(original_waveform, dtype=torch.float32), tabular
        )

        lead_idx = segment_info["lead_index"]
        start = segment_info["start_sample"]
        end = segment_info["end_sample"]

        ecg_2d = original_waveform.squeeze() if original_waveform.ndim == 3 else original_waveform
        if ecg_2d.ndim == 2 and ecg_2d.shape[0] == 12:
            ecg_2d = ecg_2d.T

        original_segment = ecg_2d[start:end, lead_idx]

        substitutes = self.library.find_substitutes(
            target_lead=segment_info["lead"],
            target_segment_type=segment_info["segment_type"],
            target_waveform=original_segment,
            n_candidates=self.config.max_candidates_per_segment,
        )

        if len(substitutes) == 0:
            return CounterfactualResult(
                ecg_idx=ecg_idx,
                original_prob=original_prob,
                counterfactual_prob=original_prob,
                flipped=False,
                n_segments_modified=0,
                modified_segments=[],
            )

        best_prob = original_prob
        best_modified = None
        best_sub_info = None

        for i, (sub_key, sub_dist, sub_wf) in enumerate(substitutes):
            modified = self.substitutor.substitute_lead_segment(
                original_waveform, lead_idx, start, end, sub_wf
            )

            modified_tensor = torch.tensor(modified[np.newaxis, :, :], dtype=torch.float32)
            prob = self._predict(modified_tensor, tabular)

            if prob < best_prob:
                best_prob = prob
                best_modified = modified
                best_sub_info = {
                    "segment": segment_info,
                    "substitute_key": sub_key,
                    "distance": sub_dist,
                    "new_prob": prob,
                }

            if prob < self.config.shd_threshold:
                return CounterfactualResult(
                    ecg_idx=ecg_idx,
                    original_prob=original_prob,
                    counterfactual_prob=prob,
                    flipped=True,
                    n_segments_modified=1,
                    modified_segments=[best_sub_info] if best_sub_info else [],
                    counterfactual_waveform=modified,
                )
            if i >= 20 and best_prob > original_prob * 0.99:
                break

        return CounterfactualResult(
            ecg_idx=ecg_idx,
            original_prob=original_prob,
            counterfactual_prob=best_prob,
            flipped=False,
            n_segments_modified=1 if best_sub_info else 0,
            modified_segments=[best_sub_info] if best_sub_info else [],
            counterfactual_waveform=best_modified,
        )

    def _try_two_segments(
        self,
        ecg_idx: int,
        original_waveform: np.ndarray,
        tabular: torch.Tensor,
        seg1_info: Dict,
        seg2_info: Dict,
    ) -> CounterfactualResult:

        original_prob = self._predict(
            torch.tensor(original_waveform, dtype=torch.float32), tabular
        )

        subs1 = self.library.find_substitutes(
            seg1_info["lead"], seg1_info["segment_type"],
            self._extract_segment(original_waveform, seg1_info),
            self.config.max_candidates_per_segment,
        )
        subs2 = self.library.find_substitutes(
            seg2_info["lead"], seg2_info["segment_type"],
            self._extract_segment(original_waveform, seg2_info),
            self.config.max_candidates_per_segment,
        )

        if len(subs1) == 0 or len(subs2) == 0:
            return CounterfactualResult(
                ecg_idx=ecg_idx,
                original_prob=original_prob,
                counterfactual_prob=original_prob,
                flipped=False,
                n_segments_modified=0,
                modified_segments=[],
            )

        best_prob = original_prob
        best_modified = None
        best_subs_info = None

        # Try top 5 candidates for each (more thorough than before)
        for sub1_key, sub1_dist, sub1_wf in subs1[:5]:
            for sub2_key, sub2_dist, sub2_wf in subs2[:5]:
                modified = self.substitutor.substitute_lead_segment(
                    original_waveform,
                    seg1_info["lead_index"], seg1_info["start_sample"], seg1_info["end_sample"],
                    sub1_wf,
                )
                modified = self.substitutor.substitute_lead_segment(
                    modified,
                    seg2_info["lead_index"], seg2_info["start_sample"], seg2_info["end_sample"],
                    sub2_wf,
                )

                modified_tensor = torch.tensor(modified[np.newaxis, :, :], dtype=torch.float32)
                prob = self._predict(modified_tensor, tabular)

                if prob < best_prob:
                    best_prob = prob
                    best_modified = modified
                    best_subs_info = [
                        {"segment": seg1_info, "substitute_key": sub1_key, "distance": sub1_dist, "new_prob": prob},
                        {"segment": seg2_info, "substitute_key": sub2_key, "distance": sub2_dist, "new_prob": prob},
                    ]

                if prob < self.config.shd_threshold:
                    return CounterfactualResult(
                        ecg_idx=ecg_idx,
                        original_prob=original_prob,
                        counterfactual_prob=prob,
                        flipped=True,
                        n_segments_modified=2,
                        modified_segments=best_subs_info,
                        counterfactual_waveform=modified,
                    )

        return CounterfactualResult(
            ecg_idx=ecg_idx,
            original_prob=original_prob,
            counterfactual_prob=best_prob,
            flipped=False,
            n_segments_modified=2 if best_subs_info else 0,
            modified_segments=best_subs_info or [],
            counterfactual_waveform=best_modified,
        )

    def _try_three_segments(
        self,
        ecg_idx: int,
        original_waveform: np.ndarray,
        tabular: torch.Tensor,
        seg1_info: Dict,
        seg2_info: Dict,
        seg3_info: Dict,
    ) -> CounterfactualResult:

        original_prob = self._predict(
            torch.tensor(original_waveform, dtype=torch.float32), tabular
        )

        subs1 = self.library.find_substitutes(
            seg1_info["lead"], seg1_info["segment_type"],
            self._extract_segment(original_waveform, seg1_info), 3,
        )
        subs2 = self.library.find_substitutes(
            seg2_info["lead"], seg2_info["segment_type"],
            self._extract_segment(original_waveform, seg2_info), 3,
        )
        subs3 = self.library.find_substitutes(
            seg3_info["lead"], seg3_info["segment_type"],
            self._extract_segment(original_waveform, seg3_info), 3,
        )

        if len(subs1) == 0 or len(subs2) == 0 or len(subs3) == 0:
            return CounterfactualResult(
                ecg_idx=ecg_idx,
                original_prob=original_prob,
                counterfactual_prob=original_prob,
                flipped=False,
                n_segments_modified=0,
                modified_segments=[],
            )

        best_prob = original_prob
        best_modified = None
        best_subs_info = None

        for sub1_key, sub1_dist, sub1_wf in subs1:
            for sub2_key, sub2_dist, sub2_wf in subs2:
                for sub3_key, sub3_dist, sub3_wf in subs3:
                    modified = original_waveform.copy()
                    for seg_info, sub_wf in [(seg1_info, sub1_wf), (seg2_info, sub2_wf), (seg3_info, sub3_wf)]:
                        modified = self.substitutor.substitute_lead_segment(
                            modified,
                            seg_info["lead_index"], seg_info["start_sample"], seg_info["end_sample"],
                            sub_wf,
                        )

                    modified_tensor = torch.tensor(modified[np.newaxis, :, :], dtype=torch.float32)
                    prob = self._predict(modified_tensor, tabular)

                    if prob < best_prob:
                        best_prob = prob
                        best_modified = modified
                        best_subs_info = [
                            {"segment": seg1_info, "substitute_key": sub1_key, "distance": sub1_dist},
                            {"segment": seg2_info, "substitute_key": sub2_key, "distance": sub2_dist},
                            {"segment": seg3_info, "substitute_key": sub3_key, "distance": sub3_dist},
                        ]

                    if prob < self.config.shd_threshold:
                        return CounterfactualResult(
                            ecg_idx=ecg_idx,
                            original_prob=original_prob,
                            counterfactual_prob=prob,
                            flipped=True,
                            n_segments_modified=3,
                            modified_segments=best_subs_info,
                            counterfactual_waveform=modified,
                        )

        return CounterfactualResult(
            ecg_idx=ecg_idx,
            original_prob=original_prob,
            counterfactual_prob=best_prob,
            flipped=False,
            n_segments_modified=3 if best_subs_info else 0,
            modified_segments=best_subs_info or [],
            counterfactual_waveform=best_modified,
        )

    def _extract_segment(self, ecg_waveform: np.ndarray, segment_info: Dict) -> np.ndarray:
        ecg_2d = ecg_waveform.squeeze() if ecg_waveform.ndim == 3 else ecg_waveform
        if ecg_2d.ndim == 2 and ecg_2d.shape[0] == 12:
            ecg_2d = ecg_2d.T
        lead_idx = segment_info["lead_index"]
        start = segment_info["start_sample"]
        end = segment_info["end_sample"]
        return ecg_2d[start:end, lead_idx]


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_counterfactual_search(
    model: nn.Module,
    n_ecgs: int = 10,
    config: Optional[CounterfactualConfig] = None,
) -> List[CounterfactualResult]:
    config = config or CounterfactualConfig()
    device = next(model.parameters()).device

    print("Loading saliency cache...")
    with open(config.saliency_cache_path, "rb") as f:
        saliency_cache = pickle.load(f)
    print(f"  Loaded rankings for {len(saliency_cache)} ECGs")

    library_loader = SegmentLibraryLoader(config)
    library_loader.load()

    engine = CounterfactualEngine(model, library_loader, config, device=str(device))

    val_waveforms = np.load(config.data_dir / "EchoNext_val_waveforms.npy")
    val_tabular = np.load(config.data_dir / "EchoNext_val_tabular_features.npy")

    results = []
    ecg_indices = list(saliency_cache.keys())[:n_ecgs]

    print(f"Running counterfactual search on {len(ecg_indices)} ECGs...")
    print("=" * 60)

    for i, ecg_idx in enumerate(ecg_indices):
        print(f"[{i+1}/{len(ecg_indices)}] ECG {ecg_idx}")

        wf = val_waveforms[ecg_idx]
        tab = torch.tensor(val_tabular[ecg_idx:ecg_idx+1], dtype=torch.float32)

        ranked = saliency_cache[ecg_idx]

        result = engine.search(ecg_idx, wf, tab, ranked)
        results.append(result)

    with open(config.results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {config.results_path}")

    n_flipped = sum(1 for r in results if r.flipped)
    n_abnormal = sum(1 for r in results if r.original_prob >= config.shd_threshold)
    print(f"Summary: {n_flipped}/{n_abnormal} abnormal ECGs flipped ({n_flipped}/{len(results)} total)")

    return results


if __name__ == "__main__":
    print("Counterfactual Engine module loaded.")