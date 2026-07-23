"""Visualize saliency rankings and segment substitutions.

Usage:
    python scripts/visualize_saliency.py --ecg_idx 42
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Phase3Config


def plot_saliency_on_ecg(
    ecg_idx: int,
    waveform: np.ndarray,
    ranked_segments: list,
    top_n: int = 5,
    save_path: Path | None = None,
):
    """Plot ECG with top-N salient segments highlighted."""
    fig, axes = plt.subplots(12, 1, figsize=(14, 20), sharex=True)

    ecg_2d = waveform.squeeze()  # (2500, 12)
    t = np.arange(len(ecg_2d)) / 250.0

    lead_names = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]

    colors = {
        "p_wave": "blue",
        "qrs_complex": "red",
        "st_segment": "green",
        "t_wave": "purple",
    }

    for lead_idx, ax in enumerate(axes):
        ax.plot(t, ecg_2d[:, lead_idx], 'k-', linewidth=0.6, alpha=0.7)
        ax.set_ylabel(lead_names[lead_idx], rotation=0, ha="right", va="center")
        ax.set_ylim(ecg_2d[:, lead_idx].min() - 0.5, ecg_2d[:, lead_idx].max() + 0.5)
        ax.set_yticks([])

        # Highlight top-N segments for this lead
        lead_segments = [s for s in ranked_segments[:top_n] if s["lead_index"] == lead_idx]
        for seg in lead_segments:
            start_t = seg["start_sample"] / 250.0
            end_t = seg["end_sample"] / 250.0
            color = colors.get(seg["segment_type"], "gray")
            ax.axvspan(start_t, end_t, alpha=0.3, color=color)

            # Add importance score as text
            mid_t = (start_t + end_t) / 2
            y_pos = ax.get_ylim()[1] - 0.1
            ax.text(mid_t, y_pos, f"{seg['importance']:.3f}",
                   ha="center", va="top", fontsize=6, color=color)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"ECG {ecg_idx} — Top {top_n} Saliency Segments", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize saliency rankings")
    parser.add_argument("--ecg_idx", type=int, required=True, help="ECG index to visualize")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top segments to highlight")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    config = Phase3Config()

    # Load saliency cache
    with open(config.saliency_cache_path, "rb") as f:
        saliency_cache = pickle.load(f)

    if args.ecg_idx not in saliency_cache:
        print(f"ECG {args.ecg_idx} not found in saliency cache.")
        print(f"Available indices: {list(saliency_cache.keys())[:10]}...")
        return

    # Load waveform
    val_waveforms = np.load(config.val_waveforms_path)
    waveform = val_waveforms[args.ecg_idx]

    # Plot
    ranked = saliency_cache[args.ecg_idx]
    save_path = Path(args.output) if args.output else config.output_dir / f"saliency_ecg_{args.ecg_idx}.png"

    plot_saliency_on_ecg(args.ecg_idx, waveform, ranked, args.top_n, save_path)

    # Print top segments
    print(f"\nTop {args.top_n} segments for ECG {args.ecg_idx}:")
    for i, seg in enumerate(ranked[:args.top_n]):
        print(f"  {i+1}. {seg['segment_type']} on {seg['lead']} "
              f"(beat {seg['beat_index']}) — importance: {seg['importance']:.6f}")


if __name__ == "__main__":
    main()