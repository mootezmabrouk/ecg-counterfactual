"""Visualize counterfactual results: original vs modified ECG.

Usage:
    python scripts/visualize_counterfactual.py --ecg_idx 3253
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Phase3Config


def plot_counterfactual(
    ecg_idx: int,
    original_waveform: np.ndarray,
    counterfactual_waveform: np.ndarray,
    modified_segments: list,
    original_prob: float,
    cf_prob: float,
    save_path: Path | None = None,
):
    """Plot original vs counterfactual ECG with modified segments highlighted."""

    fig, axes = plt.subplots(12, 1, figsize=(14, 22), sharex=True)

    # EchoNext format: (1, 2500, 12) or (2500, 12)
    orig = original_waveform.squeeze()  # (2500, 12)
    cf = counterfactual_waveform.squeeze()

    t = np.arange(len(orig)) / 250.0
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
        # Original in black
        ax.plot(t, orig[:, lead_idx], 'k-', linewidth=0.7, alpha=0.8, label='Original')
        # Counterfactual in red
        ax.plot(t, cf[:, lead_idx], 'r-', linewidth=0.7, alpha=0.6, label='Counterfactual')

        ax.set_ylabel(lead_names[lead_idx], rotation=0, ha="right", va="center")
        ax.set_yticks([])

        # Highlight modified segments for this lead
        for mod in modified_segments:
            seg = mod["segment"]
            if seg["lead_index"] == lead_idx:
                start_t = seg["start_sample"] / 250.0
                end_t = seg["end_sample"] / 250.0
                color = colors.get(seg["segment_type"], "gray")
                ax.axvspan(start_t, end_t, alpha=0.25, color=color)

    axes[-1].set_xlabel("Time (s)")

    status = "FLIPPED" if cf_prob < 0.5 else "NO FLIP"
    fig.suptitle(
        f"ECG {ecg_idx} — {status}\n"
        f"Original SHD: {original_prob:.3f} → Counterfactual SHD: {cf_prob:.3f}",
        fontsize=14
    )

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color='k', lw=1, label='Original'),
        Line2D([0], [0], color='r', lw=1, label='Counterfactual'),
        Patch(facecolor='blue', alpha=0.3, label='P-wave modified'),
        Patch(facecolor='red', alpha=0.3, label='QRS modified'),
        Patch(facecolor='green', alpha=0.3, label='ST modified'),
        Patch(facecolor='purple', alpha=0.3, label='T-wave modified'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize counterfactual")
    parser.add_argument("--ecg_idx", type=int, required=True, help="ECG index")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    config = Phase3Config()

    # Load counterfactual results
    results_path = config.output_dir / "counterfactual_results.pkl"
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    # Find result for this ECG
    result = None
    for r in results:
        if r.ecg_idx == args.ecg_idx:
            result = r
            break

    if result is None:
        print(f"ECG {args.ecg_idx} not found in results.")
        print(f"Available: {[r.ecg_idx for r in results]}")
        return

    if result.counterfactual_waveform is None:
        print(f"ECG {args.ecg_idx} has no counterfactual waveform (no flip or no modification).")
        return

    # Load original waveform
    val_waveforms = np.load(config.val_waveforms_path)
    original = val_waveforms[args.ecg_idx]

    save_path = Path(args.output) if args.output else config.output_dir / f"cf_ecg_{args.ecg_idx}.png"

    plot_counterfactual(
        args.ecg_idx,
        original,
        result.counterfactual_waveform,
        result.modified_segments,
        result.original_prob,
        result.counterfactual_prob,
        save_path,
    )


if __name__ == "__main__":
    main()