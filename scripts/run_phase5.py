"""Phase 5 entry point: Counterfactual Search Engine (Fixed).

Usage:
    python scripts/run_phase5.py --n_ecgs 10 --max_candidates 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "IntroECG/7-EchoNext Minimodel"))

from src.counterfactual_engine import CounterfactualConfig, run_counterfactual_search


def load_model():
    """Load the pretrained EchoNext Mini-Model."""
    from cradlenet.models.resnet1d_tabular import ResNet1dWithTabular

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet1dWithTabular(
        len_tabular_feature_vector=7,
        filter_size=16,
        input_channels=12,
        num_classes=12,
        dropout_value=0.5,
    )

    model_ckpt = PROJECT_ROOT / "IntroECG/7-EchoNext Minimodel/models/echonext_multilabel_minimodel/weights.pt"
    checkpoint = torch.load(model_ckpt, map_location=device, weights_only=True)
    state_dict = checkpoint["model"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print("  Model loaded successfully")

    return model


def main():
    parser = argparse.ArgumentParser(description="Run counterfactual search")
    parser.add_argument("--n_ecgs", type=int, default=10, help="Number of ECGs to process")
    parser.add_argument("--max_segments", type=int, default=3, help="Max segments to modify")
    parser.add_argument("--max_candidates", type=int, default=50, help="Candidates per segment")
    parser.add_argument("--top_qrs", type=int, default=5, help="Top QRS segments to try")
    parser.add_argument("--top_st", type=int, default=3, help="Top ST segments to try")
    parser.add_argument("--top_t", type=int, default=2, help="Top T segments to try")
    parser.add_argument("--top_p", type=int, default=1, help="Top P segments to try")
    args = parser.parse_args()

    config = CounterfactualConfig(
        max_segments_to_modify=args.max_segments,
        max_candidates_per_segment=args.max_candidates,
        top_qrs=args.top_qrs,
        top_st=args.top_st,
        top_t=args.top_t,
        top_p=args.top_p,
    )

    print("=" * 60)
    print("PHASE 5: Counterfactual Search Engine (Fixed)")
    print("=" * 60)
    print(f"Config: max_segments={args.max_segments}, candidates={args.max_candidates}")
    print(f"        QRS={args.top_qrs}, ST={args.top_st}, T={args.top_t}, P={args.top_p}")

    model = load_model()
    results = run_counterfactual_search(model, n_ecgs=args.n_ecgs, config=config)

    # Print detailed results
    print("" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)

    for r in results:
        status = "✓ FLIPPED" if r.flipped else "✗ NO FLIP"
        print(f"ECG {r.ecg_idx}: {status}")
        print(f"  Original prob:  {r.original_prob:.4f}")
        print(f"  Best prob:      {r.counterfactual_prob:.4f}")
        print(f"  Segments modified: {r.n_segments_modified}")
        for mod in r.modified_segments:
            seg = mod["segment"]
            print(f"    → {seg['segment_type']} on {seg['lead']} "
                  f"(beat {seg['beat_index']}) — dist={mod['distance']:.4f}")


if __name__ == "__main__":
    main()