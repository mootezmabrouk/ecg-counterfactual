"""Phase 5 Full-Beat: Counterfactual Search with Full-Beat Substitution.

Usage:
    python scripts/run_phase5_fullbeat.py --n_ecgs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "IntroECG/7-EchoNext Minimodel"))

from src.full_beat_engine import FullBeatConfig, run_full_beat_search


def load_model():
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
    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.to(device)
    model.eval()

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ecgs", type=int, default=10)
    parser.add_argument("--max_beats", type=int, default=3)
    parser.add_argument("--max_candidates", type=int, default=20)
    args = parser.parse_args()

    config = FullBeatConfig(
        max_beats_to_modify=args.max_beats,
        max_candidates_per_beat=args.max_candidates,
    )

    print("=" * 60)
    print("PHASE 5: Full-Beat Counterfactual Search")
    print("=" * 60)

    model = load_model()
    results = run_full_beat_search(model, n_ecgs=args.n_ecgs, config=config)

    print("" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for r in results:
        status = "✓ FLIPPED" if r.flipped else "✗ NO FLIP"
        print(f"ECG {r.ecg_idx}: {status}")
        print(f"  Original: {r.original_prob:.4f} → Best: {r.counterfactual_prob:.4f}")
        print(f"  Beats modified: {r.n_beats_modified}")
        for mod in r.modified_beats:
            beat = mod["beat"]
            print(f"    → Beat {beat['beat_index']} on {beat['lead']} — dist={mod['distance']:.4f}")


if __name__ == "__main__":
    main()