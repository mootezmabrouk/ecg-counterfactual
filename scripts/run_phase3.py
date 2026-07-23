"""Phase 3 entry point: build normal segment library + compute saliency rankings."""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "IntroECG/7-EchoNext Minimodel"))

from src.config import Phase3Config
from src.preprocessing import preprocess_echonext_record
from src.segment_library import build_segment_library, save_segment_library
from src.saliency import SaliencyPrior


def load_model(config: Phase3Config) -> torch.nn.Module:
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


def build_normal_library(config: Phase3Config) -> pd.DataFrame:
    print("=" * 60)
    print("BLOCK 4: Building Normal Segment Library")
    print("=" * 60)

    # Load metadata and filter to val split FIRST, then reset index
    metadata = pd.read_csv(config.metadata_path)
    val_meta = metadata[metadata["split"] == "val"].reset_index(drop=True)
    
    normal_mask = val_meta["shd_moderate_or_greater_flag"] == 0
    normal_indices = val_meta[normal_mask].index.tolist()
    print(f"Found {len(normal_indices)} normal ECGs in val split")

    print("Preprocessing normal ECGs...")
    processed_records = []
    skipped = 0
    
    for i, idx in enumerate(normal_indices):
        if i % 100 == 0:
            print(f"  {i}/{len(normal_indices)}...")
        try:
            # Suppress neurokit2 warnings for clean output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                processed = preprocess_echonext_record(record=idx, data_dir=config.data_dir, split="val")
            processed_records.append(processed)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Skipping record {idx}: {e}")
            continue

    print(f"Successfully preprocessed {len(processed_records)} records ({skipped} skipped)")

    library = build_segment_library(processed_records, include_waveforms=True, drop_incomplete_beats=True)
    print(f"Library built: {len(library)} segments")

    save_segment_library(library, output_csv=config.segment_library_csv, waveform_npz=config.segment_library_npz)
    print(f"Saved to {config.segment_library_csv}")
    print(f"Waveforms saved to {config.segment_library_npz}")

    return library


def compute_saliency_rankings(config: Phase3Config, model: torch.nn.Module) -> dict:
    print("\n" + "=" * 60)
    print("BLOCK 3: Computing Saliency Rankings (Integrated Gradients)")
    print("=" * 60)

    device = next(model.parameters()).device
    print(f"Using device: {device}")

    # Load data
    metadata = pd.read_csv(config.metadata_path)
    val_meta = metadata[metadata["split"] == "val"].reset_index(drop=True)
    val_waveforms = np.load(config.val_waveforms_path)
    val_tabular = np.load(config.val_tabular_path)

    abnormal_mask = val_meta["shd_moderate_or_greater_flag"] == 1
    abnormal_indices = val_meta[abnormal_mask].index.tolist()
    print(f"Found {len(abnormal_indices)} abnormal ECGs in val split")

    n_samples = config.n_saliency_samples
    if n_samples > 0 and n_samples < len(abnormal_indices):
        import random
        random.seed(42)
        sample_indices = random.sample(abnormal_indices, n_samples)
        print(f"Computing saliency for {n_samples} sampled abnormal ECGs")
    else:
        sample_indices = abnormal_indices
        print(f"Computing saliency for all {len(abnormal_indices)} abnormal ECGs")

    saliency = SaliencyPrior(model, config, device=str(device))

    saliency_cache = {}
    failed = 0
    
    for i, idx in enumerate(sample_indices):
        print(f"  [{i+1}/{len(sample_indices)}] ECG {idx}...", end=" ")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                processed = preprocess_echonext_record(record=idx, data_dir=config.data_dir, split="val")
            
            wf = torch.tensor(val_waveforms[idx:idx+1], dtype=torch.float32)
            tab = torch.tensor(val_tabular[idx:idx+1], dtype=torch.float32)

            ranked = saliency.rank_segments_from_processed(processed, wf, tab)
            saliency_cache[idx] = ranked
            print(f"OK — {len(ranked)} segments ranked")

        except Exception as e:
            failed += 1
            print(f"FAILED: {e}")
            continue

    with open(config.saliency_cache_path, "wb") as f:
        pickle.dump(saliency_cache, f)
    print(f"\nSaliency cache saved to {config.saliency_cache_path}")
    print(f"  Contains rankings for {len(saliency_cache)} ECGs ({failed} failed)")

    return saliency_cache


def main():
    config = Phase3Config()
    
    # --- BLOCK 4: Build Normal Segment Library (or skip) ---
    if config.segment_library_csv.exists() and config.segment_library_npz.exists():
        print("Segment library already exists, skipping Block 4...")
        library = pd.read_csv(config.segment_library_csv)
    else:
        library = build_normal_library(config)

    # --- BLOCK 3: Compute Saliency Rankings ---
    print("\nLoading frozen model...")
    model = load_model(config)
    saliency_cache = compute_saliency_rankings(config, model)

    print("\n" + "=" * 60)
    print("Phase 3 complete!")
    print("=" * 60)
    print(f"  Segment library: {config.segment_library_csv}")
    print(f"  Saliency cache:  {config.saliency_cache_path}")


if __name__ == "__main__":
    main()  