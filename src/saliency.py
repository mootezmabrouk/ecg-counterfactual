"""Saliency Prior — Integrated Gradients for ECG segment ranking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

if TYPE_CHECKING:
    from .config import Phase3Config
    from .preprocessing import ProcessedECG


class _ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, tabular: torch.Tensor):
        super().__init__()
        self.model = model
        self._tabular = tabular  # stored as attribute, not register_buffer
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        batch_size = waveforms.shape[0]
        tabular_broadcast = self._tabular.expand(batch_size, -1)  # <-- THIS IS THE FIX
        return self.model((waveforms, tabular_broadcast))

class SaliencyPrior:
    """Compute IG attributions and rank segments by importance."""

    def __init__(self, model: nn.Module, config: Phase3Config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = model.to(device).eval()

        if not HAS_CAPTUM:
            raise ImportError("captum is required. Install: pip install captum")

    def compute_attributions(
        self,
        waveform: torch.Tensor,
        tabular: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        target_class = target_class or self.config.ig_target_class
        waveform = waveform.to(self.device)
        tabular = tabular.to(self.device)
        baseline = torch.zeros_like(waveform)

        wrapped = _ModelWrapper(self.model, tabular)
        ig = IntegratedGradients(wrapped)

        attributions, _ = ig.attribute(
            waveform,
            baselines=baseline,
            n_steps=self.config.ig_n_steps,
            target=target_class,
            return_convergence_delta=True,
        )
        return attributions.detach().cpu()

    def rank_segments_from_processed(
        self,
        processed: ProcessedECG,
        waveform: torch.Tensor,
        tabular: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> List[Dict]:
        attrs = self.compute_attributions(waveform, tabular, target_class)
        attrs_np = attrs.squeeze().numpy()  # (2500, 12)
        segments = []
        aggregation = self.config.saliency_aggregation

        for lead_index, lead_name in enumerate(processed.segments):
            lead_segments = processed.segments[lead_name]
            r_peaks = lead_segments.r_peaks

            for seg_type in ("p_wave", "qrs_complex", "st_segment", "t_wave"):
                intervals = getattr(lead_segments, seg_type)

                for beat_idx, (start, end) in enumerate(intervals):
                    start, end = int(start), int(end)
                    if end <= start or end > attrs_np.shape[0]:
                        continue

                    seg_attrs = attrs_np[start:end, lead_index]

                    if aggregation == "mean":
                        importance = float(np.mean(np.abs(seg_attrs)))
                    elif aggregation == "max":
                        importance = float(np.max(np.abs(seg_attrs)))
                    elif aggregation == "l1":
                        importance = float(np.sum(np.abs(seg_attrs)))
                    else:
                        raise ValueError(f"Unknown aggregation: {aggregation}")

                    r_peak = self._nearest_r_peak(start, end, seg_type, r_peaks)

                    segments.append({
                        "lead": lead_name,
                        "lead_index": lead_index,
                        "segment_type": seg_type,
                        "beat_index": beat_idx,
                        "start_sample": start,
                        "end_sample": end,
                        "duration_samples": end - start,
                        "importance": importance,
                        "r_peak_sample": int(r_peak) if r_peak is not None else None,
                    })

        return sorted(segments, key=lambda x: x["importance"], reverse=True)

    @staticmethod
    def _nearest_r_peak(start: int, end: int, segment_type: str, r_peaks: np.ndarray) -> Optional[int]:
        if len(r_peaks) == 0:
            return None
        if segment_type == "p_wave":
            candidates = np.flatnonzero(r_peaks >= end)
            return int(r_peaks[candidates[0]]) if len(candidates) else None
        elif segment_type in ("st_segment", "t_wave"):
            candidates = np.flatnonzero(r_peaks <= start)
            return int(r_peaks[candidates[-1]]) if len(candidates) else None
        else:
            center = (start + end) / 2
            return int(r_peaks[int(np.argmin(np.abs(r_peaks - center)))])