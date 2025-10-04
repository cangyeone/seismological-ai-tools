from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .annotation_state import TraceAnnotation
from .event_models import EventData, TraceData


class SimplePhaseNet(nn.Module):
    """A lightweight convolutional network for phase probability estimation."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=7, padding=3, bias=False)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(8, 2, kernel_size=1)
        self._initialise_weights()

    def _initialise_weights(self) -> None:
        with torch.no_grad():
            gradient_kernel = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, -0.5])
            average_kernel = torch.ones(7) / 7.0
            envelope_kernel = torch.tensor([-1.0, 2.0, -1.0, -1.0, 2.0, -1.0, 0.0])
            self.conv1.weight.zero_()
            self.conv1.weight[0, 0] = gradient_kernel
            self.conv1.weight[1, 0] = average_kernel
            self.conv1.weight[2, 0] = envelope_kernel
            self.conv1.weight[3, 0] = torch.flip(gradient_kernel, dims=(0,))

            nn.init.xavier_uniform_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)
            nn.init.zeros_(self.conv3.weight)
            nn.init.zeros_(self.conv3.bias)

            # Encourage the final layer to focus on gradient magnitude for P and energy for S.
            self.conv3.weight[0, 0, 0] = 1.5  # Gradient-informed channel
            self.conv3.weight[0, 1, 0] = 0.5  # Average energy
            self.conv3.weight[1, 1, 0] = 1.2  # Energy-focused channel
            self.conv3.weight[1, 2, 0] = 0.3  # Envelope variation
            self.conv3.bias[1] = -0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        return x

    def predict_probabilities(self, waveform: np.ndarray) -> np.ndarray:
        device = torch.device("cpu")
        tensor = torch.from_numpy(waveform.astype(np.float32)).view(1, 1, -1)
        logits = self.forward(tensor.to(device))
        logits = logits.squeeze(0).permute(1, 0)  # (T, 2)
        probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()


def _normalise_waveform(samples: np.ndarray) -> np.ndarray:
    samples = samples.astype(np.float32)
    mean = samples.mean()
    std = samples.std() + 1e-6
    return (samples - mean) / std


def _pick_times_from_probabilities(probabilities: np.ndarray, sampling_rate: float) -> Dict[str, float]:
    picks: Dict[str, float] = {}
    window = max(int(0.5 * sampling_rate), 1)
    p_channel = probabilities[:, 0]
    s_channel = probabilities[:, 1]

    if p_channel.size:
        idx = np.argmax(p_channel)
        picks["p_pick_s"] = idx / sampling_rate
        picks["phase_confidence"] = float(np.max(p_channel))
    if s_channel.size:
        idx = np.argmax(s_channel)
        picks["s_pick_s"] = idx / sampling_rate
    return picks


@dataclass
class DNNPreLabeler:
    model: SimplePhaseNet

    @classmethod
    def create(cls) -> "DNNPreLabeler":
        model = SimplePhaseNet()
        model.eval()
        return cls(model=model)

    def annotate_trace(self, trace: TraceData) -> TraceAnnotation:
        normalised = _normalise_waveform(trace.samples)
        probabilities = self.model.predict_probabilities(normalised)
        picks = _pick_times_from_probabilities(probabilities, trace.metadata.sampling_rate)
        return TraceAnnotation(
            station_code=trace.metadata.station_code,
            network_code=trace.metadata.network_code,
            distance_km=trace.metadata.distance_km,
            p_pick_s=picks.get("p_pick_s"),
            s_pick_s=picks.get("s_pick_s"),
            phase_confidence=picks.get("phase_confidence"),
        )

    def annotate_event(self, event: EventData) -> Dict[str, TraceAnnotation]:
        results: Dict[str, TraceAnnotation] = {}
        for trace in event.traces:
            annotation = self.annotate_trace(trace)
            key = f"{annotation.network_code}.{annotation.station_code}"
            results[key] = annotation
        return results
