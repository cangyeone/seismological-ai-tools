from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt

from .event_models import TraceData


@dataclass
class FilterSettings:
    mode: str = "none"  # 'none', 'bandpass', 'lowpass', 'highpass'
    lowcut: Optional[float] = None
    highcut: Optional[float] = None
    order: int = 4

    def description(self) -> str:
        if self.mode == "bandpass" and self.lowcut and self.highcut:
            return f"Band-pass {self.lowcut:g}-{self.highcut:g} Hz"
        if self.mode == "highpass" and self.lowcut:
            return f"High-pass {self.lowcut:g} Hz"
        if self.mode == "lowpass" and self.highcut:
            return f"Low-pass {self.highcut:g} Hz"
        return "No filtering"


def _butterworth_filter(data: np.ndarray, fs: float, settings: FilterSettings) -> np.ndarray:
    nyquist = 0.5 * fs
    if settings.mode == "bandpass" and settings.lowcut and settings.highcut:
        low = settings.lowcut / nyquist
        high = settings.highcut / nyquist
        b, a = butter(settings.order, [low, high], btype="bandpass")
    elif settings.mode == "highpass" and settings.lowcut:
        low = settings.lowcut / nyquist
        b, a = butter(settings.order, low, btype="highpass")
    elif settings.mode == "lowpass" and settings.highcut:
        high = settings.highcut / nyquist
        b, a = butter(settings.order, high, btype="lowpass")
    else:
        return data
    return filtfilt(b, a, data)


def apply_filter(trace: TraceData, settings: FilterSettings) -> TraceData:
    filtered_samples = _butterworth_filter(trace.samples, trace.metadata.sampling_rate, settings)
    return trace.copy_with_samples(filtered_samples.astype(np.float32))
