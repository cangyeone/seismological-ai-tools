from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TraceMetadata:
    """Metadata for a single station trace."""

    station_code: str
    network_code: str
    distance_km: float
    sampling_rate: float
    start_time: datetime
    channel: str = "BHZ"
    location_code: str = ""
    extras: Dict[str, float] = field(default_factory=dict)

    def time_axis(self, npts: int) -> np.ndarray:
        """Return a relative time axis in seconds."""
        return np.arange(npts, dtype=float) / self.sampling_rate


@dataclass
class TraceData:
    """Waveform container holding the samples and accompanying metadata."""

    samples: np.ndarray
    metadata: TraceMetadata

    def copy_with_samples(self, samples: np.ndarray) -> "TraceData":
        return TraceData(samples=samples, metadata=self.metadata)


@dataclass
class EventMetadata:
    """Describes the earthquake level metadata."""

    event_id: str
    origin_time: datetime
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float
    event_type: str = "unspecified"


@dataclass
class EventData:
    """Collects traces belonging to the same event."""

    metadata: EventMetadata
    traces: List[TraceData]

    def sorted_traces(self) -> List[TraceData]:
        """Return traces sorted by increasing epicentral distance."""
        return sorted(self.traces, key=lambda tr: tr.metadata.distance_km)

    def copy_with_event_type(self, event_type: str) -> "EventData":
        updated = EventMetadata(
            event_id=self.metadata.event_id,
            origin_time=self.metadata.origin_time,
            latitude=self.metadata.latitude,
            longitude=self.metadata.longitude,
            depth_km=self.metadata.depth_km,
            magnitude=self.metadata.magnitude,
            event_type=event_type,
        )
        return EventData(metadata=updated, traces=self.traces)


def seconds_to_datetime(seconds: float, ref: datetime) -> datetime:
    return ref + timedelta(seconds=float(seconds))
