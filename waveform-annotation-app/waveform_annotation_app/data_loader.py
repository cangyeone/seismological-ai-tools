from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Dict, Optional

import numpy as np

from .event_models import EventData, EventMetadata, TraceData, TraceMetadata


def _default_origin() -> datetime:
    return datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def load_npz_event(file_obj: IO[bytes]) -> EventData:
    """Load an :class:`EventData` instance from a NumPy ``.npz`` archive."""
    with np.load(file_obj) as npz:
        waveforms = npz["waveforms"].astype(np.float32)
        station_codes = npz["station_codes"].astype(str)
        network_codes = npz["network_codes"].astype(str)
        distances = npz["distances_km"].astype(float)
        sampling_rate = float(npz["sampling_rate"]) if "sampling_rate" in npz else 100.0

        origin_time = datetime.fromisoformat(npz["origin_time"].item()) if "origin_time" in npz else _default_origin()
        event_id = str(npz["event_id"].item()) if "event_id" in npz else "npz-event"
        latitude = float(npz["latitude"].item()) if "latitude" in npz else 0.0
        longitude = float(npz["longitude"].item()) if "longitude" in npz else 0.0
        depth_km = float(npz["depth_km"].item()) if "depth_km" in npz else 10.0
        magnitude = float(npz["magnitude"].item()) if "magnitude" in npz else 4.0

    traces = []
    for samples, sta, net, dist in zip(waveforms, station_codes, network_codes, distances):
        metadata = TraceMetadata(
            station_code=sta,
            network_code=net,
            distance_km=float(dist),
            sampling_rate=sampling_rate,
            start_time=origin_time,
        )
        traces.append(TraceData(samples=samples, metadata=metadata))

    event_metadata = EventMetadata(
        event_id=event_id,
        origin_time=origin_time,
        latitude=latitude,
        longitude=longitude,
        depth_km=depth_km,
        magnitude=magnitude,
    )
    return EventData(metadata=event_metadata, traces=traces)


def load_demo_event() -> EventData:
    """Generate a synthetic event with realistic looking waveforms."""
    from .sample_event import generate_synthetic_event

    return generate_synthetic_event()


def save_event_to_npz(event: EventData, path: Path) -> None:
    """Utility function to persist an event in the ``.npz`` format."""
    waveforms = np.stack([trace.samples for trace in event.traces])
    station_codes = np.array([trace.metadata.station_code for trace in event.traces])
    network_codes = np.array([trace.metadata.network_code for trace in event.traces])
    distances = np.array([trace.metadata.distance_km for trace in event.traces])

    np.savez(
        path,
        waveforms=waveforms,
        station_codes=station_codes,
        network_codes=network_codes,
        distances_km=distances,
        sampling_rate=event.traces[0].metadata.sampling_rate,
        origin_time=event.metadata.origin_time.isoformat(),
        event_id=event.metadata.event_id,
        latitude=event.metadata.latitude,
        longitude=event.metadata.longitude,
        depth_km=event.metadata.depth_km,
        magnitude=event.metadata.magnitude,
    )
