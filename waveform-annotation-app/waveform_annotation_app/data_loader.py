from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import IO

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency for HDF5 support.
    h5py = None

from .event_models import (
    EventData,
    EventMetadata,
    TraceData,
    TraceMetadata,
    seconds_to_datetime,
)


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


def _ensure_h5py() -> None:
    if h5py is None:
        raise ImportError(
            "h5py is required for reading HDF5 events. Install it via `pip install h5py`."
        )


def _decode_attr(attrs: "h5py.AttributeManager", key: str, default):
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O"} and value.size == 1:
        return value[0].decode("utf-8") if isinstance(value[0], (bytes, bytearray)) else value[0]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _read_string_dataset(dataset: "h5py.Dataset") -> np.ndarray:
    data = np.asarray(dataset)
    if data.dtype.kind in {"S", "O"}:
        return np.vectorize(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x))(data)
    return data.astype(str)


def load_hdf5_event(file_obj: IO[bytes]) -> EventData:
    """Load an :class:`EventData` instance from an HDF5 file.

    The reader expects datasets named ``waveforms`` (``n_traces`` × ``n_samples``),
    ``station_codes``, ``network_codes``, and ``distances_km``. Event-level metadata
    such as latitude, longitude, depth, magnitude, origin time, and event id are
    read from HDF5 attributes on the root group.
    """

    _ensure_h5py()

    with h5py.File(file_obj, "r") as h5:
        if "waveforms" not in h5:
            raise KeyError("HDF5 file must contain a 'waveforms' dataset")
        waveforms = np.asarray(h5["waveforms"], dtype=np.float32)

        station_codes = (
            _read_string_dataset(h5["station_codes"]) if "station_codes" in h5 else np.array(["STA"] * waveforms.shape[0])
        )
        network_codes = (
            _read_string_dataset(h5["network_codes"]) if "network_codes" in h5 else np.array(["NET"] * waveforms.shape[0])
        )
        distances = (
            np.asarray(h5["distances_km"], dtype=float)
            if "distances_km" in h5
            else np.linspace(10.0, 200.0, waveforms.shape[0])
        )
        sampling_rate = float(_decode_attr(h5.attrs, "sampling_rate", 100.0))
        start_offset = float(_decode_attr(h5.attrs, "start_offset", 0.0))

        origin_attr = _decode_attr(h5.attrs, "origin_time", None)
        origin_time = (
            datetime.fromisoformat(str(origin_attr))
            if origin_attr is not None
            else _default_origin()
        )

        event_id = str(_decode_attr(h5.attrs, "event_id", "hdf5-event"))
        latitude = float(_decode_attr(h5.attrs, "latitude", 0.0))
        longitude = float(_decode_attr(h5.attrs, "longitude", 0.0))
        depth_km = float(_decode_attr(h5.attrs, "depth_km", 10.0))
        magnitude = float(_decode_attr(h5.attrs, "magnitude", 4.0))
        event_type = str(_decode_attr(h5.attrs, "event_type", "unspecified"))

        channels = (
            _read_string_dataset(h5["channels"]) if "channels" in h5 else np.array(["BHZ"] * waveforms.shape[0])
        )
        locations = (
            _read_string_dataset(h5["location_codes"]) if "location_codes" in h5 else np.array([""] * waveforms.shape[0])
        )

    traces: list[TraceData] = []
    for samples, sta, net, dist, chan, loc in zip(
        waveforms, station_codes, network_codes, distances, channels, locations
    ):
        metadata = TraceMetadata(
            station_code=str(sta),
            network_code=str(net),
            distance_km=float(dist),
            sampling_rate=sampling_rate,
            start_time=seconds_to_datetime(start_offset, origin_time),
            channel=str(chan),
            location_code=str(loc),
        )
        traces.append(TraceData(samples=np.asarray(samples, dtype=np.float32), metadata=metadata))

    event_metadata = EventMetadata(
        event_id=event_id,
        origin_time=origin_time,
        latitude=latitude,
        longitude=longitude,
        depth_km=depth_km,
        magnitude=magnitude,
        event_type=event_type,
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
