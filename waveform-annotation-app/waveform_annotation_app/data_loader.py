from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Iterable, Sequence

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency for HDF5 support.
    h5py = None

try:
    from obspy import read as obspy_read
except ImportError:  # pragma: no cover - optional dependency for SAC support.
    obspy_read = None

from .dispersion import DispersionImage
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


def _ensure_obspy() -> None:
    if obspy_read is None:
        raise ImportError(
            "obspy is required for reading SAC events. Install it via `pip install obspy`."
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


def load_dispersion_npz(file_obj: IO[bytes]) -> DispersionImage:
    """Load a dispersion image from a NumPy ``.npz`` archive."""

    with np.load(file_obj) as npz:
        if "energy" in npz:
            energy = np.asarray(npz["energy"], dtype=np.float32)
        elif "image" in npz:
            energy = np.asarray(npz["image"], dtype=np.float32)
        else:
            raise KeyError("Dispersion NPZ must contain an 'energy' or 'image' array.")

        if "periods" in npz:
            periods = np.asarray(npz["periods"], dtype=float)
        elif "frequencies" in npz:
            frequencies = np.asarray(npz["frequencies"], dtype=float)
            if np.any(frequencies <= 0):
                raise ValueError("Frequencies must be positive to convert to periods.")
            periods = 1.0 / frequencies
        else:
            raise KeyError("Dispersion NPZ must contain either 'periods' or 'frequencies'.")

        if "velocities" in npz:
            velocities = np.asarray(npz["velocities"], dtype=float)
        elif "velocity" in npz:
            velocities = np.asarray(npz["velocity"], dtype=float)
        else:
            raise KeyError("Dispersion NPZ must contain a 'velocities' vector.")

        label = str(npz["label"]) if "label" in npz else ""

    if energy.shape != (len(velocities), len(periods)):
        raise ValueError(
            "Dispersion image shape must match (len(velocities), len(periods))."
        )

    return DispersionImage(periods=periods, velocities=velocities, energy=energy, label=label)


def load_dispersion_hdf5(file_obj: IO[bytes]) -> DispersionImage:
    """Load a dispersion image from an HDF5 file."""

    _ensure_h5py()

    with h5py.File(file_obj, "r") as h5:
        if "dispersion" in h5:
            group = h5["dispersion"]
        else:
            group = h5

        if "energy" in group:
            energy = np.asarray(group["energy"], dtype=np.float32)
        elif "image" in group:
            energy = np.asarray(group["image"], dtype=np.float32)
        else:
            raise KeyError("HDF5 dispersion data must have an 'energy' dataset under 'dispersion'.")

        if "periods" in group:
            periods = np.asarray(group["periods"], dtype=float)
        elif "frequencies" in group:
            frequencies = np.asarray(group["frequencies"], dtype=float)
            if np.any(frequencies <= 0):
                raise ValueError("Frequencies must be positive to convert to periods.")
            periods = 1.0 / frequencies
        else:
            raise KeyError("HDF5 dispersion data must have a 'periods' or 'frequencies' dataset.")

        if "velocities" in group:
            velocities = np.asarray(group["velocities"], dtype=float)
        else:
            raise KeyError("HDF5 dispersion data must have a 'velocities' dataset.")

        label = str(group.attrs.get("label", ""))

    if energy.shape != (len(velocities), len(periods)):
        raise ValueError(
            "Dispersion image shape must match (len(velocities), len(periods))."
        )

    return DispersionImage(periods=periods, velocities=velocities, energy=energy, label=label)


def _valid_sac_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return False
        return float(value) not in {-12345.0, -12345}
    if isinstance(value, (int, np.integer)):
        return int(value) not in {-12345}
    if isinstance(value, str):
        stripped = value.strip()
        return stripped not in {"", "-12345"}
    return True


def _ensure_datetime_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_event_metadata_from_sac(stats) -> EventMetadata:
    sac = getattr(stats, "sac", None)
    start_time = _ensure_datetime_utc(stats.starttime.datetime)
    origin_time = start_time
    if sac is not None and _valid_sac_value(getattr(sac, "o", None)):
        origin_time = start_time + timedelta(seconds=float(sac.o))

    event_id = "sac-event"
    latitude = 0.0
    longitude = 0.0
    depth_km = 10.0
    magnitude = 4.0
    event_type = "unspecified"

    if sac is not None:
        if _valid_sac_value(getattr(sac, "kevnm", None)):
            candidate_id = str(sac.kevnm).strip()
            if candidate_id:
                event_id = candidate_id
        if _valid_sac_value(getattr(sac, "evla", None)):
            latitude = float(sac.evla)
        if _valid_sac_value(getattr(sac, "evlo", None)):
            longitude = float(sac.evlo)
        if _valid_sac_value(getattr(sac, "evdp", None)):
            depth_km = float(sac.evdp)
            if abs(depth_km) > 1000:  # convert metres to kilometres if necessary
                depth_km /= 1000.0
        if _valid_sac_value(getattr(sac, "mag", None)):
            magnitude = float(sac.mag)
        if _valid_sac_value(getattr(sac, "ictype", None)):
            event_type = str(sac.ictype).strip().lower()

    return EventMetadata(
        event_id=event_id,
        origin_time=origin_time,
        latitude=latitude,
        longitude=longitude,
        depth_km=depth_km,
        magnitude=magnitude,
        event_type=event_type,
    )


def _trace_metadata_from_sac(stats) -> TraceMetadata:
    sac = getattr(stats, "sac", None)
    start_time = _ensure_datetime_utc(stats.starttime.datetime)
    sampling_rate = float(stats.sampling_rate)
    distance_km = 0.0
    extras = {}

    if sac is not None:
        if _valid_sac_value(getattr(sac, "dist", None)):
            distance_km = float(sac.dist)
        elif _valid_sac_value(getattr(sac, "gcarc", None)):
            distance_km = float(sac.gcarc) * 111.19
        if _valid_sac_value(getattr(sac, "az", None)):
            extras["azimuth"] = float(sac.az)
        if _valid_sac_value(getattr(sac, "baz", None)):
            extras["back_azimuth"] = float(sac.baz)

    return TraceMetadata(
        station_code=getattr(stats, "station", "STA"),
        network_code=getattr(stats, "network", "NET"),
        distance_km=distance_km,
        sampling_rate=sampling_rate,
        start_time=start_time,
        channel=getattr(stats, "channel", "BHZ"),
        location_code=getattr(stats, "location", ""),
        extras=extras,
    )


def load_sac_event(file_objs: Iterable[IO[bytes]] | IO[bytes]) -> EventData:
    """Load an :class:`EventData` instance from one or more SAC files."""

    _ensure_obspy()

    if isinstance(file_objs, (io.IOBase, bytes, bytearray)):
        file_sequence: Sequence[IO[bytes]] = [file_objs]  # type: ignore[assignment]
    else:
        file_sequence = list(file_objs)  # type: ignore[arg-type]

    traces: list[TraceData] = []
    event_metadata: EventMetadata | None = None

    for file_obj in file_sequence:
        if isinstance(file_obj, (bytes, bytearray)):
            buffer = io.BytesIO(file_obj)
        else:
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            buffer = io.BytesIO(file_obj.read())
        buffer.seek(0)
        stream = obspy_read(buffer, format="SAC")
        for tr in stream:
            samples = np.asarray(tr.data, dtype=np.float32)
            metadata = _trace_metadata_from_sac(tr.stats)
            traces.append(TraceData(samples=samples, metadata=metadata))
            if event_metadata is None:
                event_metadata = _extract_event_metadata_from_sac(tr.stats)

    if not traces:
        raise ValueError("No SAC traces were loaded from the provided files.")

    if event_metadata is None:
        first_trace = traces[0]
        event_metadata = EventMetadata(
            event_id="sac-event",
            origin_time=first_trace.metadata.start_time,
            latitude=0.0,
            longitude=0.0,
            depth_km=10.0,
            magnitude=4.0,
        )

    return EventData(metadata=event_metadata, traces=traces)


def load_demo_event() -> EventData:
    """Generate a synthetic event with realistic looking waveforms."""
    from .sample_event import generate_synthetic_event

    return generate_synthetic_event()


def load_demo_dispersion() -> DispersionImage:
    """Return a synthetic dispersion image for demonstration purposes."""

    from .sample_event import generate_demo_dispersion

    return generate_demo_dispersion()


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
