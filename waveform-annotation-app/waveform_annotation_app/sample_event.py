from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple

import numpy as np

from .dispersion import DispersionImage
from .event_models import EventData, EventMetadata, TraceData, TraceMetadata


def _ricker_wavelet(points: int, a: float) -> np.ndarray:
    t = np.linspace(-1, 1, points)
    return (1 - 2 * (np.pi ** 2) * (a ** 2) * (t ** 2)) * np.exp(- (np.pi ** 2) * (a ** 2) * (t ** 2))


def generate_synthetic_event(
    num_stations: int = 6,
    sampling_rate: float = 100.0,
    duration_s: float = 120.0,
) -> EventData:
    """Create a synthetic event with P and S arrivals across stations."""
    npts = int(duration_s * sampling_rate)
    origin_time = datetime(2022, 7, 12, 3, 45, tzinfo=timezone.utc)
    base_lat, base_lon = 35.2, 102.1

    rng = np.random.default_rng(42)
    distances = np.linspace(20, 220, num_stations) + rng.uniform(-5, 5, num_stations)
    station_codes = [f"ST{idx:02d}" for idx in range(num_stations)]
    network_codes = ["XZ"] * num_stations

    traces = []
    for idx, (distance, sta, net) in enumerate(zip(distances, station_codes, network_codes)):
        time = np.arange(npts) / sampling_rate

        p_travel = 8.0 + distance / 6.0 + rng.normal(scale=0.2)
        s_travel = 14.0 + distance / 3.0 + rng.normal(scale=0.3)

        noise = rng.normal(scale=0.05, size=npts)
        p_wave = np.zeros_like(time)
        s_wave = np.zeros_like(time)

        p_center = int(p_travel * sampling_rate)
        s_center = int(s_travel * sampling_rate)

        wavelet = _ricker_wavelet(400, a=0.25)
        insert_length = len(wavelet)
        p_start = max(p_center - insert_length // 2, 0)
        s_start = max(s_center - insert_length // 2, 0)
        p_wave[p_start:p_start + insert_length] += 0.3 * wavelet[: npts - p_start]
        s_wave[s_start:s_start + insert_length] += 0.6 * wavelet[: npts - s_start]

        envelope = np.exp(-0.0005 * np.arange(npts))
        waveform = (noise + p_wave + s_wave) * envelope

        metadata = TraceMetadata(
            station_code=sta,
            network_code=net,
            distance_km=float(distance),
            sampling_rate=sampling_rate,
            start_time=origin_time,
        )
        traces.append(TraceData(samples=waveform.astype(np.float32), metadata=metadata))

    event_metadata = EventMetadata(
        event_id="demo-event",
        origin_time=origin_time,
        latitude=base_lat,
        longitude=base_lon,
        depth_km=12.0,
        magnitude=5.1,
    )
    return EventData(metadata=event_metadata, traces=traces)


def generate_demo_dispersion(
    num_periods: int = 120,
    num_velocities: int = 160,
    period_range: Tuple[float, float] = (5.0, 50.0),
    velocity_range: Tuple[float, float] = (2.0, 5.0),
) -> DispersionImage:
    """Generate a synthetic dispersion energy image for demonstration."""

    periods = np.linspace(period_range[0], period_range[1], num_periods)
    velocities = np.linspace(velocity_range[0], velocity_range[1], num_velocities)
    pp, vv = np.meshgrid(periods, velocities)

    fundamental = np.exp(-((pp - 20.0) ** 2) / 200.0) * np.exp(-((vv - 3.5) ** 2) / 0.08)
    first_higher = 0.6 * np.exp(-((pp - 30.0) ** 2) / 250.0) * np.exp(-((vv - 4.2) ** 2) / 0.05)
    noise = 0.05 * np.random.default_rng(1234).random(size=fundamental.shape)

    energy = fundamental + first_higher + noise
    return DispersionImage(periods=periods, velocities=velocities, energy=energy, label="demo-dispersion")
