"""Streamlit application for seismic waveform annotation."""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from typing import Dict, List

import pandas as pd
import streamlit as st

from waveform_annotation_app.annotation_state import (
    TraceAnnotation,
    annotations_to_dataframe,
    dataframe_to_annotations,
)
from waveform_annotation_app.data_loader import (
    load_demo_event,
    load_hdf5_event,
    load_npz_event,
    load_sac_event,
)
from waveform_annotation_app.dnn_prelabel import DNNPreLabeler
from waveform_annotation_app.event_models import EventData, TraceData
from waveform_annotation_app.filtering import FilterSettings, apply_filter
from waveform_annotation_app.plotting import create_waveform_figure

st.set_page_config(page_title="Seismic Waveform Annotation", layout="wide")


def _initialise_session_state() -> None:
    if "event" not in st.session_state:
        st.session_state.event = load_demo_event()
    if "annotations" not in st.session_state:
        st.session_state.annotations = {
            f"{tr.metadata.network_code}.{tr.metadata.station_code}": TraceAnnotation(
                station_code=tr.metadata.station_code,
                network_code=tr.metadata.network_code,
                distance_km=tr.metadata.distance_km,
            )
            for tr in st.session_state.event.traces
        }
    if "phase_picker" not in st.session_state:
        st.session_state.phase_picker = DNNPreLabeler.create()
    if "filter_settings" not in st.session_state:
        st.session_state.filter_settings = FilterSettings()
    if "vertical_scale" not in st.session_state:
        st.session_state.vertical_scale = 1.0


def _update_event(event: EventData) -> None:
    st.session_state.event = event
    st.session_state.annotations = {
        f"{tr.metadata.network_code}.{tr.metadata.station_code}": TraceAnnotation(
            station_code=tr.metadata.station_code,
            network_code=tr.metadata.network_code,
            distance_km=tr.metadata.distance_km,
        )
        for tr in event.traces
    }


def _sidebar_controls() -> None:
    st.sidebar.header("Data selection")
    source = st.sidebar.radio(
        "Choose dataset",
        ["Demo event", "Upload NPZ", "Upload HDF5", "Upload SAC"],
        key="dataset_source",
    )

    if source == "Demo event":
        if st.sidebar.button("Reload demo"):
            _update_event(load_demo_event())
    elif source == "Upload NPZ":
        uploaded = st.sidebar.file_uploader("Upload .npz file", type="npz")
        if uploaded is not None:
            bytes_buffer = io.BytesIO(uploaded.getvalue())
            event = load_npz_event(bytes_buffer)
            _update_event(event)
    elif source == "Upload HDF5":
        uploaded = st.sidebar.file_uploader("Upload .h5 or .hdf5 file", type=["h5", "hdf5"])
        if uploaded is not None:
            bytes_buffer = io.BytesIO(uploaded.getvalue())
            event = load_hdf5_event(bytes_buffer)
            _update_event(event)
    else:
        uploaded_files = st.sidebar.file_uploader(
            "Upload SAC file(s)",
            type=["sac", "SAC"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            buffers = [io.BytesIO(file.getvalue()) for file in uploaded_files]
            event = load_sac_event(buffers)
            _update_event(event)

    st.sidebar.header("Filtering")
    filter_mode = st.sidebar.selectbox(
        "Filter mode",
        options=["none", "bandpass", "highpass", "lowpass"],
        index=["none", "bandpass", "highpass", "lowpass"].index(st.session_state.filter_settings.mode),
    )
    lowcut = st.sidebar.number_input("Low cut (Hz)", min_value=0.1, max_value=40.0, value=1.0)
    highcut = st.sidebar.number_input("High cut (Hz)", min_value=0.5, max_value=80.0, value=20.0)
    order = st.sidebar.slider("Filter order", min_value=2, max_value=8, value=4)
    st.session_state.filter_settings = FilterSettings(
        mode=filter_mode,
        lowcut=lowcut if filter_mode in {"bandpass", "highpass"} else None,
        highcut=highcut if filter_mode in {"bandpass", "lowpass"} else None,
        order=order,
    )

    st.sidebar.header("Visualisation")
    st.session_state.vertical_scale = st.sidebar.slider("Amplitude scale", min_value=0.1, max_value=5.0, value=st.session_state.vertical_scale)
    st.sidebar.checkbox("Show DNN annotations", value=True, key="show_annotations")


def _prepare_annotations_dataframe(event: EventData) -> pd.DataFrame:
    records: List[TraceAnnotation] = []
    for trace in event.sorted_traces():
        key = f"{trace.metadata.network_code}.{trace.metadata.station_code}"
        annotation = st.session_state.annotations.get(key)
        if annotation is None:
            annotation = TraceAnnotation(
                station_code=trace.metadata.station_code,
                network_code=trace.metadata.network_code,
                distance_km=trace.metadata.distance_km,
            )
            st.session_state.annotations[key] = annotation
        records.append(annotation)
    df = annotations_to_dataframe(records)
    return df


def _update_annotations_from_dataframe(df: pd.DataFrame) -> None:
    annotations = dataframe_to_annotations(df)
    st.session_state.annotations = {
        f"{ann.network_code}.{ann.station_code}": ann
        for ann in annotations
    }


def _run_prelabeler(event: EventData) -> None:
    picker = st.session_state.phase_picker
    suggestions = picker.annotate_event(event)
    updated = st.session_state.annotations.copy()
    updated.update(suggestions)
    st.session_state.annotations = updated
    st.success("DNN pre-annotation completed. Suggestions inserted into the table.")


def _export_annotations(event: EventData) -> str:
    payload = {
        "event": {
            "metadata": {
                "event_id": event.metadata.event_id,
                "origin_time": event.metadata.origin_time.isoformat(),
                "latitude": event.metadata.latitude,
                "longitude": event.metadata.longitude,
                "depth_km": event.metadata.depth_km,
                "magnitude": event.metadata.magnitude,
                "event_type": event.metadata.event_type,
            },
            "traces": [
                {
                    "station_code": tr.metadata.station_code,
                    "network_code": tr.metadata.network_code,
                    "distance_km": tr.metadata.distance_km,
                    "sampling_rate": tr.metadata.sampling_rate,
                }
                for tr in event.traces
            ],
        },
        "annotations": [
            {
                "station": key,
                **asdict(annotation),
            }
            for key, annotation in st.session_state.annotations.items()
        ],
    }
    return json.dumps(payload, indent=2)


def main() -> None:
    _initialise_session_state()
    _sidebar_controls()
    event: EventData = st.session_state.event

    st.title("Seismic Waveform Annotation Workbench")
    st.caption("Visualise, filter, and annotate multi-station earthquake records with DNN-assisted picks.")

    col1, col2 = st.columns([2, 1])
    with col1:
        filtered_traces: List[TraceData] = []
        for trace in event.sorted_traces():
            filtered_traces.append(apply_filter(trace, st.session_state.filter_settings))
        annotations_dict = st.session_state.annotations
        fig = create_waveform_figure(
            filtered_traces,
            annotations=annotations_dict,
            vertical_scaling=st.session_state.vertical_scale,
            show_annotations=st.session_state.show_annotations,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Event metadata")
        md = event.metadata
        st.markdown(
            f"**ID:** {md.event_id}<br>"
            f"**Origin:** {md.origin_time.isoformat()}<br>"
            f"**Location:** {md.latitude:.2f}, {md.longitude:.2f}<br>"
            f"**Depth:** {md.depth_km:.1f} km<br>"
            f"**Magnitude:** {md.magnitude:.1f}",
            unsafe_allow_html=True,
        )
        new_type = st.selectbox(
            "Event type",
            options=["unspecified", "earthquake", "explosion", "quarry blast", "noise"],
            index=["unspecified", "earthquake", "explosion", "quarry blast", "noise"].index(md.event_type if md.event_type in {"unspecified", "earthquake", "explosion", "quarry blast", "noise"} else "unspecified"),
        )
        event.metadata.event_type = new_type

        st.markdown("---")
        if st.button("Run DNN pre-annotation"):
            _run_prelabeler(event)

        annotation_df = _prepare_annotations_dataframe(event)
        edited_df = st.data_editor(
            annotation_df,
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            key="annotation_editor",
        )
        _update_annotations_from_dataframe(edited_df)

        export_json = _export_annotations(event)
        st.download_button("Download annotations", export_json, file_name=f"{event.metadata.event_id}_annotations.json")

    st.markdown("---")
    st.info("Zoom and pan directly within the waveform plots to focus on specific arrivals. Use the DNN suggestions as a starting point and refine manually as needed.")


if __name__ == "__main__":
    main()
