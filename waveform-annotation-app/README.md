# Waveform Annotation App

This Streamlit application provides an interactive environment for annotating
seismic waveforms from individual earthquakes. It supports displaying multiple
station traces for the same event, sorting by epicentral distance, filtering,
zooming, and creating manual annotations for seismic phases and event types.
A lightweight convolutional neural network (DNN) is bundled with the app to
provide automatic pre-annotation suggestions that can be reviewed and edited by
an analyst.

## Features

- Load synthetic demonstration data or upload your own NumPy ``.npz``, HDF5,
  or SAC files containing waveform arrays and metadata.
- Visualise multiple station waveforms for a single event in a Plotly figure
  with pan/zoom support and distance-based stacking.
- Apply common signal-processing filters (band-pass, high-pass, low-pass) using
  an interactive sidebar.
- Capture manual annotations for P and S arrivals, signal quality, and overall
  event type directly within the interface.
- Run the bundled DNN to generate phase-pick suggestions that pre-populate the
  annotation table for rapid review.
- Export the curated annotations as JSON for down-stream processing.

## Getting Started

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Open the provided URL in your browser. Use the sidebar controls to choose
   between the synthetic demo event or your own uploads, adjust filtering
   options, and trigger DNN pre-annotation. The main panel shows the stacked
   waveforms and the annotation editor.

### Preparing custom data

The upload option supports three container formats:

- ``.npz`` archives with at least the following arrays:

  - ``waveforms`` – ``(N, T)`` float32 array of waveform samples for ``N``
    stations and ``T`` time samples.
  - ``station_codes`` – ``(N,)`` array of station codes.
  - ``network_codes`` – ``(N,)`` array of network identifiers.
  - ``distances_km`` – ``(N,)`` array of epicentral distances in kilometres.
  - ``sampling_rate`` – Scalar sampling rate in Hz.
  - Optional metadata fields (``origin_time``, ``event_id``, ``latitude``,
    ``longitude``, ``depth_km``, ``magnitude``) can also be provided as
    single-valued arrays.

- HDF5 files with datasets matching the names above and event-level attributes
  describing the earthquake. Attributes such as ``origin_time`` (ISO8601),
  ``event_id``, ``latitude``, ``longitude``, ``depth_km``, ``magnitude``,
  ``sampling_rate``, and ``event_type`` are read directly from the root group.
  Additional optional datasets such as ``channels`` and ``location_codes`` are
  used when present.

- One or more SAC waveforms for the same event. The loader reads the SAC header
  information to populate event metadata (e.g., ``kevnm``, ``evla``, ``evlo``,
  ``evdp``, ``mag``) and station-specific details including distance, azimuth,
  and sample rate. Upload multiple SAC files at once to build a multi-station
  event.

See ``data_loader.py`` for additional details on the expected formats and how
the files are parsed.

## Repository layout

```
waveform-annotation-app/
├── README.md
├── requirements.txt
├── app.py
└── waveform_annotation_app/
    ├── __init__.py
    ├── annotation_state.py
    ├── data_loader.py
    ├── dnn_prelabel.py
    ├── event_models.py
    ├── filtering.py
    ├── plotting.py
    └── sample_event.py
```

## License

This tool inherits the Apache-2.0 license from the root of the repository.
