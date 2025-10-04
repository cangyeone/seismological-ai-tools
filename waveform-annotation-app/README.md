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
- Annotate surface-wave dispersion images by clicking to build 2-D dispersion
  branches, editing them in a tabular view, and exporting the picks alongside
  the energy map metadata.

## Detailed feature documentation

### Data ingestion workflow

The sidebar contains a **Data source** selector that determines how an event is
initialised. Each option is described in detail below so that custom datasets
can be prepared consistently.

#### Demo events

- Generates a synthetic but realistic single-earthquake package with multiple
  stations, origin metadata, and dispersion energy so you can explore the user
  interface without preparing external files.
- Includes pre-computed epicentral distances, instrument responses, and P/S
  arrival labels to showcase the full workflow.

#### NumPy ``.npz`` uploads

- Accepts a single archive containing waveform arrays and metadata. Required
  keys are ``waveforms`` (``N`` traces × ``T`` samples), ``station_codes``,
  ``network_codes``, ``distances_km``, and ``sampling_rate``.
- Optional scalar arrays such as ``origin_time`` (ISO8601 string), ``latitude``,
  ``longitude``, ``depth_km``, ``magnitude``, and ``event_type`` populate the
  event header.
- Additional convenience fields like ``channels`` or ``location_codes`` are
  honoured when present.

#### HDF5 uploads

- Reads datasets that mirror the ``.npz`` layout and relies on file attributes
  for event-level metadata. Attributes with names matching
  ``origin_time``, ``event_id``, ``latitude``, ``longitude``, ``depth_km``,
  ``magnitude``, ``event_type``, and ``sampling_rate`` are recognised.
- Child datasets such as ``waveforms`` and ``distances_km`` are loaded from the
  root group by default, but nested groups can be referenced through attributes
  like ``data_path`` if needed (see ``data_loader.py`` for examples).
- Supports station-specific attributes when the datasets expose compound
  dtypes. When absent, the loader falls back to per-dataset arrays.

#### SAC waveform collections

- Accepts one or more SAC files describing the same earthquake. ObsPy is used
  to parse the SAC headers and populate both event and station metadata.
- Event information (ID, origin time, hypocentre, magnitude) is extracted from
  SAC header fields such as ``kevnm``, ``nzyear``/``nzjday``/``nzhour``, ``evla``
  and ``evlo``.
- Station metadata such as the sampling rate (``delta``), epicentral distance
  (``dist``), azimuth (``az``), back-azimuth (``baz``), and component code are
  preserved for display within the waveform table.
- Multiple SAC files can be uploaded simultaneously; the loader automatically
  groups them into a single ``EventData`` instance.

#### Dispersion energy uploads

- The **Dispersion** tab has dedicated controls for importing 2-D dispersion
  maps. ``.npz`` archives should contain ``energy`` (or ``image``), ``periods``
  or ``frequencies`` (Hz), and ``velocities`` arrays. ``frequencies`` are
  converted to periods internally.
- HDF5 files may store these datasets either at the root level or within a
  ``dispersion`` group. Optional attributes like ``label`` and
  ``velocity_units`` are carried through to the annotation export.
- When no external dispersion file is provided, the demo generator supplies a
  synthetic image so you can experiment with the branch-picking workflow.

### Waveform review panel

- Displays every loaded trace stacked vertically and sorted by epicentral
  distance. The layout makes it easy to compare arrival times across the
  network.
- Plotly rendering supports pan, zoom, and hover-to-inspect interactions. Use
  your mouse wheel or the toolbar controls to zoom in around arrivals of
  interest.
- The **Vertical scale** slider in the sidebar controls per-trace amplitude
  scaling. Increasing the value reduces the spacing to highlight subtle phases;
  decreasing it separates traces for busy datasets.
- Toggle **Show picks** to overlay the current manual or DNN-assisted P/S
  annotations directly on the traces.
- A tabular summary of event metadata (ID, origin, hypocentre, magnitude, and
  type) appears alongside the plot and can be updated in-place.

### Signal processing controls

- The sidebar offers high-pass, low-pass, and band-pass filtering options. Each
  filter is implemented with a fourth-order Butterworth design using SciPy.
- Selecting **Band-pass** exposes lower/upper corner sliders whose ranges adapt
  to the loaded sampling rate. The app prevents invalid combinations to avoid
  runtime filter errors.
- Filters are applied on the fly whenever you adjust the configuration. The
  raw data remain untouched so you can revert to the original view instantly by
  switching back to **None**.

### Phase and quality annotation table

- Every trace is listed with editable fields for P-pick (seconds), S-pick
  (seconds), phase confidence (0–1), quality flag, and free-form comments.
- Edits performed in the data editor immediately sync back to the session state
  and are reflected in the plot overlays.
- New rows can be appended manually to record picks for additional channels, and
  unwanted entries can be removed with the built-in delete controls.
- Use the **Event type** drop-down to categorise the earthquake (e.g.,
  ``earthquake`` vs. ``explosion``). This value is included in the JSON export.

### DNN pre-annotation

- The **Run DNN pre-annotation** button executes a lightweight convolutional
  network (`SimplePhaseNet`) packaged with the app. The model produces
  probability curves for P and S arrivals for each trace.
- Peak probability indices are converted to pick times and merged into the
  annotation table. Existing manual edits are preserved unless a DNN suggestion
  has higher confidence for the same station.
- Because the model runs entirely on CPU and is intentionally compact, it can
  serve as a baseline for rapid triage before manual refinement. Advanced users
  can replace the network weights by editing ``dnn_prelabel.py``.

### Dispersion branch editor

- The **Dispersion** tab visualises the loaded energy map using a perceptually
  uniform colour scale, normalising amplitudes to [0, 1] for consistent display.
- Select or type a branch label, then click points on the image to add
  dispersion picks. Each click records the period (x-axis), velocity (y-axis),
  branch name, sequential order, and default weight.
- A dedicated table lists all picks, allowing you to edit values, adjust order,
  or delete points. Rows are automatically sorted by branch and order when
  exported.
- Use the **Clear branch** button to remove all picks for the active label or
  **Reset annotations** to start a fresh dispersion session.

### Exported artefacts

- Clicking **Download waveform annotations** returns a JSON document containing
  event metadata, per-trace headers, and the current annotation table. Each
  record includes station/network identifiers, epicentral distance, sample rate,
  P/S picks, confidence, quality flag, and comments.
- The **Download dispersion annotations** button provides a JSON package with
  the dispersion metadata (period and velocity axes, panel label) and the set of
  branch picks sorted by branch/order.
- These exports can be ingested directly into downstream training pipelines or
  archival databases.

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
   options, and trigger DNN pre-annotation. The main panel exposes two tabs:
   **Waveforms** for trace review and **Dispersion** for frequency–velocity
   annotation.

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

Surface-wave dispersion images can be provided via the sidebar using either of
the following formats:

- ``.npz`` archives containing:

  - ``energy`` (or ``image``) – ``(N_v, N_p)`` float array of dispersion energy.
  - ``periods`` – ``(N_p,)`` periods in seconds. ``frequencies`` can be
    supplied instead and will be inverted to periods.
  - ``velocities`` – ``(N_v,)`` phase velocities in km/s.
  - Optional ``label`` string identifying the dispersion panel.

- HDF5 files with a ``dispersion`` group (or datasets at the root) providing the
  same ``energy``/``periods``/``velocities`` datasets and an optional
  ``label`` attribute.

Once loaded, switch to the **Dispersion** tab, choose a branch name, and click
points on the image to trace dispersion curves. The table beneath the plot can
be edited to refine, reorder, or delete picks, and the final annotations can be
downloaded as JSON.

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
    ├── dispersion.py
    ├── event_models.py
    ├── filtering.py
    ├── plotting.py
    └── sample_event.py
```

## License

This tool inherits the Apache-2.0 license from the root of the repository.
