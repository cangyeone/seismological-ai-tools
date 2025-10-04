# Waveform Annotation Workbench – Feature Guide

This guide provides deep dives into the major capabilities that ship with the
Streamlit-based waveform annotation workbench. It is intended for analysts and
developers who want to understand how data flows through the system, what each
interface element manipulates, and how the exported artefacts can be consumed.

## 1. Event loading pipeline

### 1.1 Demo generator

* **Purpose:** supplies a synthetic earthquake with multiple stations and a
  companion dispersion image for training or demonstrations.
* **Implementation:** located in `waveform_annotation_app/sample_event.py`.
  Produces an `EventData` object with procedurally generated traces and realistic
  metadata (origin time, magnitude, hypocentre, instrument codes).
* **Usage notes:** ideal for sanity checks when no external files are available.
  Toggle "Demo event" in the sidebar and the app immediately refreshes the plot
  and tables.

### 1.2 NumPy `.npz` loader

* **Purpose:** ingest a compact archive of arrays representing all traces from an
  event.
* **Implementation:** see `waveform_annotation_app/data_loader.py` functions
  `load_event_from_npz` and `parse_event_from_arrays`.
* **Expected content:**
  - `waveforms`: 2-D float array shaped `(N_traces, N_samples)`
  - `station_codes`, `network_codes`, `distances_km`: 1-D arrays of length
    `N_traces`
  - `sampling_rate`: scalar (float or integer)
  - Optional `origin_time`, `latitude`, `longitude`, `depth_km`, `magnitude`,
    `event_type`, `event_id`, `channels`, `location_codes`
* **Metadata extraction:** scalar arrays are converted to native Python types and
  stored on the `EventMetadata` dataclass. Missing values fall back to the demo
  defaults.
* **Error handling:** informative `ValueError` messages are raised if mandatory
  arrays are absent or dimensionally inconsistent.

### 1.3 HDF5 loader

* **Purpose:** support large multi-station events with rich metadata stored as
  datasets and attributes.
* **Implementation:** see `load_event_from_hdf5` in
  `waveform_annotation_app/data_loader.py`.
* **Attributes:** the root group attributes are interpreted as event metadata.
  The loader automatically parses ISO8601 timestamps and converts numeric values
  to floats. Attribute names are case-insensitive.
* **Datasets:** waveform and station arrays can be stored at the root or within
  nested groups. The helper `_resolve_dataset` accepts dataset paths referenced
  via the `data_path` attribute when the structure is nested.
* **Station-level metadata:** `distances_km`, `azimuths`, `back_azimuths`,
  `channels`, and `location_codes` are optional datasets. If unavailable, the app
  still functions with the core waveform and distance information.

### 1.4 SAC collection loader

* **Purpose:** aggregate multiple SAC files describing the same earthquake.
* **Implementation:** `load_event_from_sac_files` in
  `waveform_annotation_app/data_loader.py` uses ObsPy's `read` helper.
* **Metadata mapping:** fields such as `kevnm`, `evla`, `evlo`, `evdp`, `mag`,
  `dist`, `az`, `baz`, and `delta` are mapped to the internal `TraceMetadata`
  structure. Missing values default to zero but remain editable in the app.
* **Time alignment:** traces are trimmed/padded so they share a common start time
  and sampling rate. The loader emits warnings when inconsistent sampling rates
  are detected.
* **Usage tips:** upload the SAC files simultaneously to ensure they are merged
  into a single event. Individual uploads are still accepted but result in a
  single-trace session.

### 1.5 Dispersion image loader

* **Purpose:** import 2-D frequency–velocity energy maps for surface-wave
  analysis.
* **Implementation:** see `load_dispersion_from_npz` and
  `load_dispersion_from_hdf5` in `waveform_annotation_app/data_loader.py`.
* **Axis handling:** both functions ensure that periods and velocities are
  strictly increasing and cast to NumPy arrays. If only frequencies are provided,
  they are inverted to periods (`period = 1 / frequency`).
* **Normalisation:** the `DispersionImage.normalised_energy` method rescales the
  energy matrix to the [0, 1] interval for stable colour mapping.

## 2. Interactive waveform analysis

### 2.1 Plotting stack

* **Implementation:** `waveform_annotation_app/plotting.py` exposes
  `create_waveform_figure`, which constructs a Plotly figure with distance-based
  stacking, optional annotation overlays, and responsive layout defaults.
* **Controls:** vertical scaling and annotation visibility flags are passed from
  Streamlit's session state to the plotting helper, ensuring UI changes are
  reflected immediately.
* **Tooling:** the figure utilises Plotly's native toolbar for zoom, pan,
  lasso-select, and download-as-PNG actions.

### 2.2 Filtering operations

* **Implementation:** `waveform_annotation_app/filtering.py` defines the
  `FilterSettings` dataclass and the `apply_filter` dispatcher. Butterworth
  filters are constructed with SciPy and applied via `sosfiltfilt` for zero-phase
  response.
* **Configuration:** Streamlit widgets update `FilterSettings`, which dictate the
  filter type and corner frequencies. Invalid ranges trigger helper messages in
  the sidebar rather than crashing the app.
* **Performance:** filters are applied trace-by-trace during plotting. For large
  datasets consider down-sampling prior to upload.

## 3. Annotation management

### 3.1 Phase picks and metadata

* **Data model:** `TraceAnnotation` (see
  `waveform_annotation_app/annotation_state.py`) captures picks, quality, and
  free-text comments per station.
* **UI binding:** `annotations_to_dataframe` and `dataframe_to_annotations`
  bridge between the dataclass list and Streamlit's data editor.
* **Export schema:** `_export_annotations` in `app.py` serialises event metadata
  plus the annotation table into JSON for downstream use.

### 3.2 DNN-assisted suggestions

* **Model:** `SimplePhaseNet` (in `dnn_prelabel.py`) is a three-layer convolution
  network with handcrafted initial weights to emphasise gradients and envelopes.
* **Workflow:** the `DNNPreLabeler.annotate_event` method iterates over each
  trace, normalises the waveform, generates P/S probabilities, converts them to
  pick times, and merges them into the session annotation dictionary.
* **Extensibility:** swap in a different PyTorch module by modifying
  `DNNPreLabeler.create` while preserving the `annotate_trace` interface.

### 3.3 Dispersion picks

* **Data model:** `DispersionAnnotation` describes a dispersion branch sample
  (branch name, period, velocity, order, weight). Conversion helpers provide a
  round-trip between dataclasses and pandas DataFrames.
* **Sorting:** `sort_dispersion_annotations` ensures exported picks are grouped
  by branch and ordered along increasing period.
* **Export schema:** `_export_dispersion_annotations` in `app.py` combines the
  axis metadata with sorted picks when generating the download payload.

## 4. Exported files

### 4.1 Waveform annotation JSON

```
{
  "event": {
    "metadata": {
      "event_id": "demo-001",
      "origin_time": "2020-01-01T00:00:00Z",
      "latitude": 34.5,
      "longitude": 135.6,
      "depth_km": 12.3,
      "magnitude": 4.5,
      "event_type": "earthquake"
    },
    "traces": [
      {
        "station_code": "ABC",
        "network_code": "XY",
        "distance_km": 57.2,
        "sampling_rate": 100.0
      }
    ]
  },
  "annotations": [
    {
      "station": "XY.ABC",
      "station_code": "ABC",
      "network_code": "XY",
      "distance_km": 57.2,
      "p_pick_s": 12.4,
      "s_pick_s": 24.8,
      "phase_confidence": 0.93,
      "quality_flag": "A",
      "comments": "Clear arrivals"
    }
  ]
}
```

### 4.2 Dispersion annotation JSON

```
{
  "dispersion": {
    "label": "demo dispersion",
    "periods": [5.0, 6.0, 7.0],
    "velocities": [3.0, 3.1, 3.2]
  },
  "annotations": [
    {
      "branch": "fundamental",
      "period_s": 5.5,
      "velocity_kms": 3.05,
      "order": 0,
      "weight": 1.0
    }
  ]
}
```

## 5. Best practices

1. **Consistent sampling:** ensure all uploaded traces share the same sampling
   rate. Mixed rates will be resampled automatically but may reduce fidelity.
2. **Synchronised clocks:** align trace start times prior to packaging. The app
   assumes the waveforms are already time-aligned when displayed.
3. **Metadata completeness:** populate event attributes wherever possible so the
   exported JSON is self-descriptive.
4. **Dispersion quality:** smooth noisy dispersion images before upload to make
   branch picking easier and reduce mis-clicks.
5. **Model calibration:** treat the built-in DNN suggestions as initial guesses
   and validate them manually, especially for emergent or complex arrivals.

## 6. Troubleshooting

| Symptom | Potential cause | Resolution |
| --- | --- | --- |
| Waveform figure is empty | Waveform arrays contain NaN or Inf values | Clean the data or provide valid ranges before upload |
| Filters produce flat lines | Corner frequencies exceed Nyquist | Adjust the sliders to stay below half the sampling rate |
| SAC upload rejected | Mixed sampling rates across files | Resample traces to a common rate or split the event |
| Dispersion click not recorded | Branch label missing | Enter a branch name before clicking the energy map |
| JSON export missing annotations | Table rows cleared or empty | Re-run the DNN pre-labeler or enter picks manually before exporting |

For further questions, inspect the individual modules inside
`waveform_annotation_app/` – they are fully typed and documented to facilitate
extension.
