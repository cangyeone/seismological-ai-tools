from __future__ import annotations

from typing import Dict, List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .annotation_state import TraceAnnotation
from .dispersion import (
    DispersionAnnotation,
    DispersionImage,
    sort_dispersion_annotations,
)
from .event_models import TraceData


COLORS = {
    "P": "#1f77b4",
    "S": "#ff7f0e",
}


def _trace_label(trace: TraceData) -> str:
    md = trace.metadata
    return f"{md.network_code}.{md.station_code} ({md.distance_km:.1f} km)"


def create_waveform_figure(
    traces: List[TraceData],
    annotations: Dict[str, TraceAnnotation],
    vertical_scaling: float = 1.0,
    show_annotations: bool = True,
) -> go.Figure:
    if not traces:
        return go.Figure()

    rows = len(traces)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.005)

    for idx, trace in enumerate(traces, start=1):
        samples = trace.samples * vertical_scaling
        time_axis = trace.metadata.time_axis(len(samples))
        label = _trace_label(trace)
        fig.add_trace(
            go.Scatter(x=time_axis, y=samples, mode="lines", name=label, showlegend=False),
            row=idx,
            col=1,
        )

        key = f"{trace.metadata.network_code}.{trace.metadata.station_code}"
        annotation = annotations.get(key)
        if show_annotations and annotation:
            if annotation.p_pick_s is not None:
                fig.add_vline(
                    x=annotation.p_pick_s,
                    line=dict(color=COLORS["P"], dash="dash"),
                    annotation_text="P",
                    annotation_position="top right",
                    row=idx,
                    col=1,
                )
            if annotation.s_pick_s is not None:
                fig.add_vline(
                    x=annotation.s_pick_s,
                    line=dict(color=COLORS["S"], dash="dot"),
                    annotation_text="S",
                    annotation_position="top right",
                    row=idx,
                    col=1,
                )

        fig.update_yaxes(title_text=label, row=idx, col=1)

    fig.update_layout(
        height=200 * rows,
        margin=dict(l=80, r=20, t=40, b=40),
        xaxis_title="Time since trace start (s)",
    )
    return fig


def create_dispersion_figure(
    dispersion: DispersionImage,
    annotations: Optional[Dict[str, List[DispersionAnnotation]]] = None,
    colourscale: str = "Viridis",
) -> go.Figure:
    """Create a Plotly figure visualising dispersion energy with annotations."""

    fig = go.Figure()
    norm_energy = dispersion.normalised_energy()
    fig.add_trace(
        go.Heatmap(
            z=norm_energy,
            x=dispersion.periods,
            y=dispersion.velocities,
            colorscale=colourscale,
            colorbar=dict(title="Normalised energy"),
            hovertemplate="Period: %{x:.2f} s<br>Velocity: %{y:.2f} km/s<br>Energy: %{z:.2f}<extra></extra>",
        )
    )

    if annotations:
        for branch, branch_annotations in annotations.items():
            if not branch_annotations:
                continue
            ordered = sort_dispersion_annotations(branch_annotations)
            fig.add_trace(
                go.Scatter(
                    x=[ann.period_s for ann in ordered],
                    y=[ann.velocity_kms for ann in ordered],
                    mode="markers+lines",
                    name=branch or "branch",
                    marker=dict(size=8, symbol="circle", line=dict(width=1, color="white")),
                    line=dict(width=2),
                    hovertemplate="Branch: %s<br>Period: %%{x:.2f} s<br>Velocity: %%{y:.2f} km/s" % (branch or "branch"),
                )
            )

    fig.update_layout(
        xaxis_title="Period (s)",
        yaxis_title="Phase velocity (km/s)",
        margin=dict(l=80, r=40, t=40, b=60),
        height=600,
    )
    return fig
