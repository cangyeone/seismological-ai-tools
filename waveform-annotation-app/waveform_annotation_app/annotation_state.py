from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class TraceAnnotation:
    station_code: str
    network_code: str
    distance_km: float
    p_pick_s: Optional[float] = None
    s_pick_s: Optional[float] = None
    phase_confidence: Optional[float] = None
    quality_flag: str = ""
    comments: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def annotations_to_dataframe(annotations: List[TraceAnnotation]) -> pd.DataFrame:
    return pd.DataFrame([ann.to_dict() for ann in annotations])


def dataframe_to_annotations(df: pd.DataFrame) -> List[TraceAnnotation]:
    records = df.fillna({"p_pick_s": None, "s_pick_s": None, "phase_confidence": None}).to_dict("records")
    return [TraceAnnotation(**rec) for rec in records]


def merge_annotations(
    base: List[TraceAnnotation],
    updates: Dict[str, TraceAnnotation],
) -> List[TraceAnnotation]:
    merged: List[TraceAnnotation] = []
    for annotation in base:
        key = f"{annotation.network_code}.{annotation.station_code}"
        merged.append(updates.get(key, annotation))
    return merged
