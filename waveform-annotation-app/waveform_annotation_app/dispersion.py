from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class DispersionImage:
    """Container for surface-wave dispersion energy maps."""

    periods: np.ndarray
    velocities: np.ndarray
    energy: np.ndarray
    label: str = ""

    def normalised_energy(self) -> np.ndarray:
        """Return energy normalised between 0 and 1 for plotting."""

        if self.energy.size == 0:
            return self.energy
        finite = np.isfinite(self.energy)
        if not np.any(finite):
            return np.zeros_like(self.energy)
        values = self.energy[finite]
        max_val = values.max()
        min_val = values.min()
        if max_val == min_val:
            return np.zeros_like(self.energy)
        scaled = (self.energy - min_val) / (max_val - min_val)
        return np.clip(scaled, 0.0, 1.0)


@dataclass
class DispersionAnnotation:
    branch: str
    period_s: float
    velocity_kms: float
    order: int
    weight: float = 1.0

    def to_dict(self) -> Dict[str, float | int | str]:
        return {
            "branch": self.branch,
            "period_s": float(self.period_s),
            "velocity_kms": float(self.velocity_kms),
            "order": int(self.order),
            "weight": float(self.weight),
        }


def dispersion_annotations_to_dataframe(annotations: Iterable[DispersionAnnotation]) -> pd.DataFrame:
    return pd.DataFrame([ann.to_dict() for ann in annotations])


def dataframe_to_dispersion_annotations(df: pd.DataFrame) -> List[DispersionAnnotation]:
    records = df.fillna({"weight": 1.0, "branch": ""}).to_dict("records")
    annotations: List[DispersionAnnotation] = []
    for idx, record in enumerate(records):
        annotations.append(
            DispersionAnnotation(
                branch=str(record.get("branch", "")),
                period_s=float(record.get("period_s", 0.0)),
                velocity_kms=float(record.get("velocity_kms", 0.0)),
                order=int(record.get("order", idx)),
                weight=float(record.get("weight", 1.0)),
            )
        )
    return annotations


def sort_dispersion_annotations(annotations: Iterable[DispersionAnnotation]) -> List[DispersionAnnotation]:
    return sorted(annotations, key=lambda ann: (ann.branch, ann.order, ann.period_s))
