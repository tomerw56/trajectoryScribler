from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .covariance_stability import (
    CovarianceStabilityResult,
    CovarianceStabilityState,
)


@dataclass(frozen=True)
class CovarianceReliabilitySummary:
    """Operational reliability summary derived from temporal stability.

    This is *not* statistical calibration/NEES. It answers the runtime question:
    "For what fraction of classified samples was this covariance usable?"

    STABLE and CHANGING are counted as usable because CHANGING explicitly means
    "usable with caution". UNSTABLE is counted as unusable.
    """

    classified_samples: int
    stable_fraction: float
    changing_fraction: float
    unstable_fraction: float
    usable_fraction: float
    mean_roughness: float
    max_roughness: float

    @property
    def has_data(self) -> bool:
        return self.classified_samples > 0


def summarize_covariance_reliability(
    results: Iterable[CovarianceStabilityResult],
) -> CovarianceReliabilitySummary:
    classified = [
        result
        for result in results
        if result.state != CovarianceStabilityState.INSUFFICIENT_DATA
    ]

    count = len(classified)
    if count == 0:
        return CovarianceReliabilitySummary(
            classified_samples=0,
            stable_fraction=0.0,
            changing_fraction=0.0,
            unstable_fraction=0.0,
            usable_fraction=0.0,
            mean_roughness=0.0,
            max_roughness=0.0,
        )

    stable = sum(
        result.state == CovarianceStabilityState.STABLE
        for result in classified
    )
    changing = sum(
        result.state == CovarianceStabilityState.CHANGING
        for result in classified
    )
    unstable = sum(
        result.state == CovarianceStabilityState.UNSTABLE
        for result in classified
    )
    roughness = np.asarray(
        [result.roughness for result in classified],
        dtype=float,
    )

    return CovarianceReliabilitySummary(
        classified_samples=count,
        stable_fraction=stable / count,
        changing_fraction=changing / count,
        unstable_fraction=unstable / count,
        usable_fraction=(stable + changing) / count,
        mean_roughness=float(np.mean(roughness)),
        max_roughness=float(np.max(roughness)),
    )
