from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from .covariance_policy import CovarianceScope
from .models import CovarianceSamples


class CovarianceStabilityState(str, Enum):
    """Four-state runtime classification for recent covariance behavior."""

    INSUFFICIENT_DATA = "insufficient_data"
    STABLE = "stable"
    CHANGING = "changing"
    UNSTABLE = "unstable"


# Backward-compatible public name. The scope is now a system-wide policy.
CovarianceStabilityScope = CovarianceScope


@dataclass(frozen=True)
class CovarianceStabilityConfig:
    """Policy thresholds for trailing-window covariance stability.

    The change metrics are normalized Frobenius distances. Thresholds are
    operational policy values, not statistical confidence levels.

    ``normalization_floor`` prevents harmless changes in an almost-zero
    covariance from becoming huge relative ratios. ``absolute_change_deadband``
    is applied before normalization to suppress numerical noise entirely.
    """

    window_sec: float = 6.0
    min_samples: int = 3
    min_window_coverage_fraction: float = 0.75

    # Numerical epsilon remains tiny; normalization_floor is the meaningful
    # covariance scale used when the covariance itself is near zero.
    norm_epsilon: float = 1e-12
    normalization_floor: float = 0.005
    absolute_change_deadband: float = 1e-6

    # Fully stable region.
    stable_max_drift: float = 0.40
    stable_max_rms_change: float = 0.15
    stable_max_single_jump: float = 0.35

    # CHANGING / UNSTABLE policy.
    # A single large jump is not enough for UNSTABLE; repeated large changes
    # or sustained high RMS movement are required.
    changing_max_drift: float = 1.50
    changing_max_rms_change: float = 0.35
    changing_max_single_jump: float = 0.80
    unstable_large_change_fraction: float = 0.25

    # RMS normalized second difference of covariance matrices. This separates
    # smooth rotation/drift from genuinely erratic covariance evolution.
    unstable_min_roughness: float = 0.35

    def __post_init__(self) -> None:
        if self.window_sec <= 0.0:
            raise ValueError("window_sec must be > 0")
        if self.min_samples < 2:
            raise ValueError("min_samples must be >= 2")
        if not 0.0 < self.min_window_coverage_fraction <= 1.0:
            raise ValueError("min_window_coverage_fraction must be in (0, 1]")
        if self.norm_epsilon <= 0.0:
            raise ValueError("norm_epsilon must be > 0")
        if self.normalization_floor <= 0.0:
            raise ValueError("normalization_floor must be > 0")
        if self.absolute_change_deadband < 0.0:
            raise ValueError("absolute_change_deadband must be >= 0")
        if not 0.0 <= self.unstable_large_change_fraction <= 1.0:
            raise ValueError("unstable_large_change_fraction must be in [0, 1]")
        if self.unstable_min_roughness < 0.0:
            raise ValueError("unstable_min_roughness must be >= 0")

        stable = (
            self.stable_max_drift,
            self.stable_max_rms_change,
            self.stable_max_single_jump,
        )
        changing = (
            self.changing_max_drift,
            self.changing_max_rms_change,
            self.changing_max_single_jump,
        )
        if any(value < 0.0 for value in (*stable, *changing)):
            raise ValueError("stability thresholds must be >= 0")
        if any(s > c for s, c in zip(stable, changing)):
            raise ValueError("stable thresholds must not exceed changing thresholds")


@dataclass(frozen=True)
class CovarianceStabilityResult:
    state: CovarianceStabilityState
    scope: CovarianceScope
    end_time_sec: float
    requested_window_sec: float
    source_warmup_sec: float
    observed_window_sec: float
    sample_count: int
    drift: float
    rms_change: float
    max_single_jump: float
    large_change_fraction: float
    roughness: float
    reason: str

    @property
    def is_trustworthy(self) -> bool:
        """Strict trust flag: only fully STABLE covariance is trusted."""
        return self.state == CovarianceStabilityState.STABLE

    @property
    def is_usable_with_caution(self) -> bool:
        return self.state in (
            CovarianceStabilityState.STABLE,
            CovarianceStabilityState.CHANGING,
        )


@dataclass(frozen=True)
class CovarianceStabilityHistory:
    start_time_sec: float
    end_time_sec: float
    results: tuple[CovarianceStabilityResult, ...]


class CovarianceStabilityCalculator:
    """Evaluate how violently a covariance estimate changed recently.

    This calculator measures *stability*, not statistical calibration.
    It never needs future truth.

    The default scope is ``PREDICTION_XY`` because the current forward
    prediction envelope consumes the XY velocity covariance.
    """

    def __init__(self, config: CovarianceStabilityConfig | None = None) -> None:
        self.config = config or CovarianceStabilityConfig()

    @staticmethod
    def normalized_frobenius_distance(
        first: np.ndarray,
        second: np.ndarray,
        *,
        epsilon: float = 1e-12,
        normalization_floor: float = 0.0,
        absolute_deadband: float = 0.0,
    ) -> float:
        """Return normalized covariance change after an absolute deadband.

        The inputs may be any equal-shape matrices, including the 2x2 XY
        covariance or the complete 3x3 covariance.
        """
        a = np.asarray(first, dtype=float)
        b = np.asarray(second, dtype=float)

        if a.shape != b.shape or a.ndim != 2:
            raise ValueError("covariance comparison matrices must have equal 2D shape")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("covariance comparison matrices must be finite")

        numerator = float(np.linalg.norm(a - b, ord="fro"))
        if numerator <= float(absolute_deadband):
            return 0.0

        denominator = max(
            float(np.linalg.norm(a, ord="fro")),
            float(np.linalg.norm(b, ord="fro")),
            float(normalization_floor),
            float(epsilon),
        )
        return numerator / denominator

    @staticmethod
    def _project_matrices(
        matrices: np.ndarray,
        scope: CovarianceScope,
    ) -> np.ndarray:
        matrices = np.asarray(matrices, dtype=float)
        if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
            raise ValueError("covariance matrices must have shape [N, 3, 3]")

        if scope == CovarianceScope.PREDICTION_XY:
            return matrices[:, :2, :2]
        if scope == CovarianceScope.FULL_3D:
            return matrices
        raise ValueError(f"unsupported covariance stability scope: {scope!r}")

    @staticmethod
    def covariance_roughness(
        matrices: np.ndarray,
        *,
        normalization_floor: float,
        epsilon: float = 1e-12,
        absolute_deadband: float = 0.0,
    ) -> float:
        """Measure how erratically covariance evolution changes direction.

        This is the RMS normalized second difference of the covariance
        sequence. Smooth rotation or smooth drift can have a large first
        difference while still having low roughness.
        """
        values = np.asarray(matrices, dtype=float)
        if values.ndim != 3 or len(values) < 3:
            return 0.0

        first_difference = np.diff(values, axis=0)
        second_difference = np.diff(first_difference, axis=0)
        normalized: list[float] = []

        for index, delta2 in enumerate(second_difference, start=1):
            magnitude = float(np.linalg.norm(delta2, ord="fro"))
            if magnitude <= float(absolute_deadband):
                normalized.append(0.0)
                continue

            local_scale = max(
                float(np.linalg.norm(values[index - 1], ord="fro")),
                float(np.linalg.norm(values[index], ord="fro")),
                float(np.linalg.norm(values[index + 1], ord="fro")),
                float(normalization_floor),
                float(epsilon),
            )
            normalized.append(magnitude / local_scale)

        if not normalized:
            return 0.0
        return float(np.sqrt(np.mean(np.square(normalized))))

    def _window_arrays(
        self,
        samples: CovarianceSamples,
        *,
        end_time_sec: float,
        window_sec: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        times = np.asarray(samples.time_sec, dtype=float).reshape(-1)
        matrices = np.asarray(samples.matrices, dtype=float)

        if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
            raise ValueError("covariance matrices must have shape [N, 3, 3]")
        if len(times) != len(matrices):
            raise ValueError("covariance time and matrix counts differ")
        if len(times) == 0:
            return times, matrices, 0.0
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(matrices)):
            raise ValueError("covariance samples must be finite")

        order = np.argsort(times, kind="stable")
        times = times[order]
        matrices = matrices[order]

        end = float(end_time_sec)
        start = end - float(window_sec)
        last_index = int(np.searchsorted(times, end, side="right"))
        if last_index <= 0:
            return np.empty(0), np.empty((0, 3, 3)), 0.0

        times = times[:last_index]
        matrices = matrices[:last_index]

        # Include the most recent value at-or-before the window start as the
        # baseline valid at that instant, then all values inside the window.
        first_inside = int(np.searchsorted(times, start, side="left"))
        if first_inside < len(times) and np.isclose(times[first_inside], start):
            baseline = first_inside
        else:
            baseline = max(0, first_inside - 1)

        window_times = times[baseline:]
        window_matrices = matrices[baseline:]

        if len(window_times) == 0:
            return window_times, window_matrices, 0.0

        effective_start = start if window_times[0] <= start else float(window_times[0])
        observed = max(0.0, end - effective_start)
        observed = min(float(window_sec), observed)
        return window_times, window_matrices, observed

    def evaluate(
        self,
        samples: CovarianceSamples,
        *,
        end_time_sec: float | None = None,
        window_sec: float | None = None,
        source_warmup_sec: float = 0.0,
        scope: CovarianceScope = CovarianceScope.PREDICTION_XY,
    ) -> CovarianceStabilityResult:
        requested_window = float(
            self.config.window_sec if window_sec is None else window_sec
        )
        if requested_window <= 0.0:
            raise ValueError("window_sec must be > 0")

        scope = CovarianceScope(scope)

        all_times = np.asarray(samples.time_sec, dtype=float).reshape(-1)
        if end_time_sec is None:
            end = float(all_times[-1]) if len(all_times) else 0.0
        else:
            end = float(end_time_sec)

        warmup = float(source_warmup_sec)
        if warmup < 0.0:
            raise ValueError("source_warmup_sec must be >= 0")

        first_sample_time = float(all_times[0]) if len(all_times) else 0.0
        warmup_complete_time = first_sample_time + warmup
        fully_observed_after_warmup = warmup_complete_time + requested_window

        times, matrices, observed = self._window_arrays(
            samples,
            end_time_sec=end,
            window_sec=requested_window,
        )

        min_coverage = requested_window * self.config.min_window_coverage_fraction

        # A rolling covariance is itself still changing while its own source
        # window fills. Do not classify that initialization transient as
        # instability. The app passes its covariance_window_sec here, so with
        # the defaults classification starts after 3 s + 5 s = 8 s.
        if warmup > 0.0 and end + 1e-12 < fully_observed_after_warmup:
            return CovarianceStabilityResult(
                state=CovarianceStabilityState.INSUFFICIENT_DATA,
                scope=scope,
                end_time_sec=end,
                requested_window_sec=requested_window,
                source_warmup_sec=warmup,
                observed_window_sec=observed,
                sample_count=int(len(times)),
                drift=0.0,
                rms_change=0.0,
                max_single_jump=0.0,
                large_change_fraction=0.0,
                roughness=0.0,
                reason=(
                    f"warming up: need {warmup:.2f}s covariance source window "
                    f"+ {requested_window:.2f}s stability window"
                ),
            )

        if len(times) < self.config.min_samples or observed + 1e-12 < min_coverage:
            return CovarianceStabilityResult(
                state=CovarianceStabilityState.INSUFFICIENT_DATA,
                scope=scope,
                end_time_sec=end,
                requested_window_sec=requested_window,
                source_warmup_sec=warmup,
                observed_window_sec=observed,
                sample_count=int(len(times)),
                drift=0.0,
                rms_change=0.0,
                max_single_jump=0.0,
                large_change_fraction=0.0,
                roughness=0.0,
                reason=(
                    f"need >= {self.config.min_samples} samples and "
                    f"{min_coverage:.2f}s coverage"
                ),
            )

        projected = self._project_matrices(matrices, scope)

        epsilon = self.config.norm_epsilon
        deadband = self.config.absolute_change_deadband
        changes = np.array(
            [
                self.normalized_frobenius_distance(
                    projected[index - 1],
                    projected[index],
                    epsilon=epsilon,
                    normalization_floor=self.config.normalization_floor,
                    absolute_deadband=deadband,
                )
                for index in range(1, len(projected))
            ],
            dtype=float,
        )

        drift = self.normalized_frobenius_distance(
            projected[0],
            projected[-1],
            epsilon=epsilon,
            normalization_floor=self.config.normalization_floor,
            absolute_deadband=deadband,
        )
        rms_change = (
            float(np.sqrt(np.mean(np.square(changes))))
            if len(changes)
            else 0.0
        )
        max_jump = float(np.max(changes)) if len(changes) else 0.0
        large_change_mask = changes > self.config.changing_max_single_jump
        material_change_mask = changes > self.config.stable_max_single_jump
        large_change_count = int(np.count_nonzero(large_change_mask))
        material_change_count = int(np.count_nonzero(material_change_mask))
        large_change_fraction = (
            float(np.mean(large_change_mask))
            if len(changes)
            else 0.0
        )
        roughness = self.covariance_roughness(
            projected,
            normalization_floor=self.config.normalization_floor,
            epsilon=epsilon,
            absolute_deadband=deadband,
        )

        stable = (
            drift <= self.config.stable_max_drift
            and rms_change <= self.config.stable_max_rms_change
            and max_jump <= self.config.stable_max_single_jump
        )

        evolution_is_rough = roughness >= self.config.unstable_min_roughness
        repeated_large_changes = (
            evolution_is_rough
            and large_change_count >= 2
            and large_change_fraction >= self.config.unstable_large_change_fraction
        )
        sustained_high_change = (
            evolution_is_rough
            and material_change_count >= 2
            and rms_change > self.config.changing_max_rms_change
        )
        severe_sustained_drift = (
            evolution_is_rough
            and material_change_count >= 2
            and drift > self.config.changing_max_drift
            and rms_change > self.config.stable_max_rms_change
        )

        scope_name = (
            "XY prediction covariance"
            if scope == CovarianceScope.PREDICTION_XY
            else "full 3D covariance"
        )

        if stable:
            state = CovarianceStabilityState.STABLE
            reason = f"{scope_name} changes are small"
        elif repeated_large_changes:
            state = CovarianceStabilityState.UNSTABLE
            reason = (
                f"{scope_name}: repeated large covariance changes "
                f"({large_change_fraction:.0%} of recent steps)"
            )
        elif sustained_high_change:
            state = CovarianceStabilityState.UNSTABLE
            reason = f"{scope_name}: sustained covariance movement is high"
        elif severe_sustained_drift:
            state = CovarianceStabilityState.UNSTABLE
            reason = f"{scope_name}: large drift with sustained movement"
        else:
            state = CovarianceStabilityState.CHANGING
            if (
                max_jump > self.config.changing_max_single_jump
                and large_change_count < 2
            ):
                reason = (
                    f"{scope_name}: isolated large jump; use with caution"
                )
            elif not evolution_is_rough and (
                rms_change > self.config.changing_max_rms_change
                or drift > self.config.changing_max_drift
            ):
                reason = f"{scope_name}: large but smooth covariance evolution"
            elif drift > self.config.stable_max_drift:
                reason = f"{scope_name}: covariance is drifting smoothly"
            else:
                reason = f"{scope_name} is changing; use with caution"

        return CovarianceStabilityResult(
            state=state,
            scope=scope,
            end_time_sec=end,
            requested_window_sec=requested_window,
            source_warmup_sec=warmup,
            observed_window_sec=observed,
            sample_count=int(len(times)),
            drift=float(drift),
            rms_change=float(rms_change),
            max_single_jump=float(max_jump),
            large_change_fraction=float(large_change_fraction),
            roughness=float(roughness),
            reason=reason,
        )

    def evaluate_history(
        self,
        samples: CovarianceSamples,
        *,
        last_n_seconds: float,
        end_time_sec: float | None = None,
        evaluation_window_sec: float | None = None,
        source_warmup_sec: float = 0.0,
        scope: CovarianceScope = CovarianceScope.PREDICTION_XY,
    ) -> CovarianceStabilityHistory:
        """Return stability evaluations for covariance samples in the last N sec."""
        history_span = float(last_n_seconds)
        if history_span <= 0.0:
            raise ValueError("last_n_seconds must be > 0")

        times = np.asarray(samples.time_sec, dtype=float).reshape(-1)
        if len(times) == 0:
            end = float(end_time_sec or 0.0)
            return CovarianceStabilityHistory(end - history_span, end, ())

        end = float(times[-1] if end_time_sec is None else end_time_sec)
        start = end - history_span
        selected = times[(times >= start) & (times <= end)]
        results = tuple(
            self.evaluate(
                samples,
                end_time_sec=float(time_sec),
                window_sec=evaluation_window_sec,
                source_warmup_sec=source_warmup_sec,
                scope=scope,
            )
            for time_sec in selected
        )
        return CovarianceStabilityHistory(start, end, results)

    def with_window(self, window_sec: float) -> "CovarianceStabilityCalculator":
        return CovarianceStabilityCalculator(
            replace(self.config, window_sec=float(window_sec))
        )
