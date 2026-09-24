from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


TAU = 2.0 * math.pi


def _mod2pi(angle: float) -> float:
    return float(angle % TAU)


@dataclass(frozen=True)
class DubinsPose:
    x: float
    y: float
    heading_rad: float


@dataclass(frozen=True)
class DubinsPath:
    start: DubinsPose
    end: DubinsPose
    radius_m: float
    family: str
    segment_types: tuple[str, str, str]
    segment_parameters: tuple[float, float, float]

    @property
    def length_m(self) -> float:
        return float(sum(self.segment_parameters) * self.radius_m)


@dataclass(frozen=True)
class DubinsFitResult:
    """Piecewise Dubins fit of a freehand XY scribble."""

    xy: np.ndarray
    anchor_poses: np.ndarray  # [N,3] -> x, y, heading_rad
    leg_families: tuple[str, ...]
    min_turn_radius_m: float
    mean_fit_error_m: float
    max_fit_error_m: float


def _lsl(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    p2 = 2.0 + d * d - 2.0 * cab + 2.0 * d * (sa - sb)
    if p2 < -1e-12:
        return None
    p = math.sqrt(max(0.0, p2))
    tmp = math.atan2(cb - ca, d + sa - sb)
    return _mod2pi(-alpha + tmp), p, _mod2pi(beta - tmp)


def _rsr(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    p2 = 2.0 + d * d - 2.0 * cab + 2.0 * d * (-sa + sb)
    if p2 < -1e-12:
        return None
    p = math.sqrt(max(0.0, p2))
    tmp = math.atan2(ca - cb, d - sa + sb)
    return _mod2pi(alpha - tmp), p, _mod2pi(-beta + tmp)


def _lsr(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    p2 = -2.0 + d * d + 2.0 * cab + 2.0 * d * (sa + sb)
    if p2 < -1e-12:
        return None
    p = math.sqrt(max(0.0, p2))
    tmp = (
        math.atan2(-ca - cb, d + sa + sb)
        - math.atan2(-2.0, p)
    )
    return _mod2pi(-alpha + tmp), p, _mod2pi(-beta + tmp)


def _rsl(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    p2 = -2.0 + d * d + 2.0 * cab - 2.0 * d * (sa + sb)
    if p2 < -1e-12:
        return None
    p = math.sqrt(max(0.0, p2))
    tmp = (
        math.atan2(ca + cb, d - sa - sb)
        - math.atan2(2.0, p)
    )
    return _mod2pi(alpha - tmp), p, _mod2pi(beta - tmp)


def _rlr(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    value = (
        6.0 - d * d + 2.0 * cab + 2.0 * d * (sa - sb)
    ) / 8.0
    if value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
        return None
    value = float(np.clip(value, -1.0, 1.0))
    p = _mod2pi(TAU - math.acos(value))
    t = _mod2pi(
        alpha
        - math.atan2(ca - cb, d - sa + sb)
        + p / 2.0
    )
    q = _mod2pi(alpha - beta - t + p)
    return t, p, q


def _lrl(alpha: float, beta: float, d: float):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    value = (
        6.0 - d * d + 2.0 * cab + 2.0 * d * (-sa + sb)
    ) / 8.0
    if value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
        return None
    value = float(np.clip(value, -1.0, 1.0))
    p = _mod2pi(TAU - math.acos(value))
    t = _mod2pi(
        -alpha
        - math.atan2(ca - cb, d + sa - sb)
        + p / 2.0
    )
    q = _mod2pi(beta - alpha - t + p)
    return t, p, q


_FAMILIES: tuple[
    tuple[str, tuple[str, str, str], Callable[[float, float, float], tuple[float, float, float] | None]],
    ...
] = (
    ("LSL", ("L", "S", "L"), _lsl),
    ("RSR", ("R", "S", "R"), _rsr),
    ("LSR", ("L", "S", "R"), _lsr),
    ("RSL", ("R", "S", "L"), _rsl),
    ("RLR", ("R", "L", "R"), _rlr),
    ("LRL", ("L", "R", "L"), _lrl),
)


def shortest_dubins_path(
    start: DubinsPose,
    end: DubinsPose,
    radius_m: float,
) -> DubinsPath:
    """Return the shortest classical forward-only Dubins path."""

    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Dubins minimum turn radius must be > 0")

    dx = float(end.x - start.x)
    dy = float(end.y - start.y)
    distance = math.hypot(dx, dy)
    d = distance / radius
    theta = math.atan2(dy, dx)
    alpha = _mod2pi(start.heading_rad - theta)
    beta = _mod2pi(end.heading_rad - theta)

    best = None
    for family, segment_types, solver in _FAMILIES:
        params = solver(alpha, beta, d)
        if params is None:
            continue
        length = float(sum(params))
        if best is None or length < best[0]:
            best = (length, family, segment_types, params)

    if best is None:
        raise ValueError("No valid Dubins path was found")

    _, family, segment_types, params = best
    return DubinsPath(
        start=start,
        end=end,
        radius_m=radius,
        family=family,
        segment_types=segment_types,
        segment_parameters=tuple(float(v) for v in params),
    )


def _advance_pose(
    pose: DubinsPose,
    segment_type: str,
    distance_m: float,
    radius_m: float,
) -> DubinsPose:
    ds = float(distance_m)
    heading = float(pose.heading_rad)

    if segment_type == "S":
        return DubinsPose(
            pose.x + ds * math.cos(heading),
            pose.y + ds * math.sin(heading),
            heading,
        )

    curvature = (1.0 / radius_m) if segment_type == "L" else (-1.0 / radius_m)
    delta = curvature * ds
    new_heading = heading + delta
    x = pose.x + (math.sin(new_heading) - math.sin(heading)) / curvature
    y = pose.y + (-math.cos(new_heading) + math.cos(heading)) / curvature
    return DubinsPose(x, y, new_heading)


def sample_dubins_path(
    path: DubinsPath,
    spacing_m: float,
) -> np.ndarray:
    """Sample a Dubins path in XY at approximately `spacing_m`."""

    spacing = max(1e-4, float(spacing_m))
    current = path.start
    points = [[current.x, current.y]]

    for segment_type, parameter in zip(
        path.segment_types,
        path.segment_parameters,
    ):
        segment_length = float(parameter) * path.radius_m
        if segment_length <= 1e-12:
            continue
        step_count = max(1, int(math.ceil(segment_length / spacing)))
        step = segment_length / step_count
        for _ in range(step_count):
            current = _advance_pose(
                current,
                segment_type,
                step,
                path.radius_m,
            )
            points.append([current.x, current.y])

    endpoint_error = math.hypot(
        current.x - path.end.x,
        current.y - path.end.y,
    )
    heading_error = abs(
        (current.heading_rad - path.end.heading_rad + math.pi)
        % TAU
        - math.pi
    )
    if endpoint_error > 1e-5 or heading_error > 1e-5:
        raise RuntimeError(
            "Dubins sampler endpoint mismatch: "
            f"position={endpoint_error:.3e}m heading={heading_error:.3e}rad"
        )

    points[-1] = [path.end.x, path.end.y]
    return np.asarray(points, dtype=float)


def _remove_near_duplicate_points(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points.copy()
    keep = [0]
    for index in range(1, len(points)):
        if np.linalg.norm(points[index] - points[keep[-1]]) > 1e-9:
            keep.append(index)
    return points[np.asarray(keep, dtype=int)]


def _resample_polyline(points: np.ndarray, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    points = _remove_near_duplicate_points(np.asarray(points, dtype=float))
    if len(points) < 2:
        raise ValueError("Dubins drawing requires at least two distinct points")

    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment)]
    total = float(cumulative[-1])
    if total <= 1e-9:
        raise ValueError("Dubins drawing is too short")

    spacing = max(1e-4, float(spacing_m))
    distances = np.arange(0.0, total, spacing)
    if len(distances) == 0 or not math.isclose(
        float(distances[-1]),
        total,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        distances = np.r_[distances, total]

    sampled = np.c_[
        np.interp(distances, cumulative, points[:, 0]),
        np.interp(distances, cumulative, points[:, 1]),
    ]
    return sampled, distances


def _smooth_reference(points: np.ndarray, passes: int = 2) -> np.ndarray:
    out = np.asarray(points, dtype=float).copy()
    for _ in range(max(0, int(passes))):
        if len(out) < 3:
            break
        old = out.copy()
        out[1:-1] = (
            0.25 * old[:-2]
            + 0.50 * old[1:-1]
            + 0.25 * old[2:]
        )
    return out


def _interp_xy(
    reference: np.ndarray,
    cumulative: np.ndarray,
    distance_m: float,
) -> np.ndarray:
    s = float(np.clip(distance_m, cumulative[0], cumulative[-1]))
    return np.array(
        [
            np.interp(s, cumulative, reference[:, 0]),
            np.interp(s, cumulative, reference[:, 1]),
        ],
        dtype=float,
    )


def _heading_at_distance(
    reference: np.ndarray,
    cumulative: np.ndarray,
    distance_m: float,
    window_m: float,
) -> float:
    total = float(cumulative[-1])
    half = min(max(1e-4, float(window_m)), max(1e-4, total / 3.0))
    before = max(0.0, float(distance_m) - half)
    after = min(total, float(distance_m) + half)
    if after - before <= 1e-9:
        before = max(0.0, float(distance_m) - 1e-3)
        after = min(total, float(distance_m) + 1e-3)

    p0 = _interp_xy(reference, cumulative, before)
    p1 = _interp_xy(reference, cumulative, after)
    delta = p1 - p0
    if np.linalg.norm(delta) <= 1e-12:
        return 0.0
    return float(math.atan2(delta[1], delta[0]))


def _nearest_fit_errors(
    reference: np.ndarray,
    generated: np.ndarray,
) -> tuple[float, float]:
    if len(reference) == 0 or len(generated) == 0:
        return 0.0, 0.0

    distances: list[np.ndarray] = []
    chunk = 256
    for start in range(0, len(reference), chunk):
        ref = reference[start:start + chunk]
        delta = ref[:, None, :] - generated[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        distances.append(np.sqrt(np.min(squared, axis=1)))
    values = np.concatenate(distances)
    return float(np.mean(values)), float(np.max(values))


def fit_freehand_dubins(
    raw_xy,
    *,
    min_turn_radius_m: float,
    path_resolution_m: float,
    smoothing_passes: int = 2,
) -> DubinsFitResult:
    """Interpret a freehand scribble as intent and fit piecewise Dubins legs.

    Anchor spacing is chosen automatically from the requested minimum radius.
    The raw mouse points are *not* treated as hard waypoints.
    """

    radius = float(min_turn_radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Dubins minimum turn radius must be > 0")

    raw = _remove_near_duplicate_points(np.asarray(raw_xy, dtype=float))
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 2:
        raise ValueError("Dubins drawing requires at least two XY points")

    reference_spacing = max(
        float(path_resolution_m),
        min(1.0, radius / 8.0),
    )
    reference, _ = _resample_polyline(raw, reference_spacing)
    reference = _smooth_reference(reference, max(1, int(smoothing_passes)))
    reference, cumulative = _resample_polyline(
        reference,
        reference_spacing,
    )

    total_length = float(cumulative[-1])
    if total_length < max(0.5, radius * 0.25):
        raise ValueError(
            "Dubins drawing is too short for the requested minimum turn radius"
        )

    # Enough separation to prevent every mouse wiggle becoming a hard pose,
    # while still retaining the broad shape of the freehand intent.
    anchor_spacing = max(
        radius * 2.5,
        float(path_resolution_m) * 12.0,
    )
    anchor_distances = np.arange(0.0, total_length, anchor_spacing)
    if (
        len(anchor_distances) == 0
        or total_length - float(anchor_distances[-1]) > anchor_spacing * 0.35
    ):
        anchor_distances = np.r_[anchor_distances, total_length]
    else:
        anchor_distances[-1] = total_length

    if len(anchor_distances) < 2:
        anchor_distances = np.array([0.0, total_length], dtype=float)

    heading_window = max(
        radius * 0.35,
        float(path_resolution_m) * 4.0,
    )
    anchor_poses: list[DubinsPose] = []
    for distance in anchor_distances:
        point = _interp_xy(reference, cumulative, float(distance))
        heading = _heading_at_distance(
            reference,
            cumulative,
            float(distance),
            heading_window,
        )
        anchor_poses.append(
            DubinsPose(float(point[0]), float(point[1]), heading)
        )

    generated_parts: list[np.ndarray] = []
    families: list[str] = []
    for index in range(len(anchor_poses) - 1):
        leg = shortest_dubins_path(
            anchor_poses[index],
            anchor_poses[index + 1],
            radius,
        )
        part = sample_dubins_path(
            leg,
            max(1e-3, float(path_resolution_m)),
        )
        if generated_parts:
            part = part[1:]
        generated_parts.append(part)
        families.append(leg.family)

    generated = np.vstack(generated_parts)
    mean_error, max_error = _nearest_fit_errors(reference, generated)

    anchors_array = np.asarray(
        [[pose.x, pose.y, pose.heading_rad] for pose in anchor_poses],
        dtype=float,
    )
    return DubinsFitResult(
        xy=generated,
        anchor_poses=anchors_array,
        leg_families=tuple(families),
        min_turn_radius_m=radius,
        mean_fit_error_m=mean_error,
        max_fit_error_m=max_error,
    )
