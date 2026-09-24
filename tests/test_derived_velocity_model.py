from __future__ import annotations

import math

import numpy as np

from trajectory_app.core import derive_velocity_model
from trajectory_app.models import TrajectoryResult


def _trajectory(xyz: np.ndarray) -> TrajectoryResult:
    xyz = np.asarray(xyz, dtype=float)
    xy = xyz[:, :2].copy()
    terrain = np.zeros(len(xyz), dtype=float)
    cumulative = np.r_[
        0.0,
        np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)),
    ]
    return TrajectoryResult(
        xy=xy,
        xyz=xyz,
        terrain_z=terrain,
        cumulative_distance_3d=cumulative,
    )


def test_straight_trajectory_has_infinite_turn_radius():
    x = np.linspace(0.0, 20.0, 81)
    xyz = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))

    model = derive_velocity_model(
        _trajectory(xyz),
        speed_mps=4.0,
        path_resolution_m=0.25,
    )

    assert math.isinf(model.min_turn_radius_m)
    assert model.max_turn_rate_deg_s == 0.0
    assert model.max_lateral_accel_mps2 == 0.0


def test_quarter_circle_derives_approximately_its_radius():
    radius = 10.0
    theta = np.linspace(0.0, np.pi / 2.0, 201)
    xyz = np.column_stack(
        (
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.zeros_like(theta),
        )
    )

    model = derive_velocity_model(
        _trajectory(xyz),
        speed_mps=5.0,
        path_resolution_m=0.10,
    )

    assert 8.0 <= model.min_turn_radius_m <= 12.5
    assert model.max_turn_rate_deg_s > 0.0
    assert model.max_lateral_accel_mps2 > 0.0


def test_climb_angle_and_rate_are_derived_from_3d_path():
    angle_deg = 30.0
    angle_rad = math.radians(angle_deg)
    x = np.linspace(0.0, 20.0, 101)
    z = np.tan(angle_rad) * x
    xyz = np.column_stack((x, np.zeros_like(x), z))

    model = derive_velocity_model(
        _trajectory(xyz),
        speed_mps=4.0,
        path_resolution_m=0.25,
    )

    assert np.isclose(model.max_climb_angle_deg, 30.0, atol=0.2)
    assert np.isclose(model.max_climb_rate_mps, 2.0, atol=0.03)
    assert model.max_descent_angle_deg == 0.0
    assert model.max_descent_rate_mps == 0.0


def test_descent_angle_and_rate_are_reported_positive_magnitudes():
    angle_rad = math.radians(20.0)
    x = np.linspace(0.0, 20.0, 101)
    z = 10.0 - np.tan(angle_rad) * x
    xyz = np.column_stack((x, np.zeros_like(x), z))

    model = derive_velocity_model(
        _trajectory(xyz),
        speed_mps=6.0,
        path_resolution_m=0.25,
    )

    assert np.isclose(model.max_descent_angle_deg, 20.0, atol=0.2)
    assert np.isclose(
        model.max_descent_rate_mps,
        6.0 * math.sin(angle_rad),
        atol=0.03,
    )
