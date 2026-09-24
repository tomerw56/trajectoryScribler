from __future__ import annotations

import math

import numpy as np

from trajectory_app.core import fixed_velocity_covariance_from_model
from trajectory_app.models import (
    DerivedVelocityModel,
    MotionConfig,
    TrajectoryConfig,
)
from trajectory_app.solver import PrincipalSigmaDemoSolver, SolverRequest


def _model(
    *,
    speed=6.0,
    turn_rate=30.0,
    climb_rate=1.5,
    descent_rate=0.75,
):
    return DerivedVelocityModel(
        speed_mps=speed,
        min_turn_radius_m=10.0,
        max_turn_rate_deg_s=turn_rate,
        max_lateral_accel_mps2=1.0,
        max_climb_angle_deg=10.0,
        max_descent_angle_deg=8.0,
        max_climb_rate_mps=climb_rate,
        max_descent_rate_mps=descent_rate,
    )


def test_fixed_covariance_matches_documented_rule():
    model = _model()
    covariance = fixed_velocity_covariance_from_model(
        model,
        reference_dt_sec=1.0,
        sigma_factor=3.0,
    )

    sigma_xy = 6.0 * math.sin(math.radians(30.0)) / 3.0
    sigma_z = 1.5 / 3.0

    assert covariance.shape == (3, 3)
    assert np.allclose(
        covariance,
        np.diag([sigma_xy**2, sigma_xy**2, sigma_z**2]),
    )


def test_fixed_covariance_is_heading_independent_and_diagonal():
    covariance = fixed_velocity_covariance_from_model(_model())
    assert np.allclose(covariance, covariance.T)
    assert np.allclose(covariance - np.diag(np.diag(covariance)), 0.0)
    assert np.isclose(covariance[0, 0], covariance[1, 1])


def test_straight_level_model_produces_zero_fixed_covariance():
    model = _model(
        turn_rate=0.0,
        climb_rate=0.0,
        descent_rate=0.0,
    )
    covariance = fixed_velocity_covariance_from_model(model)
    assert np.allclose(covariance, np.zeros((3, 3)))
