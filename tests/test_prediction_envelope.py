from __future__ import annotations

import math
import numpy as np

from trajectory_app.core import fixed_velocity_covariance_from_model
from trajectory_app.models import DerivedVelocityModel, MotionConfig, TrajectoryConfig
from trajectory_app.solver import PrincipalSigmaDemoSolver, SolverRequest


def test_principal_sigma_solver_uses_forward_center_plus_offset():
    solver = PrincipalSigmaDemoSolver()
    request = SolverRequest(
        target_id="t",
        target_name="Target",
        sample_index=0,
        time_sec=0.0,
        current_position=np.array([1.0, 2.0, 3.0]),
        current_velocity=np.array([4.0, 0.0, 0.0]),
        velocity_covariance=np.diag([4.0, 1.0, 0.25]),
        terrain_height_m=0.0,
        motion_config=MotionConfig(
            speed_mps=4.0,
            sample_dt_sec=0.25,
            covariance_sample_rate_sec=0.5,
            covariance_window_sec=3.0,
            prediction_dt_sec=2.5,
        ),
        trajectory_config=TrajectoryConfig(),
    )
    result = solver.solve(request)
    assert np.allclose(result.estimated_position, np.array([16.0, 2.0, 3.0]))
    assert math.isclose(result.metadata["horizon_sec"], 2.5)
    assert math.isclose(result.metadata["offset_m"], 5.0)


def test_fixed_covariance_from_model_is_expected_diagonal():
    model = DerivedVelocityModel(
        speed_mps=6.0,
        min_turn_radius_m=10.0,
        max_turn_rate_deg_s=30.0,
        max_lateral_accel_mps2=3.0,
        max_climb_angle_deg=12.0,
        max_descent_angle_deg=9.0,
        max_climb_rate_mps=1.5,
        max_descent_rate_mps=1.0,
    )
    covariance = fixed_velocity_covariance_from_model(
        model,
        reference_dt_sec=1.0,
        sigma_factor=3.0,
    )
    sigma_xy = 6.0 * math.sin(math.radians(30.0)) / 3.0
    sigma_z = 1.5 / 3.0
    assert np.allclose(covariance, np.diag([sigma_xy**2, sigma_xy**2, sigma_z**2]))
