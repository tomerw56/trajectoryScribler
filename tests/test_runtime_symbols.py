from __future__ import annotations

from pathlib import Path

import numpy as np

from trajectory_app.core import fixed_velocity_covariance_from_model
from trajectory_app.models import DerivedVelocityModel


def test_main_window_contains_prediction_visibility_handler_and_covariance_import():
    source = (
        Path(__file__).resolve().parents[1]
        / "trajectory_app"
        / "main_window.py"
    ).read_text(encoding="utf-8")

    assert "def _prediction_visibility_changed" in source
    assert "fixed_velocity_covariance_from_model," in source


def test_fixed_velocity_covariance_function_imports_and_runs():
    model = DerivedVelocityModel(
        speed_mps=4.0,
        min_turn_radius_m=10.0,
        max_turn_rate_deg_s=20.0,
        max_lateral_accel_mps2=1.0,
        max_climb_angle_deg=5.0,
        max_descent_angle_deg=5.0,
        max_climb_rate_mps=0.5,
        max_descent_rate_mps=0.5,
    )

    covariance = fixed_velocity_covariance_from_model(model)

    assert isinstance(covariance, np.ndarray)
    assert covariance.shape == (3, 3)
    assert np.all(np.isfinite(covariance))
