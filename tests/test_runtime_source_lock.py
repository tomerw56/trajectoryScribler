from pathlib import Path


def test_required_runtime_symbols_are_in_local_sources():
    root = Path(__file__).resolve().parents[1]
    core = (root / "trajectory_app" / "core.py").read_text(encoding="utf-8")
    main_window = (root / "trajectory_app" / "main_window.py").read_text(encoding="utf-8")

    assert "def fixed_velocity_covariance_from_model(" in core
    assert "def _prediction_visibility_changed(" in main_window
    assert "fixed_velocity_covariance_from_model," in main_window


def test_covariance_stability_runtime_module_exists():
    root = Path(__file__).resolve().parents[1]
    source = (root / "trajectory_app" / "covariance_stability.py").read_text(
        encoding="utf-8"
    )
    assert "class CovarianceStabilityCalculator" in source
    assert "class CovarianceStabilityState" in source
