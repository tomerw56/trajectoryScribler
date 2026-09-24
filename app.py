from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGE = (PROJECT_ROOT / "trajectory_app").resolve()

# Force the package next to this launcher to be the first import location.
project_root_text = str(PROJECT_ROOT)
if not sys.path or Path(sys.path[0] or ".").resolve() != PROJECT_ROOT:
    sys.path.insert(0, project_root_text)

from trajectory_app.logging_config import configure_logging
from trajectory_app import core as _core
from trajectory_app import main_window as _main_window
from trajectory_app import covariance_stability as _covariance_stability

VERSION = "6.6-covariance-stability"


def _verify_runtime_sources() -> None:
    expected_core = (LOCAL_PACKAGE / "core.py").resolve()
    expected_main_window = (LOCAL_PACKAGE / "main_window.py").resolve()
    actual_core = Path(_core.__file__).resolve()
    actual_main_window = Path(_main_window.__file__).resolve()

    print(f"Trajectory Scribbler {VERSION}")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"core.py      : {actual_core}")
    print(f"main_window  : {actual_main_window}")

    if actual_core != expected_core or actual_main_window != expected_main_window:
        raise RuntimeError(
            "Python imported a different trajectory_app package than the one "
            "beside app.py.\n"
            f"Expected core.py: {expected_core}\n"
            f"Loaded core.py:   {actual_core}\n"
            f"Expected main_window.py: {expected_main_window}\n"
            f"Loaded main_window.py:   {actual_main_window}"
        )

    if not hasattr(_core, "fixed_velocity_covariance_from_model"):
        raise RuntimeError(
            "Local core.py is missing fixed_velocity_covariance_from_model: "
            f"{actual_core}"
        )

    if not hasattr(_main_window.MainWindow, "_prediction_visibility_changed"):
        raise RuntimeError(
            "Local MainWindow is missing _prediction_visibility_changed: "
            f"{actual_main_window}"
        )

    if not hasattr(_covariance_stability, "CovarianceStabilityCalculator"):
        raise RuntimeError(
            "Local covariance_stability.py is missing CovarianceStabilityCalculator"
        )

    if not hasattr(_main_window.MainWindow, "get_covariance_stability"):
        raise RuntimeError(
            "Local MainWindow is missing get_covariance_stability: "
            f"{actual_main_window}"
        )

    print("Runtime source verification: PASS")


if __name__ == "__main__":
    configure_logging()
    _verify_runtime_sources()
    _main_window.run_app()
