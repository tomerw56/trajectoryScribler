from __future__ import annotations

from pathlib import Path


def _source(name: str) -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "trajectory_app"
        / name
    ).read_text(encoding="utf-8")


def test_main_window_has_one_system_scope_selector_and_persists_it():
    source = _source("main_window.py")

    assert 'self.covariance_scope = CovarianceScope.PREDICTION_XY' in source
    assert 'self.covariance_scope_combo = QComboBox()' in source
    assert 'self._covariance_scope_changed' in source
    assert '"covariance_scope": self.covariance_scope.value' in source
    assert 'data.get(' in source
    assert '"covariance_scope"' in source


def test_solver_receives_scoped_covariance_and_scope_metadata():
    source = _source("main_window.py")

    assert 'active_covariance = scoped_velocity_covariance(' in source
    assert 'velocity_covariance=active_covariance.copy()' in source
    assert 'covariance_scope=self.covariance_scope' in source


def test_canvas_applies_the_same_scope_before_drawing_ellipse():
    source = _source("canvas.py")

    assert 'self.covariance_scope = CovarianceScope.PREDICTION_XY' in source
    assert 'def set_covariance_scope(' in source
    assert 'active_covariance = scoped_velocity_covariance(' in source
    assert 'self.covariance_scope.ellipse_label' in source


def test_covariance_plot_excludes_z_in_xy_mode():
    source = _source("main_window.py")

    assert 'self.czz_curve.setVisible(False)' in source
    assert 'np.asarray(covariance.cxx) + np.asarray(covariance.cyy)' in source
    assert 'self.czz_curve.setVisible(True)' in source
