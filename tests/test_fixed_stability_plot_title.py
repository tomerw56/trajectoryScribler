from __future__ import annotations

from pathlib import Path


def test_covariance_stability_uses_fixed_plot_title_not_data_text_item():
    source = (
        Path(__file__).resolve().parents[1]
        / "trajectory_app"
        / "main_window.py"
    ).read_text(encoding="utf-8")

    assert "self.cov_stability_title = QLabel" in source
    assert "covariance_layout.addWidget(self.cov_stability_title)" in source
    assert "self.cov_stability_caption = pg.TextItem" not in source
    assert "target.name" in source
    assert "drift=" in source
    assert "rms=" in source
    assert "max=" in source
