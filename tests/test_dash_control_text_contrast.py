from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dropdown_controls_use_light_surface_and_dark_text():
    css = (
        ROOT / "trajectory_dash" / "assets" / "style.css"
    ).read_text(encoding="utf-8")

    assert "Phase 01.5 — input controls use a light surface" in css
    assert "background-color: #f7f9fc !important;" in css
    assert "color: #111820 !important;" in css


def test_slider_tooltip_uses_dark_text_on_light_background():
    css = (
        ROOT / "trajectory_dash" / "assets" / "style.css"
    ).read_text(encoding="utf-8")

    assert ".timeline-row .rc-slider-tooltip-inner" in css
    assert "background-color: #f7f9fc !important;" in css
    assert "color: #111820 !important;" in css
