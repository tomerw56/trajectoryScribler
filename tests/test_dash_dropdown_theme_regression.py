from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dropdowns_do_not_use_inline_white_text_style():
    source = (ROOT / "trajectory_dash" / "layout.py").read_text(encoding="utf-8")

    assert 'style={"color": "#ffffff"}' not in source


def test_dark_dropdown_theme_defines_both_background_and_selected_text():
    css = (
        ROOT / "trajectory_dash" / "assets" / "style.css"
    ).read_text(encoding="utf-8")

    assert "Phase 01.3 dropdown fix" in css
    assert "background-color: #202732 !important;" in css
    assert ".dark-dropdown .Select-value-label" in css
    assert "color: #ffffff !important;" in css


def test_current_dropdowns_all_use_dark_dropdown_class():
    source = (ROOT / "trajectory_dash" / "layout.py").read_text(encoding="utf-8")

    for dropdown_id in (
        "scenario-select",
        "active-target-select",
        "covariance-scope-select",
    ):
        marker = f'id="{dropdown_id}"'
        index = source.index(marker)
        nearby = source[index:index + 350]
        assert 'className="dark-dropdown"' in nearby
