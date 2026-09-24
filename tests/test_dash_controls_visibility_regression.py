from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dropdown_theme_supports_generated_react_select_classes():
    css = (
        ROOT / "trajectory_dash" / "assets" / "style.css"
    ).read_text(encoding="utf-8")

    assert '[class*="-control"]' in css
    assert '[class*="-singleValue"]' in css
    assert '[class*="-menu"]' in css
    assert '[class*="-option"]' in css
    assert '[role="option"][aria-selected="true"]' in css

    # Control background and selected text must be opposite/high contrast.
    assert "background-color: #202732 !important;" in css
    assert "color: #ffffff !important;" in css


def test_prediction_checkboxes_have_visible_dark_theme_style():
    css = (
        ROOT / "trajectory_dash" / "assets" / "style.css"
    ).read_text(encoding="utf-8")

    assert '.prediction-checklist input[type="checkbox"]' in css
    assert "accent-color: #72b7ff;" in css
    assert "outline: 1px solid #dbe7f4;" in css


def test_prediction_checklist_class_is_used_by_layout():
    source = (
        ROOT / "trajectory_dash" / "layout.py"
    ).read_text(encoding="utf-8")

    assert 'className="prediction-checklist"' in source
