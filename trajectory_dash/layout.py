from __future__ import annotations

from dash import dcc, html

from trajectory_app.cnc_solver import prediction_offsets_sec
from trajectory_app.covariance_policy import CovarianceScope

from .runtime import ScenarioRepository


def build_layout(repository: ScenarioRepository):
    choices = repository.choices()
    default_path = str(repository.default_path())
    default_scenario = repository.load(default_path)
    active_target = default_scenario.targets[0].id if default_scenario.targets else None
    default_prediction_horizon = (
        float(default_scenario.targets[0].motion_config.prediction_dt_sec)
        if default_scenario.targets
        else 2.0
    )
    default_solver_dt = 1.0
    default_offsets = prediction_offsets_sec(
        default_prediction_horizon,
        default_solver_dt,
    )
    default_display_steps = min(2, len(default_offsets))

    return html.Div(
        className="app-shell",
        children=[
            dcc.Store(
                id="playback-state",
                data={"playing": False},
            ),
            dcc.Store(
                id="cnc-last-execution",
                data=None,
            ),
            dcc.Store(
                id="cnc-selection-state",
                data={},
            ),
            dcc.Store(
                id="cnc-accepted-state",
                data={},
            ),
            dcc.Store(
                id="track-state",
                data={"enabled": False},
            ),
            dcc.Interval(id="playback-interval", interval=250, disabled=True),
            html.Div(
                className="top-toolbar",
                children=[
                    html.Div(
                        className="scenario-picker",
                        children=[
                            html.Div("Trajectory project", className="control-label"),
                            dcc.Dropdown(
                                id="scenario-select",
                                options=choices,
                                value=default_path,
                                clearable=False,
                                className="friendly-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="transport-controls",
                        children=[
                            html.Button("|◀", id="first-button", title="First"),
                            html.Button("◀", id="prev-button", title="Previous sample"),
                            html.Button("▶", id="play-button", title="Play / pause"),
                            html.Button("▶", id="next-button", title="Next sample"),
                            html.Button("▶|", id="last-button", title="Last"),
                            html.Button(
                                "◎ Track",
                                id="track-button",
                                title="Follow the currently selected bird",
                                className="track-button",
                            ),
                        ],
                    ),
                    html.Div(id="time-readout", className="time-readout", children="t = 0.00 s"),
                ],
            ),
            html.Div(
                className="timeline-row",
                children=[
                    dcc.Slider(
                        id="time-slider",
                        min=0.0,
                        max=default_scenario.max_time_sec,
                        step=default_scenario.playback_step_sec,
                        value=0.0,
                        tooltip={"placement": "bottom", "always_visible": False},
                    )
                ],
            ),
            html.Div(
                className="workspace-row",
                children=[
                    html.Div(
                        className="cnc-panel",
                        children=[
                            html.Div("CnC Manager", className="panel-title"),
                            html.Div(
                                "Scenario solver",
                                className="cnc-section-title",
                            ),
                            html.Div(
                                "Prediction time (s)",
                                className="control-label",
                            ),
                            dcc.Input(
                                id="prediction-horizon-input",
                                type="number",
                                min=0.25,
                                max=30.0,
                                step=0.25,
                                value=default_prediction_horizon,
                                debounce=False,
                                className="numeric-updown",
                            ),
                            html.Div(
                                "Solver dt (s)",
                                className="control-label spaced",
                            ),
                            dcc.Input(
                                id="solver-dt-input",
                                type="number",
                                min=0.01,
                                max=30.0,
                                step=0.25,
                                value=default_solver_dt,
                                debounce=False,
                                className="numeric-updown",
                            ),
                            html.Div(
                                "Future steps to display",
                                className="control-label spaced",
                            ),
                            dcc.Dropdown(
                                id="cnc-display-steps",
                                options=[
                                    {
                                        "label": (
                                            f"{index + 1} "
                                            f"(through {'t0' if abs(offset) <= 1e-12 else f't+{offset:g}s'})"
                                        ),
                                        "value": index + 1,
                                    }
                                    for index, offset in enumerate(default_offsets)
                                ],
                                value=default_display_steps,
                                clearable=False,
                                className="friendly-dropdown",
                            ),
                            html.Div(
                                "Solution time budget (s)",
                                className="control-label spaced",
                            ),
                            dcc.Input(
                                id="solution-time-input",
                                type="number",
                                min=0.01,
                                max=600.0,
                                step=0.10,
                                value=1.0,
                                debounce=False,
                                className="numeric-updown",
                            ),
                            html.Button(
                                "Solve",
                                id="cnc-solve-button",
                                className="cnc-solve-button",
                                n_clicks=0,
                                title="Capture the current scenario and call the CnC solver.",
                            ),
                            html.Div(
                                id="cnc-result-panel",
                                className="cnc-result-panel",
                                children=[
                                    html.Div(
                                        "No solve has been run yet.",
                                        className="cnc-result-empty",
                                    )
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="plot-panel",
                        children=[
                            dcc.Graph(
                                id="terrain-graph",
                                config={
                                    "displaylogo": False,
                                    "scrollZoom": True,
                                    "responsive": True,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                },
                                responsive=True,
                                style={"height": "100%", "width": "100%"},
                            )
                        ],
                    ),
                ],
            ),
            html.Div(
                className="control-dock",
                children=[
                    html.Div("Control", className="dock-title"),
                    dcc.Tabs(
                        id="control-tabs",
                        value="active-sample",
                        children=[
                            dcc.Tab(
                                label="Active Target",
                                value="active-sample",
                                children=[
                                    html.Div(
                                        className="active-tab-grid",
                                        children=[
                                            html.Div(
                                                className="control-column",
                                                children=[
                                                    html.Div("Active target", className="control-label"),
                                                    dcc.Dropdown(
                                                        id="active-target-select",
                                                        className="friendly-dropdown",
                                                                                options=[
                                                            {"label": target.name, "value": target.id}
                                                            for target in default_scenario.targets
                                                        ],
                                                        value=active_target,
                                                        clearable=False,
                                                    ),
                                                    html.Div("Covariance mode", className="control-label spaced"),
                                                    dcc.Dropdown(
                                                        id="covariance-scope-select",
                                                        className="friendly-dropdown",
                                                        options=[
                                                            {"label": "Prediction XY", "value": CovarianceScope.PREDICTION_XY.value},
                                                            {"label": "Full 3D", "value": CovarianceScope.FULL_3D.value},
                                                        ],
                                                        value=default_scenario.covariance_scope.value,
                                                        clearable=False,
                                                    ),
                                                    dcc.Checklist(
                                                        id="prediction-visibility",
                                                        options=[
                                                            {"label": "Rolling prediction", "value": "rolling"},
                                                            {"label": "Fixed-model prediction", "value": "fixed"},
                                                        ],
                                                        value=["rolling", "fixed"],
                                                        className="prediction-checklist",
                                                    ),
                                                ],
                                            ),
                                            html.Div(id="active-sample-panel", className="sample-panel"),
                                        ],
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="Solver Metry",
                                value="solver-metry",
                                children=[
                                    html.Div(
                                        id="solver-metry-panel",
                                        className="solver-metry-panel",
                                        children=[
                                            html.Div(
                                                "Run the CnC solver to populate performance metrics.",
                                                className="future-tab-placeholder",
                                            )
                                        ],
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="Telemetry (next)",
                                value="telemetry",
                                children=[
                                    html.Div(
                                        "Reserved for telemetry/history charts and derived metrics.",
                                        className="future-tab-placeholder",
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
