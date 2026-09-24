from __future__ import annotations

from pathlib import Path
import argparse
import os

import numpy as np
from dash import ALL, Dash, Input, Output, State, ctx, html, no_update

from trajectory_app.cnc_manager import CnCManager, CnCSolveExecution
from trajectory_app.cnc_solver import (
    CnCSolver,
    SnapshotInspectionSolver,
    prediction_offsets_sec,
)
from trajectory_app.covariance_policy import CovarianceScope, scoped_velocity_covariance
from trajectory_app.covariance_stability import CovarianceStabilityState
from trajectory_app.logging_config import configure_logging, get_logger

from .analysis import solver_snapshot
from .controller import PlaybackState, apply_playback_event
from .layout import build_layout
from .plotting import STATE_LABELS, build_terrain_figure, stability_for_target
from .runtime import ScenarioRepository

logger = get_logger(__name__)


def _matrix_text(matrix: np.ndarray, scope: CovarianceScope) -> str:
    matrix = scoped_velocity_covariance(matrix, scope)
    if scope == CovarianceScope.PREDICTION_XY:
        return (
            f"[[{matrix[0,0]:.4f}, {matrix[0,1]:.4f}], "
            f"[{matrix[1,0]:.4f}, {matrix[1,1]:.4f}]]"
        )
    return np.array2string(matrix, precision=4, suppress_small=True)


def _sample_panel(
    scenario,
    target_id: str | None,
    time_sec: float,
    scope: CovarianceScope,
    prediction_horizon_sec: float | None = None,
):
    target = scenario.target_by_id(target_id)
    if target is None or target.samples is None:
        return html.Div("No active target sample.", className="empty-sample")

    index = scenario.target_index_at_time(target, time_sec)
    samples = target.samples
    sample_time = float(samples.time_sec[index])
    position = samples.position[index]
    velocity = samples.velocity[index]
    rolling_cov = scenario.covariance_at_time(target, sample_time)
    fixed_cov = target.fixed_velocity_covariance
    stability = stability_for_target(scenario, target, sample_time, scope)
    horizon = (
        float(target.motion_config.prediction_dt_sec)
        if prediction_horizon_sec is None
        else max(0.25, float(prediction_horizon_sec))
    )
    solver_data = solver_snapshot(
        scenario,
        target,
        sample_time,
        scope,
        prediction_horizon_sec=horizon,
    )

    if stability is None:
        state_text = "—"
        state_class = "reliability-unknown"
        metrics = "No covariance stability data"
    else:
        state_text = STATE_LABELS[stability.state]
        state_class = f"reliability-{stability.state.value}"
        metrics = (
            f"drift {stability.drift:.3f} | RMS {stability.rms_change:.3f} | "
            f"max {stability.max_single_jump:.3f} | large {stability.large_change_fraction:.0%} | "
            f"rough {stability.roughness:.3f}"
        )

    rolling_scoped = scoped_velocity_covariance(rolling_cov, scope)
    fixed_scoped = (
        scoped_velocity_covariance(fixed_cov, scope)
        if fixed_cov is not None
        else None
    )
    covariance_delta = (
        np.diag(fixed_scoped) - np.diag(rolling_scoped)
        if fixed_scoped is not None
        else None
    )

    def point_text(point):
        if point is None:
            return "—"
        return f"({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}) m"

    rows = [
        ("Sample", f"{index + 1}/{len(samples)}"),
        ("Global / sample time", f"{time_sec:.2f} / {sample_time:.2f} s"),
        ("Position", f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) m"),
        ("|V|", f"{np.linalg.norm(velocity):.3f} m/s"),
        ("Components", f"Vx={velocity[0]:.3f}, Vy={velocity[1]:.3f}, Vz={velocity[2]:.3f}"),
        ("Rolling covariance", _matrix_text(rolling_cov, scope)),
        ("Fixed model covariance", _matrix_text(fixed_cov, scope) if fixed_cov is not None else "—"),
        (
            "Covariance diag Δ",
            (
                np.array2string(covariance_delta, precision=4, suppress_small=True)
                if covariance_delta is not None
                else "—"
            ),
        ),
        ("Solver", solver_data.solver_name),
        ("Rolling-cov estimate", point_text(solver_data.rolling_estimate)),
        ("Fixed-cov estimate", point_text(solver_data.fixed_estimate)),
        (
            "Estimate difference",
            f"{solver_data.estimate_difference_m:.3f} m"
            if solver_data.estimate_difference_m is not None
            else "—",
        ),
        (
            f"Rolling estimate error @ +{horizon:.2f}s",
            f"{solver_data.rolling_error_m:.3f} m"
            if solver_data.rolling_error_m is not None
            else "n/a (past end)",
        ),
        (
            "Position MSE",
            f"{solver_data.mse_m2:.3f} m²" if solver_data.mse_m2 is not None else "—",
        ),
        (
            "Position RMSE",
            f"{solver_data.rmse_m:.3f} m" if solver_data.rmse_m is not None else "—",
        ),
        ("Evaluated samples", str(solver_data.evaluated_samples)),
        ("Prediction horizon", f"{horizon:.2f} s"),
    ]

    return html.Div(
        children=[
            html.Div(
                className="reliability-row",
                children=[
                    html.Div("Covariance reliability", className="sample-heading"),
                    html.Div(state_text, className=f"reliability-badge {state_class}"),
                    html.Div(metrics, className="reliability-metrics"),
                ],
            ),
            html.Div(
                className="sample-grid",
                children=[
                    html.Div(
                        className="sample-field",
                        children=[
                            html.Div(label, className="sample-key"),
                            html.Div(value, className="sample-value"),
                        ],
                    )
                    for label, value in rows
                ],
            ),
        ]
    )



def _serialize_cnc_execution(execution: CnCSolveExecution) -> dict:
    return {
        "solution_id": execution.solution_id,
        "solver_name": execution.solver_name,
        "status": execution.result.status,
        "summary": execution.result.summary,
        "solve_elapsed_sec": float(execution.solve_elapsed_sec),
        "exceeded_solution_time": bool(execution.exceeded_solution_time),
        "scenario_time_sec": float(execution.request.scenario_time_sec),
        "prediction_time_sec": float(execution.request.prediction_time_sec),
        "solver_dt_sec": float(execution.request.solver_dt_sec),
        "solution_time_sec": float(execution.request.solution_time_sec),
        "prediction_offsets_sec": [
            float(value)
            for value in execution.request.prediction_offsets_sec
        ],
        "birds": [
            {
                "id": bird.target_id,
                "name": bird.target_name,
            }
            for bird in execution.request.birds
        ],
        "cameramen": [
            {
                "id": cameraman.cameraman_id,
                "name": cameraman.cameraman_name,
            }
            for cameraman in execution.request.cameramen
        ],
        "probabilities": [
            {
                "cameraman_id": item.cameraman_id,
                "cameraman_name": item.cameraman_name,
                "target_id": item.target_id,
                "target_name": item.target_name,
                "prediction_offset_sec": float(item.prediction_offset_sec),
                "observation_probability": float(item.observation_probability),
            }
            for item in execution.result.observation_probabilities
        ],
    }



def _offset_label(offset: float) -> str:
    return "t0" if abs(float(offset)) <= 1e-12 else f"t+{float(offset):g}s"


def _offset_key(offset: float) -> str:
    return f"{float(offset):.12g}"


def _candidate_matches(left: dict | None, right: dict | None) -> bool:
    if not left or not right:
        return False
    return (
        left.get("solution_id") == right.get("solution_id")
        and left.get("cameraman_id") == right.get("cameraman_id")
        and left.get("target_id") == right.get("target_id")
        and abs(
            float(left.get("prediction_offset_sec", 0.0))
            - float(right.get("prediction_offset_sec", 0.0))
        )
        <= 1e-9
    )


def _current_candidate(
    state: dict | None,
    cameraman_id: str,
    solution_id: str,
) -> dict | None:
    candidate = (state or {}).get(cameraman_id)
    if not candidate or candidate.get("solution_id") != solution_id:
        return None
    return candidate



def _solver_metry_panel(execution_data: dict | None):
    if not execution_data:
        return html.Div(
            "Run the CnC solver to populate performance metrics.",
            className="future-tab-placeholder",
        )

    within_budget = not bool(execution_data["exceeded_solution_time"])
    budget_class = (
        "solver-metry-budget-ok"
        if within_budget
        else "solver-metry-budget-exceeded"
    )
    budget_text = "within budget" if within_budget else "EXCEEDED"

    prediction_count = len(execution_data.get("prediction_offsets_sec", []))
    metrics = [
        ("Status", execution_data["status"].upper()),
        ("Solver", execution_data["solver_name"]),
        ("Scenario time", f'{execution_data["scenario_time_sec"]:.3f} s'),
        ("Birds", str(len(execution_data.get("birds", [])))),
        ("Cameramen", str(len(execution_data.get("cameramen", [])))),
        ("Prediction times", str(prediction_count)),
        ("Prediction time", f'{execution_data["prediction_time_sec"]:.3f} s'),
        ("Solver dt", f'{execution_data["solver_dt_sec"]:.3f} s'),
        ("Solution budget", f'{execution_data["solution_time_sec"]:.3f} s'),
        (
            "Measured solve()",
            f'{execution_data["solve_elapsed_sec"] * 1000.0:.3f} ms',
        ),
        ("Budget", budget_text),
    ]

    return html.Div(
        className="solver-metry-content",
        children=[
            html.Div(
                className="solver-metry-header",
                children=[
                    html.Div(
                        "Solver execution",
                        className="solver-metry-title",
                    ),
                    html.Div(
                        execution_data.get("summary", ""),
                        className="solver-metry-summary",
                    ),
                ],
            ),
            html.Div(
                className="solver-metry-grid",
                children=[
                    html.Div(
                        className=(
                            "solver-metry-card "
                            + (
                                budget_class
                                if label == "Budget"
                                else ""
                            )
                        ).strip(),
                        children=[
                            html.Div(label, className="solver-metry-key"),
                            html.Div(value, className="solver-metry-value"),
                        ],
                    )
                    for label, value in metrics
                ],
            ),
            html.Div(
                f'Solution ID: {execution_data["solution_id"]}',
                className="solver-metry-solution-id",
            ),
        ],
    )


def _cnc_execution_panel(
    execution_data: dict | None,
    display_steps: int | None,
    selection_state: dict | None = None,
    accepted_state: dict | None = None,
):
    if not execution_data:
        return html.Div(
            "No solve has been run yet.",
            className="cnc-result-empty",
        )

    solution_id = execution_data["solution_id"]
    offsets = [
        float(value)
        for value in execution_data.get("prediction_offsets_sec", [])
    ]
    if not offsets:
        visible_offsets: list[float] = []
    else:
        requested_count = int(display_steps or 1)
        requested_count = max(1, min(requested_count, len(offsets)))
        visible_offsets = offsets[:requested_count]

    probability_lookup = {
        (
            item["cameraman_id"],
            item["target_id"],
            float(item["prediction_offset_sec"]),
        ): float(item["observation_probability"])
        for item in execution_data.get("probabilities", [])
    }

    accepted_summary_rows = []
    for cameraman in execution_data.get("cameramen", []):
        accepted = _current_candidate(
            accepted_state,
            cameraman["id"],
            solution_id,
        )
        if accepted is None:
            accepted_text = "— not assigned"
            row_class = "cnc-accepted-summary-row cnc-accepted-summary-empty"
        else:
            accepted_text = (
                f'{accepted["target_name"]} · '
                f'{100.0 * float(accepted["observation_probability"]):.1f}% '
                f'@ {_offset_label(float(accepted["prediction_offset_sec"]))}'
            )
            row_class = "cnc-accepted-summary-row"

        accepted_summary_rows.append(
            html.Div(
                className=row_class,
                children=[
                    html.Div(
                        cameraman["name"],
                        className="cnc-accepted-camera",
                    ),
                    html.Div(
                        accepted_text,
                        className="cnc-accepted-choice",
                    ),
                ],
            )
        )

    camera_blocks = []
    for cameraman in execution_data.get("cameramen", []):
        camera_id = cameraman["id"]
        selected = _current_candidate(
            selection_state,
            camera_id,
            solution_id,
        )
        accepted = _current_candidate(
            accepted_state,
            camera_id,
            solution_id,
        )

        header_cells = [html.Th("Bird")]
        header_cells.extend(
            html.Th(_offset_label(offset))
            for offset in visible_offsets
        )

        body_rows = []
        for bird in execution_data.get("birds", []):
            target_id = bird["id"]
            selected_row = bool(
                selected and selected.get("target_id") == target_id
            )
            accepted_row = bool(
                accepted and accepted.get("target_id") == target_id
            )
            row_classes = ["cnc-probability-row"]
            if selected_row:
                row_classes.append("cnc-probability-row-selected")
            if accepted_row:
                row_classes.append("cnc-probability-row-accepted")

            context_buttons = [
                html.Button(
                    "",
                    id={
                        "type": f"cnc-context-{action}",
                        "camera": camera_id,
                        "target": target_id,
                    },
                    n_clicks=0,
                    className="cnc-context-trigger",
                    **{
                        "data-cnc-action": action,
                        "data-camera-id": camera_id,
                        "data-target-id": target_id,
                    },
                )
                for action in ("active", "track", "best")
            ]

            cells = [
                html.Td(
                    children=[
                        html.Span(bird["name"]),
                        html.Span("⋮", className="cnc-bird-context-hint"),
                        *context_buttons,
                    ],
                    className="cnc-bird-name-cell cnc-bird-context-target",
                    title="Right-click for bird actions",
                    **{
                        "data-camera-id": camera_id,
                        "data-camera-name": cameraman["name"],
                        "data-target-id": target_id,
                        "data-target-name": bird["name"],
                    },
                )
            ]

            for offset in visible_offsets:
                probability = probability_lookup.get(
                    (camera_id, target_id, float(offset))
                )

                candidate = (
                    None
                    if probability is None
                    else {
                        "solution_id": solution_id,
                        "cameraman_id": camera_id,
                        "cameraman_name": cameraman["name"],
                        "target_id": target_id,
                        "target_name": bird["name"],
                        "prediction_offset_sec": float(offset),
                        "observation_probability": float(probability),
                    }
                )

                is_selected = _candidate_matches(selected, candidate)
                is_accepted = _candidate_matches(accepted, candidate)

                cell_classes = ["cnc-probability-value", "cnc-probability-cell"]
                if is_selected:
                    cell_classes.append("cnc-probability-cell-selected")
                if is_accepted:
                    cell_classes.append("cnc-probability-cell-accepted")

                if probability is None:
                    text_value = "—"
                    cell_style = {}
                    cell_id = None
                else:
                    text_value = (
                        f"✓ {100.0 * probability:.1f}%"
                        if is_accepted
                        else f"{100.0 * probability:.1f}%"
                    )
                    alpha = 0.055 + 0.32 * float(probability)
                    cell_style = {
                        "backgroundColor": (
                            f"rgba(47, 128, 237, {alpha:.3f})"
                        )
                    }
                    cell_id = {
                        "type": "cnc-probability-cell",
                        "camera": camera_id,
                        "target": target_id,
                        "offset": _offset_key(offset),
                    }

                cell_kwargs = {
                    "children": text_value,
                    "className": " ".join(cell_classes),
                    "style": cell_style,
                    "title": (
                        "Click to select this observation"
                        if probability is not None
                        else "No probability"
                    ),
                }
                if cell_id is not None:
                    cell_kwargs["id"] = cell_id
                    cell_kwargs["n_clicks"] = 0

                cells.append(html.Td(**cell_kwargs))

            body_rows.append(
                html.Tr(
                    cells,
                    className=" ".join(row_classes),
                )
            )

        if selected is None:
            selection_text = "Click a percentage to select an observation."
            action_text = "Accept selection"
            action_disabled = True
        else:
            selection_text = (
                f'Selected: {selected["target_name"]} · '
                f'{_offset_label(float(selected["prediction_offset_sec"]))} · '
                f'{100.0 * float(selected["observation_probability"]):.1f}%'
            )
            action_disabled = False
            if _candidate_matches(selected, accepted):
                action_text = "Clear accepted"
            elif accepted is not None:
                action_text = "Change acceptance"
            else:
                action_text = "Accept selection"

        if accepted is None:
            accepted_banner = html.Div(
                "No accepted observation",
                className="cnc-camera-accepted-banner cnc-camera-accepted-empty",
            )
        else:
            accepted_banner = html.Div(
                (
                    f'✓ ACCEPTED: {accepted["target_name"]} · '
                    f'{100.0 * float(accepted["observation_probability"]):.1f}% '
                    f'@ {_offset_label(float(accepted["prediction_offset_sec"]))}'
                ),
                className="cnc-camera-accepted-banner",
            )

        camera_blocks.append(
            html.Div(
                className="cnc-camera-block",
                children=[
                    html.Div(
                        className="cnc-camera-title-row",
                        children=[
                            html.Div(
                                cameraman["name"],
                                className="cnc-camera-title",
                            ),
                            accepted_banner,
                        ],
                    ),
                    html.Div(
                        className="cnc-probability-scroll",
                        children=[
                            html.Table(
                                className="cnc-probability-table",
                                children=[
                                    html.Thead(html.Tr(header_cells)),
                                    html.Tbody(body_rows),
                                ],
                            )
                        ],
                    ),
                    html.Div(
                        className="cnc-camera-selection",
                        children=[
                            html.Div(
                                selection_text,
                                className="cnc-selection-text",
                            ),
                            html.Button(
                                action_text,
                                id={
                                    "type": "cnc-accept-button",
                                    "camera": camera_id,
                                },
                                n_clicks=0,
                                disabled=action_disabled,
                                className="cnc-accept-button",
                            ),
                        ],
                    ),
                ],
            )
        )

    probability_content = (
        camera_blocks
        if camera_blocks and visible_offsets
        else [
            html.Div(
                "No observation probabilities available.",
                className="cnc-probability-empty",
            )
        ]
    )

    return html.Div(
        children=[
            html.Div(
                className="cnc-accepted-summary",
                children=[
                    html.Div(
                        "Accepted observations",
                        className="cnc-section-title",
                    ),
                    *accepted_summary_rows,
                ],
            ),
            html.Div(
                className="cnc-probability-section",
                children=[
                    html.Div(
                        (
                            "Observation probability — "
                            f"showing {len(visible_offsets)} of "
                            f"{len(offsets)} prediction times"
                        ),
                        className="cnc-section-title",
                    ),
                    *probability_content,
                ],
            ),
        ]
    )


def create_app(
    root: str | Path | None = None,
    *,
    cnc_solver: CnCSolver | None = None,
) -> Dash:
    root_path = Path(root or Path(__file__).resolve().parents[1]).resolve()
    configure_logging(log_dir=root_path / "logs")
    repository = ScenarioRepository(root_path)
    cnc_manager = CnCManager(cnc_solver or SnapshotInspectionSolver())

    app = Dash(
        __name__,
        assets_folder=str(Path(__file__).resolve().parent / "assets"),
        title="Trajectory Playback",
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout(repository)

    @app.callback(
        Output("active-target-select", "options"),
        Output("active-target-select", "value"),
        Output("covariance-scope-select", "value"),
        Output("time-slider", "max"),
        Output("time-slider", "step"),
        Output("prediction-horizon-input", "value"),
        Input("scenario-select", "value"),
    )
    def scenario_changed(path_text: str):
        scenario = repository.load(path_text)
        options = [{"label": target.name, "value": target.id} for target in scenario.targets]
        active = scenario.targets[0].id if scenario.targets else None
        logger.info("Dash selected scenario: %s", scenario.path.name)
        default_horizon = (
            float(scenario.targets[0].motion_config.prediction_dt_sec)
            if scenario.targets
            else 2.0
        )
        return (
            options,
            active,
            scenario.covariance_scope.value,
            scenario.max_time_sec,
            scenario.playback_step_sec,
            default_horizon,
        )

    @app.callback(
        Output("playback-state", "data"),
        Output("playback-interval", "disabled"),
        Output("time-slider", "value"),
        Output("play-button", "children"),
        Input("scenario-select", "value"),
        Input("first-button", "n_clicks"),
        Input("prev-button", "n_clicks"),
        Input("play-button", "n_clicks"),
        Input("next-button", "n_clicks"),
        Input("last-button", "n_clicks"),
        Input("playback-interval", "n_intervals"),
        State("time-slider", "value"),
        State("playback-state", "data"),
        prevent_initial_call=False,
    )
    def playback_controller(
        scenario_path,
        _first,
        _prev,
        _play,
        _next,
        _last,
        _ticks,
        slider_value,
        state,
    ):
        scenario = repository.load(scenario_path)
        previous = state or {"playing": False}
        current = PlaybackState(
            time_sec=float(slider_value or 0.0),
            playing=bool(previous.get("playing", False)),
        )
        triggered = ctx.triggered_id
        updated = apply_playback_event(
            current,
            triggered,
            step_sec=scenario.playback_step_sec,
            max_time_sec=scenario.max_time_sec,
            slider_value=slider_value,
        )

        if triggered in {"first-button", "last-button", "play-button"}:
            logger.info(
                "Dash playback event=%s time=%.2f playing=%s",
                triggered,
                updated.time_sec,
                updated.playing,
            )

        return (
            {"playing": updated.playing},
            not updated.playing,
            float(updated.time_sec),
            "❚❚" if updated.playing else "▶",
        )

    @app.callback(
        Output("time-readout", "children"),
        Input("scenario-select", "value"),
        Input("time-slider", "value"),
    )
    def time_readout(scenario_path, slider_value):
        scenario = repository.load(scenario_path)
        time_sec = float(slider_value or 0.0)
        return f"t = {time_sec:.2f} / {scenario.max_time_sec:.2f} s"

    @app.callback(
        Output("track-state", "data"),
        Input("track-button", "n_clicks"),
        State("track-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_track(_n_clicks, state):
        enabled = not bool((state or {}).get("enabled", False))
        logger.info("Dash target tracking changed: enabled=%s", enabled)
        return {"enabled": enabled}

    @app.callback(
        Output("track-button", "children"),
        Output("track-button", "className"),
        Input("track-state", "data"),
    )
    def render_track_button(state):
        enabled = bool((state or {}).get("enabled", False))
        return (
            "✓ Tracking" if enabled else "◎ Track",
            (
                "track-button track-button-active"
                if enabled
                else "track-button"
            ),
        )

    @app.callback(
        Output("terrain-graph", "figure"),
        Output("active-sample-panel", "children"),
        Input("scenario-select", "value"),
        Input("time-slider", "value"),
        Input("active-target-select", "value"),
        Input("covariance-scope-select", "value"),
        Input("prediction-visibility", "value"),
        Input("prediction-horizon-input", "value"),
        Input("track-state", "data"),
    )
    def refresh_view(
        scenario_path,
        slider_value,
        target_id,
        scope_value,
        visibility,
        prediction_horizon,
        track_state,
    ):
        scenario = repository.load(scenario_path)
        time_sec = float(slider_value or 0.0)
        scope = CovarianceScope(scope_value or scenario.covariance_scope.value)
        visibility = set(visibility or [])
        target = scenario.target_by_id(target_id)
        configured_horizon = (
            float(target.motion_config.prediction_dt_sec)
            if target is not None
            else 2.0
        )
        try:
            horizon = float(prediction_horizon)
        except (TypeError, ValueError):
            horizon = configured_horizon
        horizon = min(30.0, max(0.25, horizon))

        if ctx.triggered_id == "prediction-horizon-input":
            logger.info("Dash prediction horizon changed: %.2f s", horizon)

        track_enabled = bool((track_state or {}).get("enabled", False))

        figure = build_terrain_figure(
            scenario,
            time_sec=time_sec,
            active_target_id=target_id,
            covariance_scope=scope,
            show_rolling_prediction="rolling" in visibility,
            show_fixed_prediction="fixed" in visibility,
            prediction_horizon_sec=horizon,
            track_active_target=track_enabled,
        )
        panel = _sample_panel(
            scenario,
            target_id,
            time_sec,
            scope,
            prediction_horizon_sec=horizon,
        )
        return figure, panel

    @app.callback(
        Output("cnc-display-steps", "options"),
        Output("cnc-display-steps", "value"),
        Input("prediction-horizon-input", "value"),
        Input("solver-dt-input", "value"),
        State("cnc-display-steps", "value"),
    )
    def update_cnc_display_step_options(
        prediction_time,
        solver_dt,
        current_value,
    ):
        try:
            prediction = min(30.0, max(0.25, float(prediction_time)))
        except (TypeError, ValueError):
            prediction = 2.0

        try:
            dt = min(30.0, max(0.01, float(solver_dt)))
        except (TypeError, ValueError):
            dt = 1.0

        offsets = prediction_offsets_sec(prediction, dt)
        options = [
            {
                "label": (
                    f"{index + 1} "
                    f"(through {'t0' if abs(offset) <= 1e-12 else f't+{offset:g}s'})"
                ),
                "value": index + 1,
            }
            for index, offset in enumerate(offsets)
        ]

        try:
            selected = int(current_value)
        except (TypeError, ValueError):
            selected = min(2, len(offsets))

        selected = max(1, min(selected, len(offsets)))
        return options, selected
    @app.callback(
        Input("cnc-generate-scenario-button", "n_clicks"),
        State("scenario-select", "value"),

        prevent_initial_call=True,
    )
    def genrate_scenario(_n_clicks,
            scenario_path):
            scenario = repository.load(scenario_path)
            k=0
                    

    @app.callback(
        Output("cnc-last-execution", "data"),
        Output("cnc-selection-state", "data", allow_duplicate=True),
        Output("cnc-accepted-state", "data", allow_duplicate=True),
        Input("cnc-solve-button", "n_clicks"),
        State("scenario-select", "value"),
        State("time-slider", "value"),
        State("prediction-horizon-input", "value"),
        State("solver-dt-input", "value"),
        State("solution-time-input", "value"),
        State("covariance-scope-select", "value"),
        prevent_initial_call=True,
    )
    def run_cnc_solver(
        _n_clicks,
        scenario_path,
        slider_value,
        prediction_time,
        solver_dt,
        solution_time,
        scope_value,
    ):
        scenario = repository.load(scenario_path)
        scenario_time = float(slider_value or 0.0)
        scope = CovarianceScope(scope_value or scenario.covariance_scope.value)

        try:
            prediction = float(prediction_time)
        except (TypeError, ValueError):
            prediction = 2.0
        prediction = min(30.0, max(0.25, prediction))

        try:
            dt = float(solver_dt)
        except (TypeError, ValueError):
            dt = 1.0
        dt = min(30.0, max(0.01, dt))

        try:
            solution_budget = float(solution_time)
        except (TypeError, ValueError):
            solution_budget = 1.0
        solution_budget = min(600.0, max(0.01, solution_budget))

        execution = cnc_manager.solve(
            scenario_name=scenario.path.stem,
            scenario_time_sec=scenario_time,
            prediction_time_sec=prediction,
            solver_dt_sec=dt,
            solution_time_sec=solution_budget,
            covariance_scope=scope,
            targets=scenario.targets,
            cameramen=scenario.cameramen,
            terrain=scenario.terrain,
            terrain_config=scenario.terrain_config,
        )

        logger.info(
            "Dash CnC solve displayed: solver=%s elapsed=%.6fs",
            execution.solver_name,
            execution.solve_elapsed_sec,
        )
        return _serialize_cnc_execution(execution), {}, {}

    @app.callback(
        Output("cnc-selection-state", "data"),
        Input(
            {
                "type": "cnc-probability-cell",
                "camera": ALL,
                "target": ALL,
                "offset": ALL,
            },
            "n_clicks",
        ),
        State("cnc-last-execution", "data"),
        State("cnc-selection-state", "data"),
        prevent_initial_call=True,
    )
    def select_cnc_probability(_clicks, execution_data, selection_state):
        triggered_id = ctx.triggered_id
        if (
            not execution_data
            or not isinstance(triggered_id, dict)
            or triggered_id.get("type") != "cnc-probability-cell"
            or not ctx.triggered
            or not ctx.triggered[0].get("value")
        ):
            return no_update

        camera_id = triggered_id["camera"]
        target_id = triggered_id["target"]
        offset = float(triggered_id["offset"])

        probability = next(
            (
                float(item["observation_probability"])
                for item in execution_data.get("probabilities", [])
                if item["cameraman_id"] == camera_id
                and item["target_id"] == target_id
                and abs(
                    float(item["prediction_offset_sec"]) - offset
                ) <= 1e-9
            ),
            None,
        )
        if probability is None:
            return no_update

        camera_name = next(
            (
                item["name"]
                for item in execution_data.get("cameramen", [])
                if item["id"] == camera_id
            ),
            camera_id,
        )
        target_name = next(
            (
                item["name"]
                for item in execution_data.get("birds", [])
                if item["id"] == target_id
            ),
            target_id,
        )

        candidate = {
            "solution_id": execution_data["solution_id"],
            "cameraman_id": camera_id,
            "cameraman_name": camera_name,
            "target_id": target_id,
            "target_name": target_name,
            "prediction_offset_sec": offset,
            "observation_probability": probability,
        }

        updated = dict(selection_state or {})
        updated[camera_id] = candidate
        logger.info(
            "CnC probability selected: camera=%s target=%s offset=%.3fs p=%.4f",
            camera_name,
            target_name,
            offset,
            probability,
        )
        return updated

    @app.callback(
        Output("cnc-accepted-state", "data"),
        Input(
            {
                "type": "cnc-accept-button",
                "camera": ALL,
            },
            "n_clicks",
        ),
        State("cnc-selection-state", "data"),
        State("cnc-accepted-state", "data"),
        State("cnc-last-execution", "data"),
        prevent_initial_call=True,
    )
    def accept_cnc_probability(
        _clicks,
        selection_state,
        accepted_state,
        execution_data,
    ):
        triggered_id = ctx.triggered_id
        if (
            not execution_data
            or not isinstance(triggered_id, dict)
            or triggered_id.get("type") != "cnc-accept-button"
            or not ctx.triggered
            or not ctx.triggered[0].get("value")
        ):
            return no_update

        camera_id = triggered_id["camera"]
        selected = (selection_state or {}).get(camera_id)
        if (
            not selected
            or selected.get("solution_id") != execution_data["solution_id"]
        ):
            return no_update

        updated = dict(accepted_state or {})
        current = updated.get(camera_id)

        if _candidate_matches(selected, current):
            updated.pop(camera_id, None)
            logger.info(
                "CnC accepted observation cleared: camera=%s",
                selected["cameraman_name"],
            )
        else:
            accepted = dict(selected)
            accepted["accepted_scenario_time_sec"] = float(
                execution_data["scenario_time_sec"]
            )
            updated[camera_id] = accepted
            logger.info(
                "CnC observation accepted: camera=%s target=%s "
                "offset=%.3fs p=%.4f",
                accepted["cameraman_name"],
                accepted["target_name"],
                float(accepted["prediction_offset_sec"]),
                float(accepted["observation_probability"]),
            )

        return updated

    @app.callback(
        Output("active-target-select", "value", allow_duplicate=True),
        Output("track-state", "data", allow_duplicate=True),
        Output("cnc-selection-state", "data", allow_duplicate=True),
        Input(
            {
                "type": "cnc-context-active",
                "camera": ALL,
                "target": ALL,
            },
            "n_clicks",
        ),
        Input(
            {
                "type": "cnc-context-track",
                "camera": ALL,
                "target": ALL,
            },
            "n_clicks",
        ),
        Input(
            {
                "type": "cnc-context-best",
                "camera": ALL,
                "target": ALL,
            },
            "n_clicks",
        ),
        State("cnc-last-execution", "data"),
        State("cnc-display-steps", "value"),
        State("cnc-selection-state", "data"),
        prevent_initial_call=True,
    )
    def handle_bird_context_action(
        _active_clicks,
        _track_clicks,
        _best_clicks,
        execution_data,
        display_steps,
        selection_state,
    ):
        triggered_id = ctx.triggered_id
        if (
            not execution_data
            or not isinstance(triggered_id, dict)
            or not ctx.triggered
            or not ctx.triggered[0].get("value")
        ):
            return no_update, no_update, no_update

        action_type = str(triggered_id.get("type", ""))
        camera_id = triggered_id.get("camera")
        target_id = triggered_id.get("target")
        if not camera_id or not target_id:
            return no_update, no_update, no_update

        if action_type == "cnc-context-active":
            logger.info(
                "CnC bird context: make active target=%s",
                target_id,
            )
            return target_id, no_update, no_update

        if action_type == "cnc-context-track":
            logger.info(
                "CnC bird context: track target=%s",
                target_id,
            )
            return target_id, {"enabled": True}, no_update

        if action_type != "cnc-context-best":
            return no_update, no_update, no_update

        offsets = [
            float(value)
            for value in execution_data.get("prediction_offsets_sec", [])
        ]
        if not offsets:
            return no_update, no_update, no_update

        try:
            visible_count = int(display_steps or 1)
        except (TypeError, ValueError):
            visible_count = 1
        visible_count = max(1, min(visible_count, len(offsets)))
        visible_offsets = offsets[:visible_count]

        candidates = [
            item
            for item in execution_data.get("probabilities", [])
            if item["cameraman_id"] == camera_id
            and item["target_id"] == target_id
            and any(
                abs(
                    float(item["prediction_offset_sec"])
                    - float(offset)
                )
                <= 1e-9
                for offset in visible_offsets
            )
        ]
        if not candidates:
            return no_update, no_update, no_update

        best = max(
            candidates,
            key=lambda item: float(item["observation_probability"]),
        )
        camera_name = next(
            (
                item["name"]
                for item in execution_data.get("cameramen", [])
                if item["id"] == camera_id
            ),
            camera_id,
        )
        target_name = next(
            (
                item["name"]
                for item in execution_data.get("birds", [])
                if item["id"] == target_id
            ),
            target_id,
        )

        candidate = {
            "solution_id": execution_data["solution_id"],
            "cameraman_id": camera_id,
            "cameraman_name": camera_name,
            "target_id": target_id,
            "target_name": target_name,
            "prediction_offset_sec": float(best["prediction_offset_sec"]),
            "observation_probability": float(best["observation_probability"]),
        }
        updated = dict(selection_state or {})
        updated[camera_id] = candidate

        logger.info(
            "CnC bird context: selected best camera=%s target=%s "
            "offset=%.3fs p=%.4f",
            camera_name,
            target_name,
            candidate["prediction_offset_sec"],
            candidate["observation_probability"],
        )
        return no_update, no_update, updated

    @app.callback(
        Output("solver-metry-panel", "children"),
        Input("cnc-last-execution", "data"),
    )
    def render_solver_metry(execution_data):
        return _solver_metry_panel(execution_data)

    @app.callback(
        Output("cnc-result-panel", "children"),
        Input("cnc-last-execution", "data"),
        Input("cnc-display-steps", "value"),
        Input("cnc-selection-state", "data"),
        Input("cnc-accepted-state", "data"),
    )
    def render_cnc_result(
        execution_data,
        display_steps,
        selection_state,
        accepted_state,
    ):
        return _cnc_execution_panel(
            execution_data,
            display_steps,
            selection_state,
            accepted_state,
        )

    return app


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run(
    *,
    debug: bool | None = None,
    host: str = "127.0.0.1",
    port: int = 8050,
) -> None:
    """Run the Dash server.

    In debug mode the Dash/Flask reloader is deliberately disabled so an IDE
    debugger remains attached to a single Python process.
    """
    debug_enabled = (
        _env_flag("TRAJECTORY_DASH_DEBUG", False)
        if debug is None
        else bool(debug)
    )

    app = create_app()
    logger.info(
        "Starting Dash trajectory player host=%s port=%s debug=%s",
        host,
        port,
        debug_enabled,
    )
    app.run(
        debug=debug_enabled,
        host=host,
        port=int(port),
        use_reloader=False,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trajectory Dash player")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash/Flask debug mode without the auto-reloader.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(debug=args.debug, host=args.host, port=args.port)
