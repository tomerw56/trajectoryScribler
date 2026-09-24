from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from trajectory_app.core import predict_position_ellipse_xy, view_sector_boundary_xy
from trajectory_app.covariance_policy import CovarianceScope, scoped_velocity_covariance
from trajectory_app.covariance_stability import (
    CovarianceStabilityCalculator,
    CovarianceStabilityResult,
    CovarianceStabilityState,
)
from trajectory_app.models import TargetState

from .analysis import solver_snapshot
from .runtime import LoadedScenario


STATE_LABELS = {
    CovarianceStabilityState.INSUFFICIENT_DATA: "INSUFFICIENT DATA",
    CovarianceStabilityState.STABLE: "STABLE",
    CovarianceStabilityState.CHANGING: "CHANGING",
    CovarianceStabilityState.UNSTABLE: "UNSTABLE",
}


def _rgba(color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(); pass through unknown colors with white fallback."""
    text = str(color).strip()
    if text.startswith("#") and len(text) == 7:
        try:
            r = int(text[1:3], 16)
            g = int(text[3:5], 16)
            b = int(text[5:7], 16)
            return f"rgba({r},{g},{b},{alpha:.3f})"
        except ValueError:
            pass
    return f"rgba(255,255,255,{alpha:.3f})"


def stability_for_target(
    scenario: LoadedScenario,
    target: TargetState,
    time_sec: float,
    scope: CovarianceScope,
) -> CovarianceStabilityResult | None:
    if target.covariance is None or len(target.covariance.time_sec) == 0:
        return None
    calculator = CovarianceStabilityCalculator()
    return calculator.evaluate(
        target.covariance,
        end_time_sec=float(time_sec),
        window_sec=target.motion_config.covariance_stability_window_sec,
        source_warmup_sec=target.motion_config.covariance_window_sec,
        scope=scope,
    )


def _prediction_envelope_traces(
    target: TargetState,
    index: int,
    covariance: np.ndarray,
    scope: CovarianceScope,
    *,
    name: str,
    line_dash: str,
    line_color: str,
    opacity: float,
    prediction_horizon_sec: float | None = None,
    step_count: int = 5,
) -> list[go.Scatter]:
    if target.samples is None:
        return []
    horizon = (
        float(target.motion_config.prediction_dt_sec)
        if prediction_horizon_sec is None
        else max(0.0, float(prediction_horizon_sec))
    )
    if horizon <= 0:
        return []

    active_covariance = scoped_velocity_covariance(covariance, scope)
    taus = np.linspace(horizon / step_count, horizon, step_count)
    traces: list[go.Scatter] = []
    centers: list[np.ndarray] = [np.asarray(target.samples.position[index, :2], dtype=float)]

    for step_index, tau in enumerate(taus, start=1):
        center, ellipse = predict_position_ellipse_xy(
            target.samples.position[index],
            target.samples.velocity[index],
            active_covariance,
            float(tau),
            sigma_scale=1.0,
            point_count=64,
        )
        centers.append(center)
        closed = np.vstack((ellipse, ellipse[0])) if len(ellipse) else ellipse
        is_final = step_index == step_count
        traces.append(
            go.Scatter(
                x=closed[:, 0] if len(closed) else [],
                y=closed[:, 1] if len(closed) else [],
                mode="lines",
                line={
                    "color": line_color,
                    "width": 3.2 if is_final else 1.6,
                    "dash": line_dash,
                },
                fill="toself" if is_final else None,
                fillcolor=(
                    _rgba(line_color, 0.22 if line_color != "#ffffff" else 0.14)
                    if is_final
                    else None
                ),
                opacity=1.0 if is_final else 0.58,
                name=name,
                legendgroup=name,
                showlegend=is_final,
                hovertemplate=(
                    f"{name}<br>+{tau:.2f}s<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>"
                ),
            )
        )

    centers_array = np.asarray(centers)
    traces.append(
        go.Scatter(
            x=centers_array[:, 0],
            y=centers_array[:, 1],
            mode="lines+markers",
            line={"color": line_color, "width": 2.0, "dash": line_dash},
            marker={
                "size": 6,
                "color": line_color,
                "line": {"width": 1, "color": "#111820"},
            },
            opacity=0.95,
            name=f"{name} centerline",
            legendgroup=name,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    return traces


def build_terrain_figure(
    scenario: LoadedScenario,
    *,
    time_sec: float,
    active_target_id: str | None,
    covariance_scope: CovarianceScope,
    show_rolling_prediction: bool,
    show_fixed_prediction: bool,
    prediction_horizon_sec: float | None = None,
    track_active_target: bool = False,
) -> go.Figure:
    cfg = scenario.terrain_config
    x_values = np.linspace(cfg.x_min, cfg.x_max, scenario.terrain.shape[1])
    y_values = np.linspace(cfg.y_min, cfg.y_max, scenario.terrain.shape[0])

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=scenario.terrain,
            colorscale="Earth",
            hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f} m<extra></extra>",
            name="Terrain",
            showscale=False,
        )
    )

    for cameraman in scenario.cameramen:
        if cameraman.position_xyz is None:
            continue
        x, y, _ = cameraman.position_xyz
        boundary = view_sector_boundary_xy(
            x,
            y,
            cameraman.view_radius_m,
            cameraman.view_start_angle_deg,
            cameraman.view_end_angle_deg,
        )
        if len(boundary):
            sector_x = np.r_[x, boundary[:, 0], x]
            sector_y = np.r_[y, boundary[:, 1], y]
            fig.add_trace(
                go.Scatter(
                    x=sector_x,
                    y=sector_y,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(255,255,255,0.035)",
                    line={"color": cameraman.color, "width": 1},
                    name=cameraman.name,
                    legendgroup="cameramen",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker={"size": 9, "symbol": "triangle-up", "color": cameraman.color, "line": {"width": 1, "color": "white"}},
                name=cameraman.name,
                legendgroup="cameramen",
                showlegend=False,
                hovertemplate=f"{cameraman.name}<extra></extra>",
            )
        )

    active_target = scenario.target_by_id(active_target_id)
    active_index = 0

    for target in scenario.targets:
        if target.trajectory is None or target.samples is None:
            continue
        xy = target.trajectory.xy
        is_active = target.id == active_target_id
        fig.add_trace(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="lines",
                line={"color": target.color, "width": 4 if is_active else 2},
                opacity=1.0 if is_active else 0.62,
                name=target.name,
                hovertemplate=f"{target.name}<extra></extra>",
            )
        )
        index = scenario.target_index_at_time(target, time_sec)
        position = target.samples.position[index]
        fig.add_trace(
            go.Scatter(
                x=[position[0]],
                y=[position[1]],
                mode="markers",
                marker={
                    "size": 13 if is_active else 8,
                    "color": target.color,
                    "line": {"width": 2 if is_active else 1, "color": "white"},
                },
                name=f"{target.name} current",
                showlegend=False,
                hovertemplate=(
                    f"{target.name}<br>t={target.samples.time_sec[index]:.2f}s"
                    f"<br>x={position[0]:.2f}<br>y={position[1]:.2f}<br>z={position[2]:.2f}<extra></extra>"
                ),
            )
        )
        if is_active:
            active_index = index

    x_range = [float(cfg.x_min), float(cfg.x_max)]
    y_range = [float(cfg.y_min), float(cfg.y_max)]
    tracking_position: np.ndarray | None = None

    if (
        track_active_target
        and active_target is not None
        and active_target.samples is not None
        and len(active_target.samples.time_sec) > 0
    ):
        tracking_position = np.asarray(
            active_target.samples.position[active_index],
            dtype=float,
        )

        # Show roughly one third of the full terrain in each dimension while
        # keeping the selected bird exactly at the center.
        x_half_span = max(
            1.0,
            0.16 * float(cfg.x_max - cfg.x_min),
        )
        y_half_span = max(
            1.0,
            0.16 * float(cfg.y_max - cfg.y_min),
        )
        x_range = [
            float(tracking_position[0] - x_half_span),
            float(tracking_position[0] + x_half_span),
        ]
        y_range = [
            float(tracking_position[1] - y_half_span),
            float(tracking_position[1] + y_half_span),
        ]

    reliability_text = "Reliability: —"
    if active_target is not None and active_target.samples is not None:
        rolling_covariance = scenario.covariance_at_time(active_target, time_sec)
        if show_rolling_prediction:
            fig.add_traces(
                _prediction_envelope_traces(
                    active_target,
                    active_index,
                    rolling_covariance,
                    covariance_scope,
                    name="Rolling covariance 1σ",
                    line_dash="solid",
                    line_color=active_target.color,
                    opacity=0.95,
                    prediction_horizon_sec=prediction_horizon_sec,
                )
            )
        if show_fixed_prediction and active_target.fixed_velocity_covariance is not None:
            fig.add_traces(
                _prediction_envelope_traces(
                    active_target,
                    active_index,
                    active_target.fixed_velocity_covariance,
                    covariance_scope,
                    name="Fixed model covariance 1σ",
                    line_dash="dash",
                    line_color="#ffffff",
                    opacity=0.8,
                    prediction_horizon_sec=prediction_horizon_sec,
                )
            )

        solver_data = solver_snapshot(
            scenario,
            active_target,
            time_sec,
            covariance_scope,
            prediction_horizon_sec=prediction_horizon_sec,
        )
        if solver_data.rolling_estimate is not None:
            fig.add_trace(
                go.Scatter(
                    x=[solver_data.rolling_estimate[0]],
                    y=[solver_data.rolling_estimate[1]],
                    mode="markers",
                    marker={
                        "size": 12,
                        "symbol": "diamond",
                        "color": active_target.color,
                        "line": {"width": 1.5, "color": "white"},
                    },
                    name="Rolling solver estimate",
                    hovertemplate="Rolling future estimate<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
                )
            )
        if solver_data.fixed_estimate is not None:
            fig.add_trace(
                go.Scatter(
                    x=[solver_data.fixed_estimate[0]],
                    y=[solver_data.fixed_estimate[1]],
                    mode="markers",
                    marker={
                        "size": 11,
                        "symbol": "square",
                        "color": "#ffffff",
                        "line": {"width": 1.5, "color": active_target.color},
                    },
                    name="Fixed solver estimate",
                    hovertemplate="Fixed future estimate<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
                )
            )

        stability = stability_for_target(
            scenario,
            active_target,
            time_sec,
            covariance_scope,
        )
        if stability is not None:
            reliability_text = (
                f"Reliability: {STATE_LABELS[stability.state]} | "
                f"drift={stability.drift:.3f} rms={stability.rms_change:.3f} "
                f"large={stability.large_change_fraction:.0%} rough={stability.roughness:.3f}"
            )

    scope_label = (
        "Prediction XY"
        if covariance_scope == CovarianceScope.PREDICTION_XY
        else "Full 3D → XY marginal on map"
    )
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#344054", "size": 13},
        margin={"l": 58, "r": 235, "t": 74, "b": 52},
        title={
            "text": f"{scenario.path.stem} | t={time_sec:.2f}s | {scope_label}<br><sup>{reliability_text}</sup>",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.985,
            "yanchor": "top",
            "font": {"color": "#1f2937", "size": 16},
        },
        xaxis={
            "title": {"text": "X (m)", "font": {"color": "#344054"}},
            "range": x_range,
            "showgrid": False,
            "zeroline": False,
            "tickfont": {"color": "#475467"},
            "automargin": True,
        },
        yaxis={
            "title": {"text": "Y (m)", "font": {"color": "#344054"}},
            "range": y_range,
            "showgrid": False,
            "zeroline": False,
            "tickfont": {"color": "#475467"},
            "automargin": True,
        },
        legend={
            "orientation": "v",
            "xanchor": "left",
            "x": 1.01,
            "yanchor": "top",
            "y": 1.0,
            "font": {"color": "#344054", "size": 11},
            "bgcolor": "rgba(255,255,255,0.96)",
            "bordercolor": "#d7dee8",
            "borderwidth": 1,
            "itemsizing": "constant",
            "tracegroupgap": 4,
        },
        hoverlabel={
            "bgcolor": "#ffffff",
            "font_color": "#1f2937",
        },
        hovermode="closest",
        uirevision=(
            f"track:{scenario.path.stem}:{active_target_id}:{time_sec:.6f}"
            if track_active_target and tracking_position is not None
            else scenario.path.stem
        ),
    )
    return fig
