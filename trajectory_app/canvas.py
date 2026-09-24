from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from .core import predict_position_ellipse_xy, view_sector_boundary_xy
from .covariance_policy import CovarianceScope, scoped_velocity_covariance
from .logging_config import get_logger
from .models import (
    CameramanState,
    TargetState,
    TerrainConfig,
    TrajectoryGenerationMode,
)

logger = get_logger(__name__)


class TerrainCanvas(QWidget):
    scribbleChanged = Signal()
    sampleSelected = Signal(str, int)
    cameramanPlaced = Signal(str, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(720, 430)
        self.setMouseTracking(True)
        self.terrain: np.ndarray | None = None
        self.terrain_config = TerrainConfig()
        self._terrain_image: QImage | None = None
        self.targets: list[TargetState] = []
        self.cameramen: list[CameramanState] = []
        self.active_target_id: str | None = None
        self.active_cameraman_id: str | None = None
        self.draw_target_id: str | None = None
        self.place_cameraman_id: str | None = None
        self._drawing = False
        self.show_rolling_prediction = True
        self.show_fixed_prediction = True
        self.prediction_step_count = 5
        self.covariance_scope = CovarianceScope.PREDICTION_XY


    def set_prediction_visibility(
        self,
        *,
        show_rolling: bool,
        show_fixed: bool,
    ) -> None:
        self.show_rolling_prediction = bool(show_rolling)
        self.show_fixed_prediction = bool(show_fixed)
        self.update()

    def set_covariance_scope(self, scope: CovarianceScope) -> None:
        self.covariance_scope = CovarianceScope(scope)
        self.update()

    def set_terrain(self, terrain: np.ndarray, config: TerrainConfig) -> None:
        self.terrain = np.asarray(terrain, dtype=float)
        self.terrain_config = config
        self._terrain_image = self._make_heatmap_image(self.terrain)
        self.update()

    def set_targets(self, targets: list[TargetState], active_target_id: str | None) -> None:
        self.targets = targets
        self.active_target_id = active_target_id
        self.update()

    def set_cameramen(
        self,
        cameramen: list[CameramanState],
        active_cameraman_id: str | None,
    ) -> None:
        self.cameramen = cameramen
        self.active_cameraman_id = active_cameraman_id
        self.update()

    def set_draw_target(self, target_id: str | None) -> None:
        if self.draw_target_id == target_id:
            return
        self.draw_target_id = target_id
        self._drawing = False
        self._update_cursor()
        self.update()

    def set_place_cameraman(self, cameraman_id: str | None) -> None:
        if self.place_cameraman_id == cameraman_id:
            return
        self.place_cameraman_id = cameraman_id
        self._drawing = False
        self._update_cursor()
        self.update()

    def _update_cursor(self) -> None:
        if self.draw_target_id is None and self.place_cameraman_id is None:
            self.unsetCursor()
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def _active(self) -> TargetState | None:
        return next((t for t in self.targets if t.id == self.active_target_id), None)

    def _drawing_target(self) -> TargetState | None:
        active = self._active()
        if active is None or active.id != self.draw_target_id:
            return None
        return active

    def _make_heatmap_image(self, terrain: np.ndarray) -> QImage:
        lo = float(np.min(terrain))
        hi = float(np.max(terrain))
        span = max(1e-12, hi - lo)
        t = np.clip((terrain - lo) / span, 0, 1)
        stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        r = np.interp(t, stops, [20, 35, 105, 175, 245])
        g = np.interp(t, stops, [55, 120, 155, 120, 235])
        b = np.interp(t, stops, [100, 90, 75, 60, 225])
        rgb = np.stack((r, g, b), axis=-1).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb[::-1])
        height, width, _ = rgb.shape
        return QImage(
            rgb.data,
            width,
            height,
            3 * width,
            QImage.Format.Format_RGB888,
        ).copy()

    def _widget_to_world(self, pos: QPointF) -> tuple[float, float]:
        cfg = self.terrain_config
        tx = np.clip(pos.x() / max(1.0, self.width()), 0, 1)
        ty = 1 - np.clip(pos.y() / max(1.0, self.height()), 0, 1)
        return (
            float(cfg.x_min + tx * (cfg.x_max - cfg.x_min)),
            float(cfg.y_min + ty * (cfg.y_max - cfg.y_min)),
        )

    def _world_to_widget(self, x: float, y: float) -> QPointF:
        cfg = self.terrain_config
        tx = (x - cfg.x_min) / max(1e-12, cfg.x_max - cfg.x_min)
        ty = (y - cfg.y_min) / max(1e-12, cfg.y_max - cfg.y_min)
        return QPointF(float(tx * self.width()), float((1 - ty) * self.height()))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.place_cameraman_id is not None:
                x, y = self._widget_to_world(event.position())
                self.cameramanPlaced.emit(self.place_cameraman_id, x, y)
                event.accept()
                return

            target = self._drawing_target()
            if target is None:
                event.ignore()
                return
            self._drawing = True
            target.scribble_xy.append(list(self._widget_to_world(event.position())))
            target.trajectory = target.samples = target.covariance = None
            target.current_index = 0
            self.scribbleChanged.emit()
            self.update()
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton:
            self._select_nearest_sample(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        target = self._drawing_target()
        if (
            target is not None
            and self._drawing
            and (event.buttons() & Qt.MouseButton.LeftButton)
        ):
            point = np.asarray(self._widget_to_world(event.position()))
            if (
                not target.scribble_xy
                or np.linalg.norm(point - np.asarray(target.scribble_xy[-1])) > 0.03
            ):
                target.scribble_xy.append(point.tolist())
                target.trajectory = target.samples = target.covariance = None
                self.scribbleChanged.emit()
                self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _select_nearest_sample(self, pos: QPointF) -> None:
        best = None
        for target in self.targets:
            if target.samples is None or len(target.samples) == 0:
                continue
            points = np.array(
                [
                    [
                        self._world_to_widget(float(point[0]), float(point[1])).x(),
                        self._world_to_widget(float(point[0]), float(point[1])).y(),
                    ]
                    for point in target.samples.position
                ]
            )
            distances = np.linalg.norm(
                points - np.array([pos.x(), pos.y()]),
                axis=1,
            )
            index = int(np.argmin(distances))
            candidate = (float(distances[index]), target.id, index)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best and best[0] <= 30:
            self.sampleSelected.emit(best[1], best[2])

    def _draw_cameraman(self, painter: QPainter, cameraman: CameramanState) -> None:
        if cameraman.position_xyz is None:
            return
        x, y, z = cameraman.position_xyz
        color = QColor(cameraman.color)
        origin = self._world_to_widget(x, y)

        boundary = view_sector_boundary_xy(
            x,
            y,
            cameraman.view_radius_m,
            cameraman.view_start_angle_deg,
            cameraman.view_end_angle_deg,
        )
        if len(boundary):
            polygon = [origin]
            for world_x, world_y in boundary:
                polygon.append(self._world_to_widget(float(world_x), float(world_y)))
            polygon.append(origin)
            sector = QPolygonF(polygon)
            fill = QColor(color)
            fill.setAlpha(42 if cameraman.id != self.active_cameraman_id else 62)
            painter.setBrush(fill)
            painter.setPen(QPen(color, 2.5 if cameraman.id == self.active_cameraman_id else 1.8))
            painter.drawPolygon(sector)

        painter.setBrush(QColor(20, 20, 20, 220))
        painter.setPen(QPen(color, 4.0 if cameraman.id == self.active_cameraman_id else 2.5))
        painter.drawRect(QRectF(origin.x() - 7, origin.y() - 7, 14, 14))
        painter.setPen(QPen(QColor("white"), 1))
        painter.drawText(origin + QPointF(10, -10), f"{cameraman.name}  z={z:.2f}")


    def _rolling_covariance_for_target(
        self,
        target: TargetState,
        index: int,
    ) -> np.ndarray | None:
        if (
            target.samples is None
            or target.covariance is None
            or len(target.covariance.time_sec) == 0
        ):
            return None

        time_sec = float(target.samples.time_sec[index])
        covariance_index = int(
            np.searchsorted(
                target.covariance.time_sec,
                time_sec,
                side="right",
            )
            - 1
        )
        covariance_index = int(
            np.clip(
                covariance_index,
                0,
                len(target.covariance.time_sec) - 1,
            )
        )
        return target.covariance.matrices[covariance_index]

    def _prediction_for_covariance(
        self,
        target: TargetState,
        index: int,
        velocity_covariance: np.ndarray | None,
        tau_sec: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if target.samples is None or velocity_covariance is None:
            return None

        active_covariance = scoped_velocity_covariance(
            velocity_covariance,
            self.covariance_scope,
        )
        return predict_position_ellipse_xy(
            target.samples.position[index],
            target.samples.velocity[index],
            active_covariance,
            tau_sec,
            sigma_scale=1.0,
            point_count=64,
        )

    def _draw_covariance_ellipse(
        self,
        painter: QPainter,
        target: TargetState,
        center: np.ndarray,
        ellipse: np.ndarray,
        *,
        outline: QColor,
        fill: QColor,
        pen_style: Qt.PenStyle,
        pen_width: float,
        label: str | None = None,
        label_offset: QPointF | None = None,
    ) -> None:
        spread = np.ptp(ellipse, axis=0)
        if float(np.max(spread)) <= 1e-9:
            return

        first = self._world_to_widget(float(ellipse[0, 0]), float(ellipse[0, 1]))
        path = QPainterPath(first)
        for x, y in ellipse[1:]:
            path.lineTo(self._world_to_widget(float(x), float(y)))
        path.closeSubpath()

        painter.setBrush(fill)
        painter.setPen(QPen(outline, pen_width, pen_style))
        painter.drawPath(path)

        if target.id == self.active_target_id and label:
            center_widget = self._world_to_widget(
                float(center[0]),
                float(center[1]),
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(outline, max(1.0, pen_width - 0.2), pen_style))
            painter.drawText(
                center_widget + (label_offset or QPointF(6, -6)),
                label,
            )

    def _draw_prediction_envelope(
        self,
        painter: QPainter,
        target: TargetState,
        index: int,
        *,
        velocity_covariance: np.ndarray | None,
        outline: QColor,
        fill_base: QColor,
        pen_style: Qt.PenStyle,
        final_label: str | None,
        final_label_offset: QPointF,
    ) -> None:
        if target.samples is None or velocity_covariance is None:
            return

        steps = max(2, int(self.prediction_step_count))
        horizon = float(target.motion_config.prediction_dt_sec)
        if horizon <= 0.0:
            return

        tau_values = np.linspace(0.0, horizon, steps + 1)[1:]
        centers: list[QPointF] = []

        for step_index, tau_sec in enumerate(tau_values, start=1):
            prediction = self._prediction_for_covariance(
                target,
                index,
                velocity_covariance,
                float(tau_sec),
            )
            if prediction is None:
                continue

            center, ellipse = prediction
            center_widget = self._world_to_widget(float(center[0]), float(center[1]))
            centers.append(center_widget)

            alpha_scale = 0.25 + 0.75 * (step_index / len(tau_values))
            fill = QColor(fill_base)
            fill.setAlpha(max(8, int(fill.alpha() * alpha_scale)))
            edge = QColor(outline)
            edge.setAlpha(max(40, int(outline.alpha() * alpha_scale)))
            is_final = step_index == len(tau_values)

            self._draw_covariance_ellipse(
                painter,
                target,
                center,
                ellipse,
                outline=edge,
                fill=fill,
                pen_style=pen_style,
                pen_width=(
                    2.2
                    if is_final and target.id == self.active_target_id
                    else 1.8 if is_final else 1.0
                ),
                label=final_label if is_final else None,
                label_offset=final_label_offset,
            )

        if not centers:
            return

        actual_position = target.samples.position[index]
        start = self._world_to_widget(
            float(actual_position[0]),
            float(actual_position[1]),
        )
        path = QPainterPath(start)
        for point in centers:
            path.lineTo(point)

        line_color = QColor(outline)
        line_color.setAlpha(min(255, max(80, outline.alpha())))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(line_color, 1.4, pen_style))
        painter.drawPath(path)

    def _draw_prediction_ellipses(
        self,
        painter: QPainter,
        target: TargetState,
        index: int,
        color: QColor,
    ) -> None:
        if self.show_rolling_prediction:
            rolling_covariance = self._rolling_covariance_for_target(target, index)
            fill = QColor(color)
            fill.setAlpha(48)
            outline = QColor(color)
            outline.setAlpha(210)
            self._draw_prediction_envelope(
                painter,
                target,
                index,
                velocity_covariance=rolling_covariance,
                outline=outline,
                fill_base=fill,
                pen_style=Qt.PenStyle.SolidLine,
                final_label=(
                    f"+{target.motion_config.prediction_dt_sec:.1f}s "
                    f"rolling 1σ [{self.covariance_scope.ellipse_label}]"
                ),
                final_label_offset=QPointF(6, -8),
            )

        if self.show_fixed_prediction:
            fixed_outline = QColor(245, 245, 245, 220)
            fixed_fill = QColor(245, 245, 245, 24)
            self._draw_prediction_envelope(
                painter,
                target,
                index,
                velocity_covariance=target.fixed_velocity_covariance,
                outline=fixed_outline,
                fill_base=fixed_fill,
                pen_style=Qt.PenStyle.DashLine,
                final_label=(
                    f"+{target.motion_config.prediction_dt_sec:.1f}s "
                    f"fixed 1σ [{self.covariance_scope.ellipse_label}]"
                ),
                final_label_offset=QPointF(6, 14),
            )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(0, 0, float(self.width()), float(self.height()))
        painter.fillRect(rect, QColor(28, 31, 36))
        if self._terrain_image is not None:
            painter.drawImage(rect, self._terrain_image)

        # View cones intentionally use a faint fill; trajectory paths below
        # explicitly switch back to NoBrush so there is no accidental filling.
        for cameraman in self.cameramen:
            self._draw_cameraman(painter, cameraman)

        for target in self.targets:
            color = QColor(target.color)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if len(target.scribble_xy) >= 2:
                path = QPainterPath(self._world_to_widget(*target.scribble_xy[0]))
                for x, y in target.scribble_xy[1:]:
                    path.lineTo(self._world_to_widget(x, y))
                pen = QPen(color, 2.0, Qt.PenStyle.DashLine)
                pen.setColor(QColor(color.red(), color.green(), color.blue(), 150))
                painter.setPen(pen)
                painter.drawPath(path)

            if target.trajectory is not None and len(target.trajectory.xy) >= 2:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                first = target.trajectory.xy[0]
                path = QPainterPath(
                    self._world_to_widget(float(first[0]), float(first[1]))
                )
                for x, y in target.trajectory.xy[1:]:
                    path.lineTo(self._world_to_widget(float(x), float(y)))
                painter.setPen(
                    QPen(color, 4.0 if target.id == self.active_target_id else 2.5)
                )
                painter.drawPath(path)

            if (
                target.id == self.active_target_id
                and target.trajectory_config.generation_mode
                == TrajectoryGenerationMode.DUBINS
                and target.dubins_anchor_poses is not None
            ):
                # Show the fitted poses so it is obvious how the freehand
                # scribble was interpreted by the Dubins generator.
                for anchor_index, anchor in enumerate(target.dubins_anchor_poses):
                    x, y, heading = (float(anchor[0]), float(anchor[1]), float(anchor[2]))
                    center = self._world_to_widget(x, y)
                    heading_tip_world = np.array(
                        [
                            x + np.cos(heading) * max(
                                1.5,
                                target.trajectory_config.dubins_min_turn_radius_m * 0.20,
                            ),
                            y + np.sin(heading) * max(
                                1.5,
                                target.trajectory_config.dubins_min_turn_radius_m * 0.20,
                            ),
                        ]
                    )
                    tip = self._world_to_widget(
                        float(heading_tip_world[0]),
                        float(heading_tip_world[1]),
                    )

                    painter.setBrush(QColor(255, 255, 255, 220))
                    painter.setPen(QPen(color, 2.0))
                    painter.drawEllipse(center, 4.5, 4.5)
                    painter.setPen(QPen(color, 1.8))
                    painter.drawLine(center, tip)
                    painter.setPen(QPen(QColor(20, 20, 20, 210), 1.0))
                    painter.drawText(center + QPointF(6, -6), str(anchor_index))

            if target.samples is None or len(target.samples) == 0:
                continue
            index = int(np.clip(target.current_index, 0, len(target.samples) - 1))
            x, y, _ = target.samples.position[index]
            actual = self._world_to_widget(float(x), float(y))

            # Draw the future uncertainty region beneath the solver/current
            # position markers so the target remains visually prominent.
            self._draw_prediction_ellipses(painter, target, index, color)

            rolling_widget = None
            fixed_widget = None

            if target.estimated_position is not None:
                ex, ey, _ = target.estimated_position
                rolling_widget = self._world_to_widget(float(ex), float(ey))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(
                    QPen(QColor(color.red(), color.green(), color.blue(), 185), 1.5, Qt.PenStyle.DotLine)
                )
                painter.drawLine(actual, rolling_widget)
                radius = 9.0
                diamond = QPolygonF(
                    [
                        QPointF(rolling_widget.x(), rolling_widget.y() - radius),
                        QPointF(rolling_widget.x() + radius, rolling_widget.y()),
                        QPointF(rolling_widget.x(), rolling_widget.y() + radius),
                        QPointF(rolling_widget.x() - radius, rolling_widget.y()),
                    ]
                )
                painter.setPen(
                    QPen(color, 3.0 if target.id == self.active_target_id else 2.0)
                )
                painter.drawPolygon(diamond)
                if target.id == self.active_target_id:
                    painter.setPen(QPen(QColor("white"), 1))
                    painter.drawText(
                        rolling_widget + QPointF(11, 12),
                        "rolling-cov estimate",
                    )

            if target.fixed_cov_estimated_position is not None:
                ex, ey, _ = target.fixed_cov_estimated_position
                fixed_widget = self._world_to_widget(float(ex), float(ey))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                fixed_color = QColor(245, 245, 245, 225)
                painter.setPen(
                    QPen(fixed_color, 1.5, Qt.PenStyle.DashLine)
                )
                painter.drawLine(actual, fixed_widget)
                radius = 8.0
                painter.drawRect(
                    QRectF(
                        fixed_widget.x() - radius,
                        fixed_widget.y() - radius,
                        2.0 * radius,
                        2.0 * radius,
                    )
                )
                if target.id == self.active_target_id:
                    painter.drawText(
                        fixed_widget + QPointF(11, -10),
                        "fixed-cov estimate",
                    )

            if (
                target.id == self.active_target_id
                and rolling_widget is not None
                and fixed_widget is not None
            ):
                compare_color = QColor(255, 215, 0, 210)
                painter.setPen(
                    QPen(compare_color, 1.8, Qt.PenStyle.DotLine)
                )
                painter.drawLine(rolling_widget, fixed_widget)

            painter.setBrush(color)
            painter.setPen(
                QPen(
                    QColor("white")
                    if target.id == self.active_target_id
                    else QColor("black"),
                    2,
                )
            )
            painter.drawEllipse(actual, 8, 8)
            painter.setPen(QPen(QColor("white"), 1))
            painter.drawText(actual + QPointF(10, -10), target.name)
        painter.end()
