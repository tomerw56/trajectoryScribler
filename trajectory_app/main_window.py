from __future__ import annotations

from .logging_config import get_logger
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .canvas import TerrainCanvas
from .covariance_policy import CovarianceScope, scoped_velocity_covariance
from .covariance_stability import (
    CovarianceStabilityCalculator,
    CovarianceStabilityResult,
    CovarianceStabilityState,
)
from .core import (
    build_dubins_trajectory,
    build_trajectory,
    derive_velocity_model,
    fixed_velocity_covariance_from_model,
    sample_motion,
    sample_velocity_covariance,
    terrain_surface_position,
)
from .models import (
    AltitudeMode,
    MotionConfig,
    CameramanState,
    TargetState,
    TerrainConfig,
    TrajectoryConfig,
    TrajectoryGenerationMode,
    dataclass_dict,
)
from .target_manager import COLORS, TargetManager
from .cameraman_manager import CameramanManager
from .terrain_io import load_terrain
from .terrain_window import TerrainDesigner
from .solver import PrincipalSigmaDemoSolver, Solver, SolverRequest


logger = get_logger(__name__)

class MainWindow(QMainWindow):
    def __init__(self, solver: Solver | None = None):
        super().__init__()
        # The GUI talks only to this object for the estimated position.
        # Inject any object implementing Solver.solve(request) to replace it.
        self.solver: Solver = solver or PrincipalSigmaDemoSolver()
        self.covariance_stability_calculator = CovarianceStabilityCalculator()
        self.setWindowTitle("Trajectory Scribbler — Multi Target")
        self.resize(1500, 980)

        self.terrain_config: TerrainConfig | None = None
        self.terrain: np.ndarray | None = None
        self.terrain_path: Path | None = None

        self.targets: list[TargetState] = [TargetState("Target 1", COLORS[0][1])]
        self.cameramen: list[CameramanState] = [
            CameramanState("Cameraman 1", COLORS[1][1])
        ]
        self.active_cameraman_id: str | None = self.cameramen[0].id
        self.placing_cameraman_id: str | None = None
        self.global_time = 0.0
        self.show_rolling_prediction = True
        self.show_fixed_prediction = True
        self.covariance_scope = CovarianceScope.PREDICTION_XY

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._play_tick)
        self.terrain_designer: TerrainDesigner | None = None
        self.target_manager: TargetManager | None = None
        self.cameraman_manager: CameramanManager | None = None
        self._updating_ui = False
        # A drawing session is transient and is never persisted until a
        # successful Generate commits it.
        self.drawing_target_id: str | None = None

        self._build_ui()
        self._refresh_target_combo()
        self._refresh_all()

    def _build_ui(self) -> None:
        root_widget = QWidget()
        root = QVBoxLayout(root_widget)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        self.setCentralWidget(root_widget)

        top = QHBoxLayout()
        root.addLayout(top)
        for text, callback in (
            ("New Terrain…", self._new_terrain),
            ("Load Terrain…", self._load_terrain_dialog),
            ("Load Trajectory…", self._load_project),
            ("Save Trajectory…", self._save_project),
            ("Target Manager…", self._open_target_manager),
            ("Cameraman Manager…", self._open_cameraman_manager),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            top.addWidget(button)
        top.addStretch(1)
        self.terrain_label = QLabel("Terrain: <none>")
        top.addWidget(self.terrain_label)

        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split, 4)

        self.canvas = TerrainCanvas()
        self.canvas.set_prediction_visibility(
            show_rolling=self.show_rolling_prediction,
            show_fixed=self.show_fixed_prediction,
        )
        self.canvas.set_covariance_scope(self.covariance_scope)
        split.addWidget(self.canvas)
        self.canvas.scribbleChanged.connect(self._scribble_changed)
        self.canvas.sampleSelected.connect(self._canvas_sample_selected)
        self.canvas.cameramanPlaced.connect(self._canvas_cameraman_placed)

        panel = self._build_panel()
        split.addWidget(panel)
        split.setSizes([1180, 300])

        graphs = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(graphs, 3)

        velocity_group = QGroupBox("Active target — velocity")
        velocity_layout = QVBoxLayout(velocity_group)
        self.velocity_plot = pg.PlotWidget()
        velocity_layout.addWidget(self.velocity_plot)
        graphs.addWidget(velocity_group)
        self.velocity_plot.showGrid(x=True, y=True, alpha=0.25)
        self.velocity_plot.setLabel("left", "Velocity", "m/s")
        self.velocity_plot.setLabel("bottom", "Time", "s")
        self.velocity_plot.addLegend()
        self.speed_curve = self.velocity_plot.plot(name="|V|", pen=pg.mkPen("w", width=2))
        self.vx_curve = self.velocity_plot.plot(name="Vx", pen=pg.mkPen("r", width=2))
        self.vy_curve = self.velocity_plot.plot(name="Vy", pen=pg.mkPen("g", width=2))
        self.vz_curve = self.velocity_plot.plot(name="Vz", pen=pg.mkPen("b", width=2))
        self.vcursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("y", width=2))
        self.velocity_plot.addItem(self.vcursor)
        self.velocity_plot.scene().sigMouseClicked.connect(
            lambda event: self._plot_jump(event, self.velocity_plot)
        )

        covariance_group = QGroupBox("Active target — rolling velocity covariance")
        covariance_layout = QVBoxLayout(covariance_group)

        self.cov_stability_title = QLabel("Covariance stability — no active target")
        self.cov_stability_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cov_stability_title.setWordWrap(True)
        self.cov_stability_title.setMinimumHeight(44)
        self.cov_stability_title.setStyleSheet(
            "font-weight: 600; padding: 4px 8px;"
        )
        covariance_layout.addWidget(self.cov_stability_title)

        self.cov_plot = pg.PlotWidget()
        covariance_layout.addWidget(self.cov_plot)
        graphs.addWidget(covariance_group)
        self.cov_plot.showGrid(x=True, y=True, alpha=0.25)
        self.cov_plot.setLabel("left", "Variance", "(m/s)²")
        self.cov_plot.setLabel("bottom", "Time", "s")
        self.cov_plot.addLegend()
        self.cxx_curve = self.cov_plot.plot(name="Cxx", pen=pg.mkPen("r", width=2))
        self.cyy_curve = self.cov_plot.plot(name="Cyy", pen=pg.mkPen("g", width=2))
        self.czz_curve = self.cov_plot.plot(name="Czz", pen=pg.mkPen("b", width=2))
        self.trace_curve = self.cov_plot.plot(name="trace(active)", pen=pg.mkPen("w", width=2))
        self.ccursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("y", width=2))
        self.cov_plot.addItem(self.ccursor)
        self.cov_plot.scene().sigMouseClicked.connect(
            lambda event: self._plot_jump(event, self.cov_plot)
        )

    def _build_panel(self) -> QWidget:
        panel = QFrame()
        panel.setMinimumWidth(270)
        panel.setMaximumWidth(330)
        layout = QVBoxLayout(panel)

        active_group = QGroupBox("Active target")
        active_layout = QVBoxLayout(active_group)
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        active_layout.addWidget(self.target_combo)
        self.active_summary = QLabel("Configure speed and altitude in Target Manager.")
        self.active_summary.setWordWrap(True)
        active_layout.addWidget(self.active_summary)
        layout.addWidget(active_group)

        playback_group = QGroupBox("Playback — global time")
        playback_layout = QGridLayout(playback_group)
        self.first = QPushButton("|◀")
        self.prev = QPushButton("◀")
        self.play = QPushButton("▶ Play")
        self.next = QPushButton("▶")
        self.last = QPushButton("▶|")
        for column, button in enumerate((self.first, self.prev, self.play, self.next, self.last)):
            playback_layout.addWidget(button, 0, column)
        self.first.clicked.connect(lambda: self._set_global_time(0.0))
        self.prev.clicked.connect(lambda: self._step(-1))
        self.next.clicked.connect(lambda: self._step(1))
        self.play.clicked.connect(self._toggle_play)
        self.last.clicked.connect(self._go_last)
        layout.addWidget(playback_group)

        sample_group = QGroupBox("Active target sample")
        sample_form = QFormLayout(sample_group)
        self.step_label = QLabel("—")
        self.time_label = QLabel("—")
        self.pos_label = QLabel("—")
        self.vel_label = QLabel("—")
        self.comp_label = QLabel("—")
        self.cov_label = QLabel("—")
        self.fixed_cov_label = QLabel("—")
        self.cov_difference_label = QLabel("—")
        self.estimate_label = QLabel("—")
        self.fixed_estimate_label = QLabel("—")
        self.estimate_difference_label = QLabel("—")
        self.estimate_error_label = QLabel("—")
        self.mse_label = QLabel("—")
        self.rmse_label = QLabel("—")
        self.eval_count_label = QLabel("—")
        for label, widget in (
            ("Step", self.step_label),
            ("Global time", self.time_label),
            ("Position", self.pos_label),
            ("|V|", self.vel_label),
            ("Components", self.comp_label),
            ("Rolling covariance", self.cov_label),
            ("Fixed model covariance", self.fixed_cov_label),
            ("Covariance diag Δ", self.cov_difference_label),
            ("Solver", QLabel(self.solver.name)),
            ("Rolling-cov estimate", self.estimate_label),
            ("Fixed-cov estimate", self.fixed_estimate_label),
            ("Estimate difference", self.estimate_difference_label),
            ("Rolling estimate error @ +dt", self.estimate_error_label),
            ("Position MSE", self.mse_label),
            ("Position RMSE", self.rmse_label),
            ("Evaluated samples", self.eval_count_label),
        ):
            widget.setWordWrap(True)
            sample_form.addRow(label, widget)
        layout.addWidget(sample_group)


        prediction_group = QGroupBox("Covariance / prediction display")
        prediction_form = QFormLayout(prediction_group)

        self.covariance_scope_combo = QComboBox()
        self.covariance_scope_combo.addItem(
            "Prediction XY (default)",
            CovarianceScope.PREDICTION_XY.value,
        )
        self.covariance_scope_combo.addItem(
            "Full 3D (ellipse shows XY marginal)",
            CovarianceScope.FULL_3D.value,
        )
        self.covariance_scope_combo.currentIndexChanged.connect(
            self._covariance_scope_changed
        )
        prediction_form.addRow("System covariance scope", self.covariance_scope_combo)

        self.show_rolling_checkbox = QCheckBox("Show rolling / actual covariance")
        self.show_rolling_checkbox.setChecked(True)
        self.show_fixed_checkbox = QCheckBox("Show fixed velocity-model covariance")
        self.show_fixed_checkbox.setChecked(True)
        self.show_rolling_checkbox.toggled.connect(
            self._prediction_visibility_changed
        )
        self.show_fixed_checkbox.toggled.connect(
            self._prediction_visibility_changed
        )
        prediction_form.addRow(self.show_rolling_checkbox)
        prediction_form.addRow(self.show_fixed_checkbox)
        layout.addWidget(prediction_group)

        hint = QLabel(
            "Drawing is armed from Target Manager\n"
            "Filled circle = actual position\n"
            "Diamond = rolling-cov future estimate\n"
            "Square = fixed-cov future estimate\n"
            "Solid envelope = rolling covariance forward prediction\n"
            "Dashed envelope = fixed model covariance forward prediction\n"
            "Full 3D mode draws its XY marginal on the 2D terrain view\n"
            "Right-click near a path: select target/time\n"
            "Cameraman sectors: faint colored wedges"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return panel

    def _active(self) -> TargetState | None:
        index = self.target_combo.currentIndex()
        return self.targets[index] if 0 <= index < len(self.targets) else None

    def _target_by_id(self, target_id: str) -> TargetState | None:
        return next((target for target in self.targets if target.id == target_id), None)

    def _set_active_target(self, target_id: str) -> None:
        index = next((i for i, target in enumerate(self.targets) if target.id == target_id), -1)
        if index >= 0:
            self.target_combo.setCurrentIndex(index)

    def _refresh_target_combo(self, preferred_id: str | None = None) -> None:
        current = preferred_id or (self._active().id if self._active() else None)
        self._updating_ui = True
        self.target_combo.clear()
        for target in self.targets:
            self.target_combo.addItem(target.name, target.id)
        index = next((i for i, target in enumerate(self.targets) if target.id == current), 0)
        self.target_combo.setCurrentIndex(index if self.targets else -1)
        self._updating_ui = False
        self._refresh_active_summary()

    def _target_changed(self) -> None:
        if self._updating_ui:
            return
        target = self._active()
        if self.drawing_target_id is not None and (
            target is None or target.id != self.drawing_target_id
        ):
            self._cancel_uncommitted_drawing(reset_target=True)
        if self.target_manager is not None and target is not None:
            self.target_manager.select_target(target.id)
        self._refresh_all()

    def _refresh_active_summary(self) -> None:
        target = self._active()
        if target is None:
            self.active_summary.setText("No active target")
            return
        mode = (
            "Terrain clearance"
            if target.trajectory_config.altitude_mode == AltitudeMode.TERRAIN_CLEARANCE
            else "Cruise altitude"
        )
        derived = target.derived_velocity_model
        if derived is None:
            derived_text = "Derived model: generate trajectory first"
        elif np.isfinite(derived.min_turn_radius_m):
            derived_text = (
                f"Derived Rmin: {derived.min_turn_radius_m:.3f} m | "
                f"turn rate: {derived.max_turn_rate_deg_s:.2f}°/s"
            )
        else:
            derived_text = "Derived Rmin: ∞ (straight)"

        generator = (
            f"Dubins Rmin={target.trajectory_config.dubins_min_turn_radius_m:.2f} m"
            if target.trajectory_config.generation_mode
            == TrajectoryGenerationMode.DUBINS
            else "Freehand"
        )
        self.active_summary.setText(
            f"Speed: {target.motion_config.speed_mps:.3f} m/s\n"
            f"Generator: {generator}\n"
            f"Altitude mode: {mode}\n"
            f"{derived_text}\n"
            "Edit in Target Manager."
        )

    def _open_target_manager(self) -> None:
        if self.target_manager is None:
            self.target_manager = TargetManager(self.targets, self)
            self.target_manager.targetsChanged.connect(self._manager_targets_changed)
            self.target_manager.settingsChanged.connect(self._manager_settings_changed)
            self.target_manager.activeTargetRequested.connect(self._set_active_target)
            self.target_manager.generateTargetRequested.connect(self._manager_generate_target)
            self.target_manager.generateAllRequested.connect(self._generate_all)
            self.target_manager.clearScribbleRequested.connect(self._manager_clear_scribble)
            self.target_manager.beginDrawingRequested.connect(self._manager_begin_drawing)
            self.target_manager.beginDubinsDrawingRequested.connect(
                self._manager_begin_dubins_drawing
            )
            self.target_manager.stopDrawingRequested.connect(self._manager_stop_drawing)
        else:
            self.target_manager.set_targets(self.targets)

        active = self._active()
        if active is not None:
            self.target_manager.select_target(active.id)
        self.target_manager.set_drawing_target(self.drawing_target_id)
        self.target_manager.show()
        self.target_manager.raise_()
        self.target_manager.activateWindow()

    def _cameraman_by_id(self, cameraman_id: str) -> CameramanState | None:
        return next((c for c in self.cameramen if c.id == cameraman_id), None)

    def _open_cameraman_manager(self) -> None:
        if self.cameraman_manager is None:
            self.cameraman_manager = CameramanManager(self.cameramen, self)
            self.cameraman_manager.cameramenChanged.connect(
                self._manager_cameramen_changed
            )
            self.cameraman_manager.settingsChanged.connect(
                self._manager_cameraman_settings_changed
            )
            self.cameraman_manager.activeCameramanRequested.connect(
                self._set_active_cameraman
            )
            self.cameraman_manager.beginPlacementRequested.connect(
                self._manager_begin_cameraman_placement
            )
            self.cameraman_manager.stopPlacementRequested.connect(
                self._manager_stop_cameraman_placement
            )
        else:
            self.cameraman_manager.set_cameramen(self.cameramen)

        if self.active_cameraman_id is not None:
            self.cameraman_manager.select_cameraman(self.active_cameraman_id)
        self.cameraman_manager.set_placing_cameraman(self.placing_cameraman_id)
        self.cameraman_manager.show()
        self.cameraman_manager.raise_()
        self.cameraman_manager.activateWindow()

    def _set_active_cameraman(self, cameraman_id: str) -> None:
        if self._cameraman_by_id(cameraman_id) is None:
            return
        self.active_cameraman_id = cameraman_id
        self._refresh_all()

    def _manager_cameramen_changed(self, preferred_id: str) -> None:
        if (
            self.placing_cameraman_id is not None
            and self._cameraman_by_id(self.placing_cameraman_id) is None
        ):
            self._set_placing_cameraman(None)
        if preferred_id:
            self.active_cameraman_id = preferred_id
        elif self.cameramen:
            self.active_cameraman_id = self.cameramen[0].id
        else:
            self.active_cameraman_id = None
        self._refresh_all()

    def _manager_cameraman_settings_changed(self, cameraman_id: str) -> None:
        self.active_cameraman_id = cameraman_id
        self._refresh_all()

    def _set_placing_cameraman(self, cameraman_id: str | None) -> None:
        self.placing_cameraman_id = cameraman_id
        self.canvas.set_place_cameraman(cameraman_id)
        if self.cameraman_manager is not None:
            self.cameraman_manager.set_placing_cameraman(cameraman_id)

    def _manager_begin_cameraman_placement(self, cameraman_id: str) -> None:
        self._stop()
        if self.drawing_target_id is not None:
            self._cancel_uncommitted_drawing(reset_target=True)
        if self.terrain is None or self.terrain_config is None:
            QMessageBox.information(
                self,
                "Load terrain first",
                "Load or create a .terrain file before placing a cameraman.",
            )
            return
        if self._cameraman_by_id(cameraman_id) is None:
            return
        self.active_cameraman_id = cameraman_id
        self._set_placing_cameraman(cameraman_id)
        self.statusBar().showMessage(
            "Click the terrain to place the cameraman. Its Z will equal terrain altitude.",
            6000,
        )
        self._refresh_all()

    def _manager_stop_cameraman_placement(self, cameraman_id: str) -> None:
        if cameraman_id == self.placing_cameraman_id:
            self._set_placing_cameraman(None)
            self.statusBar().showMessage("Cameraman placement cancelled.", 3000)

    def _canvas_cameraman_placed(
        self,
        cameraman_id: str,
        x_m: float,
        y_m: float,
    ) -> None:
        cameraman = self._cameraman_by_id(cameraman_id)
        if (
            cameraman is None
            or self.terrain is None
            or self.terrain_config is None
        ):
            return
        position = terrain_surface_position(
            self.terrain,
            self.terrain_config,
            x_m,
            y_m,
        )
        cameraman.position_xyz = position.tolist()
        self.active_cameraman_id = cameraman_id
        self._set_placing_cameraman(None)
        logger.info(
            "Placed cameraman=%s at x=%.3f y=%.3f z=%.3f",
            cameraman.name,
            position[0],
            position[1],
            position[2],
        )
        if self.cameraman_manager is not None:
            self.cameraman_manager.refresh_from_cameramen(cameraman_id)
        self._refresh_all()

    def _manager_targets_changed(self, preferred_id: str) -> None:
        if self.drawing_target_id is not None and self._target_by_id(self.drawing_target_id) is None:
            self._set_drawing_target(None)
        self._refresh_target_combo(preferred_id)
        self._refresh_all()
        if self.target_manager is not None:
            self.target_manager.refresh_from_targets(preferred_id)

    def _reset_target_path(self, target: TargetState) -> None:
        target.scribble_xy.clear()
        target.trajectory = None
        target.samples = None
        target.covariance = None
        target.current_index = 0
        target.derived_velocity_model = None
        target.fixed_velocity_covariance = None
        target.fixed_cov_estimated_position = None
        target.fixed_cov_solver_metadata.clear()
        target.estimated_position = None
        target.solver_metadata.clear()
        target.solver_statistics.clear()
        target.dubins_anchor_poses = None
        target.dubins_leg_families = ()
        target.dubins_mean_fit_error_m = None
        target.dubins_max_fit_error_m = None

    def _set_drawing_target(self, target_id: str | None) -> None:
        self.drawing_target_id = target_id
        self.canvas.set_draw_target(target_id)
        if self.target_manager is not None:
            self.target_manager.set_drawing_target(target_id)

    def _cancel_uncommitted_drawing(self, reset_target: bool = True) -> None:
        target_id = self.drawing_target_id
        if target_id is None:
            self.canvas.set_draw_target(None)
            return
        target = self._target_by_id(target_id)
        self._set_drawing_target(None)
        if reset_target and target is not None and target.trajectory is None:
            self._reset_target_path(target)
        self._refresh_all()
        if self.target_manager is not None:
            self.target_manager.refresh_from_targets(target_id)
            self.target_manager.set_drawing_target(None)

    def _begin_target_drawing_session(
        self,
        target_id: str,
        *,
        mode: TrajectoryGenerationMode,
        dubins_min_turn_radius_m: float | None = None,
    ) -> None:
        self._stop()
        if self.placing_cameraman_id is not None:
            self._set_placing_cameraman(None)

        # Any previous uncommitted drawing is discarded first.
        if self.drawing_target_id is not None:
            self._cancel_uncommitted_drawing(reset_target=True)

        target = self._target_by_id(target_id)
        if target is None:
            return

        self._set_active_target(target_id)
        target.trajectory_config.generation_mode = mode
        if dubins_min_turn_radius_m is not None:
            target.trajectory_config.dubins_min_turn_radius_m = float(
                dubins_min_turn_radius_m
            )

        # Redraw is destructive by design: beginning a new drawing deletes the
        # old scribble and generated trajectory immediately.
        self._reset_target_path(target)
        self.global_time = 0.0
        self.show_rolling_prediction = True
        self.show_fixed_prediction = True
        self._set_drawing_target(target_id)
        self._refresh_all()

        if self.target_manager is not None:
            self.target_manager.refresh_from_targets(target_id)
            self.target_manager.set_drawing_target(target_id)

        if mode == TrajectoryGenerationMode.DUBINS:
            mode_text = (
                f"Dubins Rmin={target.trajectory_config.dubins_min_turn_radius_m:.2f} m"
            )
        else:
            mode_text = "freehand"

        self.statusBar().showMessage(
            f"Drawing {target.name} ({mode_text}). "
            "Generate trajectory to keep it; Stop drawing cancels it.",
            7000,
        )

    def _manager_begin_drawing(self, target_id: str) -> None:
        self._begin_target_drawing_session(
            target_id,
            mode=TrajectoryGenerationMode.FREEHAND,
        )

    def _manager_begin_dubins_drawing(
        self,
        target_id: str,
        min_turn_radius_m: float,
    ) -> None:
        self._begin_target_drawing_session(
            target_id,
            mode=TrajectoryGenerationMode.DUBINS,
            dubins_min_turn_radius_m=min_turn_radius_m,
        )

    def _manager_stop_drawing(self, target_id: str) -> None:
        if target_id != self.drawing_target_id:
            return
        # A stopped drawing that was never generated is intentionally discarded.
        self._cancel_uncommitted_drawing(reset_target=True)
        self.statusBar().showMessage(
            "Drawing stopped without Generate — draft discarded and target reset.",
            5000,
        )

    def _manager_settings_changed(self, target_id: str, change_kind: str) -> None:
        target = self._target_by_id(target_id)
        if target is None:
            return
        self._stop()

        if change_kind == "geometry":
            target.trajectory = None
            target.samples = None
            target.covariance = None
            target.current_index = 0
            target.derived_velocity_model = None
            target.fixed_velocity_covariance = None
            target.fixed_cov_estimated_position = None
            target.fixed_cov_solver_metadata.clear()
            target.dubins_anchor_poses = None
            target.dubins_leg_families = ()
            target.dubins_mean_fit_error_m = None
            target.dubins_max_fit_error_m = None
        elif change_kind == "motion":
            if target.trajectory is not None and self.terrain is not None:
                try:
                    self._resample_target(target)
                except Exception as exc:
                    QMessageBox.critical(self, "Motion update failed", str(exc))
        self._refresh_active_summary()
        self._refresh_all()
        if self.target_manager is not None:
            self.target_manager.refresh_from_targets(target_id)

    def _manager_generate_target(self, target_id: str) -> None:
        target = self._target_by_id(target_id)
        if target is None:
            return
        self._set_active_target(target_id)
        try:
            self._generate_target(target)
            if target.id == self.drawing_target_id:
                # Successful generation commits the new scribble and disarms drawing.
                self._set_drawing_target(None)
            self._set_global_time(self.global_time)
            self._refresh_all()
            if self.target_manager is not None:
                self.target_manager.refresh_from_targets(target.id)
        except Exception as exc:
            logger.exception("Trajectory generation failed for target=%s", target.name)
            QMessageBox.critical(self, "Generate trajectory failed", str(exc))

    def _manager_clear_scribble(self, target_id: str) -> None:
        target = self._target_by_id(target_id)
        if target is None:
            return
        if target_id == self.drawing_target_id:
            self._set_drawing_target(None)
        self._reset_target_path(target)
        self._stop()
        self._refresh_all()
        if self.target_manager is not None:
            self.target_manager.refresh_from_targets(target_id)

    def _scribble_changed(self) -> None:
        self._stop()
        self._refresh_all()
        if self.target_manager is not None:
            target = self._active()
            self.target_manager.refresh_from_targets(target.id if target else None)

    def _new_terrain(self) -> None:
        self.terrain_designer = TerrainDesigner()
        self.terrain_designer.terrainSaved.connect(self._load_terrain_path)
        self.terrain_designer.show()
        self.terrain_designer.raise_()

    def _load_terrain_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load terrain", "", "Terrain (*.terrain)")
        if path:
            self._load_terrain_path(path)

    def _load_terrain_path(self, path: str) -> None:
        if self.drawing_target_id is not None:
            self._cancel_uncommitted_drawing(reset_target=True)
        try:
            config, heights = load_terrain(path)
        except Exception as exc:
            QMessageBox.critical(self, "Terrain load failed", str(exc))
            return

        self.terrain_config = config
        self.terrain = heights
        self.terrain_path = Path(path).resolve()
        self.terrain_label.setText(f"Terrain: {self.terrain_path.name}")
        for cameraman in self.cameramen:
            cameraman.position_xyz = None
        self._set_placing_cameraman(None)
        for target in self.targets:
            target.trajectory = None
            target.samples = None
            target.covariance = None
            target.current_index = 0
            target.derived_velocity_model = None
            target.fixed_velocity_covariance = None
            target.fixed_cov_estimated_position = None
            target.fixed_cov_solver_metadata.clear()
            target.dubins_anchor_poses = None
            target.dubins_leg_families = ()
            target.dubins_mean_fit_error_m = None
            target.dubins_max_fit_error_m = None
        self.global_time = 0.0
        self.show_rolling_prediction = True
        self.show_fixed_prediction = True
        if self.cameraman_manager is not None:
            self.cameraman_manager.refresh_from_cameramen(self.active_cameraman_id)
        self._refresh_all()

    def _generate_target(self, target: TargetState) -> None:
        if self.terrain is None or self.terrain_config is None:
            raise ValueError("Load or create a .terrain file first.")
        if len(target.scribble_xy) < 2:
            raise ValueError(f"{target.name}: draw a scribble first.")
        if (
            target.trajectory_config.generation_mode
            == TrajectoryGenerationMode.DUBINS
        ):
            target.trajectory, fit = build_dubins_trajectory(
                target.scribble_xy,
                self.terrain,
                self.terrain_config,
                target.trajectory_config,
            )
            target.dubins_anchor_poses = fit.anchor_poses
            target.dubins_leg_families = fit.leg_families
            target.dubins_mean_fit_error_m = fit.mean_fit_error_m
            target.dubins_max_fit_error_m = fit.max_fit_error_m
            logger.info(
                "Dubins trajectory generated: target=%s Rmin=%.3f "
                "anchors=%d legs=%s fit_mean=%.3f fit_max=%.3f",
                target.name,
                fit.min_turn_radius_m,
                len(fit.anchor_poses),
                ",".join(fit.leg_families),
                fit.mean_fit_error_m,
                fit.max_fit_error_m,
            )
        else:
            target.trajectory = build_trajectory(
                target.scribble_xy,
                self.terrain,
                self.terrain_config,
                target.trajectory_config,
            )
            target.dubins_anchor_poses = None
            target.dubins_leg_families = ()
            target.dubins_mean_fit_error_m = None
            target.dubins_max_fit_error_m = None

        self._resample_target(target)

    def _resample_target(self, target: TargetState) -> None:
        if target.trajectory is None:
            return
        target.samples = sample_motion(
            target.trajectory,
            self.terrain,
            self.terrain_config,
            target.motion_config,
        )
        target.covariance = sample_velocity_covariance(target.samples, target.motion_config)
        target.derived_velocity_model = derive_velocity_model(
            target.trajectory,
            target.motion_config.speed_mps,
            target.trajectory_config.path_resolution_m,
        )
        target.fixed_velocity_covariance = fixed_velocity_covariance_from_model(
            target.derived_velocity_model,
            reference_dt_sec=1.0,
            sigma_factor=3.0,
        )
        target.current_index = min(target.current_index, len(target.samples) - 1)
        target.estimated_position = None
        target.fixed_cov_estimated_position = None
        target.solver_metadata.clear()
        target.fixed_cov_solver_metadata.clear()
        target.solver_statistics.clear()

    def _generate_all(self) -> None:
        errors: list[str] = []
        for target in self.targets:
            if len(target.scribble_xy) < 2:
                continue
            try:
                self._generate_target(target)
            except Exception as exc:
                errors.append(str(exc))
        if self.drawing_target_id is not None:
            drawn = self._target_by_id(self.drawing_target_id)
            if drawn is not None and drawn.trajectory is not None:
                self._set_drawing_target(None)
        self._set_global_time(self.global_time)
        self._refresh_all()
        if self.target_manager is not None:
            active = self._active()
            self.target_manager.refresh_from_targets(active.id if active else None)
        if errors:
            QMessageBox.warning(self, "Some targets failed", "\n".join(errors))

    def _covariance_for_target_time(self, target: TargetState, time_sec: float) -> np.ndarray:
        if target.covariance is None or len(target.covariance.time_sec) == 0:
            return np.zeros((3, 3), dtype=float)
        # Use the most recent covariance sample at or before the current time.
        # This avoids leaking a future covariance sample into the solver.
        index = int(np.searchsorted(target.covariance.time_sec, time_sec, side="right") - 1)
        index = int(np.clip(index, 0, len(target.covariance.time_sec) - 1))
        return target.covariance.matrices[index]

    def _solve_with_covariance(
        self,
        target: TargetState,
        index: int,
        covariance: np.ndarray,
        previous_estimate: np.ndarray | None,
    ) -> tuple[np.ndarray, dict]:
        samples = target.samples
        if samples is None:
            raise ValueError("target has no motion samples")

        active_covariance = scoped_velocity_covariance(
            covariance,
            self.covariance_scope,
        )
        request = SolverRequest(
            target_id=target.id,
            target_name=target.name,
            sample_index=index,
            time_sec=float(samples.time_sec[index]),
            current_position=samples.position[index].copy(),
            current_velocity=samples.velocity[index].copy(),
            velocity_covariance=active_covariance.copy(),
            terrain_height_m=float(samples.terrain_z[index]),
            motion_config=target.motion_config,
            trajectory_config=target.trajectory_config,
            covariance_scope=self.covariance_scope,
            previous_estimated_position=(
                None
                if previous_estimate is None
                else previous_estimate.copy()
            ),
        )
        result = self.solver.solve(request)
        estimate = np.asarray(result.estimated_position, dtype=float).reshape(3)
        if not np.all(np.isfinite(estimate)):
            raise ValueError("solver returned a non-finite estimated position")
        return estimate, dict(result.metadata)

    def _update_solver_estimate(self, target: TargetState) -> None:
        if target.samples is None or len(target.samples) == 0:
            target.estimated_position = None
            target.fixed_cov_estimated_position = None
            target.solver_metadata.clear()
            target.fixed_cov_solver_metadata.clear()
            return

        index = int(np.clip(target.current_index, 0, len(target.samples) - 1))
        samples = target.samples
        time_sec = float(samples.time_sec[index])

        rolling_covariance = self._covariance_for_target_time(target, time_sec)
        rolling_previous = (
            None
            if target.estimated_position is None
            else target.estimated_position.copy()
        )
        try:
            estimate, metadata = self._solve_with_covariance(
                target,
                index,
                rolling_covariance,
                rolling_previous,
            )
            target.estimated_position = estimate
            target.solver_metadata = metadata
            future_truth = self._future_truth_for_target(target, index)
            if future_truth is not None:
                target.solver_statistics.put(
                    sample_index=index,
                    time_sec=time_sec,
                    actual_position=future_truth,
                    estimated_position=estimate,
                )
        except Exception as exc:
            target.estimated_position = None
            logger.exception(
                "Rolling-covariance solver failed for target=%s sample=%s",
                target.name,
                index,
            )
            target.solver_metadata = {"error": str(exc)}

        if target.fixed_velocity_covariance is None:
            target.fixed_cov_estimated_position = None
            target.fixed_cov_solver_metadata.clear()
            return

        fixed_previous = (
            None
            if target.fixed_cov_estimated_position is None
            else target.fixed_cov_estimated_position.copy()
        )
        try:
            fixed_estimate, fixed_metadata = self._solve_with_covariance(
                target,
                index,
                target.fixed_velocity_covariance,
                fixed_previous,
            )
            target.fixed_cov_estimated_position = fixed_estimate
            target.fixed_cov_solver_metadata = fixed_metadata
        except Exception as exc:
            target.fixed_cov_estimated_position = None
            logger.exception(
                "Fixed-covariance solver failed for target=%s sample=%s",
                target.name,
                index,
            )
            target.fixed_cov_solver_metadata = {"error": str(exc)}

    def _update_all_solver_estimates(self) -> None:
        for target in self.targets:
            self._update_solver_estimate(target)

    def _set_global_time(self, time_sec: float) -> None:
        # Any playback/navigation action leaves edit/placement mode.
        if self.placing_cameraman_id is not None:
            self._set_placing_cameraman(None)
        # An ungenerated redraw is a draft, so leaving edit mode discards it.
        if self.drawing_target_id is not None:
            self._cancel_uncommitted_drawing(reset_target=True)
        self.global_time = max(0.0, float(time_sec))
        for target in self.targets:
            if target.samples is not None and len(target.samples):
                target.current_index = int(
                    np.argmin(np.abs(target.samples.time_sec - self.global_time))
                )
        self._update_all_solver_estimates()
        self._refresh_all()

    def _step(self, direction: int) -> None:
        target = self._active()
        step = target.motion_config.sample_dt_sec if target else 0.25
        self._set_global_time(self.global_time + direction * step)

    def _max_time(self) -> float:
        values = [
            float(target.samples.time_sec[-1])
            for target in self.targets
            if target.samples is not None and len(target.samples)
        ]
        return max(values, default=0.0)

    def _go_last(self) -> None:
        self._set_global_time(self._max_time())

    def _toggle_play(self) -> None:
        if self.timer.isActive():
            self._stop()
            return
        if self.placing_cameraman_id is not None:
            self._set_placing_cameraman(None)
        if self.drawing_target_id is not None:
            self._cancel_uncommitted_drawing(reset_target=True)
        if self._max_time() <= 0:
            return
        if self.global_time >= self._max_time():
            self._set_global_time(0.0)
        target = self._active()
        interval_ms = max(
            20,
            int(1000 * (target.motion_config.sample_dt_sec if target else 0.25)),
        )
        self.timer.start(interval_ms)
        logger.info("Playback started at global_time=%.3f s interval_ms=%d", self.global_time, interval_ms)
        self.play.setText("⏸ Pause")

    def _play_tick(self) -> None:
        target = self._active()
        dt = target.motion_config.sample_dt_sec if target else 0.25
        if self.global_time + dt > self._max_time() + 1e-9:
            self._stop()
            return
        self._set_global_time(self.global_time + dt)

    def _stop(self) -> None:
        if self.timer.isActive():
            logger.info("Playback stopped at global_time=%.3f s", self.global_time)
        self.timer.stop()
        self.play.setText("▶ Play")

    def _canvas_sample_selected(self, target_id: str, index: int) -> None:
        if self.drawing_target_id is not None:
            self._cancel_uncommitted_drawing(reset_target=True)
        target = self._target_by_id(target_id)
        if target is None:
            return
        self._set_active_target(target_id)
        if target.samples is not None:
            self._set_global_time(float(target.samples.time_sec[index]))

    def _plot_jump(self, event, plot) -> None:
        if self.drawing_target_id is not None:
            self._cancel_uncommitted_drawing(reset_target=True)
        target = self._active()
        if target is None:
            return
        scene_position = event.scenePos()
        if not plot.sceneBoundingRect().contains(scene_position):
            return
        time_sec = float(plot.plotItem.vb.mapSceneToView(scene_position).x())
        self._set_global_time(time_sec)

    def _refresh_all(self) -> None:
        # Keep estimates available after generation/settings changes that call
        # refresh directly rather than going through _set_global_time.
        for target in self.targets:
            if target.samples is not None and target.estimated_position is None:
                self._update_solver_estimate(target)
        active = self._active()
        self.canvas.set_targets(self.targets, active.id if active else None)
        self.canvas.set_cameramen(self.cameramen, self.active_cameraman_id)
        self.canvas.set_draw_target(self.drawing_target_id)
        self.canvas.set_place_cameraman(self.placing_cameraman_id)
        if self.terrain is not None and self.terrain_config is not None:
            self.canvas.set_terrain(self.terrain, self.terrain_config)
        self._refresh_active_summary()
        self._refresh_plots()
        self._refresh_status()

    def get_covariance_stability(
        self,
        target: TargetState,
        *,
        end_time_sec: float | None = None,
        window_sec: float | None = None,
        scope: CovarianceScope | None = None,
    ) -> CovarianceStabilityResult:
        if target.covariance is None:
            empty_times = np.empty(0, dtype=float)
            from .models import CovarianceSamples
            samples = CovarianceSamples(
                time_sec=empty_times,
                matrices=np.empty((0, 3, 3), dtype=float),
            )
        else:
            samples = target.covariance

        requested_window = (
            target.motion_config.covariance_stability_window_sec
            if window_sec is None
            else float(window_sec)
        )
        calculator = self.covariance_stability_calculator.with_window(
            requested_window
        )
        return calculator.evaluate(
            samples,
            end_time_sec=end_time_sec,
            source_warmup_sec=target.motion_config.covariance_window_sec,
            scope=self.covariance_scope if scope is None else CovarianceScope(scope),
        )

    def get_covariance_stability_history(
        self,
        target: TargetState,
        *,
        last_n_seconds: float,
        end_time_sec: float | None = None,
        scope: CovarianceScope | None = None,
    ):
        if target.covariance is None:
            from .models import CovarianceSamples
            samples = CovarianceSamples(
                time_sec=np.empty(0, dtype=float),
                matrices=np.empty((0, 3, 3), dtype=float),
            )
        else:
            samples = target.covariance
        calculator = self.covariance_stability_calculator.with_window(
            target.motion_config.covariance_stability_window_sec
        )
        return calculator.evaluate_history(
            samples,
            last_n_seconds=last_n_seconds,
            end_time_sec=end_time_sec,
            source_warmup_sec=target.motion_config.covariance_window_sec,
            scope=self.covariance_scope if scope is None else CovarianceScope(scope),
        )

    @staticmethod
    def _stability_caption_text(result: CovarianceStabilityResult) -> str:
        labels = {
            CovarianceStabilityState.INSUFFICIENT_DATA:
                f"INSUFFICIENT DATA — {result.reason}",
            CovarianceStabilityState.STABLE:
                "STABLE — covariance changes are small",
            CovarianceStabilityState.CHANGING:
                "CHANGING — covariance is moving; use with caution",
            CovarianceStabilityState.UNSTABLE:
                "UNSTABLE — covariance changed drastically; do not trust",
        }
        return (
            f"{labels[result.state]}\n"
            f"last {result.requested_window_sec:.1f}s | "
            f"drift={result.drift:.3f}  "
            f"rms={result.rms_change:.3f}  "
            f"max={result.max_single_jump:.3f}  "
            f"large={result.large_change_fraction:.0%}  "
            f"rough={result.roughness:.3f}"
        )

    def _update_stability_caption(
        self,
        target: TargetState,
        covariance,
    ) -> None:
        """Update the fixed stability title above the covariance plot."""
        if len(covariance.time_sec) == 0:
            self.cov_stability_title.setText(
                f"{target.name} — INSUFFICIENT DATA\n"
                "No covariance samples available yet"
            )
            self.cov_stability_title.setStyleSheet(
                "font-weight: 600; padding: 4px 8px; color: white;"
            )
            return

        current_time = float(target.samples.time_sec[target.current_index])
        result = self.get_covariance_stability(
            target,
            end_time_sec=current_time,
        )

        labels = {
            CovarianceStabilityState.INSUFFICIENT_DATA:
                "INSUFFICIENT DATA — waiting for enough covariance history",
            CovarianceStabilityState.STABLE:
                "STABLE — covariance changes are small",
            CovarianceStabilityState.CHANGING:
                "CHANGING — covariance is moving; use with caution",
            CovarianceStabilityState.UNSTABLE:
                "UNSTABLE — covariance changed drastically; do not trust",
        }
        colors = {
            CovarianceStabilityState.INSUFFICIENT_DATA: "white",
            CovarianceStabilityState.STABLE: "#69d984",
            CovarianceStabilityState.CHANGING: "#e6c95c",
            CovarianceStabilityState.UNSTABLE: "#ff6b6b",
        }

        self.cov_stability_title.setText(
            f"{target.name}  |  t={current_time:.2f}s  |  "
            f"{self.covariance_scope.display_name} stability  |  "
            f"{labels[result.state]}\n"
            f"last {result.requested_window_sec:.1f}s  |  "
            f"drift={result.drift:.3f}  "
            f"rms={result.rms_change:.3f}  "
            f"max={result.max_single_jump:.3f}"
        )
        self.cov_stability_title.setStyleSheet(
            "font-weight: 600; padding: 4px 8px; "
            f"color: {colors[result.state]};"
        )

    def _refresh_plots(self) -> None:
        target = self._active()
        empty = np.array([])
        if target is None or target.samples is None:
            self.cov_stability_title.setText(
                f"Covariance stability — {self.covariance_scope.display_name} — "
                "no active target data"
            )
            self.cov_stability_title.setStyleSheet(
                "font-weight: 600; padding: 4px 8px; color: white;"
            )
            for curve in (
                self.speed_curve,
                self.vx_curve,
                self.vy_curve,
                self.vz_curve,
                self.cxx_curve,
                self.cyy_curve,
                self.czz_curve,
                self.trace_curve,
            ):
                curve.setData(empty, empty)
            return

        samples = target.samples
        self.speed_curve.setData(samples.time_sec, samples.speed)
        self.vx_curve.setData(samples.time_sec, samples.velocity[:, 0])
        self.vy_curve.setData(samples.time_sec, samples.velocity[:, 1])
        self.vz_curve.setData(samples.time_sec, samples.velocity[:, 2])
        self.vcursor.setValue(self.global_time)

        if target.covariance is not None:
            covariance = target.covariance
            times = covariance.time_sec

            self.cxx_curve.setData(times, covariance.cxx)
            self.cyy_curve.setData(times, covariance.cyy)

            if self.covariance_scope == CovarianceScope.PREDICTION_XY:
                # The system is using only [Vx, Vy]. Hide Czz and use XY trace.
                self.czz_curve.setVisible(False)
                self.czz_curve.setData(empty, empty)
                active_trace = np.asarray(covariance.cxx) + np.asarray(covariance.cyy)
            else:
                self.czz_curve.setVisible(True)
                self.czz_curve.setData(times, covariance.czz)
                active_trace = covariance.trace

            self.trace_curve.setData(times, active_trace)
            self.ccursor.setValue(self.global_time)
            self._update_stability_caption(target, covariance)
        else:
            self.cov_stability_title.setText(
                f"{target.name} — {self.covariance_scope.display_name} — "
                "covariance not available"
            )
            self.cov_stability_title.setStyleSheet(
                "font-weight: 600; padding: 4px 8px; color: white;"
            )
            for curve in (
                self.cxx_curve,
                self.cyy_curve,
                self.czz_curve,
                self.trace_curve,
            ):
                curve.setData(empty, empty)


    def _set_covariance_scope_ui(self, scope: CovarianceScope) -> None:
        """Set the system covariance mode and keep the combo/canvas in sync."""
        selected = CovarianceScope(scope)
        self.covariance_scope = selected

        combo_index = self.covariance_scope_combo.findData(selected.value)
        if combo_index >= 0 and self.covariance_scope_combo.currentIndex() != combo_index:
            blocked = self.covariance_scope_combo.blockSignals(True)
            self.covariance_scope_combo.setCurrentIndex(combo_index)
            self.covariance_scope_combo.blockSignals(blocked)

        self.canvas.set_covariance_scope(selected)

    def _covariance_scope_changed(self) -> None:
        data = self.covariance_scope_combo.currentData()
        selected = CovarianceScope(
            data if data is not None else CovarianceScope.PREDICTION_XY.value
        )
        if selected == self.covariance_scope:
            return

        self._set_covariance_scope_ui(selected)

        # Estimates and statistics depend on the covariance interpretation.
        # Never mix samples evaluated under two different system scopes.
        for target in self.targets:
            target.estimated_position = None
            target.fixed_cov_estimated_position = None
            target.solver_metadata.clear()
            target.fixed_cov_solver_metadata.clear()
            target.solver_statistics.clear()

        self._update_all_solver_estimates()
        self._refresh_all()

    def _prediction_visibility_changed(self) -> None:
        self.show_rolling_prediction = self.show_rolling_checkbox.isChecked()
        self.show_fixed_prediction = self.show_fixed_checkbox.isChecked()
        self.canvas.set_prediction_visibility(
            show_rolling=self.show_rolling_prediction,
            show_fixed=self.show_fixed_prediction,
        )

    def _future_truth_for_target(
        self,
        target: TargetState,
        index: int,
    ) -> np.ndarray | None:
        if target.samples is None or len(target.samples) == 0:
            return None

        start_time = float(target.samples.time_sec[index])
        target_time = start_time + float(target.motion_config.prediction_dt_sec)
        all_times = np.asarray(target.samples.time_sec, dtype=float)
        if target_time < all_times[0] or target_time > all_times[-1]:
            return None

        positions = np.asarray(target.samples.position, dtype=float)
        return np.array(
            [
                np.interp(target_time, all_times, positions[:, axis])
                for axis in range(3)
            ],
            dtype=float,
        )

    def _refresh_status(self) -> None:
        target = self._active()
        enabled = bool(target and target.samples is not None and len(target.samples))
        has_playback = self._max_time() > 0
        for button in (self.first, self.prev, self.play, self.next, self.last):
            button.setEnabled(has_playback)

        if not enabled:
            for widget in (
                self.step_label,
                self.time_label,
                self.pos_label,
                self.vel_label,
                self.comp_label,
                self.cov_label,
                self.fixed_cov_label,
                self.cov_difference_label,
                self.estimate_label,
                self.fixed_estimate_label,
                self.estimate_difference_label,
                self.estimate_error_label,
                self.mse_label,
                self.rmse_label,
                self.eval_count_label,
            ):
                widget.setText("—")
            return

        index = target.current_index
        samples = target.samples
        position = samples.position[index]
        velocity = samples.velocity[index]
        self.step_label.setText(f"{index + 1}/{len(samples)}")
        self.time_label.setText(f"{self.global_time:.2f} s")
        self.pos_label.setText(
            f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) m"
        )
        self.vel_label.setText(f"{np.linalg.norm(velocity):.3f} m/s")
        self.comp_label.setText(
            f"Vx={velocity[0]:.3f}, Vy={velocity[1]:.3f}, Vz={velocity[2]:.3f}"
        )

        rolling_matrix = None
        if target.covariance is not None and len(target.covariance.time_sec):
            rolling_raw = self._covariance_for_target_time(
                target,
                float(samples.time_sec[index]),
            )
            rolling_matrix = scoped_velocity_covariance(
                rolling_raw,
                self.covariance_scope,
            )
            if self.covariance_scope == CovarianceScope.PREDICTION_XY:
                self.cov_label.setText(
                    f"XY diag=({rolling_matrix[0,0]:.3f}, "
                    f"{rolling_matrix[1,1]:.3f})"
                )
            else:
                self.cov_label.setText(
                    f"3D diag=({rolling_matrix[0,0]:.3f}, "
                    f"{rolling_matrix[1,1]:.3f}, {rolling_matrix[2,2]:.3f})"
                )
        else:
            self.cov_label.setText("—")

        fixed_matrix = None
        if target.fixed_velocity_covariance is not None:
            fixed_matrix = scoped_velocity_covariance(
                target.fixed_velocity_covariance,
                self.covariance_scope,
            )
            if self.covariance_scope == CovarianceScope.PREDICTION_XY:
                self.fixed_cov_label.setText(
                    f"XY diag=({fixed_matrix[0,0]:.3f}, "
                    f"{fixed_matrix[1,1]:.3f})"
                )
            else:
                self.fixed_cov_label.setText(
                    f"3D diag=({fixed_matrix[0,0]:.3f}, "
                    f"{fixed_matrix[1,1]:.3f}, {fixed_matrix[2,2]:.3f})"
                )
        else:
            self.fixed_cov_label.setText("—")

        if rolling_matrix is not None and fixed_matrix is not None:
            diagonal_delta = np.diag(fixed_matrix) - np.diag(rolling_matrix)
            if self.covariance_scope == CovarianceScope.PREDICTION_XY:
                self.cov_difference_label.setText(
                    f"XY fixed−rolling=({diagonal_delta[0]:+.3f}, "
                    f"{diagonal_delta[1]:+.3f})"
                )
            else:
                self.cov_difference_label.setText(
                    f"3D fixed−rolling=({diagonal_delta[0]:+.3f}, "
                    f"{diagonal_delta[1]:+.3f}, {diagonal_delta[2]:+.3f})"
                )
        else:
            self.cov_difference_label.setText("—")

        if target.estimated_position is not None:
            estimate = target.estimated_position
            future_truth = self._future_truth_for_target(target, index)
            self.estimate_label.setText(
                f"({estimate[0]:.2f}, {estimate[1]:.2f}, {estimate[2]:.2f}) m"
            )
            if future_truth is not None:
                error = float(np.linalg.norm(estimate - future_truth))
                self.estimate_error_label.setText(f"{error:.3f} m")
            else:
                self.estimate_error_label.setText("n/a (past end)")
            self.mse_label.setText(
                f"{target.solver_statistics.mse_m2:.3f} m²"
                if target.solver_statistics.count
                else "—"
            )
            self.rmse_label.setText(
                f"{target.solver_statistics.rmse_m:.3f} m"
                if target.solver_statistics.count
                else "—"
            )
            self.eval_count_label.setText(
                str(target.solver_statistics.count)
                if target.solver_statistics.count
                else "—"
            )
        else:
            message = target.solver_metadata.get("error", "—")
            self.estimate_label.setText(str(message))
            self.estimate_error_label.setText("—")
            self.mse_label.setText(
                f"{target.solver_statistics.mse_m2:.3f} m²"
                if target.solver_statistics.count else "—"
            )
            self.rmse_label.setText(
                f"{target.solver_statistics.rmse_m:.3f} m"
                if target.solver_statistics.count else "—"
            )
            self.eval_count_label.setText(
                str(target.solver_statistics.count)
                if target.solver_statistics.count else "—"
            )

        if target.fixed_cov_estimated_position is not None:
            fixed_estimate = target.fixed_cov_estimated_position
            self.fixed_estimate_label.setText(
                f"({fixed_estimate[0]:.2f}, {fixed_estimate[1]:.2f}, "
                f"{fixed_estimate[2]:.2f}) m"
            )
        else:
            message = target.fixed_cov_solver_metadata.get("error", "—")
            self.fixed_estimate_label.setText(str(message))

        if (
            target.estimated_position is not None
            and target.fixed_cov_estimated_position is not None
        ):
            difference = float(
                np.linalg.norm(
                    target.fixed_cov_estimated_position
                    - target.estimated_position
                )
            )
            self.estimate_difference_label.setText(f"{difference:.3f} m")
        else:
            self.estimate_difference_label.setText("—")

    def _serialize_target(self, target: TargetState) -> dict:
        return {
            "id": target.id,
            "name": target.name,
            "color": target.color,
            "scribble_xy": target.scribble_xy,
            "trajectory_config": dataclass_dict(target.trajectory_config),
            "motion_config": dataclass_dict(target.motion_config),
        }

    def _serialize_cameraman(self, cameraman: CameramanState) -> dict:
        return {
            "id": cameraman.id,
            "name": cameraman.name,
            "color": cameraman.color,
            "position_xyz": cameraman.position_xyz,
            "view_radius_m": cameraman.view_radius_m,
            "view_start_angle_deg": cameraman.view_start_angle_deg,
            "view_end_angle_deg": cameraman.view_end_angle_deg,
        }

    def _save_project(self) -> None:
        if self.placing_cameraman_id is not None:
            self._set_placing_cameraman(None)
        if self.drawing_target_id is not None:
            self._cancel_uncommitted_drawing(reset_target=True)
        if self.terrain_path is None or not self.terrain_path.exists():
            QMessageBox.critical(
                self,
                "Cannot save trajectory",
                "A trajectory must reference an existing .terrain file. "
                "Create/save or load a terrain first.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save trajectory project",
            "project.trajectory",
            "Trajectory project (*.trajectory)",
        )
        if not path:
            return

        project_path = Path(path)
        if project_path.suffix.lower() != ".trajectory":
            project_path = project_path.with_suffix(".trajectory")
        terrain_ref = os.path.relpath(self.terrain_path, project_path.parent)
        payload = {
            "version": 7,
            "terrain_file": terrain_ref,
            "covariance_scope": self.covariance_scope.value,
            "targets": [self._serialize_target(target) for target in self.targets],
            "cameramen": [
                self._serialize_cameraman(cameraman)
                for cameraman in self.cameramen
            ],
        }
        try:
            project_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.exception("Project save failed")
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved {project_path}", 5000)

    def _load_project(self) -> None:
        if self.placing_cameraman_id is not None:
            self._set_placing_cameraman(None)
        if self.drawing_target_id is not None:
            self._cancel_uncommitted_drawing(reset_target=True)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load trajectory project",
            "",
            "Trajectory project (*.trajectory)",
        )
        if not path:
            return

        project_path = Path(path)
        try:
            data = json.loads(project_path.read_text(encoding="utf-8"))
            version = int(data.get("version", 0))
            if version not in (2, 3, 4, 5, 6, 7):
                raise ValueError(f"Unsupported trajectory project version: {version}")

            terrain_ref = data["terrain_file"]
            terrain_path = (project_path.parent / terrain_ref).resolve()
            if not terrain_path.exists():
                raise FileNotFoundError(
                    "Referenced terrain file was not found:\n"
                    f"{terrain_path}\n\n"
                    "Trajectory loading was cancelled."
                )

            terrain_config, heights = load_terrain(terrain_path)
            loaded_covariance_scope = CovarianceScope(
                data.get(
                    "covariance_scope",
                    CovarianceScope.PREDICTION_XY.value,
                )
            )
            loaded: list[TargetState] = []
            for item in data["targets"]:
                trajectory_data = item["trajectory_config"]
                motion_data = item["motion_config"]
                trajectory_config = TrajectoryConfig(
                    path_resolution_m=trajectory_data["path_resolution_m"],
                    altitude_mode=AltitudeMode(trajectory_data["altitude_mode"]),
                    clearance_m=trajectory_data["clearance_m"],
                    cruise_altitude_m=trajectory_data["cruise_altitude_m"],
                    max_climb_angle_deg=trajectory_data["max_climb_angle_deg"],
                    smoothing_passes=trajectory_data.get("smoothing_passes", 2),
                    generation_mode=TrajectoryGenerationMode(
                        trajectory_data.get(
                            "generation_mode",
                            TrajectoryGenerationMode.FREEHAND.value,
                        )
                    ),
                    dubins_min_turn_radius_m=float(
                        trajectory_data.get("dubins_min_turn_radius_m", 10.0)
                    ),
                )
                motion_config = MotionConfig(
                    speed_mps=motion_data["speed_mps"],
                    sample_dt_sec=motion_data["sample_dt_sec"],
                    covariance_sample_rate_sec=motion_data.get(
                        "covariance_sample_rate_sec", 0.5
                    ),
                    covariance_window_sec=motion_data.get("covariance_window_sec", 5.0),
                    prediction_dt_sec=motion_data.get("prediction_dt_sec", 2.0),
                    covariance_stability_window_sec=motion_data.get(
                        "covariance_stability_window_sec", 6.0
                    ),
                )
                loaded.append(
                    TargetState(
                        name=item["name"],
                        color=item["color"],
                        id=item.get("id", TargetState("x", "#ffffff").id),
                        trajectory_config=trajectory_config,
                        motion_config=motion_config,
                        scribble_xy=item.get("scribble_xy", []),
                    )
                )
            if not loaded:
                loaded = [TargetState("Target 1", COLORS[0][1])]

            loaded_cameramen: list[CameramanState] = []
            for item in data.get("cameramen", []):
                position = item.get("position_xyz")
                loaded_cameramen.append(
                    CameramanState(
                        name=item.get("name", "Cameraman"),
                        color=item.get("color", COLORS[1][1]),
                        id=item.get("id", CameramanState("x", "#ffffff").id),
                        position_xyz=(
                            [float(value) for value in position]
                            if position is not None
                            else None
                        ),
                        view_radius_m=float(item.get("view_radius_m", 8.0)),
                        view_start_angle_deg=float(
                            item.get("view_start_angle_deg", -30.0)
                        ),
                        view_end_angle_deg=float(
                            item.get("view_end_angle_deg", 30.0)
                        ),
                    )
                )
            if not loaded_cameramen:
                loaded_cameramen = [CameramanState("Cameraman 1", COLORS[1][1])]
        except Exception as exc:
            QMessageBox.critical(self, "Trajectory load failed", str(exc))
            return

        self._stop()
        self.terrain_config = terrain_config
        self.terrain = heights
        self.terrain_path = terrain_path
        self.targets = loaded[:10]
        self.cameramen = loaded_cameramen[:10]
        self.active_cameraman_id = self.cameramen[0].id if self.cameramen else None
        self._set_placing_cameraman(None)
        self.global_time = 0.0
        self.show_rolling_prediction = True
        self.show_fixed_prediction = True
        self._set_covariance_scope_ui(loaded_covariance_scope)
        self.terrain_label.setText(f"Terrain: {terrain_path.name}")

        if self.target_manager is not None:
            self.target_manager.set_targets(self.targets)
        if self.cameraman_manager is not None:
            self.cameraman_manager.set_cameramen(self.cameramen)
        self._refresh_target_combo(self.targets[0].id)
        self._generate_all()
        self._refresh_all()


def run_app(solver: Solver | None = None) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(solver=solver)
    window.show()
    sys.exit(app.exec())
