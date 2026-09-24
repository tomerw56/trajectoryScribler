from __future__ import annotations

from .logging_config import get_logger
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .models import (
    AltitudeMode,
    MotionConfig,
    TargetState,
    TrajectoryConfig,
    TrajectoryGenerationMode,
)


COLORS = [
    ("Red", "#e74c3c"),
    ("Blue", "#3498db"),
    ("Green", "#2ecc71"),
    ("Orange", "#f39c12"),
    ("Purple", "#9b59b6"),
    ("Cyan", "#00bcd4"),
    ("Pink", "#ff6fae"),
    ("Yellow", "#f1c40f"),
    ("Lime", "#9acd32"),
    ("White", "#ecf0f1"),
]


logger = get_logger(__name__)

class TargetManager(QMainWindow):
    """Dedicated editor for all target-specific settings.

    The manager mutates the shared TargetState objects and tells the main
    window what kind of refresh is required. The main window remains the
    owner of terrain-dependent trajectory generation and playback.
    """

    targetsChanged = Signal(str)          # preferred active target id
    settingsChanged = Signal(str, str)    # target id, geometry|motion|display
    activeTargetRequested = Signal(str)
    generateTargetRequested = Signal(str)
    generateAllRequested = Signal()
    clearScribbleRequested = Signal(str)
    beginDrawingRequested = Signal(str)
    beginDubinsDrawingRequested = Signal(str, float)
    stopDrawingRequested = Signal(str)

    def __init__(self, targets: list[TargetState], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Target Manager")
        self.resize(900, 760)
        self.targets = targets
        self._updating = False
        self.drawing_target_id: str | None = None
        self._build_ui()
        self.refresh_from_targets()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        self.setCentralWidget(central)

        intro = QLabel(
            "Each target owns its own color, altitude policy, constant 3D speed, "
            "motion sample rate, covariance settings, and future-position prediction."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # Left: target list and target-level actions.
        left = QFrame()
        left_layout = QVBoxLayout(left)
        self.target_list = QListWidget()
        self.target_list.currentItemChanged.connect(self._selection_changed)
        left_layout.addWidget(self.target_list, 1)

        add_remove = QHBoxLayout()
        self.add_button = QPushButton("+ Add target")
        self.remove_button = QPushButton("− Remove")
        self.add_button.clicked.connect(self._add_target)
        self.remove_button.clicked.connect(self._remove_target)
        add_remove.addWidget(self.add_button)
        add_remove.addWidget(self.remove_button)
        left_layout.addLayout(add_remove)

        self.make_active_button = QPushButton("Make active in main window")
        self.make_active_button.clicked.connect(self._request_active)
        left_layout.addWidget(self.make_active_button)

        self.draw_button = QPushButton("Begin drawing / redraw")
        self.draw_button.setToolTip(
            "Begin Drawing immediately clears this target's existing path. "
            "Generate the trajectory to commit the new drawing. "
            "Stopping without Generate discards the draft and resets the target."
        )
        self.draw_button.clicked.connect(self._toggle_drawing)
        left_layout.addWidget(self.draw_button)

        self.dubins_draw_button = QPushButton("Begin Dubins drawing / redraw")
        self.dubins_draw_button.setToolTip(
            "Ask for a minimum turn radius, then draw the intended path freely. "
            "Generate fits a piecewise Dubins trajectory through the broad shape."
        )
        self.dubins_draw_button.clicked.connect(self._begin_dubins_drawing)
        left_layout.addWidget(self.dubins_draw_button)

        self.generate_button = QPushButton("Generate selected trajectory")
        self.generate_button.clicked.connect(self._request_generate_selected)
        left_layout.addWidget(self.generate_button)

        self.generate_all_button = QPushButton("Generate all drawn trajectories")
        self.generate_all_button.clicked.connect(self.generateAllRequested.emit)
        left_layout.addWidget(self.generate_all_button)

        self.clear_button = QPushButton("Clear selected scribble")
        self.clear_button.clicked.connect(self._request_clear)
        left_layout.addWidget(self.clear_button)

        splitter.addWidget(left)

        # Right: settings for the selected target.
        right = QFrame()
        right_layout = QVBoxLayout(right)

        identity_group = QGroupBox("Target identity")
        identity_form = QFormLayout(identity_group)
        self.name_edit = QLineEdit()
        self.color_combo = QComboBox()
        for name, color in COLORS:
            self.color_combo.addItem(self._color_icon(color), name, color)
        identity_form.addRow("Name", self.name_edit)
        identity_form.addRow("Color", self.color_combo)
        right_layout.addWidget(identity_group)

        motion_group = QGroupBox("Target motion")
        motion_form = QFormLayout(motion_group)
        self.speed = self._spin(0.01, 10000.0, 4.0, 0.5, " m/s")
        self.speed.setToolTip(
            "Constant magnitude of the target's 3D velocity vector. "
            "Climb/descent changes Vx/Vy/Vz, but |V| remains this value."
        )
        self.dt = self._spin(0.01, 60.0, 0.25, 0.05, " s")
        self.cov_rate = self._spin(0.01, 60.0, 0.50, 0.10, " s")
        self.cov_window = self._spin(0.02, 600.0, 3.0, 0.50, " s")
        self.prediction_dt = self._spin(0.0, 600.0, 2.0, 0.25, " s")
        self.stability_window = self._spin(0.1, 600.0, 5.0, 0.5, " s")
        self.stability_window.setToolTip(
            "Trailing history duration used by the covariance stability calculator."
        )
        self.prediction_dt.setToolTip(
            "Future horizon for the 1σ predicted-position ellipse. "
            "The ellipse uses Σp = dt²·Σv and is centered at p + v·dt."
        )
        motion_form.addRow("Target speed |V|", self.speed)
        motion_form.addRow("Motion sample dt", self.dt)
        motion_form.addRow("Covariance sample rate", self.cov_rate)
        motion_form.addRow("Covariance window", self.cov_window)
        motion_form.addRow("Prediction horizon", self.prediction_dt)
        motion_form.addRow("Covariance stability window", self.stability_window)
        right_layout.addWidget(motion_group)

        altitude_group = QGroupBox("Altitude policy — owned by this target")
        altitude_form = QFormLayout(altitude_group)
        self.mode = QComboBox()
        self.mode.addItem(
            "Fixed increment / terrain clearance",
            AltitudeMode.TERRAIN_CLEARANCE.value,
        )
        self.mode.addItem(
            "Fixed altitude / cruise (with obstacle climb)",
            AltitudeMode.CRUISE_ALTITUDE.value,
        )
        self.resolution = self._spin(0.01, 1000.0, 0.25, 0.05, " m")
        self.clearance = self._spin(0.0, 10000.0, 3.0, 0.5, " m")
        self.cruise = self._spin(-10000.0, 50000.0, 8.0, 1.0, " m")
        self.angle = self._spin(0.1, 89.0, 18.0, 1.0, "°")
        altitude_form.addRow("Altitude mode", self.mode)
        altitude_form.addRow("Path resolution", self.resolution)
        altitude_form.addRow("Clearance / increment", self.clearance)
        altitude_form.addRow("Cruise altitude", self.cruise)
        altitude_form.addRow("Max climb/descent", self.angle)
        right_layout.addWidget(altitude_group)

        status_group = QGroupBox("Target state")
        status_form = QFormLayout(status_group)
        self.scribble_status = QLabel("—")
        self.generated_status = QLabel("—")
        self.sample_status = QLabel("—")
        status_form.addRow("Scribble", self.scribble_status)
        status_form.addRow("Generated path", self.generated_status)
        status_form.addRow("Motion samples", self.sample_status)
        right_layout.addWidget(status_group)

        derived_group = QGroupBox("Derived velocity model — read only")
        derived_form = QFormLayout(derived_group)
        self.derived_speed = QLabel("—")
        self.derived_turn_radius = QLabel("—")
        self.derived_turn_rate = QLabel("—")
        self.derived_lateral_accel = QLabel("—")
        self.derived_climb_angle = QLabel("—")
        self.derived_descent_angle = QLabel("—")
        self.derived_climb_rate = QLabel("—")
        self.derived_descent_rate = QLabel("—")

        derived_form.addRow("Speed", self.derived_speed)
        derived_form.addRow("Minimum turn radius", self.derived_turn_radius)
        derived_form.addRow("Maximum turn rate", self.derived_turn_rate)
        derived_form.addRow("Maximum lateral accel", self.derived_lateral_accel)
        derived_form.addRow("Maximum climb angle", self.derived_climb_angle)
        derived_form.addRow("Maximum descent angle", self.derived_descent_angle)
        derived_form.addRow("Maximum climb rate", self.derived_climb_rate)
        derived_form.addRow("Maximum descent rate", self.derived_descent_rate)
        right_layout.addWidget(derived_group)

        right_layout.addStretch(1)

        splitter.addWidget(right)
        splitter.setSizes([280, 600])

        self.name_edit.editingFinished.connect(self._name_changed)
        self.color_combo.currentIndexChanged.connect(self._color_changed)
        self.mode.currentIndexChanged.connect(self._geometry_changed)
        for widget in (self.resolution, self.clearance, self.cruise, self.angle):
            widget.valueChanged.connect(self._geometry_changed)
        for widget in (self.speed, self.dt, self.cov_rate, self.cov_window):
            widget.valueChanged.connect(self._motion_changed)
        self.prediction_dt.valueChanged.connect(self._prediction_changed)
        self.stability_window.valueChanged.connect(self._stability_window_changed)

    @staticmethod
    def _spin(lo, hi, value, step, suffix) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(3)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        return spin

    @staticmethod
    def _color_icon(color: str) -> QIcon:
        pixmap = QPixmap(18, 18)
        pixmap.fill(QColor(color))
        return QIcon(pixmap)

    def set_targets(self, targets: list[TargetState]) -> None:
        self.targets = targets
        self.refresh_from_targets()

    def _selected_id(self) -> str | None:
        item = self.target_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _selected_target(self) -> TargetState | None:
        target_id = self._selected_id()
        return next((t for t in self.targets if t.id == target_id), None)

    def refresh_from_targets(self, preferred_id: str | None = None) -> None:
        preferred_id = preferred_id or self._selected_id()
        self._updating = True
        self.target_list.clear()
        selected_row = 0
        for row, target in enumerate(self.targets):
            item = QListWidgetItem(self._color_icon(target.color), target.name)
            item.setData(Qt.ItemDataRole.UserRole, target.id)
            self.target_list.addItem(item)
            if target.id == preferred_id:
                selected_row = row
        if self.targets:
            self.target_list.setCurrentRow(selected_row)
        self._updating = False
        self._load_selected()

    def select_target(self, target_id: str) -> None:
        for row in range(self.target_list.count()):
            item = self.target_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == target_id:
                self.target_list.setCurrentRow(row)
                return

    def _selection_changed(self, *_args) -> None:
        if not self._updating:
            self._load_selected()

    def _load_selected(self) -> None:
        target = self._selected_target()
        enabled = target is not None
        for widget in (
            self.name_edit,
            self.color_combo,
            self.speed,
            self.dt,
            self.cov_rate,
            self.cov_window,
            self.prediction_dt,
            self.stability_window,
            self.mode,
            self.resolution,
            self.clearance,
            self.cruise,
            self.angle,
            self.make_active_button,
            self.draw_button,
            self.dubins_draw_button,
            self.generate_button,
            self.clear_button,
            self.remove_button,
        ):
            widget.setEnabled(enabled)

        if target is None:
            self.scribble_status.setText("—")
            self.generated_status.setText("—")
            self.sample_status.setText("—")
            self._clear_derived_model()
            return

        self._updating = True
        self.name_edit.setText(target.name)
        color_index = self.color_combo.findData(target.color)
        self.color_combo.setCurrentIndex(max(0, color_index))
        mode_index = self.mode.findData(target.trajectory_config.altitude_mode.value)
        self.mode.setCurrentIndex(max(0, mode_index))
        self.resolution.setValue(target.trajectory_config.path_resolution_m)
        self.clearance.setValue(target.trajectory_config.clearance_m)
        self.cruise.setValue(target.trajectory_config.cruise_altitude_m)
        self.angle.setValue(target.trajectory_config.max_climb_angle_deg)
        self.speed.setValue(target.motion_config.speed_mps)
        self.dt.setValue(target.motion_config.sample_dt_sec)
        self.cov_rate.setValue(target.motion_config.covariance_sample_rate_sec)
        self.cov_window.setValue(target.motion_config.covariance_window_sec)
        self.prediction_dt.setValue(target.motion_config.prediction_dt_sec)
        self.stability_window.setValue(
            target.motion_config.covariance_stability_window_sec
        )
        self._updating = False

        self._update_mode_enabled()
        self._update_drawing_button()
        self._update_status(target)

    def _update_mode_enabled(self) -> None:
        cruise = self.mode.currentData() == AltitudeMode.CRUISE_ALTITUDE.value
        self.cruise.setEnabled(cruise)
        self.angle.setEnabled(cruise)

    def _update_drawing_button(self) -> None:
        target = self._selected_target()
        is_drawing = bool(target and target.id == self.drawing_target_id)
        if is_drawing:
            self.draw_button.setText("Stop drawing")
            self.dubins_draw_button.setText("Stop drawing")
        else:
            self.draw_button.setText("Begin freehand drawing / redraw")
            self.dubins_draw_button.setText("Begin Dubins drawing / redraw")

    def set_drawing_target(self, target_id: str | None) -> None:
        self.drawing_target_id = target_id
        self._update_drawing_button()
        target = self._selected_target()
        if target is not None:
            self._update_status(target)

    def _clear_derived_model(self) -> None:
        for label in (
            self.derived_speed,
            self.derived_turn_radius,
            self.derived_turn_rate,
            self.derived_lateral_accel,
            self.derived_climb_angle,
            self.derived_descent_angle,
            self.derived_climb_rate,
            self.derived_descent_rate,
        ):
            label.setText("—")

    def _update_derived_model(self, target: TargetState) -> None:
        model = target.derived_velocity_model
        if model is None:
            self._clear_derived_model()
            self.derived_speed.setText("Generate trajectory first")
            return

        self.derived_speed.setText(f"{model.speed_mps:.3f} m/s")
        if np.isfinite(model.min_turn_radius_m):
            self.derived_turn_radius.setText(
                f"{model.min_turn_radius_m:.3f} m"
            )
        else:
            self.derived_turn_radius.setText("∞ (straight)")

        self.derived_turn_rate.setText(
            f"{model.max_turn_rate_deg_s:.3f} °/s"
        )
        self.derived_lateral_accel.setText(
            f"{model.max_lateral_accel_mps2:.3f} m/s²"
        )
        self.derived_climb_angle.setText(
            f"{model.max_climb_angle_deg:.3f}°"
        )
        self.derived_descent_angle.setText(
            f"{model.max_descent_angle_deg:.3f}°"
        )
        self.derived_climb_rate.setText(
            f"{model.max_climb_rate_mps:.3f} m/s"
        )
        self.derived_descent_rate.setText(
            f"{model.max_descent_rate_mps:.3f} m/s"
        )

    def _update_status(self, target: TargetState) -> None:
        if target.trajectory_config.generation_mode == TrajectoryGenerationMode.DUBINS:
            generator = (
                f"Dubins Rmin={target.trajectory_config.dubins_min_turn_radius_m:.2f} m"
            )
        else:
            generator = "Freehand"

        if target.id == self.drawing_target_id:
            self.scribble_status.setText(
                f"DRAWING — {len(target.scribble_xy)} raw points — {generator}"
            )
        else:
            self.scribble_status.setText(
                f"{len(target.scribble_xy)} raw points — {generator}"
            )

        if target.trajectory is None:
            generated_text = "not generated"
        elif target.trajectory_config.generation_mode == TrajectoryGenerationMode.DUBINS:
            fit = (
                ""
                if target.dubins_max_fit_error_m is None
                else (
                    f" | fit mean={target.dubins_mean_fit_error_m:.2f} m "
                    f"max={target.dubins_max_fit_error_m:.2f} m"
                )
            )
            generated_text = f"{len(target.trajectory.xyz)} 3D path points{fit}"
        else:
            generated_text = f"{len(target.trajectory.xyz)} 3D path points"

        self.generated_status.setText(generated_text)
        self.sample_status.setText(
            "not sampled" if target.samples is None else f"{len(target.samples)} samples"
        )
        self._update_derived_model(target)

    def _name_changed(self) -> None:
        if self._updating:
            return
        target = self._selected_target()
        if target is None:
            return
        target.name = self.name_edit.text().strip() or target.name
        self.refresh_from_targets(target.id)
        self.targetsChanged.emit(target.id)

    def _color_changed(self) -> None:
        if self._updating:
            return
        target = self._selected_target()
        if target is None:
            return
        target.color = str(self.color_combo.currentData())
        self.refresh_from_targets(target.id)
        self.settingsChanged.emit(target.id, "display")
        self.targetsChanged.emit(target.id)

    def _geometry_changed(self) -> None:
        if self._updating:
            return
        target = self._selected_target()
        if target is None:
            return
        target.trajectory_config = TrajectoryConfig(
            path_resolution_m=self.resolution.value(),
            altitude_mode=AltitudeMode(self.mode.currentData()),
            clearance_m=self.clearance.value(),
            cruise_altitude_m=self.cruise.value(),
            max_climb_angle_deg=self.angle.value(),
            smoothing_passes=target.trajectory_config.smoothing_passes,
            generation_mode=target.trajectory_config.generation_mode,
            dubins_min_turn_radius_m=target.trajectory_config.dubins_min_turn_radius_m,
        )
        self._update_mode_enabled()
        self.settingsChanged.emit(target.id, "geometry")
        self._update_status(target)

    def _motion_changed(self) -> None:
        if self._updating:
            return
        target = self._selected_target()
        if target is None:
            return
        target.motion_config = MotionConfig(
            speed_mps=self.speed.value(),
            sample_dt_sec=self.dt.value(),
            covariance_sample_rate_sec=self.cov_rate.value(),
            covariance_window_sec=self.cov_window.value(),
            prediction_dt_sec=self.prediction_dt.value(),
            covariance_stability_window_sec=self.stability_window.value(),
        )
        self.settingsChanged.emit(target.id, "motion")
        self._update_status(target)

    def _stability_window_changed(self) -> None:
        if self._updating:
            return
        target = self._selected_target()
        if target is None:
            return
        target.motion_config.covariance_stability_window_sec = (
            self.stability_window.value()
        )
        self.settingsChanged.emit(target.id, "display")
        self._update_status(target)

    def _prediction_changed(self) -> None:
        if self._updating:
            return
        target = self._selected_target()
        if target is None:
            return
        target.motion_config.prediction_dt_sec = self.prediction_dt.value()
        self.settingsChanged.emit(target.id, "display")
        self._update_status(target)

    def _add_target(self) -> None:
        if len(self.targets) >= 10:
            QMessageBox.information(
                self,
                "Target limit",
                "Up to 10 trajectories/targets are supported.",
            )
            return

        used = {target.color for target in self.targets}
        color = next(
            (value for _, value in COLORS if value not in used),
            COLORS[len(self.targets) % len(COLORS)][1],
        )
        target = TargetState(f"Target {len(self.targets) + 1}", color)
        self.targets.append(target)
        self.refresh_from_targets(target.id)
        self.targetsChanged.emit(target.id)

    def _remove_target(self) -> None:
        target = self._selected_target()
        if target is None:
            return
        if len(self.targets) == 1:
            QMessageBox.information(self, "Cannot remove", "Keep at least one target.")
            return
        self.targets.remove(target)
        preferred = self.targets[0].id
        self.refresh_from_targets(preferred)
        self.targetsChanged.emit(preferred)

    def _request_active(self) -> None:
        target = self._selected_target()
        if target is not None:
            self.activeTargetRequested.emit(target.id)

    def _toggle_drawing(self) -> None:
        target = self._selected_target()
        if target is None:
            return
        if target.id == self.drawing_target_id:
            logger.info("Stop drawing requested from Target Manager: target=%s", target.name)
            self.stopDrawingRequested.emit(target.id)
        else:
            logger.info("Begin drawing requested from Target Manager: target=%s", target.name)
            self.beginDrawingRequested.emit(target.id)

    def _begin_dubins_drawing(self) -> None:
        target = self._selected_target()
        if target is None:
            return
        if target.id == self.drawing_target_id:
            logger.info(
                "Stop drawing requested from Target Manager: target=%s",
                target.name,
            )
            self.stopDrawingRequested.emit(target.id)
            return

        radius, accepted = QInputDialog.getDouble(
            self,
            "Dubins minimum turn radius",
            f"{target.name}: minimum turn radius (m)",
            float(target.trajectory_config.dubins_min_turn_radius_m),
            0.10,
            10000.0,
            2,
        )
        if not accepted:
            return

        logger.info(
            "Begin Dubins drawing requested: target=%s min_radius=%.3f",
            target.name,
            radius,
        )
        self.beginDubinsDrawingRequested.emit(target.id, float(radius))


    def _request_generate_selected(self) -> None:
        target = self._selected_target()
        if target is not None:
            self.generateTargetRequested.emit(target.id)

    def _request_clear(self) -> None:
        target = self._selected_target()
        if target is not None:
            self.clearScribbleRequested.emit(target.id)
