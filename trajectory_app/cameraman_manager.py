from __future__ import annotations

from .logging_config import get_logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
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

from .models import CameramanState
from .target_manager import COLORS

logger = get_logger(__name__)


class CameramanManager(QMainWindow):
    cameramenChanged = Signal(str)
    settingsChanged = Signal(str)
    activeCameramanRequested = Signal(str)
    beginPlacementRequested = Signal(str)
    stopPlacementRequested = Signal(str)

    def __init__(self, cameramen: list[CameramanState], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cameraman Manager")
        self.resize(820, 560)
        self.cameramen = cameramen
        self.placing_cameraman_id: str | None = None
        self._updating = False
        self._build_ui()
        self.refresh_from_cameramen()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        self.setCentralWidget(central)

        intro = QLabel(
            "Cameramen are static terrain entities. Place one with a map click; "
            "its altitude is the terrain altitude at that XY location. "
            "The view sector updates live when radius or angles change."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        left = QFrame()
        left_layout = QVBoxLayout(left)
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._selection_changed)
        left_layout.addWidget(self.list_widget, 1)

        row = QHBoxLayout()
        self.add_button = QPushButton("+ Add cameraman")
        self.remove_button = QPushButton("− Remove")
        self.add_button.clicked.connect(self._add)
        self.remove_button.clicked.connect(self._remove)
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        left_layout.addLayout(row)

        self.highlight_button = QPushButton("Highlight in main window")
        self.highlight_button.clicked.connect(self._request_active)
        left_layout.addWidget(self.highlight_button)

        self.place_button = QPushButton("Place / move cameraman")
        self.place_button.clicked.connect(self._toggle_placement)
        left_layout.addWidget(self.place_button)
        splitter.addWidget(left)

        right = QFrame()
        right_layout = QVBoxLayout(right)

        identity = QGroupBox("Cameraman identity")
        identity_form = QFormLayout(identity)
        self.name_edit = QLineEdit()
        self.color_combo = QComboBox()
        for name, color in COLORS:
            self.color_combo.addItem(self._color_icon(color), name, color)
        identity_form.addRow("Name", self.name_edit)
        identity_form.addRow("Color", self.color_combo)
        right_layout.addWidget(identity)

        view = QGroupBox("View cone")
        view_form = QFormLayout(view)
        self.radius = self._spin(0.01, 100000.0, 8.0, 1.0, " m")
        self.start_angle = self._spin(-360.0, 360.0, -30.0, 5.0, "°")
        self.end_angle = self._spin(-360.0, 360.0, 30.0, 5.0, "°")
        self.start_angle.setToolTip("0° = +X, 90° = +Y. If end < start, the sector wraps through 360°.")
        self.end_angle.setToolTip("0° = +X, 90° = +Y. If end < start, the sector wraps through 360°.")
        view_form.addRow("Radius", self.radius)
        view_form.addRow("Start azimuth", self.start_angle)
        view_form.addRow("End azimuth", self.end_angle)
        right_layout.addWidget(view)

        location = QGroupBox("Terrain location")
        location_form = QFormLayout(location)
        self.xy_label = QLabel("Not placed")
        self.z_label = QLabel("—")
        location_form.addRow("XY", self.xy_label)
        location_form.addRow("Terrain altitude Z", self.z_label)
        right_layout.addWidget(location)
        right_layout.addStretch(1)
        splitter.addWidget(right)
        splitter.setSizes([290, 510])

        self.name_edit.editingFinished.connect(self._name_changed)
        self.color_combo.currentIndexChanged.connect(self._settings_changed)
        self.radius.valueChanged.connect(self._settings_changed)
        self.start_angle.valueChanged.connect(self._settings_changed)
        self.end_angle.valueChanged.connect(self._settings_changed)

    @staticmethod
    def _spin(lo, hi, value, step, suffix) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(3)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        return spin

    @staticmethod
    def _color_icon(color: str) -> QIcon:
        pixmap = QPixmap(18, 18)
        pixmap.fill(QColor(color))
        return QIcon(pixmap)

    def set_cameramen(self, cameramen: list[CameramanState]) -> None:
        self.cameramen = cameramen
        self.refresh_from_cameramen()

    def set_placing_cameraman(self, cameraman_id: str | None) -> None:
        self.placing_cameraman_id = cameraman_id
        self._update_place_button()

    def select_cameraman(self, cameraman_id: str) -> None:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == cameraman_id:
                self.list_widget.setCurrentRow(row)
                return

    def _selected_id(self) -> str | None:
        item = self.list_widget.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _selected(self) -> CameramanState | None:
        selected_id = self._selected_id()
        return next((c for c in self.cameramen if c.id == selected_id), None)

    def refresh_from_cameramen(self, preferred_id: str | None = None) -> None:
        preferred_id = preferred_id or self._selected_id()
        self._updating = True
        self.list_widget.clear()
        row_to_select = 0
        for row, cameraman in enumerate(self.cameramen):
            item = QListWidgetItem(self._color_icon(cameraman.color), cameraman.name)
            item.setData(Qt.ItemDataRole.UserRole, cameraman.id)
            self.list_widget.addItem(item)
            if cameraman.id == preferred_id:
                row_to_select = row
        if self.cameramen:
            self.list_widget.setCurrentRow(row_to_select)
        self._updating = False
        self._load_selected()

    def _selection_changed(self, *_args) -> None:
        if not self._updating:
            self._load_selected()

    def _load_selected(self) -> None:
        cameraman = self._selected()
        enabled = cameraman is not None
        for widget in (
            self.name_edit,
            self.color_combo,
            self.radius,
            self.start_angle,
            self.end_angle,
            self.place_button,
            self.highlight_button,
            self.remove_button,
        ):
            widget.setEnabled(enabled)
        if cameraman is None:
            self.xy_label.setText("Not placed")
            self.z_label.setText("—")
            return

        self._updating = True
        self.name_edit.setText(cameraman.name)
        color_index = self.color_combo.findData(cameraman.color)
        self.color_combo.setCurrentIndex(max(0, color_index))
        self.radius.setValue(cameraman.view_radius_m)
        self.start_angle.setValue(cameraman.view_start_angle_deg)
        self.end_angle.setValue(cameraman.view_end_angle_deg)
        self._updating = False
        if cameraman.position_xyz is None:
            self.xy_label.setText("Not placed")
            self.z_label.setText("—")
        else:
            x, y, z = cameraman.position_xyz
            self.xy_label.setText(f"X={x:.3f}, Y={y:.3f} m")
            self.z_label.setText(f"{z:.3f} m")
        self._update_place_button()

    def _update_place_button(self) -> None:
        selected_id = self._selected_id()
        self.place_button.setText(
            "Cancel placement"
            if selected_id is not None and selected_id == self.placing_cameraman_id
            else "Place / move cameraman"
        )

    def _name_changed(self) -> None:
        if self._updating:
            return
        cameraman = self._selected()
        if cameraman is None:
            return
        name = self.name_edit.text().strip()
        if name:
            cameraman.name = name
            self.cameramenChanged.emit(cameraman.id)
            self.refresh_from_cameramen(cameraman.id)

    def _settings_changed(self) -> None:
        if self._updating:
            return
        cameraman = self._selected()
        if cameraman is None:
            return
        cameraman.color = str(self.color_combo.currentData())
        cameraman.view_radius_m = float(self.radius.value())
        cameraman.view_start_angle_deg = float(self.start_angle.value())
        cameraman.view_end_angle_deg = float(self.end_angle.value())
        self.settingsChanged.emit(cameraman.id)
        self.refresh_from_cameramen(cameraman.id)

    def _add(self) -> None:
        if len(self.cameramen) >= 10:
            QMessageBox.information(self, "Limit reached", "Up to 10 cameramen are supported.")
            return
        used = {c.color for c in self.cameramen}
        color = next((value for _name, value in COLORS if value not in used), COLORS[len(self.cameramen) % len(COLORS)][1])
        cameraman = CameramanState(f"Cameraman {len(self.cameramen) + 1}", color)
        self.cameramen.append(cameraman)
        logger.info("Added cameraman id=%s name=%s", cameraman.id, cameraman.name)
        self.cameramenChanged.emit(cameraman.id)
        self.refresh_from_cameramen(cameraman.id)

    def _remove(self) -> None:
        cameraman = self._selected()
        if cameraman is None:
            return
        if cameraman.id == self.placing_cameraman_id:
            self.stopPlacementRequested.emit(cameraman.id)
        logger.info("Removed cameraman id=%s name=%s", cameraman.id, cameraman.name)
        self.cameramen[:] = [c for c in self.cameramen if c.id != cameraman.id]
        preferred = self.cameramen[0].id if self.cameramen else ""
        self.cameramenChanged.emit(preferred)
        self.refresh_from_cameramen(preferred or None)

    def _request_active(self) -> None:
        cameraman = self._selected()
        if cameraman is not None:
            self.activeCameramanRequested.emit(cameraman.id)

    def _toggle_placement(self) -> None:
        cameraman = self._selected()
        if cameraman is None:
            return
        if cameraman.id == self.placing_cameraman_id:
            self.stopPlacementRequested.emit(cameraman.id)
        else:
            self.beginPlacementRequested.emit(cameraman.id)

