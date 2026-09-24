from __future__ import annotations

from .logging_config import get_logger
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal, QRectF
from PySide6.QtWidgets import (
    QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QPushButton,
    QDoubleSpinBox,QSpinBox,QFileDialog,QMessageBox,QLabel
)

from .core import generate_terrain
from .models import TerrainConfig
from .terrain_io import save_terrain


logger = get_logger(__name__)

class TerrainDesigner(QWidget):
    terrainSaved = Signal(str)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle("Terrain Designer")
        self.resize(1050,720)
        self.config=TerrainConfig(); self.terrain=generate_terrain(self.config)
        root=QHBoxLayout(self)
        left=QVBoxLayout(); root.addLayout(left,0)
        form=QFormLayout(); left.addLayout(form)
        self.fields={}
        specs=[
            ("seed","Seed",0,1000000,1),("grid_size","Grid size",20,300,1),
            ("x_min","X min",-10000,10000,.5),("x_max","X max",-10000,10000,.5),
            ("y_min","Y min",-10000,10000,.5),("y_max","Y max",-10000,10000,.5),
            ("base_height","Base height",-1000,10000,.5),("mountain_count","Mountains",0,30,1),
            ("valley_count","Valleys",0,20,1),("mountain_height_min","Mountain min",0,1000,.5),
            ("mountain_height_max","Mountain max",0,1000,.5),("valley_depth_min","Valley min",0,1000,.5),
            ("valley_depth_max","Valley max",0,1000,.5),("min_sigma","Sigma min",.01,1000,.25),
            ("max_sigma","Sigma max",.01,1000,.25),("noise_amplitude","Noise",0,100,.05),
        ]
        for key,label,lo,hi,step in specs:
            if key in {"seed","grid_size","mountain_count","valley_count"}:
                w=QSpinBox(); w.setRange(int(lo),int(hi)); w.setValue(int(getattr(self.config,key))); w.setSingleStep(int(step))
            else:
                w=QDoubleSpinBox(); w.setRange(lo,hi); w.setDecimals(3); w.setValue(float(getattr(self.config,key))); w.setSingleStep(step)
            self.fields[key]=w; form.addRow(label,w)
        row=QHBoxLayout(); left.addLayout(row)
        self.generate_btn=QPushButton("Generate"); self.save_btn=QPushButton("Save .terrain")
        row.addWidget(self.generate_btn); row.addWidget(self.save_btn)
        self.stats=QLabel(); left.addWidget(self.stats); left.addStretch(1)

        self.plot=pg.PlotWidget(); self.image=pg.ImageItem(); self.plot.addItem(self.image); self.plot.setAspectLocked(False)
        self.plot.setLabel("bottom","X","m"); self.plot.setLabel("left","Y","m"); root.addWidget(self.plot,1)
        self.generate_btn.clicked.connect(self.generate); self.save_btn.clicked.connect(self.save)
        self._show()

    def _read(self):
        vals={k:w.value() for k,w in self.fields.items()}; return TerrainConfig(**vals)

    def generate(self):
        try:
            self.config=self._read(); self.terrain=generate_terrain(self.config); self._show()
        except Exception as e: QMessageBox.critical(self,"Terrain generation failed",str(e))

    def _show(self):
        self.image.setImage(self.terrain.T,autoLevels=True)
        c=self.config
        self.image.setRect(QRectF(c.x_min,c.y_min,c.x_max-c.x_min,c.y_max-c.y_min))
        self.plot.setXRange(c.x_min,c.x_max,padding=.02); self.plot.setYRange(c.y_min,c.y_max,padding=.02)
        self.stats.setText(f"min={np.min(self.terrain):.2f} m   max={np.max(self.terrain):.2f} m")

    def save(self):
        path,_=QFileDialog.getSaveFileName(self,"Save terrain","terrain.terrain","Terrain (*.terrain)")
        if not path: return
        try:
            actual=save_terrain(path,self.config,self.terrain); self.terrainSaved.emit(str(actual)); self.stats.setText(self.stats.text()+f"\nSaved: {actual}")
        except Exception as e: QMessageBox.critical(self,"Save terrain failed",str(e))
