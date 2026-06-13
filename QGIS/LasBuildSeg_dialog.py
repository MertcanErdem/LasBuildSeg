"""
LasBuildSeg Dialog — with advanced False-Positive Elimination controls
"""

import os
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QGroupBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QComboBox, QFormLayout,
    QFrame, QScrollArea, QWidget
)
from qgis.gui import QgsFileWidget


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------
PRESETS = {
    "Balanced (default)": {
        "alpha":               0.7,
        "min_size":            100,
        "max_size":            50000,
        "squareness":          0.30,
        "width":               3.0,
        "kernel_size":         7,
        "use_height_var":      False,  # NEW filters OFF by default to match standalone
        "height_var_thresh":   2.5,
        "use_convex_hull":     False,
        "convex_hull_ratio_min": 0.50,
        "use_elongation":      False,
        "max_elongation":      8.0,
        "use_per_component":   False,  # Use global alpha shape like standalone
    },
    "Conservative (fewest FP)": {
        "alpha":               0.7,
        "min_size":            150,
        "max_size":            50000,
        "squareness":          0.40,
        "width":               4.0,
        "kernel_size":         9,
        "use_height_var":      True,  # Enable strict filters
        "height_var_thresh":   1.8,
        "use_convex_hull":     True,
        "convex_hull_ratio_min": 0.60,
        "use_elongation":      True,
        "max_elongation":      6.0,
        "use_per_component":   True,
    },
    "Sensitive (fewest FN)": {
        "alpha":               0.5,
        "min_size":            50,
        "max_size":            100000,
        "squareness":          0.20,
        "width":               2.0,
        "kernel_size":         3,
        "use_height_var":      False,  # Disable strict filters
        "height_var_thresh":   4.0,
        "use_convex_hull":     False,
        "convex_hull_ratio_min": 0.35,
        "use_elongation":      False,
        "max_elongation":      12.0,
        "use_per_component":   False,
    },
}


def _separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


class LasBuildSegDialog(QDialog):
    """
    Full-featured dialog for LasBuildSeg with:
      - Core extraction parameters
      - Advanced false-positive elimination controls (NEW)
      - Quick presets: Balanced / Conservative / Sensitive
      - Tooltips on every parameter
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LasBuildSeg — Building Extraction from LiDAR")
        self.setMinimumWidth(540)
        self.setMinimumHeight(680)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)

        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(scroll)

        main_layout = QVBoxLayout(inner)
        main_layout.setSpacing(8)

        # ── 1. INPUT / OUTPUT ──────────────────────────────────────────
        io_group = QGroupBox("Input / Output")
        io_form = QFormLayout(io_group)

        self.mQgsFileWidget_laz = QgsFileWidget()
        self.mQgsFileWidget_laz.setFilter("LAS/LAZ files (*.las *.laz)")
        self.mQgsFileWidget_laz.setStorageMode(QgsFileWidget.GetFile)
        self.mQgsFileWidget_laz.fileChanged.connect(self._on_input_changed)
        io_form.addRow("LAS/LAZ file:", self.mQgsFileWidget_laz)

        self.spin_epsg = QSpinBox()
        self.spin_epsg.setRange(0, 999999)
        self.spin_epsg.setValue(6457)
        self.spin_epsg.setToolTip(
            "EPSG of input LAZ CRS. Required for correct EPSG:3857 output.\n"
            "6457 = NAD83(2011) Illinois West (US Survey Feet)\n"
            "32614 = WGS84 UTM 14N (metres)   27700 = British National Grid"
        )
        io_form.addRow("Input CRS (EPSG):", self.spin_epsg)

        self.spin_resolution = QDoubleSpinBox()
        self.spin_resolution.setRange(0.1, 100.0)
        self.spin_resolution.setValue(1.0)
        self.spin_resolution.setSingleStep(0.5)
        self.spin_resolution.setDecimals(2)
        self.spin_resolution.setToolTip(
            "DSM/DTM grid resolution in the input CRS units.\n"
            "For US Survey Feet (EPSG:6457): 1.0 = 1 ft cells\n"
            "For metres (UTM): 1.0 = 1 m cells\n"
            "Lower = finer detail but much slower."
        )
        io_form.addRow("Grid resolution:", self.spin_resolution)

        out_row = QHBoxLayout()
        self.lineEdit_Output = QLineEdit()
        self.lineEdit_Output.setPlaceholderText("Same folder as input file")
        out_row.addWidget(self.lineEdit_Output)
        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._browse_output)
        out_row.addWidget(btn_browse)
        io_form.addRow("Output folder:", out_row)

        main_layout.addWidget(io_group)

        # ── 2. PRESET ─────────────────────────────────────────────────
        preset_group = QGroupBox("Quick Preset")
        preset_lay = QHBoxLayout(preset_group)
        preset_lay.addWidget(QLabel("Preset:"))
        self.combo_preset = QComboBox()
        for name in PRESETS:
            self.combo_preset.addItem(name)
        preset_lay.addWidget(self.combo_preset, stretch=1)
        btn_apply = QPushButton("Apply")
        btn_apply.setFixedWidth(60)
        btn_apply.clicked.connect(self._apply_preset)
        preset_lay.addWidget(btn_apply)
        main_layout.addWidget(preset_group)

        # ── 3. CORE EXTRACTION PARAMETERS ─────────────────────────────
        core_group = QGroupBox("Core Extraction Parameters")
        core_form = QFormLayout(core_group)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.01, 5.0)
        self.spin_alpha.setSingleStep(0.05)
        self.spin_alpha.setValue(0.7)
        self.spin_alpha.setDecimals(2)
        self.spin_alpha.setToolTip(
            "Alpha Shape parameter.\n"
            "Higher = tighter fit (can fragment large roofs).\n"
            "Lower = smoother shapes (can merge nearby buildings).\n"
            "Typical range: 0.4–1.0"
        )
        core_form.addRow("Alpha Shape (α):", self.spin_alpha)

        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(0, 25)
        self.spin_kernel.setSingleStep(2)
        self.spin_kernel.setValue(7)
        self.spin_kernel.setToolTip(
            "Morphological Opening kernel applied BEFORE alpha shape.\n"
            "Removes isolated noise pixels smaller than kernel x kernel.\n"
            "Set 0 to disable. Use odd numbers: 3, 5, 7, 9."
        )
        core_form.addRow("Morph Open kernel (pre-alpha):", self.spin_kernel)

        self.spin_min_size = QSpinBox()
        self.spin_min_size.setRange(1, 100000)
        self.spin_min_size.setSingleStep(10)
        self.spin_min_size.setValue(100)
        self.spin_min_size.setSuffix(" m2")
        self.spin_min_size.setToolTip("Minimum building footprint area to keep.")
        core_form.addRow("Min building area:", self.spin_min_size)

        self.spin_max_size = QSpinBox()
        self.spin_max_size.setRange(100, 1000000)
        self.spin_max_size.setSingleStep(1000)
        self.spin_max_size.setValue(50000)
        self.spin_max_size.setSuffix(" m2")
        self.spin_max_size.setToolTip("Maximum building footprint area to keep.")
        core_form.addRow("Max building area:", self.spin_max_size)

        self.spin_squareness = QDoubleSpinBox()
        self.spin_squareness.setRange(0.0, 1.0)
        self.spin_squareness.setSingleStep(0.05)
        self.spin_squareness.setValue(0.30)
        self.spin_squareness.setDecimals(2)
        self.spin_squareness.setToolTip(
            "Squareness = polygon area / oriented bounding box area.\n"
            "Range 0-1 where 1 = perfect rectangle.\n"
            "Candidates below this are dropped. Default 0.30."
        )
        core_form.addRow("Squareness threshold:", self.spin_squareness)

        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(0.0, 100.0)
        self.spin_width.setSingleStep(0.5)
        self.spin_width.setValue(3.0)
        self.spin_width.setSuffix(" m")
        self.spin_width.setToolTip(
            "Minimum short-side width from the oriented bounding box.\n"
            "Removes thin sliver detections (walls, roads). Default 3 m."
        )
        core_form.addRow("Min width threshold:", self.spin_width)

        main_layout.addWidget(core_group)

        # ── 4. FALSE-POSITIVE ELIMINATION (optional filters) ──────────
        fp_group = QGroupBox("False-Positive Elimination  (optional - disable to match standalone code)")
        fp_form = QFormLayout(fp_group)

        self.chk_per_component = QCheckBox(
            "Per-component alpha shape"
        )
        self.chk_per_component.setChecked(False)  # OFF by default to match standalone
        self.chk_per_component.setToolTip(
            "Labels connected pixel regions first, then alpha shape per component.\n\n"
            "Benefits:\n"
            "  • Prevents vegetation merging with building blobs\n"
            "  • Shape filters work more accurately per-building\n\n"
            "Disable to match standalone code (uses global alpha shape)."
        )
        fp_form.addRow("", self.chk_per_component)

        fp_form.addRow(_separator())

        # Height Variance (checkbox + spinbox)
        hv_row = QHBoxLayout()
        self.chk_use_height_var = QCheckBox("Enable")
        self.chk_use_height_var.setChecked(False)  # OFF by default
        self.chk_use_height_var.toggled.connect(self._update_filter_states)
        hv_row.addWidget(self.chk_use_height_var)
        hv_row.addWidget(QLabel("Max height variance:"))
        self.spin_height_var = QDoubleSpinBox()
        self.spin_height_var.setRange(0.5, 20.0)
        self.spin_height_var.setSingleStep(0.1)
        self.spin_height_var.setValue(2.5)
        self.spin_height_var.setDecimals(1)
        self.spin_height_var.setSuffix(" m")
        self.spin_height_var.setEnabled(False)  # Grayed out initially
        self.spin_height_var.setToolTip(
            "Height Variance Filter — drops candidates whose nDHM std-dev exceeds this.\n\n"
            "Buildings: flat roofs → LOW variance (< 1-2 m)\n"
            "Trees / vegetation → HIGH variance (> 3-5 m)\n\n"
            "WARNING: Can remove buildings near trees. Use carefully.\n"
            "Recommended: 2.5-3.5 m when enabled."
        )
        hv_row.addWidget(self.spin_height_var)
        hv_row.addStretch()
        fp_form.addRow("Height Variance:", hv_row)

        # Convex Hull (checkbox + spinbox)
        cvx_row = QHBoxLayout()
        self.chk_use_convex_hull = QCheckBox("Enable")
        self.chk_use_convex_hull.setChecked(False)  # OFF by default
        self.chk_use_convex_hull.toggled.connect(self._update_filter_states)
        cvx_row.addWidget(self.chk_use_convex_hull)
        cvx_row.addWidget(QLabel("Min convex hull ratio:"))
        self.spin_convex = QDoubleSpinBox()
        self.spin_convex.setRange(0.3, 1.0)
        self.spin_convex.setSingleStep(0.05)
        self.spin_convex.setValue(0.50)
        self.spin_convex.setDecimals(2)
        self.spin_convex.setEnabled(False)
        self.spin_convex.setToolTip(
            "Convex Hull Ratio = polygon area / convex hull area.\n\n"
            "Buildings: 0.7-1.0 (convex)\n"
            "Trees / branchy blobs: 0.3-0.6 (concave)\n\n"
            "Recommended: 0.45-0.60 when enabled."
        )
        cvx_row.addWidget(self.spin_convex)
        cvx_row.addStretch()
        fp_form.addRow("Convex Hull:", cvx_row)

        # Elongation (checkbox + spinbox)
        elong_row = QHBoxLayout()
        self.chk_use_elongation = QCheckBox("Enable")
        self.chk_use_elongation.setChecked(False)  # OFF by default
        self.chk_use_elongation.toggled.connect(self._update_filter_states)
        elong_row.addWidget(self.chk_use_elongation)
        elong_row.addWidget(QLabel("Max elongation:"))
        self.spin_elongation = QDoubleSpinBox()
        self.spin_elongation.setRange(2.0, 50.0)
        self.spin_elongation.setSingleStep(0.5)
        self.spin_elongation.setValue(8.0)
        self.spin_elongation.setDecimals(1)
        self.spin_elongation.setEnabled(False)
        self.spin_elongation.setToolTip(
            "Elongation = OBB long-side / short-side.\n\n"
            "Buildings: typically < 5\n"
            "Roads / power lines: 10-50+\n\n"
            "Recommended: 6-10 when enabled."
        )
        elong_row.addWidget(self.spin_elongation)
        elong_row.addStretch()
        fp_form.addRow("Elongation:", elong_row)

        main_layout.addWidget(fp_group)

        # ── 5. RE-RUN OPTIONS ─────────────────────────────────────────
        rerun_group = QGroupBox("Re-run Options")
        rerun_lay = QVBoxLayout(rerun_group)

        self.chk_skip_rasters = QCheckBox(
            "Skip DSM / DTM / nDHM generation  (use existing files)"
        )
        self.chk_skip_rasters.setChecked(False)
        self.chk_skip_rasters.setToolTip(
            "Check this on subsequent runs to reuse the DSM, DTM and nDHM\n"
            "already present in the output folder.\n\n"
            "Saves several minutes — only re-runs the building extraction\n"
            "and filtering steps with your updated parameters.\n\n"
            "Leave unchecked on first run, or after changing the input file."
        )
        rerun_lay.addWidget(self.chk_skip_rasters)

        # Show which files will be reused when checkbox is toggled
        self._reuse_label = QLabel("")
        self._reuse_label.setStyleSheet("color: #1565c0; font-size: 9pt;")
        rerun_lay.addWidget(self._reuse_label)

        self.chk_skip_rasters.toggled.connect(self._update_reuse_label)

        main_layout.addWidget(rerun_group)

        # ── 6. INFO ───────────────────────────────────────────────────
        info_label = QLabel(
            "Outputs created in the output folder:\n"
            "  dsm.tif  |  dtm.tif  |  ndhm.tif\n"
            "  buildings_filtered.geojson  (final results)\n"
            "  buildings_filtered_debug_candidates.geojson  (before filtering)"
        )
        info_label.setStyleSheet("color: gray; font-size: 9pt;")
        main_layout.addWidget(info_label)

        # ── 6. RUN BUTTON ─────────────────────────────────────────────
        self.pushButton_Run = QPushButton("Run Processing")
        self.pushButton_Run.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 11pt; padding: 10px; "
            "background-color: #2e7d32; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #388e3c; }"
            "QPushButton:pressed { background-color: #1b5e20; }"
        )
        main_layout.addWidget(self.pushButton_Run)

        self._apply_preset_by_name("Balanced (default)")

    # ── Public API ────────────────────────────────────────────────────
    def get_param(self, name, default=None):
        """Return current value of a named parameter. Called by run_logic."""
        mapping = {
            'epsg_code':             self.spin_epsg,
            'resolution':            self.spin_resolution,
            'alpha':                 self.spin_alpha,
            'min_size':              self.spin_min_size,
            'max_size':              self.spin_max_size,
            'squareness':            self.spin_squareness,
            'width':                 self.spin_width,
            'kernel_size':           self.spin_kernel,
            'height_var_thresh':     self.spin_height_var,
            'convex_hull_ratio_min': self.spin_convex,
            'max_elongation':        self.spin_elongation,
            'use_per_component':     self.chk_per_component,
            'skip_rasters':          self.chk_skip_rasters,
        }
        widget = mapping.get(name)
        if widget is None:
            return default
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        
        # For the NEW filters, return 0/disabled value if checkbox is unchecked
        if name == 'height_var_thresh' and not self.chk_use_height_var.isChecked():
            return 0.0  # Disabled
        if name == 'convex_hull_ratio_min' and not self.chk_use_convex_hull.isChecked():
            return 0.0  # Disabled
        if name == 'max_elongation' and not self.chk_use_elongation.isChecked():
            return 999.0  # Effectively disabled (no building is this elongated)
        
        return widget.value()

    # ── Internal helpers ──────────────────────────────────────────────
    def _apply_preset(self):
        self._apply_preset_by_name(self.combo_preset.currentText())

    def _apply_preset_by_name(self, name):
        p = PRESETS.get(name)
        if not p:
            return
        self.spin_alpha.setValue(p['alpha'])
        self.spin_min_size.setValue(int(p['min_size']))
        self.spin_max_size.setValue(int(p['max_size']))
        self.spin_squareness.setValue(p['squareness'])
        self.spin_width.setValue(p['width'])
        self.spin_kernel.setValue(int(p['kernel_size']))
        
        # Set checkbox states first
        self.chk_use_height_var.setChecked(p['use_height_var'])
        self.chk_use_convex_hull.setChecked(p['use_convex_hull'])
        self.chk_use_elongation.setChecked(p['use_elongation'])
        self.chk_per_component.setChecked(p['use_per_component'])
        
        # Then set values (spinboxes will auto-enable/disable via signals)
        self.spin_height_var.setValue(p['height_var_thresh'])
        self.spin_convex.setValue(p['convex_hull_ratio_min'])
        self.spin_elongation.setValue(p['max_elongation'])

    def _update_filter_states(self):
        """Enable/disable filter spinboxes based on checkbox states."""
        self.spin_height_var.setEnabled(self.chk_use_height_var.isChecked())
        self.spin_convex.setEnabled(self.chk_use_convex_hull.isChecked())
        self.spin_elongation.setEnabled(self.chk_use_elongation.isChecked())

    def _update_reuse_label(self, checked):
        """Show which files will be reused, and warn if any are missing."""
        if not checked:
            self._reuse_label.setText("")
            return

        out_dir = self.lineEdit_Output.text()
        if not out_dir:
            fp = self.mQgsFileWidget_laz.filePath()
            if fp and os.path.exists(fp):
                out_dir = os.path.dirname(os.path.abspath(fp))

        if not out_dir:
            self._reuse_label.setText("  Set output folder to check existing files.")
            return

        files = {"dsm.tif": False, "dtm.tif": False, "ndhm.tif": False}
        for fname in files:
            files[fname] = os.path.exists(os.path.join(out_dir, fname))

        missing = [f for f, found in files.items() if not found]
        if missing:
            self._reuse_label.setText(
                f"  WARNING: missing in output folder: {', '.join(missing)}\n"
                f"  Uncheck this option for first run."
            )
            self._reuse_label.setStyleSheet("color: #b71c1c; font-size: 9pt;")
        else:
            self._reuse_label.setText(
                "  Found: dsm.tif, dtm.tif, ndhm.tif — will be reused."
            )
            self._reuse_label.setStyleSheet("color: #1b5e20; font-size: 9pt;")

    def _on_input_changed(self, filepath):
        if filepath and os.path.exists(filepath) and not self.lineEdit_Output.text():
            self.lineEdit_Output.setText(os.path.dirname(os.path.abspath(filepath)))
        if self.chk_skip_rasters.isChecked():
            self._update_reuse_label(True)

    def _browse_output(self):
        start = self.lineEdit_Output.text()
        if not start:
            fp = self.mQgsFileWidget_laz.filePath()
            start = os.path.dirname(fp) if fp and os.path.exists(fp) else os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", start,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder:
            self.lineEdit_Output.setText(folder)
