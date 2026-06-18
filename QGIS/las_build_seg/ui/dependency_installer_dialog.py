from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QProgressBar, QLabel, QTextEdit, QMessageBox
)
from qgis.PyQt.QtCore import QThread, pyqtSignal
from qgis.core import QgsMessageLog, Qgis


class InstallWorker(QThread):
    """Background thread for dependency installation."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self):
        super().__init__()
        self._cancel = False
    
    def run(self):
        # Import here to avoid circular imports
        import os
        import sys
        
        # Add core directory to path
        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        core_dir = os.path.join(plugin_dir, 'core')
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)
        
        from venv_manager import create_venv_and_install
        
        def progress_callback(percent, msg):
            self.progress.emit(percent, msg)
        
        def cancel_check():
            return self._cancel
        
        success, message = create_venv_and_install(
            progress_callback=progress_callback,
            cancel_check=cancel_check
        )
        
        self.finished.emit(success, message)
    
    def cancel(self):
        self._cancel = True


class DependencyInstallerDialog(QDialog):
    """Dialog for installing plugin dependencies."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LasBuildSeg - Dependency Installer")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        
        self.worker = None
        self.init_ui()
        self.update_status()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel(
            "<b>LasBuildSeg Dependency Installation</b><br><br>"
            "This plugin requires the following Python libraries:<br>"
            "• <b>numpy</b> - Numerical computations<br>"
            "• <b>scipy</b> - Scientific computing (interpolation, morphology)<br>"
            "• <b>alphashape</b> - Alpha shape generation<br>"
            "• <b>CSF</b> (Cloth Simulation Filter) - Ground point classification<br>"
            "• <b>GDAL</b> - Geospatial data processing<br><br>"
            "Click '<b>Install Dependencies</b>' to automatically download and install them.<br>"
            "<i>Total download size: ~100-150 MB | Installation time: 2-5 minutes</i>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Separator
        layout.addSpacing(10)
        
        # Status label
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Progress message
        self.progress_msg = QLabel("")
        layout.addWidget(self.progress_msg)
        
        # Log output
        log_label = QLabel("<b>Installation Log:</b>")
        layout.addWidget(log_label)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        self.log_output.setStyleSheet("font-family: monospace; font-size: 9pt;")
        layout.addWidget(self.log_output)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.install_btn = QPushButton("Install Dependencies")
        self.install_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.install_btn.clicked.connect(self.start_installation)
        button_layout.addWidget(self.install_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_installation)
        self.cancel_btn.setEnabled(False)
        button_layout.addWidget(self.cancel_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def update_status(self):
        import os
        import sys
        
        # Add core directory to path
        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        core_dir = os.path.join(plugin_dir, 'core')
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)
        
        from venv_manager import get_venv_status
        
        is_ready, status_msg = get_venv_status()
        
        if is_ready:
            self.status_label.setText(f"<b style='color: green;'>✓ Status: {status_msg}</b>")
            self.install_btn.setText("Reinstall Dependencies")
            self.log_output.append("✓ All dependencies are installed and ready!")
        else:
            self.status_label.setText(f"<b style='color: orange;'>⚠ Status: {status_msg}</b>")
            self.install_btn.setText("Install Dependencies")
            self.log_output.append(f"⚠ {status_msg}")
            self.log_output.append("Click 'Install Dependencies' to begin installation.")
    
    def start_installation(self):
        reply = QMessageBox.question(
            self,
            "Confirm Installation",
            "This will download ~100-150 MB and may take 2-5 minutes.\n\n"
            "The installation will:\n"
            "1. Download Python standalone (~50 MB)\n"
            "2. Create an isolated virtual environment\n"
            "3. Install required packages (numpy, scipy, alphashape, CSF, GDAL)\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        self.log_output.clear()
        self.log_output.append("=" * 60)
        self.log_output.append("Starting LasBuildSeg dependency installation...")
        self.log_output.append("=" * 60)
        self.log_output.append("")
        
        self.install_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.close_btn.setEnabled(False)
        
        self.worker = InstallWorker()
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
    
    def on_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.progress_msg.setText(message)
        self.log_output.append(f"[{percent:3d}%] {message}")
        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_finished(self, success, message):
        self.progress_bar.setValue(100 if success else 0)
        self.progress_msg.setText(message)
        
        self.log_output.append("")
        self.log_output.append("=" * 60)
        
        if success:
            self.log_output.append(f"✓ SUCCESS: {message}")
            self.log_output.append("=" * 60)
            self.log_output.append("")
            self.log_output.append("You can now use LasBuildSeg to process LiDAR data!")
            
            QMessageBox.information(
                self, 
                "Installation Successful", 
                f"{message}\n\n"
                "All dependencies are now installed.\n"
                "You can close this dialog and start using LasBuildSeg!"
            )
        else:
            self.log_output.append(f"✗ FAILED: {message}")
            self.log_output.append("=" * 60)
            
            QMessageBox.warning(
                self, 
                "Installation Failed", 
                f"Installation failed:\n{message}\n\n"
                "Please check the log for details.\n"
                "You may need to reinstall or contact support."
            )
        
        self.install_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        
        self.update_status()
        self.worker = None
        
        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def cancel_installation(self):
        if self.worker:
            reply = QMessageBox.question(
                self,
                "Cancel Installation",
                "Are you sure you want to cancel the installation?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.log_output.append("")
                self.log_output.append("⚠ Cancelling installation...")
                self.worker.cancel()
                self.cancel_btn.setEnabled(False)
