from pathlib import Path

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .predictor2026_dialog import Predictor2026Dialog


class Predictor2026Plugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = Path(__file__).resolve().parent
        self.action = None
        self.dialog = None

    def initGui(self):
        icon = QIcon(str(self.plugin_dir / "icon.png"))
        self.action = QAction(icon, "Predictor2026", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&Predictor2026", self.action)

    def unload(self):
        if self.action:
            self.iface.removePluginMenu("&Predictor2026", self.action)
            self.iface.removeToolBarIcon(self.action)

    def run(self):
        if self.dialog is None:
            self.dialog = Predictor2026Dialog(self.iface, self.plugin_dir)
        self.dialog.showNormal()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        self.dialog.show_about_once()
