import json
from pathlib import Path

import numpy as np

from .predictor2026_runner import (
    run_prediction,
    _boundary_mask,
    _open_array,
    _same_grid,
    _valid_mask,
    _warp_to_reference,
)

from qgis.core import QgsMapLayerType, QgsProject, QgsRasterLayer, QgsWkbTypes
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

STANDARD_CLASSES = {
    1: "Urban / built-up",
    2: "Cultivated agricultural land",
    3: "Grassland / pasture / meadow",
    4: "Shrub / transitional vegetation",
    5: "Forest",
    6: "Bare / sparsely vegetated / idle",
    7: "Water / wetlands",
}

DEFAULT_TRANSITIONS = [
    {"from": 2, "to": 4, "label": "Agriculture -> Shrub"},
    {"from": 4, "to": 5, "label": "Shrub -> Forest"},
    {"from": 5, "to": 4, "label": "Forest -> Shrub"},
    {"from": 3, "to": 4, "label": "Grassland -> Shrub"},
    {"from": 2, "to": 1, "label": "Agriculture -> Urban"},
]

DEFAULT_SCENARIOS = [
    {
        "name": "BAU",
        "description": "Baseline scenario.",
        "enabled": True,
        "multipliers": {
            "2 -> 4": 1.0,
            "4 -> 5": 1.0,
            "5 -> 4": 1.0,
            "3 -> 4": 1.0,
            "2 -> 1": 1.0,
        },
    },
    {
        "name": "Depopulation",
        "description": "Higher abandonment and slower urbanization.",
        "enabled": True,
        "multipliers": {
            "2 -> 4": 1.5,
            "4 -> 5": 1.2,
            "5 -> 4": 0.9,
            "3 -> 4": 1.4,
            "2 -> 1": 0.7,
        },
    },
    {
        "name": "Urban expansion",
        "description": "Accelerated urbanization pressure.",
        "enabled": True,
        "multipliers": {
            "2 -> 4": 0.7,
            "4 -> 5": 0.8,
            "5 -> 4": 1.0,
            "3 -> 4": 0.8,
            "2 -> 1": 2.0,
        },
    },
]


class Predictor2026Dialog(QDialog):
    def __init__(self, iface, plugin_dir):
        super().__init__(iface.mainWindow())
        self.iface = iface
        self.plugin_dir = Path(plugin_dir)
        self._about_shown = False
        self._loading_scenario_editor = False
        self._loading_scenario_list = False
        self._updating_preview = False
        self.scenario_data = []
        self.setWindowTitle("Predictor2026")
        self.setMinimumSize(980, 720)
        self.resize(1320, 920)
        self.setSizeGripEnabled(True)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._build_ui()
        self.refresh_layer_dropdowns(silent=True)
        self._load_default_transitions()
        self._load_default_scenarios()

    def _all_layers(self):
        return list(QgsProject.instance().mapLayers().values())

    def _raster_layers(self):
        return [lyr for lyr in self._all_layers() if lyr.type() == QgsMapLayerType.RasterLayer]

    def _polygon_layers(self):
        layers = []
        for lyr in self._all_layers():
            if lyr.type() != QgsMapLayerType.VectorLayer:
                continue
            try:
                if lyr.geometryType() == QgsWkbTypes.PolygonGeometry:
                    layers.append(lyr)
            except Exception:
                pass
        return layers

    def _layer_label(self, layer):
        if layer.type() == QgsMapLayerType.RasterLayer:
            layer_type = "Raster"
        elif layer.type() == QgsMapLayerType.VectorLayer:
            try:
                geom_type = layer.geometryType()
                if geom_type == QgsWkbTypes.PolygonGeometry:
                    layer_type = "Vector Polygon"
                elif geom_type == QgsWkbTypes.LineGeometry:
                    layer_type = "Vector Line"
                elif geom_type == QgsWkbTypes.PointGeometry:
                    layer_type = "Vector Point"
                else:
                    layer_type = "Vector"
            except Exception:
                layer_type = "Vector"
        else:
            layer_type = "Layer"
        return f"{layer.name()} ({layer_type})"

    def _make_layer_combo(self, layer_kind="raster"):
        combo = QComboBox()
        combo.setEditable(False)
        combo.addItem("-- Select layer --", "")
        self._populate_combo(combo, layer_kind)
        return combo

    def _populate_combo(self, combo, layer_kind="raster", selected_id=""):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("-- Select layer --", "")
        layers = self._raster_layers() if layer_kind == "raster" else self._polygon_layers()
        selected_index = 0
        for i, layer in enumerate(layers, start=1):
            combo.addItem(self._layer_label(layer), layer.id())
            if selected_id and layer.id() == selected_id:
                selected_index = i
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def _layer_by_id(self, layer_id):
        if not layer_id:
            return None
        return QgsProject.instance().mapLayer(layer_id)

    def _make_purpose_combo(self):
        combo = QComboBox()
        combo.addItems(["TRAIN", "VALIDATION"])
        return combo

    def _set_combo_by_text(self, combo, text):
        for i in range(combo.count()):
            if combo.itemText(i) == text:
                combo.setCurrentIndex(i)
                return

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        icon_label = QLabel()
        pix = QPixmap(str(self.plugin_dir / "icon.png")).scaled(
            86, 86, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        icon_label.setPixmap(pix)
        header.addWidget(icon_label)

        title = QLabel(
            "<h2 style='margin-bottom:2px;'>Predictor2026</h2>"
            "<div><b>Transparent transition-based LULC setup for QGIS</b></div>"
            "<div style='margin-top:4px; color:#555;'>"
            "Standard 7-class workflow, one or more historical periods, user-managed transition sets, "
            "static and dynamic predictor support, scenario management, and scenario-ready project export."
            "</div>"
            "<div style='margin-top:6px; color:#777;'><b>Author:</b> Nikola Djokic + Open AI</div>"
        )
        title.setWordWrap(True)
        header.addWidget(title, 1)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self._build_input_tab()
        self._build_reclass_tab()
        self._build_transitions_tab()
        self._build_scenarios_tab()
        self._build_export_tab()

        buttons = QHBoxLayout()
        self.refresh_layers_btn = QPushButton("Refresh QGIS layers")
        self.refresh_layers_btn.clicked.connect(self.refresh_layer_dropdowns)
        self.validate_btn = QPushButton("Validate configuration")
        self.validate_btn.clicked.connect(self.validate_configuration)
        self.export_btn = QPushButton("Export project JSON")
        self.export_btn.clicked.connect(self.export_config)
        self.run_btn = QPushButton("Run prediction in QGIS")
        self.run_btn.clicked.connect(self.run_prediction_in_qgis)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)

        buttons.addWidget(self.refresh_layers_btn)
        buttons.addWidget(self.validate_btn)
        buttons.addWidget(self.export_btn)
        buttons.addWidget(self.run_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)

    def _build_input_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        project_box = QGroupBox("Project setup")
        form = QFormLayout(project_box)
        self.project_name = QLineEdit("")
        self.project_name.setPlaceholderText("Enter project name")
        form.addRow("Project name:", self.project_name)
        layout.addWidget(project_box)

        boundary_box = QGroupBox("Boundary")
        boundary_layout = QHBoxLayout(boundary_box)
        self.boundary_combo = self._make_layer_combo("polygon")
        self.boundary_combo.currentIndexChanged.connect(self.update_transition_preview)
        boundary_layout.addWidget(QLabel("Boundary layer:"))
        boundary_layout.addWidget(self.boundary_combo, 1)
        layout.addWidget(boundary_box)

        period_box = QGroupBox("LULC periods (minimum: 1 pair)")
        period_layout = QVBoxLayout(period_box)
        period_note = QLabel(
            "Add one or more LULC periods. Mark each period as TRAIN or VALIDATION. "
            "Validation periods are reserved for hindcast testing and are excluded from model calibration."
        )
        period_note.setWordWrap(True)
        period_layout.addWidget(period_note)

        self.period_table = QTableWidget(0, 5)
        self.period_table.itemChanged.connect(lambda _item: self.update_transition_preview())
        self.period_table.setHorizontalHeaderLabels(
            ["Start year", "Start raster layer", "End year", "End raster layer", "Purpose"]
        )
        self.period_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.period_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.period_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.period_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.period_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.period_table.verticalHeader().setVisible(False)
        self.period_table.setAlternatingRowColors(True)
        period_layout.addWidget(self.period_table)

        period_buttons = QHBoxLayout()
        add_btn = QPushButton("Add period")
        add_btn.clicked.connect(self.add_period_row)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self.remove_selected_period)
        period_buttons.addWidget(add_btn)
        period_buttons.addWidget(remove_btn)
        period_buttons.addStretch(1)
        period_layout.addLayout(period_buttons)
        layout.addWidget(period_box, 1)

        static_box = QGroupBox("Static predictors")
        static_layout = QVBoxLayout(static_box)
        static_note = QLabel("Static predictors are applied to all LULC periods equally.")
        static_note.setWordWrap(True)
        static_layout.addWidget(static_note)

        self.static_table = QTableWidget(0, 2)
        self.static_table.setHorizontalHeaderLabels(["Predictor name", "Raster layer"])
        self.static_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.static_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.static_table.verticalHeader().setVisible(False)
        self.static_table.setAlternatingRowColors(True)
        static_layout.addWidget(self.static_table)

        static_buttons = QHBoxLayout()
        static_add_btn = QPushButton("Add static predictor")
        static_add_btn.clicked.connect(self.add_static_row)
        static_remove_btn = QPushButton("Remove selected")
        static_remove_btn.clicked.connect(self.remove_selected_static)
        static_buttons.addWidget(static_add_btn)
        static_buttons.addWidget(static_remove_btn)
        static_buttons.addStretch(1)
        static_layout.addLayout(static_buttons)
        layout.addWidget(static_box)

        dynamic_box = QGroupBox("Dynamic predictors")
        dynamic_layout = QVBoxLayout(dynamic_box)
        dynamic_note = QLabel(
            "Dynamic predictors change through time. For each dynamic predictor, assign a year and a raster layer. "
            "The plugin will map each LULC period to the predictor year matching the start year or, if unavailable, "
            "the nearest available year."
        )
        dynamic_note.setWordWrap(True)
        dynamic_layout.addWidget(dynamic_note)

        self.dynamic_table = QTableWidget(0, 3)
        self.dynamic_table.setHorizontalHeaderLabels(["Predictor name", "Year", "Raster layer"])
        self.dynamic_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.dynamic_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.dynamic_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.dynamic_table.verticalHeader().setVisible(False)
        self.dynamic_table.setAlternatingRowColors(True)
        dynamic_layout.addWidget(self.dynamic_table)

        dynamic_buttons = QHBoxLayout()
        dynamic_add_btn = QPushButton("Add dynamic predictor")
        dynamic_add_btn.clicked.connect(self.add_dynamic_row)
        dynamic_remove_btn = QPushButton("Remove selected")
        dynamic_remove_btn.clicked.connect(self.remove_selected_dynamic)
        dynamic_buttons.addWidget(dynamic_add_btn)
        dynamic_buttons.addWidget(dynamic_remove_btn)
        dynamic_buttons.addStretch(1)
        dynamic_layout.addLayout(dynamic_buttons)
        layout.addWidget(dynamic_box)

        self.tabs.addTab(tab, "1. Inputs")

    def _build_reclass_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        mode_box = QGroupBox("Reclassification mode")
        mode_form = QFormLayout(mode_box)
        self.reclass_mode = QComboBox()
        self.reclass_mode.addItems(
            [
                "Already reclassified (1-7)",
                "Reclassify CORINE to standard 7-class scheme",
            ]
        )
        self.align_checkbox = QCheckBox("Align all rasters to the first start LULC raster")
        self.align_checkbox.setChecked(True)
        mode_form.addRow("Mode:", self.reclass_mode)
        mode_form.addRow("", self.align_checkbox)
        layout.addWidget(mode_box)

        scheme_box = QGroupBox("Standard 7-class scheme")
        scheme_layout = QVBoxLayout(scheme_box)
        info = QTextEdit()
        info.setReadOnly(True)
        info.setHtml(
            "<ol>"
            "<li><b>Urban / built-up</b></li>"
            "<li><b>Cultivated agricultural land</b></li>"
            "<li><b>Grassland / pasture / meadow</b></li>"
            "<li><b>Shrub / transitional vegetation</b></li>"
            "<li><b>Forest</b></li>"
            "<li><b>Bare / sparsely vegetated / idle</b></li>"
            "<li><b>Water / wetlands</b></li>"
            "</ol>"
            "<p>This workflow keeps the fixed 1-7 class scheme to improve reproducibility and reduce user burden.</p>"
        )
        scheme_layout.addWidget(info)
        layout.addWidget(scheme_box)
        self.tabs.addTab(tab, "2. Reclassification")

    def _build_transitions_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box = QGroupBox("Transition set")
        box_layout = QVBoxLayout(box)
        desc = QLabel(
            "You can now add, edit, or remove transitions. Each transition is defined by a From class, a To class, "
            "and an optional label. All scenarios will automatically sync to the current transition list."
        )
        desc.setWordWrap(True)
        box_layout.addWidget(desc)

        self.transition_table = QTableWidget(0, 4)
        self.transition_table.setHorizontalHeaderLabels(["Use", "From class", "To class", "Label"])
        self.transition_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.transition_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.transition_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.transition_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.transition_table.verticalHeader().setVisible(False)
        self.transition_table.setAlternatingRowColors(True)
        self.transition_table.itemChanged.connect(self._on_transition_table_changed)
        box_layout.addWidget(self.transition_table)

        buttons = QHBoxLayout()
        add_btn = QPushButton("Add transition")
        add_btn.clicked.connect(self.add_transition_row)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self.remove_selected_transition)
        buttons.addWidget(add_btn)
        buttons.addWidget(remove_btn)
        buttons.addStretch(1)
        box_layout.addLayout(buttons)
        layout.addWidget(box)

        notes_box = QGroupBox("Class reference")
        notes_layout = QVBoxLayout(notes_box)
        notes = QTextEdit()
        notes.setReadOnly(True)
        notes.setPlainText(
            "Standard classes:\n\n" + "\n".join(f"{k} = {v}" for k, v in STANDARD_CLASSES.items())
        )
        notes_layout.addWidget(notes)
        layout.addWidget(notes_box)
        self.tabs.addTab(tab, "3. Transitions")

    def _build_scenarios_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        info = QLabel(
            "Manage multiple scenarios, enable or disable them with checkboxes, review the fixed default scenarios, "
            "and edit multipliers for any custom scenario you add."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        selector_box = QGroupBox("Scenarios")
        selector_layout = QVBoxLayout(selector_box)
        self.scenario_list = QListWidget()
        self.scenario_list.currentRowChanged.connect(self._on_scenario_selection_changed)
        self.scenario_list.itemChanged.connect(self._on_scenario_list_item_changed)
        self.scenario_list.setMaximumHeight(150)
        selector_layout.addWidget(self.scenario_list)

        scenario_buttons = QHBoxLayout()
        add_scenario_btn = QPushButton("Add scenario")
        add_scenario_btn.clicked.connect(self.add_scenario)
        remove_scenario_btn = QPushButton("Remove selected")
        remove_scenario_btn.clicked.connect(self.remove_selected_scenario)
        scenario_buttons.addWidget(add_scenario_btn)
        scenario_buttons.addWidget(remove_scenario_btn)
        selector_layout.addLayout(scenario_buttons)
        left_layout.addWidget(selector_box, 0)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.fixed_scenario_groups = {}
        self.fixed_scenario_tables = {}
        self.fixed_scenario_descriptions = {}
        for scenario_name in ["BAU", "Depopulation", "Urban expansion"]:
            box = QGroupBox(scenario_name)
            box_layout = QVBoxLayout(box)
            desc = QLabel("")
            desc.setWordWrap(True)
            box_layout.addWidget(desc)
            table = QTableWidget(0, 2)
            table.setHorizontalHeaderLabels(["Transition", "Multiplier"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            table.verticalHeader().setVisible(False)
            table.setAlternatingRowColors(True)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            box_layout.addWidget(table)
            right_layout.addWidget(box)
            self.fixed_scenario_groups[scenario_name] = box
            self.fixed_scenario_tables[scenario_name] = table
            self.fixed_scenario_descriptions[scenario_name] = desc

        self.custom_scenario_box = QGroupBox("Custom scenario")
        custom_layout = QVBoxLayout(self.custom_scenario_box)
        custom_hint = QLabel("This block becomes active when you add a new scenario and select it from the list on the left.")
        custom_hint.setWordWrap(True)
        custom_layout.addWidget(custom_hint)
        custom_form = QFormLayout()
        self.scenario_name_edit = QLineEdit()
        self.scenario_name_edit.textEdited.connect(self._scenario_meta_changed)
        self.scenario_description_edit = QLineEdit()
        self.scenario_description_edit.textEdited.connect(self._scenario_meta_changed)
        custom_form.addRow("Scenario name:", self.scenario_name_edit)
        custom_form.addRow("Description:", self.scenario_description_edit)
        custom_layout.addLayout(custom_form)
        self.scenario_transition_table = QTableWidget(0, 2)
        self.scenario_transition_table.setHorizontalHeaderLabels(["Transition", "Multiplier"])
        self.scenario_transition_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.scenario_transition_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.scenario_transition_table.verticalHeader().setVisible(False)
        self.scenario_transition_table.setAlternatingRowColors(True)
        self.scenario_transition_table.itemChanged.connect(self._on_scenario_transition_changed)
        custom_layout.addWidget(self.scenario_transition_table)
        right_layout.addWidget(self.custom_scenario_box)

        target_box = QGroupBox("Target years")
        target_layout = QVBoxLayout(target_box)
        target_note = QLabel("Leave the list blank initially and add only the years you want to predict.")
        target_note.setWordWrap(True)
        target_layout.addWidget(target_note)
        self.target_year_table = QTableWidget(0, 1)
        self.target_year_table.setHorizontalHeaderLabels(["Year"])
        self.target_year_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.target_year_table.verticalHeader().setVisible(False)
        self.target_year_table.setAlternatingRowColors(True)
        self.target_year_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        target_layout.addWidget(self.target_year_table)
        target_buttons = QHBoxLayout()
        add_year_btn = QPushButton("Add year")
        add_year_btn.clicked.connect(lambda _checked=False: self.add_target_year_row())
        remove_year_btn = QPushButton("Remove selected")
        remove_year_btn.clicked.connect(self.remove_selected_target_year)
        target_buttons.addWidget(add_year_btn)
        target_buttons.addWidget(remove_year_btn)
        target_buttons.addStretch(1)
        target_layout.addLayout(target_buttons)
        right_layout.addWidget(target_box)

        preview_box = QGroupBox("Transition matrix preview")
        preview_layout = QVBoxLayout(preview_box)
        preview_top = QHBoxLayout()
        preview_top.addWidget(QLabel("Preview scenario:"))
        self.preview_scenario_combo = QComboBox()
        self.preview_scenario_combo.currentIndexChanged.connect(self.update_transition_preview)
        preview_top.addWidget(self.preview_scenario_combo, 1)
        self.preview_status_label = QLabel("")
        self.preview_status_label.setWordWrap(True)
        preview_layout.addLayout(preview_top)
        preview_layout.addWidget(self.preview_status_label)
        self.preview_table = QTableWidget(7, 7)
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_table.verticalHeader().setDefaultSectionSize(26)
        self.preview_table.horizontalHeader().setDefaultSectionSize(72)
        self.preview_table.setMinimumHeight(240)
        self.preview_table.setWordWrap(False)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setHorizontalHeaderLabels([str(i) for i in range(1, 8)])
        self.preview_table.setVerticalHeaderLabels([str(i) for i in range(1, 8)])
        preview_layout.addWidget(self.preview_table)
        left_layout.addWidget(preview_box, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([340, 980])

        self.tabs.addTab(tab, "4. Scenarios")

    def _build_export_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        model_box = QGroupBox("Spatial allocation controls")
        model_form = QFormLayout(model_box)
        self.neighborhood_enabled = QCheckBox("Use neighborhood effect during spatial allocation")
        self.neighborhood_enabled.setChecked(True)
        self.neighborhood_strength = QComboBox()
        self.neighborhood_strength.addItems(["Low", "Medium", "High"])
        self.neighborhood_strength.setCurrentText("Medium")
        strength_hint = QLabel(
            "Low = weaker neighborhood influence, Medium = balanced behavior, High = stronger compact growth."
        )
        strength_hint.setWordWrap(True)
        model_form.addRow(self.neighborhood_enabled)
        model_form.addRow("Neighborhood strength:", self.neighborhood_strength)
        model_form.addRow("Guide:", strength_hint)
        layout.addWidget(model_box)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlainText(
            "Workflow summary will appear here after validation.\n\n"
            "This version exports a structured JSON project file with editable transitions, scenario definitions, custom target years, and stable run settings."
        )
        layout.addWidget(self.summary, 1)
        self.tabs.addTab(tab, "5. Export")

    def add_period_row(self, defaults=None):
        defaults = defaults or ("", "", "", "", "TRAIN")
        row = self.period_table.rowCount()
        self.period_table.insertRow(row)
        self.period_table.setRowHeight(row, 28)
        self.period_table.setItem(row, 0, QTableWidgetItem(str(defaults[0])))
        start_combo = self._make_layer_combo("raster")
        start_combo.currentIndexChanged.connect(self.update_transition_preview)
        self.period_table.setCellWidget(row, 1, start_combo)
        self.period_table.setItem(row, 2, QTableWidgetItem(str(defaults[2])))
        end_combo = self._make_layer_combo("raster")
        end_combo.currentIndexChanged.connect(self.update_transition_preview)
        self.period_table.setCellWidget(row, 3, end_combo)
        purpose_combo = self._make_purpose_combo()
        purpose_combo.currentIndexChanged.connect(self.update_transition_preview)
        self.period_table.setCellWidget(row, 4, purpose_combo)
        if defaults[1]:
            self._populate_combo(start_combo, "raster", defaults[1])
        if defaults[3]:
            self._populate_combo(end_combo, "raster", defaults[3])
        if defaults[4]:
            self._set_combo_by_text(purpose_combo, defaults[4])

    def remove_selected_period(self):
        rows = sorted({index.row() for index in self.period_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.period_table.removeRow(row)
        self.update_transition_preview()

    def add_static_row(self, defaults=None):
        defaults = defaults or ("", "")
        row = self.static_table.rowCount()
        self.static_table.insertRow(row)
        self.static_table.setRowHeight(row, 28)
        self.static_table.setItem(row, 0, QTableWidgetItem(str(defaults[0])))
        combo = self._make_layer_combo("raster")
        self.static_table.setCellWidget(row, 1, combo)
        if defaults[1]:
            self._populate_combo(combo, "raster", defaults[1])

    def remove_selected_static(self):
        rows = sorted({index.row() for index in self.static_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.static_table.removeRow(row)

    def add_dynamic_row(self, defaults=None):
        defaults = defaults or ("", "", "")
        row = self.dynamic_table.rowCount()
        self.dynamic_table.insertRow(row)
        self.dynamic_table.setRowHeight(row, 28)
        self.dynamic_table.setItem(row, 0, QTableWidgetItem(str(defaults[0])))
        self.dynamic_table.setItem(row, 1, QTableWidgetItem(str(defaults[1])))
        combo = self._make_layer_combo("raster")
        self.dynamic_table.setCellWidget(row, 2, combo)
        if defaults[2]:
            self._populate_combo(combo, "raster", defaults[2])

    def remove_selected_dynamic(self):
        rows = sorted({index.row() for index in self.dynamic_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.dynamic_table.removeRow(row)

    def add_transition_row(self, defaults=None):
        defaults = defaults or {"enabled": True, "from": "", "to": "", "label": ""}
        row = self.transition_table.rowCount()
        self.transition_table.blockSignals(True)
        self.transition_table.insertRow(row)
        enabled_item = QTableWidgetItem()
        enabled_item.setFlags(enabled_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        enabled_item.setCheckState(Qt.Checked if defaults.get("enabled", True) else Qt.Unchecked)
        self.transition_table.setItem(row, 0, enabled_item)
        self.transition_table.setItem(row, 1, QTableWidgetItem(str(defaults.get("from", ""))))
        self.transition_table.setItem(row, 2, QTableWidgetItem(str(defaults.get("to", ""))))
        self.transition_table.setItem(row, 3, QTableWidgetItem(str(defaults.get("label", ""))))
        self.transition_table.blockSignals(False)
        self._sync_scenarios_with_transitions()

    def remove_selected_transition(self):
        rows = sorted({index.row() for index in self.transition_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.transition_table.removeRow(row)
        self._sync_scenarios_with_transitions()

    def add_target_year_row(self, year=None):
        if isinstance(year, bool):
            year = None
        if year is None:
            value, ok = QInputDialog.getInt(
                self,
                "Add target year",
                "Enter target year:",
                2030,
                1900,
                3000,
                1,
            )
            if not ok:
                return
            year = value
        row = self.target_year_table.rowCount()
        self.target_year_table.insertRow(row)
        item = QTableWidgetItem(str(year))
        self.target_year_table.setItem(row, 0, item)

    def remove_selected_target_year(self):
        rows = sorted({index.row() for index in self.target_year_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.target_year_table.removeRow(row)

    def _load_default_transitions(self):
        for transition in DEFAULT_TRANSITIONS:
            self.add_transition_row({
                "enabled": True,
                "from": transition["from"],
                "to": transition["to"],
                "label": transition["label"],
            })

    def _load_default_scenarios(self):
        self.scenario_data = []
        transition_codes = self._transition_codes(active_only=False)
        for scenario in DEFAULT_SCENARIOS:
            multipliers = {code: float(scenario["multipliers"].get(code, 1.0)) for code in transition_codes}
            self.scenario_data.append({
                "name": scenario["name"],
                "description": scenario["description"],
                "enabled": bool(scenario.get("enabled", True)),
                "multipliers": multipliers,
            })
        self._rebuild_scenario_list()
        if self.scenario_data:
            self.scenario_list.setCurrentRow(0)
        self._rebuild_preview_combo()
        self._refresh_fixed_scenario_panels()
        self._refresh_custom_scenario_panel(self.scenario_list.currentRow())
        self.update_transition_preview()

    def _default_scenario_names(self):
        return [scenario["name"] for scenario in DEFAULT_SCENARIOS]

    def _find_scenario_by_name(self, name):
        for scenario in self.scenario_data:
            if scenario.get("name") == name:
                return scenario
        return None

    def _is_custom_scenario(self, scenario):
        return scenario.get("name") not in self._default_scenario_names()

    def _fill_scenario_table(self, table, scenario):
        transitions = self.collect_transitions(active_only=False)
        table.blockSignals(True)
        table.setRowCount(0)
        for transition in transitions:
            code = transition.get("code", "")
            if not code:
                continue
            r = table.rowCount()
            table.insertRow(r)
            transition_label = code
            if transition.get("label"):
                transition_label = f"{code} | {transition['label']}"
            label_item = QTableWidgetItem(transition_label)
            label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(r, 0, label_item)
            value = 1.0
            if scenario is not None:
                value = scenario.get("multipliers", {}).get(code, 1.0)
            table.setItem(r, 1, QTableWidgetItem(str(value)))
        table.blockSignals(False)

    def _refresh_fixed_scenario_panels(self):
        for scenario_name in self._default_scenario_names():
            scenario = self._find_scenario_by_name(scenario_name)
            box = self.fixed_scenario_groups.get(scenario_name)
            table = self.fixed_scenario_tables.get(scenario_name)
            desc = self.fixed_scenario_descriptions.get(scenario_name)
            if box is None or table is None or desc is None:
                continue
            if scenario is None:
                box.setVisible(False)
                continue
            box.setVisible(True)
            box.setEnabled(bool(scenario.get("enabled", True)))
            desc.setText(scenario.get("description", ""))
            self._fill_scenario_table(table, scenario)

    def _refresh_custom_scenario_panel(self, selected_row=None):
        if selected_row is None:
            selected_row = self.scenario_list.currentRow()
        scenario = None
        if 0 <= selected_row < len(self.scenario_data):
            candidate = self.scenario_data[selected_row]
            if self._is_custom_scenario(candidate):
                scenario = candidate
        has_custom = any(self._is_custom_scenario(s) for s in self.scenario_data)
        self.custom_scenario_box.setVisible(has_custom)
        self.custom_scenario_box.setEnabled(scenario is not None and bool(scenario.get("enabled", True)))
        self._loading_scenario_editor = True
        if scenario is None:
            self.scenario_name_edit.clear()
            self.scenario_description_edit.clear()
            self.scenario_transition_table.setRowCount(0)
        else:
            self.scenario_name_edit.setText(scenario.get("name", ""))
            self.scenario_description_edit.setText(scenario.get("description", ""))
            self._fill_scenario_table(self.scenario_transition_table, scenario)
        self._loading_scenario_editor = False

    def _transition_row_data(self, row):
        enabled_item = self.transition_table.item(row, 0)
        from_item = self.transition_table.item(row, 1)
        to_item = self.transition_table.item(row, 2)
        label_item = self.transition_table.item(row, 3)
        enabled = enabled_item.checkState() == Qt.Checked if enabled_item else True
        from_raw = from_item.text().strip() if from_item else ""
        to_raw = to_item.text().strip() if to_item else ""
        label = label_item.text().strip() if label_item else ""
        return {
            "enabled": enabled,
            "from": from_raw,
            "to": to_raw,
            "label": label,
            "code": f"{from_raw} -> {to_raw}" if from_raw and to_raw else "",
        }

    def collect_transitions(self, active_only=False):
        transitions = []
        for row in range(self.transition_table.rowCount()):
            item = self._transition_row_data(row)
            if active_only and not item["enabled"]:
                continue
            if any([item["from"], item["to"], item["label"]]):
                transitions.append(item)
        return transitions

    def _transition_codes(self, active_only=False):
        codes = []
        for item in self.collect_transitions(active_only=active_only):
            if item["code"]:
                codes.append(item["code"])
        return codes

    def _scenario_display_name(self, scenario):
        prefix = "✓" if scenario.get("enabled", True) else "✗"
        return f"{prefix} {scenario.get('name', 'Scenario')}"

    def _rebuild_scenario_list(self):
        current_row = self.scenario_list.currentRow()
        self._loading_scenario_list = True
        self.scenario_list.clear()
        for scenario in self.scenario_data:
            item = QListWidgetItem(self._scenario_display_name(scenario))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if scenario.get("enabled", True) else Qt.Unchecked)
            self.scenario_list.addItem(item)
        self._loading_scenario_list = False
        if self.scenario_data:
            if current_row < 0 or current_row >= len(self.scenario_data):
                current_row = 0
            self.scenario_list.setCurrentRow(current_row)
        else:
            self._clear_scenario_editor()

    def _rebuild_preview_combo(self):
        current_name = self.preview_scenario_combo.currentText()
        self.preview_scenario_combo.blockSignals(True)
        self.preview_scenario_combo.clear()
        for scenario in self.scenario_data:
            self.preview_scenario_combo.addItem(scenario.get("name", "Scenario"))
        if current_name:
            idx = self.preview_scenario_combo.findText(current_name)
            if idx >= 0:
                self.preview_scenario_combo.setCurrentIndex(idx)
        if self.preview_scenario_combo.count() and self.preview_scenario_combo.currentIndex() < 0:
            self.preview_scenario_combo.setCurrentIndex(0)
        self.preview_scenario_combo.blockSignals(False)

    def add_scenario(self):
        custom_count = sum(1 for s in self.scenario_data if self._is_custom_scenario(s)) + 1
        base_name = f"Custom scenario {custom_count}"
        transition_codes = self._transition_codes(active_only=False)
        self.scenario_data.append({
            "name": base_name,
            "description": "",
            "enabled": True,
            "multipliers": {code: 1.0 for code in transition_codes},
        })
        self._rebuild_scenario_list()
        self._rebuild_preview_combo()
        self.scenario_list.setCurrentRow(len(self.scenario_data) - 1)
        self.update_transition_preview()

    def remove_selected_scenario(self):
        row = self.scenario_list.currentRow()
        if row < 0:
            return
        scenario = self.scenario_data[row]
        if not self._is_custom_scenario(scenario):
            QMessageBox.information(self, "Predictor2026", "Default scenarios cannot be removed. You can only enable or disable them.")
            return
        del self.scenario_data[row]
        self._rebuild_scenario_list()
        self._rebuild_preview_combo()
        self._refresh_fixed_scenario_panels()
        self._refresh_custom_scenario_panel()
        self.update_transition_preview()

    def _clear_scenario_editor(self):
        self._loading_scenario_editor = True
        self.scenario_name_edit.clear()
        self.scenario_description_edit.clear()
        self.scenario_transition_table.setRowCount(0)
        self._loading_scenario_editor = False

    def _load_scenario_into_editor(self, row):
        self._refresh_fixed_scenario_panels()
        self._refresh_custom_scenario_panel(row)

    def _save_current_scenario_editor(self):
        row = self.scenario_list.currentRow()
        if self._loading_scenario_editor or row < 0 or row >= len(self.scenario_data):
            return
        scenario = self.scenario_data[row]
        if not self._is_custom_scenario(scenario):
            return
        scenario["name"] = self.scenario_name_edit.text().strip() or scenario.get("name") or f"Custom scenario {row + 1}"
        scenario["description"] = self.scenario_description_edit.text().strip()
        multipliers = scenario.setdefault("multipliers", {})
        transitions = self.collect_transitions(active_only=False)
        for r, transition in enumerate(transitions):
            code = transition.get("code", "")
            if not code or r >= self.scenario_transition_table.rowCount():
                continue
            value_item = self.scenario_transition_table.item(r, 1)
            raw_value = value_item.text().strip() if value_item else "1.0"
            try:
                multipliers[code] = float(raw_value)
            except Exception:
                multipliers[code] = 1.0
        valid_codes = {transition.get("code", "") for transition in transitions if transition.get("code")}
        for code in list(multipliers.keys()):
            if code not in valid_codes:
                del multipliers[code]
        if 0 <= row < self.scenario_list.count():
            self.scenario_list.item(row).setText(self._scenario_display_name(scenario))
        self._rebuild_preview_combo()
        self._refresh_fixed_scenario_panels()
        self._refresh_custom_scenario_panel(row)

    def _sync_scenarios_with_transitions(self):
        transitions = self.collect_transitions(active_only=False)
        valid_codes = [transition["code"] for transition in transitions if transition["code"]]
        for scenario in self.scenario_data:
            multipliers = scenario.setdefault("multipliers", {})
            for code in valid_codes:
                if code not in multipliers:
                    multipliers[code] = 1.0
            for code in list(multipliers.keys()):
                if code not in valid_codes:
                    del multipliers[code]
        current_row = self.scenario_list.currentRow()
        self._rebuild_scenario_list()
        if self.scenario_data and current_row >= 0:
            self.scenario_list.setCurrentRow(min(current_row, len(self.scenario_data) - 1))
        self._rebuild_preview_combo()
        self.update_transition_preview()

    def _on_transition_table_changed(self, _item):
        self._sync_scenarios_with_transitions()

    def _on_scenario_selection_changed(self, row):
        self._save_current_scenario_editor()
        self._load_scenario_into_editor(row)
        self.update_transition_preview()

    def _on_scenario_list_item_changed(self, item):
        if self._loading_scenario_list:
            return
        row = self.scenario_list.row(item)
        if 0 <= row < len(self.scenario_data):
            self.scenario_data[row]["enabled"] = item.checkState() == Qt.Checked
            item.setText(self._scenario_display_name(self.scenario_data[row]))
            self._rebuild_preview_combo()
            self._refresh_fixed_scenario_panels()
            self._refresh_custom_scenario_panel(row)
            self.update_transition_preview()

    def _scenario_meta_changed(self):
        self._save_current_scenario_editor()

    def _on_scenario_transition_changed(self, item):
        if self._loading_scenario_editor or item.column() != 1:
            return
        self._save_current_scenario_editor()
        self.update_transition_preview()

    def _compute_base_transition_matrix(self):
        config = self.build_config()
        periods = [p for p in config.get("lulc_periods", []) if p.get("purpose") == "TRAIN"]
        if not periods:
            return None, "Add at least one TRAIN period to estimate transition probabilities."

        boundary_source = (config.get("boundary") or {}).get("layer_source", "")
        if not boundary_source:
            return None, "Select a boundary layer to estimate the transition matrix."

        matrix = np.zeros((7, 7), dtype=np.float64)
        exposures = np.zeros(7, dtype=np.float64)
        used_periods = 0

        try:
            ref_ds = None
            boundary = None
            for period in periods:
                start_path = period.get("start_layer_source", "")
                end_path = period.get("end_layer_source", "")
                if not start_path or not end_path:
                    continue
                ds0, a0, n0 = _open_array(start_path)
                ds1, a1, n1 = _open_array(end_path)
                if ref_ds is None:
                    ref_ds = ds0
                    boundary = _boundary_mask(boundary_source, ref_ds)
                if not _same_grid(ds0, ref_ds):
                    ds0, a0, n0 = _warp_to_reference(start_path, ref_ds)
                if not _same_grid(ds1, ref_ds):
                    ds1, a1, n1 = _warp_to_reference(end_path, ref_ds, resample_alg=0)
                if a0.shape != a1.shape:
                    continue
                valid = _valid_mask(a0, n0, boundary)
                if n1 is not None:
                    valid &= a1 != n1
                valid &= np.isin(a0, np.arange(1, 8)) & np.isin(a1, np.arange(1, 8))
                if not np.any(valid):
                    continue
                used_periods += 1
                try:
                    years = max(1, int(period.get("end_year", 0)) - int(period.get("start_year", 0)))
                except Exception:
                    years = 1
                start_vals = a0[valid].astype(np.int16)
                end_vals = a1[valid].astype(np.int16)
                for from_cls in range(1, 8):
                    class_mask = start_vals == from_cls
                    class_count = int(np.count_nonzero(class_mask))
                    if class_count == 0:
                        continue
                    exposures[from_cls - 1] += float(class_count) * float(years)
                    targets = end_vals[class_mask]
                    for to_cls in range(1, 8):
                        change_count = int(np.count_nonzero(targets == to_cls))
                        matrix[from_cls - 1, to_cls - 1] += float(change_count)
            if used_periods == 0:
                return None, "No valid TRAIN rasters were available for matrix estimation."

            annual = np.zeros((7, 7), dtype=np.float64)
            for from_idx in range(7):
                denom = exposures[from_idx]
                if denom > 0:
                    annual[from_idx, :] = matrix[from_idx, :] / denom
                    off_sum = annual[from_idx, :].sum() - annual[from_idx, from_idx]
                    annual[from_idx, from_idx] = max(0.0, 1.0 - off_sum)
            return annual, f"Estimated annual transition probabilities from {used_periods} TRAIN period(s)."
        except Exception as exc:
            return None, f"Preview fallback: could not estimate probabilities ({exc})."

    def _apply_scenario_to_matrix(self, base_matrix, scenario):
        adjusted = np.array(base_matrix, copy=True)
        multipliers = scenario.get("multipliers", {})
        transitions = self.collect_transitions(active_only=True)
        for transition in transitions:
            code = transition.get("code", "")
            if not code:
                continue
            try:
                from_cls = int(transition.get("from"))
                to_cls = int(transition.get("to"))
            except Exception:
                continue
            if not (1 <= from_cls <= 7 and 1 <= to_cls <= 7):
                continue
            mult = float(multipliers.get(code, 1.0))
            adjusted[from_cls - 1, to_cls - 1] = base_matrix[from_cls - 1, to_cls - 1] * mult

        for row in range(7):
            off_sum = adjusted[row, :].sum() - adjusted[row, row]
            if off_sum > 1.0:
                scale = 1.0 / off_sum
                for col in range(7):
                    if col != row:
                        adjusted[row, col] *= scale
                off_sum = adjusted[row, :].sum() - adjusted[row, row]
            adjusted[row, row] = max(0.0, 1.0 - off_sum)
        return adjusted

    def update_transition_preview(self):
        if self._updating_preview:
            return
        self._updating_preview = True
        try:
            for r in range(7):
                for c in range(7):
                    preview_item = QTableWidgetItem("-")
                    preview_item.setTextAlignment(Qt.AlignCenter)
                    self.preview_table.setItem(r, c, preview_item)

            idx = self.preview_scenario_combo.currentIndex()
            if idx < 0 or idx >= len(self.scenario_data):
                self.preview_status_label.setText("Select a scenario to preview.")
                return

            self._save_current_scenario_editor()
            scenario = self.scenario_data[idx]
            base_matrix, status = self._compute_base_transition_matrix()

            if base_matrix is None:
                self.preview_status_label.setText(status)
                multipliers = scenario.get("multipliers", {})
                transitions = self.collect_transitions(active_only=True)
                for transition in transitions:
                    code = transition.get("code", "")
                    if not code:
                        continue
                    try:
                        from_cls = int(transition.get("from"))
                        to_cls = int(transition.get("to"))
                    except Exception:
                        continue
                    value = float(multipliers.get(code, 1.0))
                    item = QTableWidgetItem(f"m={value:.3f}")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.preview_table.setItem(from_cls - 1, to_cls - 1, item)
                return

            adjusted = self._apply_scenario_to_matrix(base_matrix, scenario)
            self.preview_status_label.setText(status + " Showing scenario-adjusted annual probabilities.")
            for r in range(7):
                for c in range(7):
                    value = adjusted[r, c]
                    text = f"{value:.4f}" if value > 0 else "-"
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    base_value = base_matrix[r, c]
                    item.setToolTip(
                        f"From {r + 1} to {c + 1}\n"
                        f"Base annual probability: {base_value:.6f}\n"
                        f"Scenario-adjusted probability: {value:.6f}"
                    )
                    self.preview_table.setItem(r, c, item)
        finally:
            self._updating_preview = False

    def refresh_layer_dropdowns(self, silent=False):
        self._populate_combo(self.boundary_combo, "polygon", self.boundary_combo.currentData())
        for row in range(self.period_table.rowCount()):
            start_combo = self.period_table.cellWidget(row, 1)
            end_combo = self.period_table.cellWidget(row, 3)
            if start_combo is not None:
                self._populate_combo(start_combo, "raster", start_combo.currentData())
            if end_combo is not None:
                self._populate_combo(end_combo, "raster", end_combo.currentData())
        for row in range(self.static_table.rowCount()):
            combo = self.static_table.cellWidget(row, 1)
            if combo is not None:
                self._populate_combo(combo, "raster", combo.currentData())
        for row in range(self.dynamic_table.rowCount()):
            combo = self.dynamic_table.cellWidget(row, 2)
            if combo is not None:
                self._populate_combo(combo, "raster", combo.currentData())
        if not silent:
            QMessageBox.information(self, "Predictor2026", "Layer lists refreshed from the current QGIS project.")

    def show_about_once(self):
        if self._about_shown:
            return
        self._about_shown = True
        QMessageBox.information(
            self,
            "About Predictor2026",
            "Predictor2026 is a transparent LULC modeling plugin prototype.\n\n"
            "How it works:\n"
            "• standard 7-class scheme\n"
            "• one or more historical LULC periods\n"
            "• TRAIN / VALIDATION period support\n"
            "• separate static and dynamic predictors\n"
            "• editable transition set\n"
            "• scenario enable/disable and per-transition multipliers\n"
            "• custom target years and transition-matrix preview\n"
            "• layer selection directly from the current QGIS project\n\n"
            "Author: Nikola Djokic + Open AI"
        )

    def collect_periods(self):
        periods = []
        for row in range(self.period_table.rowCount()):
            start_year_item = self.period_table.item(row, 0)
            end_year_item = self.period_table.item(row, 2)
            start_combo = self.period_table.cellWidget(row, 1)
            end_combo = self.period_table.cellWidget(row, 3)
            purpose_combo = self.period_table.cellWidget(row, 4)
            start_year = start_year_item.text().strip() if start_year_item else ""
            end_year = end_year_item.text().strip() if end_year_item else ""
            start_layer_id = start_combo.currentData() if start_combo else ""
            end_layer_id = end_combo.currentData() if end_combo else ""
            purpose = purpose_combo.currentText().strip() if purpose_combo else "TRAIN"
            if any([start_year, end_year, start_layer_id, end_layer_id]):
                start_layer = self._layer_by_id(start_layer_id)
                end_layer = self._layer_by_id(end_layer_id)
                periods.append({
                    "start_year": start_year,
                    "start_layer_id": start_layer_id or "",
                    "start_layer_name": start_layer.name() if start_layer else "",
                    "start_layer_source": start_layer.source() if start_layer else "",
                    "end_year": end_year,
                    "end_layer_id": end_layer_id or "",
                    "end_layer_name": end_layer.name() if end_layer else "",
                    "end_layer_source": end_layer.source() if end_layer else "",
                    "purpose": purpose,
                })
        return periods

    def collect_static_predictors(self):
        predictors = []
        for row in range(self.static_table.rowCount()):
            name_item = self.static_table.item(row, 0)
            combo = self.static_table.cellWidget(row, 1)
            name = name_item.text().strip() if name_item else ""
            layer_id = combo.currentData() if combo else ""
            layer = self._layer_by_id(layer_id)
            if name or layer_id:
                predictors.append({
                    "name": name,
                    "layer_id": layer_id or "",
                    "layer_name": layer.name() if layer else "",
                    "layer_source": layer.source() if layer else "",
                    "type": "static",
                })
        return predictors

    def collect_dynamic_predictors(self):
        predictors = []
        for row in range(self.dynamic_table.rowCount()):
            name_item = self.dynamic_table.item(row, 0)
            year_item = self.dynamic_table.item(row, 1)
            combo = self.dynamic_table.cellWidget(row, 2)
            name = name_item.text().strip() if name_item else ""
            year = year_item.text().strip() if year_item else ""
            layer_id = combo.currentData() if combo else ""
            layer = self._layer_by_id(layer_id)
            if name or year or layer_id:
                predictors.append({
                    "name": name,
                    "year": year,
                    "layer_id": layer_id or "",
                    "layer_name": layer.name() if layer else "",
                    "layer_source": layer.source() if layer else "",
                    "type": "dynamic",
                })
        return predictors

    def collect_scenarios(self):
        self._save_current_scenario_editor()
        scenarios = []
        active_codes = set(self._transition_codes(active_only=True))
        for scenario in self.scenario_data:
            multipliers = {}
            for code, value in scenario.get("multipliers", {}).items():
                if code in active_codes:
                    try:
                        multipliers[code] = float(value)
                    except Exception:
                        multipliers[code] = 1.0
            scenarios.append({
                "name": scenario.get("name", "").strip(),
                "description": scenario.get("description", "").strip(),
                "enabled": bool(scenario.get("enabled", True)),
                "transition_multipliers": multipliers,
            })
        return scenarios

    def collect_target_years(self):
        years = []
        seen = set()
        for row in range(self.target_year_table.rowCount()):
            item = self.target_year_table.item(row, 0)
            text = item.text().strip() if item else ""
            if not text:
                continue
            try:
                year = int(text)
            except Exception:
                continue
            if year not in seen:
                seen.add(year)
                years.append(year)
        return years

    def _dynamic_mapping(self, periods, dynamic_predictors):
        mapping = []
        grouped = {}
        for pred in dynamic_predictors:
            name = pred.get("name", "").strip()
            if not name:
                continue
            grouped.setdefault(name, []).append(pred)

        for period in periods:
            try:
                start_year_int = int(period.get("start_year", ""))
            except Exception:
                start_year_int = None
            period_key = f"{period.get('start_year','?')}->{period.get('end_year','?')} ({period.get('purpose','')})"
            for pred_name, rows in grouped.items():
                chosen = None
                if start_year_int is not None:
                    exact = [r for r in rows if str(r.get("year", "")).strip().isdigit() and int(str(r.get("year")).strip()) == start_year_int]
                    if exact:
                        chosen = exact[0]
                    else:
                        numeric_rows = [r for r in rows if str(r.get("year", "")).strip().isdigit()]
                        if numeric_rows:
                            chosen = min(numeric_rows, key=lambda r: abs(int(str(r.get("year")).strip()) - start_year_int))
                if chosen is None and rows:
                    chosen = rows[0]
                mapping.append({
                    "period": period_key,
                    "predictor_name": pred_name,
                    "requested_start_year": period.get("start_year", ""),
                    "selected_predictor_year": chosen.get("year", "") if chosen else "",
                    "selected_layer_name": chosen.get("layer_name", "") if chosen else "",
                    "selected_layer_source": chosen.get("layer_source", "") if chosen else "",
                })
        return mapping

    def build_config(self):
        boundary_id = self.boundary_combo.currentData()
        boundary_layer = self._layer_by_id(boundary_id)
        periods = self.collect_periods()
        static_predictors = self.collect_static_predictors()
        dynamic_predictors = self.collect_dynamic_predictors()
        dynamic_mapping = self._dynamic_mapping(periods, dynamic_predictors)
        transitions = self.collect_transitions(active_only=True)
        scenarios = self.collect_scenarios()
        return {
            "project": {
                "name": self.project_name.text().strip(),
                "authors": "Nikola Djokic + Open AI",
            },
            "boundary": {
                "layer_id": boundary_id or "",
                "layer_name": boundary_layer.name() if boundary_layer else "",
                "layer_source": boundary_layer.source() if boundary_layer else "",
            },
            "lulc_periods": periods,
            "predictors": {
                "static": static_predictors,
                "dynamic": dynamic_predictors,
                "dynamic_mapping_by_period": dynamic_mapping,
            },
            "reclassification": {
                "mode": self.reclass_mode.currentText(),
                "align_to_reference_grid": self.align_checkbox.isChecked(),
                "standard_scheme": "1-7 fixed scheme",
            },
            "transitions": transitions,
            "scenarios": scenarios,
            "target_years": self.collect_target_years(),
            "model_settings": {
                "use_neighborhood": self.neighborhood_enabled.isChecked(),
                "neighborhood_strength": self.neighborhood_strength.currentText(),
            },
        }

    def validate_configuration(self):
        problems = []
        warnings = []
        if not self.project_name.text().strip():
            problems.append("Project name is missing.")
        if not self.boundary_combo.currentData():
            problems.append("Boundary layer is missing.")

        periods = self.collect_periods()
        if not periods:
            problems.append("At least one LULC period is required.")
        else:
            has_train = any(p.get("purpose") == "TRAIN" for p in periods)
            if not has_train:
                problems.append("At least one TRAIN period is required.")
            for i, period in enumerate(periods, start=1):
                if not all([period["start_year"], period["start_layer_id"], period["end_year"], period["end_layer_id"]]):
                    problems.append(f"Period {i} is incomplete.")
                for key in ("start_year", "end_year"):
                    try:
                        int(period[key])
                    except Exception:
                        problems.append(f"Period {i} has invalid {key.replace('_', ' ')}.")
                        break

        static_predictors = self.collect_static_predictors()
        dynamic_predictors = self.collect_dynamic_predictors()
        if not static_predictors and not dynamic_predictors:
            problems.append("At least one predictor must be registered (static or dynamic).")
        for i, predictor in enumerate(static_predictors, start=1):
            if not predictor["name"] or not predictor["layer_id"]:
                problems.append(f"Static predictor row {i} is incomplete.")
        for i, predictor in enumerate(dynamic_predictors, start=1):
            if not predictor["name"] or not predictor["year"] or not predictor["layer_id"]:
                problems.append(f"Dynamic predictor row {i} is incomplete.")
            else:
                try:
                    int(predictor["year"])
                except Exception:
                    problems.append(f"Dynamic predictor row {i} has invalid year.")

        transitions = self.collect_transitions(active_only=True)
        if not transitions:
            problems.append("At least one active transition is required.")
        else:
            seen_codes = set()
            for i, transition in enumerate(transitions, start=1):
                if not transition["from"] or not transition["to"]:
                    problems.append(f"Transition row {i} is incomplete.")
                    continue
                try:
                    from_cls = int(transition["from"])
                    to_cls = int(transition["to"])
                except Exception:
                    problems.append(f"Transition row {i} must use numeric class codes.")
                    continue
                if from_cls not in STANDARD_CLASSES or to_cls not in STANDARD_CLASSES:
                    problems.append(f"Transition row {i} must use class codes 1-7.")
                if from_cls == to_cls:
                    problems.append(f"Transition row {i} cannot have identical From and To classes.")
                if transition["code"] in seen_codes:
                    problems.append(f"Duplicate transition detected: {transition['code']}.")
                seen_codes.add(transition["code"])

        scenarios = self.collect_scenarios()
        if not scenarios:
            problems.append("At least one scenario is required.")
        else:
            enabled_count = sum(1 for s in scenarios if s.get("enabled"))
            if enabled_count == 0:
                problems.append("At least one scenario must be enabled.")
            active_codes = {t["code"] for t in transitions if t.get("code")}
            for i, scenario in enumerate(scenarios, start=1):
                if not scenario["name"]:
                    problems.append(f"Scenario {i} is missing a name.")
                for code in active_codes:
                    if code not in scenario["transition_multipliers"]:
                        warnings.append(f"Scenario '{scenario['name'] or i}' is missing multiplier for {code}; default 1.0 will be assumed.")
                for code, value in scenario["transition_multipliers"].items():
                    try:
                        float(value)
                    except Exception:
                        problems.append(f"Scenario '{scenario['name'] or i}' has invalid multiplier for {code}.")

        target_years = self.collect_target_years()
        if not target_years:
            problems.append("At least one target year is required.")

        config = self.build_config()
        pretty = json.dumps(config, indent=2)
        summary_lines = []
        if problems:
            summary_lines.append("Validation issues:")
            summary_lines.extend(f"- {p}" for p in problems)
        else:
            summary_lines.append("Configuration validated successfully.")
        if warnings:
            summary_lines.append("")
            summary_lines.append("Warnings:")
            summary_lines.extend(f"- {w}" for w in warnings)

        mapping = config["predictors"]["dynamic_mapping_by_period"]
        if mapping:
            summary_lines.append("")
            summary_lines.append("Dynamic predictor mapping summary:")
            for row in mapping:
                summary_lines.append(
                    f"- {row['period']}: {row['predictor_name']} -> year {row['selected_predictor_year']} -> {row['selected_layer_name']}"
                )

        summary_lines.append("")
        summary_lines.append("Current configuration:")
        summary_lines.append("")
        summary_lines.append(pretty)
        self.summary.setPlainText("\n".join(summary_lines))

        if problems:
            QMessageBox.warning(self, "Predictor2026 validation", "Configuration is incomplete. See the Export tab for details.")
        else:
            QMessageBox.information(self, "Predictor2026 validation", "Configuration looks good.")

    def run_prediction_in_qgis(self):
        config = self.build_config()
        self.validate_configuration()
        problems = []
        if not config["project"]["name"]:
            problems.append("Project name is missing.")
        if not config["boundary"]["layer_source"]:
            problems.append("Boundary layer is missing.")
        if not config["lulc_periods"]:
            problems.append("At least one LULC period is required.")
        if not config["transitions"]:
            problems.append("At least one active transition is required.")
        if not any(s.get("enabled") for s in config["scenarios"]):
            problems.append("At least one scenario must be enabled.")
        if not config["target_years"]:
            problems.append("At least one target year is required.")
        if problems:
            QMessageBox.warning(self, "Predictor2026", "Cannot run prediction until the configuration is fixed.\n\n" + "\n".join(f"- {p}" for p in problems))
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select output folder for prediction rasters")
        if not output_dir:
            return

        try:
            result = run_prediction(config, output_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Predictor2026 run error", str(exc))
            return

        added_layers = []
        for path in result.get("rasters", []):
            layer_name = Path(path).stem
            layer = QgsRasterLayer(path, layer_name)
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                added_layers.append(layer_name)

        lines = [
            "Prediction run completed successfully.",
            f"Base map year: {result.get('base_year')}",
            f"Training periods used: {result.get('train_period_count')}",
            f"Output folder: {result.get('output_dir')}",
        ]
        model_settings = result.get("model_settings") or {}
        lines.extend([
            f"Neighborhood effect: {'On' if model_settings.get('use_neighborhood', True) else 'Off'}",
            f"Neighborhood strength: {model_settings.get('neighborhood_strength', 'Medium')}",
            "",
            "Created rasters:",
        ])
        lines.extend(f"- {Path(path).name}" for path in result.get("rasters", []))
        if result.get("report_path"):
            lines.extend(["", f"Run report: {result['report_path']}"])
        if result.get("feature_importance_csv_path"):
            lines.append(f"Feature importance CSV: {result['feature_importance_csv_path']}")
        elif result.get("feature_importance_path"):
            lines.append(f"Feature importance JSON: {result['feature_importance_path']}")
        validation = result.get("validation")
        if validation:
            lines.extend(["", "Validation completed:"])
            first = (validation.get("results") or [{}])[0]
            if first:
                lines.append(
                    f"OA = {first.get('overall_accuracy', 0.0):.4f}, Kappa = {first.get('kappa', 0.0):.4f}, "
                    f"Precision = {first.get('macro_precision', 0.0):.4f}, Recall = {first.get('macro_recall', 0.0):.4f}"
                )
            if validation.get("report_path"):
                lines.append(f"Validation report: {validation['report_path']}")
        if added_layers:
            lines.extend(["", "Layers added to QGIS:"] + [f"- {name}" for name in added_layers])
        QMessageBox.information(self, "Predictor2026", "\n".join(lines))

    def export_config(self):
        config = self.build_config()
        path, _ = QFileDialog.getSaveFileName(self, "Export Predictor2026 project JSON", "predictor2026_project.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        QMessageBox.information(self, "Predictor2026", f"Project exported to:\n{path}")
