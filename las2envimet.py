# las2envimet.py
# Main plugin class for LAS to ENVI-met Tree Converter

import os
import time
import tempfile
import numpy as np
import laspy

from qgis.core import (QgsProject, Qgis, QgsMapLayerProxyModel, QgsMapLayerType, QgsPointCloudLayer, QgsPointXY)
from qgis.gui import QgsMessageBar, QgsFileWidget
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtCore import Qt, QStandardPaths, QCoreApplication
from qgis.PyQt.QtGui import QColor, QIcon

from typing import Optional, List, Dict, Any
from .resources import *
from .las2envimet_dialog import LAS2ENVImetDialog
from .ags_presets import AGS_PRESETS
from .file_utils import load_point_cloud, save_filtered_las
from .tree_model import TreeModel
from .export_manager import ExportManager
from .map_tools import PointTool
from . import layer_utils as lutils
from .roi_tool import ROIPolygonTool
from .lad_utils import apply_stem_only_option


class LAS2ENVImet:
    # Main plugin class controlling the GUI and processing workflow
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = self.tr('&LAS to ENVI-met Tree Converter')
        self.first_start = True
        self.dlg: Optional[LAS2ENVImetDialog] = None
        self.model = TreeModel()
        self.point_tool: Optional[PointTool] = None
        self.roi_tool: Optional[ROIPolygonTool] = None
        self.applied_threshold = 0.0
        self.lad_calculated = False
        self.last_picked_point: Optional[tuple] = None

    def tr(self, message: str) -> str:
        # Translate a string using the QGIS translation framework
        return QCoreApplication.translate('LAS2ENVImet', message)

    def add_action(self, icon_path: str, text: str, callback, parent=None):
        # Add a toolbar icon and menu entry for the plugin
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        self.iface.addToolBarIcon(action)
        self.iface.addPluginToMenu(self.menu, action)
        self.actions.append(action)
        return action

    def initGui(self):
        # Initialize the plugin GUI (called by QGIS)
        icon_path = ':/plugins/las2envimet/icon.png'
        self.add_action(
            icon_path,
            text=self.tr('LAS to ENVI-met Converter'),
            callback=self.run,
            parent=self.iface.mainWindow()
        )

    def unload(self):
        # Remove the plugin GUI (called by QGIS)
        for action in self.actions:
            self.iface.removePluginMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)

    
    # Main run method and UI setup
    def run(self):
        # Show the plugin dialog when the user activates the plugin
        try:
            if self.first_start:
                self.dlg = LAS2ENVImetDialog()
                self.setup_ui()
                self.first_start = False
            self.dlg.show()
            self.dlg.exec_()
        except Exception as e:
            self.iface.messageBar().pushMessage(
                self.tr("Critical Error"),
                self.tr("Plugin startup failed: {}").format(str(e)),
                level=Qgis.Critical
            )

    def setup_ui(self):
        # Connect all UI signals and set initial widget states
        if not self.dlg:
            return

        # Layer selection
        self.dlg.mMapLayerComboBox.setFilters(QgsMapLayerProxyModel.PointCloudLayer)

        # File source widget
        self.dlg.fileSource.setStorageMode(QgsFileWidget.GetFile)
        self.dlg.fileSource.setFilter("LAS/LAZ files (*.las *.laz)")
        self.dlg.fileSource.fileChanged.connect(self.on_source_file_changed)

        # Dialog buttons
        self.dlg.button_box.accepted.connect(self.dlg.accept)
        self.dlg.button_box.rejected.connect(self.dlg.reject)

        # Tree species presets
        self.dlg.comboSpecies.clear()
        self.dlg.comboSpecies.addItems(sorted(AGS_PRESETS.keys()))

        # Connect buttons
        self.dlg.btnSelectROI.clicked.connect(self.on_select_roi)
        self.dlg.btnSelectStem.clicked.connect(self.on_select_stem)
        self.dlg.btnFilterCloud.clicked.connect(self.on_filter_cloud)
        self.dlg.btnPreviewScale.clicked.connect(self.on_preview_scaled)
        self.dlg.btnResetTransform.clicked.connect(self.on_reset_transform)
        self.dlg.btnCreateVoxels.clicked.connect(self.on_create_voxels)
        self.dlg.btnCalculateLAD.clicked.connect(self.on_calculate_lad)
        self.dlg.btnSelectRefVoxel.clicked.connect(self.on_select_ref_voxel)
        self.dlg.btnAddRefVoxel.clicked.connect(self.on_add_ref_voxel)
        self.dlg.btnRemoveRefVoxel.clicked.connect(self.on_remove_ref_voxel)
        self.dlg.btnApplyRefinement.clicked.connect(self.on_apply_refinement)
        self.dlg.btnResetRefinement.clicked.connect(self.on_reset_refinement)
        self.dlg.btnExport.clicked.connect(self.on_export)

        # Spin, check and combo boxes
        self.dlg.spinZLevel.valueChanged.connect(self.show_z_slice)
        self.dlg.spinGmT1.valueChanged.connect(self.validate_gm_temps)
        self.dlg.spinGmT2.valueChanged.connect(self.validate_gm_temps)
        self.dlg.spinAmmaxT1.valueChanged.connect(self.validate_ammax_temps)
        self.dlg.spinAmmaxT2.valueChanged.connect(self.validate_ammax_temps)
        self.dlg.checkManualParams.toggled.connect(self.toggle_manual_physiology)
        self.dlg.comboSpecies.currentIndexChanged.connect(self.update_physiological_params)
        self.dlg.spinZFilter.valueChanged.connect(self._validate_z_filter)
        self.dlg.spinZFilterUp.valueChanged.connect(self._validate_z_filter)
        self.dlg.spinCrownZMin.valueChanged.connect(self._validate_crown_z)
        self.dlg.spinCrownZMax.valueChanged.connect(self._validate_crown_z)

        # Trunk enhancement checkbox enables the spin boxes
        self.dlg.checkTrunkEnhance.toggled.connect(self.dlg.spinTrunkHeight.setEnabled)
        self.dlg.checkTrunkEnhance.toggled.connect(self.dlg.spinTrunkLAD.setEnabled)

        # Manual parameters visibility
        self.dlg.checkManualParams.toggled.connect(self.on_manual_params_toggled)
        self.on_manual_params_toggled(self.dlg.checkManualParams.isChecked())

        # Plant ID input
        self.dlg.linePlantID.textChanged.connect(self.on_plant_id_changed)
        self.dlg.linePlantID.setInputMask(">XXXXXX")  # max 6 characters, letters/digits

        # Output file widget
        file_out_widget = self.dlg.fileOutput.fileWidget()
        if file_out_widget:
            file_out_widget.setStorageMode(QgsFileWidget.SaveFile)
            file_out_widget.setFilter(self.tr("Text files (*.txt)"))
        self.dlg.fileOutput.setEnabled(False)

        # Database file widget
        db_widget = self.dlg.fileDatabase.fileWidget()
        if db_widget:
            db_widget.setStorageMode(QgsFileWidget.GetFile)
            db_widget.setFilter(self.tr("ENVI-met Database (projectdatabase.edb)"))
        default_db_path = r"D:\enviprojects\userdatabase\projectdatabase.edb"
        self.dlg.fileDatabase.setDocumentPath(default_db_path)

        # Initial button states
        self.dlg.btnExport.setEnabled(False)
        self.dlg.btnSelectStem.setEnabled(False)
        self.dlg.spinTargetHeight.setEnabled(False)
        self.dlg.spinTargetWidthX.setEnabled(False)
        self.dlg.spinTargetWidthY.setEnabled(False)
        self.dlg.spinRotation.setEnabled(False)
        self.dlg.btnPreviewScale.setEnabled(False)
        self.dlg.btnResetTransform.setEnabled(False)
        self.dlg.btnCreateVoxels.setEnabled(False)
        self.dlg.btnCalculateLAD.setEnabled(False)
        self.dlg.btnSelectRefVoxel.setEnabled(False)
        self.dlg.btnAddRefVoxel.setEnabled(False)
        self.dlg.btnRemoveRefVoxel.setEnabled(False)
        self.dlg.btnApplyRefinement.setEnabled(False)
        self.dlg.btnResetRefinement.setEnabled(False)
        self.dlg.spinZLevel.setEnabled(False)
        self.dlg.spinLADValue.setEnabled(False)
        self.dlg.spinCrownFactor.setEnabled(False)
        self.dlg.spinCrownZMin.setEnabled(False)
        self.dlg.spinCrownZMax.setEnabled(False)
        self.dlg.checkTrunkEnhance.setEnabled(False)
        self.dlg.checkClearGround.setEnabled(False)
        self.dlg.spinTrunkHeight.setEnabled(False)
        self.dlg.spinTrunkLAD.setEnabled(False)
        self.dlg.spinLADThreshold.setEnabled(False)

        # Initialise physiological parameters
        self.update_physiological_params()
        self.toggle_manual_physiology()

    
    # UI helper methods (validation, updates)
    def validate_gm_temps(self):
        # Ensure that GM temperature 1 <= GM temperature 2
        t1 = self.dlg.spinGmT1.value()
        t2 = self.dlg.spinGmT2.value()
        if t1 > t2:
            self.dlg.spinGmT2.setValue(t1)
        self.dlg.lblGmRange.setText(
            self.tr("GM range: {:.1f}°C – {:.1f}°C").format(t1, t2)
        )

    def validate_ammax_temps(self):
        # Ensure that Amax temperature 1 <= Amax temperature 2
        t1 = self.dlg.spinAmmaxT1.value()
        t2 = self.dlg.spinAmmaxT2.value()
        if t1 > t2:
            self.dlg.spinAmmaxT2.setValue(t1)
        self.dlg.lblAmmaxRange.setText(
            self.tr("Ammax range: {:.1f}°C – {:.1f}°C").format(t1, t2)
        )

    def toggle_manual_physiology(self):
        # Enable/disable a-gs section
        manual = self.dlg.checkManualParams.isChecked()
        self.dlg.comboSpecies.setEnabled(not manual)

        fields = [
            self.dlg.spinCompPoint25,
            self.dlg.spinCompPointQ10,
            self.dlg.spinGm25,
            self.dlg.spinGmQ10,
            self.dlg.spinGmT1,
            self.dlg.spinGmT2,
            self.dlg.spinAmmax25,
            self.dlg.spinAmmaxQ10,
            self.dlg.spinAmmaxT1,
            self.dlg.spinAmmaxT2,
            self.dlg.spinE0,
            self.dlg.spinF0,
            self.dlg.spinGc,
            self.dlg.spinDmax,
        ]
        for f in fields:
            f.setEnabled(manual)

        # Albedo, transmittance, emissivity always enabled
        self.dlg.spinAlbedo.setEnabled(True)
        self.dlg.spinTransmittance.setEnabled(True)
        self.dlg.spinEmissivity.setEnabled(True)

    def update_physiological_params(self):
        # Update A‑gs parameters when a preset species is selected
        if self.dlg.checkManualParams.isChecked():
            return
        species = self.dlg.comboSpecies.currentText()
        params = AGS_PRESETS.get(species, {})
        if params:
            self.dlg.spinCompPoint25.setValue(params.get("comp_point25", 45.0))
            self.dlg.spinCompPointQ10.setValue(params.get("comp_point_q10", 1.5))
            self.dlg.spinGm25.setValue(params.get("gm25", 5.0))
            self.dlg.spinGmQ10.setValue(params.get("gm_q10", 2.0))
            self.dlg.spinGmT1.setValue(params.get("gm_t1", 5.0))
            self.dlg.spinGmT2.setValue(params.get("gm_t2", 36.0))
            self.dlg.spinAmmax25.setValue(params.get("ammax25", 2.2))
            self.dlg.spinAmmaxQ10.setValue(params.get("ammax_q10", 2.0))
            self.dlg.spinAmmaxT1.setValue(params.get("ammax_t1", 8.0))
            self.dlg.spinAmmaxT2.setValue(params.get("ammax_t2", 36.0))
            self.dlg.spinE0.setValue(params.get("e0", 0.017))
            self.dlg.spinF0.setValue(params.get("f0", 0.85))
            self.dlg.spinGc.setValue(params.get("gc", 0.25))
            self.dlg.spinDmax.setValue(params.get("dmax", 20.0))
            self.dlg.spinAlbedo.setValue(params.get("albedo", 0.18))
            self.dlg.spinTransmittance.setValue(params.get("transmittance", 0.30))
            self.dlg.spinEmissivity.setValue(params.get("eps", 0.96))
        self.validate_gm_temps()
        self.validate_ammax_temps()

    def on_manual_params_toggled(self, checked: bool):
        # Show/hide the detailed A‑gs parameter widgets
        
        widgets = [
            self.dlg.spinCompPoint25,
            self.dlg.spinF0,
            self.dlg.spinGc,
            self.dlg.spinDmax,
            self.dlg.spinE0,
            self.dlg.spinCompPointQ10,
            self.dlg.label_12,
            self.dlg.label_13,
            self.dlg.label_19,
            self.dlg.label_20,
            self.dlg.label_21,
            self.dlg.label_18,
            self.dlg.spinGm25,
            self.dlg.spinGmQ10,
            self.dlg.spinGmT1,
            self.dlg.spinAmmax25,
            self.dlg.spinAmmaxQ10,
            self.dlg.spinAmmaxT1,
            self.dlg.label_14,
            self.dlg.label_15,
            self.dlg.lblGmRange,
            self.dlg.label_16,
            self.dlg.label_17,
            self.dlg.lblAmmaxRange,
            self.dlg.spinGmT2,
            self.dlg.spinAmmaxT2,
        ]
        for w in widgets:
            w.setVisible(checked)

    def _validate_crown_z(self):
        # Ensure CrownZMin < CrownZMax
        min_val = self.dlg.spinCrownZMin.value()
        max_val = self.dlg.spinCrownZMax.value()
        if min_val >= max_val:
            self.dlg.spinCrownZMin.setValue(max_val - 1)
        if max_val <= min_val:
            self.dlg.spinCrownZMax.setValue(min_val + 1)

    def _validate_z_filter(self):
        # Ensure lower Z‑filter percent <= upper Z‑filter percent
        lower = self.dlg.spinZFilter.value()
        upper = self.dlg.spinZFilterUp.value()
        if lower > upper:
            self.dlg.spinZFilter.setValue(upper)
        if upper < lower:
            self.dlg.spinZFilterUp.setValue(lower)

    def suggest_export_path(self, text: str):
        # Update the default export path when a valid Plant ID is entered
        clean_id = text.replace("_", "").strip()
        if len(clean_id) == 6:
            default_dir = os.path.expanduser("~")
            default_path = os.path.join(default_dir, f"{clean_id}.txt")
            self.dlg.fileOutput.setDocumentPath(default_path)

    
    # Point cloud loading and layer handling
    def load_point_cloud_file(self, file_path: str) -> QgsPointCloudLayer:
        # Load a LAS/LAZ file as a point cloud layer
        layer_name = os.path.splitext(os.path.basename(file_path))[0]
        layer = QgsPointCloudLayer(file_path, layer_name, "pdal")
        if not layer.isValid():
            raise Exception("Failed to load point cloud layer.")
        QgsProject.instance().addMapLayer(layer)
        return layer

    def on_source_file_changed(self, file_path: str):
        # Handle selection of a point cloud file via the file widget
        if not file_path:
            return
        try:
            layer = self.load_point_cloud_file(file_path)
            self.dlg.mMapLayerComboBox.setLayer(layer)
        except Exception as e:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("Could not load point cloud: {}").format(str(e)),
                level=Qgis.Critical
            )


    # ROI selection
    def on_select_roi(self):
        # Activate the polygon tool to draw a region of interest
        layer = self.dlg.mMapLayerComboBox.currentLayer()
        if not layer or layer.type() != QgsMapLayerType.PointCloudLayer:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("Please select a point cloud layer!"),
                level=Qgis.Warning
            )
            return

        if self.model.points is None:
            try:
                points, _ = load_point_cloud(layer.source())
                self.model.load_points(points)
            except Exception as e:
                self.iface.messageBar().pushMessage(
                    self.tr("Error"),
                    self.tr("Could not load LAS file: {}").format(str(e)),
                    level=Qgis.Critical
                )
                return

        self.dlg.showMinimized()
        self.roi_tool = ROIPolygonTool(self.iface.mapCanvas(), self.on_roi_drawn)
        self.iface.mapCanvas().setMapTool(self.roi_tool)
        self.iface.messageBar().pushMessage(
            self.tr("Info"),
            self.tr("Draw a polygon around the tree. Right-click to finish."),
            level=Qgis.Info
        )

    def on_roi_drawn(self, polygon):
        # Process the drawn polygon: filter points, create new layer, update model
        self.dlg.showNormal()
        self.dlg.raise_()

        layer = self.dlg.mMapLayerComboBox.currentLayer()
        if not layer or layer.type() != QgsMapLayerType.PointCloudLayer:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("No point cloud layer selected."),
                level=Qgis.Critical
            )
            return

        source_path = layer.source()
        if not os.path.exists(source_path):
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("Source file not found: {}").format(source_path),
                level=Qgis.Critical
            )
            return

        try:
            las = laspy.read(source_path)
            points = np.vstack((las.x, las.y, las.z)).transpose()

            mask = self._points_in_polygon(points, polygon)
            if mask is None or np.sum(mask) == 0:
                self.iface.messageBar().pushMessage(
                    self.tr("Error"),
                    self.tr("No points inside the drawn polygon."),
                    level=Qgis.Critical
                )
                return

            temp_dir = tempfile.gettempdir()
            base_name = os.path.splitext(os.path.basename(source_path))[0]
            timestamp = int(time.time() * 1000)
            temp_path = os.path.join(temp_dir, f"roi_{base_name}_{timestamp}.las")

            header = laspy.LasHeader(
                point_format=las.header.point_format,
                version=las.header.version
            )
            filtered_points = las.points[mask]
            with laspy.open(temp_path, mode='w', header=header) as writer:
                writer.write_points(filtered_points)

            # Remove old ROI layers
            for lyr in QgsProject.instance().mapLayers().values():
                if lyr.name().startswith("ROI Points") or "roi_" in lyr.source():
                    QgsProject.instance().removeMapLayer(lyr.id())

            # Load new ROI layer
            roi_layer = QgsPointCloudLayer(temp_path, "ROI Points", "pdal")
            if not roi_layer.isValid():
                raise Exception("Failed to load filtered point cloud.")
            QgsProject.instance().addMapLayer(roi_layer)

            # Hide the original layer
            root = QgsProject.instance().layerTreeRoot()
            layer_tree_layer = root.findLayer(layer.id())
            if layer_tree_layer:
                layer_tree_layer.setItemVisibilityChecked(False)

            self.dlg.mMapLayerComboBox.setLayer(roi_layer)
            self.model.points = points[mask]
            self.model.stem_point = None
            self.model.current_dims = None
            self.model.transformed_points = None
            self.model.voxel_corners = None
            self.model.lad_values = None
            self.model.ref_voxels = []

            self.dlg.btnSelectStem.setEnabled(True)
            self.dlg.btnCreateVoxels.setEnabled(False)
            self.dlg.btnCalculateLAD.setEnabled(False)
            self.dlg.spinZLevel.setEnabled(False)
            self.dlg.btnSelectRefVoxel.setEnabled(False)

            self.iface.messageBar().pushMessage(
                self.tr("Success"),
                self.tr("ROI applied. New layer created with {} points.").format(len(self.model.points)),
                level=Qgis.Success,
                duration=5
            )

        except Exception as e:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("ROI processing failed: {}").format(str(e)),
                level=Qgis.Critical
            )

    def _points_in_polygon(self, points: np.ndarray, polygon_geom) -> np.ndarray:
        # Return a boolean mask indicating which points lie inside the given polygon
        poly = polygon_geom.asPolygon()[0]
        poly_xy = [(p.x(), p.y()) for p in poly]
        points_xy = points[:, :2]

        try:
            from matplotlib.path import Path
            path = Path(poly_xy)
            return path.contains_points(points_xy)
        except ImportError:
            # fallback
            x, y = points_xy[:, 0], points_xy[:, 1]
            n = len(poly_xy)
            inside = np.zeros(len(points), dtype=bool)
            for i in range(n):
                x1, y1 = poly_xy[i]
                x2, y2 = poly_xy[(i + 1) % n]
                cond = ((y1 > y) != (y2 > y))
                xinters = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                inside[cond & (x < xinters)] = ~inside[cond & (x < xinters)]
            return inside

    # Trunk point selection
    def on_select_stem(self):
        # Activate point tool to pick the trunk base
        if self.model.points is None or len(self.model.points) == 0:
            layer = self.dlg.mMapLayerComboBox.currentLayer()
            if not layer:
                self.iface.messageBar().pushMessage(
                    self.tr("Error"),
                    self.tr("No layer selected!"),
                    level=Qgis.Warning
                )
                return
            if layer.type() != QgsMapLayerType.PointCloudLayer:
                self.iface.messageBar().pushMessage(
                    self.tr("Error"),
                    self.tr("Please select a point cloud layer (.las/.laz)!"),
                    level=Qgis.Warning
                )
                return
            try:
                points, las = load_point_cloud(layer.source())
                self.model.load_points(points, las)
            except Exception as e:
                self.iface.messageBar().pushMessage(
                    self.tr("Error"),
                    self.tr("Could not load LAS file: {}").format(str(e)),
                    level=Qgis.Critical
                )
                return
        else:
            self.iface.messageBar().pushMessage(
                self.tr("Info"),
                self.tr("Using existing points (ROI or previous load)."),
                level=Qgis.Info
            )

        self.dlg.showMinimized()
        self.point_tool = PointTool(self.iface.mapCanvas(), self.on_stem_picked)
        self.iface.mapCanvas().setMapTool(self.point_tool)
        self.iface.messageBar().pushMessage(
            self.tr("Info"),
            self.tr("Please click on the trunk center point on the map."),
            level=Qgis.Info
        )

    def on_stem_picked(self, point: QgsPointXY):
        # Handle the trunk point picked by the user
        self.iface.mapCanvas().unsetMapTool(self.point_tool)
        self.dlg.showNormal()
        self.dlg.raise_()

        x = point.x()
        y = point.y()
        self.model.set_stem(x, y)
        self.dlg.lblStemCoords.setText(f"X: {x:.2f}, Y: {y:.2f}")

        # Remove temporary preview layers
        for layer in QgsProject.instance().mapLayers().values():
            if layer.name().startswith("Trunk_Preview") or layer.name().startswith("Filter_"):
                QgsProject.instance().removeMapLayer(layer.id())

        dims = self.model.current_dims
        if dims:
            self.dlg.lineHeight.setText(f"{dims['height']:.2f}")
            self.dlg.lineWidthX.setText(f"{dims['total_width_x']:.2f}")
            self.dlg.lineWidthY.setText(f"{dims['total_width_y']:.2f}")
            self.dlg.spinTargetHeight.setValue(dims['height'])
            self.dlg.spinTargetWidthX.setValue(dims['total_width_x'])
            self.dlg.spinTargetWidthY.setValue(dims['total_width_y'])
            self.dlg.spinRotation.setValue(0.0)

        self.dlg.spinTargetHeight.setEnabled(True)
        self.dlg.spinTargetWidthX.setEnabled(True)
        self.dlg.spinTargetWidthY.setEnabled(True)
        self.dlg.spinRotation.setEnabled(True)
        self.dlg.btnPreviewScale.setEnabled(True)
        self.dlg.btnCreateVoxels.setEnabled(True)

        self.iface.messageBar().pushMessage(
            self.tr("Info"),
            self.tr("Trunk set. Using {} points for further processing.").format(len(self.model.points)),
            level=Qgis.Info,
            duration=3
        )

    # Z‑filter preview
    def on_filter_cloud(self):
        # Create a temporary layer showing points between two Z‑percentiles
        layer = self.dlg.mMapLayerComboBox.currentLayer()
        if not layer or layer.type() != QgsMapLayerType.PointCloudLayer:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("Please select a point cloud layer!"),
                level=Qgis.Warning
            )
            return

        if self.model.points is None:
            try:
                points, _ = load_point_cloud(layer.source())
                self.model.load_points(points)
            except Exception as e:
                self.iface.messageBar().pushMessage(
                    self.tr("Error"),
                    self.tr("Could not load LAS file: {}").format(str(e)),
                    level=Qgis.Critical
                )
                return

        # Remove existing filter layers
        for lyr in QgsProject.instance().mapLayers().values():
            if lyr.name().startswith("Filter_"):
                QgsProject.instance().removeMapLayer(lyr.id())

        lower_percent = self.dlg.spinZFilter.value()
        upper_percent = self.dlg.spinZFilterUp.value()

        z = self.model.points[:, 2]
        z_min, z_max = np.min(z), np.max(z)
        lower_thresh = z_min + (lower_percent / 100.0) * (z_max - z_min)
        upper_thresh = z_min + (upper_percent / 100.0) * (z_max - z_min)

        mask = (z >= lower_thresh) & (z <= upper_thresh)
        filtered = self.model.points[mask]

        # Limit to 50k points for performance
        if len(filtered) > 50000:
            idx = np.random.choice(len(filtered), 50000, replace=False)
            filtered = filtered[idx]

        crs = layer.crs().authid()
        layer_name = f"Filter_{lower_percent:.2f}-{upper_percent:.2f}%"
        vl = lutils.create_memory_layer(crs, layer_name, "Point")
        lutils.add_points_to_layer(vl, filtered)
        QgsProject.instance().addMapLayer(vl)

        self.dlg.btnSelectStem.setEnabled(True)

        self.iface.messageBar().pushMessage(
            self.tr("Info"),
            self.tr("Filter applied: {} points displayed ({}% to {}%).").format(
                len(filtered), lower_percent, upper_percent
            ),
            level=Qgis.Info
        )

    
    # Scaling and rotation preview
    def on_preview_scaled(self):
        # Show scaled and rotated point cloud
        layer = self.dlg.mMapLayerComboBox.currentLayer()
        if not layer or layer.type() != QgsMapLayerType.PointCloudLayer:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("Please select a point cloud layer!"),
                level=Qgis.Warning
            )
            return

        if self.model.current_dims is None:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("Please select a trunk point first."),
                level=Qgis.Warning
            )
            return

        if self.model.points is None:
            try:
                points, _ = load_point_cloud(layer.source())
                self.model.load_points(points)
            except Exception as e:
                self.iface.messageBar().pushMessage(
                    self.tr("Error"),
                    self.tr("Could not load LAS file: {}").format(str(e)),
                    level=Qgis.Critical
                )
                return

        target = {
            'height': self.dlg.spinTargetHeight.value(),
            'total_width_x': self.dlg.spinTargetWidthX.value(),
            'total_width_y': self.dlg.spinTargetWidthY.value()
        }
        angle = self.dlg.spinRotation.value()
        transformed = self.model.transform_points(target, angle)
        if transformed is None:
            return

        # Remove old preview layer
        for lyr in QgsProject.instance().mapLayers().values():
            if lyr.name() == "Scaled_Tree_Preview":
                QgsProject.instance().removeMapLayer(lyr.id())

        crs = layer.crs().authid()
        vl = lutils.create_memory_layer(crs, "Scaled_Tree_Preview", "Point")
        lutils.add_points_to_layer(vl, transformed)
        QgsProject.instance().addMapLayer(vl)

        self.dlg.btnResetTransform.setEnabled(True)
        self.iface.messageBar().pushMessage(
            self.tr("Info"),
            self.tr("Preview updated."),
            level=Qgis.Info
        )

    def on_reset_transform(self):
        # Reset scaling and rotation to original values
        self.dlg.spinTargetHeight.setValue(self.model.current_dims['height'])
        self.dlg.spinTargetWidthX.setValue(self.model.current_dims['total_width_x'])
        self.dlg.spinTargetWidthY.setValue(self.model.current_dims['total_width_y'])
        self.dlg.spinRotation.setValue(0.0)

        # Remove preview layer
        for lyr in QgsProject.instance().mapLayers().values():
            if lyr.name() == "Scaled_Tree_Preview":
                QgsProject.instance().removeMapLayer(lyr.id())

        # Reset model data that depends on transformation
        self.model.transformed_points = None
        self.model.voxel_corners = None
        self.model.voxel_indices = None
        self.model.unique_voxels = None
        self.model.point_counts = None
        self.model.lad_values = None
        self.model.base_lad = None
        self.model.transformed_stem_point = None
        self.model.ref_voxels = []

        self.dlg.btnCreateVoxels.setEnabled(True)

        self.dlg.btnCalculateLAD.setEnabled(False)
        self.dlg.btnSelectRefVoxel.setEnabled(False)
        self.dlg.btnAddRefVoxel.setEnabled(False)
        self.dlg.btnRemoveRefVoxel.setEnabled(False)
        self.dlg.btnApplyRefinement.setEnabled(False)
        self.dlg.btnResetRefinement.setEnabled(False)
        self.dlg.spinZLevel.setEnabled(False)
        self.dlg.spinLADValue.setEnabled(False)
        self.dlg.spinCrownFactor.setEnabled(False)
        self.dlg.spinCrownZMin.setEnabled(False)
        self.dlg.spinCrownZMax.setEnabled(False)
        self.dlg.checkTrunkEnhance.setEnabled(False)
        self.dlg.spinTrunkHeight.setEnabled(False)
        self.dlg.spinTrunkLAD.setEnabled(False)
        self.dlg.spinLADThreshold.setEnabled(False)
        self.dlg.checkClearGround.setEnabled(False)
        self.dlg.btnResetTransform.setEnabled(False)

        self.iface.messageBar().pushMessage(
            self.tr("Info"),
            self.tr("Transform values reset. You can now create new voxels from original dimensions."),
            level=Qgis.Info
        )

    # Voxel creation
    def on_create_voxels(self):
        # Create a voxel grid from the (transformed) point cloud
        for lyr in QgsProject.instance().mapLayers().values():
            if lyr.name() == "Scaled_Tree_Preview":
                QgsProject.instance().removeMapLayer(lyr.id())

        layer = self.dlg.mMapLayerComboBox.currentLayer()
        if not layer or layer.type() != QgsMapLayerType.PointCloudLayer:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("Please select a point cloud layer!"),
                level=Qgis.Warning
            )
            return

        self.lad_calculated = False
        self.dlg.btnExport.setEnabled(False)

        if self.model.points is None:
            try:
                points, las = load_point_cloud(layer.source())
                self.model.load_points(points, las)
            except Exception as e:
                self.iface.messageBar().pushMessage(
                    self.tr("Error"),
                    self.tr("Could not load LAS file: {}").format(str(e)),
                    level=Qgis.Critical
                )
                return

        if self.model.stem_point is None:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("Please select a trunk point first."),
                level=Qgis.Warning
            )
            return

        if self.model.transformed_points is None:
            target = {
                'height': self.dlg.spinTargetHeight.value(),
                'total_width_x': self.dlg.spinTargetWidthX.value(),
                'total_width_y': self.dlg.spinTargetWidthY.value()
            }
            angle = self.dlg.spinRotation.value()
            self.model.transform_points(target, angle)

        try:
            unique, corners, counts = self.model.create_voxels(voxel_size=1.0)

            self.iface.messageBar().pushMessage(
                self.tr("Info"),
                self.tr("Voxelizing {} points.").format(len(self.model.transformed_points)),
                level=Qgis.Info,
                duration=3
            )

            z_vals = np.unique(corners[:, 2])
            min_z = int(np.min(z_vals))
            max_z = int(np.max(z_vals))

            self.dlg.spinZLevel.setMinimum(min_z)
            self.dlg.spinZLevel.setMaximum(max_z)
            self.dlg.spinZLevel.setValue(min_z)

            self.dlg.lblTreeHeightInfo.setText(
                self.tr("Tree height: {} cells (Z: {} to {})").format(max_z - min_z + 1, min_z, max_z)
            )

            self.dlg.spinCrownZMin.setMinimum(min_z)
            self.dlg.spinCrownZMin.setMaximum(max_z - 1)
            self.dlg.spinCrownZMin.setValue(min_z)

            self.dlg.spinCrownZMax.setMinimum(min_z + 1)
            self.dlg.spinCrownZMax.setMaximum(max_z)
            self.dlg.spinCrownZMax.setValue(max_z)

            self.dlg.spinTrunkHeight.setMaximum(max_z)

            self.dlg.spinZLevel.setEnabled(True)
            self.dlg.btnSelectRefVoxel.setEnabled(True)
            self.dlg.spinLADValue.setEnabled(True)
            self.dlg.btnAddRefVoxel.setEnabled(True)
            self.dlg.btnRemoveRefVoxel.setEnabled(True)
            self.dlg.btnCalculateLAD.setEnabled(True)

            self.show_z_slice(self.dlg.spinZLevel.value())

            self.iface.messageBar().pushMessage(
                self.tr("Success"),
                self.tr("Voxels created: {} cells.").format(len(unique)),
                level=Qgis.Success
            )
        except Exception as e:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("Voxel creation failed: {}").format(str(e)),
                level=Qgis.Critical
            )

    # LAD calculation and refinement
    def on_calculate_lad(self):
        # Calculate LAD values from reference voxels
        if self.model.point_counts is None:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("No voxel data available."),
                level=Qgis.Warning
            )
            return

        if len(self.model.ref_voxels) == 0:
            self.iface.messageBar().pushMessage(
                self.tr("Hint"),
                self.tr("Please add at least one reference voxel."),
                level=Qgis.Warning
            )
            return

        if not self.model.apply_refinement():
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("Calibration failed."),
                level=Qgis.Critical
            )
            return

        self.lad_calculated = True

        self.dlg.btnSelectRefVoxel.setEnabled(True)
        self.dlg.btnAddRefVoxel.setEnabled(True)
        self.dlg.btnRemoveRefVoxel.setEnabled(True)
        self.dlg.btnApplyRefinement.setEnabled(True)
        self.dlg.spinZLevel.setEnabled(True)
        self.dlg.spinLADValue.setEnabled(True)
        self.dlg.spinCrownFactor.setEnabled(True)
        self.dlg.spinCrownZMin.setEnabled(True)
        self.dlg.spinCrownZMax.setEnabled(True)
        self.dlg.checkTrunkEnhance.setEnabled(True)
        self.dlg.spinLADThreshold.setEnabled(True)
        self.dlg.checkClearGround.setEnabled(True)

        self.dlg.spinTrunkHeight.setEnabled(self.dlg.checkTrunkEnhance.isChecked())
        self.dlg.spinTrunkLAD.setEnabled(self.dlg.checkTrunkEnhance.isChecked())

        self.dlg.btnResetRefinement.setEnabled(False)

        self.applied_threshold = 0.0

        # Enable export if PlantID is valid
        clean_id = self.dlg.linePlantID.text().replace("_", "").strip()
        if len(clean_id) == 6:
            self.dlg.btnExport.setEnabled(True)

        self.show_z_slice(self.dlg.spinZLevel.value())
        self.iface.messageBar().pushMessage(
            self.tr("Success"),
            self.tr("LAD calculated. You can now apply further refinements."),
            level=Qgis.Success
        )

    def on_select_ref_voxel(self):
        # Activate point tool for picking a reference voxel
        if self.model.voxel_corners is None:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("No voxels available."),
                level=Qgis.Warning
            )
            return
        self.point_tool = PointTool(self.iface.mapCanvas(), self.on_voxel_picked)
        self.iface.mapCanvas().setMapTool(self.point_tool)
        self.iface.messageBar().pushMessage(
            self.tr("Info"),
            self.tr("Click on a voxel on the map."),
            level=Qgis.Info
        )

    def on_voxel_picked(self, point: QgsPointXY):
        # Store the picked voxel coordinates for later addition
        self.iface.mapCanvas().unsetMapTool(self.point_tool)
        self.last_picked_point = (point.x(), point.y(), self.dlg.spinZLevel.value())
        self.iface.messageBar().pushMessage(
            self.tr("Info"),
            self.tr("Voxel at ({:.2f}, {:.2f}, Z={}) selected. Enter LAD and click 'Add'.").format(
                point.x(), point.y(), self.dlg.spinZLevel.value()
            ),
            level=Qgis.Info
        )

    def on_add_ref_voxel(self):
        # Add the last picked voxel as a reference with the current LAD value
        if self.last_picked_point is None:
            self.iface.messageBar().pushMessage(
                self.tr("Warning"),
                self.tr("Please select a voxel first."),
                level=Qgis.Warning
            )
            return
        x, y, z = self.last_picked_point
        lad_val = self.dlg.spinLADValue.value()
        ref = self.model.add_reference_voxel(x, y, z, lad_val)
        if ref is None:
            self.iface.messageBar().pushMessage(
                self.tr("Warning"),
                self.tr("Could not add voxel (maybe already present?)."),
                level=Qgis.Warning
            )
            return
        text = f"LAD: {lad_val:.2f} | Pts: {ref['point_count']} (X:{ref['x']:.0f} Y:{ref['y']:.0f})"
        self.dlg.listRefVoxels.addItem(text)
        self.iface.messageBar().pushMessage(
            self.tr("Success"),
            self.tr("Reference voxel added."),
            level=Qgis.Success
        )

    def on_remove_ref_voxel(self):
        # Remove the selected reference voxel from the list
        row = self.dlg.listRefVoxels.currentRow()
        if row < 0:
            self.iface.messageBar().pushMessage(
                self.tr("Info"),
                self.tr("Please select an item in the list."),
                level=Qgis.Info
            )
            return
        self.model.remove_reference_voxel(row)
        self.dlg.listRefVoxels.takeItem(row)
        self.iface.messageBar().pushMessage(
            self.tr("Info"),
            self.tr("Reference removed."),
            level=Qgis.Info
        )

    def on_apply_refinement(self):
        # Apply crown factor, trunk enhancement, threshold and ground clearing
        if self.model.base_lad is None:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("No base LAD available. Please run 'Calculate LAD' first."),
                level=Qgis.Warning
            )
            return

        lad = self.model.base_lad.copy()

        crown_factor = self.dlg.spinCrownFactor.value()
        z_min = self.dlg.spinCrownZMin.value()
        z_max = self.dlg.spinCrownZMax.value()
        for i, corner in enumerate(self.model.voxel_corners):
            if z_min <= corner[2] <= z_max:
                lad[i] *= crown_factor

        if self.dlg.checkTrunkEnhance.isChecked():
            trunk_height = self.dlg.spinTrunkHeight.value()
            trunk_lad = self.dlg.spinTrunkLAD.value()
            stem_x, stem_y, _ = self.model.stem_point
            for i, corner in enumerate(self.model.voxel_corners):
                if (corner[2] <= trunk_height and
                        corner[0] <= stem_x <= corner[0] + 1.0 and
                        corner[1] <= stem_y <= corner[1] + 1.0):
                    lad[i] = trunk_lad

        threshold = self.dlg.spinLADThreshold.value()
        if threshold > 0:
            lad[lad < threshold] = 0.0
        self.applied_threshold = threshold

        # Keep only stem voxel at z=0
        if self.dlg.checkClearGround.isChecked():
            if hasattr(self.model, 'stem_voxel_index') and self.model.stem_voxel_index is not None:
                lad = apply_stem_only_option(
                    self.model.voxel_corners,
                    lad,
                    self.model.stem_voxel_index,
                    voxel_size=1.0
                )
            else:
                self.iface.messageBar().pushMessage(
                    self.tr("Warning"),
                    self.tr("Stem voxel index not found. Clear Ground may not work correctly."),
                    level=Qgis.Warning
                )

        self.model.lad_values = lad
        self.dlg.btnResetRefinement.setEnabled(True)

        self.show_z_slice(self.dlg.spinZLevel.value())
        self.iface.messageBar().pushMessage(
            self.tr("Success"),
            self.tr("Refinement applied."),
            level=Qgis.Success
        )

    def on_reset_refinement(self):
        # Reset LAD values to the state after 'Calculate LAD'
        if self.model.base_lad is None:
            self.iface.messageBar().pushMessage(
                self.tr("Error"),
                self.tr("No base LAD available. Please run 'Calculate LAD' first."),
                level=Qgis.Warning
            )
            return

        self.model.lad_values = self.model.base_lad.copy()
        self.applied_threshold = 0.0

        self.dlg.spinCrownFactor.setValue(1.0)
        if self.model.voxel_corners is not None:
            z_vals = np.unique(self.model.voxel_corners[:, 2])
            min_z = int(np.min(z_vals))
            max_z = int(np.max(z_vals))
            self.dlg.spinCrownZMin.setValue(min_z)
            self.dlg.spinCrownZMax.setValue(max_z)

        self.dlg.checkTrunkEnhance.setChecked(False)
        self.dlg.spinLADThreshold.setValue(0.0)
        self.dlg.btnResetRefinement.setEnabled(False)

        self.show_z_slice(self.dlg.spinZLevel.value())
        self.iface.messageBar().pushMessage(
            self.tr("Info"),
            self.tr("Refinement settings reset."),
            level=Qgis.Info
        )

    def show_z_slice(self, z: int):
        # Display a polygon layer for the given Z‑level, coloured by LAD or point count
        if self.model.voxel_corners is None:
            return

        mask, corners, counts = self.model.get_voxels_at_z(z)
        if mask is None or np.sum(mask) == 0:
            return

        # Remove old voxel layers
        for layer in QgsProject.instance().mapLayers().values():
            if layer.name().startswith("Voxel_Selection_Layer"):
                QgsProject.instance().removeMapLayer(layer.id())

        crs = self.iface.mapCanvas().mapSettings().destinationCrs().authid()
        layer_name = f"Voxel_Selection_Layer_Z{int(z)}"
        layer = lutils.create_memory_layer(crs, layer_name, "Polygon")

        lad_vals = self.model.lad_values[mask] if self.model.lad_values is not None else None
        min_lad = self.applied_threshold if self.model.lad_values is not None else 0.0
        lutils.add_voxel_polygons(layer, corners, counts, lad_vals, min_lad=min_lad)

        if self.model.lad_values is not None:
            # LAD view
            start_color = QColor(240, 255, 240)
            end_color = QColor(0, 80, 0)
            lutils.apply_dynamic_graduated_style(layer, "LAD", start_color, end_color, num_classes=6)
            lutils.enable_labels(layer, field_name="LAD", text_size=8, decimals=2)
        else:
            # Point count view
            start_color = QColor(240, 240, 255)
            end_color = QColor(0, 0, 120)
            lutils.apply_dynamic_graduated_style(layer, "points", start_color, end_color, num_classes=6)
            lutils.enable_labels(layer, field_name="points", text_size=8, decimals=0)

        QgsProject.instance().addMapLayer(layer)


    # Export
    def get_current_ags_params(self) -> Dict[str, float]:
        # Collect the current A‑gs parameters from the UI
        return {
            "comp_point25": self.dlg.spinCompPoint25.value(),
            "comp_point_q10": self.dlg.spinCompPointQ10.value(),
            "gm25": self.dlg.spinGm25.value(),
            "gm_q10": self.dlg.spinGmQ10.value(),
            "gm_t1": self.dlg.spinGmT1.value(),
            "gm_t2": self.dlg.spinGmT2.value(),
            "ammax25": self.dlg.spinAmmax25.value(),
            "ammax_q10": self.dlg.spinAmmaxQ10.value(),
            "ammax_t1": self.dlg.spinAmmaxT1.value(),
            "ammax_t2": self.dlg.spinAmmaxT2.value(),
            "e0": self.dlg.spinE0.value(),
            "f0": self.dlg.spinF0.value(),
            "gc": self.dlg.spinGc.value(),
            "dmax": self.dlg.spinDmax.value()
        }

    def on_export(self):
        # Export the tree as an ENVI‑met TXT file and optionally update the database
        plant_id = self.dlg.linePlantID.text().replace("_", "").strip()
        if len(plant_id) != 6:
            QMessageBox.warning(
                self.dlg,
                self.tr("Export"),
                self.tr("Plant ID must be exactly 6 characters long!")
            )
            return

        output_path = self.dlg.fileOutput.documentPath()
        if not output_path:
            QMessageBox.warning(
                self.dlg,
                self.tr("Export"),
                self.tr("Please choose a save location.")
            )
            return

        db_path = self.dlg.fileDatabase.documentPath()
        description = self.dlg.lineDescription.text().strip() or "Generated from LiDAR"
        albedo = self.dlg.spinAlbedo.value()
        transmittance = self.dlg.spinTransmittance.value()
        emissivity = self.dlg.spinEmissivity.value()
        ags_params = self.get_current_ags_params()
        lad_threshold = self.dlg.spinLADThreshold.value() if hasattr(self.dlg, 'spinLADThreshold') else 0.0

        try:
            xml_content = ExportManager.export_to_txt(
                model=self.model,
                plant_id=plant_id,
                description=description,
                albedo=albedo,
                transmittance=transmittance,
                emissivity=emissivity, 
                ags_params=ags_params,
                output_path=output_path,
                lad_threshold=lad_threshold
            )
            msg = self.tr("Tree file saved to: {}").format(output_path)
            if db_path and os.path.exists(db_path):
                ExportManager.update_database(db_path, xml_content, plant_id)
                msg += self.tr("\n\nDatabase updated.")
            QMessageBox.information(self.dlg, self.tr("Export successful"), msg)
        except Exception as e:
            QMessageBox.critical(self.dlg, self.tr("Export failed"), str(e))

    def on_plant_id_changed(self, text: str):
        # Handle changes to the Plant ID input field
        if len(text) > 6:
            self.dlg.linePlantID.setText(text[:6])
            return
        clean_id = text.replace("_", "").strip()
        if len(clean_id) == 6:
            self.dlg.fileOutput.setEnabled(True)
            docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
            default_path = os.path.join(docs, f"{clean_id}.txt")
            self.dlg.fileOutput.setDocumentPath(default_path)
            if self.lad_calculated:
                self.dlg.btnExport.setEnabled(True)
        else:
            self.dlg.fileOutput.setEnabled(False)
            self.dlg.btnExport.setEnabled(False)