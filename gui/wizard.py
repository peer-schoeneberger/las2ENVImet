# wizard.py
# Step-by-step wizard
import json
import os
import traceback

import numpy as np
from qgis.core import QgsProject, QgsSettings
from qgis.gui import QgsFileWidget
from qgis.PyQt.QtCore import QCoreApplication, Qt, QUrl
from qgis.PyQt.QtGui import QColor, QDesktopServices, QGuiApplication
from qgis.PyQt.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
                                 QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                                 QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                                 QRadioButton, QScrollArea, QSpinBox, QSplitter, QStackedWidget, QVBoxLayout,
                                 QWidget)

from .. import readers
from ..core import foliage, presets, project_io, roots
from ..core.geometry import z_percent_mask
from ..core.physiology import AGS_PRESETS, ags_of, preset_names
from ..core.tree_model import AGS_KEYS, DEFAULT_SEASON_PROFILE, TreeWorkflow, build_tree_data
from ..exporters import (DEFAULT_FORMAT, FORMAT_JSON6, FORMAT_LEGACY, ExportError, PlantExistsError, detect_format,
                         get_exporter, is_valid_id, normalise_id, suggest_free_id)
from . import layers, tasks
from .ascii_dialog import AsciiMappingDialog
from .compat import (MSG_CRITICAL, MSG_INFO, MSG_SUCCESS, MSG_WARNING, exec_dialog, is_point_cloud_layer)
from .map_tools import PointTool, RoiPolygonTool
from .preview import SideViewWidget

README_URL = 'https://github.com/peer-schoeneberger/las2ENVImet#'
STEP_KEYS = ('load', 'select', 'filter', 'transform', 'voxel', 'foliage', 'roots', 'physiology', 'export')
HELP_ANCHORS = {'load': 'point-cloud-formats', 'select': 'select-the-tree', 'filter': 'filter',
                'transform': 'trunk-and-transformation', 'voxel': 'voxelisation', 'foliage': 'foliage-lad-and-lai',
                'roots': 'root-system', 'physiology': 'physiology-a-gs', 'export': 'export-to-envi-met'}
AGS_LABELS = [
    ('e0', 'Max. quantum efficiency E0 [mg/J]', 4, 0.0, 1.0), ('f0', 'Stomata humidity factor f0 [-]', 3, 0.0, 1.0),
    ('gc', 'Cuticular conductance Gc [mm/s]', 3, 0.0, 10.0), ('dmax', 'Max. humidity deficit Dmax [g/kg]', 2, 0.0, 100.0),
    ('ammax25', 'Am,max at 25 °C [mg/(m² s)]', 3, 0.0, 20.0), ('ammax_q10', 'Am,max Q10 [-]', 2, 0.0, 10.0),
    ('ammax_t1', 'Am,max lower temperature T1 [°C]', 1, -20.0, 60.0),
    ('ammax_t2', 'Am,max upper temperature T2 [°C]', 1, -20.0, 60.0),
    ('gm25', 'Mesophyll conductance gm at 25 °C [mm/s]', 3, 0.0, 50.0), ('gm_q10', 'gm Q10 [-]', 2, 0.0, 10.0),
    ('gm_t1', 'gm lower temperature T1 [°C]', 1, -20.0, 60.0), ('gm_t2', 'gm upper temperature T2 [°C]', 1, -20.0, 60.0),
    ('comp_point25', 'CO₂ compensation point at 25 °C [ppm]', 2, 0.0, 200.0),
    ('comp_point_q10', 'CO₂ compensation point Q10 [-]', 2, 0.0, 10.0),
]


def _dspin(minimum, maximum, value, decimals=2, step=0.1, suffix='', tip=''):
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    box.setToolTip(tip)
    box.setKeyboardTracking(False)
    return box


def _ispin(minimum, maximum, value, tip=''):
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setToolTip(tip)
    return box


def _parse_numbers(text, count):
    parts = [p for p in text.replace(';', ' ').replace(',', ' ').split() if p]
    values = [float(p) for p in parts]
    if len(values) != count:
        raise ValueError('expected %d values, got %d' % (count, len(values)))
    return values


class WizardDialog(QDialog):
    def __init__(self, iface, plugin_dir, parent=None):
        super().__init__(parent or (iface.mainWindow() if iface else None))
        self.iface = iface
        self.plugin_dir = plugin_dir
        self.wf = TreeWorkflow()
        self.cloud = None # Readers.PointCloud as loaded (before ROI / filters)
        self.crs_authid = ''
        self.ascii_options = None
        self.roi_ring = None
        self.root_system = None
        self.tree = None # Last built TreeData
        self._task = None
        self._tool = None
        self._last_pick = None
        self._states = {key: 'todo' for key in STEP_KEYS}
        self.setWindowTitle(self.tr('las2ENVImet – LiDAR tree to ENVI-met'))
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(980, 720)
        self._build()
        self._refresh_dependencies()
        self._update_states()
        self.log(self.tr('Ready. Start with step 1: load a point cloud.'))

    # infrastructure
    def tr(self, text):
        return QCoreApplication.translate('WizardDialog', text)

    def log(self, message, level='info'):
        prefix = {'info': '', 'ok': '✔ ', 'warn': '⚠ ', 'error': '✖ '}[level]
        self.log_panel.appendPlainText(prefix + message)
        if self.iface is not None and level in ('warn', 'error'):
            qgis_level = MSG_WARNING if level == 'warn' else MSG_CRITICAL
            self.iface.messageBar().pushMessage('las2ENVImet', message.splitlines()[0], level=qgis_level, duration=8)

    def fail(self, message, details=None):
        # Readable message for the user; technical details only go to the log panel.
        self.log(message, 'error')
        if details:
            self.log_panel.appendPlainText(details)

    def guarded(self, function):
        # Wrap a slot: unexpected errors end up in the log panel instead of a Python traceback dialog.
        def wrapper(*_args):
            try:
                return function()
            except (ValueError, readers.ReaderError, ExportError) as exc:
                self.fail(str(exc))
            except Exception:  # noqa: BLE001
                self.fail(self.tr('Unexpected error – details below. Please report this.'), traceback.format_exc())
        return wrapper

    def canvas(self):
        return self.iface.mapCanvas() if self.iface is not None else None

    def map_crs(self):
        if self.crs_authid:
            return self.crs_authid
        canvas = self.canvas()
        return canvas.mapSettings().destinationCrs().authid() if canvas is not None else ''

    # layout
    def _build(self):
        outer = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        self.step_list = QListWidget()
        self.step_list.setFixedWidth(210)
        self.step_list.currentRowChanged.connect(self._goto)
        top_layout.addWidget(self.step_list)
        self.stack = QStackedWidget()
        top_layout.addWidget(self.stack, 1)
        splitter.addWidget(top)

        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumBlockCount(2000)
        self.log_panel.setToolTip(self.tr('Log: every step, warning and error is listed here.'))
        splitter.addWidget(self.log_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        self.btn_cancel_task = QPushButton(self.tr('Cancel'))
        self.btn_cancel_task.setVisible(False)
        self.btn_cancel_task.clicked.connect(self._cancel_task)
        row = QHBoxLayout()
        row.addWidget(self.progress, 1)
        row.addWidget(self.btn_cancel_task)
        outer.addLayout(row)

        buttons = QHBoxLayout()
        self.btn_load_project = QPushButton(self.tr('Open project…'))
        self.btn_load_project.setToolTip(self.tr('Load all settings from a las2ENVImet project file (*.l2e.json).'))
        self.btn_load_project.clicked.connect(self.guarded(self.on_load_project))
        self.btn_save_project = QPushButton(self.tr('Save project…'))
        self.btn_save_project.setToolTip(self.tr('Save all settings to a project file (*.l2e.json) for reproducibility.'))
        self.btn_save_project.clicked.connect(self.guarded(self.on_save_project))
        self.btn_help = QPushButton(self.tr('Help'))
        self.btn_help.setToolTip(self.tr('Open the README section for the current step.'))
        self.btn_help.clicked.connect(self.on_help)
        self.btn_back = QPushButton(self.tr('< Back'))
        self.btn_back.clicked.connect(lambda: self.step_list.setCurrentRow(max(0, self.step_list.currentRow() - 1)))
        self.btn_next = QPushButton(self.tr('Next >'))
        self.btn_next.clicked.connect(
            lambda: self.step_list.setCurrentRow(min(len(STEP_KEYS) - 1, self.step_list.currentRow() + 1)))
        self.btn_close = QPushButton(self.tr('Close'))
        self.btn_close.clicked.connect(self.close)
        for widget in (self.btn_load_project, self.btn_save_project, self.btn_help):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        for widget in (self.btn_back, self.btn_next, self.btn_close):
            buttons.addWidget(widget)
        outer.addLayout(buttons)

        titles = {
            'load': self.tr('1  Load point cloud'), 'select': self.tr('2  Select tree'),
            'filter': self.tr('3  Filter'), 'transform': self.tr('4  Trunk and transform'),
            'voxel': self.tr('5  Voxelise'), 'foliage': self.tr('6  Foliage (LAD / LAI)'),
            'roots': self.tr('7  Roots'), 'physiology': self.tr('8  Physiology'),
            'export': self.tr('9  Export'),
        }
        self._titles = titles
        builders = {'load': self._page_load, 'select': self._page_select, 'filter': self._page_filter,
                    'transform': self._page_transform, 'voxel': self._page_voxel, 'foliage': self._page_foliage,
                    'roots': self._page_roots, 'physiology': self._page_physiology, 'export': self._page_export}
        for key in STEP_KEYS:
            self.step_list.addItem(QListWidgetItem(titles[key]))
            page = QWidget()
            builders[key](page)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        self.step_list.setCurrentRow(0)

    def _goto(self, row):
        if row < 0:
            return
        self.stack.setCurrentIndex(row)
        self.btn_back.setEnabled(row > 0)
        self.btn_next.setEnabled(row < len(STEP_KEYS) - 1)
        key = STEP_KEYS[row]
        if key == 'roots':
            self.update_roots()
        elif key == 'export':
            self.on_database_changed()
        elif key == 'foliage':
            self.update_metrics()

    def _set_state(self, key, state):
        self._states[key] = state
        self._update_states()

    def _update_states(self):
        marks = {'todo': '○', 'ok': '✔', 'warn': '⚠'}
        colors = {'todo': QColor(90, 90, 90), 'ok': QColor(0, 120, 0), 'warn': QColor(190, 110, 0)}
        for row, key in enumerate(STEP_KEYS):
            item = self.step_list.item(row)
            item.setText('%s  %s' % (marks[self._states[key]], self._titles[key]))
            item.setForeground(colors[self._states[key]])

    def step_state(self, key):
        return self._states[key]

    # step 1: load
    def _page_load(self, page):
        layout = QVBoxLayout(page)
        box = QGroupBox(self.tr('Source'))
        form = QFormLayout(box)
        self.layer_combo = QComboBox()
        self.layer_combo.setToolTip(self.tr('Point cloud layers already loaded in QGIS. Their coordinate system is taken over.'))
        self.btn_refresh_layers = QPushButton(self.tr('Refresh'))
        self.btn_refresh_layers.clicked.connect(self.refresh_layers)
        row = QHBoxLayout()
        row.addWidget(self.layer_combo, 1)
        row.addWidget(self.btn_refresh_layers)
        form.addRow(self.tr('QGIS layer:'), row)
        self.file_source = QgsFileWidget()
        self.file_source.setStorageMode(QgsFileWidget.StorageMode.GetFile)
        self.file_source.setFilter(readers.file_filter())
        self.file_source.setToolTip(self.tr('Point cloud file: LAS, LAZ, COPC, E57, PLY, PCD or text (XYZ, TXT, CSV, PTS).'))
        self.file_source.fileChanged.connect(self._on_file_changed)
        form.addRow(self.tr('or file:'), self.file_source)
        self.btn_ascii = QPushButton(self.tr('Text file columns…'))
        self.btn_ascii.setToolTip(self.tr('Check or change delimiter, header lines and column assignment of a text file.'))
        self.btn_ascii.setEnabled(False)
        self.btn_ascii.clicked.connect(self.guarded(self.on_ascii_mapping))
        form.addRow('', self.btn_ascii)
        layout.addWidget(box)

        box = QGroupBox(self.tr('Options for large files'))
        form = QFormLayout(box)
        self.chk_extent = QCheckBox(self.tr('Load only the current map extent'))
        self.chk_extent.setToolTip(self.tr('Reads only points inside the visible map area (bounding box). Unit: map units.'))
        form.addRow(self.chk_extent)
        self.chk_classes = QCheckBox(self.tr('Only classes:'))
        self.chk_classes.setToolTip(self.tr('Keep only points with these classification codes (ASPRS: 3, 4, 5 = vegetation).'))
        self.edit_classes = QLineEdit('3,4,5')
        self.edit_classes.setToolTip(self.tr('Comma-separated class codes, e.g. 3,4,5.'))
        row = QHBoxLayout()
        row.addWidget(self.chk_classes)
        row.addWidget(self.edit_classes, 1)
        form.addRow(row)
        self.chk_thin = QCheckBox(self.tr('Thin to one point per cube of'))
        self.chk_thin.setToolTip(self.tr('Voxel-grid thinning while loading: keeps one point per cube. Reduces memory and time.'))
        self.spin_thin = _dspin(0.01, 1.0, 0.05, 2, 0.01, ' m', self.tr('Edge length of the thinning cube in metres.'))
        row = QHBoxLayout()
        row.addWidget(self.chk_thin)
        row.addWidget(self.spin_thin)
        row.addStretch(1)
        form.addRow(row)
        layout.addWidget(box)

        self.btn_load = QPushButton(self.tr('Load point cloud'))
        self.btn_load.clicked.connect(self.guarded(self.on_load))
        layout.addWidget(self.btn_load)
        self.lbl_cloud = QLabel(self.tr('No point cloud loaded.'))
        self.lbl_cloud.setWordWrap(True)
        layout.addWidget(self.lbl_cloud)

        box = QGroupBox(self.tr('Formats and Python packages'))
        v = QVBoxLayout(box)
        self.lbl_deps = QLabel()
        self.lbl_deps.setWordWrap(True)
        self.lbl_deps.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.lbl_deps)
        self.btn_copy_pip = QPushButton(self.tr('Copy installation command'))
        self.btn_copy_pip.setToolTip(self.tr('Copies the pip command for the missing packages. Run it in the OSGeo4W Shell, then restart QGIS.'))
        self.btn_copy_pip.clicked.connect(self.on_copy_pip)
        v.addWidget(self.btn_copy_pip)
        layout.addWidget(box)
        layout.addStretch(1)
        self.refresh_layers()

    def refresh_layers(self):
        self.layer_combo.clear()
        self.layer_combo.addItem(self.tr('(none – use file below)'), '')
        for layer in QgsProject.instance().mapLayers().values():
            if is_point_cloud_layer(layer):
                self.layer_combo.addItem(layer.name(), layer.id())

    def _refresh_dependencies(self):
        lines, missing = [], []
        for entry in readers.dependency_report():
            if entry['available'] and not entry.get('partial'):
                lines.append('✔ %s' % entry['format'])
            else:
                lines.append('✖ %s – %s' % (entry['format'], entry['hint']))
                if 'pip install' in entry['hint']:
                    missing.append(entry['hint'].split('pip install', 1)[1].strip())
        self._pip_command = 'python -m pip install ' + ' '.join(sorted(set(' '.join(missing).split()))) if missing else ''
        self.lbl_deps.setText('\n'.join(lines))
        self.btn_copy_pip.setVisible(bool(self._pip_command))
        self.file_source.setFilter(readers.file_filter(only_available=True))

    def on_copy_pip(self):
        QGuiApplication.clipboard().setText(self._pip_command)
        self.log(self.tr('Copied to clipboard: {}').format(self._pip_command))

    def _on_file_changed(self, path):
        reader = readers.reader_for(path) if path else None
        self.btn_ascii.setEnabled(bool(reader and reader.format_id == 'ascii'))
        self.ascii_options = None
        if path:
            self.layer_combo.setCurrentIndex(0)

    def on_ascii_mapping(self):
        dialog = AsciiMappingDialog(self.file_source.filePath(), self, self.ascii_options)
        if exec_dialog(dialog):
            self.ascii_options = dialog.options()
            self.log(self.tr('Column assignment set: {}').format(self.ascii_options['mapping']))

    def source_path(self):
        layer_id = self.layer_combo.currentData()
        if layer_id:
            layer = QgsProject.instance().mapLayer(layer_id)
            if layer is not None:
                self.crs_authid = layer.crs().authid()
                return layer.source().split('|')[0]
        return self.file_source.filePath()

    def load_options(self):
        options = dict(self.ascii_options or {})
        if self.chk_classes.isChecked():
            options['classes'] = [int(v) for v in self.edit_classes.text().replace(';', ',').split(',') if v.strip()]
        if self.chk_thin.isChecked():
            options['thin'] = self.spin_thin.value()
        if self.chk_extent.isChecked() and self.canvas() is not None:
            e = self.canvas().extent()
            options['bbox'] = (e.xMinimum(), e.yMinimum(), e.xMaximum(), e.yMaximum())
        return options

    def on_load(self):
        path = self.source_path()
        if not path:
            raise ValueError(self.tr('Please choose a point cloud layer or file first.'))
        reader = readers.reader_for(path)
        if reader is None:
            raise ValueError(self.tr('Unsupported point cloud format: {}').format(os.path.basename(path)))
        options = self.load_options()
        self.log(self.tr('Loading {} …').format(path))
        self._run_task(self.tr('las2ENVImet: loading point cloud'),
                       lambda progress, cancelled: reader.read(path, progress=progress, is_cancelled=cancelled, **options),
                       self._loaded)

    def _loaded(self, cloud):
        self.cloud = cloud
        self.roi_ring = None
        self.wf = TreeWorkflow()
        self.wf.load_points(cloud.xyz.copy(), cloud.attributes())
        self.wf.stem_point = None
        bounds = cloud.bounds()
        classes = cloud.class_counts()
        text = self.tr('{n} points loaded (file: {total}). Extent x {x0:.2f}…{x1:.2f}, y {y0:.2f}…{y1:.2f}, z {z0:.2f}…{z1:.2f} m.').format(
            n=len(cloud), total=cloud.total_points, x0=bounds[0], x1=bounds[3], y0=bounds[1], y1=bounds[4],
            z0=bounds[2], z1=bounds[5])
        if classes:
            text += ' ' + self.tr('Classes: {}').format(', '.join('%d (%d)' % kv for kv in sorted(classes.items())))
        self.lbl_cloud.setText(text)
        self.log(text, 'ok')
        self._show_points(self.tr('L2E Points'), self.wf.points)
        for key in STEP_KEYS:
            if key != 'physiology': # Presets are valid from the start
                self._states[key] = 'todo'
        self._set_state('load', 'ok')
        self._update_filter_page()
        self._update_dim_labels()

    def _show_points(self, name, points, zoom=True):
        if self.canvas() is None:
            return
        layers.remove_layers(name)
        layer = layers.add(layers.point_layer(name, self.map_crs(), points))
        if zoom:
            self.canvas().setExtent(layer.extent())
            self.canvas().refresh()

    # task handling
    def _run_task(self, description, function, on_success):
        if self._task is not None:
            raise ValueError(self.tr('Another calculation is still running.'))
        task = tasks.FunctionTask(description, function)
        self._task = task
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.btn_cancel_task.setVisible(True)
        task.progressChanged.connect(lambda value: self.progress.setValue(int(value)))

        def done():
            self._task = None
            self.progress.setVisible(False)
            self.btn_cancel_task.setVisible(False)

        def success(result):
            done()
            self.guarded(lambda: on_success(result))()

        def failed(message, expected):
            done()
            if expected:
                self.fail(message)
            else:
                self.fail(self.tr('Unexpected error – details below. Please report this.'), message)

        def cancelled():
            done()
            self.log(self.tr('Cancelled.'), 'warn')

        task.succeeded.connect(success)
        task.failed.connect(failed)
        task.cancelled.connect(cancelled)
        tasks.start(task)

    def _cancel_task(self):
        if self._task is not None:
            self._task.cancel()

    def busy(self):
        return self._task is not None

    # step 2: select tree
    def _page_select(self, page):
        layout = QVBoxLayout(page)
        info = QLabel(self.tr('If the cloud contains more than the one tree, draw a polygon around the tree '
                              '(left click = vertex, right click = finish). Otherwise continue with the next step.'))
        info.setWordWrap(True)
        layout.addWidget(info)
        self.btn_roi = QPushButton(self.tr('Draw polygon around the tree'))
        self.btn_roi.setToolTip(self.tr('Keeps only the points inside the polygon. All later steps are reset.'))
        self.btn_roi.clicked.connect(self.guarded(self.on_draw_roi))
        layout.addWidget(self.btn_roi)
        self.btn_roi_reset = QPushButton(self.tr('Use the whole cloud again'))
        self.btn_roi_reset.clicked.connect(self.guarded(self.on_reset_selection))
        layout.addWidget(self.btn_roi_reset)
        self.lbl_selection = QLabel('')
        self.lbl_selection.setWordWrap(True)
        layout.addWidget(self.lbl_selection)
        layout.addStretch(1)

    def _require_points(self):
        if self.wf.points is None:
            raise ValueError(self.tr('Please load a point cloud first (step 1).'))

    def on_draw_roi(self):
        self._require_points()
        if self.canvas() is None:
            return
        self._tool = RoiPolygonTool(self.canvas(), self.guarded_arg(self.apply_roi))
        self.canvas().setMapTool(self._tool)
        self.showMinimized()
        self.log(self.tr('Draw the polygon in the map. Right click finishes, Esc cancels.'))

    def guarded_arg(self, function):
        def wrapper(value):
            self.showNormal()
            self.raise_()
            if self.canvas() is not None and self._tool is not None:
                self.canvas().unsetMapTool(self._tool)
            self.guarded(lambda: function(value))()
        return wrapper

    def apply_roi(self, ring):
        # Keep the points inside ring [(x, y), ...].
        self._require_points()
        if not ring:
            self.log(self.tr('Polygon cancelled or invalid.'), 'warn')
            return
        before = len(self.wf.points)
        if not self.wf.filter_points_by_ring(ring):
            raise ValueError(self.tr('No points inside the polygon.'))
        self.roi_ring = [list(p) for p in ring]
        text = self.tr('{n} of {m} points selected.').format(n=len(self.wf.points), m=before)
        self.lbl_selection.setText(text)
        self.log(text, 'ok')
        self._show_points(self.tr('L2E Points'), self.wf.points)
        self._reset_from('select')
        self._set_state('select', 'ok')
        self._update_filter_page()

    def on_reset_selection(self):
        if self.cloud is None:
            return
        self.wf = TreeWorkflow()
        self.wf.load_points(self.cloud.xyz.copy(), self.cloud.attributes())
        self.wf.stem_point = None
        self.roi_ring = None
        self.lbl_selection.setText(self.tr('Whole cloud in use ({} points).').format(len(self.wf.points)))
        self._show_points(self.tr('L2E Points'), self.wf.points)
        self._reset_from('select')
        self._update_filter_page()

    def _reset_from(self, key):
        # Mark this and all later steps as not done (their results depend on earlier ones).
        started = False
        for k in STEP_KEYS:
            started = started or k == key
            if started and k not in ('roots', 'physiology'):
                self._states[k] = 'todo'
        self._update_states()
        self.tree = None
        if hasattr(self, 'btn_export'):
            self.lbl_metrics.setText('')
            self._update_dim_labels()

    # step 3: filter
    def _page_filter(self, page):
        layout = QVBoxLayout(page)
        box = QGroupBox(self.tr('Classification'))
        form = QFormLayout(box)
        self.lbl_classes = QLabel(self.tr('No classification in this cloud.'))
        self.lbl_classes.setWordWrap(True)
        form.addRow(self.lbl_classes)
        self.edit_filter_classes = QLineEdit('3,4,5')
        self.edit_filter_classes.setToolTip(self.tr('Classes to keep, comma separated (ASPRS: 2 ground, 3–5 vegetation, 6 building).'))
        form.addRow(self.tr('Keep classes:'), self.edit_filter_classes)
        self.btn_class_filter = QPushButton(self.tr('Apply class filter'))
        self.btn_class_filter.clicked.connect(self.guarded(self.on_class_filter))
        form.addRow(self.btn_class_filter)
        layout.addWidget(box)

        box = QGroupBox(self.tr('Height range'))
        form = QFormLayout(box)
        self.spin_z_low = _dspin(0, 100, 0, 1, 1, ' %', self.tr('Lower limit in percent of the height range of the cloud (0 % = lowest point).'))
        self.spin_z_up = _dspin(0, 100, 100, 1, 1, ' %', self.tr('Upper limit in percent of the height range of the cloud (100 % = highest point).'))
        form.addRow(self.tr('From:'), self.spin_z_low)
        form.addRow(self.tr('To:'), self.spin_z_up)
        self.btn_z_preview = QPushButton(self.tr('Show this height range in the map'))
        self.btn_z_preview.setToolTip(self.tr('Preview only – a thin slice near the ground makes the trunk easy to find.'))
        self.btn_z_preview.clicked.connect(self.guarded(self.on_z_preview))
        form.addRow(self.btn_z_preview)
        self.btn_z_apply = QPushButton(self.tr('Remove points outside this height range'))
        self.btn_z_apply.setToolTip(self.tr('Permanently removes the points outside the range (until the cloud is reloaded or reset).'))
        self.btn_z_apply.clicked.connect(self.guarded(self.on_z_apply))
        form.addRow(self.btn_z_apply)
        layout.addWidget(box)

        box = QGroupBox(self.tr('Thinning'))
        form = QFormLayout(box)
        self.spin_filter_thin = _dspin(0.01, 1.0, 0.05, 2, 0.01, ' m', self.tr('Keep one point per cube with this edge length (metres).'))
        form.addRow(self.tr('Cube size:'), self.spin_filter_thin)
        self.btn_thin = QPushButton(self.tr('Thin the cloud'))
        self.btn_thin.clicked.connect(self.guarded(self.on_thin))
        form.addRow(self.btn_thin)
        layout.addWidget(box)
        self.lbl_filter = QLabel('')
        self.lbl_filter.setWordWrap(True)
        layout.addWidget(self.lbl_filter)
        layout.addStretch(1)

    def _update_filter_page(self):
        cls = self.wf.attributes.get('classification') if self.wf.points is not None else None
        if cls is None:
            self.lbl_classes.setText(self.tr('No classification in this cloud.'))
        else:
            values, counts = np.unique(cls, return_counts=True)
            self.lbl_classes.setText(self.tr('Classes in the cloud: {}').format(
                ', '.join('%d (%d)' % (int(v), int(c)) for v, c in zip(values, counts))))
        self.btn_class_filter.setEnabled(cls is not None)
        self.edit_filter_classes.setEnabled(cls is not None)

    def _filtered(self, mask, what):
        if not np.any(mask):
            raise ValueError(self.tr('This filter would remove all points.'))
        before = len(self.wf.points)
        self.wf.apply_mask(mask)
        text = self.tr('{what}: {n} of {m} points kept.').format(what=what, n=len(self.wf.points), m=before)
        self.lbl_filter.setText(text)
        self.log(text, 'ok')
        self._show_points(self.tr('L2E Points'), self.wf.points, zoom=False)
        self._reset_from('filter')
        self._set_state('filter', 'ok')
        self._update_filter_page()

    def on_class_filter(self):
        self._require_points()
        cls = self.wf.attributes.get('classification')
        if cls is None:
            raise ValueError(self.tr('This cloud has no classification.'))
        keep = [int(v) for v in self.edit_filter_classes.text().replace(';', ',').split(',') if v.strip()]
        self._filtered(np.isin(cls, keep), self.tr('Class filter'))

    def on_z_preview(self):
        self._require_points()
        mask = z_percent_mask(self.wf.points, self.spin_z_low.value(), self.spin_z_up.value())
        self._show_points(self.tr('L2E Height range'), self.wf.points[mask], zoom=False)
        self.log(self.tr('Height range shown: {} points.').format(int(mask.sum())))

    def on_z_apply(self):
        self._require_points()
        self._filtered(z_percent_mask(self.wf.points, self.spin_z_low.value(), self.spin_z_up.value()),
                       self.tr('Height filter'))

    def on_thin(self):
        self._require_points()
        cell = self.spin_filter_thin.value()
        keys = np.floor((self.wf.points - self.wf.points.min(axis=0)) / cell).astype(np.int64)
        dims = keys.max(axis=0) + 1
        flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
        _, first = np.unique(flat, return_index=True)
        mask = np.zeros(len(self.wf.points), dtype=bool)
        mask[first] = True
        self._filtered(mask, self.tr('Thinning'))

    # step 4: trunk and transform
    def _page_transform(self, page):
        layout = QVBoxLayout(page)
        box = QGroupBox(self.tr('Trunk position'))
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        self.btn_pick_stem = QPushButton(self.tr('Click the trunk in the map'))
        self.btn_pick_stem.setToolTip(self.tr('Click the centre of the trunk in the map. It becomes the centre of the plant grid.'))
        self.btn_pick_stem.clicked.connect(self.guarded(self.on_pick_stem))
        self.btn_suggest_stem = QPushButton(self.tr('Suggest trunk position'))
        self.btn_suggest_stem.setToolTip(self.tr('Uses the median position of the lowest 30 cm of the cloud. Check the result in the map.'))
        self.btn_suggest_stem.clicked.connect(self.guarded(self.on_suggest_stem))
        row.addWidget(self.btn_pick_stem)
        row.addWidget(self.btn_suggest_stem)
        v.addLayout(row)
        self.lbl_stem = QLabel(self.tr('No trunk position yet.'))
        v.addWidget(self.lbl_stem)
        layout.addWidget(box)

        box = QGroupBox(self.tr('Dimensions (measured → target)'))
        grid = QGridLayout(box)
        self.lbl_dims = QLabel('–')
        grid.addWidget(self.lbl_dims, 0, 0, 1, 2)
        self.spin_height = _dspin(0.1, 200, 10, 2, 0.1, ' m', self.tr('Target tree height in metres. The cloud is scaled vertically to this height.'))
        self.spin_width_x = _dspin(0.1, 200, 5, 2, 0.1, ' m', self.tr('Target crown width in x direction (east–west) in metres.'))
        self.spin_width_y = _dspin(0.1, 200, 5, 2, 0.1, ' m', self.tr('Target crown width in y direction (north–south) in metres.'))
        self.spin_rotation = _dspin(-360, 360, 0, 1, 5, ' °', self.tr('Rotation around the trunk in degrees, clockwise.'))
        for n, (label, widget) in enumerate([(self.tr('Height:'), self.spin_height), (self.tr('Width x:'), self.spin_width_x),
                                             (self.tr('Width y:'), self.spin_width_y), (self.tr('Rotation:'), self.spin_rotation)]):
            grid.addWidget(QLabel(label), n + 1, 0)
            grid.addWidget(widget, n + 1, 1)
        row = QHBoxLayout()
        self.btn_preview = QPushButton(self.tr('Preview in the map'))
        self.btn_preview.clicked.connect(self.guarded(self.on_preview_transform))
        self.btn_reset_transform = QPushButton(self.tr('Reset to measured values'))
        self.btn_reset_transform.clicked.connect(self.guarded(self.on_reset_transform))
        row.addWidget(self.btn_preview)
        row.addWidget(self.btn_reset_transform)
        grid.addLayout(row, 5, 0, 1, 2)
        layout.addWidget(box)
        layout.addStretch(1)

    def on_pick_stem(self):
        self._require_points()
        if self.canvas() is None:
            return
        self._tool = PointTool(self.canvas(), self.guarded_arg(lambda p: self.set_stem(p.x(), p.y())))
        self.canvas().setMapTool(self._tool)
        self.showMinimized()
        self.log(self.tr('Click the trunk centre in the map.'))

    def on_suggest_stem(self):
        self._require_points()
        z = self.wf.points[:, 2]
        low = self.wf.points[z < z.min() + 0.3]
        self.set_stem(float(np.median(low[:, 0])), float(np.median(low[:, 1])))

    def set_stem(self, x, y):
        self._require_points()
        self.wf.reset_derived(keep_stem=True)
        self.wf.set_stem(x, y)
        d = self.wf.current_dims
        if d['height'] <= 0 or d['total_width_x'] <= 0 or d['total_width_y'] <= 0:
            raise ValueError(self.tr('The cloud has no extent in at least one direction; cannot continue.'))
        self.lbl_stem.setText(self.tr('Trunk at x = {x:.2f}, y = {y:.2f}').format(x=x, y=y))
        for box, key in ((self.spin_height, 'height'), (self.spin_width_x, 'total_width_x'), (self.spin_width_y, 'total_width_y')):
            box.setValue(d[key])
        self.spin_rotation.setValue(0.0)
        self._update_dim_labels()
        self._show_points(self.tr('L2E Trunk'), np.array([[x, y, 0.0]]), zoom=False)
        self._reset_from('voxel')
        self._set_state('transform', 'ok')
        self.log(self.tr('Trunk set. Measured: height {h:.2f} m, width {wx:.2f} × {wy:.2f} m.').format(
            h=d['height'], wx=d['total_width_x'], wy=d['total_width_y']), 'ok')

    def _update_dim_labels(self):
        d = self.wf.current_dims
        self.lbl_dims.setText('–' if not d else self.tr('Measured: height {h:.2f} m, width x {wx:.2f} m, width y {wy:.2f} m').format(
            h=d['height'], wx=d['total_width_x'], wy=d['total_width_y']))

    def _require_stem(self):
        self._require_points()
        if self.wf.current_dims is None:
            raise ValueError(self.tr('Please set the trunk position first (step 4).'))

    def target_dims(self):
        return {'height': self.spin_height.value(), 'total_width_x': self.spin_width_x.value(),
                'total_width_y': self.spin_width_y.value()}

    def on_preview_transform(self):
        self._require_stem()
        points = self.wf.transform_points(self.target_dims(), self.spin_rotation.value())
        self._show_points(self.tr('L2E Transformed'), points, zoom=False)
        self.log(self.tr('Preview of the scaled / rotated cloud shown.'))

    def on_reset_transform(self):
        self._require_stem()
        d = self.wf.current_dims
        self.spin_height.setValue(d['height'])
        self.spin_width_x.setValue(d['total_width_x'])
        self.spin_width_y.setValue(d['total_width_y'])
        self.spin_rotation.setValue(0.0)
        layers.remove_layers(self.tr('L2E Transformed'))

    # step 5: voxelise
    def _page_voxel(self, page):
        layout = QVBoxLayout(page)
        info = QLabel(self.tr('The tree is converted to 1 m × 1 m × 1 m cells (the fixed cell size of ENVI-met 3D plants). '
                              'The trunk is the centre of the grid.'))
        info.setWordWrap(True)
        layout.addWidget(info)
        self.btn_voxelise = QPushButton(self.tr('Create voxels'))
        self.btn_voxelise.clicked.connect(self.guarded(self.on_voxelise))
        layout.addWidget(self.btn_voxelise)
        self.lbl_voxels = QLabel('')
        self.lbl_voxels.setWordWrap(True)
        layout.addWidget(self.lbl_voxels)
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr('Show layer (height above ground in m):')))
        self.spin_z_level = _ispin(0, 0, 0, self.tr('Horizontal slice shown in the map; labels are point counts or LAD.'))
        self.spin_z_level.valueChanged.connect(self.guarded_arg(self._layer_changed))
        row.addWidget(self.spin_z_level)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)

    def on_voxelise(self):
        self._require_stem()
        target, angle = self.target_dims(), self.spin_rotation.value()
        wf = self.wf

        def work(progress, cancelled):
            wf.transform_points(target, angle)
            progress(0.4)
            if cancelled():
                raise readers.Cancelled()
            result = wf.create_voxels(1.0)
            progress(1.0)
            return result

        self._run_task(self.tr('las2ENVImet: voxelising'), work, self._voxelised)

    def _voxelised(self, _result):
        corners = self.wf.voxel_corners
        z_max = int(np.max(corners[:, 2]))
        for box in self._layer_boxes():
            box.blockSignals(True)
            box.setRange(0, z_max)
            box.setValue(min(z_max, max(0, z_max // 2)))
            box.blockSignals(False)
        for box in (self.spin_crown_z_min, self.spin_crown_z_max, self.spin_trunk_height):
            box.setRange(0, z_max)
        self.spin_crown_z_max.setValue(z_max)
        self.list_refs.clear()
        text = self.tr('{n} voxels with points, {z} layers (tree height {z} m), {p} points.').format(
            n=len(corners), z=z_max + 1, p=int(self.wf.point_counts.sum()))
        self.lbl_voxels.setText(text)
        self.log(text, 'ok')
        layers.remove_layers(self.tr('L2E Transformed'))
        self._reset_from('foliage')
        self._set_state('voxel', 'ok')
        self.show_slice()

    def _require_voxels(self):
        if self.wf.voxel_corners is None:
            raise ValueError(self.tr('Please create the voxels first (step 5).'))

    def _layer_boxes(self):
        # The shown layer can be chosen in step 5, in step 6 and with the reference voxel layer
        return (self.spin_z_level, self.spin_z_level_foliage, self.spin_ref_z)

    def _layer_changed(self, value):
        for box in self._layer_boxes():
            if box.value() != value:
                box.blockSignals(True)
                box.setValue(value)
                box.blockSignals(False)
        self.show_slice()

    def show_slice(self):
        if self.wf.voxel_corners is None or self.canvas() is None:
            return
        z = self.spin_z_level.value()
        mask, corners, counts = self.wf.get_voxels_at_z(z)
        layers.remove_layers(self.tr('L2E Voxels'))
        if mask is None or not np.any(mask):
            return
        lad = self.wf.lad_values[mask] if self.wf.lad_values is not None else None
        layer = layers.voxel_layer(self.tr('L2E Voxels') + ' z=%d' % z, self.map_crs(), corners, counts, lad)
        if lad is not None:
            layers.style_graduated(layer, 'LAD', QColor(240, 255, 240), QColor(0, 80, 0))
            layers.enable_labels(layer, 'LAD', 2)
        else:
            layers.style_graduated(layer, 'points', QColor(240, 240, 255), QColor(0, 0, 120))
            layers.enable_labels(layer, 'points', 0)
        layers.add(layer)

    # step 6: foliage
    def _page_foliage(self, page):
        layout = QVBoxLayout(page)
        box = QGroupBox(self.tr('Control foliage by'))
        v = QVBoxLayout(box)
        self.radio_reference = QRadioButton(self.tr('LAD reference voxels'))
        self.radio_reference.setToolTip(self.tr('You give the LAD of one or more voxels; all others are scaled by their point count.'))
        self.radio_lai = QRadioButton(self.tr('Target LAI'))
        self.radio_lai.setToolTip(self.tr('The relative distribution from the point cloud is kept and scaled to the leaf area index you enter.'))
        self.radio_area = QRadioButton(self.tr('Total leaf area'))
        self.radio_area.setToolTip(self.tr('The relative distribution from the point cloud is kept and scaled to a total one-sided leaf area in m².'))
        self.radio_lai.setChecked(True)
        for radio in (self.radio_reference, self.radio_lai, self.radio_area):
            v.addWidget(radio)
            radio.toggled.connect(self._foliage_mode_changed)
        layout.addWidget(box)

        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr('Show layer (height above ground in m):')))
        self.spin_z_level_foliage = _ispin(0, 0, 0, self.tr('Horizontal slice shown in the map; labels are point counts or LAD.'))
        self.spin_z_level_foliage.valueChanged.connect(self.guarded_arg(self._layer_changed))
        row.addWidget(self.spin_z_level_foliage)
        row.addStretch(1)
        layout.addLayout(row)

        self.box_reference = QGroupBox(self.tr('Reference voxels'))
        g = QGridLayout(self.box_reference)
        self.btn_pick_ref = QPushButton(self.tr('Click a voxel in the map'))
        self.btn_pick_ref.setToolTip(self.tr('Pick a voxel of the displayed layer whose leaf area density you know.'))
        self.btn_pick_ref.clicked.connect(self.guarded(self.on_pick_reference))
        self.spin_ref_z = _ispin(0, 0, 0, self.tr('Height layer of the reference voxel (m above ground). The map shows this layer.'))
        self.spin_ref_z.valueChanged.connect(self.guarded_arg(self._layer_changed))
        self.spin_ref_lad = _dspin(0.01, 20, 1.0, 2, 0.1, ' m²/m³', self.tr('Known leaf area density of the picked voxel in m² leaf area per m³.'))
        self.btn_add_ref = QPushButton(self.tr('Add reference'))
        self.btn_add_ref.clicked.connect(self.guarded(self.on_add_reference))
        self.btn_remove_ref = QPushButton(self.tr('Remove selected'))
        self.btn_remove_ref.clicked.connect(self.guarded(self.on_remove_reference))
        self.list_refs = QListWidget()
        self.list_refs.setMaximumHeight(90)
        g.addWidget(self.btn_pick_ref, 0, 0, 1, 2)
        g.addWidget(QLabel(self.tr('Layer:')), 1, 0)
        g.addWidget(self.spin_ref_z, 1, 1)
        g.addWidget(QLabel(self.tr('LAD:')), 2, 0)
        g.addWidget(self.spin_ref_lad, 2, 1)
        g.addWidget(self.btn_add_ref, 3, 0)
        g.addWidget(self.btn_remove_ref, 3, 1)
        g.addWidget(self.list_refs, 4, 0, 1, 2)
        layout.addWidget(self.box_reference)

        self.box_lai = QGroupBox(self.tr('Target'))
        form = QFormLayout(self.box_lai)
        self.spin_lai = _dspin(0.1, 20, 4.0, 2, 0.1, ' m²/m²', self.tr('Leaf area index: one-sided leaf area per ground area.') + '\n' +
                               self.tr('Orientation (maximum summer LAI of urban trees, Moser-Reischl et al. 2025):') + '\n' +
                               presets.lai_reference_text())
        form.addRow(self.tr('Target LAI:'), self.spin_lai)
        self.combo_lai_ref = QComboBox()
        self.combo_lai_ref.addItem(self.tr('Crown projection area (total leaf area / crown area)'), foliage.LAI_CROWN)
        self.combo_lai_ref.addItem(self.tr('Every column (each x/y column reaches the target)'), foliage.LAI_COLUMN)
        self.combo_lai_ref.setToolTip(self.tr('Reference area of the LAI. Default: the whole crown projection. '
                                              '"Every column" evens out the horizontal distribution.'))
        form.addRow(self.tr('LAI refers to:'), self.combo_lai_ref)
        self.spin_leaf_area = _dspin(0.1, 100000, 100.0, 1, 5, ' m²', self.tr('Total one-sided leaf area of the tree in m².'))
        form.addRow(self.tr('Total leaf area:'), self.spin_leaf_area)
        layout.addWidget(self.box_lai)

        self.btn_calc_lad = QPushButton(self.tr('Calculate LAD'))
        self.btn_calc_lad.clicked.connect(self.guarded(self.on_calculate_lad))
        layout.addWidget(self.btn_calc_lad)

        box = QGroupBox(self.tr('Refinement (optional)'))
        form = QFormLayout(box)
        self.spin_crown_factor = _dspin(0.0, 10, 1.0, 2, 0.1, '', self.tr('Factor applied to the LAD of all voxels between the two layers below (1 = unchanged).'))
        self.spin_crown_z_min = _ispin(0, 0, 0, self.tr('Lowest layer the crown factor applies to (m above ground).'))
        self.spin_crown_z_max = _ispin(0, 0, 0, self.tr('Highest layer the crown factor applies to (m above ground).'))
        form.addRow(self.tr('Crown factor:'), self.spin_crown_factor)
        row = QHBoxLayout()
        row.addWidget(self.spin_crown_z_min)
        row.addWidget(QLabel('–'))
        row.addWidget(self.spin_crown_z_max)
        form.addRow(self.tr('From / to layer:'), row)
        self.chk_trunk = QCheckBox(self.tr('Set trunk voxels to a fixed LAD'))
        self.chk_trunk.setToolTip(self.tr('Gives the voxel column of the trunk a fixed (usually high) density up to the given height.'))
        self.spin_trunk_height = _ispin(0, 0, 0, self.tr('Trunk voxels up to this layer are set (m above ground).'))
        self.spin_trunk_lad = _dspin(0.0, 20, 10.0, 2, 0.5, ' m²/m³', self.tr('LAD assigned to the trunk voxels.'))
        form.addRow(self.chk_trunk)
        form.addRow(self.tr('Trunk up to layer:'), self.spin_trunk_height)
        form.addRow(self.tr('Trunk LAD:'), self.spin_trunk_lad)
        self.spin_threshold = _dspin(0.0, 5, 0.0, 3, 0.01, ' m²/m³', self.tr('Voxels with a smaller LAD are emptied (removes scattered points).'))
        form.addRow(self.tr('Remove LAD below:'), self.spin_threshold)
        self.chk_clear_ground = QCheckBox(self.tr('Empty the ground layer except the trunk voxel'))
        self.chk_clear_ground.setToolTip(self.tr('Removes ground points and low vegetation in the lowest metre; the trunk voxel stays.'))
        form.addRow(self.chk_clear_ground)
        row = QHBoxLayout()
        self.btn_refine = QPushButton(self.tr('Apply refinement'))
        self.btn_refine.clicked.connect(self.guarded(self.on_refine))
        self.btn_refine_reset = QPushButton(self.tr('Undo refinement'))
        self.btn_refine_reset.clicked.connect(self.guarded(self.on_reset_refinement))
        row.addWidget(self.btn_refine)
        row.addWidget(self.btn_refine_reset)
        form.addRow(row)
        layout.addWidget(box)

        box = QGroupBox(self.tr('Result'))
        form = QFormLayout(box)
        self.spin_lad_max = _dspin(0.1, 50, foliage.DEFAULT_LAD_MAX, 1, 0.5, ' m²/m³',
                                   self.tr('Warning limit: voxels with a higher LAD are reported as implausible. No value is changed.'))
        self.spin_lad_max.valueChanged.connect(lambda _v: self.update_metrics())
        form.addRow(self.tr('Warn if LAD above:'), self.spin_lad_max)
        self.lbl_metrics = QLabel('')
        self.lbl_metrics.setWordWrap(True)
        self.lbl_metrics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.lbl_metrics)
        layout.addWidget(box)
        layout.addStretch(1)
        self._foliage_mode_changed()

    def foliage_mode(self):
        if self.radio_reference.isChecked():
            return foliage.MODE_REFERENCE
        return foliage.MODE_TARGET_LAI if self.radio_lai.isChecked() else foliage.MODE_LEAF_AREA

    def set_foliage_mode(self, mode):
        {foliage.MODE_REFERENCE: self.radio_reference, foliage.MODE_TARGET_LAI: self.radio_lai,
         foliage.MODE_LEAF_AREA: self.radio_area}[mode].setChecked(True)

    def _foliage_mode_changed(self, *_args):
        mode = self.foliage_mode()
        self.box_reference.setVisible(mode == foliage.MODE_REFERENCE)
        self.box_lai.setVisible(mode != foliage.MODE_REFERENCE)
        self.spin_lai.setEnabled(mode == foliage.MODE_TARGET_LAI)
        self.combo_lai_ref.setEnabled(mode == foliage.MODE_TARGET_LAI)
        self.spin_leaf_area.setEnabled(mode == foliage.MODE_LEAF_AREA)

    def foliage_target(self):
        return self.spin_lai.value() if self.foliage_mode() == foliage.MODE_TARGET_LAI else self.spin_leaf_area.value()

    def on_pick_reference(self):
        self._require_voxels()
        if self.canvas() is None:
            return
        self.spin_ref_z.setValue(self.spin_z_level.value())
        self._tool = PointTool(self.canvas(), self.guarded_arg(self._reference_picked))
        self.canvas().setMapTool(self._tool)
        self.showMinimized()
        self.log(self.tr('Click a voxel of the displayed layer in the map.'))

    def _reference_picked(self, point):
        self._last_pick = (point.x(), point.y())
        self.log(self.tr('Voxel at x = {x:.2f}, y = {y:.2f} picked. Enter its LAD and press "Add reference".').format(
            x=point.x(), y=point.y()))

    def on_add_reference(self):
        self._require_voxels()
        if self._last_pick is None:
            raise ValueError(self.tr('Please click a voxel in the map first.'))
        self.add_reference(self._last_pick[0], self._last_pick[1], self.spin_ref_z.value(), self.spin_ref_lad.value())

    def add_reference(self, x, y, z_level, lad):
        self._require_voxels()
        ref = self.wf.add_reference_voxel(x, y, z_level, lad)
        if ref is None:
            raise ValueError(self.tr('No voxel found there, or this voxel is already a reference.'))
        self.list_refs.addItem(self.tr('LAD {lad:.2f} | {pts} points | x {x:.1f}, y {y:.1f}, layer {z:.0f}').format(
            lad=lad, pts=int(ref['point_count']), x=ref['x'], y=ref['y'], z=ref['z']))
        self.log(self.tr('Reference voxel added.'), 'ok')

    def on_remove_reference(self):
        row = self.list_refs.currentRow()
        if row >= 0 and self.wf.remove_reference_voxel(row):
            self.list_refs.takeItem(row)

    def on_calculate_lad(self):
        self._require_voxels()
        mode = self.foliage_mode()
        if mode == foliage.MODE_REFERENCE:
            if not self.wf.ref_voxels:
                raise ValueError(self.tr('Please add at least one reference voxel, or choose "Target LAI".'))
            if not self.wf.calibrate_lad():
                raise ValueError(self.tr('LAD calibration failed.'))
        else:
            self.wf.scale_foliage(mode, self.foliage_target(), self.combo_lai_ref.currentData())
        self._refined = False
        self.tree = None
        self._set_state('foliage', 'ok')
        self.log(self.tr('LAD calculated.'), 'ok')
        self.update_metrics()
        self.show_slice()

    def refinement_settings(self):
        return {'crown_factor': self.spin_crown_factor.value(), 'crown_z_min': self.spin_crown_z_min.value(),
                'crown_z_max': self.spin_crown_z_max.value(), 'trunk_enhance': self.chk_trunk.isChecked(),
                'trunk_height': self.spin_trunk_height.value(), 'trunk_lad': self.spin_trunk_lad.value(),
                'threshold': self.spin_threshold.value(), 'clear_ground': self.chk_clear_ground.isChecked()}

    def on_refine(self):
        if self.wf.base_lad is None:
            raise ValueError(self.tr('Please calculate the LAD first.'))
        self.wf.refine(self.refinement_settings())
        mode = self.foliage_mode()
        if mode != foliage.MODE_REFERENCE:
            # The target must still be met after leaves were removed
            self.wf.rescale_after_refinement(mode, self.foliage_target(), self.combo_lai_ref.currentData())
        self.tree = None
        self._refined = True
        self.log(self.tr('Refinement applied.'), 'ok')
        self.update_metrics()
        self.show_slice()

    def on_reset_refinement(self):
        if self.wf.reset_refinement():
            self.tree = None
            self.update_metrics()
            self.show_slice()
            self.log(self.tr('Refinement undone.'))

    def update_metrics(self):
        m = self.wf.metrics(self.spin_lad_max.value()) if self.wf.lad_values is not None else None
        if m is None:
            self.lbl_metrics.setText(self.tr('No LAD calculated yet.'))
            return None
        text = self.tr('LAI {lai:.2f} m²/m² · total leaf area {la:.1f} m² · crown projection {ap:.0f} m²\n'
                       'LAD max {mx:.2f} · mean {mean:.2f} m²/m³ · {cells} voxels with leaves · '
                       'column LAI max {cmax:.2f}').format(
            lai=m['lai'], la=m['total_leaf_area'], ap=m['crown_projection_area'], mx=m['lad_max'],
            mean=m['lad_mean'], cells=m['cells'], cmax=m['lai_column_max'])
        warnings = foliage.foliage_warnings(m)
        if 'lad_above_max' in warnings:
            text += '\n⚠ ' + self.tr('{n} voxels exceed the limit of {limit:.1f} m²/m³ – check the target value or the point cloud.').format(
                n=m['cells_above_max'], limit=m['lad_limit'])
        if 'no_leaves' in warnings:
            text += '\n⚠ ' + self.tr('No voxel contains leaves.')
        self.lbl_metrics.setText(text)
        self._set_state('foliage', 'warn' if warnings else 'ok')
        return m

    # step 7: roots
    def _page_roots(self, page):
        layout = QHBoxLayout(page)
        left = QVBoxLayout()
        self.chk_roots = QCheckBox(self.tr('Define the root system'))
        self.chk_roots.setToolTip(self.tr('Off: the export writes default root values: for ENVI-met 5.x a homogeneous root '
                                          'zone 10 m wide and as deep as the plant grid is wide (y), for ENVI-met 6 a 2 m '
                                          'deep homogeneous cylinder under the crown.'))
        self.chk_roots.setChecked(True)
        self.chk_roots.toggled.connect(lambda _c: self.update_roots())
        left.addWidget(self.chk_roots)
        form = QFormLayout()
        self.combo_root_preset = QComboBox()
        self.combo_root_preset.addItem(self.tr('(choose a preset)'), '')
        for name in presets.root_preset_names():
            self.combo_root_preset.addItem(name, name)
            self.combo_root_preset.setItemData(self.combo_root_preset.count() - 1, presets.ROOT_PRESETS[name]['source'],
                                               Qt.ItemDataRole.ToolTipRole)
        self.combo_root_preset.setToolTip(self.tr('Presets set some of the values below. The source of each preset is shown under the list.'))
        self.combo_root_preset.currentIndexChanged.connect(self.guarded(self.on_root_preset))
        form.addRow(self.tr('Preset:'), self.combo_root_preset)
        row = QHBoxLayout()
        self.btn_root_preset_save = QPushButton(self.tr('Save as my preset…'))
        self.btn_root_preset_save.setToolTip(self.tr('Stores the current root settings under a name of your choice, e.g. a tree species. '
                                                     'Your presets are kept in the QGIS profile.'))
        self.btn_root_preset_save.clicked.connect(self.guarded(self.on_save_root_preset))
        self.btn_root_preset_delete = QPushButton(self.tr('Delete my preset'))
        self.btn_root_preset_delete.clicked.connect(self.guarded(self.on_delete_root_preset))
        row.addWidget(self.btn_root_preset_save)
        row.addWidget(self.btn_root_preset_delete)
        form.addRow(row)
        self._fill_user_presets()
        self.lbl_root_source = QLabel('')
        self.lbl_root_source.setWordWrap(True)
        self.lbl_root_source.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.lbl_root_source)
        self.combo_root_shape = QComboBox()
        for key, label in ((roots.SHAPE_CYLINDER, self.tr('Cylinder')), (roots.SHAPE_ELLIPSOID, self.tr('Half sphere / ellipsoid')),
                           (roots.SHAPE_HEART, self.tr('Heart root system')), (roots.SHAPE_TAP, self.tr('Tap root system')),
                           (roots.SHAPE_PLATE, self.tr('Flat / plate root system')), (roots.SHAPE_CUSTOM, self.tr('User-defined radius profile'))):
            self.combo_root_shape.addItem(label, key)
        self.combo_root_shape.setToolTip(self.tr('Outline of the root zone: how the horizontal extent changes with depth.'))
        form.addRow(self.tr('Shape:'), self.combo_root_shape)
        self.spin_root_depth = _dspin(0.1, 30, 2.0, 2, 0.1, ' m', self.tr('Depth of the root zone in metres. Orientation: roots deeper than 2 m are uncommon in most soils (Crow 2005); mean maximum coarse-root depths of mature forest trees were about 1.0 m (Norway spruce) to 1.5 m (Scots pine) (Eichhorn 1992, cited in Puhe 2003).'))
        form.addRow(self.tr('Depth:'), self.spin_root_depth)
        self.combo_root_size = QComboBox()
        self.combo_root_size.addItem(self.tr('Factor × crown radius'), 'factor')
        self.combo_root_size.addItem(self.tr('Absolute diameter'), 'diameter')
        self.combo_root_size.setToolTip(self.tr('Horizontal size either relative to the crown or as a diameter in metres.'))
        form.addRow(self.tr('Horizontal size:'), self.combo_root_size)
        self.spin_root_factor = _dspin(0.1, 10, 1.0, 2, 0.1, ' ×', self.tr('Root radius = factor × crown radius (1 = roots end at the drip line). Orientation: root systems of woody plants are on average more than twice as wide as the crown (Tumber-Dávila et al. 2022); in paved urban sites the spread is often much smaller (Day et al. 2010).'))
        self.spin_root_diameter = _dspin(0.2, 100, 6.0, 2, 0.5, ' m', self.tr('Largest diameter of the root zone in metres.'))
        form.addRow(self.tr('Radius factor:'), self.spin_root_factor)
        form.addRow(self.tr('Diameter:'), self.spin_root_diameter)
        self.combo_root_dist = QComboBox()
        for key, label in ((roots.DIST_UNIFORM, self.tr('Uniform')), (roots.DIST_LINEAR, self.tr('Linearly decreasing')),
                           (roots.DIST_EXPONENTIAL, self.tr('Exponentially decreasing (β model)')),
                           (roots.DIST_LOGISTIC, self.tr('Logistic (50 % and 95 % rooting depth)')),
                           (roots.DIST_CUSTOM, self.tr('User-defined depth profile'))):
            self.combo_root_dist.addItem(label, key)
        self.combo_root_dist.setToolTip(self.tr('How the root density changes with depth.'))
        form.addRow(self.tr('Vertical distribution:'), self.combo_root_dist)
        self.spin_root_beta = _dspin(0.80, 0.999, roots.DEFAULT_BETA, 3, 0.001, '',
                                     self.tr('β of the model Y = 1 − β^d (d = depth in cm; Gale & Grigal 1987). Larger = deeper roots. '
                                             'Jackson et al. (1996): 0.966 temperate deciduous, 0.976 temperate coniferous forest.'))
        form.addRow(self.tr('β:'), self.spin_root_beta)
        self.spin_root_d50 = _dspin(1, 500, roots.DEFAULT_D50, 0, 1, ' cm', self.tr(
            'Depth above which 50 % of the roots lie (logistic model of Schenk & Jackson 2002). '
            'Cool-temperate forest: 21 cm, warm-temperate forest: 23 cm.'))
        self.spin_root_d95 = _dspin(2, 1000, roots.DEFAULT_D95, 0, 5, ' cm', self.tr(
            'Depth above which 95 % of the roots lie (logistic model of Schenk & Jackson 2002). '
            'Cool-temperate forest: 104 cm, warm-temperate forest: 121 cm.'))
        form.addRow(self.tr('50 % rooting depth:'), self.spin_root_d50)
        form.addRow(self.tr('95 % rooting depth:'), self.spin_root_d95)
        self.combo_root_density = QComboBox()
        self.combo_root_density.addItem(self.tr('Mean root area density (RAD)'), 'rad')
        self.combo_root_density.addItem(self.tr('Total root area'), 'area')
        self.combo_root_density.setToolTip(self.tr('Either the mean density of the root zone or the total root surface of the tree.'))
        form.addRow(self.tr('Amount given as:'), self.combo_root_density)
        self.spin_root_rad = _dspin(0.0, 10, 0.1, 3, 0.01, ' m²/m³', self.tr('Mean root area density of the root zone (ENVI-met default: 0.1 m²/m³; the plants shipped with ENVI-met 6 use 0.1 to 0.9 m²/m³). Measured fine-root surfaces of forest stands are far larger, so keep to the ENVI-met range unless you know how the model reacts.'))
        self.spin_root_area = _dspin(0.0, 100000, 50.0, 1, 5, ' m²', self.tr('Total root surface area of the tree in m².'))
        form.addRow(self.tr('Mean RAD:'), self.spin_root_rad)
        form.addRow(self.tr('Total root area:'), self.spin_root_area)
        self.edit_root_range = QLineEdit('1 1 1 1 1 1 1 1 1 1')
        self.edit_root_range.setToolTip(self.tr('Ten values 0…1 from top to bottom: horizontal extent of each tenth of the depth as a fraction of the radius.'))
        form.addRow(self.tr('Radius profile:'), self.edit_root_range)
        self.edit_root_profile = QLineEdit('1 1 1 1 1 1 1 1 1 1')
        self.edit_root_profile.setToolTip(self.tr('Ten relative density values from top to bottom (any unit; only the proportions count).'))
        form.addRow(self.tr('Depth profile:'), self.edit_root_profile)
        left.addLayout(form)
        self.lbl_roots = QLabel('')
        self.lbl_roots.setWordWrap(True)
        self.lbl_roots.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left.addWidget(self.lbl_roots)
        left.addStretch(1)
        holder = QWidget()
        holder.setLayout(left)
        layout.addWidget(holder, 3)
        self.side_view = SideViewWidget()
        self.side_view.setToolTip(self.tr('Side view: crown voxels (green, darker = denser) and root zone (brown, darker = denser).'))
        layout.addWidget(self.side_view, 2)
        for widget in (self.combo_root_shape, self.combo_root_size, self.combo_root_dist, self.combo_root_density):
            widget.currentIndexChanged.connect(lambda _i: self.update_roots())
        for widget in (self.spin_root_depth, self.spin_root_factor, self.spin_root_diameter, self.spin_root_beta,
                       self.spin_root_d50, self.spin_root_d95,
                       self.spin_root_rad, self.spin_root_area):
            widget.valueChanged.connect(lambda _v: self.update_roots())
        for widget in (self.edit_root_range, self.edit_root_profile):
            widget.editingFinished.connect(self.update_roots)

    # User presets live in the QGIS settings of the profile: {name: settings dict}
    USER_PRESET_KEY = 'las2envimet/root_presets'
    USER_PREFIX = 'user:'

    def user_root_presets(self):
        try:
            data = json.loads(QgsSettings().value(self.USER_PRESET_KEY, '{}') or '{}')
            return data if isinstance(data, dict) else {}
        except ValueError:
            return {}

    def _store_user_presets(self, data):
        QgsSettings().setValue(self.USER_PRESET_KEY, json.dumps(data, ensure_ascii=False))

    def _fill_user_presets(self):
        combo = self.combo_root_preset
        combo.blockSignals(True)
        for index in range(combo.count() - 1, -1, -1):
            if str(combo.itemData(index) or '').startswith(self.USER_PREFIX):
                combo.removeItem(index)
        for name in sorted(self.user_root_presets()):
            combo.addItem('★ ' + name, self.USER_PREFIX + name)
        combo.blockSignals(False)

    def save_root_preset(self, name):
        name = (name or '').strip()
        if not name:
            raise ValueError(self.tr('Please enter a name for the preset.'))
        data = self.user_root_presets()
        settings = self.root_settings()
        settings.pop('enabled', None)
        settings.pop('preset', None)
        data[name] = settings
        self._store_user_presets(data)
        self._fill_user_presets()
        self.combo_root_preset.setCurrentIndex(self.combo_root_preset.findData(self.USER_PREFIX + name))
        self.log(self.tr('Root preset "{}" saved.').format(name), 'ok')

    def on_save_root_preset(self):
        name, ok = QInputDialog.getText(self, self.tr('Save root preset'), self.tr('Name (e.g. species and site):'))
        if ok:
            self.save_root_preset(name)

    def on_delete_root_preset(self):
        key = str(self.combo_root_preset.currentData() or '')
        if not key.startswith(self.USER_PREFIX):
            raise ValueError(self.tr('Select one of your own presets (marked ★) to delete it.'))
        data = self.user_root_presets()
        data.pop(key[len(self.USER_PREFIX):], None)
        self._store_user_presets(data)
        self._fill_user_presets()
        self.combo_root_preset.setCurrentIndex(0)

    def on_root_preset(self):
        name = self.combo_root_preset.currentData()
        if not name:
            self.lbl_root_source.setText('')
            return
        if str(name).startswith(self.USER_PREFIX):
            settings = dict(self.user_root_presets().get(name[len(self.USER_PREFIX):], {}))
            for key in ('custom_range', 'custom_depth_profile'):
                if isinstance(settings.get(key), str):
                    try:
                        settings[key] = _parse_numbers(settings[key], 10)
                    except ValueError:
                        settings[key] = None
            self.apply_root_settings(settings)
            self.lbl_root_source.setText(self.tr('Source: your own preset (no literature reference stored).'))
            return
        preset = presets.ROOT_PRESETS[name]
        self.apply_root_settings(preset['settings'])
        self.lbl_root_source.setText(self.tr('Source: {}').format(preset['source']))

    def apply_root_settings(self, s):
        def select(combo, value):
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)
        if 'shape' in s:
            select(self.combo_root_shape, s['shape'])
        if 'distribution' in s:
            select(self.combo_root_dist, s['distribution'])
        if 'size_mode' in s:
            select(self.combo_root_size, s['size_mode'])
        if 'density_mode' in s:
            select(self.combo_root_density, s['density_mode'])
        for key, box in (('depth', self.spin_root_depth), ('radius_factor', self.spin_root_factor),
                         ('diameter', self.spin_root_diameter), ('beta', self.spin_root_beta),
                         ('d50', self.spin_root_d50), ('d95', self.spin_root_d95),
                         ('rad', self.spin_root_rad), ('total_root_area', self.spin_root_area)):
            if s.get(key) is not None:
                box.setValue(s[key])
        if s.get('custom_range'):
            self.edit_root_range.setText(' '.join('%g' % v for v in s['custom_range']))
        if s.get('custom_depth_profile'):
            self.edit_root_profile.setText(' '.join('%g' % v for v in s['custom_depth_profile']))
        self.update_roots()

    def crown_diameter(self):
        tree = self.current_tree(silent=True)
        if tree is not None:
            return max(tree.nx, tree.ny) * tree.cell_size
        return max(self.spin_width_x.value(), self.spin_width_y.value())

    def root_settings(self):
        return {'enabled': self.chk_roots.isChecked(), 'preset': self.combo_root_preset.currentData() or '',
                'shape': self.combo_root_shape.currentData(), 'depth': self.spin_root_depth.value(),
                'size_mode': self.combo_root_size.currentData(), 'diameter': self.spin_root_diameter.value(),
                'radius_factor': self.spin_root_factor.value(), 'distribution': self.combo_root_dist.currentData(),
                'beta': self.spin_root_beta.value(), 'd50': self.spin_root_d50.value(), 'd95': self.spin_root_d95.value(),
                'density_mode': self.combo_root_density.currentData(),
                'rad': self.spin_root_rad.value(), 'total_root_area': self.spin_root_area.value(),
                'custom_range': self.edit_root_range.text(), 'custom_depth_profile': self.edit_root_profile.text()}

    def update_roots(self):
        # Rebuild the RootSystem from the widgets, refresh enabled states, key figures and preview.
        enabled = self.chk_roots.isChecked()
        factor_mode = self.combo_root_size.currentData() == 'factor'
        area_mode = self.combo_root_density.currentData() == 'area'
        shape, dist = self.combo_root_shape.currentData(), self.combo_root_dist.currentData()
        states = {self.combo_root_preset: True, self.combo_root_shape: True, self.spin_root_depth: True,
                  self.combo_root_size: True, self.spin_root_factor: factor_mode, self.spin_root_diameter: not factor_mode,
                  self.combo_root_dist: True, self.spin_root_beta: dist == roots.DIST_EXPONENTIAL,
                  self.spin_root_d50: dist == roots.DIST_LOGISTIC, self.spin_root_d95: dist == roots.DIST_LOGISTIC,
                  self.combo_root_density: True, self.spin_root_rad: not area_mode, self.spin_root_area: area_mode,
                  self.edit_root_range: shape == roots.SHAPE_CUSTOM, self.edit_root_profile: dist == roots.DIST_CUSTOM}
        for widget, state in states.items():
            widget.setEnabled(enabled and state)
        self.side_view.set_tree(self.current_tree(silent=True))
        if not enabled:
            self.root_system = None
            self.side_view.set_roots(None)
            self.lbl_roots.setText(self.tr('No root system defined – the exporter writes its default values.'))
            self._set_state('roots', 'ok')
            return None
        try:
            self.root_system = roots.build_root_system(
                shape=shape, depth=self.spin_root_depth.value(),
                diameter=None if factor_mode else self.spin_root_diameter.value(),
                crown_diameter=self.crown_diameter(), radius_factor=self.spin_root_factor.value(),
                distribution=dist, beta=self.spin_root_beta.value(),
                d50=self.spin_root_d50.value(), d95=self.spin_root_d95.value(),
                rad=None if area_mode else self.spin_root_rad.value(),
                total_root_area=self.spin_root_area.value() if area_mode else None,
                custom_range=_parse_numbers(self.edit_root_range.text(), 10) if shape == roots.SHAPE_CUSTOM else None,
                custom_depth_profile=_parse_numbers(self.edit_root_profile.text(), 10) if dist == roots.DIST_CUSTOM else None)
        except ValueError as exc:
            self.root_system = None
            self.side_view.set_roots(None)
            self.lbl_roots.setText('⚠ ' + self.tr('Root settings are not valid: {}').format(exc))
            self._set_state('roots', 'warn')
            return None
        r = self.root_system
        self.side_view.set_roots(r)
        self.lbl_roots.setText(self.tr(
            'Root zone: {d:.2f} m deep, {w:.2f} m wide, volume {v:.1f} m³\n'
            'Total root area {a:.1f} m², mean RAD {m:.3f} m²/m³\n'
            'RAD per layer (top → bottom): {rad}\nRadius fraction per layer: {rng}').format(
            d=r.depth, w=r.diameter, v=r.volume(), a=r.total_root_area(), m=r.mean_rad(),
            rad=' '.join('%.3f' % v for v in r.rad), rng=' '.join('%.2f' % v for v in r.root_range)))
        self._set_state('roots', 'ok')
        return r

    # step 8: physiology
    def _page_physiology(self, page):
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.combo_species = QComboBox()
        self.combo_species.addItems(preset_names())
        self.combo_species.setCurrentText('Abstract Deciduous Tree')
        self.combo_species.setToolTip(self.tr('Tree type presets of the A-gs photosynthesis model (from the ENVI-met plant library).'))
        self.combo_species.currentIndexChanged.connect(lambda _i: self.apply_species_preset())
        form.addRow(self.tr('Preset:'), self.combo_species)
        self.chk_manual = QCheckBox(self.tr('Edit the A-gs parameters manually'))
        self.chk_manual.toggled.connect(self._manual_toggled)
        form.addRow(self.chk_manual)
        self.combo_leaf_type = QComboBox()
        self.combo_leaf_type.addItem(self.tr('Deciduous'), 'decid')
        self.combo_leaf_type.addItem(self.tr('Conifer'), 'conifer')
        self.combo_leaf_type.setToolTip(self.tr('Leaf type used by ENVI-met for the leaf energy balance.'))
        form.addRow(self.tr('Leaf type:'), self.combo_leaf_type)
        self.spin_albedo = _dspin(0, 1, 0.18, 2, 0.01, '', self.tr('Short-wave albedo of the foliage (0–1).'))
        self.spin_trans = _dspin(0, 1, 0.30, 2, 0.01, '', self.tr('Short-wave transmittance of the foliage (0–1).'))
        self.spin_eps = _dspin(0, 1, 0.96, 2, 0.01, '', self.tr('Long-wave emissivity of the foliage (0–1).'))
        form.addRow(self.tr('Albedo:'), self.spin_albedo)
        form.addRow(self.tr('Transmittance:'), self.spin_trans)
        form.addRow(self.tr('Emissivity:'), self.spin_eps)
        self.edit_season = QLineEdit(' '.join('%g' % v for v in DEFAULT_SEASON_PROFILE))
        self.edit_season.setToolTip(self.tr('Twelve monthly foliage factors January…December (0–1); the LAD is multiplied by them.'))
        form.addRow(self.tr('Season profile:'), self.edit_season)
        layout.addLayout(form)
        self.box_ags = QGroupBox(self.tr('A-gs parameters'))
        grid = QGridLayout(self.box_ags)
        self.ags_spins = {}
        for n, (key, label, decimals, low, high) in enumerate(AGS_LABELS):
            spin = _dspin(low, high, 0.0, decimals, 10 ** -decimals, '', label)
            self.ags_spins[key] = spin
            grid.addWidget(QLabel(label), n // 2, (n % 2) * 2)
            grid.addWidget(spin, n // 2, (n % 2) * 2 + 1)
        layout.addWidget(self.box_ags)
        layout.addStretch(1)
        self.apply_species_preset()
        self._manual_toggled(False)
        self._states['physiology'] = 'ok'

    def _manual_toggled(self, manual):
        self.combo_species.setEnabled(not manual)
        for spin in self.ags_spins.values():
            spin.setEnabled(manual)
        if not manual:
            self.apply_species_preset()

    def apply_species_preset(self):
        name = self.combo_species.currentText()
        if name not in AGS_PRESETS:
            return
        for key, value in ags_of(name).items():
            self.ags_spins[key].setValue(value)
        preset = AGS_PRESETS[name]
        self.spin_albedo.setValue(preset.get('albedo', 0.18))
        self.spin_trans.setValue(preset.get('transmittance', 0.30))
        self.spin_eps.setValue(preset.get('eps', 0.96))
        conifer = any(word in name for word in ('Conifer', 'Pine', 'Thuja'))
        self.combo_leaf_type.setCurrentIndex(1 if conifer else 0)

    def ags_values(self):
        values = {key: self.ags_spins[key].value() for key in AGS_KEYS}
        if values['gm_t1'] > values['gm_t2'] or values['ammax_t1'] > values['ammax_t2']:
            raise ValueError(self.tr('Physiology: the lower temperature limit T1 must not be above T2.'))
        return values

    # step 9: export
    def _page_export(self, page):
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.combo_format = QComboBox()
        self.combo_format.addItem(self.tr('ENVI-met 6.x (JSON)'), FORMAT_JSON6)
        self.combo_format.addItem(self.tr('ENVI-met 5.x (classic database)'), FORMAT_LEGACY)
        self.combo_format.setCurrentIndex(self.combo_format.findData(DEFAULT_FORMAT))
        self.combo_format.setToolTip(self.tr('Format of the target ENVI-met version. ENVI-met 6 uses JSON, ENVI-met 5.x the classic text format.'))
        self.combo_format.currentIndexChanged.connect(lambda _i: self.on_database_changed())
        form.addRow(self.tr('ENVI-met version:'), self.combo_format)
        self.edit_id = QLineEdit('L2E001')
        self.edit_id.setMaxLength(6)
        self.edit_id.setToolTip(self.tr('Plant ID: exactly 6 characters, digits and capital letters. Must be unique in the database.'))
        self.edit_id.textChanged.connect(lambda _t: self.on_database_changed())
        self.btn_suggest_id = QPushButton(self.tr('Suggest free ID'))
        self.btn_suggest_id.clicked.connect(self.guarded(self.on_suggest_id))
        row = QHBoxLayout()
        row.addWidget(self.edit_id, 1)
        row.addWidget(self.btn_suggest_id)
        form.addRow(self.tr('Plant ID:'), row)
        self.edit_description = QLineEdit('Tree from las2ENVImet-Plugin')
        self.edit_description.setToolTip(self.tr('Name of the plant as shown in ENVI-met.'))
        form.addRow(self.tr('Name:'), self.edit_description)
        self.edit_binomial = QLineEdit('')
        self.edit_binomial.setToolTip(self.tr('Botanical name (optional), e.g. Tilia cordata.'))
        form.addRow(self.tr('Botanical name:'), self.edit_binomial)
        self.chk_square = QCheckBox(self.tr('Square plant grid (same width in x and y)'))
        self.chk_square.setToolTip(self.tr(
            'ENVI-met 6 stores one crown width, so its plant grid is always square: the narrower direction is '
            'padded with empty cells and the trunk stays in the centre. For ENVI-met 5.x this is optional; '
            'switched off, width x and width y of the grid may differ.'))
        self._square_legacy = False # The user's choice for ENVI-met 5.x (6.x is always square)
        self.chk_square.clicked.connect(self._square_clicked)
        form.addRow(self.chk_square)
        layout.addLayout(form)

        box = QGroupBox(self.tr('Target'))
        v = QVBoxLayout(box)
        self.radio_db = QRadioButton(self.tr('Insert into an existing database (project or user database)'))
        self.radio_file = QRadioButton(self.tr('Write a new single-plant file'))
        self.radio_file.setChecked(True)
        v.addWidget(self.radio_file)
        self.file_output = QgsFileWidget()
        self.file_output.setStorageMode(QgsFileWidget.StorageMode.SaveFile)
        self.file_output.setToolTip(self.tr('New file. ENVI-met 6: a database file (*.edb) that contains only this plant. '
                                            'ENVI-met 5.x: plant definition as text (*.txt).'))
        v.addWidget(self.file_output)
        v.addWidget(self.radio_db)
        self.file_database = QgsFileWidget()
        self.file_database.setStorageMode(QgsFileWidget.StorageMode.GetFile)
        self.file_database.setFilter(self.tr('ENVI-met database (*.edb);;All files (*.*)'))
        self.file_database.setToolTip(self.tr('projectdatabase.edb of a project or userdatabase.edb. A backup copy is made before writing.'))
        self.file_database.fileChanged.connect(lambda _p: self.on_database_changed())
        v.addWidget(self.file_database)
        self.lbl_database = QLabel('')
        self.lbl_database.setWordWrap(True)
        v.addWidget(self.lbl_database)
        self.radio_db.toggled.connect(lambda _c: self.on_database_changed())
        layout.addWidget(box)

        self.btn_export = QPushButton(self.tr('Export'))
        self.btn_export.clicked.connect(self.guarded(self.on_export))
        layout.addWidget(self.btn_export)
        self.lbl_export = QLabel('')
        self.lbl_export.setWordWrap(True)
        self.lbl_export.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.lbl_export)
        layout.addStretch(1)

    def _square_clicked(self, checked):
        self._square_legacy = bool(checked)

    def plant_id(self):
        return normalise_id(self.edit_id.text())

    def on_database_changed(self):
        # Detect the format of the chosen database, warn about contradictions and id collisions.
        use_db = self.radio_db.isChecked()
        self.file_database.setEnabled(use_db)
        self.file_output.setEnabled(not use_db)
        fmt = self.combo_format.currentData()
        try:
            forced = get_exporter(self.export_format()).square_grid
        except ValueError:
            forced = get_exporter(fmt).square_grid
        self.chk_square.setEnabled(not forced)
        self.chk_square.setChecked(True if forced else self._square_legacy)
        ext = get_exporter(fmt).single_file_extension
        self.file_output.setFilter(self.tr('ENVI-met plant file (*{})').format(ext))
        messages = []
        if not is_valid_id(self.plant_id()):
            messages.append('⚠ ' + self.tr('The plant ID needs exactly 6 characters (0–9, A–Z).'))
        path = self.file_database.filePath()
        if use_db and path:
            if not os.path.exists(path):
                messages.append('⚠ ' + self.tr('The database file does not exist.'))
            else:
                detected = detect_format(path)
                if detected is None:
                    messages.append('⚠ ' + self.tr('This file is not recognised as an ENVI-met database.'))
                else:
                    label = self.combo_format.itemText(self.combo_format.findData(detected))
                    messages.append(self.tr('Detected format: {}').format(label))
                    if detected != fmt:
                        messages.append('⚠ ' + self.tr('This contradicts the selected ENVI-met version. The database format is '
                                                       'used, so the file stays readable for its ENVI-met version.'))
                    try:
                        ids = get_exporter(detected).list_plant_ids(path)
                        messages.append(self.tr('{} 3D plants in the database.').format(len(ids)))
                        if self.plant_id() in ids:
                            messages.append('⚠ ' + self.tr('The ID {} already exists – exporting asks before replacing it.').format(self.plant_id()))
                    except ExportError as exc:
                        messages.append('⚠ ' + str(exc))
        self.lbl_database.setText('\n'.join(messages))

    def on_suggest_id(self):
        ids = []
        path = self.file_database.filePath()
        if self.radio_db.isChecked() and path and os.path.exists(path):
            fmt = detect_format(path)
            if fmt:
                ids = get_exporter(fmt).list_plant_ids(path)
        self.edit_id.setText(suggest_free_id(ids))

    def export_format(self):
        # Format actually written: the database's own format wins over the selector.
        path = self.file_database.filePath()
        if self.radio_db.isChecked() and path and os.path.exists(path):
            detected = detect_format(path)
            if detected is None:
                raise ValueError(self.tr('The selected file is not an ENVI-met database.'))
            return detected
        return self.combo_format.currentData()

    def current_tree(self, silent=False, fmt=None):
        # TreeData for the current state (None if the foliage is not ready and silent is set).
        if self.wf.voxel_corners is None or self.wf.lad_values is None:
            if silent:
                return None
            raise ValueError(self.tr('Please calculate the LAD first (step 6).'))
        fmt = fmt or (self.combo_format.currentData() if hasattr(self, 'combo_format') else DEFAULT_FORMAT)
        try:
            season = _parse_numbers(self.edit_season.text(), 12)
        except ValueError:
            if silent:
                season = list(DEFAULT_SEASON_PROFILE)
            else:
                raise ValueError(self.tr('The season profile needs exactly 12 numbers.'))
        return build_tree_data(
            self.wf, self.plant_id() if hasattr(self, 'edit_id') else 'L2E001',
            self.edit_description.text().strip() or 'Tree from las2ENVImet-Plugin',
            lad_threshold=0.0,
            square=get_exporter(fmt).square_grid or (hasattr(self, 'chk_square') and self.chk_square.isChecked()),
            albedo=self.spin_albedo.value(), transmittance=self.spin_trans.value(), emissivity=self.spin_eps.value(),
            ags=self.ags_values() if not silent else {k: self.ags_spins[k].value() for k in AGS_KEYS},
            roots=self.root_system, binomial=self.edit_binomial.text().strip() or '(none)',
            leaf_type=self.combo_leaf_type.currentData(), season_profile=season)

    def on_export(self, confirm_overwrite=None):
        # Export. confirm_overwrite: None = ask the user, True/False = answer given (tests).
        if not is_valid_id(self.plant_id()):
            raise ValueError(self.tr('The plant ID needs exactly 6 characters (0–9, A–Z).'))
        if self.chk_roots.isChecked() and self.update_roots() is None:
            raise ValueError(self.tr('The root settings are not valid (step 7).'))
        fmt = self.export_format()
        exporter = get_exporter(fmt)
        tree = self.current_tree(fmt=fmt)
        if not tree.cells:
            raise ValueError(self.tr('The tree has no voxels with leaves – nothing to export.'))
        self.tree = tree
        if self.radio_db.isChecked():
            path = self.file_database.filePath()
            if not path or not os.path.exists(path):
                raise ValueError(self.tr('Please choose an existing database file.'))
            try:
                result = exporter.insert_plant(tree, path)
            except PlantExistsError:
                if confirm_overwrite is None:
                    buttons = QMessageBox.StandardButton
                    answer = QMessageBox.question(
                        self, self.tr('Replace plant?'),
                        self.tr('The database already contains a plant with the ID {}.\nReplace it? '
                                '(A backup of the database is made first.)').format(tree.plant_id),
                        buttons.Yes | buttons.No, buttons.No)
                    confirm_overwrite = answer == buttons.Yes
                if not confirm_overwrite:
                    self.log(self.tr('Export cancelled – the existing plant was kept.'), 'warn')
                    return None
                result = exporter.insert_plant(tree, path, overwrite=True)
            text = self.tr('Plant {id} {action} in {path}\nBackup: {backup}').format(
                id=tree.plant_id, action=self.tr('replaced') if result['action'] == 'replaced' else self.tr('inserted'),
                path=path, backup=result['backup'])
        else:
            path = self.file_output.filePath()
            if not path:
                raise ValueError(self.tr('Please choose a file name for the export.'))
            if not os.path.splitext(path)[1]:
                path += exporter.single_file_extension
            exporter.write_plant_file(tree, path)
            text = self.tr('Plant {id} written to {path}').format(id=tree.plant_id, path=path)
        text += '\n' + self.tr('{nx} × {ny} × {nz} cells, {n} voxels with leaves, LAI {lai:.2f}, leaf area {la:.1f} m²').format(
            nx=tree.nx, ny=tree.ny, nz=tree.nz, n=len(tree.cells), lai=tree.crown_lai(), la=tree.total_leaf_area())
        self.lbl_export.setText(text)
        self.log(text, 'ok')
        self._set_state('export', 'ok')
        self.on_database_changed()
        if self.iface is not None:
            self.iface.messageBar().pushMessage('las2ENVImet', self.tr('Export finished.'), level=MSG_SUCCESS, duration=6)
        return path

    # project files
    def collect_settings(self):
        s = project_io.new_project()
        s['source'].update(path=self.source_path() or '', format_options=self.ascii_options or {},
                           classes=self.load_options().get('classes'), thin=self.load_options().get('thin'),
                           bbox=self.load_options().get('bbox'), crs=self.crs_authid or None)
        s['roi'] = self.roi_ring
        s['z_filter'] = {'lower_percent': self.spin_z_low.value(), 'upper_percent': self.spin_z_up.value()}
        s['stem'] = [float(self.wf.stem_point[0]), float(self.wf.stem_point[1])] if self.wf.current_dims else None
        s['transform'] = {'height': self.spin_height.value(), 'width_x': self.spin_width_x.value(),
                          'width_y': self.spin_width_y.value(), 'rotation': self.spin_rotation.value()}
        s['foliage'].update(mode=self.foliage_mode(), target_lai=self.spin_lai.value(),
                            lai_reference=self.combo_lai_ref.currentData(), total_leaf_area=self.spin_leaf_area.value(),
                            lad_max=self.spin_lad_max.value(), refinement=self.refinement_settings(),
                            refined=bool(getattr(self, '_refined', False)),
                            references=[{'x': float(r['x']) + 0.5, 'y': float(r['y']) + 0.5, 'z': float(r['z']),
                                         'lad': float(r['lad_value'])} for r in self.wf.ref_voxels])
        s['roots'] = self.root_settings()
        s['physiology'].update(preset=self.combo_species.currentText(), manual=self.chk_manual.isChecked(),
                               ags={k: self.ags_spins[k].value() for k in AGS_KEYS}, albedo=self.spin_albedo.value(),
                               transmittance=self.spin_trans.value(), emissivity=self.spin_eps.value(),
                               leaf_type=self.combo_leaf_type.currentData(), season_profile=self.edit_season.text())
        s['export'].update(format=self.combo_format.currentData(), plant_id=self.plant_id(),
                           description=self.edit_description.text(), binomial=self.edit_binomial.text(),
                           square=self._square_legacy,
                           target='database' if self.radio_db.isChecked() else 'file',
                           file=self.file_output.filePath(), database=self.file_database.filePath())
        return s

    def apply_settings(self, s):
        # Put project settings into the widgets (does not load or calculate anything).
        src = s['source']
        self.layer_combo.setCurrentIndex(0)
        self.file_source.setFilePath(src.get('path') or '')
        self.ascii_options = src.get('format_options') or None
        self.crs_authid = src.get('crs') or ''
        self.chk_classes.setChecked(bool(src.get('classes')))
        if src.get('classes'):
            self.edit_classes.setText(','.join(str(c) for c in src['classes']))
        self.chk_thin.setChecked(bool(src.get('thin')))
        if src.get('thin'):
            self.spin_thin.setValue(src['thin'])
        self.spin_z_low.setValue(s['z_filter']['lower_percent'])
        self.spin_z_up.setValue(s['z_filter']['upper_percent'])
        f = s['foliage']
        self.set_foliage_mode(f['mode'])
        self.spin_lai.setValue(f['target_lai'])
        self.combo_lai_ref.setCurrentIndex(max(0, self.combo_lai_ref.findData(f['lai_reference'])))
        self.spin_leaf_area.setValue(f['total_leaf_area'])
        self.spin_lad_max.setValue(f['lad_max'])
        r = dict(s['roots'])
        self.chk_roots.setChecked(bool(r.get('enabled')))
        for key in ('custom_range', 'custom_depth_profile'):
            if isinstance(r.get(key), str):
                try:
                    r[key] = _parse_numbers(r[key], 10)
                except ValueError:
                    r[key] = None
        self.apply_root_settings(r)
        p = s['physiology']
        self.chk_manual.setChecked(bool(p['manual']))
        if p['preset'] in AGS_PRESETS:
            self.combo_species.setCurrentText(p['preset'])
        if p['manual']:
            for key, value in p['ags'].items():
                if key in self.ags_spins:
                    self.ags_spins[key].setValue(value)
        self.spin_albedo.setValue(p['albedo'])
        self.spin_trans.setValue(p['transmittance'])
        self.spin_eps.setValue(p['emissivity'])
        self.combo_leaf_type.setCurrentIndex(max(0, self.combo_leaf_type.findData(p['leaf_type'])))
        season = p['season_profile']
        self.edit_season.setText(season if isinstance(season, str) else ' '.join('%g' % v for v in season))
        e = s['export']
        self.combo_format.setCurrentIndex(max(0, self.combo_format.findData(e['format'])))
        if e['plant_id']:
            self.edit_id.setText(e['plant_id'])
        self.edit_description.setText(e['description'])
        self.edit_binomial.setText(e['binomial'])
        self._square_legacy = bool(e.get('square'))
        (self.radio_db if e['target'] == 'database' else self.radio_file).setChecked(True)
        self.file_output.setFilePath(e['file'])
        self.file_database.setFilePath(e['database'])
        self._pending = s

    def replay_after_load(self, s):
        # Repeat the processing steps of a project on the freshly loaded cloud.
        if s.get('roi'):
            self.apply_roi([tuple(p) for p in s['roi']])
        if s.get('stem'):
            self.set_stem(s['stem'][0], s['stem'][1])
            t = s['transform']
            for box, key in ((self.spin_height, 'height'), (self.spin_width_x, 'width_x'), (self.spin_width_y, 'width_y')):
                if t.get(key):
                    box.setValue(t[key])
            self.spin_rotation.setValue(t.get('rotation') or 0.0)
            self.wf.transform_points(self.target_dims(), self.spin_rotation.value())
            self.wf.create_voxels(1.0)
            self._voxelised(None)
            f = s['foliage']
            for ref in f.get('references', []):
                self.add_reference(ref['x'], ref['y'], ref['z'], ref['lad'])
            ref_settings = f['refinement']
            self.spin_crown_factor.setValue(ref_settings['crown_factor'])
            self.spin_crown_z_min.setValue(int(ref_settings['crown_z_min']))
            self.spin_crown_z_max.setValue(int(ref_settings['crown_z_max']))
            self.chk_trunk.setChecked(bool(ref_settings['trunk_enhance']))
            self.spin_trunk_height.setValue(int(ref_settings['trunk_height']))
            self.spin_trunk_lad.setValue(ref_settings['trunk_lad'])
            self.spin_threshold.setValue(ref_settings['threshold'])
            self.chk_clear_ground.setChecked(bool(ref_settings['clear_ground']))
            if f['mode'] != foliage.MODE_REFERENCE or f.get('references'):
                self.on_calculate_lad()
                if f.get('refined'):
                    self.on_refine()
                    self._refined = True
        self.update_roots()
        self.log(self.tr('Project restored.'), 'ok')

    def on_save_project(self, path=None):
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, self.tr('Save project'), '',
                                                  self.tr('las2ENVImet project (*.l2e.json)'))
        if not path:
            return None
        if not path.lower().endswith(project_io.EXTENSION):
            path += project_io.EXTENSION
        project_io.save_project(path, self.collect_settings(), self.plugin_version())
        self.log(self.tr('Project saved: {}').format(path), 'ok')
        return path

    def on_load_project(self, path=None, on_ready=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, self.tr('Open project'), '',
                                                  self.tr('las2ENVImet project (*.l2e.json)'))
        if not path:
            return
        settings = project_io.load_project(path)
        self.apply_settings(settings)
        self.log(self.tr('Project opened: {}').format(path), 'ok')
        source = settings['source'].get('path')
        if not source or not os.path.exists(source):
            self.log(self.tr('The point cloud of the project was not found: {}. Choose it in step 1 and load it.').format(source), 'warn')
            return
        reader = readers.reader_for(source)
        options = self.load_options()
        if settings['source'].get('bbox'):
            options['bbox'] = tuple(settings['source']['bbox'])

        def loaded(cloud):
            self._loaded(cloud)
            self.replay_after_load(settings)
            if on_ready:
                on_ready()

        self._run_task(self.tr('las2ENVImet: loading point cloud'),
                       lambda progress, cancelled: reader.read(source, progress=progress, is_cancelled=cancelled, **options),
                       loaded)

    def plugin_version(self):
        try:
            with open(os.path.join(self.plugin_dir, 'metadata.txt'), encoding='utf-8') as fh:
                for line in fh:
                    if line.startswith('version='):
                        return line.split('=', 1)[1].strip()
        except OSError:
            pass
        return ''

    def on_help(self):
        key = STEP_KEYS[self.stack.currentIndex()]
        QDesktopServices.openUrl(QUrl(README_URL + HELP_ANCHORS[key]))

    def closeEvent(self, event):
        if self.canvas() is not None and self._tool is not None:
            self.canvas().unsetMapTool(self._tool)
        super().closeEvent(event)
