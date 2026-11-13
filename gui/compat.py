# compat.py
# Small compatibility layer
from qgis.core import Qgis, QgsField, QgsMapLayer, QgsWkbTypes
from qgis.PyQt.QtCore import QVariant

QGIS_VERSION_INT = Qgis.QGIS_VERSION_INT


def _first(*getters):
    for getter in getters:
        try:
            return getter()
        except (AttributeError, TypeError):
            continue
    raise AttributeError('no variant of the enum value exists in this QGIS version')


# Message levels (getattr: spelling of older QGIS versions)
MSG_INFO = _first(lambda: Qgis.MessageLevel.Info, lambda: getattr(Qgis, 'Info'))
MSG_WARNING = _first(lambda: Qgis.MessageLevel.Warning, lambda: getattr(Qgis, 'Warning'))
MSG_CRITICAL = _first(lambda: Qgis.MessageLevel.Critical, lambda: getattr(Qgis, 'Critical'))
MSG_SUCCESS = _first(lambda: Qgis.MessageLevel.Success, lambda: getattr(Qgis, 'Success'))

# Geometry type for rubber bands
POLYGON_GEOMETRY = _first(lambda: Qgis.GeometryType.Polygon, lambda: getattr(QgsWkbTypes, 'PolygonGeometry'))


def is_point_cloud_layer(layer) -> bool:
    if layer is None:
        return False
    expected = _first(lambda: Qgis.LayerType.PointCloud, lambda: QgsMapLayer.LayerType.PointCloudLayer,
                      lambda: getattr(QgsMapLayer, 'PointCloudLayer'))
    return layer.type() == expected


def make_field(name: str, kind: str) -> QgsField:
    # QgsField for 'int', 'double' or 'string' (QMetaType in QGIS 4, QVariant before).
    try:
        from qgis.PyQt.QtCore import QMetaType
        types = {'int': QMetaType.Type.Int, 'double': QMetaType.Type.Double, 'string': QMetaType.Type.QString}
        if QGIS_VERSION_INT >= 33800:
            return QgsField(name, types[kind])
    except (ImportError, AttributeError, TypeError):
        pass
    types = {'int': QVariant.Int, 'double': QVariant.Double, 'string': QVariant.String}
    return QgsField(name, types[kind])


def exec_dialog(dialog):
    # Modal execution with the Qt6-safe spelling (available in PyQt5 since 5.x on Python 3).
    return dialog.exec()


def message_box_buttons():
    from qgis.PyQt.QtWidgets import QMessageBox
    return QMessageBox.StandardButton
