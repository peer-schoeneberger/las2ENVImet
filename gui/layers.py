# layers.py
# Helper functions for the temporary map layers the wizard shows (QGIS side)
import numpy as np
from qgis.core import (QgsFeature, QgsGeometry, QgsGraduatedSymbolRenderer, QgsPalLayerSettings, QgsPointXY,
                       QgsProject, QgsRectangle, QgsRendererRange, QgsSymbol, QgsTextFormat, QgsVectorLayer,
                       QgsVectorLayerSimpleLabeling)
from qgis.PyQt.QtGui import QColor

from .compat import make_field

PREFIX = 'L2E '


def remove_layers(startswith: str):
    project = QgsProject.instance()
    ids = [lyr.id() for lyr in project.mapLayers().values() if lyr.name().startswith(startswith)]
    if ids:
        project.removeMapLayers(ids)


def remove_all_helper_layers():
    remove_layers(PREFIX)


def point_layer(name: str, crs_authid: str, points: np.ndarray, max_points: int = 30000, seed: int = 0):
    # Memory point layer with a (reproducible) random subset of the points.
    layer = QgsVectorLayer('Point?crs=%s' % crs_authid if crs_authid else 'Point', name, 'memory')
    if len(points) > max_points:
        index = np.random.RandomState(seed).choice(len(points), max_points, replace=False)
        points = points[index]
    features = []
    for x, y in points[:, :2]:
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(x), float(y))))
        features.append(feature)
    layer.dataProvider().addFeatures(features)
    layer.updateExtents()
    return layer


def voxel_layer(name: str, crs_authid: str, corners: np.ndarray, counts: np.ndarray, lad=None,
                voxel_size: float = 1.0, min_lad: float = 0.0):
    # Polygon layer with one square per voxel of a horizontal slice (attributes: points, LAD).
    layer = QgsVectorLayer('Polygon?crs=%s' % crs_authid if crs_authid else 'Polygon', name, 'memory')
    provider = layer.dataProvider()
    provider.addAttributes([make_field('points', 'int'), make_field('LAD', 'double')])
    layer.updateFields()
    features = []
    for n, (x, y, _z) in enumerate(corners):
        value = float(lad[n]) if lad is not None else 0.0
        if lad is not None and min_lad > 0 and value < min_lad:
            continue
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromRect(QgsRectangle(float(x), float(y), float(x) + voxel_size,
                                                              float(y) + voxel_size)))
        feature.setAttributes([int(counts[n]), value])
        features.append(feature)
    provider.addFeatures(features)
    layer.updateExtents()
    return layer


def style_graduated(layer, field: str, start: QColor, end: QColor, classes: int = 6):
    values = [f[field] for f in layer.getFeatures() if f[field] is not None]
    if not values:
        return
    low, high = min(values), max(values)
    if abs(high - low) < 1e-6:
        high = low + 1.0
    ranges = []
    for n in range(classes):
        lower = low + n * (high - low) / classes
        upper = high if n == classes - 1 else low + (n + 1) * (high - low) / classes
        t = n / (classes - 1) if classes > 1 else 0.5
        color = QColor(int(start.red() + t * (end.red() - start.red())),
                       int(start.green() + t * (end.green() - start.green())),
                       int(start.blue() + t * (end.blue() - start.blue())))
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol.setColor(color)
        symbol.setOpacity(0.7)
        ranges.append(QgsRendererRange(lower, upper, symbol, '%.2f – %.2f' % (lower, upper)))
    layer.setRenderer(QgsGraduatedSymbolRenderer(field, ranges))


def enable_labels(layer, field: str, decimals: int = 2, size: int = 8):
    settings = QgsPalLayerSettings()
    if decimals > 0:
        settings.fieldName = 'format_number("%s", %d)' % (field, decimals)
        settings.isExpression = True
    else:
        settings.fieldName = field
        settings.isExpression = False
    settings.enabled = True
    text_format = QgsTextFormat()
    text_format.setSize(size)
    settings.setFormat(text_format)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    layer.setLabelsEnabled(True)


def add(layer):
    QgsProject.instance().addMapLayer(layer)
    return layer
