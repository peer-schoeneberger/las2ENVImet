# layer_utils.py
# Helper functions for creating and manipulating vector layers in QGIS

import numpy as np
from qgis.core import (
    QgsVectorLayer, QgsProject, QgsFeature, QgsGeometry,
    QgsField, QgsRectangle, QgsPalLayerSettings, QgsTextFormat,
    QgsVectorLayerSimpleLabeling, QgsGraduatedSymbolRenderer,
    QgsRendererRange, QgsSymbol, QgsPointXY
)
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtCore import QMetaType
from typing import Optional, Tuple, List, Union


def create_memory_layer(crs_authid: str, name: str, geometry_type: str = "Point") -> QgsVectorLayer:
    layer = QgsVectorLayer(f"{geometry_type}?crs={crs_authid}", name, "memory")
    return layer


def add_points_to_layer(layer: QgsVectorLayer, points: np.ndarray, max_points: int = 30000) -> QgsVectorLayer:
    # If the number of points exceeds max_points, a random subset is used
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]

    provider = layer.dataProvider()
    features = []
    for x, y, _ in points:
        fet = QgsFeature()
        fet.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(x), float(y))))
        features.append(fet)

    provider.addFeatures(features)
    layer.updateExtents()
    return layer


def add_voxel_polygons(layer: QgsVectorLayer, corners: np.ndarray, counts: np.ndarray, lad_values: Optional[np.ndarray] = None, min_lad: float = 0.0) -> QgsVectorLayer:
    # Add square polygons representing voxels at a given Z-level. The polygons have attributes 'points' (number of LiDAR points) and 'LAD' if provided and bigger than min_lad
    provider = layer.dataProvider()
    provider.addAttributes([
        QgsField("points", QMetaType.Type.Int),
        QgsField("LAD", QMetaType.Type.Double)
    ])
    layer.updateFields()

    features = []
    for i, (x, y, z) in enumerate(corners):
        if lad_values is not None and min_lad > 0 and lad_values[i] < min_lad:
            continue

        rect = QgsGeometry.fromRect(QgsRectangle(x, y, x + 1.0, y + 1.0))
        fet = QgsFeature()
        fet.setGeometry(rect)

        point_count = int(counts[i])
        lad_val = float(lad_values[i]) if lad_values is not None else 0.0
        fet.setAttributes([point_count, lad_val])
        features.append(fet)

    provider.addFeatures(features)
    layer.updateExtents()
    return layer


def get_min_max_from_layer(layer: QgsVectorLayer, field_name: str) -> Tuple[Optional[float], Optional[float]]:
    # Retrieve the minimum and maximum values of a numeric field in a vector layer
    min_val = None
    max_val = None
    for feature in layer.getFeatures():
        val = feature[field_name]
        if val is not None:
            if min_val is None or val < min_val:
                min_val = val
            if max_val is None or val > max_val:
                max_val = val
    return min_val, max_val


def apply_dynamic_graduated_style(layer: QgsVectorLayer, field_name: str, start_color: QColor, end_color: QColor, num_classes: int = 5) -> None:
    # Apply an interpolted colour gradient to the layer based on the numeric fields
    min_val, max_val = get_min_max_from_layer(layer, field_name)
    if min_val is None or max_val is None:
        return

    # Avoid division by zero if all values are identical
    if abs(max_val - min_val) < 1e-6:
        max_val = min_val + 1.0

    ranges = []
    for i in range(num_classes):
        lower = min_val + i * (max_val - min_val) / num_classes
        upper = min_val + (i + 1) * (max_val - min_val) / num_classes
        if i == num_classes - 1:
            upper = max_val  # Ensure last range reaches exactly max_val

        # Interpolate colour
        t = i / (num_classes - 1) if num_classes > 1 else 0.5
        r = int(start_color.red() + t * (end_color.red() - start_color.red()))
        g = int(start_color.green() + t * (end_color.green() - start_color.green()))
        b = int(start_color.blue() + t * (end_color.blue() - start_color.blue()))
        color = QColor(r, g, b)

        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol.setColor(color)
        symbol.setOpacity(0.7)
        label = f"{lower:.2f} – {upper:.2f}"
        ranges.append(QgsRendererRange(lower, upper, symbol, label))

    renderer = QgsGraduatedSymbolRenderer(field_name, ranges)
    layer.setRenderer(renderer)
    layer.triggerRepaint()


def enable_labels(layer: QgsVectorLayer, field_name: str = "LAD", text_size: int = 8, decimals: int = 2) -> None:
    # Enable labels for a vector layer using a numeric field
    settings = QgsPalLayerSettings()

    if field_name == "points":
        settings.fieldName = field_name
        settings.isExpression = False
    else:
        settings.fieldName = f'format_number("{field_name}", {decimals})'
        settings.isExpression = True

    settings.enabled = True
    # Places label near the centroid of the polygon
    settings.placement = QgsPalLayerSettings.Placement.AroundPoint
    settings.displayAll = True

    text_format = QgsTextFormat()
    text_format.setSize(text_size)
    settings.setFormat(text_format)

    labeling = QgsVectorLayerSimpleLabeling(settings)
    layer.setLabeling(labeling)
    layer.setLabelsEnabled(True)
    layer.triggerRepaint()