# map_tools.py
# Map tools
from qgis.core import QgsGeometry
from qgis.gui import QgsMapTool, QgsMapToolEmitPoint, QgsRubberBand
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor

from .compat import POLYGON_GEOMETRY


class PointTool(QgsMapToolEmitPoint):
    # Calls callback(QgsPointXY) with the clicked map position.

    def __init__(self, canvas, callback):
        super().__init__(canvas)
        self._canvas = canvas
        self._callback = callback

    def canvasReleaseEvent(self, event):
        self._callback(self.toMapCoordinates(event.pos()))


class RoiPolygonTool(QgsMapTool):
    # Left click adds a vertex, right click closes the polygon and calls callback(ring)

    def __init__(self, canvas, callback):
        super().__init__(canvas)
        self._canvas = canvas
        self._callback = callback
        self._band = QgsRubberBand(canvas, POLYGON_GEOMETRY)
        self._band.setColor(QColor(255, 0, 0, 100))
        self._band.setWidth(2)
        self._points = []

    def canvasPressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.toMapCoordinates(event.pos())
            self._points.append(point)
            self._band.addPoint(point)
        elif event.button() == Qt.MouseButton.RightButton:
            points = list(self._points)
            self.clear()
            if len(points) < 3:
                self._callback(None)
                return
            ring = [(p.x(), p.y()) for p in points]
            ring.append(ring[0])
            geometry = QgsGeometry.fromPolygonXY([points + [points[0]]])
            self._callback(ring if not geometry.isNull() and geometry.isGeosValid() else None)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.clear()
            self._callback(None)

    def clear(self):
        self._band.reset(POLYGON_GEOMETRY)
        self._points = []

    def deactivate(self):
        self.clear()
        super().deactivate()
