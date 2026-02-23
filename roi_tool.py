# roi_tool.py
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.core import QgsWkbTypes, QgsGeometry, QgsPointXY
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor


class ROIPolygonTool(QgsMapTool):
    # Map tool for drawing a polygon ROI. Left click adds vertices, right click closes the polygon   

    def __init__(self, canvas, callback):
        super().__init__(canvas)
        self.canvas = canvas
        self.callback = callback

        # Rubber band provides a transparent overlay fpr tracking the mouse while drawing the polygon
        self.rubber_band = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self.rubber_band.setColor(QColor(255, 0, 0, 100))  # semi‑transparent red
        self.rubber_band.setWidth(2)

        self.points = []          # list of QgsPointXY vertices
        self.is_drawing = False   # True while polygon is being drawn

    def canvasPressEvent(self, event):
        # Handle mouse press events
        if event.button() == Qt.LeftButton:
            point = self.toMapCoordinates(event.pos())
            self.points.append(point)
            self.rubber_band.addPoint(point)
            self.is_drawing = True

        elif event.button() == Qt.RightButton and self.is_drawing:
            if len(self.points) < 3:
                self.reset() # Not enough points – reset and ignore
                return

            polygon = QgsGeometry.fromPolygonXY([self.points])
            self.rubber_band.reset()
            self.callback(polygon)
            self.reset()

    def reset(self):
        # Clear rubber band and vertex list
        self.rubber_band.reset()
        self.points = []
        self.is_drawing = False
        self.canvas.unsetMapTool(self)

    def deactivate(self):
        self.reset()
        super().deactivate()