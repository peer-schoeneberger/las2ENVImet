# map_tools.py
# Provides a point picking tool that emits a callback with the selected map coordinates

from qgis.gui import QgsMapToolEmitPoint
from qgis.core import QgsPointXY
from typing import Callable


class PointTool(QgsMapToolEmitPoint):
    # Inherits from QgsMapToolEmitPoint and captures canvas release events

    def __init__(self, canvas: 'QgsMapCanvas', callback: Callable[[QgsPointXY], None]) -> None:
        super().__init__(canvas)
        self.canvas = canvas
        self.callback = callback

    def canvasReleaseEvent(self, event: 'QgsMouseEvent') -> None:
        # Converts the event position to map coordinates and calls the callback.
        point = self.toMapCoordinates(event.pos())
        self.callback(point)