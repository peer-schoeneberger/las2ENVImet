# preview.py
from qgis.PyQt.QtCore import QPointF, QRectF, Qt
from qgis.PyQt.QtGui import QColor, QPainter, QPen, QPolygonF
from qgis.PyQt.QtWidgets import QSizePolicy, QWidget


class SideViewWidget(QWidget):
    # Shows the x-z projection of the plant grid (maximum LAD along y) and the root outline.

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cells = {}          # (i, k) -> max LAD
        self._nx = self._nz = 0
        self._cell = 1.0
        self._root = None         # core.roots.RootSystem
        self.setMinimumSize(260, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_tree(self, tree):
        self._cells = {}
        if tree is not None:
            for i, _j, k, lad in tree.cells:
                self._cells[(i, k)] = max(lad, self._cells.get((i, k), 0.0))
            self._nx, self._nz, self._cell = tree.nx, tree.nz, tree.cell_size
        else:
            self._nx = self._nz = 0
        self.update()

    def set_roots(self, root):
        self._root = root
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        crown_w = self._nx * self._cell
        crown_h = self._nz * self._cell
        root_w = self._root.diameter if self._root else 0.0
        root_d = self._root.depth if self._root else 0.0
        width = max(crown_w, root_w, 1.0)
        height = max(crown_h, 1.0) + root_d
        margin = 14.0
        scale = min((self.width() - 2 * margin) / width, (self.height() - 2 * margin) / height)
        cx = self.width() / 2.0
        ground = margin + max(crown_h, 1.0) * scale

        # Soil
        painter.fillRect(QRectF(0, ground, self.width(), self.height() - ground), QColor(240, 230, 215))
        # Crown voxels
        peak = max(self._cells.values()) if self._cells else 1.0
        x0 = cx - crown_w * scale / 2.0
        size = self._cell * scale
        painter.setPen(QPen(QColor(255, 255, 255), 0.5))
        for (i, k), lad in self._cells.items():
            t = min(1.0, lad / peak) if peak > 0 else 0.0
            painter.setBrush(QColor(int(200 - 200 * t), int(235 - 135 * t), int(200 - 200 * t)))
            painter.drawRect(QRectF(x0 + i * size, ground - (k + 1) * size, size, size))
        # Roots
        if self._root:
            polygon = QPolygonF()
            step = self._root.layer_thickness
            radii = self._root.layer_radii()
            right = [(r, n * step) for n, r in enumerate(radii)] + [(r, (n + 1) * step) for n, r in enumerate(radii)]
            right.sort(key=lambda p: (p[1], 0))
            ordered = []
            for n, r in enumerate(radii):
                ordered.append((r, n * step))
                ordered.append((r, (n + 1) * step))
            for r, d in ordered:
                polygon.append(QPointF(cx + r * scale, ground + d * scale))
            for r, d in reversed(ordered):
                polygon.append(QPointF(cx - r * scale, ground + d * scale))
            painter.setPen(QPen(QColor(120, 80, 40), 1.5))
            painter.setBrush(QColor(170, 120, 70, 110))
            painter.drawPolygon(polygon)
            peak_rad = max(self._root.rad) or 1.0
            for n, (r, value) in enumerate(zip(radii, self._root.rad)): # Density as darker bands
                if r <= 0 or value <= 0:
                    continue
                alpha = int(30 + 150 * value / peak_rad)
                painter.fillRect(QRectF(cx - r * scale, ground + n * step * scale, 2 * r * scale, step * scale),
                                 QColor(110, 70, 30, alpha))
        # Ground line and scale bar (1 m)
        painter.setPen(QPen(QColor(60, 60, 60), 1.5))
        painter.drawLine(QPointF(0, ground), QPointF(self.width(), ground))
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawLine(QPointF(margin, self.height() - 8), QPointF(margin + scale, self.height() - 8))
        painter.drawText(QPointF(margin + scale + 4, self.height() - 4), '1 m')
        painter.end()

    def grab_png(self, path):
        return self.grab().save(path)
