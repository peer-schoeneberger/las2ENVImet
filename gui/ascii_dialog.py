# ascii_dialog.py
# Column mapping dialog for ASCII point clouds (XYZ, TXT, CSV, PTS)
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QSpinBox,
                                 QTableWidget, QTableWidgetItem, QVBoxLayout)

from ..readers import sniff_ascii
from ..readers.ascii_reader import ROLES

_DELIMITERS = [(',', 'Comma (,)'), (';', 'Semicolon (;)'), ('\t', 'Tab'), (' ', 'Space(s)'), ('|', 'Pipe (|)')]


class AsciiMappingDialog(QDialog):
    # Shows the detected delimiter, header lines and column roles and lets the user change them.

    def __init__(self, path, parent=None, options=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr('Text file columns'))
        self._path = path
        self._info = sniff_ascii(path)
        options = options or {}
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(self.tr('Assign the columns of the text file. The first lines are shown below.')))
        form = QFormLayout()
        self.delimiter = QComboBox()
        for value, label in _DELIMITERS:
            self.delimiter.addItem(self.tr(label), value)
        self.delimiter.setCurrentIndex([d for d, _ in _DELIMITERS].index(options.get('delimiter', self._info['delimiter'])))
        self.delimiter.setToolTip(self.tr('Character that separates the columns (detected automatically).'))
        form.addRow(self.tr('Delimiter:'), self.delimiter)
        self.skiprows = QSpinBox()
        self.skiprows.setRange(0, 1000)
        self.skiprows.setValue(options.get('skiprows', self._info['skiprows']))
        self.skiprows.setToolTip(self.tr('Number of lines before the first data line (header, point count).'))
        form.addRow(self.tr('Header lines to skip:'), self.skiprows)
        self.decimal_comma = QCheckBox(self.tr('Decimal comma (1,5 instead of 1.5)'))
        self.decimal_comma.setChecked(options.get('decimal_comma', self._info['decimal_comma']))
        form.addRow('', self.decimal_comma)
        self.roles = {}
        mapping = options.get('mapping', self._info['mapping'])
        for role in ROLES:
            combo = QComboBox()
            combo.addItem(self.tr('(not used)'), -1)
            for n in range(self._info['columns']):
                name = self._info['names'][n] if self._info['names'] and n < len(self._info['names']) else ''
                combo.addItem(self.tr('Column {}').format(n + 1) + (' – ' + name if name else ''), n)
            combo.setCurrentIndex(mapping.get(role, -1) + 1)
            self.roles[role] = combo
            form.addRow(role.replace('_', ' ').capitalize() + ':', combo)
        layout.addLayout(form)
        self.table = QTableWidget(len(self._info['preview']), self._info['columns'])
        self.table.setHorizontalHeaderLabels([str(n + 1) for n in range(self._info['columns'])])
        for r, row in enumerate(self._info['preview']):
            for c, value in enumerate(row[:self._info['columns']]):
                self.table.setItem(r, c, QTableWidgetItem(value))
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def tr(self, text):
        return QCoreApplication.translate('AsciiMappingDialog', text)

    def options(self) -> dict:
        mapping = {role: combo.currentData() for role, combo in self.roles.items() if combo.currentData() >= 0}
        return {'delimiter': self.delimiter.currentData(), 'skiprows': self.skiprows.value(),
                'decimal_comma': self.decimal_comma.isChecked(), 'mapping': mapping}
