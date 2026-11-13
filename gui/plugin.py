# plugin.py
# QGIS plugin entry
import os

from qgis.PyQt.QtCore import QCoreApplication, QLocale, QSettings, QTranslator
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction


class LAS2ENVImet:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.actions = []
        self.dialog = None
        self.translator = None
        self._install_translator()
        self.menu = self.tr('&LAS to ENVI-met Tree Converter')

    def _install_translator(self):
        locale = QSettings().value('locale/userLocale', QLocale.system().name()) or 'en'
        path = os.path.join(self.plugin_dir, 'i18n', 'las2envimet_%s.qm' % str(locale)[:2])
        if os.path.exists(path):
            self.translator = QTranslator()
            if self.translator.load(path):
                QCoreApplication.installTranslator(self.translator)

    def tr(self, message):
        return QCoreApplication.translate('LAS2ENVImet', message)

    def initGui(self):
        icon = QIcon(os.path.join(self.plugin_dir, 'icon.png'))
        action = QAction(icon, self.tr('LAS to ENVI-met Converter'), self.iface.mainWindow())
        action.triggered.connect(self.run)
        self.iface.addToolBarIcon(action)
        self.iface.addPluginToMenu(self.menu, action)
        self.actions.append(action)

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)
        self.actions = []
        if self.dialog is not None:
            self.dialog.close()
            self.dialog = None
        if self.translator is not None:
            QCoreApplication.removeTranslator(self.translator)

    def run(self):
        # Open the wizard (non-modal, so the map stays usable).
        try:
            if self.dialog is None:
                from .wizard import WizardDialog
                self.dialog = WizardDialog(self.iface, self.plugin_dir)
            self.dialog.refresh_layers()
            self.dialog.show()
            self.dialog.raise_()
            self.dialog.activateWindow()
        except Exception as exc:  # never let a start-up problem end in a traceback dialog
            from .compat import MSG_CRITICAL
            self.iface.messageBar().pushMessage('las2ENVImet', self.tr('The plugin could not start: {}').format(exc),
                                                level=MSG_CRITICAL)
        return self.dialog
