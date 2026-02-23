import os

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets

# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'las2envimet_dialog_base.ui'))


class LAS2ENVImetDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        # Constructor
        super(LAS2ENVImetDialog, self).__init__(parent)
        # Set up the user interface from Designer through FORM_CLASS.
        # After self.setupUi() you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots
        self.setupUi(self)

        # Ensure UI state on startup
        self.spinTrunkHeight.setEnabled(False)
        self.spinTrunkLAD.setEnabled(False)
