# tasks.py
# Background tasks (QgsTask) so loading and voxelising never freeze the QGIS window
import traceback

from qgis.core import QgsApplication, QgsTask
from qgis.PyQt.QtCore import pyqtSignal

from ..readers import Cancelled, ReaderError


class FunctionTask(QgsTask):
    # Runs function(progress, is_cancelled) in a worker thread

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, bool)
    cancelled = pyqtSignal()

    def __init__(self, description, function):
        super().__init__(description, QgsTask.Flag.CanCancel)
        self._function = function
        self._result = None
        self._error = None
        self._expected = False
        self._was_cancelled = False

    def run(self):
        try:
            self._result = self._function(lambda fraction: self.setProgress(max(0.0, min(100.0, fraction * 100.0))),
                                          self.isCanceled)
            return not self.isCanceled()
        except Cancelled:
            self._was_cancelled = True
            return False
        except (ReaderError, ValueError) as exc:
            self._error, self._expected = str(exc), True
            return False
        except Exception:  # reported in the log panel, never as a crash
            self._error = traceback.format_exc()
            return False

    def finished(self, ok):
        if ok:
            self.succeeded.emit(self._result)
        elif self._error is not None:
            self.failed.emit(self._error, self._expected)
        else:
            self.cancelled.emit()


def start(task: FunctionTask) -> FunctionTask:
    QgsApplication.taskManager().addTask(task)
    return task
