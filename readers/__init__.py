# Point-cloud readers: one class per format behind a common interface (see base.py).
import os
from typing import List, Optional

from .ascii_reader import AsciiReader, sniff_ascii
from .base import (Cancelled, MissingDependency, PointCloud, Reader, ReaderError, VEGETATION_CLASSES,
                   apply_filters)
from .e57_reader import E57Reader
from .las_reader import LasReader
from .obj_sbf_reader import ObjReader, SbfReader
from .pcd_reader import PcdReader
from .ply_reader import PlyReader

READERS: List[Reader] = [LasReader(), E57Reader(), PlyReader(), PcdReader(), ObjReader(), SbfReader(),
                         AsciiReader()]


def reader_for(path: str) -> Optional[Reader]:
    # Reader responsible for a file name (longest matching extension wins: .copc.laz before .laz).
    lower = path.lower()
    best, best_len = None, 0
    for reader in READERS:
        for ext in reader.extensions:
            if lower.endswith(ext) and len(ext) > best_len:
                best, best_len = reader, len(ext)
    return best


def read_point_cloud(path: str, **kwargs) -> PointCloud:
    reader = reader_for(path)
    if reader is None:
        raise ReaderError("Unsupported point cloud format: %s" % (os.path.splitext(path)[1] or path))
    return reader.read(path, **kwargs)


def file_filter(only_available: bool = False) -> str:
    # File-dialog filter string, e.g. 'Point clouds (*.las *.laz ...);;LAS / LAZ / COPC (*.las ...)'.
    parts, all_ext = [], []
    for reader in READERS:
        if only_available and not reader.availability()[0]:
            continue
        patterns = ' '.join('*' + e for e in reader.extensions)
        all_ext.extend('*' + e for e in reader.extensions)
        parts.append('%s (%s)' % (reader.label, patterns))
    return ';;'.join(['Point clouds (%s)' % ' '.join(sorted(set(all_ext)))] + parts)


def dependency_report() -> List[dict]:
    # Status of every format for the start-up check: label, available, hint, pip command.
    report = []
    for reader in READERS:
        available, hint = reader.availability()
        entry = {'format': reader.label, 'available': available, 'hint': hint, 'optional': reader.optional,
                 'pip': 'python -m pip install ' + ' '.join(reader.pip_packages) if reader.pip_packages else ''}
        if isinstance(reader, LasReader) and available and not reader.laz_available():
            entry['hint'] = "LAS works; compressed LAZ/COPC needs:  python -m pip install lazrs"
            entry['partial'] = True
        report.append(entry)
    return report
