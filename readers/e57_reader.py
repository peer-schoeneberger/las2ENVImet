# e57_reader.py
# E57_reader.py E57_reader.py E57 reader (pye57)
import os
import subprocess
import sys
import tempfile

import numpy as np

from .base import Cancelled, PointCloud, Reader, ReaderError, _importable

_SELF_TEST = r'''
import os, sys, tempfile
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for p in os.environ.get("PATH", "").split(";"):
        if p and os.path.isdir(p):
            try: os.add_dll_directory(p)
            except OSError: pass
import numpy as np, pye57
path = os.path.join(tempfile.mkdtemp(prefix="l2e_e57_"), "t.e57")
e = pye57.E57(path, mode="w")
e.write_scan_raw({"cartesianX": np.arange(4.0), "cartesianY": np.arange(4.0), "cartesianZ": np.arange(4.0)})
e.close()
e = pye57.E57(path)
d = e.read_scan(0, ignore_missing_fields=True)
assert len(d["cartesianX"]) == 4
try:
    e.close()
    os.remove(path)
except Exception:
    pass # a left-over temp file is not a failed test
'''

_self_test_result = None # None = not tested yet, True / False afterwards


def find_python():
    # The Python interpreter belonging to this process (inside QGIS sys.executable is qgis-bin.exe).
    exe = sys.executable or ''
    if os.path.basename(exe).lower().startswith('python') and os.path.exists(exe):
        return exe
    names = ('python.exe', 'python3.exe') if os.name == 'nt' else ('python3', 'python')
    folders = [sys.exec_prefix, os.path.join(sys.exec_prefix, 'bin'), os.path.dirname(exe),
               os.path.join(os.path.dirname(sys.exec_prefix), 'bin'),
               os.path.join(os.path.dirname(os.path.dirname(sys.exec_prefix)), 'bin')]
    for folder in folders:
        for name in names:
            candidate = os.path.join(folder, name)
            if os.path.exists(candidate):
                return candidate
    return None


def pye57_works(force=False):
    # Run the pye57 self test in a child process (once per session).
    global _self_test_result
    if _self_test_result is not None and not force:
        return _self_test_result
    python = find_python()
    if python is None:
        _self_test_result = True # Cannot test; do not block the format
        return True
    kwargs = {}
    if os.name == 'nt':
        kwargs['creationflags'] = 0x08000000 # CREATE_NO_WINDOW
    try:
        result = subprocess.run([python, '-c', _SELF_TEST], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=60, cwd=tempfile.gettempdir(), **kwargs)
        _self_test_result = result.returncode == 0
    except Exception:  # timeout or launch failure: treat as not usable
        _self_test_result = False
    return _self_test_result


class E57Reader(Reader):
    format_id = 'e57'
    label = 'E57'
    extensions = ('.e57',)
    modules = ('pye57',)
    pip_packages = ('pye57',)

    def availability(self):
        available, hint = super().availability()
        if not available:
            return available, hint
        if not pye57_works():
            return False, ("The installed pye57 package does not work with this Python/numpy (%s) and would "
                           "crash QGIS. E57 is disabled here; use a newer QGIS version or convert the file "
                           "to LAS/LAZ." % np.__version__)
        return True, ''

    def _read(self, path, bbox, classes, progress, is_cancelled, **options):
        import pye57
        e57 = pye57.E57(path)
        count = e57.scan_count
        if count == 0:
            raise ReaderError("The E57 file contains no scans.")
        xyz, intensity, rgb = [], [], []
        for index in range(count):
            if is_cancelled():
                raise Cancelled()
            data = e57.read_scan(index, intensity=True, colors=True, ignore_missing_fields=True)
            pts = np.column_stack((data['cartesianX'], data['cartesianY'], data['cartesianZ'])).astype(np.float64)
            xyz.append(pts)
            intensity.append(np.asarray(data['intensity']) if 'intensity' in data else None)
            if {'colorRed', 'colorGreen', 'colorBlue'} <= set(data):
                rgb.append(np.column_stack((data['colorRed'], data['colorGreen'], data['colorBlue'])))
            else:
                rgb.append(None)
            progress((index + 1) / count)
        all_xyz = np.concatenate(xyz)
        all_int = np.concatenate(intensity) if all(v is not None for v in intensity) else None
        all_rgb = np.concatenate(rgb) if all(v is not None for v in rgb) else None
        return PointCloud(all_xyz, intensity=all_int, rgb=all_rgb)


def write_e57(path, cloud: PointCloud):
    # Write a PointCloud as a single-scan E57 file (test data).
    import pye57
    data = {'cartesianX': cloud.xyz[:, 0].copy(), 'cartesianY': cloud.xyz[:, 1].copy(),
            'cartesianZ': cloud.xyz[:, 2].copy()}
    if cloud.intensity is not None:
        data['intensity'] = np.asarray(cloud.intensity, dtype=np.float64)
    e57 = pye57.E57(path, mode='w')
    e57.write_scan_raw(data)
    e57.close()
    return path
