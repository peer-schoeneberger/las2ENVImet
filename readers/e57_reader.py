# e57_reader.py
# E57 reader (pye57)
import numpy as np

from .base import Cancelled, PointCloud, Reader, ReaderError, _importable

MIN_NUMPY = (1, 20)    # pye57 crashes the process with older numpy (QGIS 3.18)


def pye57_works():
    try:
        version = tuple(int(part) for part in np.__version__.split('.')[:2])
    except ValueError:
        return True
    return version >= MIN_NUMPY


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
