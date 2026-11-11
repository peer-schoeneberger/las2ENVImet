# base.py
# Uniform point-cloud reader interface (no QGIS dependency)
import importlib
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ATTRIBUTES = ('classification', 'intensity', 'return_number', 'rgb')
VEGETATION_CLASSES = (3, 4, 5)        # ASPRS: low / medium / high vegetation


class ReaderError(Exception):
    # Problem with a point-cloud file that the user can understand (no traceback needed).
    pass


class MissingDependency(ReaderError):
    def __init__(self, reader):
        available, hint = reader.availability()
        super().__init__("%s files need additional Python packages. %s" % (reader.label, hint))
        self.reader = reader


class Cancelled(Exception):
    # Raised inside a reader when the user cancelled the task.
    pass


class PointCloud:
    def __init__(self, xyz, classification=None, intensity=None, return_number=None, rgb=None,
                 crs=None, source=None, total_points=None):
        self.xyz = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 3)
        self.classification = classification
        self.intensity = intensity
        self.return_number = return_number
        self.rgb = rgb
        self.crs = crs # WKT or authority id as text, None if unknown
        self.source = source
        self.total_points = total_points if total_points is not None else len(self.xyz)

    def __len__(self):
        return len(self.xyz)

    def attributes(self) -> Dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in ATTRIBUTES if getattr(self, name) is not None}

    def select(self, mask_or_index) -> 'PointCloud':
        attrs = {k: v[mask_or_index] for k, v in self.attributes().items()}
        return PointCloud(self.xyz[mask_or_index], crs=self.crs, source=self.source,
                          total_points=self.total_points, **attrs)

    def bounds(self):
        if len(self.xyz) == 0:
            return None
        return tuple(self.xyz.min(axis=0)) + tuple(self.xyz.max(axis=0))

    def class_counts(self) -> Dict[int, int]:
        if self.classification is None:
            return {}
        values, counts = np.unique(self.classification, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}


def bbox_mask(xyz: np.ndarray, bbox: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    # bbox = (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax, ymax, zmax).
    if bbox is None:
        return None
    if len(bbox) == 4:
        xmin, ymin, xmax, ymax = bbox
        return (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    return ((xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
            (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax))


def apply_filters(cloud: PointCloud, bbox=None, classes=None, thin: Optional[float] = None) -> PointCloud:
    # Bounding box, class filter and voxel-grid thinning, in this order.
    mask = bbox_mask(cloud.xyz, bbox)
    if classes is not None and cloud.classification is not None:
        class_mask = np.isin(cloud.classification, list(classes))
        mask = class_mask if mask is None else (mask & class_mask)
    if mask is not None:
        cloud = cloud.select(mask)
    if thin and thin > 0 and len(cloud):
        keys = np.floor((cloud.xyz - cloud.xyz.min(axis=0)) / thin).astype(np.int64)
        dims = keys.max(axis=0) + 1
        flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
        _, first = np.unique(flat, return_index=True)
        first.sort()
        cloud = cloud.select(first)
    return cloud


class Reader:
    format_id = ''
    label = ''
    extensions: Tuple[str, ...] = ()
    modules: Tuple[str, ...] = () # Importable module names that must be present
    pip_packages: Tuple[str, ...] = () # What to install if they are missing
    optional = True # False: core format of the plugin

    def availability(self) -> Tuple[bool, str]:
        # (usable?, installation hint).
        missing = [m for m in self.modules if not _importable(m)]
        if not missing:
            return True, ''
        return False, "Install with:  python -m pip install %s" % ' '.join(self.pip_packages)

    def matches(self, path: str) -> bool:
        lower = path.lower()
        return any(lower.endswith(ext) for ext in self.extensions)

    def read(self, path: str, bbox=None, classes=None, thin: Optional[float] = None,
             progress: Optional[Callable[[float], None]] = None,
             is_cancelled: Optional[Callable[[], bool]] = None, **options) -> PointCloud:
        if not os.path.exists(path):
            raise ReaderError("File not found: %s" % path)
        if not self.availability()[0]:
            raise MissingDependency(self)
        try:
            cloud = self._read(path, bbox=bbox, classes=classes, progress=progress or (lambda _f: None),
                               is_cancelled=is_cancelled or (lambda: False), **options)
        except (ReaderError, Cancelled):
            raise
        except MemoryError:
            raise ReaderError("Not enough memory to load %s. Load a section (bounding box) or thin the cloud."
                              % os.path.basename(path))
        except Exception as exc:  # every library error becomes a readable message
            raise ReaderError("Could not read %s as %s: %s" % (os.path.basename(path), self.label, exc))
        cloud.source = path
        cloud = apply_filters(cloud, bbox=bbox, classes=classes, thin=thin)
        if len(cloud) == 0:
            raise ReaderError("No points left in %s after applying the filters." % os.path.basename(path))
        if not np.all(np.isfinite(cloud.xyz)):
            keep = np.all(np.isfinite(cloud.xyz), axis=1)
            cloud = cloud.select(keep)
        return cloud

    def _read(self, path, bbox, classes, progress, is_cancelled, **options) -> PointCloud:
        raise NotImplementedError


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:  # broken installs count as missing
        return False
