# las_reader.py
# LAS / LAZ / COPC reader (laspy; LAZ and COPC additionally need the lazrs or laszip backend)
import numpy as np

from .base import Cancelled, PointCloud, Reader, ReaderError, bbox_mask, _importable

CHUNK = 2_000_000


def _laz_backend_available():
    try:
        import laspy
        return len(laspy.LazBackend.detect_available()) > 0
    except Exception:  # noqa: BLE001
        return False


class LasReader(Reader):
    format_id = 'las'
    label = 'LAS / LAZ / COPC'
    extensions = ('.las', '.laz', '.copc.laz')
    modules = ('laspy',)
    pip_packages = ('laspy', 'lazrs')
    optional = False

    def laz_available(self):
        return _laz_backend_available()

    def _read(self, path, bbox, classes, progress, is_cancelled, **options):
        import laspy
        compressed = path.lower().endswith('.laz')
        if compressed and not _laz_backend_available():
            raise ReaderError("Compressed LAZ/COPC files need the 'lazrs' package. "
                              "Install with:  python -m pip install lazrs")
        if path.lower().endswith('.copc.laz') and bbox is not None and hasattr(laspy, 'CopcReader'):
            return self._read_copc(laspy, path, bbox, progress)

        parts = {'xyz': [], 'classification': [], 'intensity': [], 'return_number': [], 'rgb': []}
        with laspy.open(path) as reader:
            header = reader.header
            total = max(1, header.point_count)
            names = set(header.point_format.dimension_names)
            has_rgb = {'red', 'green', 'blue'} <= names
            done = 0
            for chunk in reader.chunk_iterator(CHUNK):
                if is_cancelled():
                    raise Cancelled()
                xyz = np.column_stack((np.asarray(chunk.x, dtype=np.float64),
                                       np.asarray(chunk.y, dtype=np.float64),
                                       np.asarray(chunk.z, dtype=np.float64)))
                done += len(xyz)
                mask = bbox_mask(xyz, bbox)
                cls = np.asarray(chunk.classification).astype(np.uint8) if 'classification' in names else None
                if classes is not None and cls is not None:
                    class_mask = np.isin(cls, list(classes))
                    mask = class_mask if mask is None else (mask & class_mask)

                def take(values):
                    return values if mask is None else values[mask]

                parts['xyz'].append(take(xyz))
                if cls is not None:
                    parts['classification'].append(take(cls))
                if 'intensity' in names:
                    parts['intensity'].append(take(np.asarray(chunk.intensity)))
                if 'return_number' in names:
                    parts['return_number'].append(take(np.asarray(chunk.return_number).astype(np.uint8)))
                if has_rgb:
                    parts['rgb'].append(take(np.column_stack((np.asarray(chunk.red), np.asarray(chunk.green),
                                                              np.asarray(chunk.blue)))))
                progress(min(1.0, done / total))
            crs = _crs_text(header)
            count = header.point_count

        def join(key, empty_shape=(0,)):
            return np.concatenate(parts[key]) if parts[key] else None

        xyz = np.concatenate(parts['xyz']) if parts['xyz'] else np.zeros((0, 3))
        return PointCloud(xyz, classification=join('classification'), intensity=join('intensity'),
                          return_number=join('return_number'), rgb=join('rgb'), crs=crs, total_points=count)

    def _read_copc(self, laspy, path, bbox, progress):
        # Spatial query of a COPC file: only the octree nodes touching the box are decompressed.
        with laspy.CopcReader.open(path) as reader:
            header = reader.header
            mins = np.array([bbox[0], bbox[1], header.mins[2] if len(bbox) == 4 else bbox[2]])
            maxs = np.array([bbox[-2] if len(bbox) == 4 else bbox[3],
                             bbox[-1] if len(bbox) == 4 else bbox[4],
                             header.maxs[2] if len(bbox) == 4 else bbox[5]])
            if len(bbox) == 4:
                mins[:2] = bbox[0], bbox[1]
                maxs[:2] = bbox[2], bbox[3]
            points = reader.query(laspy.copc.Bounds(mins=mins, maxs=maxs))
            progress(1.0)
            names = set(points.point_format.dimension_names)
            xyz = np.column_stack((np.asarray(points.x, dtype=np.float64), np.asarray(points.y, dtype=np.float64),
                                   np.asarray(points.z, dtype=np.float64)))
            rgb = None
            if {'red', 'green', 'blue'} <= names:
                rgb = np.column_stack((np.asarray(points.red), np.asarray(points.green), np.asarray(points.blue)))
            return PointCloud(
                xyz,
                classification=np.asarray(points.classification).astype(np.uint8) if 'classification' in names else None,
                intensity=np.asarray(points.intensity) if 'intensity' in names else None,
                return_number=np.asarray(points.return_number).astype(np.uint8) if 'return_number' in names else None,
                rgb=rgb, crs=_crs_text(header), total_points=header.point_count)

    def info(self, path):
        import laspy
        with laspy.open(path) as reader:
            h = reader.header
            return {'points': int(h.point_count), 'version': str(h.version), 'point_format': int(h.point_format.id),
                    'mins': [float(v) for v in h.mins], 'maxs': [float(v) for v in h.maxs], 'crs': _crs_text(h)}


def _crs_text(header):
    try:
        crs = header.parse_crs()
        return crs.to_wkt() if crs is not None else None
    except Exception:  # pyproj missing or broken VLR: CRS simply unknown
        return None


def write_las(path, cloud: PointCloud, point_format: int = None, version: str = '1.4'):
    # Write a PointCloud as LAS/LAZ (used for ROI layers and for creating test data).
    import laspy
    has_rgb = cloud.rgb is not None
    if point_format is None:
        point_format = 7 if has_rgb else 6
        if version != '1.4':
            point_format = 2 if has_rgb else 0
    header = laspy.LasHeader(point_format=point_format, version=version)
    header.scales = np.array([0.0001, 0.0001, 0.0001])
    header.offsets = np.floor(cloud.xyz.min(axis=0)) if len(cloud) else np.zeros(3)
    las = laspy.LasData(header)
    las.x, las.y, las.z = cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2]
    if cloud.classification is not None:
        las.classification = cloud.classification
    if cloud.intensity is not None:
        las.intensity = cloud.intensity
    if cloud.return_number is not None:
        las.return_number = np.maximum(cloud.return_number, 1)
        las.number_of_returns = np.maximum(cloud.return_number, 1)
    if has_rgb and {'red', 'green', 'blue'} <= set(header.point_format.dimension_names):
        las.red, las.green, las.blue = cloud.rgb[:, 0], cloud.rgb[:, 1], cloud.rgb[:, 2]
    las.write(path)
    return path
