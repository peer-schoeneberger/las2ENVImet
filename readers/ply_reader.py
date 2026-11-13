# ply_reader.py
# PLY reader (ASCII and binary)
import numpy as np

from .base import PointCloud, Reader, ReaderError

_TYPES = {
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1', 'short': 'i2', 'int16': 'i2',
    'ushort': 'u2', 'uint16': 'u2', 'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
}


def _pick(names, *candidates):
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _cloud_from_vertex(vertex, names) -> PointCloud:
    x, y, z = _pick(names, 'x'), _pick(names, 'y'), _pick(names, 'z')
    if not (x and y and z):
        raise ReaderError("The PLY file has no x/y/z vertex properties.")
    xyz = np.column_stack((np.asarray(vertex[x], dtype=np.float64), np.asarray(vertex[y], dtype=np.float64),
                           np.asarray(vertex[z], dtype=np.float64)))
    r, g, b = _pick(names, 'red', 'r'), _pick(names, 'green', 'g'), _pick(names, 'blue', 'b')
    rgb = np.column_stack((vertex[r], vertex[g], vertex[b])) if (r and g and b) else None
    cls = _pick(names, 'classification', 'class', 'scalar_classification', 'label')
    inten = _pick(names, 'intensity', 'scalar_intensity', 'reflectance')
    ret = _pick(names, 'return_number', 'returnnumber', 'scalar_returnnumber')
    return PointCloud(
        xyz, rgb=rgb,
        classification=np.asarray(vertex[cls]).astype(np.uint8) if cls else None,
        intensity=np.asarray(vertex[inten]) if inten else None,
        return_number=np.asarray(vertex[ret]).astype(np.uint8) if ret else None)


def _read_builtin(path) -> PointCloud:
    with open(path, 'rb') as fh:
        if fh.readline().strip() != b'ply':
            raise ReaderError("Not a PLY file (missing 'ply' magic line).")
        fmt, elements, current = None, [], None
        while True:
            line = fh.readline()
            if not line:
                raise ReaderError("Unexpected end of the PLY header.")
            parts = line.decode('ascii', 'replace').split()
            if not parts or parts[0] == 'comment' or parts[0] == 'obj_info':
                continue
            if parts[0] == 'format':
                fmt = parts[1]
            elif parts[0] == 'element':
                current = {'name': parts[1], 'count': int(parts[2]), 'props': []}
                elements.append(current)
            elif parts[0] == 'property':
                if parts[1] == 'list':
                    current['props'].append(('list', parts[-1]))
                else:
                    if parts[1] not in _TYPES:
                        raise ReaderError("Unsupported PLY property type: %s" % parts[1])
                    current['props'].append((_TYPES[parts[1]], parts[2]))
            elif parts[0] == 'end_header':
                break
        if not elements or elements[0]['name'] != 'vertex':
            raise ReaderError("The first PLY element must be 'vertex'.")
        vertex = elements[0]
        if any(t == 'list' for t, _ in vertex['props']):
            raise ReaderError("PLY vertex list properties need the 'plyfile' package (pip install plyfile).")
        names = [n for _, n in vertex['props']]
        if fmt == 'ascii':
            data = np.loadtxt(fh, max_rows=vertex['count'], ndmin=2)
            if data.shape[0] != vertex['count']:
                raise ReaderError("The PLY file ends before all vertices were read.")
            table = {n: data[:, i] for i, n in enumerate(names)}
        elif fmt in ('binary_little_endian', 'binary_big_endian'):
            order = '<' if fmt == 'binary_little_endian' else '>'
            dtype = np.dtype([(n, order + t) for t, n in vertex['props']])
            data = np.fromfile(fh, dtype=dtype, count=vertex['count'])
            if len(data) != vertex['count']:
                raise ReaderError("The PLY file ends before all vertices were read.")
            table = {n: data[n] for n in names}
        else:
            raise ReaderError("Unknown PLY format: %s" % fmt)
    return _cloud_from_vertex(table, names)


class PlyReader(Reader):
    format_id = 'ply'
    label = 'PLY'
    extensions = ('.ply',)
    modules = () # Built-in fallback, plyfile is optional
    pip_packages = ('plyfile',)

    def _read(self, path, bbox, classes, progress, is_cancelled, **options):
        try:
            from plyfile import PlyData
        except Exception:
            PlyData = None
        cloud = None
        if PlyData is not None:
            try:
                vertex = PlyData.read(path)['vertex']
                names = [p.name for p in vertex.properties]
                cloud = _cloud_from_vertex(vertex.data, names)
            except ReaderError:
                raise
            except Exception:
                cloud = None   # fall back to the built-in parser
        if cloud is None:
            cloud = _read_builtin(path)
        progress(1.0)
        return cloud


def write_ply(path, cloud: PointCloud, binary: bool = True):
    # Write a PointCloud as PLY (test data, no external package needed).
    fields = [('x', '<f8'), ('y', '<f8'), ('z', '<f8')]
    columns = [cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2]]
    header_types = ['double', 'double', 'double']
    if cloud.intensity is not None:
        fields.append(('intensity', '<u2')); columns.append(cloud.intensity); header_types.append('ushort')
    if cloud.classification is not None:
        fields.append(('classification', 'u1')); columns.append(cloud.classification); header_types.append('uchar')
    if cloud.rgb is not None:
        for n, name in enumerate(('red', 'green', 'blue')):
            fields.append((name, '<u2')); columns.append(cloud.rgb[:, n]); header_types.append('ushort')
    header = ['ply', 'format %s 1.0' % ('binary_little_endian' if binary else 'ascii'),
              'comment written by las2ENVImet', 'element vertex %d' % len(cloud)]
    header += ['property %s %s' % (t, f[0]) for t, f in zip(header_types, fields)]
    header.append('end_header')
    with open(path, 'wb') as fh:
        fh.write(('\n'.join(header) + '\n').encode('ascii'))
        if binary:
            data = np.zeros(len(cloud), dtype=np.dtype(fields))
            for (name, _), col in zip(fields, columns):
                data[name] = col
            data.tofile(fh)
        else:
            for row in zip(*columns):
                fh.write((' '.join(repr(float(v)) if isinstance(v, (float, np.floating)) else str(int(v))
                                   for v in row) + '\n').encode('ascii'))
    return path
