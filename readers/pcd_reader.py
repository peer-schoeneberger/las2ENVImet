# pcd_reader.py
# PCD reader (Point Cloud Library format)
import numpy as np

from .base import PointCloud, Reader, ReaderError

_NP = {('F', 4): 'f4', ('F', 8): 'f8', ('I', 1): 'i1', ('I', 2): 'i2', ('I', 4): 'i4', ('I', 8): 'i8',
       ('U', 1): 'u1', ('U', 2): 'u2', ('U', 4): 'u4', ('U', 8): 'u8'}


def lzf_decompress(data: bytes, expected: int) -> bytes:
    # Minimal LZF decoder (the compression used by binary_compressed PCD files).
    out = bytearray(expected)
    ip, op, n = 0, 0, len(data)
    while ip < n:
        ctrl = data[ip]
        ip += 1
        if ctrl < 32: # Literal run of ctrl + 1 bytes
            length = ctrl + 1
            out[op:op + length] = data[ip:ip + length]
            ip += length
            op += length
        else: # Back reference
            length = ctrl >> 5
            if length == 7:
                length += data[ip]
                ip += 1
            ref = op - ((ctrl & 0x1F) << 8) - data[ip] - 1
            ip += 1
            length += 2
            if ref < 0:
                raise ReaderError("Corrupt LZF data in PCD file.")
            for _ in range(length): # May overlap, copy byte by byte
                out[op] = out[ref]
                op += 1
                ref += 1
    if op != expected:
        raise ReaderError("Corrupt LZF data in PCD file (size mismatch).")
    return bytes(out)


class PcdReader(Reader):
    format_id = 'pcd'
    label = 'PCD'
    extensions = ('.pcd',)
    modules = ()

    def _read(self, path, bbox, classes, progress, is_cancelled, **options):
        with open(path, 'rb') as fh:
            meta = {}
            while True:
                line = fh.readline()
                if not line:
                    raise ReaderError("Unexpected end of the PCD header.")
                text = line.decode('ascii', 'replace').strip()
                if not text or text.startswith('#'):
                    continue
                key, _, value = text.partition(' ')
                meta[key.upper()] = value.split()
                if key.upper() == 'DATA':
                    break
            try:
                fields = meta['FIELDS']
                sizes = [int(v) for v in meta['SIZE']]
                types = meta['TYPE']
                counts = [int(v) for v in meta.get('COUNT', ['1'] * len(fields))]
                points = int(meta['POINTS'][0]) if 'POINTS' in meta else int(meta['WIDTH'][0]) * int(meta['HEIGHT'][0])
                mode = meta['DATA'][0].lower()
            except (KeyError, ValueError, IndexError) as exc:
                raise ReaderError("Incomplete PCD header (%s)." % exc)
            dtype_fields = []
            for name, size, kind, count in zip(fields, sizes, types, counts):
                if (kind, size) not in _NP:
                    raise ReaderError("Unsupported PCD field type %s%d." % (kind, size))
                dtype_fields.append((name, '<' + _NP[(kind, size)]) if count == 1
                                    else (name, '<' + _NP[(kind, size)], (count,)))
            dtype = np.dtype(dtype_fields)
            if mode == 'ascii':
                raw = np.loadtxt(fh, ndmin=2, max_rows=points)
                table, column = {}, 0
                for name, count in zip(fields, counts):
                    table[name] = raw[:, column] if count == 1 else raw[:, column:column + count]
                    column += count
            elif mode == 'binary':
                data = np.fromfile(fh, dtype=dtype, count=points)
                table = {name: data[name] for name in fields}
            elif mode == 'binary_compressed':
                compressed_size, uncompressed_size = np.frombuffer(fh.read(8), dtype='<u4')
                buffer = lzf_decompress(fh.read(int(compressed_size)), int(uncompressed_size))
                table, offset = {}, 0          # Stored field by field (structure of arrays)
                for name, size, kind, count in zip(fields, sizes, types, counts):
                    n = points * count
                    column = np.frombuffer(buffer, dtype='<' + _NP[(kind, size)], count=n, offset=offset)
                    table[name] = column if count == 1 else column.reshape(count, points).T
                    offset += n * size
            else:
                raise ReaderError("Unknown PCD data mode: %s" % mode)
        progress(1.0)
        for axis in ('x', 'y', 'z'):
            if axis not in table:
                raise ReaderError("The PCD file has no %s field." % axis)
        xyz = np.column_stack((table['x'], table['y'], table['z'])).astype(np.float64)
        rgb = None
        if 'rgb' in table or 'rgba' in table:
            packed = np.asarray(table.get('rgb', table.get('rgba')))
            packed = packed.view('<u4') if packed.dtype.kind == 'f' and packed.dtype.itemsize == 4 else packed.astype('<u4')
            rgb = np.column_stack(((packed >> 16) & 255, (packed >> 8) & 255, packed & 255)).astype(np.uint16)
        cls = table.get('classification', table.get('label'))
        return PointCloud(xyz, rgb=rgb, intensity=table.get('intensity'),
                          classification=np.asarray(cls).astype(np.uint8) if cls is not None else None)


def write_pcd(path, cloud: PointCloud, mode='binary'):
    # Write x y z [intensity] as PCD (test data).
    fields, columns = ['x', 'y', 'z'], [cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2]]
    if cloud.intensity is not None:
        fields.append('intensity'); columns.append(np.asarray(cloud.intensity, dtype=np.float64))
    n = len(cloud)
    header = ['# .PCD v0.7 - Point Cloud Data file format', 'VERSION 0.7', 'FIELDS ' + ' '.join(fields),
              'SIZE ' + ' '.join(['8'] * len(fields)), 'TYPE ' + ' '.join(['F'] * len(fields)),
              'COUNT ' + ' '.join(['1'] * len(fields)), 'WIDTH %d' % n, 'HEIGHT 1',
              'VIEWPOINT 0 0 0 1 0 0 0', 'POINTS %d' % n, 'DATA ' + mode]
    with open(path, 'wb') as fh:
        fh.write(('\n'.join(header) + '\n').encode('ascii'))
        if mode == 'ascii':
            np.savetxt(fh, np.column_stack(columns), fmt='%.9f')
        else:
            data = np.zeros(n, dtype=np.dtype([(f, '<f8') for f in fields]))
            for f, c in zip(fields, columns):
                data[f] = c
            data.tofile(fh)
    return path
