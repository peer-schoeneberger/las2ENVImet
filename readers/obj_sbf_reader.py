# obj_sbf_reader.py
# Two more point formats written by CloudCompare
import os

import numpy as np

from .base import PointCloud, Reader, ReaderError


class ObjReader(Reader):
    format_id = 'obj'
    label = 'OBJ (vertices)'
    extensions = ('.obj',)
    modules = ()

    def _read(self, path, bbox, classes, progress, is_cancelled, **options):
        data = None
        try:
            import pandas
            # More names than columns: pandas fills the missing colour columns with NaN
            frame = pandas.read_csv(path, sep=r'\s+', header=None, comment='#', names=list('tabcdef'),
                                    dtype={'t': str}, engine='c', skip_blank_lines=True)
            frame = frame[frame['t'] == 'v']
            data = frame[list('abcdef')].to_numpy(dtype=np.float64)
        except ImportError:
            rows = []
            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    if line.startswith('v '):
                        parts = line.split()[1:7]
                        rows.append([float(v) for v in parts] + [np.nan] * (6 - len(parts)))
            data = np.array(rows, dtype=np.float64).reshape(-1, 6)
        progress(1.0)
        if data is None or len(data) == 0:
            raise ReaderError("The OBJ file contains no vertices.")
        rgb = None
        if not np.any(np.isnan(data[:, 3:6])):
            colours = data[:, 3:6]
            rgb = (colours * 255.0 if colours.max() <= 1.0 else colours).astype(np.uint16)
        return PointCloud(data[:, :3], rgb=rgb)


class SbfReader(Reader):
    format_id = 'sbf'
    label = 'SBF (CloudCompare)'
    extensions = ('.sbf', '.sbf.data')
    modules = ()

    def _read(self, path, bbox, classes, progress, is_cancelled, **options):
        data_path = path if path.lower().endswith('.sbf.data') else path + '.data'
        text_path = data_path[:-len('.data')]
        if not os.path.exists(data_path):
            raise ReaderError("The binary part of the SBF file is missing: %s" % os.path.basename(data_path))
        names = {}
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    key, _, value = line.strip().partition('=')
                    if key.upper().startswith('SF') and key[2:].isdigit():
                        names[int(key[2:])] = value.strip().lower()
        with open(data_path, 'rb') as fh:
            header = fh.read(64)
            if len(header) < 64 or header[:2] != b'\x2a\x2a':
                raise ReaderError("Not an SBF data file (wrong signature).")
            count = int(np.frombuffer(header, dtype='>u8', count=1, offset=2)[0])
            fields = int(np.frombuffer(header, dtype='>u2', count=1, offset=10)[0])
            shift = np.frombuffer(header, dtype='>f8', count=3, offset=12).astype(np.float64)
            table = np.fromfile(fh, dtype='>f4', count=count * (3 + fields))
        if len(table) != count * (3 + fields):
            raise ReaderError("The SBF data file ends before all points were read.")
        table = table.reshape(count, 3 + fields)
        progress(1.0)
        xyz = table[:, :3].astype(np.float64) + shift # Coordinates are stored relative to the shift
        intensity = classification = None
        for index in range(fields):
            name = names.get(index + 1, '')
            if 'intens' in name:
                intensity = table[:, 3 + index].astype(np.float32)
            elif 'class' in name:
                classification = table[:, 3 + index].astype(np.uint8)
        return PointCloud(xyz, intensity=intensity, classification=classification)


def write_sbf(path, cloud: PointCloud):
    # Write x y z [intensity] as SBF (test data).
    shift = cloud.xyz[0] if len(cloud) else np.zeros(3)
    columns = [cloud.xyz - shift]
    if cloud.intensity is not None:
        columns.append(np.asarray(cloud.intensity, dtype=np.float64).reshape(-1, 1))
    table = np.hstack(columns).astype('>f4')
    header = bytearray(64)
    header[0:2] = b'\x2a\x2a'
    header[2:10] = np.array([len(cloud)], dtype='>u8').tobytes()
    header[10:12] = np.array([table.shape[1] - 3], dtype='>u2').tobytes()
    header[12:36] = np.asarray(shift, dtype='>f8').tobytes()
    with open(path + '.data', 'wb') as fh:
        fh.write(bytes(header))
        table.tofile(fh)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('[SBF]\nPoints=%d\nGlobalShift=0.000000, 0.000000, 0.000000\nSFCount=%d\n' % (len(cloud), table.shape[1] - 3))
        if cloud.intensity is not None:
            fh.write('SF1=Intensity\n')
    return path


def write_obj(path, cloud: PointCloud):
    # Write the points as OBJ vertices (test data).
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('# written by las2ENVImet\n')
        np.savetxt(fh, cloud.xyz, fmt='v %.9f %.9f %.9f')
    return path
