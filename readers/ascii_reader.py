# ascii_reader.py
# ASCII point clouds
import csv
import io
import re

import numpy as np

from .base import PointCloud, Reader, ReaderError

DELIMITERS = (',', ';', '\t', ' ', '|')
ROLES = ('x', 'y', 'z', 'intensity', 'classification', 'return_number', 'red', 'green', 'blue')
_NUMBER = re.compile(r'^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$')

_HEADER_ALIASES = {
    'x': ('x', 'east', 'easting', 'rechtswert', '//x'),
    'y': ('y', 'north', 'northing', 'hochwert'),
    'z': ('z', 'h', 'height', 'elevation', 'hoehe', 'höhe'),
    'intensity': ('intensity', 'i', 'intensität', 'reflectance', 'scalar_intensity'),
    'classification': ('classification', 'class', 'c', 'klasse', 'scalar_classification'),
    'return_number': ('return_number', 'returnnumber', 'r_num', 'return'),
    'red': ('red', 'r'), 'green': ('green', 'g'), 'blue': ('blue', 'b'),
}


def _split(line, delimiter):
    if delimiter == ' ':
        return line.split()
    return [part.strip() for part in line.split(delimiter)]


def _is_number(token):
    return bool(_NUMBER.match(token.strip()))


def sniff_ascii(path, sample_lines: int = 50) -> dict:
    # Inspect the beginning of a text file
    lines = []
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for _ in range(sample_lines):
            line = fh.readline()
            if not line:
                break
            if line.strip():
                lines.append(line.rstrip('\r\n'))
    if not lines:
        raise ReaderError("The file is empty.")

    # Delimiter: the one that yields the most consistent column count (>= 3) on the data lines
    body = lines[2:] if len(lines) > 4 else lines
    best, best_score = ' ', -1.0
    for delimiter in DELIMITERS:
        counts = [len(_split(line, delimiter)) for line in body]
        common = max(set(counts), key=counts.count)
        if common < 3:
            continue
        score = counts.count(common) / len(counts) + (0.01 * common if delimiter != ' ' else 0.0)
        if score > best_score:
            best, best_score = delimiter, score
    delimiter = best

    skiprows, names = 0, None
    for line in lines[:5]:
        tokens = _split(line, delimiter)
        if len(tokens) >= 3 and all(_is_number(t) for t in tokens[:3]):
            break
        if len(tokens) >= 3:
            names = [t.strip().strip('"').lstrip('#/').strip() for t in tokens]
        skiprows += 1       # Header line, PTS point count, comment
    data_lines = lines[skiprows:]
    if not data_lines:
        raise ReaderError("No numeric data lines found.")
    first = _split(data_lines[0], delimiter)
    columns = len(first)
    decimal_comma = delimiter != ',' and any(',' in t for t in first)

    mapping = {}
    if names and len(names) == columns:
        lowered = [n.lower() for n in names]
        for role, aliases in _HEADER_ALIASES.items():
            for alias in aliases:
                if alias in lowered:
                    mapping[role] = lowered.index(alias)
                    break
    if not {'x', 'y', 'z'} <= set(mapping):
        mapping = {'x': 0, 'y': 1, 'z': 2}
        if path.lower().endswith('.pts') or columns == 7:
            mapping.update({'intensity': 3, 'red': 4, 'green': 5, 'blue': 6} if columns >= 7 else {})
        elif columns == 6:
            mapping.update({'red': 3, 'green': 4, 'blue': 5})
        elif columns == 4:
            mapping['intensity'] = 3
    preview = [_split(line, delimiter) for line in data_lines[:10]]
    return {'delimiter': delimiter, 'skiprows': skiprows, 'columns': columns, 'names': names,
            'mapping': mapping, 'decimal_comma': decimal_comma, 'preview': preview}


def _load_table(path, delimiter, skiprows, decimal_comma, usecols, progress):
    sep = None if delimiter == ' ' else delimiter
    try:
        import pandas
        frame = pandas.read_csv(path, sep=r'\s+' if sep is None else sep, header=None, skiprows=skiprows,
                                usecols=usecols, decimal=',' if decimal_comma else '.', comment='#',
                                engine='c', dtype=np.float64, skip_blank_lines=True)
        progress(1.0)
        return frame[usecols].to_numpy()
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        raise ReaderError("Could not parse the text file: %s" % exc)
    if decimal_comma:
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            text = fh.read().replace(',', '.')
        source = io.StringIO(text)
    else:
        source = path
    data = np.loadtxt(source, delimiter=sep, skiprows=skiprows, usecols=usecols, comments='#', ndmin=2)
    progress(1.0)
    return data


class AsciiReader(Reader):
    format_id = 'ascii'
    label = 'ASCII (XYZ, TXT, CSV, PTS)'
    extensions = ('.xyz', '.txt', '.csv', '.pts', '.asc')
    modules = ()

    def _read(self, path, bbox, classes, progress, is_cancelled, delimiter=None, skiprows=None,
              mapping=None, decimal_comma=None, **options):
        guess = sniff_ascii(path)
        delimiter = guess['delimiter'] if delimiter is None else delimiter
        skiprows = guess['skiprows'] if skiprows is None else skiprows
        mapping = dict(guess['mapping'] if mapping is None else mapping)
        decimal_comma = guess['decimal_comma'] if decimal_comma is None else decimal_comma
        for role in ('x', 'y', 'z'):
            if role not in mapping:
                raise ReaderError("Column mapping incomplete: no column assigned to %s." % role.upper())
        roles = [r for r in ROLES if r in mapping and mapping[r] is not None and mapping[r] >= 0]
        usecols = sorted({mapping[r] for r in roles})
        if max(usecols) >= guess['columns']:
            raise ReaderError("The column mapping refers to column %d, but the file has only %d columns."
                              % (max(usecols) + 1, guess['columns']))
        data = _load_table(path, delimiter, skiprows, decimal_comma, usecols, progress)
        col = {c: n for n, c in enumerate(usecols)}

        def get(role):
            return data[:, col[mapping[role]]] if role in roles else None

        xyz = np.column_stack((get('x'), get('y'), get('z')))
        rgb = None
        if all(r in roles for r in ('red', 'green', 'blue')):
            rgb = np.column_stack((get('red'), get('green'), get('blue'))).astype(np.uint16)
        cls = get('classification')
        ret = get('return_number')
        return PointCloud(xyz, intensity=get('intensity'), rgb=rgb,
                          classification=cls.astype(np.uint8) if cls is not None else None,
                          return_number=ret.astype(np.uint8) if ret is not None else None)


def write_ascii(path, cloud: PointCloud, delimiter=' ', header=False, pts=False, decimals=6):
    # Write x y z [intensity] [classification] as text (test data).
    columns, names = [cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2]], ['x', 'y', 'z']
    fmt = ['%%.%df' % decimals] * 3
    if cloud.intensity is not None:
        columns.append(cloud.intensity); names.append('intensity'); fmt.append('%d')
    if cloud.classification is not None and not pts:
        columns.append(cloud.classification); names.append('classification'); fmt.append('%d')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        if pts:
            fh.write('%d\n' % len(cloud))
        elif header:
            fh.write(delimiter.join(names) + '\n')
        np.savetxt(fh, np.column_stack(columns), fmt=delimiter.join(fmt))
    return path
