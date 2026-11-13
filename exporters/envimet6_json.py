# envimet6_json.py
# Envimet6_json.py Envimet6_json.py Exporter for ENVI-met 6.x
import datetime
import json
from collections import OrderedDict
from typing import List

from .base import (FORMAT_JSON6, ExportError, Exporter, PlantExistsError, atomic_write, make_backup,
                   read_text)

SECTIONS = ('soils', 'soilProfiles', 'materials', 'walls', 'singleFaces', 'waterSources', 'emitters',
            'simplePlants', 'greening', 'plants3d')

# Plugin A-gs key -> ENVI-met 6 key (same names as the 5.x tags)
AGS_MAP = OrderedDict([
    ('comp_point25', 'agsCompPoint25'), ('comp_point_q10', 'agsCompPointQ10'),
    ('gm25', 'agsGm25'), ('gm_q10', 'agsGmQ10'), ('gm_t1', 'agsGmT1'), ('gm_t2', 'agsGmT2'),
    ('ammax25', 'agsAmmax25'), ('ammax_q10', 'agsAmmaxQ10'), ('ammax_t1', 'agsAmmaxT1'),
    ('ammax_t2', 'agsAmmaxT2'), ('e0', 'agsE0'), ('f0', 'agsF0'), ('gc', 'agsGc'), ('dmax', 'agsDmax'),
])

# Root zone written when the user did not define one:
DEFAULT_ROOT_DEPTH = 2.0
DEFAULT_RAD = [0.1] * 10
DEFAULT_ROOT_RANGE = [1.0] * 10


def _number(value, digits=5):
    # Round and write integral values without a fraction, as ENVI-met does (8 instead of 8.0).
    value = round(float(value), digits)
    return int(value) if value == int(value) else value


def utc_stamp(now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is not None:
        now = now.astimezone(datetime.timezone.utc)
    return now.strftime('%Y-%m-%dT%H:%M:%S.') + '%03dZ' % (now.microsecond // 1000)


def plant_object(tree, base=None) -> OrderedDict:
    # Plants3d element for a tree
    if tree.nx != tree.ny:
        raise ExportError("ENVI-met 6 needs a square plant grid (got %d x %d cells)." % (tree.nx, tree.ny))
    roots = tree.roots
    ags = tree.ags_complete()
    has_ags = bool(tree.ags)

    plant = OrderedDict()
    plant['id'] = tree.plant_id
    plant['desc'] = tree.description
    plant['grp'] = tree.group
    plant['cost'] = _number(tree.cost)
    plant['col'] = tree.color.lstrip('#').upper()
    plant['binomial'] = tree.binomial or '(none)'
    plant['co2Type'] = 'C4' if tree.co2_type == 'C4' else 'C3'
    plant['leafType'] = 'conifer' if tree.leaf_type == 'conifer' else 'decid'
    plant['height'] = _number(tree.nz * tree.cell_size)
    plant['width'] = _number(tree.nx * tree.cell_size)
    plant['alb'] = _number(tree.albedo)
    plant['eps'] = _number(tree.emissivity)
    plant['leafTrans'] = _number(tree.transmittance)
    plant['isoprene'] = _number(tree.isoprene)
    plant['leafWeight'] = _number(tree.leaf_weight)
    plant['stomResMin'] = _number(tree.stom_res_min)
    plant['rootDepth'] = _number(roots.depth if roots else DEFAULT_ROOT_DEPTH)
    plant['rootDia'] = _number(roots.diameter if roots else tree.nx * tree.cell_size)
    plant['hasAgsParams'] = has_ags
    for key, name in AGS_MAP.items():
        plant[name] = _number(ags[key]) if has_ags else 0

    geometry = OrderedDict()
    cells = sorted(tree.cells, key=lambda c: (c[0], c[1], c[2]))      # ENVI-met order: i, j, k
    geometry['lad'] = [[int(i), int(j), int(k), _number(lad)] for i, j, k, lad in cells]
    geometry['rad'] = [_number(v) for v in (roots.rad if roots else DEFAULT_RAD)]
    geometry['rootRange'] = [_number(v) for v in (roots.root_range if roots else DEFAULT_ROOT_RANGE)]
    geometry['seasonProfile'] = [_number(v) for v in tree.season_profile]
    plant['geometry'] = geometry

    if base is None:
        for key, value in tree.extra.items():
            plant.setdefault(key, value)
        return plant

    merged = OrderedDict()
    for key, value in base.items():
        if key == 'geometry' and isinstance(value, dict):
            geo = OrderedDict(value)
            geo.update(geometry)
            merged[key] = geo
        elif key in plant:
            merged[key] = plant[key]
        else:
            merged[key] = value            # Unknown key: keep
    for key, value in plant.items():
        merged.setdefault(key, value)
    return merged


def empty_database(now=None) -> OrderedDict:
    root = OrderedDict()
    root['header'] = OrderedDict([
        ('fileType', 'databaseJSON'), ('version', 1), ('revisionDate', utc_stamp(now)),
        ('remark', ''), ('description', ''), ('fileInfo', ''),
    ])
    for section in SECTIONS:
        root[section] = []
    return OrderedDict([('envimetDatafile', root)])


def dumps(database) -> bytes:
    # Serialise like SPACES/Albero 6: UTF-8 with BOM, 4 spaces, CRLF.
    text = json.dumps(database, indent=4, ensure_ascii=False)
    return b'\xef\xbb\xbf' + text.replace('\n', '\r\n').encode('utf-8')


def load(path) -> OrderedDict:
    text, _, _ = read_text(path)
    try:
        database = json.loads(text, object_pairs_hook=OrderedDict)
    except ValueError as exc:
        raise ExportError("The file is not valid JSON (%s): %s" % (exc, path))
    root = database.get('envimetDatafile') if isinstance(database, dict) else None
    if not isinstance(root, dict):
        raise ExportError("This file is not an ENVI-met 6 database: %s" % path)
    file_type = root.get('header', {}).get('fileType')
    if file_type != 'databaseJSON':
        raise ExportError("Unexpected file type '%s' (expected 'databaseJSON'): %s" % (file_type, path))
    return database


class Envimet6JsonExporter(Exporter):
    format_id = FORMAT_JSON6
    label = 'ENVI-met 6.x (JSON)'
    single_file_extension = '.edb'
    square_grid = True

    def write_plant_file(self, tree, path, now=None):
        # A stand-alone file is a complete, otherwise empty database with this one plant
        database = empty_database(now)
        database['envimetDatafile']['plants3d'].append(plant_object(tree))
        atomic_write(path, dumps(database))
        return path

    def create_database(self, path, now=None):
        atomic_write(path, dumps(empty_database(now)))
        return path

    def list_plant_ids(self, database_path) -> List[str]:
        root = load(database_path)['envimetDatafile']
        return [p.get('id', '') for p in root.get('plants3d', [])]

    def read_tree(self, database_path, plant_id):
        # Read a plant back into the version-neutral model (round-trip tests, comparisons).
        from ..core.roots import RootSystem
        from ..core.tree_model import TreeData
        root = load(database_path)['envimetDatafile']
        plant = next((p for p in root.get('plants3d', []) if p.get('id') == plant_id), None)
        if plant is None:
            raise ExportError("Plant '%s' not found in %s" % (plant_id, database_path))
        geometry = plant.get('geometry', {})
        cells = sorted(((int(c[0]), int(c[1]), int(c[2]), float(c[3])) for c in geometry.get('lad', [])),
                       key=lambda c: (c[2], c[1], c[0]))
        n = int(round(plant['width']))
        if cells:
            n = max(n, max(max(c[0], c[1]) for c in cells) + 1)
        ags = {key: float(plant[name]) for key, name in AGS_MAP.items()} if plant.get('hasAgsParams') else None
        roots = RootSystem(plant['rootDepth'], plant['rootDia'], geometry['rad'], geometry['rootRange'])
        known = set(AGS_MAP.values()) | {'id', 'desc', 'grp', 'cost', 'col', 'binomial', 'co2Type', 'leafType',
                                         'height', 'width', 'alb', 'eps', 'leafTrans', 'isoprene', 'leafWeight',
                                         'stomResMin', 'rootDepth', 'rootDia', 'hasAgsParams', 'geometry'}
        return TreeData(plant['id'], plant['desc'], n, n, int(round(plant['height'])), cells,
                        albedo=plant['alb'], transmittance=plant['leafTrans'], emissivity=plant['eps'],
                        ags=ags, roots=roots, binomial=plant.get('binomial', '(none)'), group=plant.get('grp', ''),
                        color=plant.get('col', '00CC00'), leaf_type=plant.get('leafType', 'decid'),
                        co2_type=plant.get('co2Type', 'C3'), season_profile=geometry['seasonProfile'],
                        cost=plant.get('cost', 0), isoprene=plant.get('isoprene', 12),
                        leaf_weight=plant.get('leafWeight', 100), stom_res_min=plant.get('stomResMin', 0),
                        extra={k: v for k, v in plant.items() if k not in known})

    def insert_plant(self, tree, database_path, overwrite=False, backup=True, now=None):
        database = load(database_path)
        root = database['envimetDatafile']
        plants = root.setdefault('plants3d', [])
        index = next((n for n, p in enumerate(plants) if p.get('id') == tree.plant_id), None)
        if index is not None and not overwrite:
            raise PlantExistsError(tree.plant_id)
        if index is None:
            plants.append(plant_object(tree))
            action = 'inserted'
        else:
            plants[index] = plant_object(tree, base=plants[index])
            action = 'replaced'
        root.setdefault('header', OrderedDict())['revisionDate'] = utc_stamp(now)
        backup_path = make_backup(database_path) if backup else None
        atomic_write(database_path, dumps(database))
        return {'action': action, 'backup': backup_path}
