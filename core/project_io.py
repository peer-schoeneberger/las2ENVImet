# project_io.py
# Las2ENVImet project files (*.l2e.json)
import copy
import datetime
import json
import os

FILE_VERSION = 1
EXTENSION = '.l2e.json'

DEFAULTS = {
    'source': {'path': '', 'format_options': {}, 'bbox': None, 'classes': None, 'thin': None, 'crs': None},
    'roi': None,                                # Polygon ring [[x, y], ...] or None
    'z_filter': {'lower_percent': 0.0, 'upper_percent': 100.0},
    'stem': None,                               # [x, y]
    'transform': {'height': None, 'width_x': None, 'width_y': None, 'rotation': 0.0},
    'voxel': {'size': 1.0},
    'foliage': {
        'mode': 'lad_reference', 'target_lai': 4.0, 'lai_reference': 'crown', 'total_leaf_area': 100.0,
        'lad_max': 3.0, 'references': [],       # [{'x':..,'y':..,'z':..,'lad':..}]
        'refinement': {'crown_factor': 1.0, 'crown_z_min': 0, 'crown_z_max': 0, 'trunk_enhance': False,
                       'trunk_height': 0, 'trunk_lad': 0.0, 'threshold': 0.0, 'clear_ground': False},
    },
    'roots': {'enabled': False, 'preset': '', 'shape': 'cylinder', 'depth': 2.0, 'size_mode': 'factor',
              'diameter': 6.0, 'radius_factor': 1.0, 'distribution': 'uniform', 'beta': 0.966, 'd50': 21.0, 'd95': 104.0,
              'density_mode': 'rad', 'rad': 0.1, 'total_root_area': 50.0,
              'custom_range': None, 'custom_depth_profile': None},
    'physiology': {'preset': 'Abstract Deciduous Tree', 'manual': False, 'ags': {}, 'albedo': 0.18,
                   'transmittance': 0.30, 'emissivity': 0.96, 'leaf_type': 'decid',
                   'season_profile': [0.3, 0.3, 0.3, 0.4, 0.7, 1.0, 1.0, 1.0, 0.8, 0.6, 0.3, 0.3]},
    'export': {'format': 'envimet6', 'plant_id': '', 'description': '', 'binomial': '', 'square': False, 'target': 'file',
               'file': '', 'database': ''},
}


def new_project():
    return copy.deepcopy(DEFAULTS)


def _merge(defaults, data):
    out = copy.deepcopy(defaults)
    for key, value in (data or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def save_project(path, settings, plugin_version=''):
    # Write the settings. Paths inside the project folder are stored relative to the project file.
    data = copy.deepcopy(settings)
    base = os.path.dirname(os.path.abspath(path))
    src = data.get('source', {}).get('path')
    if src:
        try:
            rel = os.path.relpath(src, base)
            if not rel.startswith('..'):
                data['source']['path'] = rel.replace('\\', '/')
        except ValueError:
            pass # Different drive: keep absolute
    document = {
        'las2envimet_project': FILE_VERSION,
        'plugin_version': plugin_version,
        'saved': datetime.datetime.now().replace(microsecond=0).isoformat(),
        'settings': data,
    }
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(document, fh, indent=2, ensure_ascii=False)
    return path


def load_project(path):
    # Read a project file; missing keys are filled with defaults, so old files keep working.
    with open(path, 'r', encoding='utf-8-sig') as fh:
        try:
            document = json.load(fh)
        except ValueError as exc:
            raise ValueError("Not a valid las2ENVImet project file (%s)." % exc)
    if not isinstance(document, dict) or 'las2envimet_project' not in document:
        raise ValueError("Not a las2ENVImet project file.")
    if document['las2envimet_project'] > FILE_VERSION:
        raise ValueError("This project was saved by a newer las2ENVImet version.")
    settings = _merge(DEFAULTS, document.get('settings', {}))
    src = settings['source'].get('path')
    if src and not os.path.isabs(src):
        settings['source']['path'] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), src))
    return settings
