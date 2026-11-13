# base.py
# Exporter interface and helpers shared by all ENVI-met format versions (no QGIS dependency)
import datetime
import os
import re
import shutil
import string
from typing import List, Optional

FORMAT_LEGACY = 'envimet5'    # ENVI-met 5.x: XML-like text (<ENVI-MET_Datafile>)
FORMAT_JSON6 = 'envimet6'     # ENVI-met 6.x: JSON ({"envimetDatafile": ...})

ID_LENGTH = 6
ID_ALPHABET = string.digits + string.ascii_uppercase


class ExportError(Exception):
    pass


class PlantExistsError(ExportError):
    # The database already contains a plant with this id and overwriting was not confirmed.

    def __init__(self, plant_id):
        super().__init__("A plant with the ID '%s' already exists in the database." % plant_id)
        self.plant_id = plant_id


class Exporter:
    # One subclass per ENVI-met format version.

    format_id = ''
    label = '' # Shown in the version selector
    single_file_extension = ''
    square_grid = False

    def write_plant_file(self, tree, path: str) -> str:
        # Write the tree as a stand-alone file (new file). Returns the path.
        raise NotImplementedError

    def list_plant_ids(self, database_path: str) -> List[str]:
        raise NotImplementedError

    def insert_plant(self, tree, database_path: str, overwrite: bool = False, backup: bool = True) -> dict:
        # Insert the tree into an existing database or replace the plant with the same id
        raise NotImplementedError

    def create_database(self, path: str) -> str:
        # Create an empty database of this format.
        raise NotImplementedError


def read_text(path: str):
    # Read a text file as UTF-8 (BOM kept out of the text). Returns (text, had_bom, encoding).
    with open(path, 'rb') as fh:
        raw = fh.read()
    had_bom = raw.startswith(b'\xef\xbb\xbf')
    if had_bom:
        raw = raw[3:]
    try:
        return raw.decode('utf-8'), had_bom, 'utf-8'
    except UnicodeDecodeError:
        return raw.decode('cp1252'), had_bom, 'cp1252'


def detect_format(path: str) -> Optional[str]:
    # Format of an existing ENVI-met database: FORMAT_JSON6, FORMAT_LEGACY or None (unknown).
    try:
        with open(path, 'rb') as fh:
            head = fh.read(4096)
    except OSError:
        return None
    if head.startswith(b'\xef\xbb\xbf'):
        head = head[3:]
    head = head.lstrip()
    if head.startswith(b'{'):
        return FORMAT_JSON6 if b'envimetDatafile' in head or len(head) < 20 else None
    if head.startswith(b'<'):
        return FORMAT_LEGACY if b'ENVI-MET_Datafile' in head else None
    return None


def make_backup(path: str, now: Optional[datetime.datetime] = None) -> str:
    # Copy <file> to <file>.bak_YYYYMMDD-HHMMSS next to it and return the backup path.
    now = now or datetime.datetime.now()
    backup = '%s.bak_%s' % (path, now.strftime('%Y%m%d-%H%M%S'))
    counter = 1
    while os.path.exists(backup):
        counter += 1
        backup = '%s.bak_%s_%d' % (path, now.strftime('%Y%m%d-%H%M%S'), counter)
    shutil.copy2(path, backup)
    return backup


def normalise_id(text: str) -> str:
    return (text or '').replace('_', '').strip().upper()


def is_valid_id(plant_id: str) -> bool:
    return len(plant_id) == ID_LENGTH and all(ch in ID_ALPHABET for ch in plant_id)


def suggest_free_id(existing, prefix: str = 'L2E') -> str:
    # First unused id of the form <prefix><counter>, e.g. L2E001; falls back to base-36 counting.
    taken = {normalise_id(i) for i in existing}
    prefix = re.sub('[^0-9A-Z]', '', prefix.upper())[:ID_LENGTH - 1]
    digits = ID_LENGTH - len(prefix)
    for n in range(1, 36 ** digits):
        if n < 10 ** digits:
            candidate = prefix + str(n).zfill(digits)
        else:
            value, chars = n, ''
            while value:
                value, rem = divmod(value, 36)
                chars = ID_ALPHABET[rem] + chars
            candidate = prefix + chars.zfill(digits)[-digits:]
        if candidate not in taken:
            return candidate
    raise ExportError("No free plant ID left for the prefix '%s'." % prefix)


def atomic_write(path: str, data: bytes):
    # Write through a temporary file in the same folder so a crash cannot truncate the database.
    tmp = path + '.l2e_tmp'
    with open(tmp, 'wb') as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
