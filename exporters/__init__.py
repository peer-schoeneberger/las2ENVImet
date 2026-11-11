# Exporters: one class per ENVI-met format version, selected through this small registry.
from .base import (FORMAT_JSON6, FORMAT_LEGACY, ExportError, Exporter, PlantExistsError, detect_format,
                   is_valid_id, make_backup, normalise_id, suggest_free_id)
from .envimet6_json import Envimet6JsonExporter
from .envimet_legacy import EnvimetLegacyExporter

EXPORTERS = {
    FORMAT_JSON6: Envimet6JsonExporter,
    FORMAT_LEGACY: EnvimetLegacyExporter,
}
DEFAULT_FORMAT = FORMAT_JSON6


def get_exporter(format_id: str) -> Exporter:
    try:
        return EXPORTERS[format_id]()
    except KeyError:
        raise ExportError("Unknown ENVI-met format: %r" % (format_id,))


def exporter_for_database(path: str):
    # Exporter matching an existing database file, or None if the format is not recognised.
    fmt = detect_format(path)
    return get_exporter(fmt) if fmt else None
