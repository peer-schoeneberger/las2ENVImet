# Changelog

## 2.0.1 (2026-11)

- E57: the check whether `pye57` works with the installed numpy no longer starts a child process
  (E57 is disabled for numpy older than 1.20).
- Small changes for the security and Qt6 checks of the QGIS plugin repository; no change in function.

## 2.0 (2026-11)

### ENVI-met versions
- New: export for **ENVI-met 6** (JSON database). All fields ENVI-met 6 writes for a grid tree are
  filled; files are written with the `json` module in the layout of SPACES/Albero.
- ENVI-met 5.x export kept; with default settings the output is byte-identical to version 1.2.
- Support for ENVI-met 4.x is no longer tested.
- Version selector in the export step (default: ENVI-met 6). The format of an existing database is
  detected automatically. A contradiction to the selection is reported and the database's own
  format is used.
- Export as a single-plant file or insert/replace in a project or user database.
- Automatic backup `<database>.bak_YYYYMMDD-HHMMSS` before every database change.
- Plant IDs: suggestion of a free ID, collision check, replacing only after confirmation.
- Replacing a plant keeps fields the plugin does not know (and, in 5.x databases, hand-edited wood
  mechanics values); other database content is not touched.
- ENVI-met 6 plant grids are square (the format stores one width), optional for ENVI-met 5.x.
- Both exporters were checked field by field against a tree created by hand in Albero 5.9 and 6.0.

### QGIS compatibility
- One code base for QGIS 3.18 – 4.x, Qt5 and Qt6 (`qgisMinimumVersion=3.18`, `qgisMaximumVersion=4.99`).
  Tested on Windows with QGIS 3.18.1, 3.42.0 (Qt6), 3.44.15 and 4.2.3.
- Loading and voxelising run as background tasks with progress bar and cancel button.

### Foliage
- New modes **target LAI** and **total leaf area** next to the reference-voxel method.
  LAI refers to the crown projection area by default, optionally to every column.
- Live key figures (LAI, total leaf area, maximum and mean LAD) and a warning for voxels above an
  adjustable LAD limit (default 3 m²/m³).
- Orientation values for the LAI of urban trees in the tooltip (with source).

### Roots
- New step **Roots**: shape (cylinder, half ellipsoid, heart, tap, plate, user-defined), depth,
  radius (absolute or factor of the crown radius), vertical distribution (uniform, linear,
  exponential, user-defined), mean root area density or total root area, side-view preview.
- Vertical distribution also as logistic model with the 50 % and 95 % rooting depths (Schenk &
  Jackson 2002).
- Presets with literature source (see `docs/root_presets.md`): vertical distribution by forest
  zone, root system type and depth for beech, oak, Scots pine and Norway spruce, root spread relative
  to the crown. No species presets without a checked source; own presets can be saved under any name.
- Roots are exported to the RAD and root-range profiles of both formats.
- Fixed for defined root systems: version 1.2 wrote the grid width into the root depth field of
  the 5.x format. (Without a defined root system the 1.2 values are kept for comparability.)

### Point clouds
- New formats: COPC, E57, PLY, PCD, OBJ (vertices), SBF (CloudCompare) and text files (XYZ, TXT, CSV, PTS) with delimiter detection and
  column assignment; LAS/LAZ 1.0–1.4 as before.
- Point cloud layers already loaded in QGIS can be used directly (coordinate system taken over).
- Class filter, loading only the current map extent, thinning.
- Missing optional packages disable only their format; the first page shows the installation command.

### User interface
- Step-by-step wizard with a state marker per step, log panel, tooltips with units for every
  parameter, help button per step.
- Project files (`*.l2e.json`) save and restore all settings and repeat the processing.
- Errors are reported as readable messages in the log instead of Python tracebacks.
- Trunk position can be suggested automatically.
- German translation.

### Internals
- Domain logic (`core/`), readers and exporters are independent of QGIS.
- The dialog is built in Python; `resources.py` and the `.ui` file are gone.

## 1.2 (2026-04)

- Compatibility with QGIS 4, unified decimal separator, security fix in the package installer.
