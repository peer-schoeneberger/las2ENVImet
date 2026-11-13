# tree_model.py
# Tree model without QGIS dependency
from typing import Any, Dict, List, Optional

import numpy as np

from . import foliage
from .geometry import (analyze_tree_dimensions, points_in_polygon, rotate_point_cloud,
                       scale_point_cloud_to_dimensions)
from .voxelize import create_voxel_grid_centered

DEFAULT_SEASON_PROFILE = (0.3, 0.3, 0.3, 0.4, 0.7, 1.0, 1.0, 1.0, 0.8, 0.6, 0.3, 0.3)
DEFAULT_BLOSSOM_PROFILE = (0.0,) * 12
DEFAULT_AGS = {
    'comp_point25': 45.0, 'comp_point_q10': 1.5,
    'gm25': 5.0, 'gm_q10': 2.0, 'gm_t1': 5.0, 'gm_t2': 36.0,
    'ammax25': 2.2, 'ammax_q10': 2.0, 'ammax_t1': 8.0, 'ammax_t2': 36.0,
    'e0': 0.017, 'f0': 0.85, 'gc': 0.25, 'dmax': 20.0,
}
AGS_KEYS = tuple(DEFAULT_AGS)


class TreeWorkflow:
    # Central processing state (the former TreeModel of 1.2, minus everything QGIS).

    def __init__(self):
        self.points = None                  # Original cloud, N x 3
        self.attributes = {}                # Optional per-point attributes (classification, ...)
        self.stem_point = None              # Picked trunk position (x, y, z_min)
        self.voxel_size = 1.0
        self.voxel_indices = None
        self.unique_voxels = None           # Integer voxel indices, M x 3
        self.voxel_corners = None           # World coordinates of the lower voxel corners
        self.point_counts = None
        self.lad_values = None              # Current LAD per voxel
        self.base_lad = None                # LAD after calibration / scaling, before refinement
        self.ref_voxels = []                # Dicts: index, point_count, lad_value, x, y, z
        self.current_dims = None
        self.transformed_points = None
        self.transformed_stem_point = None
        self.stem_voxel_index = None

    # points and trunk
    def load_points(self, points, attributes=None):
        self.points = points
        self.attributes = dict(attributes or {})
        if self.stem_point is None:
            z_min = np.min(points[:, 2])
            self.stem_point = np.array([0.0, 0.0, z_min])

    def reset_derived(self, keep_stem=False):
        if not keep_stem:
            self.stem_point = None
            self.current_dims = None
        self.transformed_points = None
        self.transformed_stem_point = None
        self.voxel_corners = None
        self.voxel_indices = None
        self.unique_voxels = None
        self.point_counts = None
        self.lad_values = None
        self.base_lad = None
        self.ref_voxels = []
        self.stem_voxel_index = None

    def set_stem(self, x, y):
        z_min = np.min(self.points[:, 2])
        self.stem_point = np.array([x, y, z_min])
        self.current_dims = analyze_tree_dimensions(self.points, self.stem_point)

    def filter_points_by_ring(self, ring):
        # Keep only the points inside a polygon ring [(x, y), ...]; resets all derived data.
        if self.points is None:
            return False
        mask = points_in_polygon(self.points[:, :2], ring)
        if not np.any(mask):
            return False
        self.apply_mask(mask)
        return True

    def apply_mask(self, mask):
        self.points = self.points[mask]
        self.attributes = {k: v[mask] for k, v in self.attributes.items() if v is not None}
        self.reset_derived()

    # transform and voxels
    def transform_points(self, target_dims, rotation_angle):
        if self.points is None or self.stem_point is None:
            return None
        scaled, scaled_stem, _ = scale_point_cloud_to_dimensions(
            self.points, self.stem_point, target_dims, self.current_dims)
        rotated = rotate_point_cloud(scaled, scaled_stem, rotation_angle)
        self.transformed_points = rotated
        self.transformed_stem_point = scaled_stem
        return rotated

    def create_voxels(self, voxel_size=1.0):
        if self.transformed_points is None:
            raise ValueError("No transformed points available. Call transform_points first.")
        self.voxel_size = voxel_size
        indices, unique, corners, counts = create_voxel_grid_centered(
            self.transformed_points, self.transformed_stem_point, voxel_size)

        z_offset = np.min(unique[:, 2])
        if z_offset != 0:
            unique[:, 2] -= z_offset
            corners[:, 2] -= z_offset * voxel_size
            self.transformed_stem_point[2] -= z_offset * voxel_size

        self.voxel_indices = indices
        self.unique_voxels = unique
        self.voxel_corners = corners
        self.point_counts = counts
        self.lad_values = None
        self.base_lad = None
        self.ref_voxels = []

        stem = self.transformed_stem_point
        inside = ((corners[:, 0] <= stem[0]) & (stem[0] <= corners[:, 0] + voxel_size) &
                  (corners[:, 1] <= stem[1]) & (stem[1] <= corners[:, 1] + voxel_size) &
                  (corners[:, 2] <= stem[2]) & (stem[2] <= corners[:, 2] + voxel_size))
        hits = np.where(inside)[0]
        self.stem_voxel_index = int(hits[0]) if len(hits) else None
        return unique, corners, counts

    def get_z_levels(self):
        if self.voxel_corners is None:
            return []
        return np.unique(self.voxel_corners[:, 2])

    def get_voxels_at_z(self, z):
        if self.voxel_corners is None:
            return None, None, None
        mask = np.abs(self.voxel_corners[:, 2] - z) < 0.01
        return mask, self.voxel_corners[mask], self.point_counts[mask]

    # foliage
    def add_reference_voxel(self, world_x, world_y, z_level, lad_value):
        mask = np.abs(self.voxel_corners[:, 2] - z_level) < 0.01
        candidates = self.voxel_corners[mask]
        if len(candidates) == 0:
            return None
        centers = candidates + 0.5 * self.voxel_size
        dist2 = (centers[:, 0] - world_x) ** 2 + (centers[:, 1] - world_y) ** 2
        global_idx = np.where(mask)[0][np.argmin(dist2)]
        for ref in self.ref_voxels:
            if ref['index'] == global_idx:
                return None
        ref = {
            'index': global_idx,
            'point_count': self.point_counts[global_idx],
            'lad_value': lad_value,
            'x': self.voxel_corners[global_idx, 0],
            'y': self.voxel_corners[global_idx, 1],
            'z': self.voxel_corners[global_idx, 2],
        }
        self.ref_voxels.append(ref)
        return ref

    def remove_reference_voxel(self, idx_in_list):
        if 0 <= idx_in_list < len(self.ref_voxels):
            del self.ref_voxels[idx_in_list]
            return True
        return False

    def calibrate_lad(self):
        # LAD from point counts and reference voxels (1.2 method). Returns False if impossible.
        if not self.ref_voxels or self.point_counts is None:
            return False
        new_lad, _ = foliage.calculate_lad_values_multiple_references(self.point_counts, self.ref_voxels)
        self.lad_values = new_lad
        self.base_lad = new_lad.copy()
        return True

    def scale_foliage(self, mode, value, lai_reference=foliage.LAI_CROWN):
        # LAD from point counts scaled to a target LAI or a total leaf area (no references needed).
        if self.point_counts is None:
            return False
        weights = self.point_counts.astype(float)
        if mode == foliage.MODE_TARGET_LAI:
            lad = foliage.scale_to_lai(weights, self.voxel_corners[:, :2], value, self.voxel_size, lai_reference,
                                       min_lad=MIN_EXPORT_LAD)
        elif mode == foliage.MODE_LEAF_AREA:
            lad = foliage.scale_to_leaf_area(weights, value, self.voxel_size, min_lad=MIN_EXPORT_LAD)
        else:
            raise ValueError("Unknown foliage mode: %r" % (mode,))
        self.lad_values = lad
        self.base_lad = lad.copy()
        return True

    def refine(self, settings):
        # Apply crown factor / trunk / threshold / ground clearing to the base LAD.
        if self.base_lad is None:
            return False
        self.lad_values = foliage.refine_lad(self.base_lad, self.voxel_corners, self.stem_point,
                                             self.stem_voxel_index, settings, self.voxel_size)
        return True

    def rescale_after_refinement(self, mode, value, lai_reference=foliage.LAI_CROWN):
        # In LAI / leaf-area mode the target must still be met after refinement removed leaves.
        if self.lad_values is None or float(np.sum(self.lad_values)) <= 0:
            return False
        if mode == foliage.MODE_TARGET_LAI:
            self.lad_values = foliage.scale_to_lai(self.lad_values, self.voxel_corners[:, :2], value,
                                                   self.voxel_size, lai_reference, min_lad=MIN_EXPORT_LAD)
        elif mode == foliage.MODE_LEAF_AREA:
            self.lad_values = foliage.scale_to_leaf_area(self.lad_values, value, self.voxel_size,
                                                         min_lad=MIN_EXPORT_LAD)
        return True

    def reset_refinement(self):
        if self.base_lad is None:
            return False
        self.lad_values = self.base_lad.copy()
        return True

    def metrics(self, lad_max=foliage.DEFAULT_LAD_MAX):
        if self.lad_values is None:
            return None
        return foliage.foliage_metrics(self.lad_values, self.voxel_corners[:, :2], self.voxel_size, lad_max)


class TreeData:
    # Finished tree, independent of file format and ENVI-met version

    def __init__(self, plant_id, description, nx, ny, nz, cells, cell_size=1.0, albedo=0.18,
                 transmittance=0.30, emissivity=0.96, ags=None, roots=None, binomial='(none)',
                 group='', color='00CC00', leaf_type='decid', co2_type='C3', author='LAS2ENVImet QGIS Plugin',
                 season_profile=DEFAULT_SEASON_PROFILE, blossom_profile=DEFAULT_BLOSSOM_PROFILE,
                 cost=0.0, isoprene=12.0, leaf_weight=100.0, stom_res_min=0.0, extra=None):
        self.plant_id = plant_id
        self.description = description
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.cells = list(cells)
        self.cell_size = float(cell_size)
        self.albedo = float(albedo)
        self.transmittance = float(transmittance)
        self.emissivity = float(emissivity)
        self.ags = dict(ags) if ags else None
        self.roots = roots
        self.binomial = binomial
        self.group = group
        self.color = color
        self.leaf_type = leaf_type
        self.co2_type = co2_type
        self.author = author
        self.season_profile = tuple(float(v) for v in season_profile)
        self.blossom_profile = tuple(float(v) for v in blossom_profile)
        self.cost = float(cost)
        self.isoprene = float(isoprene)
        self.leaf_weight = float(leaf_weight)
        self.stom_res_min = float(stom_res_min)
        self.extra = dict(extra or {})

    @property
    def height(self):
        return self.nz * self.cell_size

    @property
    def width(self):
        return max(self.nx, self.ny) * self.cell_size

    def ags_complete(self):
        # A-gs parameters with defaults filled in (what the file gets when has_ags is set).
        merged = dict(DEFAULT_AGS)
        if self.ags:
            merged.update({k: float(v) for k, v in self.ags.items() if k in DEFAULT_AGS})
        return merged

    def lad_array(self):
        return np.array([c[3] for c in self.cells], dtype=float)

    def total_leaf_area(self):
        return float(self.lad_array().sum() * self.cell_size ** 3) if self.cells else 0.0

    def crown_lai(self):
        if not self.cells:
            return 0.0
        columns = {(c[0], c[1]) for c in self.cells}
        return self.total_leaf_area() / (len(columns) * self.cell_size ** 2)


MIN_EXPORT_LAD = 0.001   # ENVI-met ignores smaller values


def build_plant_grid(voxel_corners_world, lad_values, stem_point, voxel_size=1.0, lad_threshold=0.0):
    # Map world voxels to the ENVI-met plant grid
    lad_values = np.asarray(lad_values, dtype=float)
    if lad_threshold > 0:
        lad_values = foliage.apply_lad_threshold(lad_values.copy(), lad_threshold)

    min_x = np.min(voxel_corners_world[:, 0])
    min_y = np.min(voxel_corners_world[:, 1])
    min_z = np.min(voxel_corners_world[:, 2])

    shifted = voxel_corners_world.copy()
    shifted[:, 0] -= min_x
    shifted[:, 1] -= min_y
    shifted[:, 2] -= min_z
    stem_shifted = np.array(stem_point) - [min_x, min_y, min_z]

    nx = int(np.ceil(np.max(shifted[:, 0] + voxel_size) / voxel_size))
    ny = int(np.ceil(np.max(shifted[:, 1] + voxel_size) / voxel_size))
    nz = int(np.ceil(np.max(shifted[:, 2] + voxel_size) / voxel_size))
    if nx % 2 == 0:
        nx += 1
    if ny % 2 == 0:
        ny += 1

    stem_i = int(np.floor(stem_shifted[0] / voxel_size))
    stem_j = int(np.floor(stem_shifted[1] / voxel_size))
    delta_i = nx // 2 - stem_i
    delta_j = ny // 2 - stem_j
    if delta_i != 0 or delta_j != 0:
        shifted[:, 0] += delta_i * voxel_size
        shifted[:, 1] += delta_j * voxel_size

    grid_i = np.clip(np.floor(shifted[:, 0] / voxel_size).astype(int), 0, nx - 1)
    grid_j = np.clip(np.floor(shifted[:, 1] / voxel_size).astype(int), 0, ny - 1)
    grid_k = np.clip(np.floor(shifted[:, 2] / voxel_size).astype(int), 0, nz - 1)

    cells = []
    for i, j, k, lad in zip(grid_i, grid_j, grid_k, lad_values):
        if lad > MIN_EXPORT_LAD:
            cells.append((int(i), int(j), int(k), float('%.5f' % lad)))
    cells.sort(key=lambda c: (c[2], c[1], c[0]))
    return nx, ny, nz, cells


def build_tree_data(workflow: TreeWorkflow, plant_id: str, description: str, lad_threshold: float = 0.0,
                    square: bool = False, **attributes: Any) -> TreeData:
    # Create the export model from a finished workflow
    if workflow.voxel_corners is None or workflow.lad_values is None:
        raise ValueError("No voxel data available for export.")
    nx, ny, nz, cells = build_plant_grid(workflow.voxel_corners, workflow.lad_values, workflow.stem_point,
                                         workflow.voxel_size, lad_threshold)
    tree = TreeData(plant_id, description, nx, ny, nz, cells, cell_size=workflow.voxel_size, **attributes)
    return make_square(tree) if square else tree


def make_square(tree: TreeData) -> TreeData:
    # Return the tree on a square grid (nx == ny, odd), cells shifted so the trunk stays centred.
    n = max(tree.nx, tree.ny)
    if tree.nx == tree.ny:
        return tree
    di = n // 2 - tree.nx // 2
    dj = n // 2 - tree.ny // 2
    cells = [(i + di, j + dj, k, lad) for i, j, k, lad in tree.cells]
    cells.sort(key=lambda c: (c[2], c[1], c[0]))
    clone = TreeData.__new__(TreeData)
    clone.__dict__.update(tree.__dict__)
    clone.nx = clone.ny = n
    clone.cells = cells
    return clone
