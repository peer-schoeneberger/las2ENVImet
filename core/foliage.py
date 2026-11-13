# foliage.py
# Foliage
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MODE_REFERENCE = 'lad_reference'     # calibrate with reference voxels
MODE_TARGET_LAI = 'target_lai'       # Scale the relative distribution to a target LAI
MODE_LEAF_AREA = 'total_leaf_area'   # Scale the relative distribution to a total leaf area
MODES = (MODE_REFERENCE, MODE_TARGET_LAI, MODE_LEAF_AREA)

LAI_CROWN = 'crown'      # LAI = total leaf area / crown projection area
LAI_COLUMN = 'column'    # Every x/y column is scaled to the target LAI individually
DEFAULT_LAD_MAX = 3.0    # m2/m3, warning threshold


# 1.2 calibration

def calculate_lad_values_multiple_references(point_counts: np.ndarray,
                                             reference_voxels: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    # LAD of every voxel from point counts and reference voxels with known LAD
    if not reference_voxels:
        raise ValueError("No reference voxels provided.")

    ref_indices = [ref['index'] for ref in reference_voxels]
    ref_lad_values = [ref['lad_value'] for ref in reference_voxels]

    ratios = []
    for ref in reference_voxels:
        pts = ref['point_count']
        if pts > 0:
            ratios.append(ref['lad_value'] / pts)
    if not ratios:
        raise ValueError("No valid reference voxels with point_count > 0.")

    scaling_factor = sum(ratios) / len(ratios)

    lad_values = np.zeros_like(point_counts, dtype=float)
    for idx, lad in zip(ref_indices, ref_lad_values):
        lad_values[idx] = lad

    non_ref_mask = np.ones(len(point_counts), dtype=bool)
    non_ref_mask[ref_indices] = False
    lad_values[non_ref_mask] = point_counts[non_ref_mask] * scaling_factor
    return lad_values, reference_voxels


def apply_stem_only_option(voxel_corners: np.ndarray, lad_values: np.ndarray, stem_voxel_index: int,
                           voxel_size: float = 1.0, tolerance: float = 0.01) -> np.ndarray:
    # Clear the lowest layer (z = 0) except for the trunk voxel. Modifies and returns lad_values.
    z0_mask = np.abs(voxel_corners[:, 2]) < tolerance
    if not np.any(z0_mask):
        return lad_values
    for i in np.where(z0_mask)[0]:
        if i != stem_voxel_index:
            lad_values[i] = 0.0
    return lad_values


def apply_lad_threshold(lad_values: np.ndarray, threshold: float) -> np.ndarray:
    # Set every LAD below the threshold to zero. Modifies and returns lad_values.
    if threshold > 0:
        lad_values[lad_values < threshold] = 0.0
    return lad_values


def default_refinement() -> Dict:
    return {
        'crown_factor': 1.0, 'crown_z_min': 0, 'crown_z_max': 0,
        'trunk_enhance': False, 'trunk_height': 0, 'trunk_lad': 0.0,
        'threshold': 0.0, 'clear_ground': False,
    }


def refine_lad(base_lad: np.ndarray, voxel_corners: np.ndarray, stem_point: Sequence[float],
               stem_voxel_index: Optional[int], settings: Dict, voxel_size: float = 1.0) -> np.ndarray:
    # Apply crown factor, trunk enhancement, threshold and ground clearing to a copy of base_lad
    lad = base_lad.copy()
    z = voxel_corners[:, 2]

    crown = (settings['crown_z_min'] <= z) & (z <= settings['crown_z_max'])
    lad[crown] *= settings['crown_factor']

    if settings.get('trunk_enhance'):
        stem_x, stem_y = stem_point[0], stem_point[1]
        trunk = ((z <= settings['trunk_height']) &
                 (voxel_corners[:, 0] <= stem_x) & (stem_x <= voxel_corners[:, 0] + voxel_size) &
                 (voxel_corners[:, 1] <= stem_y) & (stem_y <= voxel_corners[:, 1] + voxel_size))
        lad[trunk] = settings['trunk_lad']

    threshold = settings.get('threshold', 0.0)
    if threshold > 0:
        lad[lad < threshold] = 0.0

    if settings.get('clear_ground') and stem_voxel_index is not None:
        lad = apply_stem_only_option(voxel_corners, lad, stem_voxel_index, voxel_size=voxel_size)
    return lad


# LAI / leaf area

def _columns(voxel_xy: np.ndarray) -> Tuple[np.ndarray, int]:
    # Column id of every voxel (same x/y -> same id) and the number of columns.
    xy = np.round(np.asarray(voxel_xy, dtype=float)[:, :2], 6)
    _, inverse = np.unique(xy, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    return inverse, int(inverse.max()) + 1 if len(inverse) else 0


def total_leaf_area(lad: np.ndarray, voxel_size: float = 1.0) -> float:
    return float(np.sum(lad) * voxel_size ** 3)


def crown_projection_area(lad: np.ndarray, voxel_xy: np.ndarray, voxel_size: float = 1.0) -> float:
    # Ground area covered by columns that contain at least one voxel with LAD > 0.
    filled = np.asarray(lad) > 0
    if not np.any(filled):
        return 0.0
    _, n = _columns(np.asarray(voxel_xy)[filled])
    return float(n * voxel_size ** 2)


def column_lai(lad: np.ndarray, voxel_xy: np.ndarray, voxel_size: float = 1.0) -> np.ndarray:
    # LAI of every x/y column that contains leaves.
    filled = np.asarray(lad) > 0
    if not np.any(filled):
        return np.zeros(0)
    col, n = _columns(np.asarray(voxel_xy)[filled])
    return np.bincount(col, weights=np.asarray(lad, dtype=float)[filled], minlength=n) * voxel_size


def crown_lai(lad: np.ndarray, voxel_xy: np.ndarray, voxel_size: float = 1.0) -> float:
    area = crown_projection_area(lad, voxel_xy, voxel_size)
    return total_leaf_area(lad, voxel_size) / area if area > 0 else 0.0


def _scale_once_leaf_area(weights, target_leaf_area, voxel_size):
    return weights * (target_leaf_area / (float(np.sum(weights)) * voxel_size ** 3))


def _scale_once_lai(weights, voxel_xy, target_lai, voxel_size, reference):
    if reference == LAI_CROWN:
        area = crown_projection_area(weights, voxel_xy, voxel_size)
        return _scale_once_leaf_area(weights, target_lai * area, voxel_size)
    col, n = _columns(voxel_xy)
    sums = np.bincount(col, weights=weights, minlength=n) * voxel_size
    factor = np.zeros(n)
    nonzero = sums > 0
    factor[nonzero] = target_lai / sums[nonzero]
    return weights * factor[col]


def _scale_with_cutoff(weights, scale, min_lad):
    # Scale, then drop cells that would end at or below min_lad and scale the rest again
    weights = np.array(weights, dtype=float)
    for _ in range(50):
        if float(np.sum(weights)) <= 0:
            return np.zeros_like(weights)
        lad = scale(weights)
        small = (lad <= min_lad) & (weights > 0)
        if min_lad <= 0 or not np.any(small):
            return lad
        weights[small] = 0.0
    return scale(weights)


def scale_to_leaf_area(weights: np.ndarray, target_leaf_area: float, voxel_size: float = 1.0,
                       min_lad: float = 0.0) -> np.ndarray:
    # Keep the relative distribution, scale so that sum(LAD * V) equals the target leaf area.
    weights = np.asarray(weights, dtype=float)
    if float(np.sum(weights)) <= 0:
        raise ValueError("The relative distribution is empty (all weights are zero).")
    if target_leaf_area < 0:
        raise ValueError("The total leaf area must not be negative.")
    return _scale_with_cutoff(weights, lambda w: _scale_once_leaf_area(w, target_leaf_area, voxel_size), min_lad)


def scale_to_lai(weights: np.ndarray, voxel_xy: np.ndarray, target_lai: float,
                 voxel_size: float = 1.0, reference: str = LAI_CROWN, min_lad: float = 0.0) -> np.ndarray:
    # Keep the relative distribution, scale to a target LAI
    weights = np.asarray(weights, dtype=float)
    if target_lai < 0:
        raise ValueError("The target LAI must not be negative.")
    if float(np.sum(weights)) <= 0:
        raise ValueError("The relative distribution is empty (all weights are zero).")
    if reference not in (LAI_CROWN, LAI_COLUMN):
        raise ValueError("Unknown LAI reference: %r" % (reference,))
    return _scale_with_cutoff(
        weights, lambda w: _scale_once_lai(w, voxel_xy, target_lai, voxel_size, reference), min_lad)


def foliage_metrics(lad: np.ndarray, voxel_xy: np.ndarray, voxel_size: float = 1.0,
                    lad_max: float = DEFAULT_LAD_MAX) -> Dict:
    # Key figures shown live in the dialog and stored in reports.
    lad = np.asarray(lad, dtype=float)
    filled = lad > 0
    cols = column_lai(lad, voxel_xy, voxel_size)
    area = crown_projection_area(lad, voxel_xy, voxel_size)
    leaf_area = total_leaf_area(lad, voxel_size)
    return {
        'cells': int(np.sum(filled)),
        'total_leaf_area': leaf_area,
        'crown_projection_area': area,
        'lai': leaf_area / area if area > 0 else 0.0,
        'lai_column_max': float(cols.max()) if len(cols) else 0.0,
        'lai_column_mean': float(cols.mean()) if len(cols) else 0.0,
        'lad_max': float(lad.max()) if len(lad) else 0.0,
        'lad_mean': float(lad[filled].mean()) if np.any(filled) else 0.0,
        'cells_above_max': int(np.sum(lad > lad_max)),
        'lad_limit': float(lad_max),
    }


def foliage_warnings(metrics: Dict) -> List[str]:
    # Plain-language plausibility messages (untranslated keys with values for the GUI).
    out = []
    if metrics['cells'] == 0:
        out.append('no_leaves')
    if metrics['cells_above_max'] > 0:
        out.append('lad_above_max')
    return out
