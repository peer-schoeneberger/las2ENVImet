# lad_utils.py
# Utility functions for LAD calculation and refinement

import numpy as np
from typing import List, Dict, Tuple, Optional


def calculate_lad_values_multiple_references(point_counts: np.ndarray, reference_voxels: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    # Calculate LAD values for all voxels based on multiple reference voxels (average of ratio: LAD / point_count is used as scaling factor)
    if not reference_voxels:
        raise ValueError("No reference voxels provided.")

    ref_indices = [ref['index'] for ref in reference_voxels]
    ref_lad_values = [ref['lad_value'] for ref in reference_voxels]

    # Calculate scaling factor as mean of LAD/point_count ratios
    ratios = []
    for ref in reference_voxels:
        pts = ref['point_count']
        if pts > 0:
            ratios.append(ref['lad_value'] / pts)
        else:
            print(f"Warning: Reference voxel {ref['index']} has zero points – ignored.")

    if not ratios:
        raise ValueError("No valid reference voxels with point_count > 0.")

    scaling_factor = sum(ratios) / len(ratios)

    lad_values = np.zeros_like(point_counts, dtype=float)

    # Set reference voxels to their fixed LAD values
    for idx, lad in zip(ref_indices, ref_lad_values):
        lad_values[idx] = lad

    # Scale non‑reference voxels
    non_ref_mask = np.ones(len(point_counts), dtype=bool)
    non_ref_mask[ref_indices] = False
    lad_values[non_ref_mask] = point_counts[non_ref_mask] * scaling_factor
    print(f"  All voxels: min={np.min(lad_values):.3f}, max={np.max(lad_values):.3f}, mean={np.mean(lad_values):.3f}")

    return lad_values, reference_voxels


def apply_stem_only_option(voxel_corners: np.ndarray, lad_values: np.ndarray, stem_voxel_index: int, voxel_size: float = 1.0, tolerance: float = 0.01) -> np.ndarray:
    # Set all voxels in the lowest layer (z ≈ 0) to LAD = 0, except the trunk voxel
    z0_mask = np.abs(voxel_corners[:, 2]) < tolerance

    if not np.any(z0_mask):
        return lad_values  # No ground voxels => nothing to do

    # Set all ground voxels to zero, except the stem voxel
    for i in np.where(z0_mask)[0]:
        if i != stem_voxel_index:
            lad_values[i] = 0.0

    return lad_values


def apply_lad_threshold(lad_values: np.ndarray, threshold: float) -> np.ndarray:
    # Set all LAD values below a given threshold to zero
    if threshold > 0:
        original_nonzero = np.sum(lad_values > 0)
        lad_values[lad_values < threshold] = 0.0
        new_nonzero = np.sum(lad_values > 0)
        print(f"LAD threshold {threshold} applied: "
              f"{original_nonzero - new_nonzero} voxels set to 0")
    return lad_values