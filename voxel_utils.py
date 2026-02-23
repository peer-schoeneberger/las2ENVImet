# voxel_utils.py
import numpy as np

def create_voxel_grid_centered(points, stem_point, voxel_size=1.0):
    #Create a voxel grid centered on the stem point, with the base at Z=0.

    stem_point = np.array(stem_point)

    # Shift points so that the stem point lies at (voxel_size/2, voxel_size/2, 0) and thereby places the stem exactly in the center of the base voxel.
    points_centered = points.copy()
    points_centered[:, 0] = points[:, 0] - (stem_point[0] - voxel_size / 2)
    points_centered[:, 1] = points[:, 1] - (stem_point[1] - voxel_size / 2)
    points_centered[:, 2] = points[:, 2] - stem_point[2]  # Z relative to ground

    # Compute voxel indices (floor division (rounding down to the nearest integer))
    voxel_indices = np.floor(points_centered / voxel_size).astype(int)

    # Find unique voxels and count points per voxel
    unique_voxels, counts = np.unique(voxel_indices, axis=0, return_counts=True)

    # Compute world coordinates of the lower corner of each voxel
    voxel_corners_world = np.zeros_like(unique_voxels, dtype=float)
    voxel_corners_world[:, 0] = unique_voxels[:, 0] * voxel_size + (stem_point[0] - voxel_size / 2)
    voxel_corners_world[:, 1] = unique_voxels[:, 1] * voxel_size + (stem_point[1] - voxel_size / 2)
    voxel_corners_world[:, 2] = unique_voxels[:, 2] * voxel_size  # Z starts at 0

    return voxel_indices, unique_voxels, voxel_corners_world, counts