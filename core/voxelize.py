# voxelize.py
# Voxelisation of a point cloud around the trunk (pure numpy)
from typing import Optional, Tuple

import numpy as np


def create_voxel_grid_centered(points: np.ndarray, stem_point, voxel_size: float = 1.0
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Voxel grid whose base cell is centred on the trunk and whose floor is the trunk base
    stem_point = np.array(stem_point)

    points_centered = points.copy()
    points_centered[:, 0] = points[:, 0] - (stem_point[0] - voxel_size / 2)
    points_centered[:, 1] = points[:, 1] - (stem_point[1] - voxel_size / 2)
    points_centered[:, 2] = points[:, 2] - stem_point[2]

    voxel_indices = np.floor(points_centered / voxel_size).astype(int)
    unique_voxels, counts = np.unique(voxel_indices, axis=0, return_counts=True)

    voxel_corners_world = np.zeros_like(unique_voxels, dtype=float)
    voxel_corners_world[:, 0] = unique_voxels[:, 0] * voxel_size + (stem_point[0] - voxel_size / 2)
    voxel_corners_world[:, 1] = unique_voxels[:, 1] * voxel_size + (stem_point[1] - voxel_size / 2)
    voxel_corners_world[:, 2] = unique_voxels[:, 2] * voxel_size

    return voxel_indices, unique_voxels, voxel_corners_world, counts


def thin_voxel_grid(points: np.ndarray, cell: float, extra: Optional[dict] = None):
    # Thin a cloud:
    if cell <= 0 or len(points) == 0:
        return points, dict(extra or {})
    keys = np.floor((points - points.min(axis=0)) / cell).astype(np.int64)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, first = np.unique(flat, return_index=True)
    first.sort()
    return points[first], {k: v[first] for k, v in (extra or {}).items() if v is not None}
