# geometry.py
# Geometric analysis and transformation of point clouds
from typing import Dict, Sequence, Tuple

import numpy as np

def analyze_tree_dimensions(points: np.ndarray, stem_point: np.ndarray) -> Dict[str, float]:
    # Extent of the cloud in the four cardinal directions and total height, relative to the trunk.
    points_relative = points - stem_point

    north_extent = np.max(points_relative[:, 1])
    south_extent = np.abs(np.min(points_relative[:, 1]))
    east_extent = np.max(points_relative[:, 0])
    west_extent = np.abs(np.min(points_relative[:, 0]))
    height = np.max(points[:, 2]) - stem_point[2]

    return {
        'north': float(north_extent),
        'south': float(south_extent),
        'east': float(east_extent),
        'west': float(west_extent),
        'height': float(height),
        'total_width_x': float(east_extent + west_extent),
        'total_width_y': float(north_extent + south_extent),
    }


def scale_point_cloud_to_dimensions(points: np.ndarray, stem_point: np.ndarray,
                                    target_dimensions: Dict[str, float],
                                    current_dimensions: Dict[str, float]
                                    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    # Scale a cloud around the trunk to the target height / widths; the trunk base ends at z = 0.
    scale_x = target_dimensions['total_width_x'] / current_dimensions['total_width_x']
    scale_y = target_dimensions['total_width_y'] / current_dimensions['total_width_y']
    scale_z = target_dimensions['height'] / current_dimensions['height']

    points_relative = points - stem_point
    points_relative[:, 0] *= scale_x
    points_relative[:, 1] *= scale_y
    points_relative[:, 2] *= scale_z

    scaled_points = points_relative + stem_point
    scaled_points[:, 2] -= stem_point[2]
    scaled_stem_point = stem_point.copy()
    scaled_stem_point[2] = 0.0

    new_dimensions = analyze_tree_dimensions(scaled_points, scaled_stem_point)
    return scaled_points, scaled_stem_point, new_dimensions


def rotate_point_cloud(points: np.ndarray, stem_point: np.ndarray, angle_degrees: float) -> np.ndarray:
    # Rotate around the vertical axis through the trunk, positive angles turn clockwise.
    angle_rad = np.radians(-angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1],
    ])
    points_relative = points - stem_point
    points_rotated = points_relative @ rotation_matrix.T
    return points_rotated + stem_point


def points_in_polygon(points_xy: np.ndarray, ring: Sequence[Tuple[float, float]]) -> np.ndarray:
    # Boolean mask of the points inside a polygon ring (list of (x, y) vertices)
    points_xy = np.asarray(points_xy, dtype=float)[:, :2]
    ring = [(float(x), float(y)) for x, y in ring]
    if len(ring) < 3 or len(points_xy) == 0:
        return np.zeros(len(points_xy), dtype=bool)
    try:
        from matplotlib.path import Path
        return np.asarray(Path(ring).contains_points(points_xy), dtype=bool)
    except ImportError:
        pass
    x, y = points_xy[:, 0], points_xy[:, 1]
    inside = np.zeros(len(points_xy), dtype=bool)
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if y1 == y2:
            continue
        crosses = (y1 > y) != (y2 > y)
        x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= crosses & (x < x_cross)
    return inside


def z_percent_mask(points: np.ndarray, lower_percent: float, upper_percent: float) -> np.ndarray:
    # Mask of the points whose height lies between two percentages of the total height range.
    z = points[:, 2]
    z_min, z_max = np.min(z), np.max(z)
    lower = z_min + (lower_percent / 100.0) * (z_max - z_min)
    upper = z_min + (upper_percent / 100.0) * (z_max - z_min)
    return (z >= lower) & (z <= upper)
