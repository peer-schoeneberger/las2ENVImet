# geometry_utils.py
# Utility functions for geometric analysis and transformation of point clouds

import numpy as np
from typing import Dict, Tuple


def analyze_tree_dimensions(points: np.ndarray, stem_point: np.ndarray) -> Dict[str, float]:
    # Calculate tree dimensions (in all cardinal directions and total height) based on a point cloud and the trunk position
    points_relative = points - stem_point

    north_extent = np.max(points_relative[:, 1])   # max positive Y
    south_extent = np.abs(np.min(points_relative[:, 1]))  # max negative Y
    east_extent = np.max(points_relative[:, 0])    # max positive X
    west_extent = np.abs(np.min(points_relative[:, 0]))   # max negative X
    height = np.max(points[:, 2]) - stem_point[2]   # total height

    dimensions = {
        'north': float(north_extent),
        'south': float(south_extent),
        'east': float(east_extent),
        'west': float(west_extent),
        'height': float(height),
        'total_width_x': float(east_extent + west_extent),
        'total_width_y': float(north_extent + south_extent)
    }
    return dimensions


def scale_point_cloud_to_dimensions(points: np.ndarray, stem_point: np.ndarray, target_dimensions: Dict[str, float], current_dimensions: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    # Scale a point cloud to fit target tree dimensions
    scale_x = target_dimensions['total_width_x'] / current_dimensions['total_width_x']
    scale_y = target_dimensions['total_width_y'] / current_dimensions['total_width_y']
    scale_z = target_dimensions['height'] / current_dimensions['height']

    points_relative = points - stem_point
    points_relative[:, 0] *= scale_x
    points_relative[:, 1] *= scale_y
    points_relative[:, 2] *= scale_z

    # Shift so that stem point is at Z=0
    scaled_points = points_relative + stem_point
    scaled_points[:, 2] -= stem_point[2]
    scaled_stem_point = stem_point.copy()
    scaled_stem_point[2] = 0.0

    # Recompute dimensions after scaling
    new_dimensions = analyze_tree_dimensions(scaled_points, scaled_stem_point)
    return scaled_points, scaled_stem_point, new_dimensions


def rotate_point_cloud(points: np.ndarray, stem_point: np.ndarray, angle_degrees: float) -> np.ndarray:
    # Rotate a point cloud around the trunk point (positive angle = clockwise => achieved by negating the angle before applying the standard counter-clockwise rotation matrix)
    
    angle_rad = np.radians(-angle_degrees) # Convert to radians and invert for clockwise rotation

    # 2D rotation matrix around Z‑axis
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])

    points_relative = points - stem_point
    points_rotated = points_relative @ rotation_matrix.T  # Multiply matrizes
    return points_rotated + stem_point