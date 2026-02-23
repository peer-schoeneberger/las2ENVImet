# tree_model.py
import numpy as np

from qgis.core import QgsVectorLayer, QgsFeature, QgsGeometry, QgsField, QgsPointXY
from qgis.PyQt.QtCore import QVariant

from .voxel_utils import create_voxel_grid_centered
from .geometry_utils import analyze_tree_dimensions, scale_point_cloud_to_dimensions, rotate_point_cloud
from .lad_utils import calculate_lad_values_multiple_references


class TreeModel:
    # Central data model for tree processing

    def __init__(self):
        self.points = None # Original point cloud; numpy array for las points = Nx3
        self.las_object = None # LAS (for ROI filtering)
        self.stem_point = None # Selected stem point (x, y, z)
        self.voxel_indices = None # Integer indices of each point in the voxel grid
        self.unique_voxels = None # Integer indices of voxels (numpy array for voxel = Mx3)
        self.voxel_corners = None # World coordinates of the lower corner of each voxel
        self.point_counts = None # Number of points per Voxel
        self.lad_values = None # LAD values per voxel
        self.ref_voxels = [] # List of reference voxels (dicts with keys: index, point_count, lad_value, x, y, z)
        self.current_dims = None # Original tree dimensions (from analyze_tree_dimensions)
        self.transformed_points = None  # Transformed (scaled + rotated) point cloud for preview and voxelization
        self.transformed_stem_point = None # Stem point after transformation (used for voxelization)
        self.base_lad = None # Base LAD after calibration (unchanged by refinements)
        self.stem_voxel_index = None # Index of the voxel containing the transformed stem point


    def load_points(self, points, las_object=None):
        # Load point cloud and optionally store the LAS object
        self.points = points
        self.las_object = las_object
        if self.stem_point is None:
            z_min = np.min(points[:, 2])
            self.stem_point = np.array([0.0, 0.0, z_min])

    def set_stem(self, x, y):
        # Set the stem point to (x, y) and the minimum Z of the current point cloud and compute the inital tree dimensions
        z_min = np.min(self.points[:, 2])
        self.stem_point = np.array([x, y, z_min])
        self.current_dims = analyze_tree_dimensions(self.points, self.stem_point)

    def get_points_in_polygon_mask(self, polygon_geom):
        # Return a boolean mask for points inside the given polygon
        if self.points is None:
            return None
        poly = polygon_geom.asPolygon()[0] #extract outer ring from polygon
        poly_xy = [(p.x(), p.y()) for p in poly]
        points_xy = self.points[:, :2] # extract all xy coordinates (z not inclusive)
        try:
            from matplotlib.path import Path
            path = Path(poly_xy)
            return path.contains_points(points_xy) # bool array
        except ImportError:
            return self._points_in_polygon(points_xy, poly_xy)

    def transform_points(self, target_dims, rotation_angle):
        # Scale and rotate point cloud using the target dimensions and rotation angle
        if self.points is None or self.stem_point is None:
            return None
        scaled, scaled_stem, _ = scale_point_cloud_to_dimensions(
            self.points, self.stem_point, target_dims, self.current_dims
        )
        rotated = rotate_point_cloud(scaled, scaled_stem, rotation_angle)
        self.transformed_points = rotated
        self.transformed_stem_point = scaled_stem
        return rotated

    def create_voxels(self, voxel_size=1.0):
        # Create a voxel grid from the transformed point cloud (Voxel indices are shiftet to start at 0 in Z)
        if self.transformed_points is None:
            raise ValueError("No transformed points available. Call transform_points first.")

        indices, unique, corners, counts = create_voxel_grid_centered(
            self.transformed_points, self.transformed_stem_point, voxel_size
        )

        # Shift Z to start at 0
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
        self.ref_voxels = []

        # Find the voxel that contains the transformed stem point
        self.stem_voxel_index = None
        for i, corner in enumerate(corners):
            if (corner[0] <= self.transformed_stem_point[0] <= corner[0] + voxel_size and
                corner[1] <= self.transformed_stem_point[1] <= corner[1] + voxel_size and
                corner[2] <= self.transformed_stem_point[2] <= corner[2] + voxel_size):
                self.stem_voxel_index = i
                break

        return unique, corners, counts

    def add_reference_voxel(self, world_x, world_y, z_level, lad_value):
        # Add a reference voxel by finding the closest voxel in the given Z level
        mask = np.abs(self.voxel_corners[:, 2] - z_level) < 0.01
        candidates = self.voxel_corners[mask]
        if len(candidates) == 0:
            return None

        centers = candidates + 0.5  # because voxel size = 1m
        dist2 = (centers[:, 0] - world_x) ** 2 + (centers[:, 1] - world_y) ** 2
        idx_in_layer = np.argmin(dist2)
        global_idx = np.where(mask)[0][idx_in_layer]

        # Check if already in list
        for ref in self.ref_voxels:
            if ref['index'] == global_idx:
                return None

        ref = {
            'index': global_idx,
            'point_count': self.point_counts[global_idx],
            'lad_value': lad_value,
            'x': self.voxel_corners[global_idx, 0],
            'y': self.voxel_corners[global_idx, 1],
            'z': self.voxel_corners[global_idx, 2]
        }
        self.ref_voxels.append(ref)
        return ref

    def remove_reference_voxel(self, idx_in_list):
        # Remove a reference voxel by its position in the list
        if 0 <= idx_in_list < len(self.ref_voxels):
            del self.ref_voxels[idx_in_list]
            return True
        return False

    def get_z_levels(self):
        # Return all existing Z levels from the voxel corners
        if self.voxel_corners is None:
            return []
        return np.unique(self.voxel_corners[:, 2])

    def get_voxels_at_z(self, z):
        # For a given Z level return: boolean mask of voxels in that level, corner coordinates and point counts
        if self.voxel_corners is None:
            return None, None, None
        mask = np.abs(self.voxel_corners[:, 2] - z) < 0.01
        return mask, self.voxel_corners[mask], self.point_counts[mask]

    def apply_refinement(self):
        # Calibrate LAD using the reference voxels and store the result as base LAD
        if not self.ref_voxels or self.point_counts is None:
            return False
        new_lad, _ = calculate_lad_values_multiple_references(self.point_counts, self.ref_voxels)
        self.lad_values = new_lad
        self.base_lad = new_lad.copy()
        return True

    def filter_points_by_polygon(self, polygon_geom):
        # Only keep points inside the given polygon. Resets all data (trunk point, dimensions, voxels etc.)
        mask = self.get_points_in_polygon_mask(polygon_geom)
        if mask is None or np.sum(mask) == 0:
            return False
        self.points = self.points[mask]

        # Reset derived data
        self.stem_point = None
        self.current_dims = None
        self.transformed_points = None
        self.transformed_stem_point = None
        self.voxel_corners = None
        self.voxel_indices = None
        self.unique_voxels = None
        self.point_counts = None
        self.lad_values = None
        self.ref_voxels = []
        self.base_lad = None
        self.stem_voxel_index = None

        return True

    def _points_in_polygon(self, points, poly):
        # Simple fallback algorithm, if matplotlib isnt installed
        x, y = points[:, 0], points[:, 1]
        n = len(poly)
        inside = np.zeros(len(points), dtype=bool)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            cond = ((y1 > y) != (y2 > y))
            xinters = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            inside[cond & (x < xinters)] = ~inside[cond & (x < xinters)]
        return inside

    def get_filtered_points_layer(self, crs_authid, layer_name="ROI Points"):
        # Create temp. vector point layer from current points => Display the ROI selection
        if self.points is None or len(self.points) == 0:
            return None
        vl = QgsVectorLayer(f"Point?crs={crs_authid}", layer_name, "memory")
        provider = vl.dataProvider()
        provider.addAttributes([QgsField("orig_index", QVariant.Int)])
        vl.updateFields()
        features = []
        for i, p in enumerate(self.points):
            fet = QgsFeature()
            fet.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(p[0], p[1])))
            fet.setAttributes([i])
            features.append(fet)
        provider.addFeatures(features)
        vl.updateExtents()
        return vl