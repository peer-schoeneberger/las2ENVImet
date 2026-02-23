# export_manager.py
# Manager class for handling ENVI‑met export operations

import os
import shutil
from typing import Optional

from .file_utils import create_envi_met_txt_file, update_envi_met_database
from .tree_model import TreeModel


class ExportManager:

    @staticmethod
    def export_to_txt(model: TreeModel, plant_id: str, description: str, albedo: float, transmittance: float, emissivity: float,
        ags_params: dict, output_path: str, lad_threshold: float = 0.0) -> str:
        # Generate an ENVI‑met plant TXT file and move it to the desired location
        if model.voxel_corners is None or model.lad_values is None:
            raise ValueError("No voxel data available for export.")

        # Writes a file named '<plant_id>.txt'
        filename, xml_content = create_envi_met_txt_file(
            voxel_corners_world=model.voxel_corners,
            lad_values=model.lad_values,
            stem_point=model.stem_point,
            reference_voxels=model.ref_voxels,
            plant_id=plant_id,
            description=description,
            voxel_size=1.0,
            lad_threshold=lad_threshold,
            ags_params=ags_params,
            albedo=albedo,
            transmittance=transmittance,
            emissivity=emissivity
        )

        if os.path.exists(filename):
            shutil.move(filename, output_path)

        return xml_content

    @staticmethod
    def update_database(database_path: str, xml_content: str, plant_id: str) -> None:
        # Insert a new plant definition into an ENVI‑met projectdatabase.edb file.
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"Database not found: {database_path}")

        update_envi_met_database(database_path, xml_content, plant_id)