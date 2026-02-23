# file_utils.py
# Utility functions for file operations, point cloud loading, and ENVI‑met export.

from __future__ import annotations

import os
import re
import tempfile
import numpy as np

from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from .lad_utils import apply_lad_threshold


# Point cloud I/O Section

def _ensure_laspy():
    # Import laspy module. Raise an ImportError if not available with ionstallation instructions
    try:
        import laspy
        return laspy
    except ImportError:
        raise ImportError("The 'laspy' Python package is required to read and write LAS/LAZ files. Please install it in the QGIS OSGeo4W shell using: pip install laspy[lazrs] ")

def load_point_cloud(file_path: str) -> Tuple[np.ndarray, laspy.LasData]:
    # Load a LAS/LAZ file and return the points as a numpy array and the laspy object
    print(f"Loading point cloud from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in ('.las', '.laz'):
        raise ValueError(f"Unsupported file format: {file_ext}. Only .las and .laz files are supported.")
    
    laspy = _ensure_laspy()

    try:
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        return points, las
    except Exception as e:
        if file_ext == '.laz':
            print("Error loading LAZ file. This may be due to missing LAZ support.")
            print("Please ensure you have laspy with LAZ support installed:")
            print("  pip install laspy[lazrs] or laspy[laszip]")
        raise e


def save_filtered_las(original_las: laspy.LasData, point_mask: np.ndarray, output_path: Optional[str] = None) -> str:
    # Create a new LAS file containing only the points selected by point_mask (ROI)
    laspy = _ensure_laspy()
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".las")

    header = laspy.LasHeader(
        point_format=original_las.header.point_format,
        version=original_las.header.version
    )
    filtered_points = original_las.points[point_mask]

    with laspy.open(output_path, mode='w', header=header) as writer:
        writer.write_points(filtered_points)

    return output_path


# ENVI‑met file creation

def create_envi_met_txt_file(voxel_corners_world: np.ndarray, lad_values: np.ndarray, stem_point: np.ndarray, reference_voxels: List[Dict[str, Any]], plant_id: str = "TREE12", 
                             description: str = "Generated using the LAS2ENVImet QGIS Plugin", voxel_size: float = 1.0, lad_threshold: float = 0.0, 
                             ags_params: Optional[Dict[str, float]] = None, albedo: float = 0.18, transmittance: float = 0.3, emissivity: float = 0.96) -> Tuple[str, str]:
    # Generate the content of an ENVI‑met plant TXT file and write it to disk

    if lad_threshold > 0:
        lad_values = apply_lad_threshold(lad_values.copy(), lad_threshold)

    # Shift all coordinates so that min_x, min_y, min_z become 0
    min_x = np.min(voxel_corners_world[:, 0])
    min_y = np.min(voxel_corners_world[:, 1])
    min_z = np.min(voxel_corners_world[:, 2])

    shifted_corners = voxel_corners_world.copy()
    shifted_corners[:, 0] -= min_x
    shifted_corners[:, 1] -= min_y
    shifted_corners[:, 2] -= min_z

    stem_shifted = np.array(stem_point) - [min_x, min_y, min_z]

    max_x = np.max(shifted_corners[:, 0] + voxel_size)
    max_y = np.max(shifted_corners[:, 1] + voxel_size)
    max_z = np.max(shifted_corners[:, 2] + voxel_size)

    nx = int(np.ceil(max_x / voxel_size))
    ny = int(np.ceil(max_y / voxel_size))
    nz = int(np.ceil(max_z / voxel_size))

    # ENVI‑met requires odd numbers in X and Y to have a clear center cell
    if nx % 2 == 0:
        nx += 1
    if ny % 2 == 0:
        ny += 1

    # Center trunk in the grid
    stem_i = int(np.floor(stem_shifted[0] / voxel_size))
    stem_j = int(np.floor(stem_shifted[1] / voxel_size))
    target_i = nx // 2
    target_j = ny // 2
    delta_i = target_i - stem_i
    delta_j = target_j - stem_j

    if delta_i != 0 or delta_j != 0:
        shifted_corners[:, 0] += delta_i * voxel_size
        shifted_corners[:, 1] += delta_j * voxel_size
        stem_shifted[0] += delta_i * voxel_size
        stem_shifted[1] += delta_j * voxel_size
        print(f"Voxels shifted by ({delta_i}, {delta_j}) cells to centre the trunk.")

    # Calcutlate final grid indices
    grid_i = np.floor(shifted_corners[:, 0] / voxel_size).astype(int)
    grid_j = np.floor(shifted_corners[:, 1] / voxel_size).astype(int)
    grid_k = np.floor(shifted_corners[:, 2] / voxel_size).astype(int)

    # Safety clip (should not be necessary)
    grid_i = np.clip(grid_i, 0, nx - 1)
    grid_j = np.clip(grid_j, 0, ny - 1)
    grid_k = np.clip(grid_k, 0, nz - 1)

    # Build LAD entries (only where LAD > 0.001, as ENVI‑met ignores smaller values)
    entries = []
    for i, j, k, lad in zip(grid_i, grid_j, grid_k, lad_values):
        if lad > 0.001:
            entries.append(f"{i},{j},{k},{lad:.5f}")

    # Sort entries by Z, then Y, then X (as ENVI‑met expects)
    entries.sort(key=lambda x: (int(x.split(',')[2]), int(x.split(',')[1]), int(x.split(',')[0])))

    default_ags = {
        'comp_point25': 45.0, 'comp_point_q10': 1.5,
        'gm25': 5.0, 'gm_q10': 2.0, 'gm_t1': 5.0, 'gm_t2': 36.0,
        'ammax25': 2.2, 'ammax_q10': 2.0, 'ammax_t1': 8.0, 'ammax_t2': 36.0,
        'e0': 0.017, 'f0': 0.85, 'gc': 0.25, 'dmax': 20.0
    }
    if ags_params:
        default_ags.update(ags_params)

    has_ags = 1 if ags_params else 0

    txt_content = f"""<PLANT3D>
     <ID> {plant_id} </ID>
     <Description> {description} </Description>
     <AlternativeName> (none) </AlternativeName>
     <Planttype> 0 </Planttype>
     <Leaftype> 1 </Leaftype>
     <Albedo> {albedo:.5f} </Albedo>
     <Eps> {emissivity:.5f} </Eps>
     <Transmittance> {transmittance:.5f} </Transmittance>
     <isoprene> 12.00000 </isoprene>
     <leafweigth> 100.00000 </leafweigth>
     <rs_min> 0.00000 </rs_min>
     <Height> {nz:.5f} </Height>
     <Width> {nx:.5f} </Width>
     <Depth> {ny:.5f} </Depth>
     <RootDiameter> 10.00000 </RootDiameter>
     <cellsize> {voxel_size:.5f} </cellsize>
     <xy_cells> {nx} </xy_cells>
     <z_cells> {nz} </z_cells>
     <scalefactor> 0.00000 </scalefactor>
     <hasAgsParams> {has_ags} </hasAgsParams>
     <agsCompPoint25> {default_ags['comp_point25']:.5f} </agsCompPoint25>
     <agsCompPointQ10> {default_ags['comp_point_q10']:.5f} </agsCompPointQ10>
     <agsGm25> {default_ags['gm25']:.5f} </agsGm25>
     <agsGmQ10> {default_ags['gm_q10']:.5f} </agsGmQ10>
     <agsGmT1> {default_ags['gm_t1']:.5f} </agsGmT1>
     <agsGmT2> {default_ags['gm_t2']:.5f} </agsGmT2>
     <agsAmmax25> {default_ags['ammax25']:.5f} </agsAmmax25>
     <agsAmmaxQ10> {default_ags['ammax_q10']:.5f} </agsAmmaxQ10>
     <agsAmmaxT1> {default_ags['ammax_t1']:.5f} </agsAmmaxT1>
     <agsAmmaxT2> {default_ags['ammax_t2']:.5f} </agsAmmaxT2>
     <agsE0> {default_ags['e0']:.5f} </agsE0>
     <agsF0> {default_ags['f0']:.5f} </agsF0>
     <agsGc> {default_ags['gc']:.5f} </agsGc>
     <agsDmax> {default_ags['dmax']:.5f} </agsDmax>
     <LAD-Profile type="sparematrix-3D" dataI="{nx}" dataJ="{ny}" zlayers="{nz}" defaultValue="0.00000">
"""
    for entry in entries:
        txt_content += f"     {entry}\n"

    txt_content += """     </LAD-Profile>
     <RAD-Profile> 0.10000,0.10000,0.10000,0.10000,0.10000,0.10000,0.10000,0.10000,0.10000,0.10000 </RAD-Profile>
     <Root-Range-Profile> 1.00000,1.00000,1.00000,1.00000,1.00000,1.00000,1.00000,1.00000,1.00000,1.00000 </Root-Range-Profile>
     <Season-Profile> 0.30000,0.30000,0.30000,0.40000,0.70000,1.00000,1.00000,1.00000,0.80000,0.60000,0.30000,0.30000 </Season-Profile>
     <Blossom-Profile> 0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000 </Blossom-Profile>
     <DensityWood> 690.00000 </DensityWood>
     <YoungsModulus> 8770000896.00000 </YoungsModulus>
     <YoungRatioRtoL> 0.12000 </YoungRatioRtoL>
     <MORBranch> 65.00000 </MORBranch>
     <MORConnection> 45.00000 </MORConnection>
     <PlantGroup> 0 </PlantGroup>
     <Color> 52224 </Color>
     <Group>  </Group>
     <Author> LAS2ENVImet QGIS Plugin </Author>
     <costs> 0.00000 </costs>
     <ColorStem> 0 </ColorStem>
     <ColorBlossom> 0 </ColorBlossom>
     <BlossomRadius> 0.00000 </BlossomRadius>
     <L-SystemBased> 0 </L-SystemBased>
     <Axiom> V </Axiom>
     <IterationDepth> 0 </IterationDepth>
     <hasUserEdits> 0 </hasUserEdits>
     <LADMatrix_generated> 1 </LADMatrix_generated>
     <InitialSegmentLength> 0.00000 </InitialSegmentLength>
     <SmallSegmentLength> 0.00000 </SmallSegmentLength>
     <ChangeSegmentLength> 0.00000 </ChangeSegmentLength>
     <SegmentResolution> 0.00000 </SegmentResolution>
     <TurtleAngle> 0.00000 </TurtleAngle>
     <RadiusOuterBranch> 0.00000 </RadiusOuterBranch>
     <PipeFactor> 0.00000 </PipeFactor>
     <LeafPosition> 0 </LeafPosition>
     <LeafsPerNode> 0 </LeafsPerNode>
     <LeafInternodeLength> 0.00000 </LeafInternodeLength>
     <LeafMinSegmentOrder> 0 </LeafMinSegmentOrder>
     <LeafWidth> 0.00000 </LeafWidth>
     <LeafLength> 0.00000 </LeafLength>
     <LeafSurface> 0.00000 </LeafSurface>
     <PetioleAngle> 0.00000 </PetioleAngle>
     <PetioleLength> 0.00000 </PetioleLength>
     <LeafRotationalAngle> 0.00000 </LeafRotationalAngle>
     <FactorHorizontal> 0.00000 </FactorHorizontal>
     <TropismVector> 0.000000,0.000000,0.000000 </TropismVector>
     <TropismElstaicity> 0.00000 </TropismElstaicity>
     <SegmentRemovallist>  </SegmentRemovallist>
     <NrRules> 0 </NrRules>
     <Rules_Variable>  </Rules_Variable>
     <Rules_Replacement>  </Rules_Replacement>
     <Rules_isConditional>  </Rules_isConditional>
     <Rules_Condition>  </Rules_Condition>
     <Rules_Remark>  </Rules_Remark>
     <TermLString>   </TermLString>
     <ApplyTermLString> 0 </ApplyTermLString>
  </PLANT3D>"""

    filename = f"{plant_id}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(txt_content)

    print(f"\n=== ENVI-met TXT FILE CREATED ===")
    print(f"Filename: {filename}")
    print(f"Grid dimensions: {nx}x{ny}x{nz} cells")
    print(f"Grid centre: i={nx // 2}, j={ny // 2}")
    print(f"Number of LAD entries: {len(entries)}")
    print(f"Albedo: {albedo:.3f}, Transmittance: {transmittance:.3f}, Emissivity: {emissivity:.3f}")
    if ags_params:
        print("AGS parameters included.")

    return filename, txt_content


# ENVI‑met database update
def update_envi_met_database(database_path: str, plant_xml_content: str, plant_id: str) -> str:
    # Insert the voxelized tree into an ENVI‑met projectdatabase.edb file
    print(f"\nUpdating ENVI-met database at: {database_path}")

    with open(database_path, 'r', encoding='utf-8') as f:
        content = f.read()

    current_date = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    if '<PLANT3D>' in content:
        # Append after the last existing plant
        last_plant_end = content.rfind('</PLANT3D>')
        if last_plant_end == -1:
            raise ValueError("Database format error: <PLANT3D> found but no closing </PLANT3D>")
        insert_pos = last_plant_end + len('</PLANT3D>')
        new_content = content[:insert_pos] + '\n  ' + plant_xml_content + content[insert_pos:]
    else:
        # No plant section yet – insert before the final closing tag
        end_pos = content.rfind('</ENVI-MET_Datafile>')
        if end_pos == -1:
            raise ValueError("Invalid database format: </ENVI-MET_Datafile> not found")
        new_content = content[:end_pos] + '  ' + plant_xml_content + '\n' + content[end_pos:]

    # Update checksum (simply add 1, checksum is not important)
    def update_checksum(text: str) -> str:
        match = re.search(r'<checksum>(\d+)</checksum>', text)
        if match:
            old = int(match.group(1))
            new = old + 1
            return re.sub(r'<checksum>\d+</checksum>', f'<checksum>{new}</checksum>', text)
        match2 = re.search(r'checksum>(\d+)</checksum>', text)
        if match2:
            old = int(match2.group(1))
            new = old + 1
            return re.sub(r'checksum>\d+</checksum>', f'checksum>{new}</checksum>', text)
        return text

    new_content = update_checksum(new_content)

    # Update revision date
    rev_pattern = r'<revisiondate>.*?</revisiondate>'
    new_rev = f'<revisiondate>{current_date}</revisiondate>'
    if re.search(rev_pattern, new_content):
        new_content = re.sub(rev_pattern, new_rev, new_content)
    else:
        rev_pattern2 = r'revisiondate>.*?</revisiondate>'
        if re.search(rev_pattern2, new_content):
            new_content = re.sub(rev_pattern2, f'revisiondate>{current_date}</revisiondate>', new_content)

    with open(database_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Successfully added plant '{plant_id}' to database")
    return database_path