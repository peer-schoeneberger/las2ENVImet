# envimet_legacy.py
# Exporter for ENVI-met 5.x
import datetime
import re
from typing import List

from .base import (FORMAT_LEGACY, ExportError, Exporter, PlantExistsError, atomic_write, make_backup,
                   read_text)

_PLANT_HEAD = """<PLANT3D>
     <ID> {plant_id} </ID>
     <Description> {description} </Description>
     <AlternativeName> {binomial} </AlternativeName>
     <Planttype> {planttype} </Planttype>
     <Leaftype> {leaftype} </Leaftype>
     <Albedo> {albedo:.5f} </Albedo>
     <Eps> {emissivity:.5f} </Eps>
     <Transmittance> {transmittance:.5f} </Transmittance>
     <isoprene> {isoprene:.5f} </isoprene>
     <leafweigth> {leaf_weight:.5f} </leafweigth>
     <rs_min> {stom_res_min:.5f} </rs_min>
     <Height> {height:.5f} </Height>
     <Width> {width:.5f} </Width>
     <Depth> {depth:.5f} </Depth>
     <RootDiameter> {root_diameter:.5f} </RootDiameter>
     <cellsize> {cell_size:.5f} </cellsize>
     <xy_cells> {nx} </xy_cells>
     <z_cells> {nz} </z_cells>
     <scalefactor> 0.00000 </scalefactor>
     <hasAgsParams> {has_ags} </hasAgsParams>
     <agsCompPoint25> {comp_point25:.5f} </agsCompPoint25>
     <agsCompPointQ10> {comp_point_q10:.5f} </agsCompPointQ10>
     <agsGm25> {gm25:.5f} </agsGm25>
     <agsGmQ10> {gm_q10:.5f} </agsGmQ10>
     <agsGmT1> {gm_t1:.5f} </agsGmT1>
     <agsGmT2> {gm_t2:.5f} </agsGmT2>
     <agsAmmax25> {ammax25:.5f} </agsAmmax25>
     <agsAmmaxQ10> {ammax_q10:.5f} </agsAmmaxQ10>
     <agsAmmaxT1> {ammax_t1:.5f} </agsAmmaxT1>
     <agsAmmaxT2> {ammax_t2:.5f} </agsAmmaxT2>
     <agsE0> {e0:.5f} </agsE0>
     <agsF0> {f0:.5f} </agsF0>
     <agsGc> {gc:.5f} </agsGc>
     <agsDmax> {dmax:.5f} </agsDmax>
     <LAD-Profile type="sparematrix-3D" dataI="{nx}" dataJ="{ny}" zlayers="{nz}" defaultValue="0.00000">
"""

_PLANT_TAIL = """     </LAD-Profile>
     <RAD-Profile> {rad} </RAD-Profile>
     <Root-Range-Profile> {root_range} </Root-Range-Profile>
     <Season-Profile> {season} </Season-Profile>
     <Blossom-Profile> {blossom} </Blossom-Profile>
     <DensityWood> 690.00000 </DensityWood>
     <YoungsModulus> 8770000896.00000 </YoungsModulus>
     <YoungRatioRtoL> 0.12000 </YoungRatioRtoL>
     <MORBranch> 65.00000 </MORBranch>
     <MORConnection> 45.00000 </MORConnection>
     <PlantGroup> 0 </PlantGroup>
     <Color> {color} </Color>
     <Group> {group} </Group>
     <Author> {author} </Author>
     <costs> {cost:.5f} </costs>
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

_EMPTY_DATABASE = """<ENVI-MET_Datafile>
<Header>
<filetype>DATA</filetype>
<version>1</version>
<revisiondate>{date}</revisiondate>
<remark>Envi-Data</remark>
<fileInfo></fileInfo>
<checksum>0</checksum>
<encryptionlevel>0</encryptionlevel>
</Header>
</ENVI-MET_Datafile>
"""

_BLOCK_RE = re.compile(r'<PLANT3D>.*?</PLANT3D>', re.S)
_ID_RE = re.compile(r'<ID>\s*(.*?)\s*</ID>')
# A complete element on one or more lines, e.g. "     <Foo> 1 </Foo>" or a multi-line profile
_ELEMENT_RE = re.compile(r'^[ \t]*<([A-Za-z0-9_\-]+)(?:\s[^>]*)?>.*?</\1>[ \t]*$', re.S | re.M)


def _profile(values):
    return ','.join('%.5f' % v for v in values)


def hex_to_tcolor(color: str) -> int:
    # 'RRGGBB' -> Delphi TColor integer (BBGGRR), the colour encoding of the 5.x format.
    color = color.lstrip('#')
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return (b << 16) | (g << 8) | r


def plant_xml(tree) -> str:
    # The <PLANT3D> element for a tree (lines separated by \n).
    ags = tree.ags_complete()
    roots = tree.roots
    head = _PLANT_HEAD.format(
        plant_id=tree.plant_id, description=tree.description, binomial=tree.binomial,
        planttype=1 if tree.co2_type == 'C4' else 0,
        leaftype=2 if tree.leaf_type == 'conifer' else 1,
        albedo=tree.albedo, emissivity=tree.emissivity, transmittance=tree.transmittance,
        isoprene=tree.isoprene, leaf_weight=tree.leaf_weight, stom_res_min=tree.stom_res_min,
        height=tree.nz, width=tree.nx,
        depth=roots.depth if roots else tree.ny,
        root_diameter=roots.diameter if roots else 10.0,
        cell_size=tree.cell_size, nx=tree.nx, ny=tree.ny, nz=tree.nz,
        has_ags=1 if tree.ags else 0, **ags)
    body = ''.join('     %d,%d,%d,%.5f\n' % cell for cell in tree.cells)
    tail = _PLANT_TAIL.format(
        rad=_profile(roots.rad if roots else [0.1] * 10),
        root_range=_profile(roots.root_range if roots else [1.0] * 10),
        season=_profile(tree.season_profile), blossom=_profile(tree.blossom_profile),
        color=hex_to_tcolor(tree.color), group=tree.group, author=tree.author, cost=tree.cost)
    return head + body + tail


# Tags the plugin writes only as constants because it has no setting for them
PRESERVED_ON_REPLACE = ('DensityWood', 'YoungsModulus', 'YoungRatioRtoL', 'MORBranch', 'MORConnection',
                        'ColorStem', 'ColorBlossom', 'BlossomRadius')


def _keep_unmanaged_values(old_block: str, new_block: str) -> str:
    for name in PRESERVED_ON_REPLACE:
        old = re.search(r'<%s>(.*?)</%s>' % (name, name), old_block, re.S)
        if old:
            new_block = re.sub(r'<%s>.*?</%s>' % (name, name),
                               lambda _m, text=old.group(0): text, new_block, count=1, flags=re.S)
    return new_block


def _carry_unknown_elements(old_block: str, new_block: str) -> str:
    # Append elements of the replaced plant that this exporter does not write (future fields).
    known = set(re.findall(r'<([A-Za-z0-9_\-]+)[ >]', new_block))
    inner = old_block[len('<PLANT3D>'):-len('</PLANT3D>')]
    extra = [m.group(0) for m in _ELEMENT_RE.finditer(inner) if m.group(1) not in known]
    if not extra:
        return new_block
    closing = '  </PLANT3D>'
    return new_block[:-len(closing)] + '\n'.join(extra) + '\n' + closing


class EnvimetLegacyExporter(Exporter):
    format_id = FORMAT_LEGACY
    label = 'ENVI-met 5.x (classic database)'
    single_file_extension = '.txt'
    square_grid = False

    def write_plant_file(self, tree, path):
        with open(path, 'w', encoding='utf-8', newline='\r\n') as fh:
            fh.write(plant_xml(tree))
        return path

    def create_database(self, path):
        text = _EMPTY_DATABASE.format(date=datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S'))
        with open(path, 'w', encoding='utf-8', newline='\r\n') as fh:
            fh.write(text)
        return path

    def _read(self, database_path):
        text, had_bom, encoding = read_text(database_path)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        if '<ENVI-MET_Datafile>' not in text:
            raise ExportError("This file is not an ENVI-met 5.x database: %s" % database_path)
        return text, had_bom, encoding

    def list_plant_ids(self, database_path) -> List[str]:
        text, _, _ = self._read(database_path)
        ids = []
        for block in _BLOCK_RE.findall(text):
            match = _ID_RE.search(block)
            if match:
                ids.append(match.group(1))
        return ids

    def read_tree(self, database_path, plant_id):
        # Read a plant back into the version-neutral model (round-trip tests, comparisons).
        from ..core.roots import RootSystem
        from ..core.tree_model import TreeData
        text, _, _ = self._read(database_path)
        block = next((b for b in _BLOCK_RE.findall(text)
                      if (_ID_RE.search(b) or [None, None])[1] == plant_id), None)
        if block is None:
            raise ExportError("Plant '%s' not found in %s" % (plant_id, database_path))

        def tag(name, default=None):
            match = re.search(r'<%s>(.*?)</%s>' % (re.escape(name), re.escape(name)), block, re.S)
            return match.group(1).strip() if match else default

        def numbers(name):
            return [float(v) for v in tag(name, '').split(',') if v.strip()]

        lad = re.search(r'<LAD-Profile[^>]*dataI="(\d+)" dataJ="(\d+)" zlayers="(\d+)"[^>]*>(.*?)</LAD-Profile>',
                        block, re.S)
        cells = []
        for row in lad.group(4).strip().splitlines():
            parts = row.strip().split(',')
            if len(parts) == 4:
                cells.append((int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])))
        cells.sort(key=lambda c: (c[2], c[1], c[0]))
        ags = None
        if tag('hasAgsParams', '0') == '1':
            names = {'comp_point25': 'agsCompPoint25', 'comp_point_q10': 'agsCompPointQ10', 'gm25': 'agsGm25',
                     'gm_q10': 'agsGmQ10', 'gm_t1': 'agsGmT1', 'gm_t2': 'agsGmT2', 'ammax25': 'agsAmmax25',
                     'ammax_q10': 'agsAmmaxQ10', 'ammax_t1': 'agsAmmaxT1', 'ammax_t2': 'agsAmmaxT2',
                     'e0': 'agsE0', 'f0': 'agsF0', 'gc': 'agsGc', 'dmax': 'agsDmax'}
            ags = {key: float(tag(name)) for key, name in names.items()}
        tcolor = int(tag('Color', '52224'))
        color = '%02X%02X%02X' % (tcolor & 255, (tcolor >> 8) & 255, (tcolor >> 16) & 255)
        roots = RootSystem(float(tag('Depth')), float(tag('RootDiameter')), numbers('RAD-Profile'),
                           numbers('Root-Range-Profile'))
        return TreeData(plant_id, tag('Description', ''), int(lad.group(1)), int(lad.group(2)), int(lad.group(3)),
                        cells, cell_size=float(tag('cellsize', '1')), albedo=float(tag('Albedo')),
                        transmittance=float(tag('Transmittance')), emissivity=float(tag('Eps')), ags=ags,
                        roots=roots, binomial=tag('AlternativeName', '(none)'), group=tag('Group', ''),
                        color=color, leaf_type='conifer' if tag('Leaftype') == '2' else 'decid',
                        co2_type='C4' if tag('Planttype') == '1' else 'C3', author=tag('Author', ''),
                        season_profile=numbers('Season-Profile'), blossom_profile=numbers('Blossom-Profile'),
                        cost=float(tag('costs', '0')), isoprene=float(tag('isoprene', '12')),
                        leaf_weight=float(tag('leafweigth', '100')), stom_res_min=float(tag('rs_min', '0')))

    def insert_plant(self, tree, database_path, overwrite=False, backup=True, now=None):
        content, had_bom, encoding = self._read(database_path)
        xml = plant_xml(tree)

        existing = None
        for match in _BLOCK_RE.finditer(content):
            found = _ID_RE.search(match.group(0))
            if found and found.group(1) == tree.plant_id:
                existing = match
                break
        if existing is not None and not overwrite:
            raise PlantExistsError(tree.plant_id)

        if existing is not None:
            xml = _keep_unmanaged_values(existing.group(0), xml)
            xml = _carry_unknown_elements(existing.group(0), xml)
            new_content = content[:existing.start()] + xml + content[existing.end():]
            action = 'replaced'
        elif '<PLANT3D>' in content:
            last_end = content.rfind('</PLANT3D>')
            if last_end == -1:
                raise ExportError("Database format error: <PLANT3D> found but no closing </PLANT3D>")
            pos = last_end + len('</PLANT3D>')
            new_content = content[:pos] + '\n  ' + xml + content[pos:]
            action = 'inserted'
        else:
            end_pos = content.rfind('</ENVI-MET_Datafile>')
            if end_pos == -1:
                raise ExportError("Invalid database format: </ENVI-MET_Datafile> not found")
            new_content = content[:end_pos] + '  ' + xml + '\n' + content[end_pos:]
            action = 'inserted'

        # Header: checksum + 1 and new revision date
        checksum = re.search(r'<checksum>(\d+)</checksum>', new_content)
        if checksum:
            new_content = re.sub(r'<checksum>\d+</checksum>',
                                 '<checksum>%d</checksum>' % (int(checksum.group(1)) + 1), new_content)
        stamp = (now or datetime.datetime.now()).strftime('%d.%m.%Y %H:%M:%S')
        new_content = re.sub(r'<revisiondate>.*?</revisiondate>',
                             lambda _m: '<revisiondate>%s</revisiondate>' % stamp, new_content)

        backup_path = make_backup(database_path, now) if backup else None
        data = new_content.replace('\n', '\r\n').encode(encoding)
        atomic_write(database_path, (b'\xef\xbb\xbf' if had_bom else b'') + data)
        return {'action': action, 'backup': backup_path}
